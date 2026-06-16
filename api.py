import os
import time
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import JSONResponse

from data_pipeline import get_bvl_data
from econometrics  import get_pen_prices_and_volumes, filter_assets_by_volume, fit_garch_models
from optimization  import simulate_garch_paths, optimize_portfolio, compute_portfolio_metrics
from dcc           import get_latest_ewma_correlation, get_ewma_correlation_series
from backtesting   import run_var_backtest
from stress_testing import run_all_stress_tests

# ─────────────────────────────────────────────────────────────────
# SIMPLE TTL CACHE — avoids re-downloading & refitting every call
# ─────────────────────────────────────────────────────────────────
_CACHE_TTL_SECONDS = 4 * 3600   # 4-hour TTL

_cache = {
    "timestamp":    0.0,
    "valid_assets": None,
    "garch_models": None,
    "std_residuals":None,
    "cond_vols":    None,
    "log_returns":  None,
    "dates":        None,
}


def _is_cache_valid() -> bool:
    return (time.time() - _cache["timestamp"]) < _CACHE_TTL_SECONDS


def _refresh_cache():
    """Download data, filter, and fit GARCH models — caches results."""
    print("\n[CACHE MISS] Refreshing data and GARCH models...")
    bvl_data                          = get_bvl_data()
    prices, volumes                   = get_pen_prices_and_volumes(bvl_data)
    prices, volumes, valid_assets     = filter_assets_by_volume(prices, volumes)

    if not valid_assets:
        raise RuntimeError("No assets met the liquidity criteria.")

    log_returns, std_residuals, garch_models, cond_vols = fit_garch_models(prices)

    _cache["timestamp"]     = time.time()
    _cache["valid_assets"]  = valid_assets
    _cache["garch_models"]  = garch_models
    _cache["std_residuals"] = std_residuals
    _cache["cond_vols"]     = cond_vols
    _cache["log_returns"]   = log_returns
    _cache["dates"]         = list(prices.index.astype(str))
    print("[CACHE] Refresh complete.")


# ─────────────────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────────────────
app = FastAPI(title="BVL Institutional Quant Dashboard API")


class OptimizationRequest(BaseModel):
    confidence_level: float = 0.95
    capital_base:     float = 3200.0


@app.post("/api/calculate")
async def calculate_portfolio(req: OptimizationRequest):
    try:
        # ── 1. Load (cached) models ──────────────────────────────
        if not _is_cache_valid():
            _refresh_cache()
        else:
            print("[CACHE HIT] Using cached GARCH models.")

        valid_assets   = _cache["valid_assets"]
        garch_models   = _cache["garch_models"]
        std_residuals  = _cache["std_residuals"]
        cond_vols      = _cache["cond_vols"]
        log_returns    = _cache["log_returns"]
        dates          = _cache["dates"]

        # ── 2. Monte Carlo simulation ────────────────────────────
        print(f"[SIM] Running 10,000 paths × 20 days for {len(valid_assets)} assets (EWMA correlated)...")
        sim_cum, sim_daily = simulate_garch_paths(
            garch_models, std_residuals,
            n_sims=10_000, n_days=20, block_size=5,
            log_returns_for_ewma=log_returns,   # <-- EWMA dynamic correlation
        )

        # ── 3. Portfolio optimization ────────────────────────────
        print("[OPT] Minimizing CVaR via SLSQP + HHI penalty...")
        weights, opt_res = optimize_portfolio(
            sim_cum, valid_assets, alpha=req.confidence_level
        )

        # ── 4. Risk metrics (corrected) ──────────────────────────
        metrics = compute_portfolio_metrics(weights, sim_daily,
                                            confidence=req.confidence_level)

        exp_ret     = metrics["exp_return_20d"]
        exp_vol_ann = metrics["exp_vol_annual"]
        var_20d     = metrics["var_20d"]
        cvar_20d    = metrics["cvar_20d"]
        sharpe      = metrics["sharpe_20d"]
        port_sim    = metrics["port_sim_ret_20d"]

        # ── 5. Per-asset timeseries & GARCH params ───────────────
        log_returns_dict  = {}
        volatility_dict   = {}
        garch_params_dict = {}

        # Compute per-asset annualized stats from simulations
        asset_stats = {}
        n_days_sim  = sim_daily.shape[0]
        for i, asset in enumerate(valid_assets):
            daily_rets  = sim_daily[:, :, i]           # (20, 10000)
            ann_vol_asset = daily_rets.std(axis=0).mean() * np.sqrt(252)
            cum_ret_asset = (np.exp(daily_rets.sum(axis=0)) - 1).mean()
            asset_stats[asset] = {
                "ann_vol":    float(ann_vol_asset),
                "exp_ret_20d": float(cum_ret_asset),
            }

        for asset in valid_assets:
            if asset in log_returns.columns:
                lr = log_returns[asset].replace({np.nan: None})
                log_returns_dict[asset] = lr.tolist()
            else:
                log_returns_dict[asset] = []

            if asset in cond_vols.columns:
                cv = cond_vols[asset].replace({np.nan: None})
                volatility_dict[asset] = cv.tolist()
            else:
                volatility_dict[asset] = []

            if asset in garch_models:
                res = garch_models[asset]
                nu  = res.params.get("nu", None)
                lam = res.params.get("lambda", None)
                dist = getattr(res, "best_dist", "t")
                bic  = getattr(res, "best_bic", 0.0)
                gamma = getattr(res, "gamma_val", None)

                garch_params_dict[asset] = {
                    "dist":       dist,
                    "bic":        float(bic),
                    "mu":         float(res.params.get("mu",       0)),
                    "omega":      float(res.params.get("omega",    0)),
                    "alpha":      float(res.params.get("alpha[1]", 0)),
                    "beta":       float(res.params.get("beta[1]",  0)),
                    "gamma":      float(gamma) if gamma is not None else None,
                    "nu":         float(nu) if nu is not None else None,
                    "lam":        float(lam) if lam is not None else None,
                    "alpha_pval": float(res.pvalues.get("alpha[1]", 1.0)),
                    "beta_pval":  float(res.pvalues.get("beta[1]",  1.0)),
                    # Annualized stats from simulation
                    "ann_vol_sim":    asset_stats[asset]["ann_vol"],
                    "exp_ret_20d_sim": asset_stats[asset]["exp_ret_20d"],
                }

        # ── 6. Response ──────────────────────────────────────────
        response_data = {
            "assets":  valid_assets,
            "weights": [float(w) for w in weights],
            "metrics": {
                "expected_return":     float(exp_ret),      # 20D simple return
                "expected_volatility": float(exp_vol_ann),  # annualized
                "var":                 float(var_20d),      # 20D VaR as fraction
                "cvar":                float(cvar_20d),     # 20D CVaR as fraction
                "sharpe_ratio":        float(sharpe),       # 20D excess return / vol, adj. for BCRP rf
                "rf_annual":           0.065,               # BCRP reference rate used
            },
            "garch_params": garch_params_dict,
            "timeseries": {
                "dates":       dates,
                "log_returns": log_returns_dict,
                "volatility":  volatility_dict,
            },
            "monte_carlo": {
                "portfolio_returns": [float(x) for x in port_sim],
            },
            "optimization": {
                "success":  bool(opt_res.success),
                "message":  str(opt_res.message),
                "n_assets_active": int((weights > 0.005).sum()),
            },
        }

        return response_data

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] {e}\n{tb}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "detail": tb},
        )


@app.post("/api/reset-cache")
async def reset_cache():
    """Force a cache refresh on next calculation request."""
    _cache["timestamp"] = 0.0
    return {"message": "Cache invalidated. Next /api/calculate will refresh."}


@app.get("/api/dynamic-correlation")
async def dynamic_correlation():
    """Returns the latest EWMA correlation matrix (λ=0.94) for the portfolio assets."""
    try:
        if not _is_cache_valid():
            _refresh_cache()
        log_returns  = _cache["log_returns"]
        valid_assets = _cache["valid_assets"]
        corr_df = get_latest_ewma_correlation(log_returns[valid_assets])
        return {
            "tickers": corr_df.columns.tolist(),
            "matrix":  corr_df.round(4).values.tolist(),
            "lambda":  0.94,
            "method":  "EWMA RiskMetrics",
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/api/backtest")
async def backtest_var(confidence: float = 0.95):
    """
    Runs walk-forward VaR backtesting with Kupiec + Christoffersen tests.
    NOTE: This endpoint takes 5-15 minutes on first call; results are cached.
    """
    try:
        if not _is_cache_valid():
            _refresh_cache()
        log_returns  = _cache["log_returns"]
        valid_assets = _cache["valid_assets"]

        # Only backtest assets with enough history
        lr_subset = log_returns[valid_assets].dropna(how="all", axis=1)
        print(f"[BACKTEST] Running walk-forward VaR backtest for {len(lr_subset.columns)} assets...")
        results = run_var_backtest(lr_subset, garch_models=_cache["garch_models"], confidence=confidence)
        print("[BACKTEST] Complete.")
        return {"confidence": confidence, "assets": results}
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()})


@app.get("/api/stress-test")
async def stress_test(capital: float = 1_000_000.0, confidence: float = 0.95):
    """Applies 6 real historical and parametric stress scenarios to the current optimal portfolio."""
    try:
        if not _is_cache_valid():
            _refresh_cache()

        # Re-run optimization to get current weights
        garch_models  = _cache["garch_models"]
        std_residuals = _cache["std_residuals"]
        log_returns   = _cache["log_returns"]
        valid_assets  = _cache["valid_assets"]

        sim_cum, sim_daily = simulate_garch_paths(
            garch_models, std_residuals,
            n_sims=5_000, n_days=20, block_size=5,
            log_returns_for_ewma=log_returns,
        )
        weights, _ = optimize_portfolio(sim_cum, valid_assets, alpha=confidence)

        print("[STRESS] Running stress test scenarios...")
        results = run_all_stress_tests(weights, valid_assets, capital=capital)
        print("[STRESS] Complete.")
        return {"capital": capital, "scenarios": results}
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={"error": str(e), "detail": traceback.format_exc()})


# ── Static files ──────────────────────────────────────────────────
os.makedirs("static/css", exist_ok=True)
os.makedirs("static/js",  exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)
