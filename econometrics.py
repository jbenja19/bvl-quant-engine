import sys
import os
import numpy as np
import pandas as pd
import yfinance as yf
from arch import arch_model

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_pipeline import get_bvl_data

# Tickers whose prices are quoted in USD on the Lima Stock Exchange
USD_TICKERS = {"BAP.LM", "SCCO.LM", "BVN.LM", "IFS.LM", "INRETC1.LM"}


def get_pen_prices_and_volumes(df):
    """
    Converts USD-denominated tickers to PEN using the USDPEN=X spot rate.
    Uses last-known rate as fallback instead of a fixed hard-coded value.
    """
    print("Downloading USD/PEN exchange rate to standardize currencies...")
    fx_raw = yf.download("USDPEN=X", period="5y", progress=False)

    if isinstance(fx_raw.columns, pd.MultiIndex):
        fx_series = fx_raw[("Close", "USDPEN=X")].copy()
    else:
        fx_series = fx_raw["Close"].copy()

    # Align to BVL calendar; forward-fill then fall back to 30-day trailing mean
    fx_series = fx_series.reindex(df.index, method="ffill")
    fallback   = fx_series.rolling(30, min_periods=1).mean()
    fx_series  = fx_series.fillna(fallback).fillna(3.72)   # last-resort: recent spot

    df_prices_pen = pd.DataFrame(index=df.index)
    df_volume_pen = pd.DataFrame(index=df.index)

    tickers = df["Close"].columns

    for t in tickers:
        close  = df[("Close",  t)].copy()
        volume = df[("Volume", t)].copy()
        is_usd = t in USD_TICKERS

        if is_usd:
            close_pen  = close * fx_series
            volume_pen = volume * close * fx_series
        else:
            close_pen  = close
            volume_pen = volume * close

        df_prices_pen[t] = close_pen
        df_volume_pen[t] = volume_pen

    return df_prices_pen, df_volume_pen


def calculate_amihud_ratio(prices_pen, volume_pen):
    """
    Amihud (2002) illiquidity ratio:
        ILLIQ_i = Mean( |R_t| / DVOL_t )
    where DVOL_t = daily dollar volume in PEN.
    """
    amihud_ratios = {}
    for t in prices_pen.columns:
        p      = prices_pen[t]
        v_pen  = volume_pen[t]
        ret    = p.pct_change().dropna()
        v_aligned = v_pen.loc[ret.index]
        valid  = v_aligned > 0

        if valid.sum() > 0:
            amihud_ratios[t] = (ret.abs().loc[valid] / v_aligned.loc[valid]).mean()
        else:
            amihud_ratios[t] = np.nan

    return amihud_ratios


def filter_assets_by_volume(prices_pen, volume_pen, threshold=50_000.0):
    """
    Excludes assets whose mean daily PEN volume is below threshold.
    """
    avg_volumes  = volume_pen.mean()
    valid_tickers = avg_volumes[avg_volumes >= threshold].index.tolist()

    print("\n--- Asset Liquidity Filter ---")
    for t in prices_pen.columns:
        status = "PASSED" if t in valid_tickers else "EXCLUDED"
        print(f"{t:12s} | Avg Daily Volume: {avg_volumes[t]:14,.2f} PEN | Status: {status}")

    return prices_pen[valid_tickers], volume_pen[valid_tickers], valid_tickers


def fit_garch_models(prices_pen):
    """
    Fits GARCH(1,1) models to daily log returns testing multiple distributions:
    Normal, Student-t, Skewed Student-t, and Generalized Error Distribution (GED).
    Selects the distribution that minimizes the Bayesian Information Criterion (BIC).

    Key design decisions:
    - Returns are scaled ×100 for numerical stability in the arch library.
    - BIC penalizes complexity, so skewed/fat-tailed models only win if they significantly
      improve the fit.

    Returns:
        log_returns   (pd.DataFrame): Daily log returns in decimal scale.
        std_residuals (pd.DataFrame): Standardized residuals z_t = eps_t / sigma_t.
        garch_models  (dict):         ticker → fitted arch_model result (×100 scale).
        cond_vols     (pd.DataFrame): Conditional volatility in DECIMAL (not %) scale.
    """
    # Full log-return series in decimal scale (NaN on day-1 per ticker)
    log_returns = np.log(prices_pen).diff()

    std_residuals = pd.DataFrame(index=log_returns.dropna().index)
    cond_vols     = pd.DataFrame(index=log_returns.dropna().index)
    garch_models  = {}

    candidate_dists = ["Normal", "t", "skewt", "ged"]

    print("\n--- Fitting GARCH(1,1) Models (BIC Selection) ---")
    for t in prices_pen.columns:
        ret_series = log_returns[t].dropna() * 100.0   # scale to %
        ret_clean = ret_series.copy()

        if len(ret_clean) < 60:
            print(f"{t:12s} | SKIPPED — insufficient observations ({len(ret_clean)})")
            continue

        best_bic = float('inf')
        best_res = None
        best_dist = None

        print(f"{t:12s} | Testing distributions...")
        for dist in candidate_dists:
            try:
                model = arch_model(
                    ret_clean,
                    mean  = "Constant",
                    vol   = "Garch",
                    p     = 1,
                    q     = 1,
                    dist  = dist,
                )
                # Suppress convergence warnings to avoid console spam
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=Warning)
                    res = model.fit(disp="off", options={"maxiter": 500})
                
                if res.bic < best_bic:
                    best_bic = res.bic
                    best_res = res
                    best_dist = dist
            except Exception as e:
                pass # Model failed to converge for this distribution

        if best_res is not None:
            res = best_res
            alpha_pval = res.pvalues.get("alpha[1]", np.nan)
            beta_pval  = res.pvalues.get("beta[1]",  np.nan)
            mu_val     = res.params.get("mu",        np.nan)
            omega_val  = res.params.get("omega",     np.nan)
            alpha_val  = res.params.get("alpha[1]",  np.nan)
            beta_val   = res.params.get("beta[1]",   np.nan)
            nu_val     = res.params.get("nu",        np.nan)   # df
            lam_val    = res.params.get("lambda",    np.nan)   # skew

            print(f"{t:12s} | Winner: {best_dist} (BIC: {best_bic:.2f})")
            
            # Standardized residuals: z_t = eps_t / sigma_t (scale-free)
            std_resid = res.resid / res.conditional_volatility

            # Align to the common index (drop leading NaN dates)
            common_idx = std_residuals.index.intersection(std_resid.index)
            std_residuals.loc[common_idx, t]  = std_resid.loc[common_idx]
            cond_vols.loc[common_idx, t]      = res.conditional_volatility.loc[common_idx] / 100.0

            # Tag the result object with the best distribution name so the API can read it
            res.best_dist = best_dist
            res.best_bic = best_bic
            garch_models[t] = res
        else:
            print(f"{t:12s} | ERROR: All models failed to converge.")

    return log_returns.dropna(), std_residuals, garch_models, cond_vols


if __name__ == "__main__":
    raw_data                           = get_bvl_data()
    prices_pen, volume_pen             = get_pen_prices_and_volumes(raw_data)
    amihud                             = calculate_amihud_ratio(prices_pen, volume_pen)
    prices_filtered, vol_filt, tickers = filter_assets_by_volume(prices_pen, volume_pen)
    log_ret, std_res, models, cond_v   = fit_garch_models(prices_filtered)

    print("\nStandardized Residuals (tail):")
    print(std_res.tail())
    print("\nConditional Volatility (decimal, tail):")
    print(cond_v.tail())
