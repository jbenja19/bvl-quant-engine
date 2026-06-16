"""
stress_testing.py — Historical & Parametric Stress Testing
===========================================================
Applies real historical crisis returns to the current optimal portfolio
and computes parametric factor shocks (copper, FX, etc.).

All historical scenarios use REAL market data downloaded at runtime
from Yahoo Finance — nothing is hard-coded or simulated.

Scenarios implemented:
    1. COVID-19 Crash        (2020-02-20 to 2020-03-23)
    2. Global Financial Crisis (2008-09-01 to 2008-11-30)
    3. Commodity Crash 2015  (2015-06-01 to 2015-12-31)
    4. Peru Political Risk   (2022-06-01 to 2022-07-31)
    5. Copper Shock (–25%)   parametric
    6. FX Shock (+15%)       parametric

References:
    - Basel Committee (2009). "Principles for sound stress testing practices"
    - BIS (2018). "Stress testing: a review of key concepts"
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL SCENARIO DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

HISTORICAL_SCENARIOS = {
    "COVID-19 Crash": {
        "start": "2020-02-20",
        "end":   "2020-03-23",
        "description": "Derrumbe global por pandemia COVID-19. Mayor caída mensual desde 1987."
    },
    "Crisis Financiera Global 2008": {
        "start": "2008-09-01",
        "end":   "2008-11-30",
        "description": "Colapso de Lehman Brothers y contagio global del sistema financiero."
    },
    "Desplome de Commodities 2015": {
        "start": "2015-06-01",
        "end":   "2015-12-31",
        "description": "Caída del cobre y materias primas por desaceleración China. Impacto severo en BVL minera."
    },
    "Crisis Política Perú 2022": {
        "start": "2022-06-01",
        "end":   "2022-07-31",
        "description": "Riesgo político por inicio del gobierno Castillo y propuestas de nationalización minera."
    },
}

# Factor tickers relevant for parametric shocks
COPPER_TICKERS  = {"SCCO.LM", "BVN.LM"}       # most exposed to copper price
USD_TICKERS     = {"BAP.LM", "SCCO.LM", "BVN.LM", "IFS.LM", "INRETC1.LM"}


# ─────────────────────────────────────────────────────────────────────────────
# HISTORICAL STRESS TEST
# ─────────────────────────────────────────────────────────────────────────────

def apply_historical_scenario(weights: np.ndarray,
                               tickers: list[str],
                               scenario_name: str,
                               scenario_def: dict,
                               capital: float = 1_000_000.0) -> dict:
    """
    Applies a historical crisis period's actual returns to the current portfolio.

    Downloads historical data for the crisis window and computes the
    cumulative portfolio return over that period using real market prices.

    Args:
        weights       (np.ndarray): Current optimal weights.
        tickers       (list):       Asset tickers in same order as weights.
        scenario_name (str):        Human-readable name.
        scenario_def  (dict):       {"start", "end", "description"}.
        capital       (float):      Portfolio capital in PEN.

    Returns:
        dict: Scenario results with cumulative return, P&L, and daily series.
    """
    start = scenario_def["start"]
    end   = scenario_def["end"]

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            prices = yf.download(
                tickers, start=start, end=end,
                group_by="ticker", progress=False, auto_adjust=True
            )

        # Extract Close prices
        if isinstance(prices.columns, pd.MultiIndex):
            close_data = {}
            for t in tickers:
                if t in prices.columns.get_level_values(0):
                    series = prices[(t, "Close")] if (t, "Close") in prices.columns else prices[t]["Close"]
                    close_data[t] = series
            df_close = pd.DataFrame(close_data)
        else:
            df_close = prices[["Close"]] if "Close" in prices.columns else prices

        # Log returns (decimal)
        log_rets = np.log(df_close).diff().dropna()

        if log_rets.empty or len(log_rets) < 2:
            return _empty_scenario(scenario_name, scenario_def, "Insufficient data for this period")

        # Align tickers and weights
        available = [t for t in tickers if t in log_rets.columns]
        avail_idx  = [tickers.index(t) for t in available]
        w_avail    = weights[avail_idx]
        if w_avail.sum() > 0:
            w_avail = w_avail / w_avail.sum()   # renormalize to available assets

        rets_matrix = log_rets[available].values   # (T, n_avail)

        # Daily portfolio log returns → simple returns
        port_log_daily = rets_matrix @ w_avail
        port_daily_simple = np.exp(port_log_daily) - 1.0

        cum_return = float((np.exp(port_log_daily.sum())) - 1.0)
        pnl        = cum_return * capital

        return {
            "scenario":     scenario_name,
            "description":  scenario_def["description"],
            "period":       f"{start} → {end}",
            "trading_days": int(len(log_rets)),
            "cum_return":   round(cum_return, 6),
            "pnl_pen":      round(pnl, 2),
            "daily_returns": [round(float(r), 6) for r in port_daily_simple],
            "dates":        [str(d)[:10] for d in log_rets.index],
            "status":       "ok",
        }

    except Exception as e:
        return _empty_scenario(scenario_name, scenario_def, str(e))


def _empty_scenario(name, defn, error_msg):
    return {
        "scenario":     name,
        "description":  defn["description"],
        "period":       f"{defn['start']} → {defn['end']}",
        "trading_days": 0,
        "cum_return":   None,
        "pnl_pen":      None,
        "daily_returns": [],
        "dates":        [],
        "status":       f"error: {error_msg}",
    }


# ─────────────────────────────────────────────────────────────────────────────
# PARAMETRIC FACTOR SHOCKS
# ─────────────────────────────────────────────────────────────────────────────

def apply_copper_shock(weights: np.ndarray,
                        tickers: list[str],
                        shock: float = -0.25,
                        capital: float = 1_000_000.0) -> dict:
    """
    Parametric shock: copper price drops by `shock` fraction instantaneously.
    Affected tickers: SCCO.LM, BVN.LM (empirical copper beta ≈ 0.85).

    Args:
        shock (float): e.g. -0.25 for –25% copper crash.
    """
    COPPER_BETA = 0.85   # empirical sensitivity of mining stocks to copper
    instant_rets = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        if t in COPPER_TICKERS:
            instant_rets[i] = shock * COPPER_BETA

    port_ret = float(np.dot(weights, instant_rets))
    return {
        "scenario":     f"Shock Cobre {shock*100:.0f}%",
        "description":  f"Caída instantánea del precio del cobre en {shock*100:.0f}%. Beta cobre ≈ {COPPER_BETA}.",
        "period":       "Paramétrico",
        "trading_days": 1,
        "cum_return":   round(port_ret, 6),
        "pnl_pen":      round(port_ret * capital, 2),
        "daily_returns": [round(port_ret, 6)],
        "dates":        ["Day 0"],
        "status":       "ok",
    }


def apply_fx_shock(weights: np.ndarray,
                    tickers: list[str],
                    shock_usdpen: float = 0.15,
                    capital: float = 1_000_000.0) -> dict:
    """
    Parametric shock: USD/PEN appreciates by `shock_usdpen` (e.g. +15%).
    USD-denominated stocks gain in PEN terms, local stocks unaffected.

    Args:
        shock_usdpen (float): e.g. +0.15 → sol weakens 15% vs USD.
    """
    instant_rets = np.zeros(len(tickers))
    for i, t in enumerate(tickers):
        if t in USD_TICKERS:
            # USD stocks appreciate in PEN when sol depreciates
            instant_rets[i] = shock_usdpen

    port_ret = float(np.dot(weights, instant_rets))
    return {
        "scenario":     f"Shock FX +{shock_usdpen*100:.0f}% USDPEN",
        "description":  f"Depreciación del sol en {shock_usdpen*100:.0f}%. Activos en USD aprecian en términos PEN.",
        "period":       "Paramétrico",
        "trading_days": 1,
        "cum_return":   round(port_ret, 6),
        "pnl_pen":      round(port_ret * capital, 2),
        "daily_returns": [round(port_ret, 6)],
        "dates":        ["Day 0"],
        "status":       "ok",
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER — All Scenarios
# ─────────────────────────────────────────────────────────────────────────────

def run_all_stress_tests(weights: np.ndarray,
                          tickers: list[str],
                          capital: float = 1_000_000.0) -> list[dict]:
    """
    Run all historical and parametric stress tests and return results list.

    Args:
        weights (np.ndarray): Portfolio weights.
        tickers (list):       Asset tickers.
        capital (float):      Portfolio capital in PEN.

    Returns:
        list of scenario result dicts, sorted by P&L (worst first).
    """
    results = []

    # Historical scenarios
    for name, defn in HISTORICAL_SCENARIOS.items():
        result = apply_historical_scenario(weights, tickers, name, defn, capital)
        results.append(result)

    # Parametric shocks
    results.append(apply_copper_shock(weights, tickers, shock=-0.25, capital=capital))
    results.append(apply_fx_shock(weights, tickers, shock_usdpen=0.15, capital=capital))

    # Sort: worst P&L first (most informative for risk management)
    valid = [r for r in results if r["pnl_pen"] is not None]
    invalid = [r for r in results if r["pnl_pen"] is None]
    valid.sort(key=lambda r: r["pnl_pen"])

    return valid + invalid
