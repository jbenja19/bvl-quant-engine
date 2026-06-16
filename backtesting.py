"""
backtesting.py — VaR Backtesting: Kupiec POF + Christoffersen Independence Tests
==================================================================================
Validates the model's VaR predictions against realized historical returns.

Methodology:
    Walk-Forward Validation — for each day t in the test window, fit GARCH
    on all data up to t-1, estimate 1-day VaR(α) for day t, then check if
    the actual return on day t exceeded the loss threshold (violation/hit).

Statistical Tests (Basilea II/III compliant):
    1. Kupiec (1995) Proportion of Failures (POF) test:
       H0: violation rate p̂ = α_expected (e.g. 5% for 95% VaR)
       Statistic: LR_uc ~ χ²(1)

    2. Christoffersen (1998) Independence test:
       H0: violations are independent (not clustered in time)
       Statistic: LR_ind ~ χ²(1)

    3. Conditional Coverage (CC) = LR_uc + LR_ind ~ χ²(2)

Basel Traffic Light:
    Green  (✅): 0–4 violations in 250 days → model accepted
    Yellow (⚠️): 5–9 violations → under scrutiny
    Red    (❌): ≥10 violations → model likely rejected

References:
    - Kupiec, P.H. (1995). "Techniques for Verifying the Accuracy of Risk
      Measurement Models". JoD.
    - Christoffersen, P.F. (1998). "Evaluating Interval Forecasts". IER.
"""

import numpy as np
import pandas as pd
from scipy import stats
from arch import arch_model
import warnings


# ─────────────────────────────────────────────────────────────────────────────
# KUPIEC PROPORTION-OF-FAILURES TEST
# ─────────────────────────────────────────────────────────────────────────────

def kupiec_pof_test(violations: np.ndarray, confidence: float = 0.95) -> dict:
    """
    Kupiec (1995) Unconditional Coverage test.

    H0: actual violation rate = expected failure rate (1 - confidence)

    Args:
        violations (np.ndarray): Binary array. 1 = VaR exceeded, 0 = not.
        confidence (float): VaR confidence level (e.g. 0.95).

    Returns:
        dict with keys: n_obs, n_violations, violation_rate, expected_rate,
                        lr_stat, p_value, reject_h0
    """
    T = len(violations)
    V = int(violations.sum())
    p_hat = V / T if T > 0 else 0.0
    p_exp = 1 - confidence

    if V == 0:
        lr = -2 * T * np.log(1 - p_exp)
    elif V == T:
        lr = -2 * T * np.log(p_exp)
    else:
        lr = -2 * (
            V * np.log(p_exp / p_hat) + (T - V) * np.log((1 - p_exp) / (1 - p_hat))
        )

    lr = max(lr, 0.0)
    p_value = 1 - stats.chi2.cdf(lr, df=1)

    return {
        "n_obs":           T,
        "n_violations":    V,
        "violation_rate":  round(p_hat, 4),
        "expected_rate":   p_exp,
        "lr_uc":           round(lr, 4),
        "p_value_uc":      round(p_value, 4),
        "reject_h0_uc":    bool(p_value < 0.05),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHRISTOFFERSEN INDEPENDENCE TEST
# ─────────────────────────────────────────────────────────────────────────────

def christoffersen_independence_test(violations: np.ndarray) -> dict:
    """
    Christoffersen (1998) Independence test.

    H0: violations are serially independent (no clustering).

    Args:
        violations (np.ndarray): Binary array. 1 = VaR exceeded.

    Returns:
        dict with keys: lr_ind, p_value_ind, reject_h0_ind
    """
    v = violations.astype(int)
    T = len(v)

    # Count transitions
    n00 = n01 = n10 = n11 = 0
    for t in range(1, T):
        prev, curr = v[t - 1], v[t]
        if   prev == 0 and curr == 0: n00 += 1
        elif prev == 0 and curr == 1: n01 += 1
        elif prev == 1 and curr == 0: n10 += 1
        else:                          n11 += 1

    # Transition probabilities
    denom_0 = n00 + n01
    denom_1 = n10 + n11
    pi_01 = n01 / denom_0 if denom_0 > 0 else 0.0
    pi_11 = n11 / denom_1 if denom_1 > 0 else 0.0
    pi    = (n01 + n11) / (n00 + n01 + n10 + n11) if T > 1 else 0.0

    def _safe_log(x):
        return np.log(x) if x > 0 else 0.0

    # Likelihood under independence (pi constant)
    ll_ind = (
        (n00 + n10) * _safe_log(1 - pi) +
        (n01 + n11) * _safe_log(pi)
    )
    # Likelihood under Markov alternative
    ll_markov = (
        n00 * _safe_log(1 - pi_01) + n01 * _safe_log(pi_01) +
        n10 * _safe_log(1 - pi_11) + n11 * _safe_log(pi_11)
    )

    lr_ind  = max(-2 * (ll_ind - ll_markov), 0.0)
    p_value = 1 - stats.chi2.cdf(lr_ind, df=1)

    return {
        "lr_ind":          round(lr_ind, 4),
        "p_value_ind":     round(p_value, 4),
        "reject_h0_ind":   bool(p_value < 0.05),
        "pi_01":           round(pi_01, 4),
        "pi_11":           round(pi_11, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BASEL TRAFFIC LIGHT
# ─────────────────────────────────────────────────────────────────────────────

def basel_traffic_light(n_violations: int, n_obs: int = 250) -> dict:
    """
    Basilea II/III Traffic Light classification.

    Args:
        n_violations (int): Number of VaR breaches.
        n_obs (int): Number of observation days (typically 250).

    Returns:
        dict with color, label, and capital_multiplier.
    """
    # Scale thresholds to actual observation window
    scale = n_obs / 250
    if n_violations <= round(4 * scale):
        return {"color": "green",  "label": "Aceptado",         "capital_multiplier": 3.0}
    elif n_violations <= round(9 * scale):
        return {"color": "yellow", "label": "Zona de Alerta",   "capital_multiplier": 3.5}
    else:
        return {"color": "red",    "label": "Modelo Rechazado", "capital_multiplier": 4.0}


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD BACKTEST
# ─────────────────────────────────────────────────────────────────────────────

def run_var_backtest(log_returns: pd.DataFrame,
                     garch_models: dict,
                     confidence: float = 0.95,
                     train_window: int = 504,
                     test_window: int = 252) -> dict:
    """
    Fast Walk-forward VaR backtesting on each asset individually.

    Uses the best selected GARCH specification from the cache to perform a 
    pseudo-out-of-sample forecast. It fits the model on the train window, 
    and uses the estimated parameters to filter the out-of-sample data, 
    generating 1-step-ahead VaR predictions for the test window.

    Args:
        log_returns  (pd.DataFrame): Full log return history, decimal scale.
        garch_models (dict):         Cached models dict from API containing winning specs.
        confidence   (float):        VaR confidence level.
        train_window (int):          Trading days for GARCH training (~2 years).
        test_window  (int):          Days to backtest (~1 year).

    Returns:
        dict: per-asset results + aggregate portfolio-level results.
    """
    results = {}
    clean = log_returns.dropna(how="all", axis=1)
    T = len(clean)

    if T < train_window + test_window:
        raise ValueError(f"Insufficient history: need {train_window + test_window} days, have {T}.")

    test_start = T - test_window

    for ticker in clean.columns:
        ret_series = clean[ticker].dropna()
        if len(ret_series) < train_window + test_window:
            continue
            
        if ticker not in garch_models:
            continue
            
        # Get the cached winning specification for this asset
        cached_res = garch_models[ticker]
        dist_name = getattr(cached_res, "best_dist", "t")
        
        if "GJR" in dist_name:
            o_param = 1
        else:
            o_param = 0
            
        dist = dist_name.split("-")[-1]

        # Ensure correct indices relative to the individual asset series length
        asset_T = len(ret_series)
        asset_test_start = asset_T - test_window
        data_pct = ret_series * 100.0

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                
                # Setup model with the winning specs
                model = arch_model(data_pct, vol="GARCH", p=1, o=o_param, q=1, dist=dist, mean="Constant")
                
                # Fit the model up to the start of the test window
                res = model.fit(last_obs=asset_test_start, disp="off", options={"maxiter": 300})
                
                # Generate 1-step-ahead out-of-sample forecasts
                forecasts = res.forecast(horizon=1, start=asset_test_start)
                
            # Extract the out-of-sample variance sequence
            # arch .forecast() might return only the forecasted rows or the full index with NaNs. dropna() is robust.
            oos_var = forecasts.variance["h.1"].dropna()
            mu_1d_pct = res.params.get("mu", 0.0)

            # Determine the critical value (z_q) for VaR
            if dist == "t":
                nu  = res.params.get("nu", 5.0)
                z_q = stats.t.ppf(1 - confidence, df=nu)
            elif dist == "skewt":
                nu  = res.params.get("nu", 5.0)
                z_q = stats.t.ppf(1 - confidence, df=nu)
            else:
                z_q = stats.norm.ppf(1 - confidence)

            var_estimates = []
            violations    = []
            test_dates    = []
            actual_rets   = []

            # Iterate through the out-of-sample period
            for i in range(len(oos_var)):
                idx = asset_test_start + i
                if idx >= asset_T: 
                    break
                
                sigma_1d_pct = np.sqrt(oos_var.iloc[i])
                var_1d = -(mu_1d_pct + z_q * sigma_1d_pct) / 100.0
                
                actual_ret = ret_series.iloc[idx]
                violation  = int(-actual_ret > var_1d)
                
                var_estimates.append(var_1d)
                violations.append(violation)
                test_dates.append(ret_series.index[idx])
                actual_rets.append(actual_ret)

            if len(violations) == 0:
                continue

            violations_arr = np.array(violations)
            var_arr        = np.array(var_estimates)
            actual_rets    = np.array(actual_rets)

            kupiec   = kupiec_pof_test(violations_arr, confidence)
            christo  = christoffersen_independence_test(violations_arr)
            traffic  = basel_traffic_light(int(violations_arr.sum()), len(violations_arr))

            lr_cc    = kupiec["lr_uc"] + christo["lr_ind"]
            p_cc     = 1 - stats.chi2.cdf(lr_cc, df=2)

            results[ticker] = {
                "dates":            [str(d)[:10] for d in test_dates],
                "var_estimates":    [round(float(v), 6) for v in var_arr],
                "actual_returns":   [round(float(r), 6) for r in actual_rets],
                "violations":       [int(x) for x in violations],
                "kupiec":           kupiec,
                "christoffersen":   christo,
                "conditional_coverage": {
                    "lr_cc":      round(lr_cc, 4),
                    "p_value_cc": round(p_cc, 4),
                    "reject":     bool(p_cc < 0.05),
                },
                "traffic_light":    traffic,
            }
            
        except Exception as e:
            print(f"[BACKTEST] Skipping {ticker} due to error: {e}")
            continue

    return results
