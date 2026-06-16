import numpy as np
import pandas as pd
from scipy.optimize import minimize
from dcc import get_cholesky_for_simulation

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
# BCRP reference rate as of mid-2025: ~6.5% annual
# Converted to 20-day horizon: (1 + 0.065)^(20/252) - 1
RF_ANNUAL    = 0.065
RF_20D       = (1 + RF_ANNUAL) ** (20 / 252) - 1
RF_DAILY     = (1 + RF_ANNUAL) ** (1 / 252) - 1


def simulate_garch_paths(garch_models, std_residuals, n_sims=10000, n_days=20,
                         block_size=5, log_returns_for_ewma=None):
    """
    Simulates joint future trajectories of daily log returns over a given horizon
    using GARCH(1,1)/GJR-GARCH parameter projections and a BLOCK bootstrap of
    historical standardized residuals (to preserve short-term autocorrelation).

    EWMA Cholesky Correlation: If log_returns_for_ewma is provided, the simulated
    i.i.d. residual blocks are transformed by the Cholesky factor of the latest
    EWMA correlation matrix (λ=0.94), introducing realistic time-varying
    cross-asset correlations into the Monte Carlo paths.

    All internal computation is done in the ×100 scale used by the arch library,
    with a final /100 conversion to decimal returns.

    Args:
        garch_models  (dict):         ticker → fitted arch_model result
        std_residuals (pd.DataFrame): standardized residuals (z_t), scale-free
        n_sims        (int):          Monte Carlo paths (default 10,000)
        n_days        (int):          forecast horizon in trading days (default 20)
        block_size    (int):          block length for block bootstrap (default 5)

    Returns:
        sim_cum_returns_simple (np.ndarray): shape (n_sims, n_assets)
            Cumulative simple returns at the horizon end.
        sim_returns_daily (np.ndarray): shape (n_days, n_sims, n_assets)
            Daily log returns for each path (decimal scale).
    """
    tickers   = list(garch_models.keys())
    n_assets  = len(tickers)

    # ── EWMA Cholesky (time-varying correlations) ─────────────────────────
    if log_returns_for_ewma is not None:
        try:
            # Align to available tickers in the right order
            lr_aligned = log_returns_for_ewma[tickers]
            L_chol = get_cholesky_for_simulation(lr_aligned)
        except Exception:
            L_chol = np.eye(n_assets)   # fallback: identity (no cross-correlation)
    else:
        L_chol = np.eye(n_assets)

    # ── Extract GARCH parameters (arch library uses ×100 scale internally) ──
    # Parameters: omega is in (%²), mu is in (%), alpha/beta are dimensionless
    omega = np.zeros(n_assets)
    alpha = np.zeros(n_assets)
    beta  = np.zeros(n_assets)
    mu    = np.zeros(n_assets)

    # Last observed epsilon (in % units) and sigma (in % units)
    last_epsilon = np.zeros(n_assets)
    last_sigma   = np.zeros(n_assets)

    for i, t in enumerate(tickers):
        res          = garch_models[t]
        omega[i]     = res.params['omega']
        alpha[i]     = res.params['alpha[1]']
        beta[i]      = res.params['beta[1]']
        mu[i]        = res.params['mu']
        # resid and conditional_volatility are already in % scale
        last_epsilon[i] = res.resid.iloc[-1]
        last_sigma[i]   = res.conditional_volatility.iloc[-1]

    # ── Standardized residuals pool: shape (T, n_assets) ──
    z_pool = std_residuals[tickers].dropna().values
    T_pool = z_pool.shape[0]

    # ── BLOCK BOOTSTRAP ──────────────────────────────────────────
    # Build an index pool of valid block start positions
    max_start    = T_pool - block_size
    n_blocks_needed = int(np.ceil(n_days / block_size))

    # Pre-sample block start indices for all simulations × needed blocks
    # Shape: (n_sims, n_blocks_needed)
    block_starts = np.random.randint(0, max_start + 1,
                                     size=(n_sims, n_blocks_needed))

    # Expand blocks into a flat index array of shape (n_sims, n_blocks_needed * block_size)
    offsets      = np.arange(block_size)                          # (block_size,)
    flat_indices = (block_starts[:, :, None] + offsets[None, None, :]) \
                   .reshape(n_sims, -1)                           # (n_sims, total_days)

    # Clip to valid range and take only the first n_days columns
    flat_indices = np.clip(flat_indices[:, :n_days], 0, T_pool - 1)

    # z_draws shape: (n_days, n_sims, n_assets)
    z_draws = z_pool[flat_indices, :]                             # (n_sims, n_days, n_assets)
    z_draws = z_draws.transpose(1, 0, 2)                         # (n_days, n_sims, n_assets)

    # ── GARCH RECURSION (in % scale) ─────────────────────────────
    sim_returns_daily_pct = np.zeros((n_days, n_sims, n_assets))

    prev_epsilon = np.tile(last_epsilon, (n_sims, 1))   # (n_sims, n_assets) in %
    prev_sigma2  = np.tile(last_sigma**2, (n_sims, 1))  # (n_sims, n_assets) in %²

    for h in range(n_days):
        # GARCH variance update: sigma²_t = omega + alpha·eps²_{t-1} + beta·sigma²_{t-1}
        current_sigma2 = omega + alpha * (prev_epsilon ** 2) + beta * prev_sigma2
        # Numerical safety: clamp variance to a minimum
        current_sigma2 = np.maximum(current_sigma2, 1e-8)
        current_sigma  = np.sqrt(current_sigma2)

        # Simulated innovation: eps_t = sigma_t * z_t  (% scale)
        z_raw          = z_draws[h]                     # (n_sims, n_assets) i.i.d. raw
        # Apply EWMA Cholesky to introduce cross-asset correlation
        z_t            = z_raw @ L_chol.T              # (n_sims, n_assets) correlated
        current_epsilon = current_sigma * z_t

        # Simulated log return: r_t = mu + eps_t  (% scale)
        sim_returns_daily_pct[h] = mu + current_epsilon

        prev_epsilon = current_epsilon
        prev_sigma2  = current_sigma2

    # ── Convert to decimal returns ────────────────────────────────
    sim_returns_daily = sim_returns_daily_pct / 100.0  # (n_days, n_sims, n_assets)

    # ── Cumulative log returns → simple returns ───────────────────
    sim_cum_log   = np.sum(sim_returns_daily, axis=0)           # (n_sims, n_assets)
    sim_cum_simple = np.exp(sim_cum_log) - 1.0

    return sim_cum_simple, sim_returns_daily


def calculate_cvar(weights, simulated_returns, alpha=0.95):
    """
    Calculates CVaR (Expected Shortfall) of a portfolio.

    Args:
        weights            (np.ndarray): shape (n_assets,)
        simulated_returns  (np.ndarray): shape (n_sims, n_assets) — simple returns
        alpha              (float):      confidence level, e.g. 0.95

    Returns:
        cvar_val (float): Expected tail loss (positive = loss)
    """
    port_ret  = np.dot(simulated_returns, weights)
    losses    = -port_ret

    # VaR threshold at the alpha quantile
    var_alpha = np.percentile(losses, alpha * 100)

    # ES: strict — only losses strictly exceeding VaR
    tail_losses = losses[losses > var_alpha]
    if len(tail_losses) == 0:
        return float(var_alpha)
    return float(tail_losses.mean())


def optimize_portfolio(simulated_returns, tickers, alpha=0.95,
                       max_weight=0.15, min_active=5):
    """
    Minimizes CVaR subject to:
      - sum(w) = 1
      - 0 ≤ w_i ≤ max_weight  (no short-selling, diversification cap)
      - Soft HHI penalty to discourage hyper-concentration

    Returns:
        optimal_weights (np.ndarray): optimal weight vector
        opt_res:                      scipy OptimizeResult
    """
    n_assets = len(tickers)

    # Equal-weight initialization
    init_weights = np.ones(n_assets) / n_assets

    bounds      = [(0.0, max_weight) for _ in range(n_assets)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def objective(w):
        cvar = calculate_cvar(w, simulated_returns, alpha=alpha)
        # Soft HHI penalty: penalize portfolios with high concentration
        hhi  = np.sum(w ** 2)                  # min=1/N, max=1
        hhi_penalty = 0.1 * hhi                # weight of penalty
        return cvar + hhi_penalty

    opt_res = minimize(
        objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-9},
    )

    weights = opt_res.x
    # Clip numerical noise
    weights = np.clip(weights, 0.0, max_weight)
    weights /= weights.sum()

    return weights, opt_res


def compute_portfolio_metrics(weights, sim_returns_daily, confidence=0.95):
    """
    Computes annualized portfolio metrics from daily simulation paths.

    Args:
        weights            (np.ndarray): shape (n_assets,)
        sim_returns_daily  (np.ndarray): shape (n_days, n_sims, n_assets) decimal returns
        confidence         (float):      VaR/CVaR confidence level

    Returns:
        dict with keys: exp_return_20d, exp_vol_annual, var_20d, cvar_20d,
                        sharpe_20d, port_sim_ret_20d (array of simple returns)
    """
    n_days, n_sims, n_assets = sim_returns_daily.shape

    # Daily portfolio returns: (n_days, n_sims)
    port_daily = np.einsum('dsa,a->ds', sim_returns_daily, weights)

    # Annualized volatility: mean over simulations of daily std, then annualize
    daily_std_per_sim  = port_daily.std(axis=0)          # (n_sims,) — std across days
    exp_vol_daily      = daily_std_per_sim.mean()         # expected daily vol
    exp_vol_annual     = exp_vol_daily * np.sqrt(252)

    # 20-day cumulative simple returns per simulation
    cum_log_returns    = port_daily.sum(axis=0)           # (n_sims,) log returns
    port_sim_ret_20d   = np.exp(cum_log_returns) - 1.0

    # Expected 20-day return
    exp_return_20d     = port_sim_ret_20d.mean()

    # VaR and CVaR (20-day horizon)
    losses             = -port_sim_ret_20d
    var_20d            = float(np.percentile(losses, confidence * 100))
    tail               = losses[losses > var_20d]
    cvar_20d           = float(tail.mean()) if len(tail) > 0 else var_20d

    # Sharpe Ratio (20-day, adjusted for BCRP risk-free rate)
    excess_ret         = exp_return_20d - RF_20D
    vol_20d            = port_sim_ret_20d.std()
    sharpe_20d         = float(excess_ret / vol_20d) if vol_20d > 0 else 0.0

    return {
        "exp_return_20d":   float(exp_return_20d),
        "exp_vol_annual":   float(exp_vol_annual),
        "var_20d":          var_20d,
        "cvar_20d":         cvar_20d,
        "sharpe_20d":       sharpe_20d,
        "port_sim_ret_20d": port_sim_ret_20d,
    }


if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from econometrics import get_pen_prices_and_volumes, filter_assets_by_volume, fit_garch_models
    from data_pipeline import get_bvl_data

    print("Testing optimization module...")
    df = get_bvl_data()
    prices_pen, volume_pen = get_pen_prices_and_volumes(df)
    prices_filtered, _, valid_tickers = filter_assets_by_volume(prices_pen, volume_pen)
    _, std_residuals, garch_models, _ = fit_garch_models(prices_filtered)

    sim_rets, sim_daily = simulate_garch_paths(garch_models, std_residuals)
    w_opt, res          = optimize_portfolio(sim_rets, valid_tickers, 0.95)
    metrics             = compute_portfolio_metrics(w_opt, sim_daily)

    print("\nOptimal Weights:")
    for t, w in zip(valid_tickers, w_opt):
        print(f"  {t:12s}: {w:.4%}")
    print(f"\nSum of weights:    {w_opt.sum():.6f}")
    print(f"Expected return:   {metrics['exp_return_20d']:.4%}  (20D)")
    print(f"Annual volatility: {metrics['exp_vol_annual']:.4%}")
    print(f"VaR  (95%, 20D):   {metrics['var_20d']:.4%}")
    print(f"CVaR (95%, 20D):   {metrics['cvar_20d']:.4%}")
    print(f"Sharpe (20D, adj): {metrics['sharpe_20d']:.4f}")
