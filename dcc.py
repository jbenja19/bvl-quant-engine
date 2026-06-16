"""
dcc.py — Dynamic Conditional Correlations via EWMA (JPMorgan RiskMetrics)
==========================================================================
Implements the industry-standard EWMA covariance estimator:

    Σ_t = λ·Σ_{t-1} + (1-λ)·r_{t-1}·r'_{t-1}

with λ=0.94 (daily), the value calibrated by JPMorgan RiskMetrics.

This captures time-varying correlations — a critical feature because
correlations spike toward 1 during crises (the "correlation breakdown"
phenomenon observed in 2008, 2015 commodity crash, and COVID-2020).

References:
    - Engle, R.F. (2002). "Dynamic Conditional Correlation". JBES.
    - J.P. Morgan/Reuters (1996). "RiskMetrics Technical Document", 4th ed.
"""

import numpy as np
import pandas as pd


# JPMorgan RiskMetrics standard decay factor (daily)
LAMBDA_DAILY = 0.94


def compute_ewma_covariance(log_returns: pd.DataFrame,
                             lam: float = LAMBDA_DAILY) -> list[np.ndarray]:
    """
    Compute a time series of EWMA covariance matrices for the given log returns.

    Args:
        log_returns (pd.DataFrame): T×N matrix of daily log returns (decimal scale).
        lam (float): Decay factor. Default 0.94 (RiskMetrics standard).

    Returns:
        cov_matrices (list[np.ndarray]): List of N×N covariance matrices,
            one per time step starting from day 1.
    """
    rets = log_returns.dropna().values   # (T, N)
    T, N = rets.shape

    # Initialise with the sample covariance of the first 60 days
    init_window = min(60, T // 4)
    sigma = np.cov(rets[:init_window].T)  # N×N seed

    cov_matrices = []
    for t in range(init_window, T):
        r = rets[t - 1].reshape(-1, 1)   # N×1 column vector
        sigma = lam * sigma + (1 - lam) * (r @ r.T)
        cov_matrices.append(sigma.copy())

    return cov_matrices


def cov_to_correlation(cov: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    std = np.sqrt(np.diag(cov))
    # Guard against zero or negative variances
    std = np.where(std < 1e-12, 1.0, std)
    corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    return corr


def get_latest_ewma_correlation(log_returns: pd.DataFrame,
                                 lam: float = LAMBDA_DAILY) -> pd.DataFrame:
    """
    Return the EWMA correlation matrix estimated from the most recent data point.
    This is the correlation matrix used in the Monte Carlo simulation.

    Args:
        log_returns (pd.DataFrame): T×N log return matrix.
        lam (float): Decay factor.

    Returns:
        corr_df (pd.DataFrame): N×N correlation matrix with ticker labels.
    """
    cov_matrices = compute_ewma_covariance(log_returns, lam=lam)
    if not cov_matrices:
        # Fallback: static sample correlation
        return log_returns.corr()

    latest_cov = cov_matrices[-1]
    latest_corr = cov_to_correlation(latest_cov)
    tickers = log_returns.dropna(how="all", axis=1).columns.tolist()

    # Align size if some tickers were dropped
    n = latest_corr.shape[0]
    tickers = tickers[:n]
    return pd.DataFrame(latest_corr, index=tickers, columns=tickers)


def get_ewma_correlation_series(log_returns: pd.DataFrame,
                                 lam: float = LAMBDA_DAILY,
                                 pair: tuple[str, str] = None) -> pd.Series:
    """
    Return the time series of EWMA pairwise correlation for a specific asset pair.
    Useful for charting how two assets' correlation evolved over time.

    Args:
        log_returns (pd.DataFrame): T×N log returns.
        lam (float): Decay factor.
        pair (tuple): (ticker_a, ticker_b). If None, returns full tensor.

    Returns:
        pd.Series of pairwise correlations indexed by date.
    """
    clean = log_returns.dropna()
    tickers = clean.columns.tolist()
    cov_matrices = compute_ewma_covariance(clean, lam=lam)

    dates = clean.index[min(60, len(clean) // 4):]

    if pair is not None:
        i = tickers.index(pair[0])
        j = tickers.index(pair[1])
        corr_series = []
        for cov in cov_matrices:
            corr = cov_to_correlation(cov)
            corr_series.append(corr[i, j])
        return pd.Series(corr_series, index=dates[:len(corr_series)], name=f"{pair[0]}/{pair[1]}")

    return cov_matrices  # raw if no pair specified


def get_cholesky_for_simulation(log_returns: pd.DataFrame,
                                 lam: float = LAMBDA_DAILY) -> np.ndarray:
    """
    Compute the Cholesky decomposition of the latest EWMA correlation matrix
    for use in correlated Monte Carlo sampling.

    Returns:
        L (np.ndarray): Lower triangular Cholesky factor. Shape (N, N).
    """
    corr_df = get_latest_ewma_correlation(log_returns, lam=lam)
    corr = corr_df.values.astype(float)

    # Ensure positive semi-definiteness (numerical fix)
    min_eig = np.linalg.eigvalsh(corr).min()
    if min_eig < 1e-8:
        corr += np.eye(corr.shape[0]) * (abs(min_eig) + 1e-6)

    try:
        L = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Fallback: identity (uncorrelated)
        L = np.eye(corr.shape[0])

    return L
