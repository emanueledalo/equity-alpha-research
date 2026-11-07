"""Small statistics helpers shared by the evaluation modules."""

from __future__ import annotations

import numpy as np
import pandas as pd


def newey_west_se(x: np.ndarray, lags: int) -> float:
    """Newey-West (Bartlett kernel) standard error of the mean of `x`.

    Needed whenever observations overlap: with a 21-day forward return measured
    every day, consecutive observations share 20 of their 21 days, so they are
    strongly autocorrelated and the naive std / sqrt(n) overstates significance.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    e = x - x.mean()
    var = e @ e / n
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        var += 2 * weight * (e[lag:] @ e[:-lag]) / n
    return float(np.sqrt(max(var, 0) / n))


def newey_west_tstat(x: pd.Series | np.ndarray, lags: int) -> float:
    x = np.asarray(x, dtype=float)
    mean = np.nanmean(x)
    return float(mean / newey_west_se(x, lags))


def ols_newey_west(y: np.ndarray, X: np.ndarray, lags: int) -> tuple[np.ndarray, np.ndarray]:
    """OLS coefficients and Newey-West (HAC) standard errors. X must include a constant."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    xe = X * resid[:, None]
    S = xe.T @ xe / n
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = xe[lag:].T @ xe[:-lag] / n
        S += weight * (gamma + gamma.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    cov = XtX_inv @ S @ XtX_inv / n
    return beta, np.sqrt(np.diag(cov))
