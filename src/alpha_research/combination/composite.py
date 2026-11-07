"""Build the composite score from signal weights, and the ridge (ML) combination."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.signals.preprocess import zscore


def signal_cube(z: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> tuple[np.ndarray, list[str], pd.Index]:
    """(dates x tickers x signals) array of z-scores on the given dates."""
    names = list(z)
    tickers = z[names[0]].columns
    cube = np.stack([z[n].reindex(index=dates, columns=tickers).to_numpy() for n in names], axis=-1)
    return cube, names, tickers


def composite_scores(z: dict[str, pd.DataFrame], weights: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """score_i = sum_k w_k z_ik, then re-standardised across stocks each date.

    A missing signal counts as 0 (= cross-sectional average, i.e. no view); a stock
    needs at least `min_share` of the signals to get a score at all.
    """
    cube, names, tickers = signal_cube(z, weights.index)
    w = weights[names].to_numpy()                                  # dates x signals
    available = (~np.isnan(cube)).sum(axis=-1) >= min_share * len(names)
    score = np.einsum("dts,ds->dt", np.nan_to_num(cube), w)
    score[~available] = np.nan
    return zscore(pd.DataFrame(score, index=weights.index, columns=tickers))


def ridge_walk_forward(z: dict[str, pd.DataFrame], hret: pd.DataFrame, known_at: pd.Series,
                       dates: pd.DatetimeIndex, lam: float = 1.0, min_periods: int = 36) -> pd.DataFrame:
    """Pooled ridge regression of next-period stock returns on the signals.

    For each training period m, regress the cross-sectionally demeaned stock return
    (removing the market move, which no stock-selection signal can predict) on the
    K z-scores. Pooled over all known periods:

        beta = (X'X / n + lam * I)^-1 X'y / n

    With unit-variance signals, X'X/n is roughly their correlation matrix, so `lam`
    is on an interpretable scale; lam = 1 is strong shrinkage, fixed a priori rather
    than tuned (tuning it on the same history would be another source of overfitting).
    The per-period sums are cached so each refit is a K x K solve.
    """
    cube, names, _ = signal_cube(z, hret.index)
    k = len(names)
    xtx, xty, nobs = {}, {}, {}
    for i, period in enumerate(hret.index):
        y = hret.loc[period].to_numpy()
        X = cube[i]
        ok = ~np.isnan(y) & ((~np.isnan(X)).sum(axis=1) >= k / 2)
        X, y = np.nan_to_num(X[ok]), y[ok] - np.nanmean(y[ok])
        xtx[period], xty[period], nobs[period] = X.T @ X, X.T @ y, ok.sum()

    known_at = known_at.reindex(hret.index)
    rows = {}
    for date in dates:
        used = hret.index[known_at <= date]
        if len(used) < min_periods:
            continue
        n = sum(nobs[p] for p in used)
        A = sum(xtx[p] for p in used) / n + lam * np.eye(k)
        b = sum(xty[p] for p in used) / n
        beta = np.linalg.solve(A, b)
        rows[date] = pd.Series(beta / np.abs(beta).sum(), index=names)
    return pd.DataFrame(rows).T
