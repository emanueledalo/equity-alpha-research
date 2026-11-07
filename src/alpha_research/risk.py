"""A small statistical risk model: market beta + principal components of returns.

Why: the baseline book is hedged against the market only. The largest remaining
common moves in large caps are sector/industry and style rotations (e.g. tech vs
energy, growth vs value). Point-in-time sector data is not free, but these common
moves can be *estimated* from returns: the leading principal components of the
trailing return covariance are, in practice, a market factor followed by sector
and style factors. Neutralising signals, training returns and the portfolio to
them removes noise that the signals are not meant to bet on (and that otherwise
dominates the P&L of a long-short book).

Exposures on rebalance date t use only returns up to t (trailing window), so they
are known at decision time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PCA = 5            # fixed a priori: a market-like PC plus ~4 sector/style factors
PCA_WINDOW = 252     # one year of daily returns
MIN_OBS_SHARE = 0.8


def exposures(ret: pd.DataFrame, beta: pd.DataFrame, dates: pd.DatetimeIndex,
              n_pca: int = N_PCA, window: int = PCA_WINDOW) -> dict[pd.Timestamp, pd.DataFrame]:
    """Per rebalance date: (tickers x [const, beta, pc1..pcK]) exposure matrix.

    PCA on the standardised trailing returns of stocks with enough history (missing
    days filled with 0 after standardisation). Stocks without enough history get
    zero PC loadings: they are only neutralised to the constant and beta.
    """
    out = {}
    pos = ret.index.get_indexer(dates)
    for date, p in zip(dates, pos):
        cols = ret.columns
        b = beta.loc[date].reindex(cols).fillna(1.0).clip(-1, 3)
        E = pd.DataFrame({"const": 1.0, "beta": b}, index=cols)
        if n_pca > 0 and p >= window:
            block = ret.iloc[p - window + 1: p + 1]
            ok = block.notna().mean() >= MIN_OBS_SHARE
            X = block.loc[:, ok]
            X = (X - X.mean()) / X.std().replace(0, np.nan)
            X = X.fillna(0.0).to_numpy()
            if X.shape[1] > n_pca * 5:
                # Right singular vectors = eigenvectors of the stock-by-stock correlation matrix.
                _, _, vt = np.linalg.svd(X, full_matrices=False)
                load = pd.DataFrame(vt[:n_pca].T, index=block.columns[ok],
                                    columns=[f"pc{k + 1}" for k in range(n_pca)])
                E = E.join(load).fillna({f"pc{k + 1}": 0.0 for k in range(n_pca)})
        out[date] = E
    return out


def neutralise_frame(values: pd.DataFrame, expo: dict[pd.Timestamp, pd.DataFrame], min_count: int = 20
                     ) -> pd.DataFrame:
    """Row by row, the residual of `values` after a cross-sectional regression on the
    exposures of that date (only stocks with a value are used)."""
    out = values * np.nan
    for date in values.index:
        if date not in expo:
            continue
        y = values.loc[date]
        ok = y.notna()
        if ok.sum() < min_count:
            continue
        A = expo[date].reindex(y.index[ok]).fillna(0.0).to_numpy()
        coef, *_ = np.linalg.lstsq(A, y[ok].to_numpy(), rcond=None)
        out.loc[date, ok] = y[ok].to_numpy() - A @ coef
    return out
