"""Information coefficient (IC) analysis.

The rank IC on day t is the Spearman correlation, across the stocks in the
universe that day, between the signal and the subsequent return. It measures
ranking skill directly, without any portfolio construction choices, and is
robust to outliers because it uses ranks. In equities a mean IC of 0.02-0.05
is already a useful signal: what matters is that it is *consistently* positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.evaluation.stats import newey_west_tstat

TRADING_DAYS = 252


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame, min_count: int = 30) -> pd.Series:
    """Daily Spearman correlation between `signal` and `fwd` (both dates x tickers)."""
    fwd = fwd.reindex_like(signal)
    valid = signal.notna() & fwd.notna()
    s = signal.where(valid).rank(axis=1)
    f = fwd.where(valid).rank(axis=1)
    s = s.sub(s.mean(axis=1), axis=0)
    f = f.sub(f.mean(axis=1), axis=0)
    ic = (s * f).sum(axis=1) / np.sqrt((s ** 2).sum(axis=1) * (f ** 2).sum(axis=1))
    return ic.where(valid.sum(axis=1) >= min_count)


def ic_summary(ic: pd.Series, horizon: int) -> dict[str, float]:
    """Mean IC, Newey-West t-stat (horizon-1 lags for overlapping returns), IC IR, hit rate."""
    ic = ic.dropna()
    # IC IR: mean/std of IC, annualised with the number of *independent* bets per year.
    ir = ic.mean() / ic.std() * np.sqrt(TRADING_DAYS / horizon)
    return {
        "mean_ic": ic.mean(),
        "t_stat": newey_west_tstat(ic, lags=max(horizon - 1, 1)),
        "ic_ir": ir,
        "hit_rate": (ic > 0).mean(),
        "n_days": len(ic),
    }


def ic_decay(signal: pd.DataFrame, fwd: pd.DataFrame, lags=(0, 5, 10, 21, 42, 63)) -> pd.Series:
    """Mean IC of the signal observed `lag` days ago. A fast decay means the signal
    must be traded quickly (high turnover); a slow decay allows monthly rebalancing."""
    return pd.Series({lag: rank_ic(signal.shift(lag), fwd).mean() for lag in lags}, name="mean_ic")


def ic_by_year(ic: pd.Series) -> pd.Series:
    return ic.groupby(ic.index.year).mean()
