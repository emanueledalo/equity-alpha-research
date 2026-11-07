"""Cross-sectional preprocessing of signals.

A raw signal (e.g. 12-month return) is not comparable across days: its level and
spread change with the market regime. What matters for a long-short portfolio is
the *ranking* of stocks on a given day. So each day, among the stocks in the
universe that day only:

  1. winsorize - clip to the 1st/99th percentile so a handful of outliers
                 (e.g. a stock up 400%) do not dominate the z-score;
  2. z-score   - subtract the cross-sectional mean, divide by the std.

After this every signal has mean 0 and std 1 each day, and signals can be
compared and combined on the same scale.
"""

from __future__ import annotations

import pandas as pd


def universe_mask(members: pd.DataFrame, adj_close: pd.DataFrame) -> pd.DataFrame:
    """True where a stock is tradable on day t: an index member *and* priced that day."""
    members = members.reindex(index=adj_close.index, columns=adj_close.columns, fill_value=False)
    return members.astype(bool) & adj_close.notna()


def winsorize(signal: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """Clip each row (day) to its own [lower, upper] quantiles."""
    lo = signal.quantile(lower, axis=1)
    hi = signal.quantile(upper, axis=1)
    return signal.clip(lower=lo, upper=hi, axis=0)


def zscore(signal: pd.DataFrame, min_count: int = 20) -> pd.DataFrame:
    """Cross-sectional z-score per day; days with fewer than `min_count` stocks -> NaN."""
    mean = signal.mean(axis=1)
    std = signal.std(axis=1)
    z = signal.sub(mean, axis=0).div(std, axis=0)
    too_few = signal.notna().sum(axis=1) < min_count
    z.loc[too_few] = float("nan")
    return z


def process(raw: pd.DataFrame, mask: pd.DataFrame, sign: int = 1) -> pd.DataFrame:
    """Raw signal -> oriented (higher = better), universe-restricted, winsorized z-score."""
    oriented = sign * raw.reindex(index=mask.index, columns=mask.columns)
    return zscore(winsorize(oriented.where(mask)))
