"""Quantile portfolios: the portfolio view of a signal.

At each rebalance date (month-end) stocks are sorted into quantiles of the signal
and held, equally weighted, until the next rebalance. If a signal works, average
returns should rise from Q1 (lowest signal) to Q5 (highest), and the long-short
portfolio Q5 - Q1 isolates the signal from the market's direction.

Timing (must match signals/targets.py): the signal of close t is traded at the
close of t+1, so the first daily return the position earns is that of day t+2.
Portfolio returns are built from *daily* stock returns rather than from 21-day
forward returns: a stock that delists mid-month simply drops out from that day on,
instead of making the whole month's return missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_research.combination.periods import EXECUTION_DELAY  # one timing convention for the whole project


def quantile_labels(signal: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """1..n per stock and date (n = highest signal), NaN where the signal is missing."""
    pct = signal.rank(axis=1, pct=True)
    return np.ceil(pct * n_quantiles).clip(1, n_quantiles)


def holdings(signal: pd.DataFrame, rebalance: pd.DatetimeIndex, n_quantiles: int = 5) -> pd.DataFrame:
    """Quantile label each stock is *held in* on each day, accounting for the execution delay.

    The label decided on rebalance date t applies from day t+1+1 (first return day)
    until the first return day of the next rebalance.
    """
    calendar = signal.index
    rebalance = calendar.intersection(rebalance)
    labels = quantile_labels(signal.loc[rebalance], n_quantiles).to_numpy()

    # Position (in the calendar) of the first return day of each rebalance.
    start = calendar.get_indexer(rebalance) + EXECUTION_DELAY + 1
    keep = start < len(calendar)
    start, labels = start[keep], labels[keep]

    # Each day uses the *whole row* of the latest rebalance that has started, so a
    # stock missing from the new sort is not carried over from the previous one.
    which = np.searchsorted(start, np.arange(len(calendar)), side="right") - 1
    out = np.full((len(calendar), signal.shape[1]), np.nan)
    active = which >= 0
    out[active] = labels[which[active]]
    return pd.DataFrame(out, index=calendar, columns=signal.columns)


def quantile_returns(held: pd.DataFrame, ret: pd.DataFrame, n_quantiles: int = 5) -> pd.DataFrame:
    """Equal-weighted daily return of each quantile portfolio (columns Q1..Qn)."""
    ret = ret.reindex_like(held)
    out = {}
    for q in range(1, n_quantiles + 1):
        out[f"Q{q}"] = ret.where(held == q).mean(axis=1)
    return pd.DataFrame(out)


def turnover(signal: pd.DataFrame, rebalance: pd.DatetimeIndex, quantile: int, n_quantiles: int = 5) -> pd.Series:
    """Share of the names in a quantile replaced at each rebalance (0 = same names, 1 = all new)."""
    labels = quantile_labels(signal.loc[signal.index.intersection(rebalance)], n_quantiles)
    members = labels == quantile
    prev = members.shift(1, fill_value=False)
    changed = (members & ~prev).sum(axis=1)
    return (changed / members.sum(axis=1)).iloc[1:]
