"""Rebalance schedule, holding-period returns and per-signal returns.

Everything downstream works on *holding periods*: a position decided at the close
of rebalance date t_m is entered at the close of t_m + 1 and earns daily returns
from day t_m + 2 up to and including day t_{m+1} + 1 (when the next rebalance's
trades are executed).

The crucial bookkeeping is *when each holding-period return becomes known*: the
return of period m is only known at the close of t_{m+1} + 1. So at rebalance date
t_m the most recent fully known period is m-2, not m-1. Using period m-1 to set the
weights at t_m would leak one day of future returns into the weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EXECUTION_DELAY = 1  # trade at the close of t+1 (same convention as signals/targets.py)


def rebalance_dates(calendar: pd.DatetimeIndex, freq: str = "M") -> pd.DatetimeIndex:
    """Last trading day of each month ("M") or week ("W") in the calendar."""
    period = calendar.to_period(freq)
    return pd.DatetimeIndex(calendar.to_series().groupby(period).max())


def holding_periods(calendar: pd.DatetimeIndex, rebalance: pd.DatetimeIndex) -> pd.DataFrame:
    """One row per rebalance date: first/last return day (calendar positions) and the
    date at which the period's return is known. Incomplete last periods are dropped."""
    pos = calendar.get_indexer(rebalance)
    start = pos + EXECUTION_DELAY + 1
    end = np.append(start[1:] - 1, len(calendar))   # last period: no end yet
    df = pd.DataFrame({"start": start, "end": end}, index=rebalance)
    df = df[(df["start"] < len(calendar)) & (df["end"] < len(calendar)) & (df["end"] >= df["start"])]
    df["known_at"] = calendar[df["end"].to_numpy()]
    return df


def holding_returns(ret: pd.DataFrame, periods: pd.DataFrame, min_share: float = 0.5) -> pd.DataFrame:
    """Compounded return of each stock over each holding period (rebalance dates x tickers).

    Days with a missing return count as 0 (e.g. after a delisting the position is
    assumed closed at the last price); a stock needs returns on at least
    `min_share` of the period's days, otherwise NaN.
    """
    lr = np.log1p(ret.to_numpy())
    cum = np.vstack([np.zeros(lr.shape[1]), np.nancumsum(lr, axis=0)])
    cnt = np.vstack([np.zeros(lr.shape[1]), np.cumsum(~np.isnan(lr), axis=0)])
    s, e = periods["start"].to_numpy(), periods["end"].to_numpy() + 1
    total = np.expm1(cum[e] - cum[s])
    n_obs = cnt[e] - cnt[s]
    total[n_obs < min_share * (e - s)[:, None]] = np.nan
    return pd.DataFrame(total, index=periods.index, columns=ret.columns)


def signal_returns(z: dict[str, pd.DataFrame], hret: pd.DataFrame) -> pd.DataFrame:
    """Return of each signal over each holding period (rebalance dates x signals).

    f_k,m = sum_i z_ik R_im / sum_i z_ik^2 is the cross-sectional regression slope of
    stock returns on the signal: the return of a portfolio with unit exposure to the
    signal (Fama-MacBeth). It is the natural 'payoff' of a signal and is what the
    combination methods learn from.
    """
    out = {}
    for name, sig in z.items():
        s = sig.reindex(index=hret.index, columns=hret.columns)
        valid = s.notna() & hret.notna()
        s0, r0 = s.where(valid, 0.0), hret.where(valid, 0.0)
        out[name] = (s0 * r0).sum(axis=1) / (s0 ** 2).sum(axis=1).replace(0, np.nan)
    return pd.DataFrame(out)
