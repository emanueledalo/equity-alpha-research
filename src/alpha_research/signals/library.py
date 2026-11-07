"""Price- and volume-based signals.

Every function takes wide (dates x tickers) inputs and returns a wide frame of the
*raw* signal on day t, computed from data up to and including the close of t only.
`shift` and backward-looking `rolling` windows are the only time operations used,
which is what guarantees there is no look-ahead (tested in tests/test_signals.py).

The `sign` in the registry orients each raw signal so that higher = higher expected
return, following the paper that documented the effect. Whether these effects
still exist in large US stocks after 2010 is exactly what Stage 3 will test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

MIN_OBS_SHARE = 0.8  # a rolling window needs >= 80% non-missing observations


@dataclass(frozen=True)
class MarketData:
    ret: pd.DataFrame            # daily simple returns (NaN across data gaps)
    adj_close: pd.DataFrame      # adjusted close
    dollar_volume: pd.DataFrame  # close * volume
    mkt_ret: pd.Series           # benchmark (SPY) daily return on the same calendar

    @property
    def log_ret(self) -> pd.DataFrame:
        return np.log1p(self.ret)


def _min_obs(window: int) -> int:
    return math.ceil(MIN_OBS_SHARE * window)


def _rolling(df: pd.DataFrame, window: int):
    return df.rolling(window, min_periods=_min_obs(window))


# --------------------------------------------------------------------------- momentum / reversal

def past_return(d: MarketData, start: int, end: int = 0) -> pd.DataFrame:
    """Cumulative log return from t-start to t-end (end > 0 skips the most recent days)."""
    return _rolling(d.log_ret.shift(end), start - end).sum()


def mom_12_1(d: MarketData) -> pd.DataFrame:
    # Skip the last month: it carries short-term reversal, which works against momentum.
    return past_return(d, 252, 21)


def mom_6_1(d: MarketData) -> pd.DataFrame:
    return past_return(d, 126, 21)


def rev_1m(d: MarketData) -> pd.DataFrame:
    return past_return(d, 21)


def rev_1w(d: MarketData) -> pd.DataFrame:
    return past_return(d, 5)


def high_52w(d: MarketData) -> pd.DataFrame:
    return d.adj_close / _rolling(d.adj_close, 252).max()


# --------------------------------------------------------------------------- risk

def volatility(d: MarketData, window: int = 63) -> pd.DataFrame:
    return _rolling(d.ret, window).std()


def market_model(d: MarketData, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rolling beta and residual (idiosyncratic) variance of r_i = a + b * r_m + e.

    Vectorised over all stocks with rolling sums. The market return is masked to
    the days each stock has a return, so every moment uses the same sample.
    """
    r = d.ret
    m = pd.DataFrame(np.where(r.notna(), d.mkt_ret.reindex(r.index).to_numpy()[:, None], np.nan),
                     index=r.index, columns=r.columns)
    roll = lambda x: _rolling(x, window).sum()  # noqa: E731
    n = _rolling(r, window).count()
    s_r, s_m = roll(r), roll(m)
    cov = (roll(r * m) - s_r * s_m / n) / (n - 1)
    var_m = (roll(m * m) - s_m ** 2 / n) / (n - 1)
    var_r = (roll(r * r) - s_r ** 2 / n) / (n - 1)
    beta = cov / var_m
    resid_var = (var_r - beta ** 2 * var_m).clip(lower=0)
    return beta, resid_var


def beta(d: MarketData) -> pd.DataFrame:
    return market_model(d, 252)[0]


def idio_vol(d: MarketData) -> pd.DataFrame:
    return np.sqrt(market_model(d, 63)[1])


def residual_returns(d: MarketData, window: int = 252) -> pd.DataFrame:
    """Daily market-model residuals e_t = r_t - beta_{t-1} * m_t.

    beta_{t-1} is estimated on data up to the previous day only, so today's residual
    does not use today's return to estimate its own beta.
    """
    b, _ = market_model(d, window)
    return d.ret - b.shift(1).mul(d.mkt_ret, axis=0)


def residual_momentum(d: MarketData) -> pd.DataFrame:
    """12-1 month sum of residual returns divided by their volatility.

    Raw momentum is dominated by the market and factor moves of the past year and
    crashes when they reverse; momentum in the stock-specific part of returns is
    more stable (Blitz, Huij & Martens, 2011). Dividing by residual volatility makes
    it a t-statistic-like measure of how *consistently* the stock outperformed.
    Simplification: market-model residuals instead of Fama-French 3-factor residuals.
    """
    e = residual_returns(d).shift(21)
    window = 231
    return _rolling(e, window).sum() / (_rolling(e, window).std() * np.sqrt(window))


def residual_reversal(d: MarketData) -> pd.DataFrame:
    """1-month sum of residual returns: reversal of the stock-specific move only."""
    return _rolling(residual_returns(d), 21).sum()


def max_ret(d: MarketData) -> pd.DataFrame:
    return _rolling(d.ret, 21).max()


def skew(d: MarketData) -> pd.DataFrame:
    return _rolling(d.ret, 63).skew()


# --------------------------------------------------------------------------- liquidity / volume

def _dollar_volume(d: MarketData) -> pd.DataFrame:
    return d.dollar_volume.where(d.dollar_volume > 0)


def amihud(d: MarketData) -> pd.DataFrame:
    """Price impact per dollar traded; log because it spans orders of magnitude."""
    impact = d.ret.abs() / _dollar_volume(d) * 1e6  # |return| per $1M traded
    return np.log(_rolling(impact, 63).mean())


def volume_shock(d: MarketData) -> pd.DataFrame:
    dv = _dollar_volume(d)
    return np.log(_rolling(dv, 21).mean() / _rolling(dv, 252).mean())


# --------------------------------------------------------------------------- registry

@dataclass(frozen=True)
class Signal:
    name: str
    func: Callable[[MarketData], pd.DataFrame]
    sign: int
    description: str
    reference: str


SIGNALS: list[Signal] = [
    Signal("mom_12_1", mom_12_1, +1, "12-month return skipping the last month", "Jegadeesh & Titman (1993)"),
    Signal("mom_6_1", mom_6_1, +1, "6-month return skipping the last month", "Jegadeesh & Titman (1993)"),
    Signal("rev_1m", rev_1m, -1, "1-month return (reversal)", "Jegadeesh (1990)"),
    Signal("rev_1w", rev_1w, -1, "1-week return (reversal)", "Lehmann (1990)"),
    Signal("high_52w", high_52w, +1, "price / 52-week high", "George & Hwang (2004)"),
    Signal("low_vol", lambda d: volatility(d, 63), -1, "3-month daily volatility",
           "Ang, Hodrick, Xing & Zhang (2006)"),
    Signal("beta", beta, -1, "12-month market beta vs SPY", "Frazzini & Pedersen (2014)"),
    Signal("idio_vol", idio_vol, -1, "3-month idiosyncratic volatility", "Ang, Hodrick, Xing & Zhang (2006)"),
    Signal("max_ret", max_ret, -1, "max daily return over 1 month (lottery demand)",
           "Bali, Cakici & Whitelaw (2011)"),
    Signal("skew", skew, -1, "3-month skewness of daily returns", "Boyer, Mitton & Vorkink (2010)"),
    Signal("amihud", amihud, +1, "Amihud illiquidity, 3 months", "Amihud (2002)"),
    Signal("volume_shock", volume_shock, +1, "1-month vs 12-month dollar volume",
           "Gervais, Kaniel & Mingelgrin (2001)"),
    # Extensions added after the first evaluation round (see docs): held to the same t >= 3 hurdle.
    Signal("res_mom", residual_momentum, +1, "12-1 month residual (market-model) momentum, vol-scaled",
           "Blitz, Huij & Martens (2011)"),
    Signal("res_rev", residual_reversal, -1, "1-month residual return (reversal)",
           "Blitz, Huij, Lansdorp & Verbeek (2013)"),
]

SIGNALS_BY_NAME = {s.name: s for s in SIGNALS}
