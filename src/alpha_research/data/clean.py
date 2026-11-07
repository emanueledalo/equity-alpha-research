"""Turn cached raw prices into one clean, analysis-ready panel.

Work is done in *wide* format (rows = trading days, columns = tickers) because
every step is a vectorised operation across the whole cross-section.

Cleaning steps, in order:
  1. Align every series to the benchmark's trading calendar.
  2. Invalid prices      - non-positive or missing close/adj_close  -> removed.
  3. Stale prices        - unchanged close with zero volume for >= N days -> removed
                           (the stock was not trading; zero returns would be fake).
  4. Bad ticks (spikes)  - a large move fully reversed the next day -> removed.
  5. Returns             - simple returns of adj_close, only between *consecutive*
                           trading days (never across a data gap, never forward-filled).
  6. Extreme returns     - |return| > 100% kept but flagged for manual review.

Nothing is forward-filled: a missing price stays missing, and the research code
must handle NaN explicitly. Filling would invent returns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_research.config import DataConfig
from alpha_research.data.prices import COLUMNS, PriceCache

PRICE_FIELDS = ["open", "high", "low", "close", "adj_close"]


@dataclass
class CleanResult:
    wide: dict[str, pd.DataFrame]   # field -> (dates x tickers)
    issues: pd.DataFrame            # date, ticker, issue
    source: pd.Series               # ticker -> data source


def check_series(df: pd.DataFrame, member_dates: pd.DatetimeIndex, cfg: DataConfig) -> tuple[bool, float, str]:
    """Is this series really the company that was in the index?

    Free sources map a ticker to its *current* owner. When a ticker was reused
    (e.g. 'GR' was Goodrich, now an OTC penny stock) the source can return the
    wrong company *during* the membership period. S&P 500 members trade tens of
    millions of dollars a day, so a tiny median dollar volume on member days
    exposes the mismatch.

    Returns (accepted, coverage of member days, reason if rejected).
    """
    if len(member_dates) == 0:
        return False, 0.0, "never a member in window"
    on = df.reindex(member_dates).dropna(subset=["adj_close"])
    coverage = len(on) / len(member_dates)
    if on.empty:
        return False, 0.0, "no price on member days"
    dollar_volume = (on["close"] * on["volume"]).median()
    if dollar_volume < cfg.min_member_dollar_volume:
        return False, coverage, f"median $ volume on member days {dollar_volume:,.0f} (wrong company?)"
    return True, coverage, ""


def choose_sources(caches: list[PriceCache], tickers: list[str], members: pd.DataFrame,
                   cfg: DataConfig = DataConfig()) -> tuple[dict[str, PriceCache], pd.DataFrame]:
    """For each ticker pick the cache whose series passes `check_series` and covers
    the most member days (ties -> earlier cache in the list).

    Returns the choice per ticker and a log of every candidate that was checked.
    """
    chosen, log = {}, []
    for ticker in tickers:
        member_dates = members.index[members[ticker]] if ticker in members else pd.DatetimeIndex([])
        best = None
        for cache in caches:
            if not cache.has(ticker):
                continue
            ok, coverage, reason = check_series(cache.load(ticker), member_dates, cfg)
            log.append({"ticker": ticker, "source": cache.source, "accepted": ok,
                        "coverage": coverage, "reason": reason})
            if ok and (best is None or coverage > best[1]):
                best = (cache, coverage)
        if best:
            chosen[ticker] = best[0]
    return chosen, pd.DataFrame(log, columns=["ticker", "source", "accepted", "coverage", "reason"])


def assemble_raw(chosen: dict[str, PriceCache]) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Load each ticker from its chosen cache into wide (dates x tickers) frames."""
    if not chosen:
        raise ValueError("No price data found in any cache.")
    frames = {ticker: cache.load(ticker) for ticker, cache in chosen.items()}
    big = pd.concat(frames, axis=1)  # columns: (ticker, field)
    wide = {f: big.xs(f, axis=1, level=1) for f in COLUMNS}
    for df in wide.values():
        df.columns.name = "ticker"
    source = pd.Series({t: c.source for t, c in chosen.items()}, name="source")
    return wide, source


def _run_mask(flag: pd.DataFrame, min_len: int) -> pd.DataFrame:
    """True where `flag` is part of a run of at least `min_len` consecutive Trues."""
    out = pd.DataFrame(False, index=flag.index, columns=flag.columns)
    for col in flag.columns:
        f = flag[col]
        run_id = (~f).cumsum()
        run_len = f.groupby(run_id).transform("sum")
        out[col] = f & (run_len >= min_len)
    return out


def _issues(mask: pd.DataFrame, label: str) -> pd.DataFrame:
    stacked = mask.stack()
    hits = stacked[stacked].index.to_frame(index=False)
    hits.columns = ["date", "ticker"]
    hits["issue"] = label
    return hits


def clean_prices(wide: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex,
                 cfg: DataConfig = DataConfig()) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    w = {f: df.reindex(calendar) for f, df in wide.items()}
    issues = []

    def drop(mask: pd.DataFrame, label: str) -> None:
        issues.append(_issues(mask & w["adj_close"].notna(), label))
        for f in COLUMNS:
            w[f] = w[f].mask(mask)

    # 2. invalid prices
    invalid = (w["close"] <= 0) | (w["adj_close"] <= 0) | w["close"].isna() | w["adj_close"].isna()
    drop(invalid, "invalid_price")

    # 3. stale prices: unchanged close and no volume, for a long enough run
    unchanged = (w["close"] == w["close"].shift(1)) & (w["volume"].fillna(0) == 0)
    drop(_run_mask(unchanged, cfg.stale_run), "stale_price")

    # 4. bad ticks: big log move that is (mostly) undone the next day
    lr = np.log(w["adj_close"]).diff()
    nxt = lr.shift(-1)
    spike = ((lr.abs() > cfg.spike_threshold) & (nxt.abs() > cfg.spike_threshold)
             & (np.sign(lr) != np.sign(nxt)) & ((lr + nxt).abs() <= cfg.spike_reversal * lr.abs()))
    drop(spike, "spike_reverted")

    # 5. returns between consecutive trading days only (NaN if either day is missing)
    w["ret"] = w["adj_close"] / w["adj_close"].shift(1) - 1.0
    # Adjusted open: same adjustment factor as the close; needed for open-to-open execution.
    w["adj_open"] = w["open"] * w["adj_close"] / w["close"]
    w["dollar_volume"] = w["close"] * w["volume"]

    # 6. extreme returns: flag only
    issues.append(_issues(w["ret"].abs() > cfg.extreme_return, "extreme_return_flag"))

    issues_df = pd.concat(issues, ignore_index=True) if issues else pd.DataFrame(columns=["date", "ticker", "issue"])
    return w, issues_df


def trim_after_membership(wide: dict[str, pd.DataFrame], members: pd.DataFrame,
                          keep_days: int = 63) -> dict[str, pd.DataFrame]:
    """Drop each stock's prices from `keep_days` trading days after it last left the index.

    A stock is only needed while it can be held: during membership, plus a short
    tail to close the position after it leaves. Beyond that, data is not just
    useless but risky: a reused ticker can point to another company (e.g. Tiingo's
    'CPWR' continues years after Compuware was taken private, as a different firm).
    Data *before* membership is kept, because signals need a year of history.
    """
    members = members.reindex(index=wide["adj_close"].index, columns=wide["adj_close"].columns, fill_value=False)
    last_member_pos = members.to_numpy()[::-1].argmax(axis=0)          # rows from the end
    ever = members.any().to_numpy()
    cutoff = len(members) - 1 - last_member_pos + keep_days             # last row we keep
    rows = np.arange(len(members))[:, None]
    keep = ~ever[None, :] | (rows <= cutoff[None, :])
    keep_df = pd.DataFrame(keep, index=members.index, columns=members.columns)
    return {field: df.where(keep_df) for field, df in wide.items()}


def coverage_checks(adj_close: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Per ticker: does price data cover the period the stock was in the index?

    A price series that starts well after the stock joined the index often means
    the ticker was reused by another company (e.g. 'AAL' was AMR Corp in 1996),
    so the source only has the newer company. Those days are simply missing,
    never mixed with the wrong company's prices.
    """
    members = members.reindex(columns=adj_close.columns.union(members.columns), fill_value=False)
    has_price = adj_close.reindex(columns=members.columns).notna()
    member_days = members.sum()
    covered = (members & has_price).sum()
    out = pd.DataFrame({
        "member_days": member_days,
        "covered_days": covered,
        "coverage": (covered / member_days.replace(0, np.nan)),
        "first_member": members.idxmax().where(member_days > 0),
        "first_price": has_price.idxmax().where(has_price.any()),
    })
    return out[out["member_days"] > 0].sort_values("coverage")


def to_long(w: dict[str, pd.DataFrame], members: pd.DataFrame, source: pd.Series) -> pd.DataFrame:
    """Stack into a long (date, ticker) panel; keep a row if there is a price or
    the stock was an index member that day (so missing data stays visible)."""
    fields = ["adj_open", "adj_close", "close", "volume", "dollar_volume", "ret"]
    tickers = w["adj_close"].columns.union(members.columns)
    members = members.reindex(index=w["adj_close"].index, columns=tickers, fill_value=False)
    data = {f: w[f].reindex(columns=tickers).stack(future_stack=True) for f in fields}
    data["in_sp500"] = members.stack(future_stack=True)
    panel = pd.DataFrame(data)
    panel = panel[panel["adj_close"].notna() | panel["in_sp500"]]
    panel["source"] = panel.index.get_level_values("ticker").map(source).fillna("none")
    panel.index.names = ["date", "ticker"]
    return panel
