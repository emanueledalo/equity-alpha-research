"""Point-in-time S&P 500 membership.

Source: https://github.com/fja05680/sp500 (MIT licence). The CSV has one row per
index change date with the full list of constituents effective from that date.
Membership on any trading day is therefore the most recent snapshot on or before
that day. Using it avoids survivorship bias in the *universe*: we only trade
stocks that were actually in the index at the time, including those later
removed, acquired or bankrupt.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

MEMBERSHIP_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)


def normalize_ticker(ticker: str) -> str:
    """Use one ticker convention everywhere: upper case, class shares with '-'
    (BRK.B -> BRK-B), which is what Yahoo and Tiingo expect."""
    return ticker.strip().upper().replace(".", "-")


def download_membership(dest: Path, url: str = MEMBERSHIP_URL, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_snapshots(path: Path) -> pd.Series:
    """Return a Series indexed by effective date; each value is a frozenset of tickers."""
    raw = pd.read_csv(path, parse_dates=["date"])
    raw = raw.sort_values("date").drop_duplicates("date", keep="last")
    members = [
        frozenset(normalize_ticker(t) for t in row.split(",") if t.strip())
        for row in raw["tickers"]
    ]
    return pd.Series(members, index=pd.DatetimeIndex(raw["date"], name="date"), name="tickers")


def tickers_between(snapshots: pd.Series, start: str, end: str | None = None) -> list[str]:
    """Every ticker that was a member at some point in [start, end]."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else snapshots.index.max()
    # The snapshot already in effect on `start` plus all later changes up to `end`.
    first = max(snapshots.index.searchsorted(start_ts, side="right") - 1, 0)
    in_window = snapshots.iloc[first:][snapshots.index[first:] <= end_ts]
    return sorted(set().union(*in_window))


def membership_matrix(snapshots: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Boolean (dates x tickers) frame: True if the ticker was in the index that day.

    Dates before the first snapshot are all False.
    """
    tickers = sorted(set().union(*snapshots))
    column = {t: i for i, t in enumerate(tickers)}
    onehot = np.zeros((len(snapshots), len(tickers)), dtype=bool)
    for i, members in enumerate(snapshots):
        onehot[i, [column[t] for t in members]] = True

    # For each date, position of the last snapshot with effective date <= date.
    pos = snapshots.index.searchsorted(dates, side="right") - 1
    out = np.zeros((len(dates), len(tickers)), dtype=bool)
    valid = pos >= 0
    out[valid] = onehot[pos[valid]]
    return pd.DataFrame(out, index=dates, columns=pd.Index(tickers, name="ticker"))
