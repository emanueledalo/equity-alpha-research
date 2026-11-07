"""Daily price downloads with a per-ticker parquet cache.

Two sources, used in this order:

1. yfinance  - free, no key, fast, but it has NO delisted stocks (e.g. XLNX, CELG,
               TWTR all return nothing). Using it alone would bias the universe
               towards survivors.
2. Tiingo    - free API key, keeps history for delisted tickers. The free tier
               allows 50 requests/hour and 500 distinct symbols/month, so it is
               only used for the tickers yfinance cannot serve.

Every series is stored with the same columns:
    open, high, low, close, volume  - as reported by the source
    adj_close                       - split- and dividend-adjusted close (for returns)
Only `adj_close` (returns) and `close * volume` (dollar volume) are used
downstream; both mean the same thing for both sources.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

COLUMNS = ["open", "high", "low", "close", "volume", "adj_close"]


class PriceCache:
    """One parquet file per ticker under <root>/<source>/, plus a list of tickers
    the source does not have, so reruns do not ask for them again."""

    def __init__(self, root: Path, source: str):
        self.source = source
        self.dir = root / source
        self.dir.mkdir(parents=True, exist_ok=True)
        self._missing_path = self.dir / "_missing.json"
        self._missing = set(json.loads(self._missing_path.read_text())) if self._missing_path.exists() else set()

    def path(self, ticker: str) -> Path:
        return self.dir / f"{ticker}.parquet"

    def has(self, ticker: str) -> bool:
        return self.path(ticker).exists()

    def is_missing(self, ticker: str) -> bool:
        return ticker in self._missing

    def load(self, ticker: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(ticker))

    def save(self, ticker: str, df: pd.DataFrame) -> None:
        df = df[COLUMNS].astype("float64").sort_index()
        df.index = pd.DatetimeIndex(df.index, name="date").tz_localize(None)
        df.to_parquet(self.path(ticker))

    def mark_missing(self, ticker: str) -> None:
        self._missing.add(ticker)
        self._missing_path.write_text(json.dumps(sorted(self._missing), indent=0))

    def todo(self, tickers: Iterable[str]) -> list[str]:
        return [t for t in tickers if not self.has(t) and not self.is_missing(t)]


# --------------------------------------------------------------------------- yfinance

_YF_RENAME = {"Open": "open", "High": "high", "Low": "low", "Close": "close",
              "Volume": "volume", "Adj Close": "adj_close"}


def download_yfinance(tickers: Iterable[str], start: str, end: str | None,
                      cache: PriceCache, chunk_size: int = 100) -> None:
    import yfinance as yf

    todo = cache.todo(tickers)
    for i in range(0, len(todo), chunk_size):
        chunk = todo[i:i + chunk_size]
        print(f"[yfinance] {i + len(chunk)}/{len(todo)}")
        raw = yf.download(chunk, start=start, end=end, auto_adjust=False, actions=False,
                          group_by="ticker", progress=False, threads=True)
        for ticker in chunk:
            if isinstance(raw.columns, pd.MultiIndex):
                if ticker not in raw.columns.get_level_values(0):
                    cache.mark_missing(ticker)
                    continue
                df = raw[ticker]
            else:
                df = raw
            df = df.rename(columns=_YF_RENAME).dropna(subset=["close", "adj_close"], how="all")
            if df.empty:
                cache.mark_missing(ticker)
            else:
                cache.save(ticker, df)


# --------------------------------------------------------------------------- Tiingo

TIINGO_SUPPORTED_URL = "https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip"
_US_EXCHANGES = {"NYSE", "NASDAQ", "NYSE ARCA", "NYSE MKT", "AMEX", "BATS", "PINK", "OTC", "NYSE NAT"}


def tiingo_supported_tickers(dest: Path, force: bool = False) -> pd.DataFrame:
    """Tiingo's public list of tickers with start/end dates (no API key needed)."""
    if force or not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(TIINGO_SUPPORTED_URL, timeout=120)
        response.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
            dest.write_bytes(zf.read(zf.namelist()[0]))
    df = pd.read_csv(dest, dtype=str)
    df["ticker"] = df["ticker"].str.upper()
    return df[(df["assetType"] == "Stock") & df["exchange"].isin(_US_EXCHANGES)]


class TiingoRateLimited(RuntimeError):
    pass


class TiingoClient:
    BASE = "https://api.tiingo.com/tiingo/daily"

    def __init__(self, api_key: str, min_interval: float = 75.0):
        # Free tier: 50 requests/hour -> one every 72 s; 75 s leaves a margin.
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Token {api_key}"
        self.min_interval = min_interval
        self._last = 0.0

    def fetch(self, ticker: str, start: str, end: str | None) -> pd.DataFrame | None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        params = {"startDate": start, "format": "json"}
        if end:
            params["endDate"] = end
        response = self.session.get(f"{self.BASE}/{ticker}/prices", params=params, timeout=60)
        self._last = time.monotonic()
        if response.status_code == 404:
            return None
        if response.status_code == 429 or "rate limit" in response.text[:300].lower():
            raise TiingoRateLimited(response.text[:300])
        response.raise_for_status()
        rows = response.json()
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df.pop("date")).dt.tz_localize(None).dt.normalize()
        return df.rename(columns={"adjClose": "adj_close"})


def download_tiingo(tickers: Iterable[str], start: str, end: str | None, cache: PriceCache,
                    client: TiingoClient, max_symbols: int | None = None) -> None:
    """Resumable: already cached or known-missing tickers are skipped, so the
    script can be stopped and restarted (e.g. when the monthly quota runs out)."""
    todo = cache.todo(tickers)
    if max_symbols is not None:
        todo = todo[:max_symbols]
    for i, ticker in enumerate(todo, 1):
        print(f"[tiingo] {i}/{len(todo)} {ticker}", flush=True)
        try:
            df = client.fetch(ticker, start, end)
        except TiingoRateLimited as exc:
            print(f"[tiingo] rate limit reached, stopping (rerun later to resume): {exc}")
            return
        if df is None or df.empty:
            cache.mark_missing(ticker)
        else:
            cache.save(ticker, df)
