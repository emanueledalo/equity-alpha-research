"""Project paths and dataset settings, kept in one place."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class DataConfig:
    # First date a stock can be in the research universe.
    universe_start: str = "2010-01-01"
    # Prices start earlier so look-back signals (e.g. 12-month momentum)
    # are defined from the first universe date.
    history_start: str = "2008-01-01"
    # None = up to the latest available date.
    end: str | None = None
    # Benchmark used for the trading calendar and for beta later on.
    benchmark: str = "SPY"

    # Cleaning thresholds (applied to daily log returns of adjusted close).
    spike_threshold: float = 0.25   # |log return| that counts as a large move
    spike_reversal: float = 0.25    # a spike is a bad tick if the next day undoes >=75% of it
    extreme_return: float = 1.0     # |simple return| above this is flagged for review
    stale_run: int = 5              # identical closes with zero volume for this many days -> stale

    # Identity check for reused tickers: a source's series is only accepted for an
    # index member if it actually trades like an S&P 500 stock while in the index.
    min_member_dollar_volume: float = 1e7   # median $ volume on member days
    min_source_coverage: float = 0.9        # below this share of member days, try the next source too


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Minimal .env reader: KEY=VALUE lines, existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def save_results(section: str, values: dict) -> None:
    """Merge headline numbers of one stage into reports/results.json (used by the docs)."""
    import json

    path = REPORTS_DIR / "results.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data[section] = values
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=float))
