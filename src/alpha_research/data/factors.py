"""Daily Fama-French 5 factors + momentum and the risk-free rate.

Source: Kenneth French Data Library (Dartmouth). Used later to risk-adjust
strategy returns (alpha vs. known factors) and for the risk-free rate in Sharpe ratios.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FF5_URL = _BASE + "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
MOM_URL = _BASE + "F-F_Momentum_Factor_daily_CSV.zip"


def _read_french_zip(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        text = zf.read(zf.namelist()[0]).decode("latin-1")

    header, rows = None, []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if header is None and len(parts) > 1 and parts[0] == "" and parts[1]:
            header = parts[1:]                      # e.g. ",Mkt-RF,SMB,..."
        elif len(parts[0]) == 8 and parts[0].isdigit():  # YYYYMMDD data rows
            rows.append(parts)
    df = pd.DataFrame([r[1:] for r in rows], columns=header,
                      index=pd.to_datetime([r[0] for r in rows], format="%Y%m%d"))
    df.index.name = "date"
    return df.astype(float) / 100.0  # the library publishes percentages


def download_factors(dest: Path, force: bool = False) -> pd.DataFrame:
    if dest.exists() and not force:
        return pd.read_parquet(dest)
    ff5 = _read_french_zip(FF5_URL)
    mom = _read_french_zip(MOM_URL)
    mom.columns = ["MOM"]
    factors = ff5.join(mom, how="left")
    dest.parent.mkdir(parents=True, exist_ok=True)
    factors.to_parquet(dest)
    return factors
