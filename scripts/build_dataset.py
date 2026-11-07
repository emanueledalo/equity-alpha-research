"""Build the research dataset end to end.

    python scripts/build_dataset.py                 # yfinance + Tiingo (if TIINGO_API_KEY is set)
    python scripts/build_dataset.py --no-tiingo     # yfinance only (survivorship-biased, for a quick start)

Every download is cached under data/raw/, so reruns only fetch what is missing.
Outputs:
    data/processed/prices.parquet      long panel (date, ticker) of clean prices and returns
    data/processed/membership.parquet  boolean (date x ticker) S&P 500 membership
    data/processed/factors.parquet     Fama-French 5 factors + momentum + risk-free rate
    reports/data_quality.md            coverage and cleaning report
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from alpha_research.config import PROCESSED_DIR, RAW_DIR, REPORTS_DIR, DataConfig, load_dotenv, save_results
from alpha_research.data import clean, factors, prices, quality, universe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-tiingo", action="store_true", help="skip the Tiingo fallback for delisted tickers")
    parser.add_argument("--tiingo-max-symbols", type=int, default=None,
                        help="cap Tiingo symbols this run (free tier: 500 distinct symbols/month)")
    parser.add_argument("--tiingo-interval", type=float, default=75.0,
                        help="seconds between Tiingo requests (free tier: 50/hour)")
    parser.add_argument("--refresh-membership", action="store_true")
    args = parser.parse_args()

    cfg = DataConfig()
    load_dotenv()

    # 1. Point-in-time membership ------------------------------------------------
    snap_path = universe.download_membership(RAW_DIR / "sp500_membership.csv", force=args.refresh_membership)
    snapshots = universe.load_snapshots(snap_path)
    needed = universe.tickers_between(snapshots, cfg.universe_start, cfg.end)
    print(f"{len(needed)} tickers were S&P 500 members since {cfg.universe_start}")

    # 2. Prices: yfinance first --------------------------------------------------------
    yf_cache = prices.PriceCache(RAW_DIR / "prices", "yfinance")
    prices.download_yfinance([cfg.benchmark, *needed], cfg.history_start, cfg.end, yf_cache)

    # Trading calendar from the benchmark, and point-in-time membership on it.
    calendar = yf_cache.load(cfg.benchmark).loc[cfg.history_start:cfg.end].index
    members = universe.membership_matrix(snapshots, calendar)
    members = members.reindex(columns=sorted(set(members.columns) | set(needed)), fill_value=False)
    members.loc[members.index < pd.Timestamp(cfg.universe_start)] = False  # universe starts at universe_start
    window = calendar[calendar >= pd.Timestamp(cfg.universe_start)]

    # Tickers yfinance cannot serve correctly: missing, wrong company, or poor coverage.
    _, yf_log = clean.choose_sources([yf_cache], needed, members, cfg)
    good_yf = set(yf_log.loc[yf_log["accepted"] & (yf_log["coverage"] >= cfg.min_source_coverage), "ticker"])
    no_yf = [t for t in needed if t not in good_yf]

    # 3. Tiingo for the rest --------------------------------------------------------------
    tg_cache = prices.PriceCache(RAW_DIR / "prices", "tiingo")
    api_key = os.environ.get("TIINGO_API_KEY", "").strip()
    if args.no_tiingo:
        print(f"Skipping Tiingo: {len(no_yf)} tickers will have no prices.")
    elif not api_key:
        print(f"TIINGO_API_KEY not set: {len(no_yf)} delisted/renamed tickers will have no prices. "
              "See .env.example.")
    else:
        supported = set(prices.tiingo_supported_tickers(RAW_DIR / "tiingo_supported_tickers.csv")["ticker"])
        candidates = [t for t in no_yf if t in supported]
        for t in no_yf:
            if t not in supported:
                tg_cache.mark_missing(t)
        print(f"Tiingo: {len(candidates)} of {len(no_yf)} remaining tickers are listed by Tiingo")
        client = prices.TiingoClient(api_key, min_interval=args.tiingo_interval)
        prices.download_tiingo(candidates, cfg.history_start, cfg.end, tg_cache, client,
                               max_symbols=args.tiingo_max_symbols)

    # 4. Factors ---------------------------------------------------------------------
    ff = factors.download_factors(PROCESSED_DIR / "factors.parquet", force=True)
    print(f"Factors: {ff.index.min():%Y-%m-%d} to {ff.index.max():%Y-%m-%d}")

    # 5. Pick a verified source per ticker, then clean -----------------------------------
    chosen, source_log = clean.choose_sources([yf_cache, tg_cache], needed, members, cfg)
    wide_raw, source = clean.assemble_raw(chosen)
    wide_raw = clean.trim_after_membership({f: df.reindex(calendar) for f, df in wide_raw.items()}, members)
    wide, issues = clean.clean_prices(wide_raw, calendar, cfg)

    panel = clean.to_long(wide, members, source)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PROCESSED_DIR / "prices.parquet")
    members.loc[window].to_parquet(PROCESSED_DIR / "membership.parquet")
    yf_cache.load(cfg.benchmark).loc[cfg.history_start:cfg.end].to_parquet(PROCESSED_DIR / "benchmark.parquet")

    # 6. Report ----------------------------------------------------------------------
    cov = quality.daily_coverage(wide["adj_close"].loc[window], members.loc[window])
    per_ticker = clean.coverage_checks(wide["adj_close"].loc[window], members.loc[window])
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    quality.write_report(REPORTS_DIR / "data_quality.md", cfg, cov, per_ticker, issues, source,
                         len(needed), source_log)

    yearly = cov.groupby(cov.index.year)["coverage"].mean()
    save_results("data", {
        "n_tickers": len(needed), "by_source": source.value_counts().to_dict(),
        "n_unpriced": int((per_ticker["covered_days"] == 0).sum()),
        "n_rejected": int((~source_log["accepted"]).sum()),
        "coverage_avg": cov["coverage"].mean(), "coverage_min": cov["coverage"].min(),
        "coverage_first_year": yearly.iloc[0], "coverage_last_year": yearly.iloc[-1],
        "start": f"{window.min():%Y-%m-%d}", "end": f"{window.max():%Y-%m-%d}",
        "issues": issues["issue"].value_counts().to_dict(), "rows": len(panel)})
    print(f"Panel: {len(panel):,} rows, {panel.index.get_level_values('ticker').nunique()} tickers")
    print(f"Average member coverage: {cov['coverage'].mean():.1%}  (report: reports/data_quality.md)")


if __name__ == "__main__":
    main()
