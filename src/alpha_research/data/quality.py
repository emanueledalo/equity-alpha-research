"""Data-quality report (markdown + figure) written to reports/."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from alpha_research.config import DataConfig


def daily_coverage(adj_close: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    members = members.reindex(index=adj_close.index, columns=members.columns, fill_value=False)
    has_price = adj_close.reindex(columns=members.columns).notna()
    n_members = members.sum(axis=1)
    n_priced = (members & has_price).sum(axis=1)
    return pd.DataFrame({"members": n_members, "priced": n_priced, "coverage": n_priced / n_members})


def write_report(path: Path, cfg: DataConfig, coverage: pd.DataFrame, per_ticker: pd.DataFrame,
                 issues: pd.DataFrame, source: pd.Series, n_needed: int,
                 source_log: pd.DataFrame | None = None) -> None:
    fig_dir = path.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(coverage.index, 100 * coverage["coverage"], lw=1.2, color="#2a6fb0")
    ax.set_ylabel("% of index members with a price")
    ax.set_ylim(min(80, 100 * coverage["coverage"].min() - 2), 100.5)
    ax.set_title("Point-in-time S&P 500: price coverage of index members")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "coverage.png", dpi=150)
    plt.close(fig)

    yearly = coverage.groupby(coverage.index.year).agg(
        avg_members=("members", "mean"), avg_priced=("priced", "mean"), min_coverage=("coverage", "min"))
    yearly["avg_coverage"] = yearly["avg_priced"] / yearly["avg_members"]

    missing = per_ticker[per_ticker["covered_days"] == 0]
    partial = per_ticker[(per_ticker["covered_days"] > 0) & (per_ticker["coverage"] < 0.95)]
    by_source = source.value_counts()
    issue_counts = issues["issue"].value_counts() if len(issues) else pd.Series(dtype=int)
    extreme = issues[issues["issue"] == "extreme_return_flag"].sort_values("date") if len(issues) else issues

    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    lines = [
        "# Data quality report",
        "",
        f"Universe window: **{coverage.index.min():%Y-%m-%d} to {coverage.index.max():%Y-%m-%d}** · "
        f"price history from {cfg.history_start} · benchmark/calendar: {cfg.benchmark}",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Tickers ever in the index during the window | {n_needed} |",
        *[f"| Tickers priced from {src} | {n} |" for src, n in by_source.items()],
        f"| Tickers with no price at all | {len(missing)} |",
        f"| Tickers with partial coverage (<95% of member days) | {len(partial)} |",
        f"| Average daily coverage of index members | {pct(coverage['coverage'].mean())} |",
        f"| Worst day coverage | {pct(coverage['coverage'].min())} |",
        "",
        "![coverage](figures/coverage.png)",
        "",
        "## Coverage by year",
        "",
        "| Year | Avg members | Avg priced | Avg coverage | Min coverage |",
        "|---|---|---|---|---|",
        *[f"| {y} | {r.avg_members:.0f} | {r.avg_priced:.0f} | {pct(r.avg_coverage)} | {pct(r.min_coverage)} |"
          for y, r in yearly.iterrows()],
        "",
        "## Cleaning actions",
        "",
        "| Issue | Cells |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in issue_counts.items()],
        "",
        "`invalid_price`, `stale_price` and `spike_reverted` cells were removed (set to NaN). "
        "`extreme_return_flag` cells were kept and are listed below for review.",
        "",
    ]
    if len(extreme):
        lines += ["### Extreme daily returns (|r| > {:.0%})".format(cfg.extreme_return), "",
                  "| Date | Ticker |", "|---|---|",
                  *[f"| {r.date:%Y-%m-%d} | {r.ticker} |" for r in extreme.itertuples()], ""]
    rejected = source_log[~source_log["accepted"]] if source_log is not None else pd.DataFrame()
    if len(rejected):
        lines += ["## Series rejected by the identity check", "",
                  "Price series a source returned for the ticker, but which do not trade like the S&P 500 "
                  "member during its membership (usually a reused ticker now owned by another company). "
                  "They are discarded, never mixed in.", "",
                  "| Ticker | Source | Reason |", "|---|---|---|",
                  *[f"| {r.ticker} | {r.source} | {r.reason} |" for r in rejected.itertuples()], ""]
    lines += ["## Index members with no price data", "",
              ", ".join(missing.index) if len(missing) else "None.", "",
              "## Partially covered members (lowest coverage first)", "",
              "| Ticker | Member days | Covered | First in index | First price |", "|---|---|---|---|---|",
              *[f"| {t} | {r.member_days} | {pct(r.coverage)} | {r.first_member:%Y-%m-%d} | "
                f"{'' if pd.isna(r.first_price) else f'{r.first_price:%Y-%m-%d}'} |"
                for t, r in partial.head(40).iterrows()], ""]
    path.write_text("\n".join(lines))
