"""Compute all signals and forward-return targets from the clean dataset.

    python scripts/build_signals.py

Inputs : data/processed/prices.parquet, benchmark.parquet (from build_dataset.py)
Outputs: data/processed/signals_raw.parquet  raw signals, universe days only
         data/processed/signals.parquet      oriented, winsorized, z-scored signals
         data/processed/targets.parquet      forward returns (1, 5, 21 days, trade at t+1 close)
         reports/signals.md                  coverage and signal correlation report
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_research.config import PROCESSED_DIR, REPORTS_DIR
from alpha_research.signals import preprocess, targets
from alpha_research.signals.library import SIGNALS, MarketData


def load_market_data() -> tuple[MarketData, pd.DataFrame]:
    panel = pd.read_parquet(PROCESSED_DIR / "prices.parquet",
                            columns=["adj_close", "ret", "dollar_volume", "in_sp500"])
    wide = {c: panel[c].unstack("ticker") for c in panel.columns}
    bench = pd.read_parquet(PROCESSED_DIR / "benchmark.parquet")
    mkt_ret = (bench["adj_close"] / bench["adj_close"].shift(1) - 1).reindex(wide["ret"].index)
    data = MarketData(ret=wide["ret"].astype(float), adj_close=wide["adj_close"].astype(float),
                      dollar_volume=wide["dollar_volume"].astype(float), mkt_ret=mkt_ret)
    return data, wide["in_sp500"].fillna(False).astype(bool)


def stack(frames: dict[str, pd.DataFrame], mask: pd.DataFrame) -> pd.DataFrame:
    """Wide frames -> one long (date, ticker) frame, universe days only."""
    keep = mask.stack()
    keep = keep[keep].index
    return pd.DataFrame({name: df.stack(future_stack=True).reindex(keep) for name, df in frames.items()})


def mean_cross_corr(z: dict[str, pd.DataFrame], dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Average over `dates` of the cross-sectional correlation between signals."""
    mats = []
    for date in dates:
        snap = pd.DataFrame({n: df.loc[date] for n, df in z.items()}).dropna()
        if len(snap) > 50:
            mats.append(snap.corr().to_numpy())
    return pd.DataFrame(np.nanmean(mats, axis=0), index=list(z), columns=list(z))


def write_report(z: dict[str, pd.DataFrame], mask: pd.DataFrame, corr: pd.DataFrame) -> None:
    fig_dir = REPORTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=60, ha="right")
    ax.set_yticks(range(len(corr)), corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.iat[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Average cross-sectional correlation of signals (month-ends)")
    fig.tight_layout()
    fig.savefig(fig_dir / "signal_correlation.png", dpi=150)
    plt.close(fig)

    universe_size = mask.sum(axis=1)
    live = universe_size > 0
    rows = []
    for s in SIGNALS:
        available = z[s.name].notna().sum(axis=1)[live]
        first = available[available > 0].index.min()
        rows.append(f"| `{s.name}` | {s.description} | {'+' if s.sign > 0 else '−'} | {s.reference} | "
                    f"{(available / universe_size[live]).mean():.1%} | {first:%Y-%m-%d} |")

    # The pair of distinct signals with the highest |correlation| is worth calling out.
    off = corr.where(~np.eye(len(corr), dtype=bool)).abs().stack()
    top = off.sort_values(ascending=False).iloc[::2].head(5)

    lines = [
        "# Signal report",
        "",
        "Signals are computed from data up to the close of day *t*; trades are assumed at the close of "
        "*t+1*. Each signal is oriented so that **higher = higher expected return**, then winsorized "
        "(1%/99%) and z-scored across the S&P 500 members of that day.",
        "",
        "## Signals",
        "",
        "| Signal | Definition | Sign | Reference | Avg coverage of universe | First date |",
        "|---|---|---|---|---|---|",
        *rows,
        "",
        "## Correlation between signals",
        "",
        "![correlation](figures/signal_correlation.png)",
        "",
        "Most correlated pairs (|ρ|): " + ", ".join(f"`{a}`/`{b}` {v:.2f}" for (a, b), v in top.items()) + ".",
        "",
        "Highly correlated signals carry largely the same information; the combination step "
        "(Stage 4) has to account for this rather than count the same bet twice.",
    ]
    (REPORTS_DIR / "signals.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    t0 = time.time()
    data, members = load_market_data()
    mask = preprocess.universe_mask(members, data.adj_close)

    raw, z = {}, {}
    for s in SIGNALS:
        raw[s.name] = s.func(data)
        z[s.name] = preprocess.process(raw[s.name], mask, s.sign)
        print(f"{s.name:<13} done ({time.time() - t0:5.1f}s)")

    fwd = {f"fwd_ret_{h}": targets.forward_returns(data.adj_close, h) for h in targets.HORIZONS}

    stack(raw, mask).to_parquet(PROCESSED_DIR / "signals_raw.parquet")
    stack(z, mask).to_parquet(PROCESSED_DIR / "signals.parquet")
    stack(fwd, mask).to_parquet(PROCESSED_DIR / "targets.parquet")

    month_ends = pd.DatetimeIndex(mask.index.to_series().groupby(mask.index.to_period("M")).max())
    month_ends = month_ends[mask.loc[month_ends].sum(axis=1).to_numpy() > 0]
    corr = mean_cross_corr(z, month_ends)
    write_report(z, mask, corr)
    print(f"Done in {time.time() - t0:.0f}s -> reports/signals.md")


if __name__ == "__main__":
    main()
