"""Stage 5: portfolio construction and cost-aware backtest of every composite.

    python scripts/backtest.py

Outputs: reports/backtest.md, reports/figures/bt_*.png,
         data/processed/backtest_daily.parquet (daily gross/net returns per book)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_research import pipeline as pl
from alpha_research.backtest.engine import TRADING_DAYS, CostConfig, monthly_turnover, summary
from alpha_research.config import PROCESSED_DIR, REPORTS_DIR
from alpha_research.evaluation.factor_model import factor_regression

FIG = REPORTS_DIR / "figures"
PRIMARY = "bayes"     # declared before running the backtest: the project's thesis
BASELINE = "equal"
LABELS = {"equal": "Equal weight", "ic_weighted": "IC-weighted", "bayes": "Bayesian shrinkage",
          "ridge": "Ridge regression"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = pl.load(PROCESSED_DIR)
    comb = pl.combine(data)
    costs = CostConfig()

    ls, lo, bench = {}, {}, None
    for m in pl.METHODS:
        ls[m] = pl.long_short_book(data, comb.scores[m], comb.expo, costs)
        lo[m], bench = pl.long_only_book(data, comb.scores[m], costs)
        print(f"{m:<12} L/S net SR {summary(ls[m].net)['sharpe']:+.2f}")

    daily = pd.DataFrame({**{f"ls_{m}_gross": r.gross for m, r in ls.items()},
                          **{f"ls_{m}_net": r.net for m, r in ls.items()},
                          **{f"lo_{m}_net": r.net for m, r in lo.items()},
                          "lo_benchmark_net": bench.net, "spy": data.mkt_ret})
    daily.to_parquet(PROCESSED_DIR / "backtest_daily.parquet")

    from alpha_research.config import save_results
    out = {}
    for m in pl.METHODS:
        g, n = summary(ls[m].gross, data.mkt_ret), summary(ls[m].net, data.mkt_ret)
        ff = factor_regression(ls[m].net, data.factors)
        lo_s = summary(lo[m].net, bench.net)
        out[m] = {"gross_sharpe": g["sharpe"], "net_sharpe": n["sharpe"], "net_return": n["ann_return"],
                  "vol": n["ann_vol"], "max_dd": n["max_drawdown"], "beta": n["beta"],
                  "turnover": monthly_turnover(ls[m]), "cost_drag": ls[m].costs.mean() * TRADING_DAYS,
                  "ff_alpha": ff.at["alpha", "coef"], "ff_alpha_t": ff.at["alpha", "t_stat"],
                  "lo_return": lo_s["ann_return"], "lo_active": lo_s["active_return"], "lo_ir": lo_s["info_ratio"],
                  "lo_te": lo_s["tracking_error"]}
    out["benchmark_return"] = summary(bench.net)["ann_return"]
    out["start"], out["end"] = f"{ls[PRIMARY].net.index.min():%Y-%m-%d}", f"{ls[PRIMARY].net.index.max():%Y-%m-%d}"
    save_results("backtest", out)

    # ---------------------------------------------------------------- long-short table
    ls_rows = []
    for m, r in ls.items():
        g, n = summary(r.gross, data.mkt_ret), summary(r.net, data.mkt_ret)
        ff = factor_regression(r.net, data.factors)
        ls_rows.append(
            f"| {LABELS[m]} | {g['sharpe']:.2f} | {n['sharpe']:.2f} | {pct(n['ann_return'])} | {pct(n['ann_vol'])} | "
            f"{pct(n['max_drawdown'])} | {n['beta']:+.2f} | {pct(monthly_turnover(r))} | "
            f"{pct(r.costs.mean() * TRADING_DAYS)} | {pct(r.borrow.mean() * TRADING_DAYS)} | "
            f"{pct(ff.at['alpha', 'coef'])} ({ff.at['alpha', 't_stat']:.1f}) |")

    # ---------------------------------------------------------------- long-only table
    lo_rows = []
    b = summary(bench.net, data.mkt_ret)
    lo_rows.append(f"| Benchmark (equal-weighted universe) | {pct(b['ann_return'])} | {pct(b['ann_vol'])} | "
                   f"{b['sharpe']:.2f} | – | – | – | {b['beta']:.2f} | {pct(monthly_turnover(bench))} |")
    for m, r in lo.items():
        s = summary(r.net, bench.net)
        beta_mkt = summary(r.net, data.mkt_ret)["beta"]
        lo_rows.append(f"| {LABELS[m]} top quintile | {pct(s['ann_return'])} | {pct(s['ann_vol'])} | {s['sharpe']:.2f} | "
                       f"{pct(s['active_return'])} | {pct(s['tracking_error'])} | {s['info_ratio']:.2f} | "
                       f"{beta_mkt:.2f} | {pct(monthly_turnover(r))} |")

    # ---------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = {"equal": "#7f7f7f", "ic_weighted": "#e39b3a", "bayes": "#2a6fb0", "ridge": "#b04a4a"}
    for m, r in ls.items():
        ax.plot(r.net.index, (1 + r.net).cumprod(), color=cmap[m], lw=1.6 if m == PRIMARY else 1.0,
                label=f"{LABELS[m]} (net SR {summary(r.net)['sharpe']:.2f})")
    p = ls[PRIMARY]
    ax.plot(p.gross.index, (1 + p.gross).cumprod(), color=cmap[PRIMARY], ls="--", lw=1,
            label=f"{LABELS[PRIMARY]} before costs")
    ax.axhline(1, color="black", lw=0.6)
    ax.set_ylabel("growth of $1")
    ax.set_title("Dollar- and beta-neutral long-short books, net of 5 bps costs and 25 bps borrow")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "bt_longshort.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for m in (PRIMARY, BASELINE):
        w = (1 + ls[m].net).cumprod()
        axes[0].plot(w.index, 100 * (w / w.cummax() - 1), color=cmap[m], label=LABELS[m])
        roll = ls[m].net.rolling(TRADING_DAYS)
        axes[1].plot(w.index, roll.mean() / roll.std() * np.sqrt(TRADING_DAYS), color=cmap[m], label=LABELS[m])
    axes[0].set_ylabel("drawdown %")
    axes[1].set_ylabel("rolling 1y Sharpe")
    axes[1].axhline(0, color="black", lw=0.6)
    axes[0].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.3)
    axes[0].set_title("Risk profile of the long-short books (net)")
    fig.tight_layout()
    fig.savefig(FIG / "bt_risk.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for m, r in lo.items():
        active = r.net - bench.net.reindex(r.net.index).fillna(0)
        ax.plot(active.index, 100 * active.cumsum(), color=cmap[m], lw=1.6 if m == PRIMARY else 1.0, label=LABELS[m])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("cumulative active return %")
    ax.set_title("Long-only top quintile vs equal-weighted universe (net of costs)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "bt_longonly.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- factor exposures of the primary book
    ff = factor_regression(ls[PRIMARY].net, data.factors)
    ff_rows = [f"| {k} | {v.coef:+.3f} | {v.t_stat:+.2f} |" for k, v in ff.iterrows()]
    ff_rows[0] = f"| alpha (annualised) | {pct(ff.at['alpha', 'coef'])} | {ff.at['alpha', 't_stat']:+.2f} |"

    start, end = ls[PRIMARY].net.index.min(), ls[PRIMARY].net.index.max()
    lines = [
        "# Stage 5 — Portfolio construction and backtest",
        "",
        f"Period: **{start:%Y-%m-%d} to {end:%Y-%m-%d}**, monthly rebalance, trades at the close of the day after "
        "the signal. Costs: 5 bps per dollar traded (one way) and 25 bps/year borrow on shorts. "
        f"Primary method declared in advance: **{LABELS[PRIMARY]}**; baseline: {LABELS[BASELINE]}.",
        "",
        "## Long-short book (market neutral)",
        "",
        "Rank-linear weights, projected to be dollar-neutral and beta-neutral (252-day beta), 100% long / 100% "
        "short, max 2% per stock.",
        "",
        "| Method | Sharpe gross | Sharpe net | Return net | Vol | Max DD | Realised beta | Turnover/month | "
        "Cost drag/yr | Borrow/yr | FF6 alpha net (t) |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
        *ls_rows,
        "",
        "![long-short](figures/bt_longshort.png)",
        "",
        "![risk](figures/bt_risk.png)",
        "",
        f"### Factor exposures of the {LABELS[PRIMARY]} book (net returns, Newey-West t)",
        "",
        "| Factor | Loading | t |",
        "|---|---|---|",
        *ff_rows,
        f"| R² | {ff.attrs['r2']:.2f} | |",
        "",
        "## Long-only book",
        "",
        "Equal weight in the top quintile of the composite (~100 stocks), against the equal-weighted universe "
        "built with the same engine and costs.",
        "",
        "| Portfolio | Return | Vol | Sharpe | Active return | Tracking error | Information ratio | Beta (SPY) | "
        "Turnover/month |",
        "|---|---|---|---|---|---|---|---|---|",
        *lo_rows,
        "",
        "![long-only](figures/bt_longonly.png)",
        "",
        "Sharpe ratios are computed on total returns (not excess of the risk-free rate), which slightly flatters "
        "the long-only books relative to the self-financing long-short books.",
    ]
    (REPORTS_DIR / "backtest.md").write_text("\n".join(lines) + "\n")
    print("Done -> reports/backtest.md")


if __name__ == "__main__":
    main()
