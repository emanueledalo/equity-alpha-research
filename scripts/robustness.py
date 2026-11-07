"""Stage 6: robustness of the conclusions to the choices made along the way.

    python scripts/robustness.py

Every test changes ONE assumption of the Stage 5 setup and reports the long-short
book again. None of these variations was used to pick the main configuration:
they show how fragile (or not) the result is.

Outputs: reports/robustness.md, reports/figures/rob_*.png
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from alpha_research import pipeline as pl
from alpha_research.backtest.engine import CostConfig, monthly_turnover, summary
from alpha_research.config import DATA_DIR, PROCESSED_DIR, REPORTS_DIR
from alpha_research.signals.library import SIGNALS

FIG = REPORTS_DIR / "figures"
SNAPSHOT = DATA_DIR / "snapshots" / "yfinance_only"
LABELS = {"equal": "Equal weight", "ic_weighted": "IC-weighted", "bayes": "Bayesian shrinkage",
          "ridge": "Ridge regression"}
MAIN = ["equal", "bayes"]
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:5.0f}s] {msg}", flush=True)


def stats(r) -> dict[str, float]:
    g, n = summary(r.gross), summary(r.net)
    return {"gross": g["sharpe"], "net": n["sharpe"], "ret": n["ann_return"], "vol": n["ann_vol"],
            "dd": n["max_drawdown"], "turnover": monthly_turnover(r)}


def row(label: str, s: dict) -> str:
    return (f"| {label} | {s['gross']:.2f} | {s['net']:.2f} | {100 * s['ret']:.1f}% | {100 * s['vol']:.1f}% | "
            f"{100 * s['dd']:.1f}% | {100 * s['turnover']:.0f}% |")


def net_sharpe_at(r, cost_bps: float) -> float:
    """Net Sharpe if every dollar traded cost `cost_bps`. Costs are linear in the traded
    amount, so the net series can be rebuilt from the gross one without a new backtest."""
    traded_cost_per_bps = r.costs / CostConfig().cost_bps
    return summary(r.gross - traded_cost_per_bps * cost_bps - r.borrow)["sharpe"]


def breakeven_cost(r, max_bps: int = 30):
    """Smallest whole-bps cost at which the net Sharpe is no longer positive."""
    for c in range(max_bps + 1):
        if net_sharpe_at(r, c) <= 0:
            return c
    return f">{max_bps}"


HEADER = ["| Variant | Sharpe gross | Sharpe net | Return net | Vol | Max DD | Turnover/month |",
          "|---|---|---|---|---|---|---|"]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = pl.load(PROCESSED_DIR)
    comb = pl.combine(data)
    base = {m: pl.long_short_book(data, comb.scores[m], comb.expo) for m in pl.METHODS}
    log("baseline done")
    sections: list[str] = []
    results: dict = {}

    # 1. Transaction costs ---------------------------------------------------------------
    levels = [0, 2, 5, 10, 20, 30]
    table = {}
    for m in pl.METHODS:
        table[m] = [net_sharpe_at(base[m], c) for c in levels]
    cost_df = pd.DataFrame(table, index=levels)
    fig, ax = plt.subplots(figsize=(7, 4))
    for m in pl.METHODS:
        ax.plot(levels, cost_df[m], marker="o", label=LABELS[m])
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(5, color="grey", ls=":", lw=1)
    ax.set_xlabel("transaction cost, bps per dollar traded (one way)")
    ax.set_ylabel("net Sharpe ratio")
    ax.set_title("Sensitivity to transaction costs (long-short books)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "rob_costs.png", dpi=150)
    plt.close(fig)
    breakeven = {m: breakeven_cost(base[m]) for m in pl.METHODS}
    sections += ["## 1. Transaction costs", "",
                 "Net Sharpe ratio for different cost assumptions (5 bps is the base case).", "",
                 "| Method | " + " | ".join(f"{c} bps" for c in levels) + " | Break-even cost |",
                 "|---|" + "---|" * (len(levels) + 1),
                 *[f"| {LABELS[m]} | " + " | ".join(f"{v:.2f}" for v in cost_df[m]) + f" | {breakeven[m]} bps |"
                   for m in pl.METHODS],
                 "", "![costs](figures/rob_costs.png)", ""]
    log("costs done")

    # 2. Turnover control: partial rebalancing ------------------------------------------------
    rows = []
    for m in MAIN:
        for frac in (1.0, 0.5, 0.25):
            r = pl.long_short_book(data, comb.scores[m], comb.expo, CostConfig(trade_fraction=frac))
            s = stats(r)
            results.setdefault("partial", {})[f"{m}_{frac}"] = s
            rows.append(row(f"{LABELS[m]}, trade {frac:.0%} of the way to target", s))
    sections += ["## 2. Turnover control (partial rebalancing)", "",
                 "Each month the book moves only part of the way from its drifted weights to the new target. "
                 "Less trading lowers costs but also dilutes the signal (slower to react).", "",
                 *HEADER, *rows, ""]
    log("turnover done")

    # 3. Sub-periods -----------------------------------------------------------------------------
    rows = []
    split = pd.Timestamp("2020-01-01")
    for m in pl.METHODS:
        r = base[m]
        for label, sl in (("first half", slice(None, split - pd.Timedelta(days=1))), ("second half", slice(split, None))):
            g, n = summary(r.gross.loc[sl]), summary(r.net.loc[sl])
            results.setdefault("subperiods", {})[f"{m}_{label}"] = {"gross": g["sharpe"], "net": n["sharpe"]}
            rows.append(f"| {LABELS[m]} | {label} | {g['sharpe']:.2f} | {n['sharpe']:.2f} | "
                        f"{100 * n['ann_return']:.1f}% |")
    start = base["equal"].net.index.min()
    sections += ["## 3. Sub-periods", "",
                 f"First half: {start:%Y-%m} to 2019-12; second half: 2020-01 to the end (includes COVID and the "
                 "2022 rate shock).", "",
                 "| Method | Period | Sharpe gross | Sharpe net | Return net |", "|---|---|---|---|---|", *rows, ""]
    log("subperiods done")

    # 4. Prior sensitivity (Bayesian method) -----------------------------------------------------
    rows = []
    for prior in ((0.0, 0.01), (0.0, 0.02), (0.02, 0.02), (0.05, 0.02), (0.02, 0.05)):
        c = pl.combine(data, methods=["bayes"], prior=prior)
        s = stats(pl.long_short_book(data, c.scores["bayes"], c.expo))
        results.setdefault("prior", {})[f"{prior[0]}_{prior[1]}"] = s
        rows.append(row(f"prior IC {prior[0]:.2f} ± {prior[1]:.2f}" + (" (base)" if prior == (0.02, 0.02) else ""), s))
    sections += ["## 4. Prior sensitivity (Bayesian shrinkage)", "",
                 "The prior only sets the *level* of the common premium; with non-negative weights it matters "
                 "mostly when the data push the common level towards zero (then the book stops trading).", "",
                 *HEADER, *rows, ""]
    log("prior done")

    # 5. Leave-one-signal-out --------------------------------------------------------------------
    rows = []
    names = [s.name for s in SIGNALS]
    full = stats(base["bayes"])
    for name in names:
        sub = pl.ResearchData(z={k: v for k, v in data.z.items() if k != name}, beta=data.beta, ret=data.ret,
                              mkt_ret=data.mkt_ret, factors=data.factors)
        c = pl.combine(sub, methods=["equal", "bayes"])
        s_eq = stats(pl.long_short_book(data, c.scores["equal"], c.expo))
        s_b = stats(pl.long_short_book(data, c.scores["bayes"], c.expo))
        results.setdefault("loo", {})[name] = {"equal": s_eq["gross"], "bayes": s_b["gross"]}
        rows.append(f"| without `{name}` | {s_eq['gross']:.2f} | {s_b['gross']:.2f} | "
                    f"{s_b['gross'] - full['gross']:+.2f} |")
        log(f"  without {name}")
    sections += ["## 5. Leave one signal out", "",
                 "Gross Sharpe when a single signal is removed before combining. A large drop means the result "
                 "depends on that one signal.", "",
                 f"All signals: equal weight {stats(base['equal'])['gross']:.2f}, Bayesian {full['gross']:.2f}.", "",
                 "| Variant | Equal weight (gross SR) | Bayesian (gross SR) | Bayesian change |", "|---|---|---|---|",
                 *rows, ""]

    # 6. Rebalance frequency ---------------------------------------------------------------------
    rows = [row(f"{LABELS[m]}, monthly (base)", stats(base[m])) for m in MAIN]
    wk = pl.combine(data, freq="W", methods=MAIN, min_periods=156)
    for m in MAIN:
        s = stats(pl.long_short_book(data, wk.scores[m], wk.expo))
        results.setdefault("weekly", {})[m] = s
        rows.append(row(f"{LABELS[m]}, weekly", s))
    sections += ["## 6. Rebalance frequency", "",
                 "Weekly rebalancing reacts faster to the short-horizon signals (reversal) but trades ~4x as often. "
                 "Weekly weights use a 156-week burn-in (same 3 years).", "",
                 *HEADER, *rows, ""]
    log("frequency done")

    # 7. Survivorship bias -----------------------------------------------------------------------
    if (SNAPSHOT / "signals.parquet").exists():
        common = [c for c in pd.read_parquet(SNAPSHOT / "signals.parquet").columns if c in data.z]
        rows = []
        for label, path in (("with delisted stocks (main)", PROCESSED_DIR), ("survivors only (yfinance)", SNAPSHOT)):
            d = pl.load(path, signals=common)
            c = pl.combine(d)
            for m in pl.METHODS:
                s = stats(pl.long_short_book(d, c.scores[m], c.expo))
                rows.append(row(f"{LABELS[m]}, {label}", s))
                results.setdefault("survivorship", {})[f"{m}_{'main' if path == PROCESSED_DIR else 'survivors'}"] = s
        sections += ["## 7. Survivorship bias", "",
                     f"Same pipeline and the same {len(common)} signals on two datasets: the main one (delisted "
                     "stocks filled from Tiingo) and a snapshot built from yfinance only, which silently drops "
                     "every stock that no longer trades.", "",
                     *HEADER, *rows, ""]
        log("survivorship done")

    lines = ["# Stage 6 — Robustness", "",
             "Each section changes one assumption of the Stage 5 long-short setup (monthly rebalance, beta-neutral, "
             "5 bps costs, 25 bps borrow, full rebalancing). None of these variants was used to choose the main "
             "configuration.", "", *sections]
    (REPORTS_DIR / "robustness.md").write_text("\n".join(lines) + "\n")
    results["costs"] = {m: dict(zip(map(str, levels), cost_df[m])) for m in pl.METHODS}
    results["breakeven_bps"] = breakeven
    from alpha_research.config import save_results
    save_results("robustness", results)
    log("Done -> reports/robustness.md")


if __name__ == "__main__":
    main()
