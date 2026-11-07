"""Stage 7: principled improvements, chosen on a development period, judged on a holdout.

    python scripts/improvements.py

Protocol (fixed before running):
  * development period: start of trading to 2022-12-31; holdout: 2023-01-01 to the end.
  * every variant is motivated a priori (see the report) and parameters are not tuned;
  * selection rule: the variant with the highest *development* net Sharpe for each
    method is selected; its holdout result is then reported, whatever it is.
Caveat stated honestly in the report: aggregate full-sample results of the baseline
were seen before this protocol was set up, so the holdout is not perfectly virgin.

Outputs: reports/improvements.md, reports/figures/imp_*.png, results.json["improvements"]
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_research import pipeline as pl
from alpha_research.backtest.construction import ConstructionConfig
from alpha_research.backtest.engine import CostConfig, monthly_turnover, summary
from alpha_research.config import DATA_DIR, PROCESSED_DIR, REPORTS_DIR, save_results
from alpha_research.evaluation.factor_model import factor_regression

FIG = REPORTS_DIR / "figures"
SNAPSHOT = DATA_DIR / "snapshots" / "yfinance_only"
N_PLACEBO = 50
DEV_END = pd.Timestamp("2022-12-31")
METHODS = ["equal", "bayes"]
LABELS = {"equal": "Equal weight", "bayes": "Bayesian shrinkage"}

# Cumulative ladder: each step adds one idea to the previous one.
VARIANTS = [
    ("V0 baseline (beta-neutral, rank weights, full rebalance)", dict(n_pca=0, risk=False, frac=1.0)),
    ("V1 + statistical risk model (5 PCA factors)", dict(n_pca=5, risk=False, frac=1.0)),
    ("V2 + risk-scaled weights (z / idio vol)", dict(n_pca=5, risk=True, frac=1.0)),
    ("V3 + trade 50% towards target", dict(n_pca=5, risk=True, frac=0.5)),
    ("V4 + trade 25% towards target", dict(n_pca=5, risk=True, frac=0.25)),
]


def split_stats(r) -> dict:
    out = {}
    for name, sl in (("dev", slice(None, DEV_END)), ("holdout", slice(DEV_END + pd.Timedelta(days=1), None)),
                     ("full", slice(None, None))):
        g, n = summary(r.gross.loc[sl]), summary(r.net.loc[sl])
        out[name] = {"gross": g["sharpe"], "net": n["sharpe"], "ret": n["ann_return"], "vol": n["ann_vol"],
                     "dd": n["max_drawdown"]}
    out["turnover"] = monthly_turnover(r)
    return out


def main() -> None:
    t0 = time.time()
    FIG.mkdir(parents=True, exist_ok=True)
    data = pl.load(PROCESSED_DIR)
    combos = {n: pl.combine(data, methods=METHODS, n_pca=n) for n in sorted({v["n_pca"] for _, v in VARIANTS})}
    print(f"combinations done ({time.time() - t0:.0f}s)")

    results, books = {}, {}
    for label, v in VARIANTS:
        comb = combos[v["n_pca"]]
        for m in METHODS:
            r = pl.long_short_book(data, comb.scores[m], comb.expo, CostConfig(trade_fraction=v["frac"]),
                                   ConstructionConfig(risk_scaled=v["risk"]))
            results[(label, m)] = split_stats(r)
            books[(label, m)] = r
            s = results[(label, m)]
            print(f"{label[:3]} {m:<6} dev net {s['dev']['net']:+.2f} holdout net {s['holdout']['net']:+.2f} "
                  f"turnover {s['turnover']:.0%} ({time.time() - t0:.0f}s)")

    chosen = {m: max(VARIANTS, key=lambda v: results[(v[0], m)]["dev"]["net"])[0] for m in METHODS}

    # ---------------------------------------------------------------- figure: cumulative net, baseline vs chosen
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = {"equal": "#7f7f7f", "bayes": "#2a6fb0"}
    for m in METHODS:
        for label, ls in ((VARIANTS[0][0], ":"), (chosen[m], "-")):
            net = books[(label, m)].net
            ax.plot(net.index, (1 + net).cumprod(), color=colors[m], ls=ls, lw=1.5 if ls == "-" else 1.1,
                    label=f"{LABELS[m]}: {label.split(' ')[0]} (net SR {summary(net)['sharpe']:.2f})")
    ax.axvline(DEV_END, color="black", lw=0.8, ls="--")
    ax.text(DEV_END, ax.get_ylim()[1], "  holdout →", va="top", fontsize=8)
    ax.axhline(1, color="black", lw=0.5)
    ax.set_ylabel("growth of $1 (net)")
    ax.set_title("Baseline (dotted) vs variant selected on 2013–2022 (solid)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "imp_ladder.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------------- factor attribution of the chosen Bayesian book
    ff = factor_regression(books[(chosen["bayes"], "bayes")].net, data.factors)
    sel = dict(VARIANTS)[chosen["bayes"]]
    sel_comb = combos[sel["n_pca"]]
    sel_cons = ConstructionConfig(risk_scaled=sel["risk"])

    # ---------------------------------------------------------------- placebo: random scores through the same machine
    real_gross = summary(books[(chosen["bayes"], "bayes")].gross)["sharpe"]
    placebo = []
    real_scores = sel_comb.scores["bayes"]
    for seed in range(N_PLACEBO):
        rng = np.random.default_rng(seed)
        fake = pd.DataFrame(rng.normal(size=real_scores.shape), index=real_scores.index,
                            columns=real_scores.columns).where(real_scores.notna())
        r = pl.long_short_book(data, fake, sel_comb.expo, CostConfig(trade_fraction=sel["frac"]), sel_cons)
        placebo.append(summary(r.gross)["sharpe"])
    placebo = np.array(placebo)
    p_value = (1 + (placebo >= real_gross).sum()) / (1 + len(placebo))
    print(f"placebo done ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- costs for the selected book
    sel_book = books[(chosen["bayes"], "bayes")]
    per_bps = sel_book.costs / CostConfig().cost_bps
    cost_levels = [0, 2, 5, 10, 20, 30]
    cost_sr = {c: summary(sel_book.gross - per_bps * c - sel_book.borrow)["sharpe"] for c in cost_levels}

    # ---------------------------------------------------------------- survivorship for the selected configuration
    surv_rows, surv = [], {}
    if (SNAPSHOT / "signals.parquet").exists():
        common = [c for c in pd.read_parquet(SNAPSHOT / "signals.parquet").columns if c in data.z]
        for lab, path in (("with delisted stocks (main)", PROCESSED_DIR), ("survivors only (yfinance)", SNAPSHOT)):
            dd = pl.load(path, signals=common)
            cc = pl.combine(dd, methods=METHODS, n_pca=sel["n_pca"])
            for m in METHODS:
                r = pl.long_short_book(dd, cc.scores[m], cc.expo, CostConfig(trade_fraction=sel["frac"]), sel_cons)
                s = split_stats(r)["full"]
                surv[f"{m}_{'main' if path == PROCESSED_DIR else 'survivors'}"] = s
                surv_rows.append(f"| {LABELS[m]} | {lab} | {s['gross']:.2f} | {s['net']:.2f} | "
                                 f"{100 * s['ret']:.1f}% | {100 * s['vol']:.1f}% |")
        print(f"survivorship done ({time.time() - t0:.0f}s)")

    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.hist(placebo, bins=15, color="#c4c4c4", edgecolor="white", label=f"{N_PLACEBO} random-score placebos")
    ax.axvline(real_gross, color="#2a6fb0", lw=2, label=f"real composite ({real_gross:.2f})")
    ax.set_xlabel("gross Sharpe ratio")
    ax.set_ylabel("count")
    ax.set_title("Placebo test: same pipeline, random scores")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "imp_placebo.png", dpi=150)
    plt.close(fig)

    rows = []
    for label, _ in VARIANTS:
        for m in METHODS:
            s = results[(label, m)]
            mark = " **(selected)**" if chosen[m] == label else ""
            rows.append(f"| {label}{mark} | {LABELS[m]} | {s['dev']['gross']:.2f} | {s['dev']['net']:.2f} | "
                        f"{s['holdout']['gross']:.2f} | {s['holdout']['net']:.2f} | {s['full']['net']:.2f} | "
                        f"{100 * s['full']['vol']:.1f}% | {100 * s['full']['dd']:.1f}% | {100 * s['turnover']:.0f}% |")

    lines = [
        "# Stage 7 — Improvements (development vs holdout)",
        "",
        "**Protocol, fixed before running.** Development period: start of trading to 2022-12-31; holdout: "
        "2023-01-01 to the end of the data. Each variant adds one idea motivated by theory, with parameters set "
        "a priori (5 PCA factors, 50% / 25% trade rates). For each method, the variant with the best "
        "*development* net Sharpe is selected and its holdout result is reported as is. Caveat: full-sample "
        "results of the baseline had been seen before this protocol was set, so the holdout is not perfectly "
        "untouched; the improvements themselves were never evaluated on it before selection.",
        "",
        "## The ideas",
        "",
        "1. **Statistical risk model.** Neutralise signals, training returns and the portfolio to the first five "
        "principal components of trailing one-year returns, on top of market beta. In large caps these are "
        "market, sector and style factors; without point-in-time sector data this is the standard way to remove "
        "them, and it removes noise the signals are not meant to bet on.",
        "2. **Risk-scaled weights.** Weight ∝ score / idiosyncratic volatility. With alpha = IC × σ × z "
        "(Grinold), the optimal weight under a factor model is ∝ z / σ: the same conviction takes a smaller "
        "position in a more volatile stock, so risk is spread more evenly.",
        "3. **Cost-aware trading.** Move only part of the way to the target each month. Under quadratic "
        "trading costs this is the optimal policy (Gârleanu & Pedersen, 2013): trade towards an aim portfolio "
        "at a rate set by costs and alpha decay. Stage 6 showed turnover is the binding constraint.",
        "",
        "## Results",
        "",
        "| Variant | Method | Dev Sharpe gross | Dev Sharpe net | Holdout Sharpe gross | Holdout Sharpe net | "
        "Full net | Vol | Max DD | Turnover/month |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "![ladder](figures/imp_ladder.png)",
        "",
        f"## Factor attribution of the selected Bayesian book ({chosen['bayes'].split(' ')[0]})",
        "",
        "| Factor | Loading | t (Newey-West) |",
        "|---|---|---|",
        *[f"| {k} | {v.coef:+.3f} | {v.t_stat:+.2f} |" for k, v in ff.iterrows()],
        f"| R² | {ff.attrs['r2']:.2f} | |",
        "",
        "Alpha is annualised. A book neutral to statistical factors should load little on the Fama-French factors.",
        "",
        "## Placebo test (is the machine itself producing returns?)",
        "",
        f"The selected configuration is fed {N_PLACEBO} sets of *random* scores (same dates, same stocks). A correct "
        "pipeline must earn nothing from noise; a positive average would reveal look-ahead or a construction bias.",
        "",
        f"Placebo gross Sharpe: mean {placebo.mean():+.2f}, standard deviation {placebo.std():.2f}, "
        f"max {placebo.max():+.2f}. Real signal: **{real_gross:+.2f}**, empirical p-value {p_value:.3f}.",
        "",
        "![placebo](figures/imp_placebo.png)",
        "",
        "## Cost sensitivity of the selected Bayesian book",
        "",
        "| Cost (bps per $ traded) | " + " | ".join(str(c) for c in cost_levels) + " |",
        "|---|" + "---|" * len(cost_levels),
        "| Net Sharpe | " + " | ".join(f"{cost_sr[c]:.2f}" for c in cost_levels) + " |",
        "",
        *(["## Survivorship bias, selected configuration", "",
           "Same configuration and the 12 original signals on the main data and on a yfinance-only snapshot "
           "(which silently drops every delisted stock).", "",
           "| Method | Dataset | Sharpe gross | Sharpe net | Return net | Vol |", "|---|---|---|---|---|---|",
           *surv_rows, ""] if surv_rows else []),
    ]
    (REPORTS_DIR / "improvements.md").write_text("\n".join(lines) + "\n")
    save_results("improvements", {
        "chosen": chosen,
        "table": {f"{label.split(' ')[0]}_{m}": results[(label, m)] for label, _ in VARIANTS for m in METHODS},
        "labels": {label.split(" ")[0]: label for label, _ in VARIANTS},
        "chosen_ff_alpha": ff.at["alpha", "coef"], "chosen_ff_alpha_t": ff.at["alpha", "t_stat"],
        "chosen_ff_r2": ff.attrs["r2"], "placebo_mean": placebo.mean(), "placebo_sd": placebo.std(),
        "placebo_max": placebo.max(), "real_gross": real_gross, "p_value": p_value,
        "cost_sr": {str(k): v for k, v in cost_sr.items()}, "survivorship": surv})
    print(f"Done in {time.time() - t0:.0f}s -> reports/improvements.md; chosen: {chosen}")


if __name__ == "__main__":
    main()
