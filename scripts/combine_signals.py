"""Stage 4: combine the signals into one composite score (walk-forward).

    python scripts/combine_signals.py

Outputs: reports/combination.md, reports/figures/comb_*.png,
         data/processed/composite_<method>.parquet (scores per rebalance date)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_research import pipeline as pl
from alpha_research.config import PROCESSED_DIR, REPORTS_DIR
from alpha_research.combination.weights import PRIOR_IC, PRIOR_IC_SD, bayes_posterior, ledoit_wolf
from alpha_research.evaluation.ic import rank_ic

FIG = REPORTS_DIR / "figures"
LABELS = {"equal": "Equal weight", "ic_weighted": "IC-weighted", "bayes": "Bayesian shrinkage",
          "ridge": "Ridge regression"}


def composite_ic(scores: pd.DataFrame, hret: pd.DataFrame) -> dict[str, float]:
    ic = rank_ic(scores, hret.reindex(index=scores.index, columns=scores.columns)).dropna()
    return {"mean_ic": ic.mean(), "t_stat": ic.mean() / ic.std() * np.sqrt(len(ic)),
            "ic_ir": ic.mean() / ic.std() * np.sqrt(12), "hit_rate": (ic > 0).mean(), "months": len(ic)}


def fig_weights(weights: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    cmap = plt.get_cmap("tab20")
    for ax, (m, w) in zip(axes.flat, weights.items()):
        if (w >= -1e-12).all().all():
            ax.stackplot(w.index, w.T.to_numpy(), labels=w.columns, colors=[cmap(i) for i in range(w.shape[1])])
            ax.set_ylim(0, 1)
        else:
            for i, c in enumerate(w.columns):
                ax.plot(w.index, w[c], color=cmap(i), lw=1, label=c)
            ax.axhline(0, color="black", lw=0.6)
        ax.set_title(LABELS[m], fontsize=10)
    axes[0, 0].legend(fontsize=6.5, ncol=2, loc="upper left")
    fig.suptitle("Signal weights over time (walk-forward: each date uses only data known by then)")
    fig.tight_layout()
    fig.savefig(FIG / "comb_weights.png", dpi=150)
    plt.close(fig)


def fig_payoffs(f: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab20")
    cum = f.fillna(0).cumsum()
    for i, c in enumerate(cum.columns):
        ax.plot(cum.index, cum[c], lw=1.1, color=cmap(i), label=c)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("cumulative payoff per unit signal exposure")
    ax.set_title("Beta-neutral signal payoffs (Fama-MacBeth slopes), monthly")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "comb_payoffs.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    data = pl.load(PROCESSED_DIR)
    comb = pl.combine(data)

    for m, s in comb.scores.items():
        s.to_parquet(PROCESSED_DIR / f"composite_{m}.parquet")

    fig_weights(comb.weights)
    fig_payoffs(comb.f)

    # Out-of-sample IC of each composite, against beta-neutral and raw period returns.
    from alpha_research.combination.periods import holding_returns
    raw = holding_returns(data.ret, comb.periods)
    rows = []
    ic_results = {}
    for m, s in comb.scores.items():
        a, b = composite_ic(s, comb.hret), composite_ic(s, raw)
        ic_results[m] = {"ic": a["mean_ic"], "t": a["t_stat"], "ic_ir": a["ic_ir"], "hit": a["hit_rate"],
                         "ic_raw": b["mean_ic"], "t_raw": b["t_stat"], "months": a["months"]}
        rows.append(f"| {LABELS[m]} | {a['mean_ic']:.4f} | {a['t_stat']:.2f} | {a['ic_ir']:.2f} | "
                    f"{a['hit_rate']:.0%} | {b['mean_ic']:.4f} | {b['t_stat']:.2f} | {a['months']} |")

    # Bayesian posterior at the last date, and Ledoit-Wolf shrinkage intensity.
    known = (comb.periods["known_at"] <= comb.periods.index[-1]).to_numpy()
    f_known = comb.f[known].dropna()
    scale = float(comb.hret.std(axis=1)[known].mean())
    post = bayes_posterior(f_known, scale)
    _, delta = ledoit_wolf(f_known.to_numpy())
    last_w = comb.weights["bayes"].iloc[-1]
    post_rows = [f"| `{k}` | {r.sample_ic:+.4f} | {r.post_ic:+.4f} | {r.shrinkage:.2f} | {last_w[k]:.2f} |"
                 for k, r in post.iterrows()]
    from alpha_research.config import save_results
    save_results("combination", {
        "composite_ic": ic_results, "tau_ic": float(post["tau"].iloc[0] / scale), "lw_delta": delta,
        "post_ic": float(post["post_ic"].mean()),
        "bayes_last_weights": last_w.to_dict(), "sample_ic": post["sample_ic"].to_dict(),
        "start": f"{comb.scores['equal'].index.min():%Y-%m}", "end": f"{comb.scores['equal'].index.max():%Y-%m}"})

    start, end = comb.scores["equal"].index.min(), comb.scores["equal"].index.max()
    lines = [
        "# Stage 4 — Signal combination",
        "",
        f"Out-of-sample period: **{start:%Y-%m} to {end:%Y-%m}** (monthly rebalances). Weights at each date "
        f"are estimated only from holding periods whose returns were fully known by that date, after a "
        f"{pl.MIN_TRAIN_PERIODS}-month burn-in. Training returns are beta-neutralised each period, matching "
        "the beta-neutral portfolios of Stage 5 (\"train on what you trade\").",
        "",
        "## Methods",
        "",
        "| Method | Idea | Estimation risk |",
        "|---|---|---|",
        "| Equal weight | 1/K per signal | none |",
        "| IC-weighted | ∝ historical mean payoff, floored at 0 | high: noisy means are taken at face value |",
        f"| Bayesian shrinkage | hierarchical model: each signal's premium is shrunk towards a common level "
        f"whose prior is literature-informed (average IC {PRIOR_IC} ± {PRIOR_IC_SD}); Ledoit-Wolf covariance; "
        "non-negative mean-variance weights | low: shrinkage on means and covariance |",
        "| Ridge regression | pooled regression of stock returns on all signals, λ = 1 fixed a priori | "
        "medium: weights can turn negative |",
        "",
        "## Out-of-sample composite IC (rebalance → next rebalance)",
        "",
        "| Method | IC (beta-neutral returns) | t | IC IR | Hit rate | IC (raw returns) | t | Months |",
        "|---|---|---|---|---|---|---|---|",
        *rows,
        "",
        "## Signal weights over time",
        "",
        "![weights](figures/comb_weights.png)",
        "",
        "## What the signals actually paid (beta-neutral)",
        "",
        "![payoffs](figures/comb_payoffs.png)",
        "",
        f"## Bayesian posterior at {comb.periods.index[-1]:%Y-%m-%d}",
        "",
        f"Estimated dispersion of true premia across signals τ = {post['tau'].iloc[0] / scale:.4f} (in IC units); "
        f"Ledoit-Wolf shrinkage intensity of the payoff covariance δ = {delta:.2f}.",
        "",
        "| Signal | Sample IC | Posterior IC | Shrinkage to common level | Weight |",
        "|---|---|---|---|---|",
        *post_rows,
        "",
        "When τ ≈ 0 the data cannot distinguish the signals' premia from noise, so every signal receives "
        "the same posterior premium and the weights are driven purely by the covariance: the model "
        "favours signals that diversify each other.",
    ]
    (REPORTS_DIR / "combination.md").write_text("\n".join(lines) + "\n")
    print("Done -> reports/combination.md")


if __name__ == "__main__":
    main()
