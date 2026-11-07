"""Stage 3: evaluate every signal on its own.

    python scripts/evaluate_signals.py

Inputs : data/processed/{signals,targets,prices,factors}.parquet
Outputs: reports/evaluation.md, reports/figures/eval_*.png,
         data/processed/eval_longshort.parquet (daily Q5-Q1 returns, for later stages)
"""

from __future__ import annotations

import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha_research.config import PROCESSED_DIR, REPORTS_DIR
from alpha_research.evaluation import ic as ic_mod
from alpha_research.evaluation import portfolios as pf
from alpha_research.backtest.engine import TRADING_DAYS, summary
from alpha_research.combination.periods import rebalance_dates
from alpha_research.evaluation.factor_model import factor_regression
from alpha_research.signals.library import SIGNALS
from alpha_research.signals.targets import HORIZONS

FIG = REPORTS_DIR / "figures"
N_Q = 5


def load() -> tuple[dict[str, pd.DataFrame], dict[int, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    sig = pd.read_parquet(PROCESSED_DIR / "signals.parquet")
    signals = {c: sig[c].unstack("ticker") for c in sig.columns}
    calendar = next(iter(signals.values())).index
    tgt = pd.read_parquet(PROCESSED_DIR / "targets.parquet")
    fwd = {h: tgt[f"fwd_ret_{h}"].unstack("ticker").reindex(calendar) for h in HORIZONS}
    ret = pd.read_parquet(PROCESSED_DIR / "prices.parquet", columns=["ret"])["ret"].unstack("ticker")
    factors = pd.read_parquet(PROCESSED_DIR / "factors.parquet")
    return signals, fwd, ret.reindex(calendar), factors


def evaluate(name: str, signal: pd.DataFrame, fwd: dict[int, pd.DataFrame], ret: pd.DataFrame,
             factors: pd.DataFrame, rebalance: pd.DatetimeIndex) -> dict:
    out: dict = {"name": name}
    ics = {h: ic_mod.rank_ic(signal, fwd[h]) for h in HORIZONS}
    out["ic"] = {h: ic_mod.ic_summary(ics[h], h) for h in HORIZONS}
    out["ic_year"] = ic_mod.ic_by_year(ics[21])
    out["decay"] = ic_mod.ic_decay(signal, fwd[5])

    held = pf.holdings(signal, rebalance, N_Q)
    q = pf.quantile_returns(held, ret, N_Q)
    q = q.loc[q.notna().all(axis=1)]
    ls = q[f"Q{N_Q}"] - q["Q1"]
    out["quantiles"] = q.mean() * TRADING_DAYS
    out["ls"] = ls
    out["perf"] = summary(ls)
    out["turnover"] = np.mean([pf.turnover(signal, rebalance, k, N_Q).mean() for k in (1, N_Q)])
    out["ff"] = factor_regression(ls, factors)
    return out


# --------------------------------------------------------------------------- figures

def fig_ic_bars(results: list[dict]) -> None:
    names = [r["name"] for r in results]
    ic = [r["ic"][21]["mean_ic"] for r in results]
    t = [r["ic"][21]["t_stat"] for r in results]
    colors = ["#2a6fb0" if abs(x) >= 3 else "#7fa9d1" if abs(x) >= 2 else "#c4c4c4" for x in t]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(names, ic, color=colors)
    for i, (v, tt) in enumerate(zip(ic, t)):
        ax.text(i, v + (0.001 if v >= 0 else -0.001), f"t={tt:.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("mean rank IC, 21-day forward return")
    ax.set_title("Signal IC (dark: |t| ≥ 3, light: |t| ≥ 2, grey: not significant)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(FIG / "eval_ic.png", dpi=150)
    plt.close(fig)


def fig_ic_year(results: list[dict]) -> None:
    mat = pd.DataFrame({r["name"]: r["ic_year"] for r in results}).T
    lim = np.nanmax(np.abs(mat.to_numpy()))
    fig, ax = plt.subplots(figsize=(11, 5))
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns)
    ax.set_yticks(range(mat.shape[0]), mat.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.8, label="mean rank IC (21d)")
    ax.set_title("IC by year: is the signal stable across regimes?")
    fig.tight_layout()
    fig.savefig(FIG / "eval_ic_by_year.png", dpi=150)
    plt.close(fig)


def fig_cumulative(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(sorted(results, key=lambda r: -r["perf"]["sharpe"])):
        wealth = (1 + r["ls"]).cumprod()
        ax.plot(wealth.index, wealth, lw=1.1, color=cmap(i), label=f"{r['name']} (SR {r['perf']['sharpe']:.2f})")
    ax.set_yscale("log")
    ax.axhline(1, color="black", lw=0.6)
    ax.set_ylabel("growth of $1, long Q5 / short Q1 (log)")
    ax.set_title("Long-short quintile portfolios, monthly rebalance, before costs")
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "eval_longshort.png", dpi=150)
    plt.close(fig)


def fig_quantiles(results: list[dict]) -> None:
    n_rows = int(np.ceil(len(results) / 4))
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 2.5 * n_rows), sharey=True)
    for ax, r in zip(axes.flat, results):
        q = r["quantiles"]
        ax.bar(q.index, 100 * q, color=["#c9d9ea", "#9dbcdc", "#6f9fcd", "#4682be", "#2a6fb0"])
        ax.set_title(r["name"], fontsize=9)
        ax.axhline(0, color="black", lw=0.6)
    for ax in axes[:, 0]:
        ax.set_ylabel("ann. return %")
    fig.suptitle("Annualised return by signal quintile (Q1 = lowest, Q5 = highest signal)")
    fig.tight_layout()
    fig.savefig(FIG / "eval_quantiles.png", dpi=150)
    plt.close(fig)


def fig_decay(results: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cmap = plt.get_cmap("tab20")
    for i, r in enumerate(results):
        ax.plot(r["decay"].index, r["decay"], marker="o", ms=3, lw=1, color=cmap(i), label=r["name"])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xlabel("signal age (trading days)")
    ax.set_ylabel("mean rank IC vs 5-day forward return")
    ax.set_title("IC decay: how fast does the information fade?")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "eval_decay.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- report

def verdict(t: float) -> str:
    if abs(t) >= 3:
        return "**significant (t ≥ 3)**" if t > 0 else "**reversed (t ≤ −3)**"
    if abs(t) >= 2:
        return "weak (2 ≤ |t| < 3)" if t > 0 else "weakly reversed"
    return "not significant"


def write_report(results: list[dict], start: pd.Timestamp, end: pd.Timestamp) -> None:
    results = sorted(results, key=lambda r: -r["ic"][21]["t_stat"])
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731

    summary = [
        "| Signal | IC 21d | t (NW) | IC IR | Hit rate | L/S ann. return | L/S Sharpe | L/S max DD | "
        "FF6 alpha | alpha t | Turnover/month | Verdict |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        i, p, ff = r["ic"][21], r["perf"], r["ff"]
        summary.append(
            f"| `{r['name']}` | {i['mean_ic']:.4f} | {i['t_stat']:.2f} | {i['ic_ir']:.2f} | {pct(i['hit_rate'])} | "
            f"{pct(p['ann_return'])} | {p['sharpe']:.2f} | {pct(p['max_drawdown'])} | "
            f"{pct(ff.loc['alpha', 'coef'])} | {ff.loc['alpha', 't_stat']:.2f} | {pct(r['turnover'])} | "
            f"{verdict(i['t_stat'])} |")

    horizons = ["| Signal | " + " | ".join(f"IC {h}d (t)" for h in HORIZONS) + " |",
                "|---|" + "---|" * len(HORIZONS)]
    for r in results:
        horizons.append(f"| `{r['name']}` | " + " | ".join(
            f"{r['ic'][h]['mean_ic']:.4f} ({r['ic'][h]['t_stat']:.1f})" for h in HORIZONS) + " |")

    loadings = ["| Signal | alpha (ann.) | " + " | ".join(results[0]["ff"].index[1:]) + " | R² |",
                "|---|---|" + "---|" * (len(results[0]["ff"]) - 1) + "---|"]
    for r in results:
        ff = r["ff"]
        cells = [f"{ff.at[f, 'coef']:.2f} ({ff.at[f, 't_stat']:.1f})" for f in ff.index[1:]]
        loadings.append(f"| `{r['name']}` | {pct(ff.at['alpha', 'coef'])} ({ff.at['alpha', 't_stat']:.1f}) | "
                        + " | ".join(cells) + f" | {ff.attrs['r2']:.2f} |")

    n_sig = sum(abs(r["ic"][21]["t_stat"]) >= 3 for r in results)
    lines = [
        "# Stage 3 — Single-signal evaluation",
        "",
        f"Sample: **{start:%Y-%m-%d} to {end:%Y-%m-%d}**, point-in-time S&P 500 members. "
        "Signals observed at the close of *t*, traded at the close of *t+1*. **No transaction costs** "
        "(Stage 5); turnover is shown so their impact can be judged.",
        "",
        "## Summary (sorted by IC t-stat)",
        "",
        "- **IC**: mean daily Spearman correlation between the signal and the next 21-day return. "
        "t-stats use Newey-West with 20 lags, because consecutive 21-day returns overlap.",
        "- **IC IR**: mean/std of the IC, annualised with 12 independent periods per year.",
        "- **L/S**: equal-weighted top quintile minus bottom quintile, rebalanced at month-ends.",
        "- **FF6 alpha**: intercept of the L/S returns regressed on the Fama-French 5 factors + momentum.",
        "",
        *summary,
        "",
        f"{n_sig} of {len(results)} signals clear |t| ≥ 3. With {len(results)} signals × 3 horizons tested, "
        "t ≥ 3 is the relevant hurdle (Harvey, Liu & Zhu, 2016): at t ≥ 2 roughly two false "
        "positives would be expected by chance alone.",
        "",
        "![IC](figures/eval_ic.png)",
        "",
        "## IC across horizons",
        "",
        *horizons,
        "",
        "## Stability across years",
        "",
        "![IC by year](figures/eval_ic_by_year.png)",
        "",
        "## Long-short performance",
        "",
        "![Long-short](figures/eval_longshort.png)",
        "",
        "## Monotonicity across quintiles",
        "",
        "A real effect should show returns rising (roughly) steadily from Q1 to Q5, not only in one extreme quintile.",
        "",
        "![Quintiles](figures/eval_quantiles.png)",
        "",
        "## IC decay",
        "",
        "![Decay](figures/eval_decay.png)",
        "",
        "## Factor exposures of the long-short portfolios",
        "",
        "Coefficient (Newey-West t-stat). A significant alpha means the signal is not just a repackaged known factor.",
        "",
        *loadings,
        "",
        "## Caveats",
        "",
        "- Large-cap universe: anomalies are known to be weaker among S&P 500 stocks than in small caps.",
        "- Equal-weighted quintiles of ~100 stocks; no costs, borrow fees or capacity limits yet.",
        "- Delisting returns are not available in free data; a delisted stock drops out of its portfolio "
        "from its last trading day.",
        "- Fama-French factors end at the latest published month; the regression uses the overlapping period.",
    ]
    (REPORTS_DIR / "evaluation.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    t0 = time.time()
    signals, fwd, ret, factors = load()
    calendar = next(iter(signals.values())).index
    rebalance = rebalance_dates(calendar)
    FIG.mkdir(parents=True, exist_ok=True)

    results = []
    for s in SIGNALS:
        results.append(evaluate(s.name, signals[s.name], fwd, ret, factors, rebalance))
        r = results[-1]
        print(f"{s.name:<13} IC21={r['ic'][21]['mean_ic']:+.4f} t={r['ic'][21]['t_stat']:+.2f} "
              f"L/S SR={r['perf']['sharpe']:+.2f} ({time.time() - t0:4.0f}s)")

    pd.DataFrame({r["name"]: r["ls"] for r in results}).to_parquet(PROCESSED_DIR / "eval_longshort.parquet")
    fig_ic_bars(results)
    fig_ic_year(results)
    fig_cumulative(results)
    fig_quantiles(results)
    fig_decay(results)
    ls_index = results[0]["ls"].index
    write_report(results, ls_index.min(), ls_index.max())
    from alpha_research.config import save_results
    save_results("evaluation", {r["name"]: {
        **{f"ic{h}": r["ic"][h]["mean_ic"] for h in HORIZONS}, **{f"t{h}": r["ic"][h]["t_stat"] for h in HORIZONS},
        "ls_sharpe": r["perf"]["sharpe"], "ff_alpha": r["ff"].at["alpha", "coef"],
        "ff_alpha_t": r["ff"].at["alpha", "t_stat"], "ff_mkt": r["ff"].at["Mkt-RF", "coef"],
        "ff_mom": r["ff"].at["MOM", "coef"], "ff_r2": r["ff"].attrs["r2"], "turnover": r["turnover"]}
        for r in results})
    print(f"Done in {time.time() - t0:.0f}s -> reports/evaluation.md")


if __name__ == "__main__":
    main()
