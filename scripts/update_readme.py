"""Rewrite the results block of README.md from reports/results.json.

    python scripts/update_readme.py

Only the text between the RESULTS:START and RESULTS:END markers is replaced, so the
headline numbers in the README always match the latest run of the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
LABELS = {"equal": "Equal weight", "ic_weighted": "IC-weighted", "bayes": "Bayesian shrinkage",
          "ridge": "Ridge regression"}


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main() -> None:
    r = json.loads((ROOT / "reports" / "results.json").read_text())
    bt, cb, rob, imp = r["backtest"], r["combination"], r["robustness"], r.get("improvements", {})

    lines = [START, "",
             f"**Stage 5 — as specified up front.** Dollar- and beta-neutral long-short books, monthly rebalance, "
             f"trading at the close after the signal, 5 bps per dollar traded + 25 bps/year borrow, out of sample "
             f"{bt['start']} to {bt['end']}:", "",
             "| Combination method | OOS composite IC (t) | Sharpe gross | Sharpe net | Vol | Max DD | "
             "Turnover / month | Break-even cost |",
             "|---|---|---|---|---|---|---|---|"]
    for m, lab in LABELS.items():
        b, c = bt[m], cb["composite_ic"][m]
        lines.append(f"| {lab} | {c['ic']:.4f} ({c['t']:.1f}) | {b['gross_sharpe']:.2f} | {b['net_sharpe']:.2f} | "
                     f"{pct(b['vol'])} | {pct(b['max_dd'])} | {pct(b['turnover'])} | "
                     f"{rob['breakeven_bps'][m]} bps |")

    if imp:
        tab, labels = imp["table"], imp["labels"]
        lines += ["",
                  "**Stage 7 — improvements selected on 2013–2022, then judged on the 2023–2026 holdout.** Each "
                  "step adds one theory-driven idea (see [improvements](reports/improvements.md)):", "",
                  "| Variant | Method | Net Sharpe 2013–22 (dev) | Net Sharpe 2023– (holdout) | Vol | Turnover / month |",
                  "|---|---|---|---|---|---|"]
        for key in ["V0", "V1", "V2", "V3", "V4"]:
            for m in ["equal", "bayes"]:
                v = tab[f"{key}_{m}"]
                sel = " **(selected)**" if imp["chosen"][m].startswith(key) else ""
                lines.append(f"| {labels[key]}{sel} | {LABELS[m]} | {v['dev']['net']:.2f} | "
                             f"{v['holdout']['net']:.2f} | {pct(v['full']['vol'])} | {pct(v['turnover'])} |")
        lines += ["",
                  f"Placebo test: the selected configuration fed 50 sets of random scores earns a gross Sharpe of "
                  f"{imp['placebo_mean']:+.2f} on average (sd {imp['placebo_sd']:.2f}); the real composite earns "
                  f"{imp['real_gross']:+.2f} (empirical p-value {imp['p_value']:.3f})."]
    lines += ["", END]

    readme = ROOT / "README.md"
    text = readme.read_text()
    a, b = text.index(START), text.index(END) + len(END)
    readme.write_text(text[:a] + "\n".join(lines) + text[b:])
    print("README results block updated")


if __name__ == "__main__":
    main()
