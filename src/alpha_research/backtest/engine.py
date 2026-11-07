"""Daily backtest engine with drifting weights, transaction costs and borrow fees.

Timeline of one trading day i (all at the close):
  1. Positions held since the previous close earn day i's returns.
  2. Weights drift with those returns (a winner becomes a bigger position).
  3. Borrow fee is charged on the short side.
  4. If day i is an execution day (rebalance date + 1), trade from the drifted
     weights to the new targets and pay costs on the traded amount.

Weights are fractions of capital, so the daily strategy return is sum_i w_i r_i.
A stock with no return on a day (halted, delisted) earns 0; after a delisting its
position is effectively closed at the last price and sold at the next rebalance.

Cost model: cost = cost_bps * sum_i |trade_i|, i.e. a fixed cost per dollar traded
covering half-spread, commissions and impact. For liquid S&P 500 names 5 bps per
side is a reasonable institutional estimate; the robustness stage shows results
from 0 to 30 bps so the reader can apply their own assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_research.combination.periods import EXECUTION_DELAY

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostConfig:
    cost_bps: float = 5.0             # per unit traded, one way
    borrow_bps_annual: float = 25.0   # general-collateral borrow fee on short positions
    trade_fraction: float = 1.0       # 1 = trade fully to target; <1 = partial rebalancing


@dataclass
class BacktestResult:
    gross: pd.Series                  # daily return before costs
    net: pd.Series                    # daily return after costs and borrow
    costs: pd.Series                  # daily trading cost
    borrow: pd.Series                 # daily borrow fee
    turnover: pd.Series               # one-way turnover on each execution day
    exposures: pd.DataFrame = field(default_factory=pd.DataFrame)  # long, short, net, gross per execution day


def run_backtest(targets: pd.DataFrame, ret: pd.DataFrame, costs: CostConfig = CostConfig()) -> BacktestResult:
    calendar = ret.index
    tickers = targets.columns
    r_all = ret.reindex(columns=tickers).to_numpy()
    tgt = targets.to_numpy()

    exec_pos = calendar.get_indexer(targets.index) + EXECUTION_DELAY
    valid = (exec_pos >= EXECUTION_DELAY) & (exec_pos < len(calendar))
    exec_of_day = dict(zip(exec_pos[valid], np.flatnonzero(valid)))
    first = exec_pos[valid].min()

    w = np.zeros(len(tickers))
    gross, cost, borrow = np.zeros(len(calendar)), np.zeros(len(calendar)), np.zeros(len(calendar))
    turnover, exposures = {}, {}
    daily_borrow = costs.borrow_bps_annual / 1e4 / TRADING_DAYS

    for i in range(first, len(calendar)):
        r = np.nan_to_num(r_all[i])
        port = w @ r
        gross[i] = port
        # Drift: position values grow with their returns; capital grows with the portfolio.
        w = w * (1 + r) / (1 + port) if port > -1 else w * 0.0
        borrow[i] = daily_borrow * -w[w < 0].sum()
        if i in exec_of_day:
            k = exec_of_day[i]
            new_w = w + costs.trade_fraction * (tgt[k] - w)
            traded = np.abs(new_w - w).sum()
            cost[i] = costs.cost_bps / 1e4 * traded
            turnover[calendar[i]] = traded / 2
            w = new_w
            exposures[calendar[i]] = {"long": w[w > 0].sum(), "short": w[w < 0].sum(),
                                      "net": w.sum(), "gross": np.abs(w).sum()}

    idx = calendar[first:]
    s = lambda a: pd.Series(a[first:], index=idx)  # noqa: E731
    return BacktestResult(gross=s(gross), net=s(gross - cost - borrow), costs=s(cost), borrow=s(borrow),
                          turnover=pd.Series(turnover), exposures=pd.DataFrame(exposures).T)


def monthly_turnover(result: BacktestResult) -> float:
    """Average one-way turnover per calendar month (comparable across rebalance frequencies)."""
    t = result.turnover
    return float(t.groupby(t.index.to_period("M")).sum().mean())


def summary(daily: pd.Series, benchmark: pd.Series | None = None) -> dict[str, float]:
    """Performance statistics of a daily return series (optionally vs a benchmark)."""
    r = daily.dropna()
    ann_ret = r.mean() * TRADING_DAYS
    ann_vol = r.std() * np.sqrt(TRADING_DAYS)
    wealth = (1 + r).cumprod()
    out = {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": ann_ret / ann_vol if ann_vol > 0 else np.nan,
        "t_stat": r.mean() / r.std() * np.sqrt(len(r)) if r.std() > 0 else np.nan,
        "max_drawdown": (wealth / wealth.cummax() - 1).min(),
        "hit_rate_monthly": (r.groupby(r.index.to_period("M")).apply(lambda x: (1 + x).prod() - 1) > 0).mean(),
    }
    if benchmark is not None:
        b = benchmark.reindex(r.index).fillna(0)
        cov = np.cov(r, b)
        out["beta"] = cov[0, 1] / cov[1, 1]
        active = r - b
        te = active.std() * np.sqrt(TRADING_DAYS)
        out["active_return"] = active.mean() * TRADING_DAYS
        out["tracking_error"] = te
        out["info_ratio"] = out["active_return"] / te if te > 0 else np.nan
    return out
