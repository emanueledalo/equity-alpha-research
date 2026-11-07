"""End-to-end research pipeline on a processed-data directory.

Kept as plain functions so the same code runs on the main dataset and on the
survivor-biased snapshot (robustness stage), with nothing copied between scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_research.backtest import construction as cons
from alpha_research.backtest.engine import BacktestResult, CostConfig, run_backtest
from alpha_research.combination import periods as per
from alpha_research.combination.composite import composite_scores, ridge_walk_forward
from alpha_research.combination.weights import PRIOR_IC, PRIOR_IC_SD, walk_forward
from alpha_research.risk import exposures, neutralise_frame
from alpha_research.signals.library import SIGNALS
from alpha_research.signals.preprocess import zscore

METHODS = ["equal", "ic_weighted", "bayes", "ridge"]
MIN_TRAIN_PERIODS = 36  # three years of monthly payoffs before the first trade


@dataclass
class ResearchData:
    z: dict[str, pd.DataFrame]        # oriented z-scored signals (dates x tickers)
    beta: pd.DataFrame                # raw 252-day beta, for neutralisation
    ret: pd.DataFrame                 # daily stock returns
    mkt_ret: pd.Series                # SPY daily return
    factors: pd.DataFrame             # FF5 + MOM + RF (daily)
    idio: pd.DataFrame | None = None  # raw 63-day idiosyncratic volatility (risk-scaled weights)

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return next(iter(self.z.values())).index


def load(processed: Path, signals: list[str] | None = None) -> ResearchData:
    names = signals or [s.name for s in SIGNALS]
    sig = pd.read_parquet(processed / "signals.parquet", columns=names)
    z = {c: sig[c].unstack("ticker") for c in names}
    calendar = z[names[0]].index
    raw = pd.read_parquet(processed / "signals_raw.parquet", columns=["beta", "idio_vol"])
    beta, idio = raw["beta"].unstack("ticker"), raw["idio_vol"].unstack("ticker")
    ret = pd.read_parquet(processed / "prices.parquet", columns=["ret"])["ret"].unstack("ticker")
    bench = pd.read_parquet(processed / "benchmark.parquet")["adj_close"]
    mkt = (bench / bench.shift(1) - 1).reindex(calendar)
    factors = pd.read_parquet(processed / "factors.parquet")
    tickers = z[names[0]].columns
    return ResearchData(z=z, beta=beta.reindex(index=calendar, columns=tickers),
                        ret=ret.reindex(index=calendar, columns=tickers), mkt_ret=mkt, factors=factors,
                        idio=idio.reindex(index=calendar, columns=tickers))


@dataclass
class Combination:
    periods: pd.DataFrame             # holding-period bookkeeping
    hret: pd.DataFrame                # holding-period stock returns
    f: pd.DataFrame                   # signal payoffs per period
    weights: dict[str, pd.DataFrame]  # method -> (rebalance dates x signals)
    scores: dict[str, pd.DataFrame]   # method -> (rebalance dates x tickers)
    expo: dict                        # rebalance date -> (tickers x risk factors) exposures


def combine(data: ResearchData, freq: str = "M", methods: list[str] | None = None,
            min_periods: int = MIN_TRAIN_PERIODS, prior: tuple[float, float] = (PRIOR_IC, PRIOR_IC_SD),
            n_pca: int = 0) -> Combination:
    """Walk-forward combination of all signals into one composite score per method.

    n_pca > 0 adds statistical (PCA) risk factors to the market beta that signals,
    training returns and later the portfolio are neutralised to (see risk.py).
    """
    methods = methods or METHODS

    # 1. Rebalance dates: month-ends where every signal covers more than 50 stocks.
    rebalance = per.rebalance_dates(data.calendar, freq)
    min_coverage = pd.concat([z.loc[rebalance].notna().sum(axis=1) for z in data.z.values()], axis=1).min(axis=1)
    rebalance = rebalance[(min_coverage > 50).to_numpy()]

    # 2. Holding periods and the stock returns earned over each of them.
    periods = per.holding_periods(data.calendar, rebalance)
    raw_hret = per.holding_returns(data.ret, periods)

    # 3. Train on what you trade: returns and signals are made neutral to the same
    #    risk factors (constant, beta, optional PCA factors) as the portfolio will be.
    expo = exposures(data.ret, data.beta, rebalance, n_pca=n_pca)
    hret = neutralise_frame(raw_hret, expo)
    z = {name: zscore(neutralise_frame(sig.reindex(index=rebalance), expo)) for name, sig in data.z.items()}

    # 4. What each signal paid in each period, and the cross-sectional return volatility
    #    (needed to express the Bayesian prior, set in IC units, in return units).
    f = per.signal_returns(z, hret)
    scale = hret.std(axis=1)

    # 5. Walk-forward weights per method, then the composite score.
    weights, scores = {}, {}
    for m in methods:
        if m == "ridge":
            w = ridge_walk_forward(z, hret, periods["known_at"], rebalance, min_periods=min_periods)
        else:
            w = walk_forward(f, periods["known_at"], rebalance, m, min_periods=min_periods, scale=scale,
                             prior=prior)
        weights[m] = w
        scores[m] = composite_scores(z, w)

    # 6. Common start date so all methods are compared over exactly the same months.
    start = max(w.index.min() for w in weights.values())
    weights = {m: w.loc[start:] for m, w in weights.items()}
    scores = {m: s.loc[start:] for m, s in scores.items()}
    return Combination(periods=periods, hret=hret, f=f, weights=weights, scores=scores, expo=expo)


def long_short_book(data: ResearchData, scores: pd.DataFrame, expo: dict, costs: CostConfig = CostConfig(),
                    cfg: cons.ConstructionConfig = cons.ConstructionConfig()) -> BacktestResult:
    """Backtest the market-neutral book: on each rebalance date the weights are neutral
    to that date's risk exposures `expo[date]` (from `combine`)."""
    def weights(date, score):
        return cons.long_short_weights(score, expo[date], cfg, data.idio.loc[date])
    return run_backtest(cons.build_targets(scores, weights), data.ret, costs)


def long_only_book(data: ResearchData, scores: pd.DataFrame, costs: CostConfig = CostConfig()
                   ) -> tuple[BacktestResult, BacktestResult]:
    """Top-quintile book and its equal-weighted universe benchmark (same engine, same costs)."""
    book = run_backtest(cons.build_targets(scores, lambda d, s: cons.long_only_weights(s)), data.ret, costs)
    bench = run_backtest(cons.build_targets(scores, lambda d, s: cons.benchmark_weights(s)), data.ret, costs)
    return book, bench
