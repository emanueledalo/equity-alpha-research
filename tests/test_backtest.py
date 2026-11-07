import numpy as np
import pandas as pd
import pytest

from alpha_research.backtest.construction import ConstructionConfig, long_only_weights, long_short_weights
from alpha_research.backtest.engine import CostConfig, monthly_turnover, run_backtest

CAL = pd.bdate_range("2021-01-04", periods=10)


def test_long_short_weights_are_neutral_and_capped():
    rng = np.random.default_rng(0)
    tickers = [f"S{i}" for i in range(300)]
    score = pd.Series(rng.normal(size=300), index=tickers)
    beta = pd.Series(rng.uniform(0.4, 1.8, 300), index=tickers)
    cfg = ConstructionConfig(gross=2.0, max_weight=0.01)
    expo = pd.DataFrame({"const": 1.0, "beta": beta})
    w = long_short_weights(score, expo, cfg)
    assert w.sum() == pytest.approx(0, abs=1e-9)
    assert (w * beta).sum() == pytest.approx(0, abs=1e-9)
    assert np.abs(w).sum() == pytest.approx(2.0, rel=1e-6)
    assert np.abs(w).max() <= 0.01 * 1.01
    assert np.corrcoef(w, score)[0, 1] > 0.8       # still follows the score


def test_long_only_top_quintile():
    score = pd.Series(np.arange(100, dtype=float), index=[f"S{i}" for i in range(100)])
    w = long_only_weights(score)
    assert len(w) == 20 and w.sum() == pytest.approx(1)
    assert set(w.index) == {f"S{i}" for i in range(80, 100)}


def test_engine_timing_costs_and_drift():
    ret = pd.DataFrame({"A": [0.0, 0.10, 0.10, 0.10, 0.0, 0, 0, 0, 0, 0],
                        "B": [0.0, -0.10, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0]}, index=CAL, dtype=float)
    targets = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=[CAL[0]])     # decided at close of day 0
    res = run_backtest(targets, ret, CostConfig(cost_bps=10, borrow_bps_annual=0))

    assert res.gross.loc[CAL[1]] == 0.0                     # traded at the close of day 1: no return yet
    assert res.costs.loc[CAL[1]] == pytest.approx(10e-4 * 1.0)  # traded 0.5 + 0.5
    assert res.gross.loc[CAL[2]] == pytest.approx(0.5 * 0.10)   # first return earned on day 2
    # drift: after day 2 A is worth 0.55 of capital 1.05
    assert res.gross.loc[CAL[3]] == pytest.approx(0.55 / 1.05 * 0.10)
    assert res.net.loc[CAL[1]] == pytest.approx(-10e-4)
    assert res.turnover.iloc[0] == pytest.approx(0.5)


def test_borrow_fee_on_short_side_only():
    ret = pd.DataFrame(0.0, index=CAL, columns=["A", "B"])
    targets = pd.DataFrame({"A": [0.7], "B": [-0.3]}, index=[CAL[0]])
    res = run_backtest(targets, ret, CostConfig(cost_bps=0, borrow_bps_annual=252))
    assert res.borrow.loc[CAL[2]] == pytest.approx(0.3 * 1e-4)


def test_partial_rebalancing_moves_part_of_the_way():
    ret = pd.DataFrame(0.0, index=CAL, columns=["A", "B"])
    targets = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[CAL[0], CAL[4]])
    res = run_backtest(targets, ret, CostConfig(cost_bps=0, trade_fraction=0.5))
    assert res.exposures.loc[CAL[1], "gross"] == pytest.approx(0.5)
    assert res.exposures.loc[CAL[5], "long"] == pytest.approx(0.75)  # 0.25 A + 0.5 B
    assert monthly_turnover(res) == pytest.approx(0.25 + 0.375)  # turnover = traded / 2


def test_risk_scaled_weights_shrink_volatile_stocks():
    rng = np.random.default_rng(1)
    tickers = [f"S{i}" for i in range(200)]
    score = pd.Series(rng.normal(size=200), index=tickers)
    expo = pd.DataFrame({"const": 1.0}, index=tickers)
    idio = pd.Series(np.where(np.arange(200) < 100, 0.01, 0.03), index=tickers)   # two volatility groups
    w = long_short_weights(score, expo, ConstructionConfig(risk_scaled=True, max_weight=0.05), idio)
    calm, wild = np.abs(w.iloc[:100]).mean(), np.abs(w.iloc[100:]).mean()
    assert calm > 2 * wild                                  # ~3x smaller positions in the volatile group


def test_engine_and_holding_periods_use_the_same_timing():
    """A book 100% long one stock must earn exactly the holding-period return that the
    combination stage trains on: both modules must agree on when a position starts and ends."""
    from alpha_research.combination import periods as per

    cal = pd.bdate_range("2022-01-03", "2022-06-30")
    rng = np.random.default_rng(7)
    ret = pd.DataFrame({"A": rng.normal(0, 0.01, len(cal)), "B": 0.0}, index=cal)
    reb = per.rebalance_dates(cal)
    p = per.holding_periods(cal, reb)
    hret = per.holding_returns(ret, p)

    targets = pd.DataFrame({"A": 1.0, "B": 0.0}, index=p.index[:1])      # hold A from the first rebalance
    res = run_backtest(targets, ret, CostConfig(cost_bps=0, borrow_bps_annual=0))
    first = p.iloc[0]
    earned = (1 + res.gross.loc[cal[first["start"]]:cal[first["end"]]]).prod() - 1
    assert earned == pytest.approx(hret["A"].iloc[0], rel=1e-10)
