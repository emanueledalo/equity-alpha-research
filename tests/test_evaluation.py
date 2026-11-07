import numpy as np
import pandas as pd
import pytest

from alpha_research.evaluation import ic as ic_mod
from alpha_research.evaluation import portfolios as pf
from alpha_research.evaluation.factor_model import FACTORS, factor_regression
from alpha_research.evaluation.stats import newey_west_se

DATES = pd.bdate_range("2020-01-01", periods=60)
TICKERS = [f"S{i}" for i in range(50)]


def _frame(values) -> pd.DataFrame:
    return pd.DataFrame(values, index=DATES[: len(values)], columns=TICKERS[: np.shape(values)[1]])


def test_rank_ic_perfect_and_inverse_signal():
    rng = np.random.default_rng(1)
    fwd = _frame(rng.normal(size=(10, 50)))
    assert np.allclose(ic_mod.rank_ic(fwd * 3 + 1, fwd), 1.0)
    assert np.allclose(ic_mod.rank_ic(-fwd, fwd), -1.0)


def test_rank_ic_needs_enough_stocks():
    rng = np.random.default_rng(2)
    fwd = _frame(rng.normal(size=(3, 50)))
    sig = fwd.copy()
    sig.iloc[0, 10:] = np.nan                      # only 10 stocks on day 0
    ic = ic_mod.rank_ic(sig, fwd, min_count=30)
    assert np.isnan(ic.iloc[0]) and ic.iloc[1] == pytest.approx(1.0)


def test_newey_west_matches_iid_and_grows_with_autocorrelation():
    rng = np.random.default_rng(3)
    iid = rng.normal(size=20000)
    assert newey_west_se(iid, 0) == pytest.approx(iid.std() / np.sqrt(len(iid)), rel=1e-3)
    # Overlapping 21-day sums are strongly autocorrelated: NW se must be much larger than naive.
    overlapping = pd.Series(iid).rolling(21).sum().dropna().to_numpy()
    naive = overlapping.std() / np.sqrt(len(overlapping))
    assert newey_west_se(overlapping, 20) > 3 * naive


def test_quantile_labels_split_evenly():
    sig = _frame([np.arange(50, dtype=float)])
    labels = pf.quantile_labels(sig, 5).iloc[0]
    assert labels.value_counts().tolist() == [10] * 5
    assert labels["S0"] == 1 and labels["S49"] == 5


def test_holdings_respect_execution_delay_and_do_not_carry_dropped_stocks():
    sig = _frame(np.tile(np.arange(50, dtype=float), (len(DATES), 1)))
    sig.loc[DATES[20]:, "S49"] = np.nan            # S49 leaves the universe before the 2nd rebalance
    rebalance = pd.DatetimeIndex([DATES[5], DATES[25]])
    held = pf.holdings(sig, rebalance, 5)

    assert held.iloc[:7].isna().all().all()        # decided at close of day 5, traded at close of 6
    assert held.iloc[7]["S49"] == 5                # first return earned on day 7
    assert held.iloc[27]["S48"] == 5
    assert np.isnan(held.iloc[27]["S49"])          # not carried over from the previous sort
    assert held.iloc[26]["S49"] == 5               # still held until the new sort takes effect


def test_quantile_returns_skip_missing_stock_returns():
    held = _frame([[1, 1, 2, 2]] * 2)
    ret = _frame([[0.01, np.nan, 0.03, 0.05], [0.02, 0.04, -0.01, 0.01]])
    q = pf.quantile_returns(held, ret, n_quantiles=2)
    assert q.loc[DATES[0], "Q1"] == pytest.approx(0.01)    # delisted stock dropped, not NaN
    assert q.loc[DATES[1], "Q2"] == pytest.approx(0.0)


def test_turnover():
    sig = _frame([np.arange(50, dtype=float), np.arange(50, dtype=float)[::-1]])
    t = pf.turnover(sig, DATES[:2], quantile=5)
    assert t.iloc[0] == pytest.approx(1.0)         # complete reversal -> all names replaced


def test_factor_regression_recovers_alpha_and_loadings():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2010-01-01", periods=3000)
    factors = pd.DataFrame(rng.normal(0, 0.01, (len(idx), len(FACTORS))), index=idx, columns=FACTORS)
    y = 0.0004 + 0.5 * factors["MOM"] - 0.3 * factors["Mkt-RF"] + rng.normal(0, 0.002, len(idx))
    res = factor_regression(y, factors)
    assert res.at["alpha", "coef"] == pytest.approx(0.0004 * 252, rel=0.15)
    assert res.at["MOM", "coef"] == pytest.approx(0.5, abs=0.02)
    assert res.at["Mkt-RF", "coef"] == pytest.approx(-0.3, abs=0.02)
    assert res.at["alpha", "t_stat"] > 3
