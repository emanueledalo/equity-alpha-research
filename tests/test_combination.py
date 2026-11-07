import numpy as np
import pandas as pd
import pytest

from alpha_research.combination import periods as per
from alpha_research.combination.weights import (bayes_posterior, ledoit_wolf, nonneg_mean_variance,
                                                walk_forward)

CAL = pd.bdate_range("2020-01-01", "2020-12-31")


def test_holding_periods_chain_without_gaps_or_overlap():
    reb = per.rebalance_dates(CAL)
    p = per.holding_periods(CAL, reb)
    # first return day = rebalance + 2 trading days; next period starts the day after this one ends
    assert p["start"].iloc[0] == CAL.get_loc(reb[0]) + 2
    assert (p["start"].iloc[1:].to_numpy() == p["end"].iloc[:-1].to_numpy() + 1).all()
    # a period's return is known only at its last day, i.e. one day AFTER the next rebalance
    assert (p["known_at"].iloc[:-1].to_numpy() > p.index[1:].to_numpy()).all()


def test_holding_returns_compound_daily_returns():
    ret = pd.DataFrame({"A": 0.01, "B": np.nan}, index=CAL)
    p = per.holding_periods(CAL, per.rebalance_dates(CAL))
    h = per.holding_returns(ret, p)
    n_days = p["end"].iloc[0] - p["start"].iloc[0] + 1
    assert h["A"].iloc[0] == pytest.approx(1.01 ** n_days - 1)
    assert h["B"].isna().all()                      # no data at all -> NaN, not 0


def test_walk_forward_never_uses_unknown_periods():
    idx = pd.date_range("2015-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(0)
    f = pd.DataFrame(rng.normal(0.001, 0.01, (60, 3)), index=idx, columns=list("abc"))
    known_at = pd.Series(idx + pd.offsets.MonthEnd(1) + pd.Timedelta(days=1), index=idx)
    date = idx[45]
    w1 = walk_forward(f, known_at, pd.DatetimeIndex([date]), "ic_weighted").iloc[0]
    f2 = f.copy()
    f2.loc[idx[44]:, "a"] = 10.0                    # periods not yet known at `date`
    w2 = walk_forward(f2, known_at, pd.DatetimeIndex([date]), "ic_weighted").iloc[0]
    pd.testing.assert_series_equal(w1, w2)


def test_bayes_shrinks_noisy_signal_more():
    rng = np.random.default_rng(1)
    n = 240
    f = pd.DataFrame({"steady": rng.normal(0.004, 0.005, n), "noisy": rng.normal(0.004, 0.05, n),
                      "zero": rng.normal(0.0, 0.005, n)})
    post = bayes_posterior(f, scale=0.08)
    assert post.at["noisy", "shrinkage"] > post.at["steady", "shrinkage"]
    assert (post["post_sd"] <= post["se"] + 1e-12).all()   # the posterior is never less certain than the data


def test_bayes_pools_to_common_level_when_signals_indistinguishable():
    rng = np.random.default_rng(2)
    f = pd.DataFrame(rng.normal(0.001, 0.02, (120, 4)), columns=list("abcd"))
    post = bayes_posterior(f, scale=0.08)
    if post["tau"].iloc[0] == 0:
        assert post["post_mean"].nunique() == 1


def test_prior_pulls_common_level():
    rng = np.random.default_rng(3)
    f = pd.DataFrame(rng.normal(0.0, 0.02, (36, 4)), columns=list("abcd"))
    low = bayes_posterior(f, 0.08, prior_ic=-0.05, prior_ic_sd=0.005)["mu0"].iloc[0]
    high = bayes_posterior(f, 0.08, prior_ic=0.05, prior_ic_sd=0.005)["mu0"].iloc[0]
    assert high > low


def test_ledoit_wolf_is_well_conditioned():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(40, 30))                   # few observations, many variables
    cov, delta = ledoit_wolf(x)
    assert 0 <= delta <= 1
    assert np.linalg.eigvalsh(cov).min() > 0
    assert np.linalg.cond(cov) < np.linalg.cond(np.cov(x, rowvar=False, bias=True) + 1e-12 * np.eye(30))


def test_nonneg_mean_variance():
    cov = np.array([[1.0, 0.3], [0.3, 1.0]])
    mu = np.array([0.5, 0.4])
    assert np.allclose(nonneg_mean_variance(cov, mu), np.linalg.solve(cov, mu), atol=1e-6)
    assert np.allclose(nonneg_mean_variance(cov, np.array([-0.1, -0.2])), 0)
    w = nonneg_mean_variance(cov, np.array([0.5, -0.4]))
    assert w[1] == 0 and w[0] > 0


def test_risk_neutralisation_removes_beta_and_pca_factors():
    from alpha_research.risk import exposures, neutralise_frame

    rng = np.random.default_rng(5)
    days = pd.bdate_range("2019-01-01", periods=300)
    tickers = [f"S{i}" for i in range(120)]
    sector = np.repeat([0, 1, 2], 40)
    sector_ret = rng.normal(0, 0.01, (300, 3))
    ret = pd.DataFrame(sector_ret[:, sector] + rng.normal(0, 0.01, (300, 120)), index=days, columns=tickers)
    beta = pd.DataFrame(rng.uniform(0.5, 1.5, (300, 120)), index=days, columns=tickers)
    date = days[-1]
    expo = exposures(ret, beta, pd.DatetimeIndex([date]), n_pca=3)
    E = expo[date]
    assert list(E.columns) == ["const", "beta", "pc1", "pc2", "pc3"]
    # a signal that is mostly "sector 0 vs the rest" plus noise
    y = pd.DataFrame([(sector == 0) * 1.0 + rng.normal(0, 0.1, 120)], index=[date], columns=tickers)
    res = neutralise_frame(y, expo).loc[date]
    for col in E.columns:
        assert abs(np.dot(res.to_numpy(), E[col].to_numpy())) < 1e-8
    assert res.std() < 0.5 * y.loc[date].std()          # the sector bet is mostly removed


def test_risk_exposures_use_only_past_returns():
    from alpha_research.risk import exposures

    rng = np.random.default_rng(6)
    days = pd.bdate_range("2019-01-01", periods=320)
    tickers = [f"S{i}" for i in range(80)]
    ret = pd.DataFrame(rng.normal(0, 0.01, (320, 80)), index=days, columns=tickers)
    beta = pd.DataFrame(1.0, index=days, columns=tickers)
    date = days[280]
    e1 = exposures(ret, beta, pd.DatetimeIndex([date]), n_pca=3)[date]
    ret2 = ret.copy()
    ret2.iloc[281:] = rng.normal(0, 0.05, (39, 80))           # rewrite the future
    e2 = exposures(ret2, beta, pd.DatetimeIndex([date]), n_pca=3)[date]
    pd.testing.assert_frame_equal(e1, e2)
