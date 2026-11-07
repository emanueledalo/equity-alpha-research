import numpy as np
import pandas as pd
import pytest

from alpha_research.signals import preprocess, targets
from alpha_research.signals.library import SIGNALS, MarketData, beta, idio_vol, mom_12_1

N_DAYS, N_STOCKS = 400, 30
DATES = pd.bdate_range("2015-01-01", periods=N_DAYS)


def _data(seed: int = 0, betas=None, noise: float = 0.01) -> MarketData:
    rng = np.random.default_rng(seed)
    mkt = pd.Series(rng.normal(0.0004, 0.01, N_DAYS), index=DATES)
    betas = np.ones(N_STOCKS) if betas is None else np.asarray(betas)
    eps = rng.normal(0, noise, (N_DAYS, N_STOCKS))
    ret = pd.DataFrame(mkt.to_numpy()[:, None] * betas + eps, index=DATES,
                       columns=[f"S{i}" for i in range(N_STOCKS)])
    ret.iloc[0] = np.nan
    adj = 100 * (1 + ret.fillna(0)).cumprod()
    dv = pd.DataFrame(rng.uniform(1e7, 1e8, ret.shape), index=DATES, columns=ret.columns)
    return MarketData(ret=ret, adj_close=adj, dollar_volume=dv, mkt_ret=mkt)


def test_no_lookahead_in_any_signal():
    """Changing prices after day T must not change any signal on or before T."""
    base = _data()
    cut = 300
    ret2 = base.ret.copy()
    ret2.iloc[cut + 1:] = ret2.iloc[cut + 1:] * -3 + 0.05          # rewrite the future
    adj2 = base.adj_close.copy()
    adj2.iloc[cut + 1:] = adj2.iloc[cut + 1:] * 7
    dv2 = base.dollar_volume.copy()
    dv2.iloc[cut + 1:] = 1.0
    mkt2 = base.mkt_ret.copy()
    mkt2.iloc[cut + 1:] = 0.2
    future = MarketData(ret=ret2, adj_close=adj2, dollar_volume=dv2, mkt_ret=mkt2)

    for s in SIGNALS:
        a = s.func(base).iloc[: cut + 1]
        b = s.func(future).iloc[: cut + 1]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, obj=s.name)


def test_momentum_skips_last_month():
    d = _data()
    t = 350
    lr = np.log1p(d.ret["S0"])
    expected = lr.iloc[t - 251: t - 20].sum()     # days t-251 .. t-21 inclusive = 231 days
    assert mom_12_1(d)["S0"].iloc[t] == pytest.approx(expected)


def test_beta_and_idio_vol_recover_true_values():
    true_betas = np.linspace(0.5, 1.8, N_STOCKS)
    d = _data(betas=true_betas, noise=0.005)
    est = beta(d).iloc[-1].to_numpy()
    assert np.allclose(est, true_betas, atol=0.1)
    assert np.allclose(idio_vol(d).iloc[-1].to_numpy(), 0.005, atol=0.0015)


def test_rolling_window_needs_enough_observations():
    d = _data()
    ret = d.ret.copy()
    ret.iloc[100:300, 0] = np.nan                  # long data gap for one stock
    d2 = MarketData(ret=ret, adj_close=d.adj_close, dollar_volume=d.dollar_volume, mkt_ret=d.mkt_ret)
    assert np.isnan(mom_12_1(d2)["S0"].iloc[320])
    assert not np.isnan(mom_12_1(d2)["S1"].iloc[320])


def test_process_zscores_universe_only_and_orients_sign():
    raw = pd.DataFrame(np.arange(60, dtype=float).reshape(2, 30), index=DATES[:2],
                       columns=[f"S{i}" for i in range(30)])
    mask = pd.DataFrame(True, index=raw.index, columns=raw.columns)
    mask.iloc[:, :5] = False                       # five stocks not in the index
    z = preprocess.process(raw, mask, sign=-1)
    assert z.iloc[:, :5].isna().all().all()
    row = z.iloc[0].dropna()
    assert row.mean() == pytest.approx(0, abs=1e-12)
    assert row.std() == pytest.approx(1)
    assert row.iloc[0] > row.iloc[-1]              # sign = -1 flips the ranking


def test_winsorize_clips_outliers():
    raw = pd.DataFrame([list(range(100)) + [1e6]], dtype=float)
    w = preprocess.winsorize(raw)
    assert w.iloc[0].max() < 1e3


def test_forward_returns_enter_at_next_close():
    adj = pd.DataFrame({"A": [100.0, 110.0, 121.0, 133.1]}, index=DATES[:4])
    f1 = targets.forward_returns(adj, horizon=1)
    assert f1["A"].iloc[0] == pytest.approx(121 / 110 - 1)    # signal on day 0, enter day 1, exit day 2
    assert f1["A"].iloc[2:].isna().all()
