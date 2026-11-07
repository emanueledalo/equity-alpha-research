import numpy as np
import pandas as pd
import pytest

from alpha_research.config import DataConfig
from alpha_research.data.clean import clean_prices

DATES = pd.bdate_range("2021-01-04", periods=12)


def _wide(adj_close, volume=None):
    adj = pd.DataFrame({"X": adj_close}, index=DATES[: len(adj_close)])
    vol = pd.DataFrame({"X": volume if volume is not None else [1e6] * len(adj_close)}, index=adj.index)
    return {"open": adj, "high": adj, "low": adj, "close": adj, "adj_close": adj, "volume": vol}


def test_returns_do_not_span_gaps():
    prices = [100, 101, np.nan, 103, 104] + [104] * 7
    w, _ = clean_prices(_wide(prices), DATES)
    ret = w["ret"]["X"]
    assert ret.iloc[1] == pytest.approx(0.01)
    assert np.isnan(ret.iloc[2]) and np.isnan(ret.iloc[3])   # no return across the missing day
    assert ret.iloc[4] == pytest.approx(104 / 103 - 1)


def test_reverted_spike_is_removed():
    prices = [100, 100, 100, 300, 101, 102] + [102] * 6
    w, issues = clean_prices(_wide(prices), DATES)
    assert np.isnan(w["adj_close"]["X"].iloc[3])
    assert (issues["issue"] == "spike_reverted").sum() == 1


def test_real_jump_that_persists_is_kept():
    prices = [100, 100, 100, 160, 161, 162] + [162] * 6       # e.g. takeover announcement
    w, issues = clean_prices(_wide(prices), DATES)
    assert w["adj_close"]["X"].iloc[3] == 160
    assert "spike_reverted" not in set(issues["issue"])


def test_stale_run_with_zero_volume_is_removed():
    prices = [100, 101] + [101] * 6 + [102, 103, 104, 105]
    volume = [1e6, 1e6] + [0] * 6 + [1e6] * 4
    w, issues = clean_prices(_wide(prices, volume), DATES, DataConfig(stale_run=5))
    adj = w["adj_close"]["X"]
    assert adj.iloc[1] == 101                       # last real trade kept
    assert adj.iloc[2:8].isna().all()               # 6 stale days removed
    assert (issues["issue"] == "stale_price").sum() == 6


def test_nonpositive_prices_are_invalid():
    prices = [100, -1, 100] + [100] * 9
    w, issues = clean_prices(_wide(prices), DATES)
    assert np.isnan(w["adj_close"]["X"].iloc[1])


def test_identity_check_rejects_reused_ticker():
    from alpha_research.data.clean import check_series

    member_dates = DATES[:10]
    real = pd.DataFrame({"close": 60.0, "volume": 2e6, "adj_close": 60.0}, index=DATES)
    penny = pd.DataFrame({"close": 1.9, "volume": 5e3, "adj_close": 1.9}, index=DATES)  # same ticker, other company

    ok, coverage, _ = check_series(real, member_dates, DataConfig())
    assert ok and coverage == 1.0
    ok, _, reason = check_series(penny, member_dates, DataConfig())
    assert not ok and "wrong company" in reason


def test_trim_after_membership_keeps_history_and_short_tail():
    from alpha_research.data.clean import trim_after_membership

    idx = pd.bdate_range("2020-01-01", periods=200)
    px = pd.DataFrame({"OLD": 10.0, "NEVER": 5.0}, index=idx)
    members = pd.DataFrame({"OLD": False, "NEVER": False}, index=idx)
    members.iloc[50:100, 0] = True                     # OLD is a member on rows 50..99
    out = trim_after_membership({"adj_close": px}, members, keep_days=10)["adj_close"]
    assert out["OLD"].iloc[:110].notna().all()         # history before + membership + 10-day tail
    assert out["OLD"].iloc[110:].isna().all()          # later data (possibly another company) dropped
    assert out["NEVER"].notna().all()                  # never a member: untouched
