import pandas as pd

from alpha_research.data import universe


def _write(tmp_path, rows):
    path = tmp_path / "m.csv"
    pd.DataFrame(rows, columns=["date", "tickers"]).to_csv(path, index=False)
    return universe.load_snapshots(path)


def test_normalize_ticker():
    assert universe.normalize_ticker(" brk.b ") == "BRK-B"


def test_membership_uses_last_snapshot_on_or_before_date(tmp_path):
    snaps = _write(tmp_path, [("2020-01-02", "AAA,BBB"), ("2020-01-06", "AAA,CCC")])
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"])
    m = universe.membership_matrix(snaps, dates)

    assert not m.loc["2020-01-01"].any()                       # before first snapshot
    assert m.loc["2020-01-03", ["AAA", "BBB"]].all()           # carried forward
    assert not m.loc["2020-01-03", "CCC"]                      # not yet added: no look-ahead
    assert m.loc["2020-01-06", "CCC"] and not m.loc["2020-01-06", "BBB"]  # change effective that day


def test_tickers_between_includes_snapshot_in_effect_at_start(tmp_path):
    snaps = _write(tmp_path, [("2019-06-01", "OLD,AAA"), ("2020-03-01", "AAA,NEW"), ("2021-01-01", "AAA,LATE")])
    assert universe.tickers_between(snaps, "2020-01-01", "2020-12-31") == ["AAA", "NEW", "OLD"]
