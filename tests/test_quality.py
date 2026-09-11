import pandas as pd

from alpha_lab.data.quality import check_bars
from alpha_lab.data.schema import normalize_bars


def _bars(ts):
    return normalize_bars(pd.DataFrame({
        "ts": pd.to_datetime(ts, utc=True),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 10.0, "quote_volume": 1000.0, "trades": 5,
        "taker_buy_volume": 6.0,
    }))


def test_clean_series_has_no_issues():
    df = _bars(pd.date_range("2024-01-01", periods=100, freq="1min"))

    rep = check_bars(df, "1m")

    assert rep.is_clean
    assert rep.total_rows == 100
    assert rep.gaps == 0


def test_detects_gap():
    ts = list(pd.date_range("2024-01-01", periods=50, freq="1min"))
    ts += list(pd.date_range("2024-01-01 02:00", periods=50, freq="1min"))

    rep = check_bars(_bars(ts), "1m")

    assert not rep.is_clean
    assert rep.gaps == 1


def test_detects_zero_volume():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[3, "volume"] = 0.0

    rep = check_bars(df, "1m")

    assert rep.zero_volume == 1


def test_detects_impossible_ohlc():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[2, "high"] = 50.0   # high ниже low

    rep = check_bars(df, "1m")

    assert rep.anomalous == 1


def test_duplicates_removed_by_normalize():
    df = _bars(pd.date_range("2024-01-01", periods=5, freq="1min"))
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

    rep = check_bars(normalize_bars(df), "1m")

    assert rep.duplicates == 0
    assert rep.total_rows == 5


def test_summary_mentions_problems():
    ts = list(pd.date_range("2024-01-01", periods=10, freq="1min"))
    ts += list(pd.date_range("2024-01-01 05:00", periods=10, freq="1min"))

    text = check_bars(_bars(ts), "1m").summary()

    assert "пропуск" in text.lower()
