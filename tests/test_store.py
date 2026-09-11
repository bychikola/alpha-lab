import pandas as pd
import pytest

from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import read_bars, read_funding, write_bars, write_funding


def _bars(n=100, start="2024-01-01"):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100.0 + i * 0.1 for i in range(n)],
        "volume": 10.0,
        "quote_volume": 1000.0,
        "trades": 5,
        "taker_buy_volume": 6.0,
    })


def test_normalize_bars_sorts_and_dedups():
    df = _bars(5)
    shuffled = pd.concat([df.iloc[[3, 1]], df.iloc[[3, 1]]], ignore_index=True)

    out = normalize_bars(shuffled)

    assert out["ts"].is_monotonic_increasing
    assert len(out) == 2
    assert out["ts"].dt.tz is not None


def test_normalize_bars_rejects_missing_columns():
    df = _bars(5).drop(columns=["volume"])

    with pytest.raises(ValueError, match="volume"):
        normalize_bars(df)


def test_write_read_roundtrip(tmp_path):
    df = normalize_bars(_bars(100))

    paths = write_bars(df, tmp_path, "BTCUSDT", "1m")
    back = read_bars(tmp_path, "BTCUSDT", "1m")

    assert len(paths) == 1
    pd.testing.assert_frame_equal(back, df)


def test_read_bars_filters_by_date(tmp_path):
    write_bars(normalize_bars(_bars(100, "2024-01-01")), tmp_path, "BTCUSDT", "1m")

    out = read_bars(tmp_path, "BTCUSDT", "1m",
                    start="2024-01-01T00:30", end="2024-01-01T00:39")

    assert len(out) == 10


def test_read_bars_missing_symbol_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Нет данных"):
        read_bars(tmp_path, "NOPEUSDT", "1m")


def test_funding_roundtrip(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    df = pd.DataFrame({"ts": ts, "rate": [0.0001, -0.0002, 0.0003],
                       "interval_hours": [8, 8, 8]})

    write_funding(df, tmp_path, "BTCUSDT")
    back = read_funding(tmp_path, "BTCUSDT")

    pd.testing.assert_frame_equal(back, df)
