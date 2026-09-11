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


def test_read_bars_date_only_end_includes_full_day(tmp_path):
    """Та же семантика, что у query.load_bars: дата включает весь конечный день."""
    write_bars(normalize_bars(_bars(2880)), tmp_path, "BTCUSDT", "1m")

    whole_day = read_bars(tmp_path, "BTCUSDT", "1m",
                          start="2024-01-01", end="2024-01-01")
    assert len(whole_day) == 1440
    assert whole_day["ts"].iloc[-1] == pd.Timestamp("2024-01-01 23:59", tz="UTC")

    exact = read_bars(tmp_path, "BTCUSDT", "1m", end="2024-01-01T00:00")
    assert len(exact) == 1


def test_read_bars_missing_symbol_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Нет данных"):
        read_bars(tmp_path, "NOPEUSDT", "1m")


def test_read_bars_ignores_foreign_parquet_in_same_dir(tmp_path):
    """Глоб обязан читать только файлы своего символа и таймфрейма.

    Соседний parquet (другой символ или другой таймфрейм) при глобе *.parquet
    молча подмешивался бы в ряд — цены чужого актива выглядели бы как дыры
    и скачки своего.
    """
    write_bars(normalize_bars(_bars(10)), tmp_path, "BTCUSDT", "1m")
    other_symbol = normalize_bars(_bars(50, "2024-02-01"))
    other_symbol.to_parquet(
        tmp_path / "bars" / "BTCUSDT" / "1m" / "ETHUSDT-1m-2024-02.parquet",
        index=False)
    other_freq = normalize_bars(_bars(7, "2024-03-01"))
    other_freq.to_parquet(
        tmp_path / "bars" / "BTCUSDT" / "1m" / "BTCUSDT-4h-2024-03.parquet",
        index=False)

    back = read_bars(tmp_path, "BTCUSDT", "1m")

    assert len(back) == 10


def test_funding_roundtrip(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    df = pd.DataFrame({"ts": ts, "rate": [0.0001, -0.0002, 0.0003],
                       "interval_hours": [8, 8, 8]})

    write_funding(df, tmp_path, "BTCUSDT")
    back = read_funding(tmp_path, "BTCUSDT")

    pd.testing.assert_frame_equal(back, df)
