import pandas as pd

from alpha_lab.data.query import (
    align_funding_to_bars, data_version, load_bars, load_funding,
)
from alpha_lab.data.store import write_bars, write_funding
from alpha_lab.data.schema import normalize_bars


# pandas 3.0 больше не принимает "m" как минуту (теперь это месяц) —
# для date_range переводим в "min"; в хранилище freq остаётся "1m".
_DATE_RANGE_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min"}


def _make(root, n=600, freq="1m"):
    ts = pd.date_range("2024-01-01", periods=n,
                       freq=_DATE_RANGE_FREQ.get(freq, freq), tz="UTC")
    close = pd.Series(range(n), dtype="float64") + 100.0
    df = normalize_bars(pd.DataFrame({
        "ts": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 10.0, "quote_volume": 1000.0,
        "trades": 5, "taker_buy_volume": 6.0,
    }))
    write_bars(df, root, "BTCUSDT", freq)


def test_load_bars_reads_all(tmp_path):
    _make(tmp_path, n=120)

    df = load_bars(tmp_path, "BTCUSDT", "1m")

    assert len(df) == 120


def test_load_bars_date_only_end_includes_whole_day(tmp_path):
    """end="2024-01-01" — это дата, а не полночь: день обязан входить целиком.

    Раньше фильтр ts <= end отсекал всё после 00:00, и последние 23 часа
    периода молча выпадали из прогона.
    """
    _make(tmp_path, n=2880)   # 2024-01-01 00:00 .. 2024-01-02 23:59

    out = load_bars(tmp_path, "BTCUSDT", "1m", "2024-01-01", "2024-01-01")

    assert len(out) == 1440
    assert out["ts"].iloc[-1] == pd.Timestamp("2024-01-01 23:59", tz="UTC")


def test_load_bars_date_only_end_resamples_full_day(tmp_path):
    _make(tmp_path, n=2880)

    out = load_bars(tmp_path, "BTCUSDT", "1m", "2024-01-01", "2024-01-01",
                    resample="1h")

    assert len(out) == 24
    assert out["ts"].iloc[-1] == pd.Timestamp("2024-01-01 23:00", tz="UTC")


def test_load_bars_explicit_timestamp_end_is_exact(tmp_path):
    """Явный момент времени — точная граница, даже если это полночь.

    Проверка на «полночь по значению, а не по формату»: иначе end=
    "2024-01-01T00:00" неожиданно включал бы весь день.
    """
    _make(tmp_path, n=2880)

    exact = load_bars(tmp_path, "BTCUSDT", "1m", "2024-01-01",
                      "2024-01-01T00:30")
    assert len(exact) == 31
    assert exact["ts"].iloc[-1] == pd.Timestamp("2024-01-01 00:30", tz="UTC")

    midnight = load_bars(tmp_path, "BTCUSDT", "1m", "2024-01-01",
                         "2024-01-01T00:00")
    assert len(midnight) == 1
    assert midnight["ts"].iloc[-1] == pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_load_bars_resamples_to_hourly(tmp_path):
    _make(tmp_path, n=600)   # 10 часов по минутам

    df = load_bars(tmp_path, "BTCUSDT", "1m", resample="1h")

    assert len(df) == 10
    assert df["close"].iloc[0] == 159.0     # последняя минута первого часа
    assert df["open"].iloc[0] == 100.0      # первая минута первого часа
    assert df["volume"].iloc[0] == 600.0    # 60 минут × 10


def test_data_version_is_stable_and_changes_with_data(tmp_path):
    _make(tmp_path, n=60)
    v1 = data_version(tmp_path)
    v2 = data_version(tmp_path)

    _make(tmp_path, n=120)
    v3 = data_version(tmp_path)

    assert v1 == v2
    assert v1 != v3


def test_align_funding_places_rate_at_funding_timestamps(tmp_path):
    _make(tmp_path, n=600)
    bars = load_bars(tmp_path, "BTCUSDT", "1m")
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True),
        "rate": [0.0001, 0.0002],
        "interval_hours": [8, 8],
    })

    series = align_funding_to_bars(bars, funding)

    assert len(series) == len(bars)
    assert series.iloc[0] == 0.0001
    assert series.iloc[480] == 0.0002
    assert series.sum() == 0.0001 + 0.0002
