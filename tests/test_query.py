import numpy as np
import pandas as pd
import pytest

from alpha_lab.data.query import (
    align_funding_to_bars, data_version, load_bars, load_funding,
    matched_funding_events,
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


def test_align_funding_sums_events_into_containing_daily_bar():
    """Решающий тест 1d: учтены ВСЕ ставки, а не только полуночные.

    Ставки Binance (00:00, 08:00, 16:00 UTC) на дневном баре попадают в один
    бар; точный матч выживлял только 00:00 и терял 2/3 выплат — в сторону,
    которая льстит стратегии.
    """
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")})
    funding = pd.DataFrame({
        "ts": pd.to_datetime([
            "2024-01-01 00:00", "2024-01-01 08:00", "2024-01-01 16:00",
            "2024-01-02 00:00", "2024-01-02 08:00", "2024-01-02 16:00",
        ], utc=True),
        "rate": [1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4],
    })

    series = align_funding_to_bars(bars, funding)

    assert series.sum() == pytest.approx(float(funding["rate"].sum()))
    assert series.iloc[0] == pytest.approx(6e-4)
    assert series.iloc[1] == pytest.approx(15e-4)
    assert series.iloc[2] == 0.0


def test_align_funding_boundary_event_belongs_to_opening_bar():
    """Ставка ровно на границе — бару, который ею открывается.

    Соглашение то же, что у точного матча на 1h, и совпадает с движком:
    funding бара t начисляется на held[t] — позицию, удерживаемую в баре
    [t, t+step), поэтому выплата в момент t оплачивается этим баром.
    """
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")})
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-02 00:00"], utc=True),
        "rate": [0.001],
    })

    series = align_funding_to_bars(bars, funding, timeframe="1d")

    assert series.tolist() == [0.0, 0.001, 0.0]


def test_align_funding_hourly_places_each_event_in_its_own_bar():
    """1h: каждое событие — в свой бар, суммы не меняются.

    Регрессия на W2: обобщение на «охватывающий бар» не должно сдвигать
    часовое выравнивание — иначе BTC-прогон изменится там, где funding и так
    учитывался полностью.
    """
    ts = pd.date_range("2024-01-01", periods=48, freq="1h", tz="UTC")
    bars = pd.DataFrame({"ts": ts})
    funding = pd.DataFrame({
        "ts": ts[::8],                      # 00:00, 08:00, 16:00 каждые сутки
        "rate": np.arange(1, 7) * 1e-4,
    })

    series = align_funding_to_bars(bars, funding, timeframe="1h")

    assert series.sum() == pytest.approx(float(funding["rate"].sum()))
    for i, rate in zip(range(0, 48, 8), funding["rate"]):
        assert series.iloc[i] == pytest.approx(rate)
    assert (series.drop(series.index[::8]) == 0.0).all()


def test_align_funding_explicit_timeframe_drops_events_in_missing_bars():
    """Явный timeframe задаёт интервал бара: событие из пропущенного бара
    не приклеивается к соседнему.

    1d-ряд с дырой (Jan 1 и Jan 4): ставка Jan 2 08:00 лежит в несуществующем
    баре Jan 2 и не принадлежит ни одному бару. Без явного timeframe шаг
    вывелся бы из медианы (3 дня) и молча приписал бы её бару Jan 1.
    """
    bars = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01", "2024-01-04"], utc=True)})
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 08:00", "2024-01-02 08:00",
                              "2024-01-04 00:00"], utc=True),
        "rate": [1e-4, 2e-4, 3e-4],
    })

    series = align_funding_to_bars(bars, funding, timeframe="1d")

    assert series.tolist() == pytest.approx([1e-4, 3e-4])


def test_align_funding_unknown_timeframe_raises():
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")})
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 08:00"], utc=True),
        "rate": [1e-4],
    })

    with pytest.raises(ValueError, match="Неизвестный таймфрейм"):
        align_funding_to_bars(bars, funding, timeframe="2h")


def test_align_funding_ignores_events_after_last_bar():
    """Событие за последним баром не приклеивается к последнему бару."""
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=2, freq="1D", tz="UTC")})
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 08:00", "2024-01-05 08:00"], utc=True),
        "rate": [1e-4, 9e-4],
    })

    series = align_funding_to_bars(bars, funding, timeframe="1d")

    assert series.tolist() == pytest.approx([1e-4, 0.0])


def test_matched_funding_events_counts_contained_including_zero_rates():
    """Счётчик привязки: сколько событий попало в существующие бары.

    Нулевая ставка — тоже учтённое событие (её нельзя отличить от «данных
    нет» суммой), поэтому счётчик считается по попаданию, а не по значению.
    """
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=3, freq="1D", tz="UTC")})
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00",
                              "2024-01-02 16:00", "2024-01-09 00:00"], utc=True),
        "rate": [0.0, 1e-4, 2e-4, 3e-4],
    })

    assert matched_funding_events(bars, funding, timeframe="1d") == 3
