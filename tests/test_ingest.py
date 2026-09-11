import pandas as pd
import pytest

from alpha_lab.data.ingest import (
    archive_url, month_range, parse_funding_csv, parse_kline_csv,
)

KLINE_WITH_HEADER = (
    b"open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    b"taker_buy_volume,taker_buy_quote_volume,ignore\n"
    b"1704067200000,42314.00,42335.80,42289.60,42331.90,289.641,"
    b"1704067259999,12256155.25625,3310,175.211,7414459.86355,0\n"
    b"1704067260000,42331.90,42353.10,42331.80,42350.40,202.444,"
    b"1704067319999,8572240.95470,1885,154.353,6535804.80720,0\n"
)

# Старые файлы архива не содержат строки заголовка
KLINE_WITHOUT_HEADER = b"\n".join(KLINE_WITH_HEADER.split(b"\n")[1:])


def test_parse_kline_with_header():
    df = parse_kline_csv(KLINE_WITH_HEADER)

    assert len(df) == 2
    assert df["close"].iloc[0] == pytest.approx(42331.90)
    assert df["trades"].iloc[0] == 3310
    assert str(df["ts"].dt.tz) == "UTC"


def test_parse_kline_without_header():
    df = parse_kline_csv(KLINE_WITHOUT_HEADER)

    assert len(df) == 2
    assert df["open"].iloc[1] == pytest.approx(42331.90)


def test_parse_kline_timestamps_are_minutes_apart():
    df = parse_kline_csv(KLINE_WITH_HEADER)

    assert (df["ts"].diff().dropna() == pd.Timedelta(minutes=1)).all()


FUNDING_CSV = (
    b"calc_time,funding_interval_hours,last_funding_rate\n"
    b"1704067200000,8,0.00037409\n"
    b"1704096000000,8,0.00027213\n"
)


def test_parse_funding():
    df = parse_funding_csv(FUNDING_CSV)

    assert list(df.columns) == ["ts", "rate", "interval_hours"]
    assert df["rate"].iloc[0] == pytest.approx(0.00037409)
    assert df["interval_hours"].iloc[0] == 8


def test_archive_url_klines():
    url = archive_url("futures-um", "klines", "BTCUSDT", "1m", "2024-01")

    assert url == ("https://data.binance.vision/data/futures/um/monthly/"
                   "klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip")


def test_archive_url_funding():
    url = archive_url("futures-um", "fundingRate", "BTCUSDT", None, "2024-01")

    assert url == ("https://data.binance.vision/data/futures/um/monthly/"
                   "fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip")


def test_month_range():
    months = month_range("2024-01-15", "2024-04-02")

    assert months == ["2024-01", "2024-02", "2024-03", "2024-04"]


# Бар с пустой ценой normalize_bars отбрасывает. Это не должно происходить молча:
# число отброшенных строк обязан вернуть парсер и донести ingest до вызывающего.
KLINE_WITH_NAN_PRICE = KLINE_WITH_HEADER + (
    b"1704067320000,,42353.10,42331.80,42350.40,150.0,"
    b"1704067379999,6349710.0,1000,80.0,3386512.0,0\n"
)


def test_parse_kline_reports_dropped_nan_price_rows():
    stats = {}
    df = parse_kline_csv(KLINE_WITH_NAN_PRICE, stats=stats)

    assert stats["dropped_rows"] == 1
    assert len(df) == 2
    assert df["close"].iloc[-1] == pytest.approx(42350.40)


def test_ingest_reports_dropped_rows(tmp_path, monkeypatch):
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_download", lambda url: KLINE_WITH_NAN_PRICE)
    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)

    assert len(results) == 1
    result = results[0]
    assert result.ok
    assert result.rows == 2
    assert result.dropped_rows == 1
    assert "отброшено" in result.quality
