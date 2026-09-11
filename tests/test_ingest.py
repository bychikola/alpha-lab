import io
import zipfile

import pandas as pd
import pytest
import requests

from alpha_lab.data.ingest import (
    archive_url, month_range, parse_funding_csv, parse_kline_csv,
)


def _zip_payload(content: bytes, name: str = "data.csv") -> bytes:
    """Zip в памяти: архив Binance отдаёт zip, и _download проверяет его целость."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(name, content)
    return buf.getvalue()


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

# То, что реально отдаёт архив: zip с CSV внутри. _download обязан проверить
# целость zip, поэтому тела HTTP-ответов в тестах ниже — zip, а не голый CSV.
ZIP_CSV = _zip_payload(KLINE_WITH_HEADER)


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

    monkeypatch.setattr(ingest_module, "_download",
                        lambda url, **kwargs: KLINE_WITH_NAN_PRICE)
    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)

    assert len(results) == 1
    result = results[0]
    assert result.ok
    assert result.rows == 2
    assert result.dropped_rows == 1
    assert "отброшено" in result.quality


# Строка с неразбираемой меткой времени получает NaT. write_bars группирует по
# месяцу и молча выбрасывает NaT-ключи, поэтому до диска такой бар не доходит.
# Парсер обязан сразу посчитать его отброшенным, а не отчитаться как о загруженном.
KLINE_WITH_BAD_TIMESTAMP = KLINE_WITH_HEADER + (
    b"not-a-timestamp,42354.00,42360.00,42350.00,42355.00,100.0,"
    b"1704067379999,4235500.0,900,50.0,2117750.0,0\n"
)


def test_parse_kline_reports_dropped_nat_timestamp_rows():
    stats = {}
    df = parse_kline_csv(KLINE_WITH_BAD_TIMESTAMP, stats=stats)

    assert stats["dropped_rows"] == 1
    assert len(df) == 2
    assert df["ts"].notna().all()


# Сбой загрузки funding (таймаут, 5xx) не должен вылетать из ingest_symbol:
# klines уже загружены, и отчёт по ним обязан уцелеть.
def test_funding_download_error_is_isolated(tmp_path, monkeypatch):
    import alpha_lab.data.ingest as ingest_module

    def fake_download(url, **kwargs):
        if "fundingRate" in url:
            raise requests.RequestException("funding timeout")
        return KLINE_WITH_HEADER

    monkeypatch.setattr(ingest_module, "_download", fake_download)
    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=True)

    assert len(results) == 2
    assert results[0].ok
    funding = results[1]
    assert funding.kind == "funding"
    assert not funding.ok
    assert "funding timeout" in funding.error


# Один сбойный символ не должен уносить с собой весь прогон по вселенной:
# цикл обязан дойти до последнего символа и вернуть отчёт по каждому.
def test_ingest_universe_isolates_failing_symbol(tmp_path, monkeypatch):
    import alpha_lab.data.ingest as ingest_module

    class Universe:
        symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
        start = "2024-01-01"
        end = "2024-01-31"
        market = "futures-um"

    def fake_ingest(symbol, freq, start, end, root=None, market="futures-um",
                    **kwargs):
        if symbol == "BBBUSDT":
            raise RuntimeError("битый символ")
        return [ingest_module.IngestResult(symbol, "klines", 1, 10, "OK")]

    monkeypatch.setattr(ingest_module, "ingest_symbol", fake_ingest)
    results = ingest_module.ingest_universe(Universe(), "1m", tmp_path)

    assert [r.symbol for r in results] == ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    assert results[0].ok and results[2].ok
    assert not results[1].ok
    assert "битый символ" in results[1].error


# 404 на месячном файле — норма (ранние месяцы), но оператор должен видеть,
# сколько месяцев реально отсутствовало, а не молчаливое «ок».
def test_ingest_reports_missing_months(tmp_path, monkeypatch):
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(
        ingest_module, "_download",
        lambda url, **kwargs: None if "2024-02" in url else KLINE_WITH_HEADER)
    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-02-29", tmp_path, with_funding=False)

    assert len(results) == 1
    assert results[0].ok
    assert "пропущено месяцев: 1" in results[0].quality


def _kline_month(period: str, day: int | None = None) -> bytes:
    """Месячный CSV: одна минута в конце месяца (или в указанный день).

    Файлы архива не фильтруются по start/end при ingest — месячный файл
    покрывает весь месяц, поэтому «полное покрытие» обязано кончаться в
    последнюю минуту месяца.
    """
    end = pd.Period(period, freq="M").end_time
    if day is not None:
        end = end.replace(day=day, hour=12, minute=0, second=0, microsecond=0)
    ms = int(end.tz_localize("UTC").timestamp() * 1000)
    header = (b"open_time,open,high,low,close,volume,close_time,quote_volume,"
              b"count,taker_buy_volume,taker_buy_quote_volume,ignore\n")
    row = f"{ms},100,101,99,100,10,{ms + 59999},1000,5,6,600,0\n".encode()
    return header + row


def _download_by_month(available: dict[str, bytes]):
    """Подменяет _download: отдаёт файл только для перечисленных месяцев."""
    import re

    def download(url: str, **kwargs):
        if "fundingRate" in url:
            return None
        match = re.search(r"(\d{4}-\d{2})\.zip", url)
        if match is None:
            return None
        return available.get(match.group(1))

    return download


def test_ingest_marks_tail_truncation_as_delisted(tmp_path, monkeypatch):
    """Хвост из отсутствующих месяцев — делистинг/обрыв истории, а не «ок».

    Пара, у которой данные кончаются раньше периода, обязана быть помечена:
    молчаливый пропуск хвоста — это и есть ошибка выживаемости из spec 5.
    """
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_download",
                        _download_by_month({"2024-01": _kline_month("2024-01")}))
    results = ingest_module.ingest_symbol(
        "OLDUSDT", "1m", "2024-01-01", "2024-03-31", tmp_path,
        with_funding=False)

    assert len(results) == 1
    result = results[0]
    assert result.ok and result.delisted and not result.excluded
    assert result.last_bar_ts >= pd.Timestamp("2024-01-31 23:59", tz="UTC")
    assert "делистинг" in result.quality


def test_ingest_marks_partial_final_month_as_delisted(tmp_path, monkeypatch):
    """Обрыв внутри последнего месяца виден по last_bar_ts, а не по 404."""
    import alpha_lab.data.ingest as ingest_module

    available = {period: _kline_month("2024-03", day=15)
                 for period in ("2024-01", "2024-02", "2024-03")}
    monkeypatch.setattr(ingest_module, "_download", _download_by_month(available))
    results = ingest_module.ingest_symbol(
        "OLDUSDT", "1m", "2024-01-01", "2024-03-31", tmp_path,
        with_funding=False)

    result = results[0]
    assert result.ok and result.delisted
    assert result.last_bar_ts == pd.Timestamp("2024-03-15 12:00", tz="UTC")


def test_ingest_full_coverage_is_not_delisted(tmp_path, monkeypatch):
    """Контроль: полное покрытие периода не считается делистингом."""
    import alpha_lab.data.ingest as ingest_module

    available = {period: _kline_month(period)
                 for period in ("2024-01", "2024-02", "2024-03")}
    monkeypatch.setattr(ingest_module, "_download", _download_by_month(available))
    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-03-31", tmp_path,
        with_funding=False)

    result = results[0]
    assert result.ok and not result.delisted and not result.excluded
    assert result.last_bar_ts is not None


def test_ingest_include_delisted_false_excludes_truncated_symbol(
        tmp_path, monkeypatch):
    """include_delisted=false — явный отказ от усечённой пары, и он виден.

    Пара не пишется в хранилище, но результат помечен excluded и объясняет,
    что отказ создаёт ошибку выживаемости: молчаливого исключения нет.
    """
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_download",
                        _download_by_month({"2024-01": _kline_month("2024-01")}))
    results = ingest_module.ingest_symbol(
        "OLDUSDT", "1m", "2024-01-01", "2024-03-31", tmp_path,
        with_funding=False, include_delisted=False)

    result = results[0]
    assert result.excluded and result.delisted
    assert result.rows == 0 and result.files == 0
    assert "выживаем" in result.quality
    assert not (tmp_path / "bars" / "OLDUSDT").exists()


def test_ingest_universe_passes_include_delisted(tmp_path, monkeypatch):
    """Флаг юниверса обязан доходить до ingest, а не лежать мёртвым грузом."""
    import alpha_lab.data.ingest as ingest_module

    class Universe:
        symbols = ["OLDUSDT"]
        start = "2024-01-01"
        end = "2024-03-31"
        market = "futures-um"

        def __init__(self, include_delisted):
            self.include_delisted = include_delisted

    monkeypatch.setattr(ingest_module, "_download",
                        _download_by_month({"2024-01": _kline_month("2024-01")}))

    excluded = ingest_module.ingest_universe(
        Universe(False), "1m", tmp_path / "excluded")
    kept = ingest_module.ingest_universe(
        Universe(True), "1m", tmp_path / "kept")

    assert excluded[0].excluded and excluded[0].rows == 0
    assert kept[0].delisted and not kept[0].excluded and kept[0].rows > 0


class _FakeResponse:
    """Минимальный ответ requests: тесты не выходят в сеть."""

    def __init__(self, status_code: int = 200, content: bytes = b"",
                 headers: dict | None = None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} Client Error", response=self)

    def close(self):
        pass


def test_download_retries_transient_timeout(monkeypatch):
    """Read timed out — временный сбой: повтор, а не потеря символа."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        if len(calls) < 3:
            raise requests.Timeout("read timed out")
        return _FakeResponse(200, ZIP_CSV)

    sleeps = []
    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", sleeps.append)

    raw = ingest_module._download("http://example.test/x.zip",
                                  retries=3, base_delay=0.1)

    assert raw == ZIP_CSV
    assert len(calls) == 3
    # Пауза только между попытками: после успеха спать нечего.
    assert len(sleeps) == 2


def test_download_persistent_timeout_fails_after_bounded_attempts(
        tmp_path, monkeypatch):
    """Мёртвый эндпоинт обязан упасть за конечное число попыток, отдав ошибку."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        raise requests.Timeout("read timed out")

    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-03-31", tmp_path,
        with_funding=False, retries=2, retry_base_delay=0.1)

    assert len(results) == 1
    assert not results[0].ok
    assert "read timed out" in results[0].error
    # 1 исходная попытка + 2 повтора; дальше ошибка обязана всплыть наружу.
    assert len(calls) == 3


def test_download_retries_5xx(monkeypatch):
    """5xx — временный сбой сервера: повторяем."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse(503)
        return _FakeResponse(200, ZIP_CSV)

    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    raw = ingest_module._download("http://example.test/x.zip",
                                  retries=2, base_delay=0)

    assert raw == ZIP_CSV
    assert len(calls) == 2


def test_download_404_is_not_retried(monkeypatch):
    """404 — «месяца ещё нет»: норма, повтор тратил бы минуты на полном прогоне."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        return _FakeResponse(404)

    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep",
                        lambda seconds: pytest.fail("404 повторять нельзя"))

    assert ingest_module._download("http://example.test/gone.zip") is None
    assert len(calls) == 1


def test_download_429_honours_retry_after(monkeypatch):
    """429 с Retry-After: пауза берётся из заголовка, а не из догадки."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse(429, headers={"Retry-After": "2"})
        return _FakeResponse(200, ZIP_CSV)

    sleeps = []
    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", sleeps.append)

    raw = ingest_module._download("http://example.test/x.zip",
                                  retries=2, base_delay=0.1)

    assert raw == ZIP_CSV
    assert len(calls) == 2
    assert sleeps[0] == pytest.approx(2.0, abs=0.05)


def test_download_retry_after_is_bounded(monkeypatch):
    """Чужой Retry-After не должен вешать прогон на часы."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        if len(calls) == 1:
            return _FakeResponse(429, headers={"Retry-After": "100000"})
        return _FakeResponse(200, ZIP_CSV)

    sleeps = []
    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", sleeps.append)

    raw = ingest_module._download("http://example.test/x.zip",
                                  retries=1, base_delay=0.1)

    assert raw == ZIP_CSV
    assert 0 < sleeps[0] <= ingest_module.MAX_RETRY_AFTER


def test_download_retry_is_logged_to_stderr(monkeypatch, capsys):
    """Долгий прогон обязан быть диагностируемым: URL, попытка, ошибка."""
    import alpha_lab.data.ingest as ingest_module

    calls = []

    def fake_get(url, headers=None, timeout=None, stream=False):
        calls.append(url)
        if len(calls) == 1:
            raise requests.Timeout("read timed out")
        return _FakeResponse(200, ZIP_CSV)

    monkeypatch.setattr(ingest_module.requests, "get", fake_get)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    ingest_module._download("http://example.test/x.zip", retries=2, base_delay=0)

    err = capsys.readouterr().err
    assert "http://example.test/x.zip" in err
    assert "1/3" in err
    assert "Timeout" in err
    assert err.count("[retry]") == 1


def test_ingest_skips_months_already_in_lake(tmp_path, monkeypatch):
    """Повторный прогон не перекачивает уже лежащие месяцы."""
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_download",
                        _download_by_month({"2024-01": _kline_month("2024-01")}))
    first = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)
    assert first[0].ok and first[0].files == 1 and first[0].skipped == 0

    def forbid(url, **kwargs):
        raise AssertionError(f"месяц уже в хранилище, повторная загрузка: {url}")

    monkeypatch.setattr(ingest_module, "_download", forbid)
    second = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)

    result = second[0]
    assert result.ok
    assert result.files == 0 and result.skipped == 1
    assert result.rows == 1
    assert "уже в хранилище" in result.quality


def test_ingest_partial_resume_downloads_only_absent_month(tmp_path, monkeypatch):
    """Докачивается только отсутствующий месяц, а не весь период символа."""
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_download",
                        _download_by_month({"2024-01": _kline_month("2024-01")}))
    ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-02-29", tmp_path, with_funding=False)

    requested: list[str] = []
    full = _download_by_month({"2024-01": _kline_month("2024-01"),
                               "2024-02": _kline_month("2024-02")})

    def recording(url, **kwargs):
        import re

        requested.append(re.search(r"(\d{4}-\d{2})\.zip", url).group(1))
        return full(url)

    monkeypatch.setattr(ingest_module, "_download", recording)
    result = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-02-29", tmp_path, with_funding=False)[0]

    assert result.ok
    assert requested == ["2024-02"]
    assert result.skipped == 1 and result.files == 1
    assert result.rows == 2 and result.missing == 0


def test_ingest_force_redownloads_present_months(tmp_path, monkeypatch):
    """force=True — явный перекач поверх уже лежащего (смена схемы, порча)."""
    import alpha_lab.data.ingest as ingest_module

    available = {"2024-01": _kline_month("2024-01")}
    monkeypatch.setattr(ingest_module, "_download", _download_by_month(available))
    ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)

    calls = []

    def recording(url, **kwargs):
        calls.append(url)
        return _download_by_month(available)(url)

    monkeypatch.setattr(ingest_module, "_download", recording)
    result = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False,
        force=True)[0]

    assert result.ok
    assert len(calls) == 1
    assert result.files == 1 and result.skipped == 0


@pytest.mark.parametrize("damaged", [b"", b"ne-tot-parquet\n\x00\x01"])
def test_ingest_does_not_trust_damaged_month_file(tmp_path, monkeypatch, damaged):
    """Существующий файл != дописанный: нулевой/битый перекачивается.

    write_bars пишет parquet прямо по конечному пути, поэтому обрыв процесса
    оставляет усечённый файл; считать его «месяц уже загружен» нельзя.
    """
    import alpha_lab.data.ingest as ingest_module

    path = tmp_path / "bars" / "BTCUSDT" / "1m" / "BTCUSDT-1m-2024-01.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(damaged)

    calls = []

    def fake_download(url, **kwargs):
        calls.append(url)
        return _kline_month("2024-01")

    monkeypatch.setattr(ingest_module, "_download", fake_download)
    result = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)[0]

    assert len(calls) == 1
    assert result.ok and result.files == 1 and result.skipped == 0
    assert path.stat().st_size > 0


def test_ingest_result_distinguishes_downloaded_skipped_missing(tmp_path,
                                                                 monkeypatch):
    """В отчёте видны все четыре категории, а не только «ок»."""
    import alpha_lab.data.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "_download",
                        _download_by_month({"2024-01": _kline_month("2024-01")}))
    ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)

    # 2024-01 уже есть (skip), 2024-02 отсутствует в архиве (404),
    # 2024-03 скачивается.
    available = {"2024-01": _kline_month("2024-01"),
                 "2024-03": _kline_month("2024-03")}
    monkeypatch.setattr(ingest_module, "_download", _download_by_month(available))
    result = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-03-31", tmp_path, with_funding=False)[0]

    assert result.ok
    assert result.files == 1 and result.skipped == 1 and result.missing == 1
    assert "скачано месяцев: 1" in result.quality
    assert "уже в хранилище: 1" in result.quality
    assert "пропущено месяцев: 1" in result.quality


# --- Чанкованные Range-загрузки -------------------------------------------
#
# Промежуточный узел сети обрывает соединение примерно на 17 КБ тела, поэтому
# один запрос на файл не докачивает ничего крупнее порога. Крупные файлы
# режутся на диапазоны по chunk_size и качаются пулом; сборка — по номеру
# диапазона, а не по порядку завершения запросов.


def _parse_range(value: str) -> tuple[int, int]:
    import re

    match = re.fullmatch(r"bytes=(\d+)-(\d+)", value)
    assert match is not None, value
    return int(match.group(1)), int(match.group(2))


class _RangeServer:
    """Псевдо-сервер архива: отдаёт диапазоны, не выходя в сеть.

    `requests` (заголовки Range) записывается по каждому обращению — тесты
    проверяют и число запросов, и то, что мелкий файл не режется на чанки.
    """

    def __init__(self, payload: bytes, *, ignore_range: bool = False,
                 fail_ranges: dict[str, int] | None = None,
                 truncate_last: int = 0,
                 delays: dict[int, float] | None = None):
        self.payload = payload
        self.ignore_range = ignore_range
        # "start-end" -> сколько первых обращений упасть таймаутом
        self.fail_ranges = dict(fail_ranges or {})
        self.truncate_last = truncate_last
        self.delays = delays or {}
        self.requests: list[str | None] = []

    def __call__(self, url, headers=None, timeout=None, stream=False):
        import time as _time

        rng = (headers or {}).get("Range")
        self.requests.append(rng)
        length = {"Content-Length": str(len(self.payload))}
        if self.ignore_range or rng is None:
            # Проба без Range; сервер, игнорирующий Range, отвечает так же.
            return _FakeResponse(200, self.payload, length)
        start, end = _parse_range(rng)
        key = f"{start}-{end}"
        if self.fail_ranges.get(key, 0) > 0:
            self.fail_ranges[key] -= 1
            raise requests.Timeout("read timed out")
        body = self.payload[start:end + 1]
        if self.truncate_last and end + 1 >= len(self.payload):
            body = body[: max(0, len(body) - self.truncate_last)]
        delay = self.delays.get(start, 0.0)
        if delay:
            _time.sleep(delay)
        return _FakeResponse(206, body, {
            "Content-Range": f"bytes {start}-{end}/{len(self.payload)}",
            "Content-Length": str(len(body)),
        })


def _chunked_zip(size: int = 9000) -> bytes:
    """Zip с неповторяющимся телом: перестановка чанков не останется незамеченной."""
    import random

    return _zip_payload(random.Random(42).randbytes(size))


def _n_chunks(size: int, chunk_size: int) -> int:
    return -(-size // chunk_size)


def test_download_chunked_reassembles_in_order(monkeypatch):
    """Диапазоны собираются по номерам, даже если завершаются вразнобой."""
    import alpha_lab.data.ingest as ingest_module

    payload = _chunked_zip()
    # Чем раньше чанк, тем он медленнее: завершение заведомо не по порядку.
    server = _RangeServer(payload, delays={1024: 0.2, 2048: 0.1})
    monkeypatch.setattr(ingest_module.requests, "get", server)

    raw = ingest_module._download("http://example.test/x.zip", retries=1,
                                  base_delay=0, chunk_size=1024, concurrency=4)

    assert raw == payload
    assert server.requests[0] is None  # проба без Range: сначала узнаём размер
    ranges = [r for r in server.requests if r is not None]
    assert len(ranges) == _n_chunks(len(payload), 1024)


def test_download_range_ignored_returns_whole_body(monkeypatch):
    """200 вместо 206: Range не поддержан — берём тело целиком, не режем его."""
    import alpha_lab.data.ingest as ingest_module

    payload = _chunked_zip()
    server = _RangeServer(payload, ignore_range=True)
    monkeypatch.setattr(ingest_module.requests, "get", server)

    raw = ingest_module._download("http://example.test/x.zip", retries=1,
                                  base_delay=0, chunk_size=1024, concurrency=4)

    assert raw == payload
    # Проба + первый диапазон: получив 200, пул чанков не запускаем.
    assert len(server.requests) == 2


def test_download_small_file_is_single_request(monkeypatch, capsys):
    """Файл не крупнее чанка качается одним запросом — Range для него лишний."""
    import alpha_lab.data.ingest as ingest_module

    server = _RangeServer(ZIP_CSV)  # ~350 Б много меньше chunk_size
    monkeypatch.setattr(ingest_module.requests, "get", server)

    raw = ingest_module._download("http://example.test/small.zip")

    assert raw == ZIP_CSV
    assert server.requests == [None]  # один запрос и ни одного Range
    assert "[download]" not in capsys.readouterr().err  # мелочь не логируем


def test_download_retries_failing_chunk_without_restarting_file(monkeypatch):
    """Сбойный чанк повторяется на месте; остальные чанки не перекачиваются."""
    import alpha_lab.data.ingest as ingest_module

    payload = _chunked_zip()
    server = _RangeServer(payload, fail_ranges={"1024-2047": 1})
    monkeypatch.setattr(ingest_module.requests, "get", server)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    raw = ingest_module._download("http://example.test/x.zip", retries=2,
                                  base_delay=0, chunk_size=1024, concurrency=4)

    assert raw == payload
    assert server.requests.count("bytes=1024-2047") == 2  # сбой + один повтор
    assert server.requests.count("bytes=0-1023") == 1     # соседей не тронули
    assert server.requests.count("bytes=2048-3071") == 1


def test_download_logs_large_file_progress(monkeypatch, capsys):
    """Крупный файл отмечается в stderr пофайлово, а не почленно."""
    import alpha_lab.data.ingest as ingest_module

    payload = _chunked_zip()
    server = _RangeServer(payload)
    monkeypatch.setattr(ingest_module.requests, "get", server)

    raw = ingest_module._download("http://example.test/big.zip", retries=1,
                                  base_delay=0, chunk_size=1024, concurrency=4)

    assert raw == payload
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if "[download]" in line]
    assert len(lines) == 2  # старт и финиш, не больше
    assert "чанков" in lines[0]
    assert "http://example.test/big.zip" in lines[0]


def test_download_rejects_short_chunk(monkeypatch):
    """Недоданный чанк не «почти получилось»: сборка обязана упасть."""
    import alpha_lab.data.ingest as ingest_module

    payload = _chunked_zip()
    server = _RangeServer(payload, truncate_last=5)
    monkeypatch.setattr(ingest_module.requests, "get", server)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(requests.ConnectionError):
        ingest_module._download("http://example.test/x.zip", retries=1,
                                base_delay=0, chunk_size=1024, concurrency=4)


def test_download_rejects_truncated_zip(monkeypatch):
    """Байты сошлись, но это обрезок zip: BadZipFile, а не короткий датафрейм."""
    import alpha_lab.data.ingest as ingest_module

    full = _zip_payload(KLINE_WITH_HEADER * 200)
    server = _RangeServer(full[: len(full) // 2])
    monkeypatch.setattr(ingest_module.requests, "get", server)
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(zipfile.BadZipFile):
        ingest_module._download("http://example.test/x.zip", retries=1,
                                base_delay=0, chunk_size=1024, concurrency=4)


def test_ingest_reports_corrupt_zip_as_error(tmp_path, monkeypatch):
    """Битый архив — ошибка пары в отчёте, а не исключение из ingest_symbol."""
    import alpha_lab.data.ingest as ingest_module

    def broken(url, **kwargs):
        raise zipfile.BadZipFile("скачанный архив не читается")

    monkeypatch.setattr(ingest_module, "_download", broken)
    results = ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path, with_funding=False)

    assert len(results) == 1
    assert not results[0].ok
    assert "архив" in results[0].error


def test_ingest_forwards_chunk_settings_to_download(tmp_path, monkeypatch):
    """chunk_size/concurrency доходят до загрузчика, а не висят мёртвыми."""
    import alpha_lab.data.ingest as ingest_module

    seen: dict = {}

    def fake_download(url, **kwargs):
        seen.update(kwargs)
        return _kline_month("2024-01")

    monkeypatch.setattr(ingest_module, "_download", fake_download)
    ingest_module.ingest_symbol(
        "BTCUSDT", "1m", "2024-01-01", "2024-01-31", tmp_path,
        with_funding=False, chunk_size=4096, concurrency=3)

    assert seen["chunk_size"] == 4096
    assert seen["concurrency"] == 3
