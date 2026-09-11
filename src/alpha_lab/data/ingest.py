"""Загрузка публичного архива Binance (data.binance.vision).

Архив бесплатный, без API-ключа и содержит всю историю. Это основной источник;
ccxt используется только как резерв, если архив недоступен из региона.
"""
from __future__ import annotations

import io
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from alpha_lab.data.quality import check_bars
from alpha_lab.data.schema import RAW_KLINE_COLUMNS, normalize_bars
from alpha_lab.data.store import (
    DEFAULT_ROOT,
    bars_path,
    parquet_row_count,
    write_bars,
    write_funding,
)

BASE_URL = "https://data.binance.vision/data"
MARKET_PREFIX = {"futures-um": "futures/um", "spot": "spot"}
REQUEST_TIMEOUT = 60

# Политика повторов. В прогоне по вселенной ~2000 файлов; единичный Read
# timed out — не редкость, а закономерность, и без повтора он стоит символа
# целиком (четыре года истории из-за одного запроса из ~96).
DEFAULT_RETRIES = 3              # повторов после первой попытки (итого до 4)
DEFAULT_RETRY_BASE_DELAY = 0.5   # база экспоненциального backoff, сек
MAX_RETRY_DELAY = 30.0           # потолок одной паузы backoff, сек
MAX_RETRY_AFTER = 60.0           # потолок уважения чужого Retry-After, сек


@dataclass(frozen=True)
class IngestResult:
    """Итог по одной паре (symbol, kind).

    Категории не растворяются в «ок»: `files` — сколько файлов записано в
    этом прогоне (downloaded), `skipped` — сколько месяцев пропущено как уже
    лежащие в хранилище (resume), `missing` — сколько месяцев отсутствует в
    архиве (404), сбой загрузки виден по `error` (ok == False). Оператор
    завершившегося прогона обязан по отчёту видеть, что именно не качалось.
    """
    symbol: str
    kind: str
    files: int
    rows: int
    quality: str
    error: str | None = None
    dropped_rows: int = 0
    # Сколько месяцев не перекачивалось, потому что валидный parquet уже лежит
    # в хранилище (при force=False). Ноль означает, что качалось всё.
    skipped: int = 0
    # Сколько месяцев архива отсутствовало (404) — норма для ранних месяцев,
    # но масштаб обязан быть виден.
    missing: int = 0
    # История пары обрывается раньше запрошенного периода (делистинг или
    # неполный архивный файл). По spec 5 такие пары обязаны оставаться в
    # юниверсе, иначе возникает ошибка выживаемости; поле делает обрыв
    # наблюдаемым, а не молчаливым.
    delisted: bool = False
    last_bar_ts: pd.Timestamp | None = None
    # Пара исключена из ingest явным include_delisted=false. Отдельное поле, а
    # не error: отказ от пары — решение конфига, а не сбой загрузки.
    excluded: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def _is_truncated(months: list[str], present: list[str],
                  last_bar_ts: pd.Timestamp, now: pd.Timestamp) -> bool:
    """Обрывается ли история раньше запрошенного периода.

    Два признака: (а) после последнего доступного месяца идут отсутствующие
    завершённые месяцы; (б) файл последнего месяца неполон — последний бар
    заметно раньше конца месяца. Текущий (незавершённый) месяц пропуском не
    считается: месячный архив публикуется только после его окончания, поэтому
    живая пара в середине месяца не должна выглядеть делистингованной.
    """
    if not present:
        return False
    trailing = months[months.index(present[-1]) + 1:]
    if any(pd.Period(m, freq="M").end_time.tz_localize("UTC") < now
           for m in trailing):
        return True
    month_end = pd.Period(present[-1], freq="M").end_time.tz_localize("UTC")
    return month_end < now and last_bar_ts < month_end - pd.Timedelta(days=1)


def archive_url(market: str, kind: str, symbol: str, freq: str | None,
                period: str) -> str:
    """URL месячного файла архива. period в формате YYYY-MM."""
    prefix = MARKET_PREFIX.get(market)
    if prefix is None:
        raise ValueError(f"Неизвестный рынок: {market}")
    if kind == "klines":
        name = f"{symbol}-{freq}-{period}.zip"
        return f"{BASE_URL}/{prefix}/monthly/klines/{symbol}/{freq}/{name}"
    if kind == "fundingRate":
        name = f"{symbol}-fundingRate-{period}.zip"
        return f"{BASE_URL}/{prefix}/monthly/fundingRate/{symbol}/{name}"
    raise ValueError(f"Неизвестный тип данных: {kind}")


def month_range(start: str, end: str | None) -> list[str]:
    """Список месяцев YYYY-MM, покрывающих [start, end]."""
    first = pd.Period(pd.Timestamp(start), freq="M")
    last = pd.Period(pd.Timestamp(end or date.today().isoformat()), freq="M")
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def _read_zip_csv(raw: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        return zf.read(name)


def _has_header(first_line: bytes) -> bool:
    """Заголовок присутствует, если первое поле не является числом."""
    head = first_line.split(b",", 1)[0].strip()
    try:
        float(head)
    except ValueError:
        return True
    return False


def parse_kline_csv(raw: bytes, *,
                    stats: dict[str, int] | None = None) -> pd.DataFrame:
    """Парсит CSV klines. Терпим к наличию и отсутствию заголовка.

    Если передан `stats`, в него записывается `dropped_rows` — сколько строк
    нормализация выбросила (например, бары с пустой ценой). Молча терять бары
    нельзя: число уходит наружу через `IngestResult`.
    """
    content = _read_zip_csv(raw) if raw[:2] == b"PK" else raw
    first_line = content.split(b"\n", 1)[0]
    skip = 1 if _has_header(first_line) else 0

    df = pd.read_csv(
        io.BytesIO(content), header=None, skiprows=skip,
        names=RAW_KLINE_COLUMNS, usecols=range(len(RAW_KLINE_COLUMNS)),
    )
    df = df.rename(columns={
        "open_time": "ts", "count": "trades",
        "taker_buy_quote_volume": "_taker_quote", "ignore": "_ignore",
    })
    # open_time в миллисекундах; в очень старых файлах — в микросекундах
    ts = pd.to_numeric(df["ts"], errors="coerce")
    if ts.dropna().median() > 1e15:
        ts = ts / 1000.0
    df["ts"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.drop(columns=[c for c in ("_taker_quote", "_ignore") if c in df.columns])

    parsed_rows = len(df)
    # Бар с NaT-меткой write_bars молча выбросил бы при группировке по месяцу
    # (pandas не создаёт группу для NaT), и он числился бы загруженным. Убираем
    # такие строки здесь и считаем их отброшенными — отчёт обязан совпадать с
    # тем, что реально ляжет на диск.
    df = df.loc[df["ts"].notna()]
    out = normalize_bars(df)
    if stats is not None:
        stats["dropped_rows"] = parsed_rows - len(out)
    return out


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    """Парсит CSV funding. Ставка в долях; интервал читается из файла."""
    content = _read_zip_csv(raw) if raw[:2] == b"PK" else raw
    first_line = content.split(b"\n", 1)[0]
    has_header = first_line.lower().startswith(b"calc_time") or _has_header(first_line)

    df = pd.read_csv(
        io.BytesIO(content), header=0 if has_header else None,
        names=None if has_header else ["calc_time", "funding_interval_hours",
                                       "last_funding_rate"],
    )
    df = df.rename(columns={"calc_time": "ts", "last_funding_rate": "rate",
                            "funding_interval_hours": "interval_hours"})
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").astype("float64")
    df["interval_hours"] = pd.to_numeric(
        df.get("interval_hours", 8), errors="coerce").fillna(8).astype("int64")
    return (df.loc[:, ["ts", "rate", "interval_hours"]]
              .dropna(subset=["rate"])
              .drop_duplicates(subset="ts", keep="last")
              .sort_values("ts")
              .reset_index(drop=True))


def _backoff_delay(base_delay: float, attempt: int) -> float:
    """Пауза перед повтором номер attempt (1-based): экспонента с джиттером.

    Потолок MAX_RETRY_DELAY не даёт мёртвому эндпоинту растянуть прогон на
    часы; джиттер разводит одновременные повторы разных символов, чтобы они не
    били в архив синхронно.
    """
    capped = min(base_delay * (2 ** (attempt - 1)), MAX_RETRY_DELAY)
    if capped <= 0:
        return 0.0
    return random.uniform(capped / 2, capped)


def _retry_after_delay(resp: requests.Response) -> float | None:
    """Retry-After в секундах, если он есть и разбирается.

    Заголовок может быть и HTTP-датой; такой формат не поддерживаем и
    откатываемся на обычный backoff — главное, что чужое значение не может
    превысить MAX_RETRY_AFTER.
    """
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, min(seconds, MAX_RETRY_AFTER))


def _log_retry(url: str, attempt: int, total: int, exc: BaseException,
               delay: float) -> None:
    """Строка в stderr на каждый повтор: длинный прогон должен быть разбираем."""
    print(f"[retry] {url}: попытка {attempt}/{total} не удалась "
          f"({exc.__class__.__name__}: {exc}); повтор через {delay:.1f} с",
          file=sys.stderr)


def _download(url: str, *, retries: int = DEFAULT_RETRIES,
              base_delay: float = DEFAULT_RETRY_BASE_DELAY,
              timeout: float = REQUEST_TIMEOUT) -> bytes | None:
    """Скачивает файл, повторяя только временные сбои.

    None означает «файла нет» (404) — норма для ранних месяцев и текущего
    незавершённого: 404 не повторяется никогда, иначе полный прогон тратит
    минуты на заведомо отсутствующие файлы.

    Повторяются: Timeout, ConnectionError, 429 и 5xx. Прочие 4xx — ошибка
    запроса (URL, права), повтор её не исправит и только откладывает провал.
    После исчерпания повторов последняя ошибка всплывает наружу, а не
    превращается в тихий None.
    """
    total = retries + 1
    for attempt in range(1, total + 1):
        try:
            resp = requests.get(url, timeout=timeout)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == total:
                raise
            delay = _backoff_delay(base_delay, attempt)
            _log_retry(url, attempt, total, exc, delay)
            time.sleep(delay)
            continue

        if resp.status_code == 404:
            return None
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == total:
                resp.raise_for_status()
            retry_after = (_retry_after_delay(resp)
                           if resp.status_code == 429 else None)
            delay = (retry_after if retry_after is not None
                     else _backoff_delay(base_delay, attempt))
            _log_retry(url, attempt, total,
                       requests.HTTPError(f"HTTP {resp.status_code}", response=resp),
                       delay)
            time.sleep(delay)
            continue

        resp.raise_for_status()
        return resp.content
    raise RuntimeError("цикл повторов завершился без результата")  # pragma: no cover


def _parquet_max_ts(path: Path) -> pd.Timestamp | None:
    """Максимальная метка ts месячного parquet (читается только колонка ts)."""
    ts = pd.read_parquet(path, columns=["ts"])["ts"]
    if ts.empty:
        return None
    return pd.Timestamp(pd.to_datetime(ts, utc=True).max())


def ingest_symbol(symbol: str, freq: str, start: str, end: str | None,
                  root: Path = DEFAULT_ROOT, market: str = "futures-um",
                  with_funding: bool = True,
                  include_delisted: bool = True,
                  force: bool = False,
                  retries: int = DEFAULT_RETRIES,
                  retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY
                  ) -> list[IngestResult]:
    """Скачивает и раскладывает все месяцы для одного символа.

    Повторный прогон не перекачивает то, что уже лежит в хранилище: месяц
    пропускается, если его parquet читается и непуст (см. store.parquet_row_count).
    force=True отменяет пропуск — явная перекачка поверх существующего.

    retries/retry_base_delay настраивают политику повторов HTTP.

    include_delisted=False исключает пару с оборванной историей из области
    исследования — явно и с пометкой excluded, а не молча (иначе отказ от
    делистингованных пар незаметно вносил бы ошибку выживаемости).
    """
    results: list[IngestResult] = []
    months = month_range(start, end)

    bar_frames: list[pd.DataFrame] = []
    present: list[str] = []
    downloaded_periods: list[str] = []
    downloaded = skipped = missing = dropped = skipped_rows = 0

    for period in months:
        if not force:
            rows_on_disk = parquet_row_count(bars_path(root, symbol, freq, period))
            if rows_on_disk is not None:
                # Валидный непустой parquet — месяц уже загружен, сеть не нужна.
                skipped += 1
                skipped_rows += rows_on_disk
                present.append(period)
                continue
        try:
            raw = _download(archive_url(market, "klines", symbol, freq, period),
                            retries=retries, base_delay=retry_base_delay)
        except requests.RequestException as exc:
            results.append(IngestResult(
                symbol, "klines", 0, skipped_rows, "", str(exc),
                skipped=skipped, missing=missing))
            return results
        if raw is None:
            missing += 1
            continue
        present.append(period)
        downloaded_periods.append(period)
        downloaded += 1
        stats: dict[str, int] = {}
        bar_frames.append(parse_kline_csv(raw, stats=stats))
        dropped += stats["dropped_rows"]

    bars = pd.concat(bar_frames, ignore_index=True) if bar_frames else None
    last_bar_ts: pd.Timestamp | None = None
    if bars is not None and present and present[-1] in downloaded_periods:
        last_bar_ts = pd.Timestamp(pd.to_datetime(bars["ts"], utc=True).max())
    elif present:
        # Последний присутствующий месяц не перекачивался: его максимум ts
        # читается с диска (одна колонка). Иначе last_bar_ts оказался бы раньше
        # реального конца ряда, и пара ложно выглядела бы делистингованной.
        last_bar_ts = _parquet_max_ts(
            bars_path(root, symbol, freq, present[-1]))

    delisted = bool(present) and last_bar_ts is not None and _is_truncated(
        months, present, last_bar_ts, pd.Timestamp.now(tz="UTC"))
    if delisted and not include_delisted:
        results.append(IngestResult(
            symbol, "klines", 0, 0,
            "исключён: include_delisted=false, история обрывается раньше "
            f"периода (последний бар {last_bar_ts:%Y-%m-%d %H:%M} UTC); "
            "отказ от делистингованных пар создаёт ошибку выживаемости",
            skipped=skipped, missing=missing,
            delisted=True, last_bar_ts=last_bar_ts, excluded=True))
        return results

    total_rows = (len(bars) if bars is not None else 0) + skipped_rows
    if bars is not None:
        bar_files = len(write_bars(bars, root, symbol, freq))
        rep = check_bars(normalize_bars(bars), freq)
        detail = rep.summary()
        detail += f"; скачано месяцев: {downloaded}"
        if skipped:
            # Пропуск по resume виден: иначе «ок» скрывал бы, что месяц не качался.
            detail += f"; уже в хранилище: {skipped}"
        if missing:
            # 404 — норма для ранних месяцев, но оператор должен видеть масштаб.
            detail += f"; пропущено месяцев: {missing}"
        if dropped:
            # Отброшенные бары не проглатываем: они видны в отчёте.
            detail += f"; отброшено строк: {dropped}"
        if delisted:
            # Пара остаётся в юниверсе (spec 5), но обрыв истории виден.
            detail += ("; история обрывается раньше периода "
                       f"(делистинг/обрезка): последний бар "
                       f"{last_bar_ts:%Y-%m-%d %H:%M} UTC")
        results.append(IngestResult(symbol, "klines", bar_files, total_rows,
                                    detail, dropped_rows=dropped,
                                    skipped=skipped, missing=missing,
                                    delisted=delisted, last_bar_ts=last_bar_ts))
    elif skipped:
        # Все месяцы уже в хранилище: это успех, а не «нет файлов». Строки
        # считаются по метаданным parquet, чтобы отчёт не показывал ноль.
        detail = (f"OK: данные уже в хранилище (месяцев: {skipped}); "
                  f"скачано месяцев: 0; проверка качества не выполнялась")
        if missing:
            detail += f"; пропущено месяцев: {missing}"
        results.append(IngestResult(symbol, "klines", 0, total_rows, detail,
                                    skipped=skipped, missing=missing,
                                    delisted=delisted, last_bar_ts=last_bar_ts))
    else:
        results.append(IngestResult(symbol, "klines", 0, 0, "",
                                    f"нет файлов за {len(months)} мес.",
                                    missing=missing))

    if with_funding:
        fund_frames = []
        for period in months:
            try:
                raw = _download(archive_url(market, "fundingRate", symbol, None,
                                            period),
                                retries=retries, base_delay=retry_base_delay)
            except requests.RequestException as exc:
                # Сбой funding не должен уносить с собой уже загруженные klines.
                results.append(IngestResult(symbol, "funding", 0, 0, "", str(exc)))
                return results
            if raw is not None:
                fund_frames.append(parse_funding_csv(raw))
        if fund_frames:
            funding = pd.concat(fund_frames, ignore_index=True)
            write_funding(funding, root, symbol)
            results.append(IngestResult(symbol, "funding", 1, len(funding), "OK"))
        else:
            results.append(IngestResult(symbol, "funding", 0, 0, "",
                                        "нет файлов fundingRate"))

    return results


def ingest_universe(universe, freq: str, root: Path = DEFAULT_ROOT,
                    force: bool = False,
                    retries: int = DEFAULT_RETRIES,
                    retry_base_delay: float = DEFAULT_RETRY_BASE_DELAY
                    ) -> list[IngestResult]:
    out: list[IngestResult] = []
    # getattr: юниверс может быть duck-typed (тесты), дефолт spec 5 — true.
    include_delisted = bool(getattr(universe, "include_delisted", True))
    for symbol in universe.symbols:
        try:
            out.extend(ingest_symbol(symbol, freq, universe.start, universe.end,
                                     root=root, market=universe.market,
                                     include_delisted=include_delisted,
                                     force=force, retries=retries,
                                     retry_base_delay=retry_base_delay))
        except Exception as exc:
            # Изоляция символов: один сбойный символ не должен уносить с собой
            # отчёт по остальным — цикл обязан дойти до последнего.
            out.append(IngestResult(symbol, "universe", 0, 0, "", str(exc)))
    return out
