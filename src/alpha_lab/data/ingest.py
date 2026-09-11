"""Загрузка публичного архива Binance (data.binance.vision).

Архив бесплатный, без API-ключа и содержит всю историю. Это основной источник;
ccxt используется только как резерв, если архив недоступен из региона.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from alpha_lab.data.quality import check_bars
from alpha_lab.data.schema import RAW_KLINE_COLUMNS, normalize_bars
from alpha_lab.data.store import DEFAULT_ROOT, write_bars, write_funding

BASE_URL = "https://data.binance.vision/data"
MARKET_PREFIX = {"futures-um": "futures/um", "spot": "spot"}
REQUEST_TIMEOUT = 60


@dataclass(frozen=True)
class IngestResult:
    symbol: str
    kind: str
    files: int
    rows: int
    quality: str
    error: str | None = None
    dropped_rows: int = 0
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


def _download(url: str) -> bytes | None:
    """Скачивает файл. None означает «файла нет» (404) — это норма для ранних месяцев."""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def ingest_symbol(symbol: str, freq: str, start: str, end: str | None,
                  root: Path = DEFAULT_ROOT, market: str = "futures-um",
                  with_funding: bool = True,
                  include_delisted: bool = True) -> list[IngestResult]:
    """Скачивает и раскладывает все месяцы для одного символа.

    include_delisted=False исключает пару с оборванной историей из области
    исследования — явно и с пометкой excluded, а не молча (иначе отказ от
    делистингованных пар незаметно вносил бы ошибку выживаемости).
    """
    results: list[IngestResult] = []
    months = month_range(start, end)

    bar_frames, bar_files, missing, dropped = [], 0, 0, 0
    present: list[str] = []
    for period in months:
        try:
            raw = _download(archive_url(market, "klines", symbol, freq, period))
        except requests.RequestException as exc:
            results.append(IngestResult(symbol, "klines", 0, 0, "", str(exc)))
            return results
        if raw is None:
            missing += 1
            continue
        present.append(period)
        stats: dict[str, int] = {}
        bar_frames.append(parse_kline_csv(raw, stats=stats))
        dropped += stats["dropped_rows"]

    if bar_frames:
        bars = pd.concat(bar_frames, ignore_index=True)
        last_bar_ts = pd.Timestamp(pd.to_datetime(bars["ts"], utc=True).max())
        delisted = _is_truncated(months, present, last_bar_ts,
                                 pd.Timestamp.now(tz="UTC"))
        if delisted and not include_delisted:
            results.append(IngestResult(
                symbol, "klines", 0, 0,
                "исключён: include_delisted=false, история обрывается раньше "
                f"периода (последний бар {last_bar_ts:%Y-%m-%d %H:%M} UTC); "
                "отказ от делистингованных пар создаёт ошибку выживаемости",
                delisted=True, last_bar_ts=last_bar_ts, excluded=True))
            return results
        bar_files = len(write_bars(bars, root, symbol, freq))
        rep = check_bars(normalize_bars(bars), freq)
        detail = rep.summary()
        if dropped:
            # Отброшенные бары не проглатываем: они видны в отчёте.
            detail += f"; отброшено строк: {dropped}"
        if missing:
            # 404 — норма для ранних месяцев, но оператор должен видеть масштаб.
            detail += f"; пропущено месяцев: {missing}"
        if delisted:
            # Пара остаётся в юниверсе (spec 5), но обрыв истории виден.
            detail += ("; история обрывается раньше периода "
                       f"(делистинг/обрезка): последний бар "
                       f"{last_bar_ts:%Y-%m-%d %H:%M} UTC")
        results.append(IngestResult(symbol, "klines", bar_files, len(bars),
                                    detail, dropped_rows=dropped,
                                    delisted=delisted, last_bar_ts=last_bar_ts))
    else:
        results.append(IngestResult(symbol, "klines", 0, 0, "",
                                    f"нет файлов за {len(months)} мес."))

    if with_funding:
        fund_frames = []
        for period in months:
            try:
                raw = _download(archive_url(market, "fundingRate", symbol, None,
                                            period))
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


def ingest_universe(universe, freq: str, root: Path = DEFAULT_ROOT
                    ) -> list[IngestResult]:
    out: list[IngestResult] = []
    # getattr: юниверс может быть duck-typed (тесты), дефолт spec 5 — true.
    include_delisted = bool(getattr(universe, "include_delisted", True))
    for symbol in universe.symbols:
        try:
            out.extend(ingest_symbol(symbol, freq, universe.start, universe.end,
                                     root=root, market=universe.market,
                                     include_delisted=include_delisted))
        except Exception as exc:
            # Изоляция символов: один сбойный символ не должен уносить с собой
            # отчёт по остальным — цикл обязан дойти до последнего.
            out.append(IngestResult(symbol, "universe", 0, 0, "", str(exc)))
    return out
