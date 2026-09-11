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

    @property
    def ok(self) -> bool:
        return self.error is None


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
                  with_funding: bool = True) -> list[IngestResult]:
    """Скачивает и раскладывает все месяцы для одного символа."""
    results: list[IngestResult] = []
    months = month_range(start, end)

    bar_frames, bar_files, missing, dropped = [], 0, 0, 0
    for period in months:
        try:
            raw = _download(archive_url(market, "klines", symbol, freq, period))
        except requests.RequestException as exc:
            results.append(IngestResult(symbol, "klines", 0, 0, "", str(exc)))
            return results
        if raw is None:
            missing += 1
            continue
        stats: dict[str, int] = {}
        bar_frames.append(parse_kline_csv(raw, stats=stats))
        dropped += stats["dropped_rows"]

    if bar_frames:
        bars = pd.concat(bar_frames, ignore_index=True)
        bar_files = len(write_bars(bars, root, symbol, freq))
        rep = check_bars(normalize_bars(bars), freq)
        detail = rep.summary()
        if dropped:
            # Отброшенные бары не проглатываем: они видны в отчёте.
            detail += f"; отброшено строк: {dropped}"
        results.append(IngestResult(symbol, "klines", bar_files, len(bars),
                                    detail, dropped_rows=dropped))
    else:
        results.append(IngestResult(symbol, "klines", 0, 0, "",
                                    f"нет файлов за {len(months)} мес."))

    if with_funding:
        fund_frames = []
        for period in months:
            raw = _download(archive_url(market, "fundingRate", symbol, None, period))
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
    for symbol in universe.symbols:
        out.extend(ingest_symbol(symbol, freq, universe.start, universe.end,
                                 root=root, market=universe.market))
    return out
