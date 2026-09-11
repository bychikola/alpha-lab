"""Parquet-хранилище на D:\\alpha-lab\\data.

Раскладка: <root>/bars/<symbol>/<freq>/<symbol>-<freq>-<YYYY-MM>.parquet
           <root>/funding/<symbol>/<symbol>-funding.parquet
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from alpha_lab.data.schema import FUNDING_COLUMNS, normalize_bars

DEFAULT_ROOT = Path(r"D:\alpha-lab\data")

_DATE_ONLY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def end_bound(end) -> tuple[pd.Timestamp, bool]:
    """Граница периода по end: (момент времени, строгое ли неравенство).

    Семантика: end-дата без времени ("2025-12-31") означает конец этого дня
    включительно — иначе фильтр ts <= end покрывал бы только полночь, и
    последние 23 часа периода молча выпадали из прогона. Явный момент времени
    ("2025-12-31T12:00" или datetime) остаётся точной границей: ts <= end.
    """
    ts = pd.Timestamp(end, tz="UTC")
    if isinstance(end, datetime):
        return ts, False
    if isinstance(end, date):
        # date без времени — это дата, а не полночь: следующий день, ts < bound.
        return ts + pd.Timedelta(days=1), True
    if _DATE_ONLY.fullmatch(str(end).strip()):
        return ts + pd.Timedelta(days=1), True
    return ts, False


def bars_dir(root: Path, symbol: str, freq: str) -> Path:
    return Path(root) / "bars" / symbol / freq


def bars_path(root: Path, symbol: str, freq: str, period: str) -> Path:
    """Путь месячного parquet: <freq>/<symbol>-<freq>-<YYYY-MM>.parquet.

    Единственный источник правды о раскладке: ingest проверяет по этому пути
    «месяц уже загружен», а write_bars по нему же пишет — разойтись не могут.
    """
    return bars_dir(root, symbol, freq) / f"{symbol}-{freq}-{period}.parquet"


def funding_path(root: Path, symbol: str) -> Path:
    return Path(root) / "funding" / symbol / f"{symbol}-funding.parquet"


def parquet_row_count(path: Path) -> int | None:
    """Число строк parquet или None, если файлу доверять нельзя.

    «Файл существует» не равно «файл дописан»: write_bars/write_funding пишут
    parquet прямо по конечному пути, поэтому обрыв процесса или диска может
    оставить усечённый (вплоть до нулевого) файл. None — файла нет, он пуст,
    его footer не читается или в нём ноль строк; вызывающий обязан перекачать
    месяц, а не считать его готовым.
    """
    path = Path(path)
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return None
        rows = pq.ParquetFile(path).metadata.num_rows
    except Exception:
        # Битый/усечённый parquet кидает разные типы (ArrowInvalid, OSError);
        # для нас любой из них означает одно — данным доверять нельзя.
        return None
    return rows if rows > 0 else None


def write_bars(df: pd.DataFrame, root: Path, symbol: str, freq: str) -> list[Path]:
    df = normalize_bars(df)
    out_dir = bars_dir(root, symbol, freq)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for period, chunk in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        path = bars_path(root, symbol, freq, period)
        chunk.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")
        written.append(path)
    return sorted(written)


def read_bars(root: Path, symbol: str, freq: str,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out_dir = bars_dir(root, symbol, freq)
    # Только файлы своего символа и таймфрейма: глоб *.parquet молча подмешал бы
    # в ряд чужой parquet, случайно оказавшийся в каталоге.
    files = sorted(out_dir.glob(f"{symbol}-{freq}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет данных: {out_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = normalize_bars(df)

    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        bound, strict = end_bound(end)
        df = df[df["ts"] < bound] if strict else df[df["ts"] <= bound]
    return df.reset_index(drop=True)


def write_funding(df: pd.DataFrame, root: Path, symbol: str) -> Path:
    out_dir = Path(root) / "funding" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = funding_path(root, symbol)

    out = df.loc[:, FUNDING_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = (out.drop_duplicates(subset="ts", keep="last")
              .sort_values("ts")
              .reset_index(drop=True))
    out.to_parquet(path, index=False, compression="zstd")
    return path


def read_funding(root: Path, symbol: str) -> pd.DataFrame:
    path = funding_path(root, symbol)
    if not path.exists():
        raise FileNotFoundError(f"Нет данных о funding: {path}")
    out = pd.read_parquet(path)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values("ts").reset_index(drop=True)
