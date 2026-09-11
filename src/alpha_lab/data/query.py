"""Чтение данных через DuckDB.

DuckDB читает только нужные колонки и партиции, не поднимая весь датасет в память —
критично при 16 ГБ RAM и десятках гигабайт истории.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pandas as pd

from alpha_lab.data.store import DEFAULT_ROOT, end_bound

# Правила агрегации минутных баров в старший таймфрейм
_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "trades": "sum",
    "taker_buy_volume": "sum",
}
_PANDAS_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min",
                "1h": "1h", "4h": "4h", "1d": "1D"}


def load_bars(root: Path = DEFAULT_ROOT, symbol: str = "BTCUSDT",
              freq: str = "1m", start: str | None = None, end: str | None = None,
              resample: str | None = None) -> pd.DataFrame:
    """Читает бары символа. Если задан resample — агрегирует из freq в resample.

    Границы периода: start включается с начала дня/момента; end-дата включает
    весь конечный день, явный момент времени — точная граница (см. store.end_bound).
    """
    # Только файлы своего символа и таймфрейма: глоб *.parquet подмешал бы в ряд
    # чужой parquet, случайно оказавшийся в каталоге.
    pattern = str(Path(root) / "bars" / symbol / freq / f"{symbol}-{freq}-*.parquet")
    where, params = [], []
    if start:
        where.append("ts >= ?")
        params.append(pd.Timestamp(start, tz="UTC").to_pydatetime())
    if end:
        bound, strict = end_bound(end)
        where.append("ts < ?" if strict else "ts <= ?")
        params.append(bound.to_pydatetime())
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    try:
        df = duckdb.sql(
            f"SELECT * FROM read_parquet('{pattern}') {clause} ORDER BY ts", params=params
        ).df()
    except duckdb.IOException as exc:
        raise FileNotFoundError(
            f"Нет данных: {pattern}. Сначала выполните ingest."
        ) from exc

    if df.empty:
        raise FileNotFoundError(f"Нет данных: {pattern} за указанный период")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    if resample and resample != freq:
        df = _resample(df, resample)
    return df.reset_index(drop=True)


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule not in _PANDAS_FREQ:
        raise ValueError(f"Неизвестный таймфрейм для ресемплинга: {rule}")
    out = (df.set_index("ts")
             .resample(_PANDAS_FREQ[rule], label="left", closed="left")
             .agg(_AGG)
             .dropna(subset=["close"])
             .reset_index())
    return out


def load_funding(root: Path = DEFAULT_ROOT, symbol: str = "BTCUSDT") -> pd.DataFrame:
    pattern = str(Path(root) / "funding" / symbol / "*.parquet")
    try:
        df = duckdb.sql(
            f"SELECT * FROM read_parquet('{pattern}') ORDER BY ts"
        ).df()
    except duckdb.IOException as exc:
        raise FileNotFoundError(f"Нет данных о funding: {pattern}") from exc
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.reset_index(drop=True)


def align_funding_to_bars(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Ставка funding, привязанная к барам.

    Ненулевое значение только в бары, совпадающие с моментом выплаты.
    """
    if funding.empty:
        return pd.Series(0.0, index=bars.index, name="funding_rate")

    lookup = dict(zip(pd.to_datetime(funding["ts"], utc=True),
                      funding["rate"].astype(float)))
    ts = pd.to_datetime(bars["ts"], utc=True)
    return ts.map(lookup).fillna(0.0).astype("float64").rename("funding_rate")


def data_version(root: Path = DEFAULT_ROOT) -> str:
    """Хэш содержимого хранилища. Участвует в experiment_id."""
    digest = hashlib.sha256()
    root = Path(root)
    if not root.exists():
        return "empty"
    for path in sorted(root.rglob("*.parquet")):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()[:16]
