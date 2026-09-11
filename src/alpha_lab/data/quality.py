"""Гейты качества данных.

Принцип: бар с проблемой не интерполируется и не чинится — он помечается,
и стратегия на нём не торгует. Тихая починка данных создаёт тихо неверный бэктест.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FREQ_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


@dataclass(frozen=True)
class QualityReport:
    total_rows: int
    gaps: int
    duplicates: int
    zero_volume: int
    anomalous: int

    @property
    def bad_rows(self) -> int:
        return self.zero_volume + self.anomalous

    @property
    def is_clean(self) -> bool:
        return self.gaps == 0 and self.duplicates == 0 and self.bad_rows == 0

    def summary(self) -> str:
        if self.is_clean:
            return f"OK: {self.total_rows} баров, проблем нет"
        parts = []
        if self.gaps:
            parts.append(f"пропусков: {self.gaps}")
        if self.duplicates:
            parts.append(f"дубликатов: {self.duplicates}")
        if self.zero_volume:
            parts.append(f"нулевой объём: {self.zero_volume}")
        if self.anomalous:
            parts.append(f"аномалий OHLC: {self.anomalous}")
        return f"ПРОБЛЕМЫ ({self.total_rows} баров): " + ", ".join(parts)


def check_bars(df: pd.DataFrame, freq: str) -> QualityReport:
    if freq not in FREQ_DELTA:
        raise ValueError(f"Неизвестный таймфрейм для проверки: {freq}")
    if df.empty:
        return QualityReport(0, 0, 0, 0, 0)

    delta = FREQ_DELTA[freq]
    ts = pd.to_datetime(df["ts"], utc=True)

    duplicates = int(ts.duplicated().sum())
    gaps = int((ts.diff().dropna() > delta).sum())

    zero_volume = int((pd.to_numeric(df["volume"], errors="coerce") <= 0).sum())

    high, low = df["high"], df["low"]
    open_, close = df["open"], df["close"]
    anomalous = int((
        (high < low)
        | (high < open_) | (high < close)
        | (low > open_) | (low > close)
        | (close <= 0) | (open_ <= 0)
    ).sum())

    return QualityReport(
        total_rows=len(df),
        gaps=gaps,
        duplicates=duplicates,
        zero_volume=zero_volume,
        anomalous=anomalous,
    )


def clean_mask(df: pd.DataFrame) -> pd.Series:
    """Маска баров, пригодных для торговли. Непригодные исключаются, а не чинятся."""
    bad = (
        (pd.to_numeric(df["volume"], errors="coerce") <= 0)
        | (df["high"] < df["low"])
        | (df["close"] <= 0) | (df["open"] <= 0)
        | (df[["open", "high", "low", "close"]].isna().any(axis=1))
    )
    return ~bad
