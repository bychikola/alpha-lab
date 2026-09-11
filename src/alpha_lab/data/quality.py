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


def periods_per_year(timeframe: str) -> int:
    """Число баров таймфрейма в году (крипта торгуется 24/7, 365 дней).

    Единственный источник правды о годовом множителе: он выводится из
    FREQ_DELTA, поэтому новый таймфрейм нельзя добавить в одну карту и забыть
    в другой. Неизвестный таймфрейм — ValueError, а не молчаливый дефолт:
    неверный множитель невидим в отчёте (Sharpe отличается на sqrt(24),
    Calmar — ровно в 24 раза) и тихо портит все метрики вердикта.
    """
    if timeframe not in FREQ_DELTA:
        raise ValueError(
            f"Неизвестный таймфрейм для аннуализации: {timeframe!r}. "
            f"Допустимые: {', '.join(FREQ_DELTA)}"
        )
    return int(round(pd.Timedelta(days=365) / FREQ_DELTA[timeframe]))


@dataclass(frozen=True)
class QualityReport:
    total_rows: int
    gaps: int
    duplicates: int
    zero_volume: int
    anomalous: int

    @property
    def bad_rows(self) -> int:
        """Сумма счётчиков проблем, а не число неторгуемых баров.

        Бар может иметь и объёмную, и ценовую проблему и попасть сразу в оба
        счётчика, поэтому `total_rows - bad_rows` — не количество торгуемых
        баров: торгуемость определяет `clean_mask`.
        """
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


def _bad_bar_mask(
    df: pd.DataFrame, *, volume: bool = True, prices: bool = True
) -> pd.Series:
    """Единый источник правды о проблемных барах: True — бар исключается.

    Флаги `volume`/`prices` лишь разбивают проверки на группы, чтобы `check_bars`
    мог посчитать счётчики; решения о торговле всегда принимаются по объединению
    (флаги по умолчанию), поэтому репортёр и фильтр не могут разойтись.
    """
    masks = []

    if volume:
        vol = pd.to_numeric(df["volume"], errors="coerce")
        # NaN и нечитаемые значения тоже проблема, а не «объём в норме».
        masks.append(vol.isna() | (vol <= 0))

    if prices:
        open_ = df["open"]
        high = df["high"]
        low = df["low"]
        close = df["close"]
        masks.append(
            (high < low)
            | (high < open_) | (high < close)
            | (low > open_) | (low > close)
            | (open_ <= 0) | (high <= 0) | (low <= 0) | (close <= 0)
            | df[["open", "high", "low", "close"]].isna().any(axis=1)
        )

    bad = masks[0]
    for mask in masks[1:]:
        bad = bad | mask
    return bad


def check_bars(df: pd.DataFrame, freq: str) -> QualityReport:
    if freq not in FREQ_DELTA:
        raise ValueError(f"Неизвестный таймфрейм для проверки: {freq}")
    if df.empty:
        return QualityReport(0, 0, 0, 0, 0)

    delta = FREQ_DELTA[freq]
    ts = pd.to_datetime(df["ts"], utc=True)

    duplicates = int(ts.duplicated().sum())
    gaps = int((ts.diff().dropna() > delta).sum())

    # Счётчики берутся из того же источника правды, что и clean_mask.
    zero_volume = int(_bad_bar_mask(df, prices=False).sum())
    anomalous = int(_bad_bar_mask(df, volume=False).sum())

    return QualityReport(
        total_rows=len(df),
        gaps=gaps,
        duplicates=duplicates,
        zero_volume=zero_volume,
        anomalous=anomalous,
    )


def clean_mask(df: pd.DataFrame) -> pd.Series:
    """Маска баров, пригодных для торговли. Непригодные исключаются, а не чинятся."""
    return ~_bad_bar_mask(df)
