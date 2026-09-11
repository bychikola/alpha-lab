"""Признаки волатильности и funding.

ATR считается методом Уайлдера (RMA с alpha = 1/length), как ta.atr в Pine
Script: рекурсия заводится SMA первых length TR (а не первым TR, как pandas
ewm(adjust=False)), поэтому результаты сходятся с TradingView на общем периоде.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1)
    tr = ranges.max(axis=1)
    tr.iloc[0] = bars["high"].iloc[0] - bars["low"].iloc[0]
    return tr.rename("tr")


def atr(bars: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR методом Уайлдера: RMA с alpha = 1/length и SMA-затравкой первых length TR."""
    tr = true_range(bars)
    if len(tr) < length:
        # Вырожденный случай: SMA первых length TR не существует. Как Pine ta.rma
        # на недостатке баров — отдаём NaN, а не исключение.
        return pd.Series(np.nan, index=tr.index, name="atr")
    seeded = tr.copy()
    seeded.iloc[:length - 1] = np.nan
    seeded.iloc[length - 1] = tr.iloc[:length].mean()
    # Дальше ewm(adjust=False) продолжает ровно рекурсию Уайлдера:
    # rma[i] = (1 - 1/length) * rma[i-1] + (1/length) * tr[i].
    return seeded.ewm(alpha=1.0 / length, adjust=False).mean().rename("atr")


def atr_zscore(bars: pd.DataFrame, atr_len: int = 14, window: int = 200) -> pd.Series:
    """Насколько текущая волатильность отклоняется от своей нормы."""
    a = atr(bars, atr_len)
    mean = a.rolling(window, min_periods=window).mean()
    std = a.rolling(window, min_periods=window).std(ddof=0)
    return ((a - mean) / std.where(std > 1e-12, np.nan)).rename("atr_z")


def volatility_regime(bars: pd.DataFrame, atr_len: int = 14, window: int = 200,
                      z_low: float = -1.0, z_high: float = 1.0) -> pd.Series:
    """Режим волатильности: -1 (тихо), 0 (норма), +1 (бурно)."""
    z = atr_zscore(bars, atr_len, window)
    out = pd.Series(0.0, index=bars.index, name="vol_regime")
    out[z >= z_high] = 1.0
    out[z <= z_low] = -1.0
    out[z.isna()] = np.nan
    return out


def funding_features(rate: pd.Series, window: int = 90) -> pd.DataFrame:
    """Скользящее среднее funding и его z-скор.

    Положительный funding означает, что лонги платят шортам — держать лонг дорого.
    """
    r = pd.Series(rate).astype("float64")
    ma = r.rolling(window, min_periods=1).mean()
    std = r.rolling(window, min_periods=window).std(ddof=0)
    z = ((r - ma) / std.where(std > 1e-12, np.nan)).fillna(0.0)
    return pd.DataFrame({"funding_ma": ma, "funding_z": z})
