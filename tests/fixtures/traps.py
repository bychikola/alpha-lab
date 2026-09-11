"""Заведомо сломанные стратегии.

Назначение — тестировать НЕ их, а валидатор: он обязан убивать каждую.
Если валидатор пропускает ловушку, значит сломан валидатор.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LookAheadStrategy:
    """Торгует по цене СЛЕДУЮЩЕГО бара — классическое подглядывание в будущее."""

    name = "trap_lookahead"

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        future_return = close.shift(-1) / close - 1.0
        # «Знаем» направление следующего бара и ставим позицию заранее
        signal = np.sign(future_return.fillna(0.0))
        return signal.astype("float64")


class OverfitNoiseStrategy:
    """Подогнана под конкретный исторический отрезок: торгует только там."""

    name = "trap_overfit"

    def __init__(self, start: int = 0, end: int = 50):
        self.start = start
        self.end = end

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        pos = pd.Series(0.0, index=bars.index)
        pos.iloc[self.start:self.end] = 1.0
        return pos


class AlwaysLongStrategy:
    """Всегда в лонге. На растущем ряде выглядит прибыльной без всякой альфы."""

    name = "trap_always_long"

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


class PerfectForesightStrategy:
    """Знает весь будущий ряд целиком — эталон максимального подглядывания."""

    name = "trap_foresight"

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64").to_numpy()
        pos = np.zeros(len(close), dtype="float64")
        pos[:-1] = np.sign(close[1:] - close[:-1])
        return pd.Series(pos, index=bars.index)
