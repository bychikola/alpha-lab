"""Протокол стратегии. Одна функция — намеренно узкий интерфейс."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    name: str

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        """Целевая позиция: −1.0 (полный шорт) … 0.0 … +1.0 (полный лонг).

        Обязан использовать только информацию по бар t включительно.
        Исполнение происходит на баре t+1 (см. engine.backtest).
        """
        ...


def build_strategy(name: str, params: dict) -> Strategy:
    from alpha_lab.strategies.mean_reversion import MeanReversionStrategy

    registry = {"mean_reversion": MeanReversionStrategy}
    if name not in registry:
        raise ValueError(
            f"Неизвестная стратегия: '{name}'. Доступные: {sorted(registry)}"
        )
    return registry[name](params)
