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

        **Индексная конвенция.** Результат обязан быть Series с индексом,
        совпадающим с bars.index: позиция подписана баром, а не позицией в
        массиве. run_backtest потребляет позиции позиционно и проверяет лишь
        длину, поэтому стратегия с дефолтным RangeIndex в движке отработает, но
        причинностный harness (tests/fixtures/causality.py) её отвергнет —
        молчаливое выравнивание по меткам скрыло бы перестановку порядка.

        **Причинность.** Решение не должно зависеть от баров позже t. Это
        проверяется не статистикой (см. spec 8.1 — статистический валидатор
        бессилен против look-ahead по построению), а harness'ом усечения:
        generate(bars.iloc[:k]) обязана побитово совпасть с
        generate(bars).iloc[:k]. Новая стратегия обязана проходить его.
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
