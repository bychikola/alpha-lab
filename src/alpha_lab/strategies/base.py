"""Протокол стратегии. Одна функция — намеренно узкий интерфейс.

Двухногая книга объявляется ОТДЕЛЬНЫМ необязательным методом generate_legs
(см. PositionLegs/TwoLegStrategy): одноногие стратегии его не имеют и их путь
не меняется, а причинностный harness проверяет все три базы двухногой явно.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Protocol, runtime_checkable

import pandas as pd

# Консервативный дефолт истории для стратегии, не объявившей history_bars.
# Единица означала бы «решение зависит только от текущего бара» — для любой
# стратегии со скользящим окном это ложь, а заниженное требование искажает
# диагностику памяти (extra.history_bars) и любой будущий путь без сегментации.
# 500 — наибольшая память среди штатных стратегий полигона (max_bars у MR);
# стратегия с более длинной памятью обязана объявить своё значение явно.
DEFAULT_HISTORY_BARS = 500


@runtime_checkable
class Strategy(Protocol):
    name: str

    # Сколько хвостовых баров (включая текущий) использует решение на баре t.
    # Подлинная память стратегии: диагностика (extra.history_bars) и требование
    # для несегментированного пути. В рабочем пути CLI маску разрывов она не
    # задаёт: ряд режется на непрерывные участки, и generate после дыры
    # стартует заново, поэтому окна сквозь пропуск не тянутся — вперёд
    # маскировать нечего (внутри участка история остаётся требованием).
    history_bars: int

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        """Целевая позиция: −1.0 (полный шорт) … 0.0 … +1.0 (полный лонг).

        Обязан использовать только информацию по бар t включительно.
        Исполнение происходит на баре t+1 (см. engine.backtest).

        **Индексная конвенция.** Результат обязан быть Series с индексом,
        совпадающим с bars.index: позиция подписана баром, а не позицией в
        массиве. run_backtest потребляет позиции позиционно и проверяет лишь
        длину, поэтому стратегия с дефолтным RangeIndex в движке отработает, но
        причинностный harness (alpha_lab.causality) её отвергнет —
        молчаливое выравнивание по меткам скрыло бы перестановку порядка.

        **Причинность.** Решение не должно зависеть от баров позже t. Это
        проверяется не статистикой (см. spec 8.1 — статистический валидатор
        бессилен против look-ahead по построению), а harness'ом усечения:
        generate(bars.iloc[:k]) обязана побитово совпасть с
        generate(bars).iloc[:k]. Новая стратегия обязана проходить его.
        """
        ...


@dataclass(frozen=True)
class PositionLegs:
    """Двухногая позиция: три базы одного решения, а не один скаляр.

    Одноногий движок держал net и применял к нему три вещи: ценовой P&L,
    издержки оборота и funding. Для дельта-нейтральной книги это неверно:
    net ≡ 0, но торгуются две ноги (издержки не ноль) и funding начисляется на
    ногу перпа (это и есть доход). Поэтому базы разделены:

      * net   — спот + перп: ценовая экспозиция; ценовой P&L, trade_returns и
                вся статистика вердикта;
      * gross — |спот| + |перп|: база комиссий и проскальзывания;
      * carry — ноционал ноги перпа (знаковый): база funding.

    Все три ряда обязаны иметь индекс, равный bars.index, — та же индексная
    конвенция, что и у generate; её проверяет причинностный harness. Длины и
    конечность значений проверяет движок (run_backtest).
    """
    net: pd.Series
    gross: pd.Series
    carry: pd.Series


class TwoLegStrategy:
    """Базовая реализация двухногой стратегии: решение — generate_legs.

    Наследник объявляет name, history_bars, PARAM_NAMES и generate_legs.
    generate выводится из решения (ровно legs.net) и существует для
    потребителей одноногого интерфейса: два разных ответа на один вопрос
    недопустимы. history_bars означает ровно то же, что и раньше, — сколько
    хвостовых баров (включая текущий) требует решение по всем трём базам.

    Причинностный harness для такой стратегии проверяет generate_legs
    (все три базы), а CLI ведёт вердикт по legs.net — той же функции решения,
    а не по производной от неё.
    """

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        raise NotImplementedError(
            f"Стратегия '{getattr(self, 'name', type(self).__name__)}' не "
            f"реализовала generate_legs: двухногий протокол требует решения по "
            f"всем трём базам (net/gross/carry)"
        )

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        return self.generate_legs(bars).net


def legs_generator(strategy):
    """generate_legs стратегии, если она двухногая, иначе None.

    Необязательный метод — точка расширения, не ломающая одноногие классы:
    у них generate_legs отсутствует, и прежний путь (generate) не меняется
    ни на одном байте.
    """
    fn = getattr(strategy, "generate_legs", None)
    return fn if callable(fn) else None


def history_bars_of(strategy) -> int:
    """Требование истории стратегии: объявленное history_bars или дефолт.

    Дефолт консервативен (DEFAULT_HISTORY_BARS), а не 1: заниженное требование
    исказило бы диагностику памяти стратегии и любой будущий путь без
    сегментации. Некорректное объявление (bool, нецелое, < 1) — ошибка, а не
    тихий фолбэк: неверное требование испортило бы диагностику незаметно.
    """
    value = getattr(strategy, "history_bars", DEFAULT_HISTORY_BARS)
    label = getattr(strategy, "name", type(strategy).__name__)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(
            f"history_bars стратегии '{label}' должен быть целым, "
            f"получено {value!r}"
        )
    if value < 1:
        raise ValueError(
            f"history_bars стратегии '{label}' должен быть ≥ 1, "
            f"получено {value!r}"
        )
    return int(value)


def _registry() -> dict[str, type]:
    """Реестр стратегий: штатные плюс добавленные извне.

    REGISTRY — точка расширения: новая стратегия регистрируется в нём, не
    требуя правки ни build_strategy, ни модуля сетки. Штатная mean_reversion
    подмешивается каждый раз, чтобы тестовая подмена REGISTRY не теряла её.
    """
    from alpha_lab.strategies.mean_reversion import MeanReversionStrategy

    return {"mean_reversion": MeanReversionStrategy, **REGISTRY}


# Стратегии, зарегистрированные вне штатного набора (например, расширением).
REGISTRY: dict[str, type] = {}


def strategy_class(name: str) -> type:
    registry = _registry()
    if name not in registry:
        raise ValueError(
            f"Неизвестная стратегия: '{name}'. Доступные: {sorted(registry)}"
        )
    return registry[name]


def strategy_param_names(name: str) -> frozenset[str]:
    """Имена параметров, которые стратегия принимает.

    Пространство параметров спрашивается у самого класса (PARAM_NAMES), а не
    берётся из списка в модуле сетки: иначе каждая новая стратегия требовала бы
    правки grid.py, а забытая правка молча пропускала бы опечатку в имени
    параметра. Класс без объявленного PARAM_NAMES — громкая ошибка: сетка по
    стратегии с неизвестным пространством невалидируема.
    """
    cls = strategy_class(name)
    names = getattr(cls, "PARAM_NAMES", None)
    if names is None:
        raise ValueError(
            f"Стратегия '{name}' не объявила PARAM_NAMES: пространство её "
            f"параметров неизвестно, поэтому сетку по ней построить нельзя. "
            f"Объявите кортеж/множество имён на классе стратегии."
        )
    return frozenset(str(n) for n in names)


def build_strategy(name: str, params: dict) -> Strategy:
    return strategy_class(name)(params)
