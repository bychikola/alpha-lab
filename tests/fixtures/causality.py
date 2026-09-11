"""Общий harness причинности стратегий.

Соглашение о времени (engine/backtest.py): positions[t] — решение по информации
до бара t включительно, исполняется на баре t+1. Нарушение этого соглашения
(подглядывание) статистикой по результатам не ловится: гипотезы «сигнал посчитан
по прошлому и коррелирует с r_t» и «сигнал посчитан из самого r_t» дают
одинаковое совместное распределение (r_t, h_t), см. spec 8.1. Единственная
рабочая защита — запрос к самой функции решения: стратегия обязана быть
причинным отображением входа в выход.

Индексная конвенция (проверяется явно, reset_index НЕ применяется):
  * generate(bars) возвращает pd.Series с индексом, равным bars.index: позиция
    подписана баром, а не порядковым номером;
  * generate(bars.iloc[:k]) возвращает pd.Series с индексом bars.index[:k] —
    ровно тем же, что у generate(bars).iloc[:k];
  * значения сравниваются поэлементно (NaN считается равным NaN), а не
    выравниванием по меткам: молчаливый reset_index/переиндексация скрыли бы
    расхождение, которое harness обязан поймать.

Не покрыто harness'ом: утечка внутри признака (окно нормализации, читающее
t+1) и стратегия, причинная только под усечением.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _name(strategy) -> str:
    return str(getattr(strategy, "name", type(strategy).__name__))


def _default_cut_points(n: int) -> list[int]:
    return [n // 4, n // 2, 3 * n // 4, n - 1]


def _diff_mask(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """True там, где значения различаются; NaN сравнивается с NaN как равный."""
    return ~((actual == expected) | (np.isnan(actual) & np.isnan(expected)))


def _check_output(series, expected_index, strategy, k: int | None) -> pd.Series:
    """Результат generate обязан быть Series с ожидаемым индексом баров."""
    where = "generate(bars)" if k is None else f"generate(bars.iloc[:{k}])"
    if not isinstance(series, pd.Series):
        raise AssertionError(
            f"Стратегия '{_name(strategy)}': {where} обязан вернуть pd.Series, "
            f"получено {type(series).__name__}"
        )
    if not series.index.equals(expected_index):
        raise AssertionError(
            f"Стратегия '{_name(strategy)}': {where} нарушил индексную конвенцию — "
            f"индекс результата не совпадает с индексом переданных баров. Позиция "
            "обязана быть подписана баром; harness не применяет reset_index и не "
            "выравнивает ряды молча. Приведите индекс результата к bars.index."
        )
    return series


def assert_strategy_is_causal(strategy, bars: pd.DataFrame, cut_points=None) -> None:
    """Проверяет, что стратегия не читает будущее и не зависит от длины ряда.

    Для каждой точки k из [len//4, len//2, 3*len//4, len-1] и переданных
    cut_points требует побитового равенства значений:
        generate(bars.iloc[:k]) == generate(bars).iloc[:k].
    Если будущее на баре t влияет на решение, усечение ряда это вскроет:
    префикс решений изменится. Сравнение поэлементное по позициям баров, с
    явной проверкой индексной конвенции (см. docstring модуля).

    cut_points — дополнительные k; каждое обязано быть целым и 0 < k < len(bars).

    Бросает AssertionError с русским сообщением: первое расхождение k и число
    разошедшихся позиций. ValueError — на некорректные аргументы.
    """
    n = len(bars)
    if n < 2:
        raise ValueError(f"Нужно минимум 2 бара для проверки причинности, получено {n}")

    full = _check_output(strategy.generate(bars), bars.index, strategy, None)

    cuts = {k for k in _default_cut_points(n) if 0 < k < n}
    if cut_points is not None:
        for k in cut_points:
            if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
                raise ValueError(f"Точка усечения должна быть целой, получено {k!r}")
            if not 0 < int(k) < n:
                raise ValueError(
                    f"Точка усечения k={k} вне диапазона 0 < k < {n}"
                )
            cuts.add(int(k))
    if not cuts:
        raise ValueError(f"Нет допустимых точек усечения для ряда длины {n}")

    for k in sorted(cuts):
        truncated_bars = bars.iloc[:k]
        truncated = _check_output(
            strategy.generate(truncated_bars), truncated_bars.index, strategy, k
        )
        expected = full.iloc[:k]

        actual_v = np.asarray(truncated, dtype="float64")
        expected_v = np.asarray(expected, dtype="float64")
        if actual_v.shape != expected_v.shape:
            raise AssertionError(
                f"Стратегия '{_name(strategy)}' не причинна: при усечении до k={k} "
                f"длина позиций {actual_v.shape} вместо {expected_v.shape}"
            )

        diff = _diff_mask(actual_v, expected_v)
        if diff.any():
            first = int(np.flatnonzero(diff)[0])
            raise AssertionError(
                f"Стратегия '{_name(strategy)}' не причинна: первое расхождение "
                f"при k={k} — позиций {int(diff.sum())} из {k}, первая из них на "
                f"баре {expected.index[first]} (позиция {first}). "
                f"generate(bars.iloc[:{k}]) обязан совпадать с "
                f"generate(bars).iloc[:{k}]: либо сигнал читает будущее, либо "
                "результат зависит от длины ряда."
            )
