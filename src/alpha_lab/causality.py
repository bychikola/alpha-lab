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

Двухногая стратегия объявляет решение через generate_legs (PositionLegs:
net/gross/carry). Тогда проверяются все три базы, каждая — по той же индексной
конвенции и тому же усечению; generate (производный, ровно net) не вызывается
повторно. Одноногие стратегии generate_legs не имеют и проверяются как раньше.

Не покрыто harness'ом: утечка внутри признака (окно нормализации, читающее
t+1) и стратегия, причинная только под усечением.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.strategies.base import PositionLegs, legs_generator


def _name(strategy) -> str:
    return str(getattr(strategy, "name", type(strategy).__name__))


# До этой длины точки усечения берутся сплошь: k = 2 .. n-1.
EXHAUSTIVE_LIMIT = 600
# Выше — плотная сетка: не реже одной точки на каждые TARGET_CUTS позиций.
TARGET_CUTS = 200


def _default_cut_points(n: int) -> list[int]:
    """Плотная сетка точек усечения.

    Точка k проверяет зависимость от будущего ровно в одной позиции — последней
    в префиксе (k−1): все предыдущие позиции присутствуют и в усечённом прогоне,
    поэтому утечка в них невидима. Четыре структурные точки проверяли четыре
    позиции и пропускали утечку, ограниченную отрезком (например, прогревом или
    окном переобучения) — замерено на _SegmentLeak и _WarmupLeak.

    Политика:
        n <= EXHAUSTIVE_LIMIT — сплошь, k = 2 .. n−1 (полное покрытие);
        иначе — шаг max(1, n // TARGET_CUTS), объединённый со структурными
        точками n//4, n//2, 3n//4, n−1.
    """
    if n <= EXHAUSTIVE_LIMIT:
        return list(range(2, n))
    stride = max(1, n // TARGET_CUTS)
    cuts = set(range(2, n, stride))
    cuts.update({n // 4, n // 2, 3 * n // 4, n - 1})
    return sorted(k for k in cuts if 0 < k < n)


def _diff_mask(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """True там, где значения различаются; NaN сравнивается с NaN как равный."""
    return ~((actual == expected) | (np.isnan(actual) & np.isnan(expected)))


def _check_output(series, expected_index, strategy, where: str) -> pd.Series:
    """Ряд решения (generate или колонка generate_legs) с индексом баров."""
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


def _check_legs(legs, expected_index, strategy, where: str) -> PositionLegs:
    """Двухногий результат обязан быть PositionLegs с индексом баров в каждой базе.

    Проверяются все три базы, а не только net: непроверенная carry-база могла бы
    читать будущее, оставаясь невидимой для harness'а, — а вердикт по
    дельта-нейтральной книге считается именно по ней.
    """
    if not isinstance(legs, PositionLegs):
        raise AssertionError(
            f"Стратегия '{_name(strategy)}': {where} обязан вернуть PositionLegs "
            f"(net/gross/carry), получено {type(legs).__name__}"
        )
    for field in ("net", "gross", "carry"):
        _check_output(getattr(legs, field), expected_index, strategy,
                      f"{where}.{field}")
    return legs


def _assert_causal(actual, expected, strategy, k: int, full: str,
                   trunc: str) -> None:
    """Побитовое сравнение усечённого и полного ряда; первое расхождение — в текст.

    NaN считается равным NaN (усечение не обязано совпадать с полным рядом по
    «пустоте»), значения сравниваются поэлементно, без выравнивания по меткам.
    """
    actual_v = np.asarray(actual, dtype="float64")
    expected_v = np.asarray(expected, dtype="float64")
    if actual_v.shape != expected_v.shape:
        raise AssertionError(
            f"Стратегия '{_name(strategy)}' не причинна: при усечении до k={k} "
            f"длина {full} — {actual_v.shape} вместо {expected_v.shape}"
        )
    diff = _diff_mask(actual_v, expected_v)
    if diff.any():
        first = int(np.flatnonzero(diff)[0])
        raise AssertionError(
            f"Стратегия '{_name(strategy)}' не причинна: первое расхождение "
            f"при k={k} — позиций {int(diff.sum())} из {k}, первая из них на "
            f"баре {expected.index[first]} (позиция {first}). "
            f"{trunc} обязан совпадать с {full}.iloc[:{k}]: либо сигнал "
            "читает будущее, либо результат зависит от длины ряда."
        )


def assert_strategy_is_causal(strategy, bars: pd.DataFrame, cut_points=None) -> int:
    """Проверяет причинность стратегии: решение на баре t не зависит от баров > t.

    Для каждой точки k требует побитового равенства значений:
        generate(bars.iloc[:k]) == generate(bars).iloc[:k].

    **Что именно проверяется.** Точка k вскрывает зависимость от будущего ровно
    в одной позиции — последней в префиксе (k−1). Все предыдущие позиции t ≤ k−2
    присутствуют и в усечённом прогоне, поэтому утечка в них при данном k
    невидима: сравнивать не с чем. Отсюда политика плотности (см.
    _default_cut_points) — при n <= EXHAUSTIVE_LIMIT покрываются ВСЕ позиции;
    выше этого порога сетка плотная, но не сплошная, и утечка, целиком лежащая
    между её точками, теоретически может остаться незамеченной. Для полной
    гарантии на длинном ряде передайте cut_points="all".

    **Двухногая стратегия.** Если у стратегии есть generate_legs, решением
    считается он, и проверяются ВСЕ три базы (net/gross/carry) — CLI ведёт
    вердикт по generate_legs, поэтому harness обязан судить ровно ту функцию,
    которая строит вердикт. Производный generate для двухногой стратегии не
    вызывается: он по контракту равен legs.net, а двойной вызов решения удвоил
    бы цену проверки, ничего не добавив.

    **Чего не проверяется** (зафиксировано в spec 8.1): утечка на уровне
    признаков — если стратегия получает уже посчитанные признаки, harness судит
    лишь по её решению; поведение последнего бара полного ряда (позиция n−1 в
    точки усечения не входит); стратегия, ведущая себя причинно только под
    усечением.

    cut_points — дополнительные k; каждое обязано быть целым и 0 < k < len(bars).
    cut_points="all" — сплошное покрытие 2..n−1 независимо от длины ряда.

    Возвращает число проверенных точек усечения. Гарантия выборочная: на длинном
    ряде точки идут с шагом n // TARGET_CUTS, поэтому возвращённое число — это
    мера покрытия, а не «все позиции». Вызывающий обязан записать его в отчёт:
    без счётчика выборочная проверка неотличима от сплошной.

    Бросает AssertionError с русским сообщением: первое расхождение k и число
    разошедшихся позиций. ValueError — на некорректные аргументы.
    """
    n = len(bars)
    if n < 2:
        raise ValueError(f"Нужно минимум 2 бара для проверки причинности, получено {n}")

    legs_fn = legs_generator(strategy)
    if legs_fn is not None:
        full_legs = _check_legs(legs_fn(bars), bars.index, strategy,
                                "generate_legs(bars)")
    else:
        full = _check_output(strategy.generate(bars), bars.index, strategy,
                             "generate(bars)")

    if isinstance(cut_points, str):
        if cut_points != "all":
            raise ValueError(
                f"cut_points как строка допускает только 'all', получено {cut_points!r}"
            )
        cuts = set(range(2, n))
    else:
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
        if legs_fn is not None:
            where_trunc = f"generate_legs(bars.iloc[:{k}])"
            truncated_legs = _check_legs(legs_fn(truncated_bars),
                                         truncated_bars.index, strategy,
                                         where_trunc)
            for field in ("net", "gross", "carry"):
                _assert_causal(
                    getattr(truncated_legs, field),
                    getattr(full_legs, field).iloc[:k],
                    strategy, k,
                    full=f"generate_legs(bars).{field}",
                    trunc=f"{where_trunc}.{field}",
                )
        else:
            truncated = _check_output(
                strategy.generate(truncated_bars), truncated_bars.index,
                strategy, f"generate(bars.iloc[:{k}])",
            )
            _assert_causal(
                truncated, full.iloc[:k], strategy, k,
                full="generate(bars)", trunc=f"generate(bars.iloc[:{k}])",
            )
    return len(cuts)
