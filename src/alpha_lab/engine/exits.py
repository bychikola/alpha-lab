"""Путь-зависимая симуляция выходов по стопу и тейку.

Векторизовать нельзя: результат каждого бара зависит от того, были ли мы в
позиции на предыдущем. Цикл компилируется numba; при недоступности numba тот
же алгоритм исполняется чистым Python (медленнее). Чисто-Python функция
`_simulate_py` тестируется напрямую, поэтому ветка отката не мёртвая.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # numba ускоряет цикл на порядки; без неё — тихий откат на чистый Python
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover — numba объявлена зависимостью проекта
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        """Минимальный шим: без numba декоратор возвращает функцию как есть."""
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def wrapper(fn):
            return fn

        return wrapper


def _simulate_py(high, low, entry_idx, sl, tp, max_bars):
    """Последовательный цикл. Семантика бара — в simulate_bracket_exits."""
    n = len(high)
    pos = np.zeros(n, dtype=np.float64)
    direction = 0.0
    entry_bar = -1
    cur_sl = 0.0
    cur_tp = 0.0

    for i in range(n):
        # Вариант A: стоп и тейк проверяются и в баре входа. Вход не выделен
        # веткой continue — открылись и в том же баре смотрим его диапазон.
        # Для лонга это консервативно: бар мог сходить ниже стопа ещё до
        # входа, и записывать такую позицию удержанной — систематически
        # занижать убыток. По той же причине не переоткрываемся в баре выхода.
        if direction == 0.0 and entry_idx[i] != 0.0:
            direction = entry_idx[i]
            entry_bar = i
            cur_sl = sl[i]
            cur_tp = tp[i]

        if direction != 0.0:
            pos[i] = direction
            hit_sl = (low[i] <= cur_sl) if direction > 0 else (high[i] >= cur_sl)
            hit_tp = (high[i] >= cur_tp) if direction > 0 else (low[i] <= cur_tp)
            # Консервативно: если бар пробил и стоп, и тейк — считаем стоп.
            if hit_sl or (i - entry_bar) >= max_bars:
                direction = 0.0
                pos[i] = 0.0
            elif hit_tp:
                direction = 0.0
                pos[i] = 0.0

    return pos


if HAS_NUMBA:
    _simulate = njit(cache=True)(_simulate_py)
else:  # pragma: no cover — с установленной numba ветка недостижима в тестах
    _simulate = _simulate_py


def simulate_bracket_exits(bars: pd.DataFrame, entries: pd.Series,
                           sl_price: pd.Series, tp_price: pd.Series,
                           max_bars: int = 500) -> pd.Series:
    """Позиция по барам для входов с фиксированными стопом и тейком.

    entries: 0 — нет входа, +1 — лонг, −1 — шорт. Допускается bool (→ +1).
    Вход исполняется в баре сигнала, и стоп/тейк проверяются в том же баре
    (вариант A): если диапазон бара входа уже пробил стоп, позиция в этом
    баре равна 0, а не ±1 — иначе убыток такого бара систематически
    занижался бы. Если бар пробил и стоп, и тейк, приоритет у стопа.
    Бар выхода не переоткрывается, даже если на нём есть сигнал.

    NaN в уровнях заполняется как −inf для стопа и +inf для тейка: для лонга
    это «нет уровня», но у шорта незаданный стоп из-за этого сработает в баре
    входа — передавайте конечные значения.
    """
    n = len(bars)
    high = bars["high"].astype("float64").to_numpy()
    low = bars["low"].astype("float64").to_numpy()

    raw = pd.Series(entries)
    if raw.dtype == bool:
        entry_idx = raw.astype("float64").to_numpy()
    else:
        entry_idx = raw.astype("float64").fillna(0.0).to_numpy()

    sl = pd.Series(sl_price).astype("float64").fillna(-np.inf).to_numpy()
    tp = pd.Series(tp_price).astype("float64").fillna(np.inf).to_numpy()

    pos = _simulate(high, low, entry_idx, sl, tp, int(max_bars))
    return pd.Series(pos, index=bars.index, name="position")


def atr_brackets(close: pd.Series, atr: pd.Series, direction: int,
                 sl_atr: float, tp_atr: float) -> tuple[pd.Series, pd.Series]:
    """Цены стопа и тейка на основе ATR. direction: +1 лонг, −1 шорт."""
    if direction not in (1, -1):
        raise ValueError("direction должен быть +1 или −1")
    c = pd.Series(close).astype("float64")
    a = pd.Series(atr).astype("float64")
    sl = c - direction * sl_atr * a
    tp = c + direction * tp_atr * a
    return sl.rename("sl"), tp.rename("tp")
