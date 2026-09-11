"""Путь-зависимая симуляция выходов по стопу и тейку.

Векторизовать нельзя: результат каждого бара зависит от того, были ли мы в
позиции на предыдущем. Цикл компилируется numba; при недоступности numba тот
же алгоритм исполняется чистым Python (медленнее). Чисто-Python функция
`_simulate_py` тестируется напрямую, поэтому ветка отката не мёртвая.
"""
from __future__ import annotations

import math

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
            # Нефинитный уровень (NaN или ±inf) = «этой защиты нет». Сентинел
            # выбирается по направлению, иначе проверка шорта ложно сработает на
            # баре входа: при -inf-стопе high >= -inf истинно всегда, при
            # +inf-тейке — low <= +inf. math.isfinite поддерживается numba.
            cur_sl = sl[i] if math.isfinite(sl[i]) else -direction * np.inf
            cur_tp = tp[i] if math.isfinite(tp[i]) else direction * np.inf

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


def _checked_float_array(values, n, name, fill):
    """float64-массив длины n с заполненными NaN; иначе ValueError.

    Проверка длины обязана жить вне numba: скомпилированный цикл собирается с
    boundscheck=False, и короткий ряд читал бы чужую память вместо IndexError.
    """
    arr = pd.Series(values).astype("float64").fillna(fill).to_numpy()
    if len(arr) != n:
        raise ValueError(
            f"{name}: длина {len(arr)} не совпадает с длиной bars ({n})"
        )
    return arr


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

    Нефинитный уровень (NaN или ±inf) означает «этой защиты нет»: при входе
    стоп заменяется на -direction * inf, тейк — на +direction * inf. Знак
    выбран по направлению, поэтому проверка не срабатывает ложно ни для лонга,
    ни для шорта (у шорта отсутствующий стоп не закрывает позицию в баре
    входа). NaN — штатное состояние уровней ATR на прогреве, не экзотика.

    Ряды entries, sl_price и tp_price обязаны совпадать по длине с bars, а
    entries — содержать только -1.0/0.0/+1.0 (bool приводится к 1.0/0.0);
    иначе ValueError. Проверки выполняются до компилируемого цикла.
    """
    n = len(bars)
    high = bars["high"].astype("float64").to_numpy()
    low = bars["low"].astype("float64").to_numpy()

    entry_idx = _checked_float_array(entries, n, "entries", 0.0)
    valid_entry = np.isin(entry_idx, (-1.0, 0.0, 1.0))
    if not valid_entry.all():
        bad = float(entry_idx[~valid_entry][0])
        raise ValueError(
            f"entries: недопустимое значение {bad}; "
            "разрешены только -1.0, 0.0, +1.0"
        )

    sl = _checked_float_array(sl_price, n, "sl_price", -np.inf)
    tp = _checked_float_array(tp_price, n, "tp_price", np.inf)

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
