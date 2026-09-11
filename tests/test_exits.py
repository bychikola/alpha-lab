import numpy as np
import pandas as pd
import pytest

from alpha_lab.engine.exits import (
    HAS_NUMBA,
    _simulate,
    _simulate_py,
    atr_brackets,
    simulate_bracket_exits,
)


def _bars(highs, lows):
    n = len(highs)
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": mid, "high": highs, "low": lows, "close": mid,
        "volume": 1e6, "quote_volume": 1e9, "trades": 100, "taker_buy_volume": 1e5,
    })


# --- лонг -------------------------------------------------------------------------------


def test_long_take_profit_closes_position():
    """Тейк фиксируется в баре пробоя, а не на следующем.

    Бар 2: high 105 >= tp 104 — тейк пробит, позиция закрывается в этом же
    баре (pos[2] == 0.0). Запись «ещё держим на баре пробоя» завышала бы
    удержание.
    """
    bars = _bars(highs=[100, 101, 105, 106, 107], lows=[100, 100, 101, 102, 103])
    entries = pd.Series([True, False, False, False, False])
    sl = pd.Series([98.0] * 5)
    tp = pd.Series([104.0] * 5)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[0] == 1.0     # вошли (бар 0: 100..100, ничего не пробито)
    assert pos.iloc[1] == 1.0     # high 101 < tp 104 — держим
    assert pos.iloc[2] == 0.0     # high 105 пробил tp 104 — вышли в этом же баре
    assert pos.iloc[3] == 0.0     # после выхода не переоткрываемся


def test_long_stop_loss_closes_position():
    bars = _bars(highs=[100, 101, 101, 101], lows=[100, 99, 97, 96])
    entries = pd.Series([True, False, False, False])
    sl = pd.Series([98.0] * 4)
    tp = pd.Series([110.0] * 4)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[0] == 1.0
    assert pos.iloc[1] == 1.0
    assert pos.iloc[2] == 0.0     # low 97 пробил стоп 98
    assert pos.iloc[3] == 0.0


# --- шорт -------------------------------------------------------------------------------


def test_short_take_profit():
    """Шорт задаётся entry = -1.0: bool True — это всегда лонг (+1)."""
    bars = _bars(highs=[100, 100, 100, 100], lows=[100, 99, 94, 93])
    entries = pd.Series([0.0, -1.0, 0.0, 0.0])
    sl = pd.Series([102.0] * 4)
    tp = pd.Series([95.0] * 4)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[1] == -1.0
    assert pos.iloc[2] == 0.0     # low 94 <= tp 95 — тейк взят
    assert pos.iloc[3] == 0.0


def test_short_stop_loss():
    bars = _bars(highs=[100, 100, 103, 104], lows=[100, 99, 99, 98])
    entries = pd.Series([0.0, -1.0, 0.0, 0.0])
    sl = pd.Series([102.0] * 4)
    tp = pd.Series([90.0] * 4)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[1] == -1.0
    assert pos.iloc[2] == 0.0     # high 103 >= sl 102
    assert pos.iloc[3] == 0.0


# --- границы удержания ------------------------------------------------------------------


def test_max_bars_forces_exit():
    """Тайм-стоп: (i - entry_bar) >= max_bars закрывает позицию в этом баре.

    Вход в баре 0, max_bars=3 — держим бары 0..2, выходим в баре 3.
    """
    bars = _bars(highs=[100] * 10, lows=[100] * 10)
    entries = pd.Series([True] + [False] * 9)
    sl = pd.Series([50.0] * 10)
    tp = pd.Series([150.0] * 10)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=3)

    assert pos.iloc[2] == 1.0     # (2 - 0) < 3 — ещё держим
    assert pos.iloc[3] == 0.0     # (3 - 0) >= 3 — вышли по времени
    assert pos.iloc[4] == 0.0


def test_no_entry_means_no_position():
    bars = _bars(highs=[100] * 5, lows=[100] * 5)
    entries = pd.Series([False] * 5)

    pos = simulate_bracket_exits(bars, entries, pd.Series([90.0] * 5),
                                 pd.Series([110.0] * 5), max_bars=5)

    assert (pos == 0.0).all()


def test_position_length_matches_bars():
    bars = _bars(highs=[100] * 20, lows=[100] * 20)
    entries = pd.Series([True] + [False] * 19)

    pos = simulate_bracket_exits(bars, entries, pd.Series([90.0] * 20),
                                 pd.Series([110.0] * 20), max_bars=5)

    assert len(pos) == 20


# --- приоритет и вариант A (бар входа) --------------------------------------------------


def test_stop_takes_priority_when_bar_hits_both():
    """Если бар пробил и стоп, и тейк — консервативно считаем стоп."""
    bars = _bars(highs=[100, 120], lows=[100, 80])
    entries = pd.Series([True, False])
    sl = pd.Series([90.0] * 2)
    tp = pd.Series([110.0] * 2)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=5)

    assert pos.iloc[1] == 0.0


@pytest.mark.parametrize(
    "highs, lows, entry, sl, tp",
    [
        ([100.0, 101.0], [100.0, 95.0], 1.0, 98.0, 110.0),    # лонг: low 95 <= sl 98
        ([100.0, 105.0], [100.0, 99.0], -1.0, 102.0, 90.0),   # шорт: high 105 >= sl 102
    ],
    ids=["long_stop", "short_stop"],
)
def test_stop_breach_on_entry_bar_produces_flat_position(highs, lows, entry, sl, tp):
    """Вариант A: стоп проверяется и в баре входа.

    Бар 1 даёт сигнал на вход, но его же диапазон уже пробил стоп. Записать
    pos[1] = ±1 значило бы удержать позицию через бар, в котором стоп был
    пробит до входа, — систематическое занижение убытков. Ожидаем 0.0.
    """
    bars = _bars(highs=highs, lows=lows)
    entries = pd.Series([0.0, entry])

    pos = simulate_bracket_exits(bars, entries, pd.Series([sl, sl]),
                                 pd.Series([tp, tp]), max_bars=10)

    assert pos.iloc[0] == 0.0     # до бара 1 входа не было
    assert pos.iloc[1] == 0.0     # вход и пробой стопа в одном баре


def test_take_profit_on_entry_bar_also_flattens():
    """Вариант A проверяет в баре входа и тейк, а не только стоп."""
    bars = _bars(highs=[100.0, 105.0], lows=[100.0, 99.0])
    entries = pd.Series([False, True])

    pos = simulate_bracket_exits(bars, entries, pd.Series([98.0, 98.0]),
                                 pd.Series([104.0, 104.0]), max_bars=10)

    assert pos.iloc[1] == 0.0     # high 105 >= tp 104 — цель в баре входа


# --- сквозная ручная сверка -------------------------------------------------------------


def test_hand_derived_position_series():
    """Серия позиций, посчитанная вручную, целиком.

    high = [100, 102, 104, 101, 103, 103, 103, 103, 103, 103]
    low  = [100, 100, 103,  97, 100, 100, 100, 100, 100, 100]
    entry= [  1,   0,   0,   0,   0,  -1,   0,   0,   0,   0]
    sl   = [ 98,  98,  98,  98,   0, 120, 120, 120, 120, 120]
    tp   = [150, 150, 150, 150,   0,  80,  80,  80,  80,  80], max_bars=4

    бар 0: лонг, 100..100 — уровни целы                       ->  1
    бары 1-2: 102..100, 104..103 — держим                     ->  1
    бар 3: low 97 <= sl 98 — стоп (3 < 4, тайм-стоп ни при чём) -> 0
    бар 4: позиции нет, входа нет                             ->  0
    бар 5: шорт, 103..100 — уровни целы                       -> -1
    бары 6-8: 1..3 < 4 — держим                               -> -1
    бар 9: (9 - 5) >= 4 — вышли по времени                    ->  0
    """
    highs = [100.0, 102.0, 104.0, 101.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0]
    lows = [100.0, 100.0, 103.0, 97.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    entries = pd.Series([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    sl = pd.Series([98.0, 98.0, 98.0, 98.0, 0.0, 120.0, 120.0, 120.0, 120.0, 120.0])
    tp = pd.Series([150.0, 150.0, 150.0, 150.0, 0.0, 80.0, 80.0, 80.0, 80.0, 80.0])

    pos = simulate_bracket_exits(_bars(highs, lows), entries, sl, tp, max_bars=4)
    expected = [1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0]

    assert pos.tolist() == expected


# --- numba и чистый Python --------------------------------------------------------------


def test_numba_and_python_loops_agree():
    """Компилированный цикл и чистый Python обязаны давать один результат.

    Тест держит ветку отката не мёртвой: без numba _simulate — это _simulate_py,
    и здесь проверяется живая реализация; с numba — доказывается, что
    компиляция не меняет семантику (стоп, шорт, тайм-стоп, бар входа).
    """
    cases = [
        # стоп, шорт и тайм-стоп
        (
            np.array([100.0, 102.0, 104.0, 101.0, 103.0, 103.0, 103.0, 103.0,
                      103.0, 103.0]),
            np.array([100.0, 100.0, 103.0, 97.0, 100.0, 100.0, 100.0, 100.0,
                      100.0, 100.0]),
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([98.0, 98.0, 98.0, 98.0, 0.0, 120.0, 120.0, 120.0, 120.0,
                      120.0]),
            np.array([150.0, 150.0, 150.0, 150.0, 0.0, 80.0, 80.0, 80.0, 80.0,
                      80.0]),
            4,
        ),
        # пробой стопа в баре входа (вариант A)
        (
            np.array([100.0, 101.0]),
            np.array([100.0, 95.0]),
            np.array([0.0, 1.0]),
            np.array([98.0, 98.0]),
            np.array([110.0, 110.0]),
            10,
        ),
    ]

    for high, low, entry, sl, tp, max_bars in cases:
        compiled = _simulate(high, low, entry, sl, tp, max_bars)
        python = _simulate_py(high, low, entry, sl, tp, max_bars)
        np.testing.assert_array_equal(compiled, python)

    if HAS_NUMBA:
        assert _simulate is not _simulate_py   # боевой путь действительно скомпилирован
    else:
        assert _simulate is _simulate_py


# --- уровни ATR -------------------------------------------------------------------------


def test_atr_brackets_long_and_short():
    close = pd.Series([100.0, 100.0])
    a = pd.Series([2.0, 2.0])

    sl_l, tp_l = atr_brackets(close, a, direction=1, sl_atr=2.0, tp_atr=6.0)
    sl_s, tp_s = atr_brackets(close, a, direction=-1, sl_atr=2.0, tp_atr=6.0)

    assert sl_l.iloc[0] == pytest.approx(96.0)
    assert tp_l.iloc[0] == pytest.approx(112.0)
    assert sl_s.iloc[0] == pytest.approx(104.0)
    assert tp_s.iloc[0] == pytest.approx(88.0)
