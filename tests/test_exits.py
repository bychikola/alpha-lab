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


# --- нефинитные уровни (NaN = защиты нет) ----------------------------------------------


def test_long_nan_stop_held_until_take_profit():
    """NaN-стоп у лонга — защиты нет: провал цены позицию не закрывает.

    Сентинел выбирается по направлению: лонгу подставляется -inf, и
    `low <= -inf` не может сработать.
    """
    bars = _bars(highs=[100.0, 101.0, 102.0, 105.0, 106.0],
                 lows=[100.0, 85.0, 70.0, 100.0, 101.0])
    entries = pd.Series([True, False, False, False, False])
    sl = pd.Series([np.nan] * 5)
    tp = pd.Series([104.0] * 5)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    # low 85 и 70 не стоп; тейк 104 пробит на баре 3 (high 105).
    assert pos.tolist() == [1.0, 1.0, 1.0, 0.0, 0.0]


def test_long_nan_target_never_takes_profit():
    """NaN-тейк у лонга — цели нет: рост high не закрывает позицию.

    До тайм-стопа доходим с позицией, хотя high доходил до 140.
    """
    bars = _bars(highs=[100.0, 120.0, 130.0, 140.0, 100.0],
                 lows=[100.0, 100.0, 100.0, 100.0, 100.0])
    entries = pd.Series([True, False, False, False, False])
    sl = pd.Series([50.0] * 5)
    tp = pd.Series([np.nan] * 5)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=3)

    assert pos.iloc[2] == 1.0     # high 130 — тейка нет, держим
    assert pos.iloc[3] == 0.0     # (3 - 0) >= 3 — только тайм-стоп


def test_short_nan_stop_is_held_not_exited_on_entry():
    """Регрессия: NaN-стоп шорта не должен закрывать позицию в баре входа.

    До фикса шорту подставлялся -inf, и `high >= -inf` было истинно всегда:
    позиция выходила на баре входа, а сигнал молча терялся. Серия позиций
    была [0, 0, 0, 0, 0, 0]; теперь шорт держится до тайм-стопа.
    """
    bars = _bars(highs=[100.0] * 6, lows=[100.0] * 6)
    entries = pd.Series([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    sl = pd.Series([np.nan] * 6)
    tp = pd.Series([80.0] * 6)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=3)

    # Бар 1 — вход, а не выход; (4 - 1) >= 3 — тайм-стоп.
    assert pos.tolist() == [0.0, -1.0, -1.0, -1.0, 0.0, 0.0]


def test_short_nan_target_never_takes_profit():
    """NaN-тейк у шорта — цели нет: low не может её пробить.

    До фикса шорту подставлялся +inf, и `low <= +inf` закрывало позицию на
    баре входа. Стей без стопа (sl=200 не достижим) держится до тайм-стопа.
    """
    bars = _bars(highs=[100.0] * 5, lows=[100.0] * 5)
    entries = pd.Series([0.0, -1.0, 0.0, 0.0, 0.0])
    sl = pd.Series([200.0] * 5)
    tp = pd.Series([np.nan] * 5)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=2)

    assert pos.tolist() == [0.0, -1.0, -1.0, 0.0, 0.0]


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
        # нефинитные уровни: nan-стоп и nan-тейк у шорта
        (
            np.array([100.0] * 5),
            np.array([100.0] * 5),
            np.array([0.0, -1.0, 0.0, 0.0, 0.0]),
            np.array([np.nan] * 5),
            np.array([np.nan] * 5),
            2,
        ),
        # нефинитный стоп у лонга при провале цены
        (
            np.array([100.0, 130.0, 90.0, 100.0]),
            np.array([100.0, 100.0, 90.0, 100.0]),
            np.array([1.0, 0.0, 0.0, 0.0]),
            np.array([np.nan] * 4),
            np.array([200.0] * 4),
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


# --- валидация входов (до jit) ----------------------------------------------------------


@pytest.mark.parametrize("bad_field", ["entries", "sl_price", "tp_price"])
def test_length_mismatch_raises_value_error(bad_field):
    """Короткий ряд — ValueError, а не чтение за границей массива в numba.

    У скомпилированного цикла boundscheck=False: без проверки короткий ряд
    дал бы молчаливый мусор вместо IndexError чисто-Python пути.
    """
    bars = _bars(highs=[100.0] * 4, lows=[100.0] * 4)
    kwargs = {
        "entries": pd.Series([True, False, False, False]),
        "sl_price": pd.Series([90.0] * 4),
        "tp_price": pd.Series([110.0] * 4),
    }
    kwargs[bad_field] = kwargs[bad_field].iloc[:3]

    with pytest.raises(ValueError, match="длина"):
        simulate_bracket_exits(bars, **kwargs)


def test_invalid_entry_value_raises():
    """entries = 2.0 нарушает контракт 1/-1/0 и падает до цикла."""
    bars = _bars(highs=[100.0] * 3, lows=[100.0] * 3)
    entries = pd.Series([0.0, 2.0, 0.0])

    with pytest.raises(ValueError, match="entries"):
        simulate_bracket_exits(bars, entries, pd.Series([90.0] * 3),
                               pd.Series([110.0] * 3))


def test_bool_entries_still_supported():
    """bool — валидный вход: True → +1.0, False → 0.0 (регрессия для Task 11)."""
    bars = _bars(highs=[100.0, 105.0], lows=[100.0, 100.0])
    sl = pd.Series([90.0, 90.0])
    tp = pd.Series([104.0, 104.0])

    for entries in (pd.Series([True, False]), np.array([True, False]),
                    [True, False]):
        pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=5)
        assert pos.tolist() == [1.0, 0.0]     # лонг, затем тейк 104 на баре 1


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
