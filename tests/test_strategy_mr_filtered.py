"""Фильтрующий вариант MR: ADX, эмпирический квантиль, сжатие волатильности.

Три идеи из чужого индикатора, приведённые к виду, который можно проверить:
не «улучшить модель», а сформулировать гипотезу и дать полигону её убить.

Ключевой инвариант всей затеи: **выключенные фильтры не меняют ничего**.
Без него нельзя было бы утверждать, что разница в вердикте вызвана фильтром,
а не тем, что стратегия — другая программа.
"""
import numpy as np
import pandas as pd
import pytest
from alpha_lab.causality import assert_strategy_is_causal
from fixtures.synthetic import ou_bars, random_walk_bars

from alpha_lab.features.price import zscore
from alpha_lab.features.volatility import adx_decay_bars
from alpha_lab.strategies.base import Strategy, build_strategy, strategy_param_names
from alpha_lab.strategies.mean_reversion import (
    ATR_DECAY_TOLERANCE, MeanReversionStrategy,
)
from alpha_lab.strategies.mean_reversion_filtered import (
    MeanReversionFilteredStrategy,
)

BASE = {"window": 20, "k": 2.0}


def _entry_bars(pos: pd.Series) -> np.ndarray:
    """Бары, на которых позиция открывалась (переход из нуля в ненулевое)."""
    p = pos.to_numpy()
    return np.flatnonzero((p != 0) & (np.r_[0.0, p[:-1]] == 0))


# ═════════════════════ Инвариант «выключено = не существует» ═══════════════


def test_filters_off_is_bit_identical_to_base():
    """Ноль фильтров — ровно базовая модель, а не «похожая».

    Это то, на чём стоит вся постановка эксперимента: если выключенный фильтр
    всё равно что-то меняет, разница в вердикте перестаёт быть свойством
    фильтра.
    """
    bars = ou_bars(n=4000, theta=0.05, seed=23)

    base = MeanReversionStrategy(dict(BASE)).generate(bars)
    filtered = MeanReversionFilteredStrategy(dict(BASE)).generate(bars)

    assert filtered.equals(base)


def test_filters_off_is_identical_on_a_random_walk():
    """Тот же инвариант на трендовом ряде: совпадение не свойство OU-фикстуры."""
    bars = random_walk_bars(n=4000, seed=24)

    base = MeanReversionStrategy(dict(BASE)).generate(bars)
    filtered = MeanReversionFilteredStrategy(dict(BASE)).generate(bars)

    assert filtered.equals(base)


# ═════════════════════════ ADX: фильтр режима ══════════════════════════════


def test_adx_min_above_the_range_blocks_everything():
    """ADX ограничен сотней, поэтому `adx > 101` не выполняется нигде.

    Проверка не «фильтр что-то отсекает», а «фильтр вообще подключён к
    решению»: если бы параметр не читался, тест упал бы.
    """
    bars = ou_bars(n=3000, seed=25)
    s = MeanReversionFilteredStrategy({**BASE, "adx_min": 101.0})

    assert (s.generate(bars) == 0).all()


def test_adx_max_below_zero_blocks_everything():
    bars = ou_bars(n=3000, seed=25)
    s = MeanReversionFilteredStrategy({**BASE, "adx_max": -1.0})

    assert (s.generate(bars) == 0).all()


def test_adx_max_in_range_cuts_but_does_not_kill():
    """Обратная сторона: настоящий порог режет часть, а не всё.

    Без этого теста предыдущие два не отличали бы работающий фильтр от
    стратегии, которая просто ничего не делает.
    """
    bars = ou_bars(n=3000, seed=25)
    base = MeanReversionStrategy(dict(BASE)).generate(bars)
    cut = MeanReversionFilteredStrategy({**BASE, "adx_max": 25.0}).generate(bars)

    assert (cut != 0).any()
    assert not cut.equals(base)


def test_adx_warmup_is_an_entry_boundary():
    """Пока ADX не посчитан, входа нет — прогрев фильтра это граница, а не мелочь.

    ADX заводится на баре 2L−1: до него strength = NaN, сравнение с порогом
    ложно, и вход запрещён. Прогрев сдвигает начало торговли, и это обязано
    быть видно, а не выясняться по разнице в числах.
    """
    period = 20
    warmup = 2 * period - 1                 # ADX конечен начиная с этого бара
    bars = ou_bars(n=800, seed=31)
    base = MeanReversionStrategy(dict(BASE)).generate(bars)
    filtered = MeanReversionFilteredStrategy(
        {**BASE, "adx_period": period, "adx_max": 100.0}).generate(bars)

    # Порог «100» не режет ничего из посчитанного ADX (он ограничен сотней),
    # поэтому единственная разница с базой — прогревочная граница. Проверка
    # непуста только если база успела войти до неё.
    early = _entry_bars(base)
    early = early[early < warmup]
    assert len(early) > 0, "база не входила до прогрева — тест ничего не проверяет"

    opened = _entry_bars(filtered)
    assert len(opened) > 0
    assert (opened >= warmup).all()


# ══════════════════════════ Сжатие волатильности ═══════════════════════════


def test_squeeze_of_zero_blocks_everything():
    """σ > 0 всегда, поэтому `σ < 0 · sma(σ)` не выполняется никогда."""
    bars = ou_bars(n=3000, seed=32)
    s = MeanReversionFilteredStrategy({**BASE, "squeeze_ratio": 0.0})

    assert (s.generate(bars) == 0).all()


def test_squeeze_ratio_in_range_bites_but_does_not_kill():
    bars = ou_bars(n=3000, seed=32)
    base = MeanReversionStrategy(dict(BASE)).generate(bars)
    tight = MeanReversionFilteredStrategy(
        {**BASE, "squeeze_window": 50, "squeeze_ratio": 0.8}).generate(bars)

    assert (tight != 0).any()
    assert not tight.equals(base)


def test_squeeze_warmup_is_an_entry_boundary():
    """σ берётся по окну z-скора, значит сжатию нужно window + squeeze_window − 1.

    Недосчитанное сжатие — запрет входа (сравнение с NaN ложно), а не
    разрешение: пропустить вход дёшево, открыть его по несуществующему
    фильтру — нет.
    """
    window, squeeze_window = 20, 100
    bars = ou_bars(n=800, seed=33)
    s = MeanReversionFilteredStrategy({
        **BASE, "window": window,
        "squeeze_window": squeeze_window, "squeeze_ratio": 0.9})

    opened = _entry_bars(s.generate(bars))

    assert len(opened) > 0
    assert opened[0] >= window + squeeze_window - 2


# ══════════════════════ Эмпирический квантиль вместо k·σ ═══════════════════


def test_quantile_threshold_is_the_declared_empirical_quantile():
    """Порог — квантиль РАСПРЕДЕЛЕНИЯ z, а не k·σ и не смесь того и другого.

    Классическая ошибка этого метода: взять эмпирический 90 %-диапазон и
    сравнить его с теоретическим 2·k·σ, то есть с 4σ = 95.4 %, — тогда порог
    систематически занижен (0.82 даже на идеально нормальных данных).
    Проверяем, что теоретического плеча в пороге нет вообще: значение равно
    квантилю окна [i − w + 1, i], посчитанному независимо через numpy.
    """
    window, quantile_window, coverage = 20, 250, 90.0
    bars = ou_bars(n=2000, seed=27)
    s = MeanReversionFilteredStrategy({
        **BASE, "quantile_window": quantile_window, "coverage": coverage})
    z = zscore(bars["close"], window)

    low, high = s._signal_thresholds(bars, bars["close"].astype("float64"), z)

    i = 500
    chunk = z.iloc[i - quantile_window + 1: i + 1].to_numpy()
    assert low.iloc[i] == pytest.approx(np.quantile(chunk, 0.05))
    assert high.iloc[i] == pytest.approx(np.quantile(chunk, 0.95))


def test_quantile_coverage_orders_the_thresholds():
    """Покрытие 90 % обязано давать порог ближе к нулю, чем 99 %.

    Монотонность квантиля по вероятности — свойство определения, а не выборки.
    """
    bars = ou_bars(n=2000, seed=34)
    close = bars["close"].astype("float64")
    z = zscore(close, 20)

    def thresholds(coverage):
        s = MeanReversionFilteredStrategy(
            {**BASE, "quantile_window": 250, "coverage": coverage})
        return s._signal_thresholds(bars, close, z)

    low90, high90 = thresholds(90.0)
    low99, high99 = thresholds(99.0)

    ready = low90.notna() & low99.notna()
    assert ready.any()
    assert (low90[ready] >= low99[ready]).all()
    assert (high90[ready] <= high99[ready]).all()
    assert (low90[ready] > low99[ready]).any()      # не тождество


def test_quantile_ignores_k():
    """В квантильном режиме k не читается — зафиксировано, а не умолчано.

    Порог берётся из распределения; держать k живым параметром значило бы
    считать одну гипотезу дважды под разными config_id и дарить себе лишние
    «попытки» в штрафе DSR за дубликаты.
    """
    bars = ou_bars(n=3000, seed=26)
    narrow = MeanReversionFilteredStrategy(
        {**BASE, "k": 1.0, "quantile_window": 250}).generate(bars)
    wide = MeanReversionFilteredStrategy(
        {**BASE, "k": 5.0, "quantile_window": 250}).generate(bars)

    assert narrow.equals(wide)


def test_quantile_mode_differs_from_base():
    bars = ou_bars(n=3000, seed=26)
    base = MeanReversionStrategy(dict(BASE)).generate(bars)
    q = MeanReversionFilteredStrategy(
        {**BASE, "quantile_window": 250}).generate(bars)

    assert (q != 0).any()
    assert not q.equals(base)


def test_quantile_warmup_is_an_entry_boundary():
    """Прогревочные нули z — заглушка «сигнала нет», а не наблюдения.

    Они не должны попадать в эмпирический квантиль: окно, набранное из нулей,
    сжимает порог к нулю, и первые бары торговали бы по несуществующему
    распределению. Порог конечен только когда набрано quantile_window
    НАСТОЯЩИХ значений z, то есть с бара window + quantile_window − 2.
    """
    window, quantile_window = 20, 500
    bars = ou_bars(n=2000, seed=28)
    s = MeanReversionFilteredStrategy({
        **BASE, "window": window, "quantile_window": quantile_window})
    base = MeanReversionStrategy(dict(BASE)).generate(bars)

    opened = _entry_bars(s.generate(bars))

    assert len(opened) > 0
    assert opened[0] >= window + quantile_window - 2
    assert opened[0] > _entry_bars(base)[0]


# ══════════════════════════ Причинность и память ═══════════════════════════


def test_causality_in_every_filter_mode():
    """Каждый фильтр причинен: он читает ПРОШЛОЕ окно, а не будущее.

    Harness усечения — единственная работающая защита от подглядывания
    (spec 8.1). Режимы проверяются раздельно: это разные ветки кода, и
    причинность одной не доказывает причинности другой.
    """
    bars = ou_bars(n=400, seed=29)
    modes = {
        "off": {},
        "adx_max": {"adx_period": 10, "adx_max": 30.0},
        "adx_min": {"adx_period": 10, "adx_min": 15.0},
        "adx_band": {"adx_period": 10, "adx_min": 10.0, "adx_max": 40.0},
        "quantile": {"quantile_window": 100, "coverage": 95.0},
        "squeeze": {"squeeze_window": 20, "squeeze_ratio": 0.9},
        "all": {"adx_period": 10, "adx_max": 40.0, "quantile_window": 100,
                "squeeze_window": 20, "squeeze_ratio": 1.2},
    }

    for label, extra in modes.items():
        strategy = MeanReversionFilteredStrategy({**BASE, **extra})
        cuts = assert_strategy_is_causal(strategy, bars)
        assert cuts >= 300, f"режим {label}: покрытие усечений {cuts}"


def test_history_bars_extends_with_each_switched_on_filter():
    """Память растёт только от ВКЛЮЧЁННЫХ фильтров — как и у hl_window.

    Ненастроенный фильтр решению не нужен и в требование не входит: иначе
    диагностика памяти показывала бы завышенное число у стратегии, которая
    ничего не фильтрует.
    """
    off = MeanReversionFilteredStrategy({"window": 20, "atr_len": 14})
    assert off.history_bars == 94

    # Задан period, но фильтр не включён — требование не меняется.
    dormant = MeanReversionFilteredStrategy(
        {"window": 20, "atr_len": 14, "adx_period": 50})
    assert dormant.history_bars == 94

    adx_on = MeanReversionFilteredStrategy(
        {"window": 20, "atr_len": 14, "adx_period": 14, "adx_max": 30.0})
    assert adx_on.history_bars == adx_decay_bars(14, ATR_DECAY_TOLERANCE)

    quantile = MeanReversionFilteredStrategy(
        {"window": 20, "atr_len": 14, "quantile_window": 250})
    assert quantile.history_bars == 20 + 250 - 1

    squeeze = MeanReversionFilteredStrategy(
        {"window": 20, "atr_len": 14, "squeeze_window": 100,
         "squeeze_ratio": 0.8})
    assert squeeze.history_bars == 20 + 100 - 1


# ═════════════════════════════ Регистрация ═════════════════════════════════


def test_build_strategy_returns_filtered():
    s = build_strategy("mean_reversion_filtered", {"quantile_window": 250})

    assert isinstance(s, MeanReversionFilteredStrategy)
    assert isinstance(s, Strategy)
    assert s.name == "mean_reversion_filtered"


def test_param_names_cover_every_filter():
    """Сетка спрашивает пространство параметров у класса: пропущенное имя
    означало бы, что ось по нему отвергается как опечатка."""
    names = strategy_param_names("mean_reversion_filtered")

    for name in ("adx_period", "adx_max", "adx_min", "quantile_window",
                 "coverage", "squeeze_window", "squeeze_ratio"):
        assert name in names
    assert names >= strategy_param_names("mean_reversion")


def test_positions_stay_bounded_with_every_filter_on():
    bars = ou_bars(n=2000, seed=35)
    s = MeanReversionFilteredStrategy({
        **BASE, "adx_max": 40.0, "adx_min": 5.0, "quantile_window": 150,
        "coverage": 95.0, "squeeze_window": 30, "squeeze_ratio": 1.1})

    pos = s.generate(bars)

    assert len(pos) == len(bars)
    assert pos.index.equals(bars.index)
    assert pos.between(-1.0, 1.0).all()
    assert np.isfinite(pos.to_numpy()).all()
