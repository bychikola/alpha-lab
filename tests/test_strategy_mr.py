import math

import numpy as np
import pandas as pd
import pytest
from alpha_lab.causality import assert_strategy_is_causal
from fixtures.synthetic import ou_bars, random_walk_bars

from alpha_lab.data.quality import check_bars
from alpha_lab.features.price import zscore
from alpha_lab.strategies import mean_reversion as mr_module
from alpha_lab.strategies.base import (
    DEFAULT_HISTORY_BARS, Strategy, build_strategy, history_bars_of,
)
from alpha_lab.strategies.mean_reversion import (
    ATR_DECAY_TOLERANCE, MeanReversionStrategy, atr_decay_bars,
)


def test_position_is_bounded():
    bars = ou_bars(n=3000, theta=0.05, sigma=1.0, seed=11)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    assert pos.between(-1.0, 1.0).all()
    assert len(pos) == len(bars)


def test_no_lookahead_position_depends_only_on_past():
    """Изменение будущих баров не должно менять прошлые позиции.

    Проверку ведёт общий harness (alpha_lab.causality): он сравнивает
    generate(bars.iloc[:k]) с generate(bars).iloc[:k] в нескольких точках
    усечения и падает с русским сообщением о первом расхождении. Дублировать
    сравнение здесь не нужно — harness и есть контракт.
    """
    bars = ou_bars(n=2000, seed=12)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    assert_strategy_is_causal(s, bars)


def test_no_lookahead_with_half_life_filter():
    """Фильтр полужизни тоже обязан быть причинным: окно [i-w, i) — только прошлое.

    Та же проверка harness'ом, но с включённым фильтром: если бы _half_life_ok
    заглядывал в текущий или будущий бар, префикс позиций изменился бы при
    усечении ряда. Это отдельный (опциональный) путь кода, поэтому он покрыт
    своим тестом, а не только общим случаем выше.
    """
    bars = ou_bars(n=1200, seed=12)
    s = MeanReversionStrategy({
        "window": 20, "k": 2.0, "use_hl_filter": True,
        "hl_window": 100, "hl_min": 0.5, "hl_max": 50.0,
    })

    assert_strategy_is_causal(s, bars)


def test_generates_trades_on_mean_reverting_series():
    bars = ou_bars(n=5000, theta=0.10, sigma=1.0, seed=13)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    assert (pos != 0).sum() > 50      # на возвращающемся ряде входы обязаны быть


def test_random_walk_bars_ohlc_invariants_and_quality_gate():
    """Бар случайного блуждания физически возможен и проходит гейт качества.

    Прежняя конструкция теста подменяла close у ou_bars и оставляла high/low
    от OU-ряда: на 4868 из 5000 баров close выходил за [low, high], ATR был
    раздут до ~33, а метрики стратегии — артефактом чужого конверта.
    """
    bars = random_walk_bars(n=5000)

    assert (bars["low"] <= bars["open"]).all(), "low > open"
    assert (bars["open"] <= bars["high"]).all(), "open > high"
    assert (bars["low"] <= bars["close"]).all(), "low > close"
    assert (bars["close"] <= bars["high"]).all(), "close > high"
    assert (bars["high"] >= bars["low"]).all(), "high < low"

    report = check_bars(random_walk_bars(n=500), "1m")
    assert report.is_clean, report.summary()


def test_random_walk_is_traded_not_avoided():
    """Опровержение посылки «на случайном блуждании z-скор редко даёт входы».

    Замер на физически согласованных барах (окно 20, k=2, max_bars=500):
      * сырой сигнал |z| >= 2 — 11.6% баров против 9.3% на OU(theta=0.10,
        seed=13): блуждание даёт входы не реже, а чаще;
      * 96 входов, среднее удержание 41.5 бара, 68/27/0 стоп/тейк/тайм-стоп
        (+1 сделка открыта в конце), позиция занята 77.1% баров.
    Голый z-порог без фильтров Pine (режим волатильности, ADX, RSI, объём)
    не «редко торгует» на трендовом ряде. Проверяем измеренную картину, а не
    ложную интуицию: входы есть в обе стороны и позиция занята больше
    половины баров.
    """
    bars = random_walk_bars(n=5000, seed=14)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    ou = ou_bars(n=5000, theta=0.10, sigma=1.0, seed=13)
    rw_rate = (zscore(bars["close"], 20).abs() >= 2.0).iloc[19:].mean()
    ou_rate = (zscore(ou["close"], 20).abs() >= 2.0).iloc[19:].mean()
    assert rw_rate >= ou_rate                # посылка «сигналов мало» неверна

    assert (pos > 0).any() and (pos < 0).any()   # входы в обе стороны
    assert (pos != 0).mean() > 0.5               # и позиция занята больше половины баров


def test_zero_signals_during_warmup():
    bars = ou_bars(n=500, seed=15)
    s = MeanReversionStrategy({"window": 100, "k": 2.0})

    pos = s.generate(bars)

    assert (pos.iloc[:100] == 0).all()

    # Граница прогрева z-скора: первые window-1 = 99 баров — структурные нули
    # («сигнала нет»), бар 99 — первый настоящий z. На нём сигнала нет:
    # z = 0.8836 < k = 2.0, поэтому первые 100 позиций нулевые по существу,
    # а не из-за округления или случайного seed.
    z = zscore(bars["close"], 100)
    assert z.iloc[98] == 0.0
    assert abs(z.iloc[99]) < 2.0


def test_no_entry_during_atr_warmup():
    """Прогрев ATR — такая же явная граница входа, как и прогрев z-скора.

    seed=1: z-скор (окно 20) даёт сигналы уже на барах 24-26, но ATR(50)
    конечен только с бара 49. Вход на баре с NaN-уровнями запрещён, поэтому
    до бара 49 позиций нет; после границы стратегия обязана снова торговать —
    иначе тест не отличал бы запрет прогрева от мёртвой стратегии.
    """
    bars = ou_bars(n=300, seed=1)
    s = MeanReversionStrategy({"window": 20, "k": 2.0, "atr_len": 50})

    z = zscore(bars["close"], 20)
    assert (z.iloc[19:49].abs() >= 2.0).any()   # сигнал до границы существует

    pos = s.generate(bars)

    assert (pos.iloc[:49] == 0).all()           # до atr_len баров входов нет
    assert (pos.iloc[49:] != 0).any()           # после границы входы есть


def test_no_entry_when_atr_brackets_not_finite():
    """Сигнал есть, но уровни ATR ещё NaN — входа быть не может.

    atr_len=200: ATR конечен только с бара 199, а z-скор (окно 20) на
    seed=1 сигналит уже с бара 24. NaN-стоп/тейк означал бы позицию без
    защиты, которую simulate_bracket_exits молча держит до тайм-стопа.
    """
    bars = ou_bars(n=300, seed=1)
    s = MeanReversionStrategy({"window": 20, "k": 2.0, "atr_len": 200})

    z = zscore(bars["close"], 20)
    assert (z.iloc[19:199].abs() >= 2.0).any()  # конечный z-сигнал до бара 199

    pos = s.generate(bars)

    assert (pos.iloc[:199] == 0).all()


def test_nan_brackets_guard_suppresses_entry(monkeypatch):
    """Страховка от нефинитных уровней работает и вне прогрева ATR.

    Штатно после прогрева уровни конечны, поэтому единственный способ
    проверить саму ветку — подменить atr_brackets на NaN. Без страховки
    позиция открылась бы без стопа и тейка и удерживалась бы до тайм-стопа.
    """
    bars = ou_bars(n=2000, seed=12)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    baseline = s.generate(bars)
    assert (baseline != 0).any()            # без патча стратегия торгует

    def nan_brackets(close, atr, direction, sl_atr, tp_atr):
        idx = pd.Series(close).index
        nan = pd.Series(np.nan, index=idx)
        return nan, nan.rename("tp")

    monkeypatch.setattr(mr_module, "atr_brackets", nan_brackets)

    pos = s.generate(bars)

    assert (pos == 0).all()


def test_nan_tp_only_suppresses_entry(monkeypatch):
    """Второй конъюнкт страховки: тейк нефинитен, а стоп конечен.

    Страховка — это `isfinite(sl) & isfinite(tp)`; тест фиксирует вторую
    половину конъюнкции, которую общий NaN-тест выше не проверяет (там оба
    уровня NaN сразу). Сырые сигналы z в обе стороны на этих барах есть —
    иначе тест прошёл бы потому, что сигнала нет вовсе, а не потому, что
    страховка сработала.
    """
    bars = ou_bars(n=2000, seed=12)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    z = zscore(bars["close"], 20)
    assert (z.iloc[19:] <= -2.0).any()      # сырой лонг-сигнал есть
    assert (z.iloc[19:] >= 2.0).any()       # сырой шорт-сигнал есть

    def finite_sl_nan_tp(close, atr, direction, sl_atr, tp_atr):
        c = pd.Series(close).astype("float64")
        a = pd.Series(atr).astype("float64")
        sl = (c - direction * sl_atr * a).rename("sl")
        nan = pd.Series(np.nan, index=c.index, name="tp")
        return sl, nan

    monkeypatch.setattr(mr_module, "atr_brackets", finite_sl_nan_tp)

    pos = s.generate(bars)

    assert (pos == 0).all()


def test_nan_sl_only_suppresses_entry(monkeypatch):
    """Первая половина конъюнкции изолированно: стоп нефинитен, тейк конечен.

    Мутация «проверять только тейк» обязана валить этот тест: без проверки
    конечности стопа позиция открылась бы без защиты и держалась бы до
    тайм-стопа. Сырые сигналы z на этих барах есть (см. тест выше).
    """
    bars = ou_bars(n=2000, seed=12)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    z = zscore(bars["close"], 20)
    assert (z.iloc[19:] <= -2.0).any()
    assert (z.iloc[19:] >= 2.0).any()

    def nan_sl_finite_tp(close, atr, direction, sl_atr, tp_atr):
        c = pd.Series(close).astype("float64")
        a = pd.Series(atr).astype("float64")
        nan = pd.Series(np.nan, index=c.index, name="sl")
        tp = (c + direction * tp_atr * a).rename("tp")
        return nan, tp

    monkeypatch.setattr(mr_module, "atr_brackets", nan_sl_finite_tp)

    pos = s.generate(bars)

    assert (pos == 0).all()


def test_half_life_filter_suppresses_signals():
    bars = ou_bars(n=3000, theta=0.30, sigma=1.0, seed=16)   # быстрый возврат
    strict = MeanReversionStrategy({
        "window": 20, "k": 2.0, "use_hl_filter": True,
        "hl_min": 0.01, "hl_max": 0.5,                        # заведомо узкое окно
    })
    loose = MeanReversionStrategy({"window": 20, "k": 2.0, "use_hl_filter": False})

    assert (strict.generate(bars) != 0).sum() < (loose.generate(bars) != 0).sum()


def test_build_strategy_returns_mr():
    s = build_strategy("mean_reversion", {"window": 10})

    assert isinstance(s, MeanReversionStrategy)


def test_build_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        build_strategy("nope", {})


def test_satisfies_strategy_protocol():
    """MeanReversionStrategy обязан удовлетворять runtime-checkable протоколу."""
    s = MeanReversionStrategy({"window": 10})

    assert isinstance(s, Strategy)
    assert s.name == "mean_reversion"


def test_history_bars_reflects_used_windows():
    """history_bars — максимум всего, от чего зависит решение, а не константа.

    Это z-скор (window), decay-горизонт ATR (НЕ atr_len: рекурсия Уайлдера с
    одним seed помнит бесконечно, см. atr_decay_bars) и — только при
    включённом фильтре — окно полужизни (hl_window; цикл _half_life_ok берёт
    w баров до i). Выключенный hl_window решению не нужен и в требование не
    входит.
    """
    assert MeanReversionStrategy({"window": 20, "atr_len": 14}).history_bars == 94
    assert MeanReversionStrategy({"window": 500, "atr_len": 14}).history_bars == 500
    # Окно z-скора больше decay-горизонта — требование задаёт окно.
    assert MeanReversionStrategy({"window": 200, "atr_len": 14}).history_bars == 200

    huge_atr = MeanReversionStrategy({"window": 20, "atr_len": 50}).history_bars
    assert huge_atr == atr_decay_bars(50)
    assert huge_atr > 50          # прежняя формула max(window, atr_len) занижала

    filtered = MeanReversionStrategy({
        "window": 20, "atr_len": 14, "use_hl_filter": True, "hl_window": 200})
    assert filtered.history_bars == 200

    ignored = MeanReversionStrategy({
        "window": 20, "atr_len": 14, "use_hl_filter": False, "hl_window": 200})
    assert ignored.history_bars == 94


def test_history_bars_follows_atr_decay_formula():
    """Граница истории выводится из рекурсии ATR, а не подбирается литералом.

    RMA(length) — rma[i] = (1 − 1/L)·rma[i−1] + (1/L)·tr[i], поэтому вклад TR
    бара k шагов назад равен (1 − 1/L)^k. Требование k таково, что этот вклад
    уже ниже допуска ATR_DECAY_TOLERANCE; проверяем и саму формулу, и
    монотонность по atr_len (больше длина — длиннее память).
    """
    tol = mr_module.ATR_DECAY_TOLERANCE
    assert 0.0 < tol < 0.01          # допуск действительно малый, не «на глаз»

    previous = 0
    for atr_len in (2, 7, 14, 50, 200):
        k = MeanReversionStrategy(
            {"window": 1, "atr_len": atr_len}).history_bars
        assert k == math.ceil(math.log(tol) / math.log1p(-1.0 / atr_len))
        assert (1.0 - 1.0 / atr_len) ** k <= tol
        assert k > previous          # монотонно растёт с atr_len
        previous = k

    # L=1: RMA совпадает с TR, память ровно один бар — формула не применима.
    assert atr_decay_bars(1) == 1
    assert MeanReversionStrategy({"window": 1, "atr_len": 1}).history_bars == 1


def test_history_bars_default_for_undeclared_strategy_is_conservative():
    """Стратегия без history_bars получает консервативный дефолт, а не 1.

    Единица означала бы «окно не пересекает разрыв» для любой стратегии —
    ровно та ошибка, из-за которой маскировался один бар вместо двадцати.
    """
    class Bare:
        name = "bare"

        def generate(self, bars):
            return pd.Series(0.0, index=bars.index)

    assert history_bars_of(Bare()) == DEFAULT_HISTORY_BARS
    assert DEFAULT_HISTORY_BARS > 1


def test_history_bars_of_rejects_invalid_declaration():
    class Bad:
        history_bars = 0

    with pytest.raises(ValueError, match="history_bars"):
        history_bars_of(Bad())
