import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_bars, random_walk

from alpha_lab.features.price import zscore
from alpha_lab.strategies import mean_reversion as mr_module
from alpha_lab.strategies.base import Strategy, build_strategy
from alpha_lab.strategies.mean_reversion import MeanReversionStrategy


def test_position_is_bounded():
    bars = ou_bars(n=3000, theta=0.05, sigma=1.0, seed=11)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    assert pos.between(-1.0, 1.0).all()
    assert len(pos) == len(bars)


def test_no_lookahead_position_depends_only_on_past():
    """Изменение будущих баров не должно менять прошлые позиции."""
    bars = ou_bars(n=2000, seed=12)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    full = s.generate(bars)
    truncated = s.generate(bars.iloc[:1000])

    pd.testing.assert_series_equal(full.iloc[:1000], truncated, check_names=False)


def test_no_lookahead_with_half_life_filter():
    """Фильтр полужизни тоже обязан быть причинным: окно [i-w, i) — только прошлое.

    Тот же приём, что и в тесте выше, но с включённым фильтром: если бы
    _half_life_ok заглядывал в текущий или будущий бар, префикс позиций
    изменился бы при усечении ряда.
    """
    bars = ou_bars(n=1200, seed=12)
    s = MeanReversionStrategy({
        "window": 20, "k": 2.0, "use_hl_filter": True,
        "hl_window": 100, "hl_min": 0.5, "hl_max": 50.0,
    })

    full = s.generate(bars)
    truncated = s.generate(bars.iloc[:600])

    pd.testing.assert_series_equal(full.iloc[:600], truncated, check_names=False)


def test_generates_trades_on_mean_reverting_series():
    bars = ou_bars(n=5000, theta=0.10, sigma=1.0, seed=13)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    assert (pos != 0).sum() > 50      # на возвращающемся ряде входы обязаны быть


def test_random_walk_is_traded_not_avoided():
    """Опровержение посылки «на случайном блуждании z-скор редко даёт входы».

    Замер на этом же фикстуре (окно 20, k=2, max_bars=500):
      * сырой сигнал |z| >= 2 — 11.6% баров против 9.3% на OU(theta=0.10,
        seed=13): блуждание даёт входы не реже, а чаще;
      * позиция занята 97.5% баров, 8 из 11 входов закрыл тайм-стоп
        (high/low фикстуры остались от OU-ряда, ATR раздут до ~33, поэтому
        стопы и цели почти не достигаются, а держит позицию max_bars).
    Голый z-порог без фильтров Pine (режим волатильности, ADX, RSI, объём)
    не «редко торгует» на трендовом ряде. Проверяем измеренную картину, а не
    ложную интуицию: входы есть в обе стороны и позиция занята больше
    половины баров.
    """
    bars = ou_bars(n=5000, seed=14)
    bars["close"] = random_walk(n=5000, seed=14).to_numpy()
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
