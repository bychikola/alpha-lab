"""S1: двухногий движок — три базы позиции (net / gross / carry).

Одноногий движок держал один скаляр и применял к нему три вещи: ценовой P&L,
издержки оборота и funding. Для дельта-нейтральной книги это неверно: net ≡ 0
(ценовой P&L ноль), но торгуются ДВЕ ноги (издержки не ноль) и funding
начисляется на ногу перпа (это и есть доход). Здесь проверяются:

* побитовая совместимость дефолтного пути (gross=None, carry=None) с прежним
  движком — литеральные числа сняты с движка ДО изменения;
* раздельная экономика двух ног: ценовой P&L от net, издержки от изменения
  gross, funding от carry;
* валидация новых рядов (длина, нефинитность);
* решение по trade_returns для net ≡ 0;
* протокол Strategy: generate_legs, причинностный harness и путь CLI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.causality import assert_strategy_is_causal
from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost, ZeroCost
from alpha_lab.strategies.base import PositionLegs, TwoLegStrategy


def _bars(close, quote_volume=1e9):
    close = pd.Series(close, dtype="float64")
    n = len(close)
    qv = quote_volume if np.isscalar(quote_volume) else pd.Series(quote_volume)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6, "quote_volume": qv, "trades": 100, "taker_buy_volume": 1e5,
    })


# --- 1. Обратная совместимость: дефолт обязан быть побитово прежним ------------
#
# Числа сняты прогоном ДО изменения движка на фикстуре с ненулевым оборотом
# (входы, выходы, прямой разворот -1 -> +1, дробная позиция 0.5) и ненулевым
# funding. Разворот здесь несущий: |Δ|net|| дал бы 0 вместо заявки 2, поэтому
# литералы фиксируют и его.

_LEGACY_CLOSE = [100., 102., 99., 105., 105., 97., 110., 108., 112., 112., 90., 95., 101., 100.]
_LEGACY_QV = [1e6, 8e5, 1.2e6, 9e5, 7e5, 1.1e6, 6e5, 1.3e6, 9.5e5, 8.8e5, 5e5, 1.4e6, 1e6, 1.1e6]
_LEGACY_POS = [0., 1., 1., 1., 0., -1., -1., 1., 1., 0., 0., .5, .5, 0.]
_LEGACY_FUND = [0., 0., 0., 1e-4, 0., -2e-4, 0., 0., 3e-4, 0., 0., 1e-4, 0., 0.]

_LEGACY_RETURNS = [
    0.0, 0.0, -0.030795098039215693, 0.06050606060606055, 0.0,
    -0.001459090909090909, -0.13623728522336767, 0.018181818181818188,
    0.031426510721247504, 0.0, -0.00255, 0.0, 0.03105394736842104,
    -0.004950495049504955,
]
_LEGACY_GROSS = [
    0.0, 0.0, -0.02941176470588236, 0.06060606060606055, 0.0, -0.0,
    -0.134020618556701, 0.018181818181818188, 0.03703703703703698, 0.0,
    -0.0, 0.0, 0.03157894736842104, -0.004950495049504955,
]
_LEGACY_HELD = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, -1.0, -1.0, 1.0, 1.0, 0.0, 0.0, 0.5, 0.5]
_LEGACY_TURNOVER = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.5, 0.0]
_LEGACY_PRICE_RET = [
    0.0, 0.020000000000000018, -0.02941176470588236, 0.06060606060606055,
    0.0, -0.07619047619047614, 0.134020618556701, -0.018181818181818188,
    0.03703703703703698, 0.0, -0.1964285714285714, 0.05555555555555558,
    0.06315789473684208, -0.00990099009900991,
]
_LEGACY_FEE = [0.0, 0.0, 0.0005, 0.0, 0.0, 0.0005, 0.0005, 0.0, 0.001, 0.0, 0.0005, 0.0, 0.00025, 0.0]
_LEGACY_SLIP = [
    0.0, 0.0, 0.0008833333333333334, 0.0, 0.0, 0.000959090909090909,
    0.001716666666666667, 0.0, 0.004310526315789474, 0.0, 0.00205, 0.0,
    0.000275, 0.0,
]
_LEGACY_FUNDING = [0.0, 0.0, 0.0, 0.0001, 0.0, -0.0, -0.0, -0.0, 0.0003, 0.0, 0.0, 0.0, 0.0, 0.0]
_LEGACY_EQUITY = [
    1.0, 1.0, 0.9692049019607843, 1.0278476724985146, 1.0278476724985146,
    1.0263479493036418, 0.8865210909959431, 0.9026396562867783,
    0.9310064711224979, 0.9310064711224979, 0.9286324046211355,
    0.9286324046211355, 0.9574701064388506, 0.952730155416876,
]
_LEGACY_TRADES = [
    0.028251871657753946, -0.12071073019944423, 0.031531773879142244,
    0.026103452318916084,
]


def test_defaults_are_bit_identical_to_legacy_engine():
    """Только positions: результат обязан совпасть с прежним движком бит-в-бит.

    Это главное свойство изменения: двухногий путь не имеет права сдвинуть ни
    одно число одноногих прогонов. Литералы сняты с движка до правки.
    """
    bars = _bars(_LEGACY_CLOSE, quote_volume=_LEGACY_QV)
    positions = pd.Series(_LEGACY_POS, dtype="float64")
    funding = pd.Series(_LEGACY_FUND, dtype="float64")

    res = run_backtest(bars, positions, RealisticCost(), capital=10_000.0,
                       funding_rate=funding)
    trades = trade_returns(res)

    assert res.returns.tolist() == _LEGACY_RETURNS
    assert res.gross_returns.tolist() == _LEGACY_GROSS
    assert res.positions.tolist() == _LEGACY_HELD
    assert res.turnover.tolist() == _LEGACY_TURNOVER
    assert res.price_returns.tolist() == _LEGACY_PRICE_RET
    assert res.costs["fee"].tolist() == _LEGACY_FEE
    assert res.costs["slippage"].tolist() == _LEGACY_SLIP
    assert res.costs["funding"].tolist() == _LEGACY_FUNDING
    assert res.equity.tolist() == _LEGACY_EQUITY
    assert res.total_return == -0.047269844583124
    assert res.max_drawdown == -0.13749759354810986
    assert res.bars == 14
    assert res.cap_hits == 3
    assert res.max_participation_observed == 0.021052631578947368
    assert trades.tolist() == _LEGACY_TRADES
    assert res.cost_totals == {
        "fee": 0.0032500000000000003,
        "slippage": 0.010194617224880383,
        "funding": 0.00039999999999999996,
    }


def test_explicit_none_defaults_match_implicit_defaults():
    """Явные gross_position=None/carry_position=None — тот же побитовый путь."""
    bars = _bars(_LEGACY_CLOSE, quote_volume=_LEGACY_QV)
    positions = pd.Series(_LEGACY_POS, dtype="float64")
    funding = pd.Series(_LEGACY_FUND, dtype="float64")

    implicit = run_backtest(bars, positions, RealisticCost(), capital=10_000.0,
                            funding_rate=funding)
    explicit = run_backtest(bars, positions, RealisticCost(), capital=10_000.0,
                            funding_rate=funding, gross_position=None,
                            carry_position=None)

    assert explicit.returns.tolist() == implicit.returns.tolist()
    assert explicit.costs.to_numpy().tolist() == implicit.costs.to_numpy().tolist()
    assert explicit.turnover.tolist() == implicit.turnover.tolist()
    assert explicit.cap_hits == implicit.cap_hits
    assert (explicit.max_participation_observed
            == implicit.max_participation_observed)


# --- 2. Экономика двух ног: каждая компонента отдельно ------------------------

_TWO_LEG_COST = RealisticCost(taker_fee_bps=10.0, min_slippage_bps=0.0,
                              impact_coef=0.0)


def test_two_leg_economics_components_separately():
    """net ≡ 0, gross ≡ 2, carry ≡ −1: ценовой P&L ноль, издержки на две ноги,
    funding — доход.

    Цена стоит (close = 100), поэтому gross_returns обязаны быть ровно нулём.
    Ставки на баров 0–1 ненулевые, но held carry там ещё 0 (сдвиг на бар) —
    funding обязан их игнорировать. Вход и выход торгуют по 2 единицы каждая,
    хотя net не меняется вовсе: издержки идут от gross, funding — от carry.
    """
    n = 6
    bars = _bars([100.0] * n)
    net = pd.Series(0.0, index=bars.index)
    gross = pd.Series([0., 2., 2., 2., 0., 0.], index=bars.index)
    carry = pd.Series([0., -1., -1., -1., 0., 0.], index=bars.index)
    rate = pd.Series([0.5, 0.5, 1e-3, 2e-3, 3e-3, 0.0], index=bars.index)

    res = run_backtest(bars, net, _TWO_LEG_COST, capital=10_000.0,
                       funding_rate=rate, gross_position=gross,
                       carry_position=carry)

    # Ценовой P&L — строго ноль: net ≡ 0.
    assert res.gross_returns.tolist() == [0.0] * n
    assert res.positions.tolist() == [0.0] * n

    # Оборот идёт от изменения gross: вход 0 -> 2 и выход 2 -> 0.
    assert res.turnover.tolist() == [0.0, 0.0, 2.0, 0.0, 0.0, 2.0]

    # Издержки: комиссия на обе ноги (2 единицы), проскальзывания нет
    # (min_slippage=0, impact=0).
    assert res.costs["fee"].tolist() == pytest.approx(
        [0.0, 0.0, 0.002, 0.0, 0.0, 0.002], rel=0, abs=0)
    assert res.costs["slippage"].tolist() == [0.0] * n

    # Funding — от carry (не от net): held carry = [0,0,-1,-1,-1,0].
    assert res.costs["funding"].tolist() == pytest.approx(
        [0.0, 0.0, -1e-3, -2e-3, -3e-3, 0.0], rel=0, abs=0)
    # Доход равен сумме ставок на удерживаемом carry: 1e-3 + 2e-3 + 3e-3.
    # Ставки баров 0–1 (0.5) в сумму не входят: carry там ещё не удерживался.
    funding_income = -float(res.costs["funding"].sum())
    assert funding_income == pytest.approx(6e-3, rel=1e-12)
    assert funding_income == pytest.approx(
        float(rate.iloc[2:5].sum()) * 1.0, rel=1e-12)  # |carry| = 1

    # Итог: funding-доход минус две пары комиссий (0.002 + 0.002).
    assert float(res.returns.sum()) == pytest.approx(6e-3 - 4e-3, rel=1e-12)
    # total_return — произведение (1 + r), а не сумма: второй порядок ~4e-6.
    assert res.total_return == pytest.approx(6e-3 - 4e-3, abs=1e-5)

    # Ёмкость — от торгуемого ноционала gross: заявка 2 * capital.
    assert res.max_participation_observed == pytest.approx(2e-5, rel=1e-12)
    assert res.cap_hits == 0


def test_cost_basis_follows_gross_when_net_changes():
    """net меняется, gross стоит — издержки обязаны молчать (следуют gross).

    net_held: [0, 0, 0.5, 1, 0.5, 0.75, 0.25] — меняется почти каждый бар,
    gross_held: [0, 0, 1, 1, 1, 1, 0] — меняется только на входе и выходе.
    Если бы комиссия шла от net, бар 4 (net 1 -> 0.5) стоил бы 0.5 единицы.
    """
    n = 7
    bars = _bars([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0])
    net = pd.Series([0., 0.5, 1.0, 0.5, 0.75, 0.25, 0.0], index=bars.index)
    gross = pd.Series([0., 1., 1., 1., 1., 0., 0.], index=bars.index)
    carry = pd.Series(0.0, index=bars.index)      # вторая нога не торгуется

    res = run_backtest(bars, net, _TWO_LEG_COST, capital=10_000.0,
                       gross_position=gross, carry_position=carry)

    assert res.turnover.tolist() == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert res.costs["fee"].tolist() == pytest.approx(
        [0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3], rel=0, abs=0)
    # Бар 4: net изменился на 0.5, gross — нет; комиссии нет.
    assert res.costs["fee"].iloc[4] == 0.0
    # Ценовой P&L при этом живёт на net — он не обнулён.
    assert any(v != 0.0 for v in res.gross_returns.tolist())


def test_cost_basis_gross_profile_charges_same_regardless_of_net():
    """Один и тот же gross даёт те же издержки при любом net — база одна."""
    n = 7
    bars = _bars([100.0] * n)
    gross = pd.Series([0., 1., 1., 1., 1., 0., 0.], index=bars.index)
    net_a = pd.Series([0., 0.5, 1.0, 0.5, 0.75, 0.25, 0.0], index=bars.index)
    net_b = pd.Series(0.0, index=bars.index)

    a = run_backtest(bars, net_a, _TWO_LEG_COST, capital=10_000.0,
                     gross_position=gross)
    b = run_backtest(bars, net_b, _TWO_LEG_COST, capital=10_000.0,
                     gross_position=gross)

    assert a.costs["fee"].tolist() == b.costs["fee"].tolist()
    assert a.turnover.tolist() == b.turnover.tolist()
    assert a.cost_totals["fee"] > 0.0


# --- 3. Валидация новых рядов --------------------------------------------------


def test_gross_position_length_mismatch_raises():
    bars = _bars([100.0] * 4)
    positions = pd.Series([0.0] * 4)
    with pytest.raises(ValueError, match="Длина gross_position"):
        run_backtest(bars, positions, ZeroCost(),
                     gross_position=pd.Series([1.0] * 3))


def test_carry_position_length_mismatch_raises():
    bars = _bars([100.0] * 4)
    positions = pd.Series([0.0] * 4)
    with pytest.raises(ValueError, match="Длина carry_position"):
        run_backtest(bars, positions, ZeroCost(),
                     carry_position=pd.Series([1.0] * 5))


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_gross_position_raises(bad):
    """Молчаливый NaN в gross отравил бы и издержки, и диагностику ёмкости."""
    bars = _bars([100.0] * 4)
    positions = pd.Series([0.0] * 4)
    gross = pd.Series([0.0, 1.0, bad, 1.0])
    with pytest.raises(ValueError, match="gross_position.*нефинитн"):
        run_backtest(bars, positions, ZeroCost(), gross_position=gross)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_carry_position_raises(bad):
    """NaN в carry испортил бы funding-арифметику — это отказ, а не пропуск."""
    bars = _bars([100.0] * 4)
    positions = pd.Series([0.0] * 4)
    carry = pd.Series([0.0, 1.0, bad, 1.0])
    with pytest.raises(ValueError, match="carry_position.*нефинитн"):
        run_backtest(bars, positions, ZeroCost(), carry_position=carry)


# --- 4. trade_returns при net ≡ 0 ---------------------------------------------


def test_trade_returns_empty_for_delta_neutral_net_zero():
    """У дельта-нейтральной книги нет НАПРАВЛЕННЫХ сделок — ряд пуст.

    Это честный ответ, а не потеря данных: trade_returns описывает сделки по
    знаку net, а он нулевой. Издержки и funding двухногой книги не атрибутируются
    ни одной направленной сделке, поэтому инвариант суммы здесь СОЗНАТЕЛЬНО не
    выполняется — min_trades тогда обязан провалить такую стратегию, и это
    консервативно правильно: число независимых ставок в carry-режиме считает
    не число баров, и подменять его выдуманными сделками нельзя.
    """
    n = 8
    bars = _bars([100.0] * n)
    net = pd.Series(0.0, index=bars.index)
    gross = pd.Series([0., 2., 2., 2., 2., 2., 2., 0.], index=bars.index)
    carry = pd.Series([0., -1., -1., -1., -1., -1., -1., 0.], index=bars.index)
    rate = pd.Series([0.0] + [1e-3] * 7, index=bars.index)

    res = run_backtest(bars, net, _TWO_LEG_COST, capital=10_000.0,
                       funding_rate=rate, gross_position=gross,
                       carry_position=carry)
    trades = trade_returns(res)

    assert trades.empty
    assert trades.dtype == np.float64
    assert len(trades) == 0
    # Funding при этом заработан: пустой ряд — не «нет доходности».
    assert float(res.returns.sum()) > 0.0
    # Инвариант сохранения суммы для двухногой книги не действует — явно.
    assert float(trades.sum()) != pytest.approx(float(res.returns.sum()))


# --- 5. Протокол Strategy: generate_legs, harness, CLI ------------------------


class _ConstantCarry(TwoLegStrategy):
    """Дельта-нейтральный сбор: net ≡ 0, gross = 2, carry = −1."""

    name = "test_constant_carry"
    history_bars = 1
    PARAM_NAMES = frozenset()

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        idx = bars.index
        return PositionLegs(
            net=pd.Series(0.0, index=idx),
            gross=pd.Series(2.0, index=idx),
            carry=pd.Series(-1.0, index=idx),
        )


class _LeakyNetLegs(TwoLegStrategy):
    name = "test_leaky_net_legs"
    history_bars = 1
    PARAM_NAMES = frozenset()

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        idx = bars.index
        future = (bars["close"].shift(-1) / bars["close"] - 1.0).fillna(0.0)
        return PositionLegs(
            net=pd.Series(np.sign(future), index=idx),
            gross=pd.Series(np.abs(np.sign(future)), index=idx),
            carry=pd.Series(0.0, index=idx),
        )


class _LeakyCarryLegs(TwoLegStrategy):
    """net и gross причинны, а carry читает следующий бар — harness обязан поймать."""

    name = "test_leaky_carry_legs"
    history_bars = 1
    PARAM_NAMES = frozenset()

    def __init__(self, params: dict | None = None):
        self.params = params or {}

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        idx = bars.index
        future = (bars["close"].shift(-1) / bars["close"] - 1.0).fillna(0.0)
        return PositionLegs(
            net=pd.Series(0.0, index=idx),
            gross=pd.Series(2.0, index=idx),
            carry=pd.Series(-1.0 - future, index=idx),
        )


class _MisalignedLegs(_ConstantCarry):
    """Индексная конвенция: позиция подписана позицией в массиве, а не баром."""

    name = "test_misaligned_legs"

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        legs = super().generate_legs(bars)
        wrong = pd.Index(bars["ts"], name="ts")
        return PositionLegs(net=legs.net.set_axis(wrong),
                            gross=legs.gross.set_axis(wrong),
                            carry=legs.carry.set_axis(wrong))


def _ou_bars(n=300, seed=7):
    from fixtures.synthetic import ou_bars
    return ou_bars(n=n, seed=seed)


def test_two_leg_strategy_generate_derives_net():
    """generate (одноногий интерфейс) обязан вернуть ровно legs.net."""
    bars = _ou_bars(n=50)
    strategy = _ConstantCarry()
    legs = strategy.generate_legs(bars)

    pd.testing.assert_series_equal(strategy.generate(bars), legs.net)
    assert legs.gross.tolist() == [2.0] * len(bars)
    assert legs.carry.tolist() == [-1.0] * len(bars)


def test_causality_harness_covers_generate_legs():
    """Двухногая причинность проверяется, а не остаётся на честном слове."""
    bars = _ou_bars(n=300)
    assert_strategy_is_causal(_ConstantCarry(), bars)


def test_causality_harness_catches_lookahead_in_legs():
    """Утечка в любой из трёх баз обязана валить harness."""
    bars = _ou_bars(n=300)
    for strategy in (_LeakyNetLegs(), _LeakyCarryLegs()):
        with pytest.raises(AssertionError, match="generate_legs"):
            assert_strategy_is_causal(strategy, bars)


def test_causality_harness_enforces_legs_index_convention():
    bars = _ou_bars(n=200)
    with pytest.raises(AssertionError, match="индексную конвенцию"):
        assert_strategy_is_causal(_MisalignedLegs(), bars)


def _loaded_data(bars, funding_rate=None, tradable=None):
    from alpha_lab.cli import LoadedData
    ts_index = pd.Index(bars["ts"].to_numpy(), name="ts")
    if tradable is None:
        tradable = np.ones(len(bars), dtype=bool)
    return LoadedData(
        symbol="BTCUSDT", timeframe="1h", bars=bars, ts_index=ts_index,
        tradable=tradable, dirty=0, gaps=0,
        missing_bars=0, gap_masked=0, data_warnings=(), funding_rate=funding_rate,
        funding_available=funding_rate is not None, funding_events=0,
        funding_matched=0, ppy=8760,
    )


def test_run_config_runs_two_leg_strategy_end_to_end(monkeypatch):
    """CLI обязан уметь двухногий путь: harness → generate_legs → движок.

    Проверяется, что через рабочий путь (run_config) стратегия доезжает до
    движка с gross/carry, funding начисляется как доход, а не теряется.
    """
    import alpha_lab.cli as cli
    import alpha_lab.strategies.base as base
    from alpha_lab.config import Experiment

    monkeypatch.setitem(base.REGISTRY, "test_constant_carry", _ConstantCarry)

    n = 12
    bars = _bars([100.0 + (i % 3) for i in range(n)])
    rate = pd.Series(0.0, index=bars.index)
    rate.iloc[3:9] = 1e-3
    exp = Experiment(
        name="carry_probe", strategy="test_constant_carry", params={},
        timeframe="1h", start="2024-01-01", end=None,
        costs={"taker_fee_bps": 10.0, "min_slippage_bps": 0.0,
               "impact_coef": 0.0},
        validation={},
    )

    outcome = cli.run_config(_loaded_data(bars, funding_rate=rate), exp)

    assert outcome.history == 1
    assert outcome.causality_cuts > 0
    result = outcome.result
    assert result.positions.tolist() == [0.0] * n          # net ≡ 0
    assert result.costs["fee"].sum() > 0.0                 # обе ноги торгуются
    assert result.costs["funding"].sum() < 0.0             # funding — доход
    assert float(result.returns.sum()) == pytest.approx(
        -float(result.costs["fee"].sum())
        - float(result.costs["funding"].sum()), rel=1e-12)


def test_run_config_masks_all_three_legs_on_untradable_bars(monkeypatch):
    """Грязный/разрывный бар обнуляет все три базы: книга закрывается целиком."""
    import alpha_lab.cli as cli
    import alpha_lab.strategies.base as base
    from alpha_lab.config import Experiment

    monkeypatch.setitem(base.REGISTRY, "test_constant_carry", _ConstantCarry)

    n = 10
    bars = _bars([100.0] * n)
    tradable = np.array(
        [True, True, False, True, True, True, True, True, True, True])
    data = _loaded_data(bars, tradable=tradable)
    exp = Experiment(
        name="carry_mask", strategy="test_constant_carry", params={},
        timeframe="1h", start="2024-01-01", end=None,
        costs={"taker_fee_bps": 10.0, "min_slippage_bps": 0.0,
               "impact_coef": 0.0},
        validation={},
    )

    outcome = cli.run_config(data, exp)
    result = outcome.result

    assert result.positions.tolist() == [0.0] * n
    # Бар 2 обнулён: цель по всем базам — 0, позиция выходит на баре 3 и
    # входит заново на баре 4; каждая заявка — две ноги, по 2 единицы.
    assert result.turnover.tolist() == [0.0, 2.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert result.costs["fee"].tolist() == pytest.approx(
        [0.0, 2e-3, 0.0, 2e-3, 2e-3, 0.0, 0.0, 0.0, 0.0, 0.0], rel=1e-15)
