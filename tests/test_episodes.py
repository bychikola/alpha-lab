"""S4: эпизод удержания книги — единица наблюдения для carry-класса.

Эпизод — максимальный непрерывный отрезок баров, где торгуемый ноционал
(`gross_positions`) ненулевой, плюс бар выхода: движок удерживает позицию на
баре t и заряжает закрытие на баре t (turnover = |Δgross|), поэтому бар с
нулевым удержанием после открытого отрезка принадлежит закрываемому эпизоду.
Иначе издержка выхода не попала бы ни в один эпизод и распределение доходностей
эпизодов было бы систематически лучше правды.

Это единица НАБЛЮДЕНИЯ, а не гейт. У дельта-нейтральной книги нет направленных
сделок (`trade_returns` пуст), и эпизод отвечает на вопрос «сколько было
независимых удержаний и что они принесли», но min_trades по нему НЕ считается
и alive не выносится: единица, выбранная ради прохода порога, сдала бы защиту
от самообмана (spec 6.6). Что эпизод НЕ измеряет: независимость (режимы funding
автокоррелированы), хвостовые структурные риски (ликвидация, депег, биржа) и
цену базиса на входе/выходе — они не видны в ценовом ряду и ряде ставок.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.engine.backtest import position_episodes, run_backtest
from alpha_lab.engine.costs import RealisticCost, ZeroCost


def _bars(close, quote_volume=1e9):
    close = pd.Series(close, dtype="float64")
    n = len(close)
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6, "quote_volume": quote_volume, "trades": 100,
        "taker_buy_volume": 1e5,
    })


# --- 1. gross_positions: удерживаемая магнитуда торгуемого ноционала ----------


def test_default_gross_positions_is_abs_held_net():
    """Одноногая книга: торгуемый ноционал — |удержанный net|."""
    bars = _bars([100.0] * 5)
    positions = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0])

    res = run_backtest(bars, positions, ZeroCost())

    held = positions.shift(1).fillna(0.0)
    assert res.gross_positions.tolist() == held.abs().tolist()


def test_explicit_gross_positions_is_held_gross():
    bars = _bars([100.0] * 5)
    net = pd.Series(0.0, index=bars.index)
    gross = pd.Series([0.0, 2.0, 2.0, 0.0, 0.0], index=bars.index)
    carry = pd.Series([0.0, -1.0, -1.0, 0.0, 0.0], index=bars.index)

    res = run_backtest(bars, net, ZeroCost(), gross_position=gross,
                       carry_position=carry)

    assert res.gross_positions.tolist() == [0.0, 0.0, 2.0, 2.0, 0.0]


# --- 2. Форма эпизода: вход, удержание, бар выхода ----------------------------


def test_episode_includes_exit_bar_and_charges_its_cost():
    """Эпизод — отрезок удержания ПЛЮС бар выхода, где закрытие оплачено.

    gross-цель [0,0,2,2,2,0,0] => held [0,0,0,2,2,2,0]: удержание на барах
    3–5, выход заряжается на баре 6 (turnover 2). Без бара 6 комиссия выхода
    не попала бы в эпизод, и его доходность была бы завышена.
    """
    n = 7
    bars = _bars([100.0] * n)
    cost = RealisticCost(taker_fee_bps=10.0, min_slippage_bps=0.0,
                         impact_coef=0.0)
    net = pd.Series(0.0, index=bars.index)
    gross = pd.Series([0.0, 0.0, 2.0, 2.0, 2.0, 0.0, 0.0], index=bars.index)
    carry = pd.Series([0.0, 0.0, -1.0, -1.0, -1.0, 0.0, 0.0], index=bars.index)
    rate = pd.Series([1e-3] * n, index=bars.index)

    res = run_backtest(bars, net, cost, capital=10_000.0, funding_rate=rate,
                       gross_position=gross, carry_position=carry)
    episodes = position_episodes(res)

    assert list(episodes.columns) == [
        "episode", "start", "end", "bars", "price_pnl", "funding_income",
        "fee", "slippage", "net_return",
    ]
    assert len(episodes) == 1
    row = episodes.iloc[0]
    assert row["episode"] == 0
    assert row["bars"] == 4                       # бары 3,4,5 (держали) + 6 (выход)
    # Метки — позиции бара в индексе движка (в рабочем пути RangeIndex, время
    # живёт колонкой ts); Series-строка смешанных типов сравнилась бы не с тем.
    assert episodes["start"].iloc[0] == 3
    assert episodes["end"].iloc[0] == 6
    assert row["price_pnl"] == 0.0                # net ≡ 0
    assert row["funding_income"] == pytest.approx(3e-3, rel=1e-12)
    # Вход (бар 3) и выход (бар 6): по 2 единицы оборота, 10 bps.
    assert row["fee"] == pytest.approx(4e-3, rel=1e-12)
    assert row["slippage"] == 0.0
    # Все издержки и доход эпизода — ровно то, что в result.returns.
    assert row["net_return"] == pytest.approx(float(res.returns.sum()), rel=1e-12)
    assert row["net_return"] == pytest.approx(3e-3 - 4e-3, rel=1e-12)


# --- 3. Несколько эпизодов и сохранение суммы ---------------------------------


def test_episodes_split_on_flat_gap_and_conserve_total_return():
    n = 12
    bars = _bars([100.0] * n)
    cost = RealisticCost(taker_fee_bps=10.0, min_slippage_bps=0.0,
                         impact_coef=0.0)
    gross = pd.Series([0., 2., 2., 0., 0., 2., 2., 2., 0., 0., 2., 0.],
                      index=bars.index)
    carry = -gross / 2.0
    net = pd.Series(0.0, index=bars.index)
    rate = pd.Series(1e-3, index=bars.index)

    res = run_backtest(bars, net, cost, capital=10_000.0, funding_rate=rate,
                       gross_position=gross, carry_position=carry)
    episodes = position_episodes(res)

    assert len(episodes) == 3
    # Ни один бар доходности не потерян и не посчитан дважды.
    assert float(episodes["net_return"].sum()) == pytest.approx(
        float(res.returns.sum()), rel=1e-12)
    # Число баров с ненулевым оборотом = входы + выходы; каждый принадлежит
    # ровно одному эпизоду (бары удержания эпизода — его же).
    assert float(episodes["fee"].sum()) == pytest.approx(
        float(res.costs["fee"].sum()), rel=1e-12)
    assert float(episodes["funding_income"].sum()) == pytest.approx(
        -float(res.costs["funding"].sum()), rel=1e-12)


def test_one_leg_direct_flip_is_pinned_as_single_episode():
    """Одноногая книга при развороте не закрывается — эпизод один.

    Это сознательное отличие от trade_returns, которая режет по знаку net:
    +1 -> -1 — одна заявка на 2 единицы, но книга не возвращалась в ноль.
    Эпизод считает периоды ОТКРЫТОЙ книги, поэтому здесь он один; тест
    закрепляет контракт, чтобы смена трактовки не прошла незамеченной.
    """
    n = 6
    bars = _bars([100.0] * n)
    positions = pd.Series([1.0, 1.0, -1.0, -1.0, 0.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())
    episodes = position_episodes(res)

    # Разворот +1 -> −1 — одна заявка на 2 единицы (бар 3), но книга не
    # возвращалась в ноль, поэтому эпизод один.
    assert res.turnover.tolist() == [0.0, 1.0, 0.0, 2.0, 0.0, 1.0]
    assert len(episodes) == 1
    assert episodes.iloc[0]["bars"] == 5          # бары 1..4 + бар выхода 5
    assert float(episodes["net_return"].sum()) == pytest.approx(
        float(res.returns.sum()), rel=1e-12)


def test_flat_book_has_no_episodes():
    bars = _bars([100.0] * 4)
    positions = pd.Series([0.0] * 4)

    episodes = position_episodes(run_backtest(bars, positions, ZeroCost()))

    assert len(episodes) == 0
    assert list(episodes.columns) == [
        "episode", "start", "end", "bars", "price_pnl", "funding_income",
        "fee", "slippage", "net_return",
    ]


def test_episode_sum_is_total_return_for_funding_strategy_contract():
    """Контракт сохранения суммы на книге с реинвестированием и ставкой.

    Ставка меняет знак, carry ходит 0/-1; сумма доходностей эпизодов обязана
    совпасть с суммой доходностей прогона бит-в-бит по построению (каждый бар
    принадлежит эпизоду или является плоским с нулевой доходностью).
    """
    n = 20
    bars = _bars([100.0 + (i % 3) for i in range(n)])
    gross = pd.Series(
        np.where(np.arange(n) % 7 < 4, 2.0, 0.0), index=bars.index)
    carry = -gross / 2.0
    net = pd.Series(0.0, index=bars.index)
    rate = pd.Series(
        np.where(np.arange(n) % 5 == 0, -1e-3, 2e-3), index=bars.index)

    res = run_backtest(bars, net, ZeroCost(), funding_rate=rate,
                       gross_position=gross, carry_position=carry)
    episodes = position_episodes(res)

    assert len(episodes) >= 2
    assert float(episodes["net_return"].sum()) == pytest.approx(
        float(res.returns.sum()), rel=1e-12)
