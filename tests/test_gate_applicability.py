"""S4: применимость гейтов вердикта к книге без направленной экспозиции.

Валидатор остаётся слепым слоем (spec 3): применимость определяется ТОЛЬКО по
данным, которые он и так получает (`positions`, `returns`, `trade_returns`), и
никогда — по имени или классу стратегии. Классификация не является обходом
гейта: неприменимый гейт не «пройден», `alive` по нему не выносится, и ни одна
комбинация прошедших гейтов не даёт `alive=True`, пока есть неприменимый.

Дельта-нейтральная книга: `net ≡ 0` (направленных сделок нет, `trade_returns`
пуст), но P&L ненулевой — его приносят не сделки по цене. Для такой книги:

* `min_trades` неприменим: единицы «направленная сделка» не существует;
  0 сделок — свойство конструкции, а не недостаток свидетельств;
* permutation-тест вырожден: нулевой ряд позиций инвариантен к перестановке,
  p-value ≡ 1.0 по построению и не зависит от дохода стратегии вообще.

Оба факта здесь измеряются, а не провозглашаются.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.validation.significance import permutation_pvalue
from alpha_lab.validation.validator import validate


def _non_directional(n=5000, income=1e-4, noise=2e-4, seed=41):
    """Книга без ценовой экспозиции с ненулевым P&L (аналог carry-дохода).

    Доходности не константа: у константы нулевая дисперсия, и DSR выродился бы
    в 0 (не «прошёл»), а тест перестал бы доказывать анти-обходной инвариант.
    """
    rng = np.random.default_rng(seed)
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    positions = pd.Series(np.zeros(n))
    returns = pd.Series(income + noise * rng.standard_normal(n))
    return returns, price_ret, positions


def _directional(n=20000, seed=42, strength=0.02):
    """Направленная книга с настоящим краем — контроль применимости."""
    rng = np.random.default_rng(seed)
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    sign = np.sign(price_ret.to_numpy())
    random_side = rng.choice([-1.0, 1.0], size=n)
    positions = pd.Series(np.where(rng.random(n) < strength, sign, random_side))
    return positions * price_ret, price_ret, positions


def _verdict(case, trades, **overrides):
    returns, price_ret, positions = case
    kwargs = {"config": {}, "n_trials": 1, "strategy_name": "s",
              "experiment_id": "x", "price_returns": price_ret,
              "positions": positions}
    kwargs.update(overrides)
    return validate(returns, trades, (1 + returns).cumprod(), **kwargs)


def _entry(verdict, gate: str) -> str:
    matches = [text for text in verdict.inapplicable if gate in text]
    assert len(matches) == 1, (
        f"ожидалась ровно одна запись про {gate}: {verdict.inapplicable}")
    return matches[0]


# --- 1. Неприменимость: не проход и не провал ---------------------------------


def test_non_directional_book_marks_min_trades_and_permutation_inapplicable():
    """Гейты называют ограничение, а не выдумывают провал.

    DSR при этом ПРОХОДИТ (0.9999...), сделок 0, p-value вырожден в 1.0.
    Читатель обязан увидеть «неприменим», а не «недостаточно сделок» и не
    «неотличимо от случая»: оба прежних текста были категориальной ошибкой.
    """
    verdict = _verdict(_non_directional(), pd.Series(dtype="float64"))

    assert not verdict.alive
    assert verdict.reasons == (), (
        "неприменимость не имеет права попадать в причины смерти: "
        f"{verdict.reasons}")
    assert len(verdict.inapplicable) == 2
    assert "неприменим" in _entry(verdict, "min_trades")
    assert "неприменим" in _entry(verdict, "permutation")
    # Ни один текст не выдает неприменимость за проход.
    for text in verdict.inapplicable:
        assert "не пройден и не провален" in text
    # Метрики остаются измеренными: 0 сделок и вырожденный p-value видны.
    assert verdict.trades == 0
    assert verdict.p_value == 1.0
    assert verdict.dsr > 0.95


def test_inapplicable_gate_never_yields_alive_even_when_others_pass():
    """Анти-обходной инвариант: неприменимый гейт не сертифицируется.

    DSR проходит, PBO проходит (готовое значение по валидной по форме
    матрице), min_trades и permutation неприменимы. Вердикт обязан остаться
    не-alive: иначе «гейт не проверялся» стало бы «гейт пройден».
    """
    case = _non_directional()
    n = len(case[0])
    matrix = np.zeros((n, 2))

    verdict = _verdict(case, pd.Series(dtype="float64"),
                       returns_matrix=matrix, pbo_value=0.1)

    assert verdict.dsr > 0.95
    assert verdict.pbo == 0.1
    assert not verdict.alive
    assert verdict.reasons == ()
    assert len(verdict.inapplicable) == 2


def test_missing_permutation_inputs_remain_a_failure_not_inapplicable():
    """Нет positions/price_returns — это отказ входа, а не свойство класса."""
    returns, _, _ = _non_directional()
    verdict = validate(returns, pd.Series(dtype="float64"),
                       (1 + returns).cumprod(), config={}, n_trials=1,
                       strategy_name="s", experiment_id="x")

    assert not verdict.alive
    assert any("permutation" in reason and "positions" in reason
               for reason in verdict.reasons)
    assert verdict.inapplicable == ()


# --- 2. Направленный путь не задет --------------------------------------------


def test_directional_book_keeps_min_trades_and_permutation_applicable():
    case = _directional()
    trades = pd.Series(np.random.default_rng(43).normal(0.002, 0.01, 400))

    verdict = _verdict(case, trades)

    assert verdict.alive, f"край убит: {verdict.reasons}"
    assert verdict.inapplicable == ()
    assert verdict.trades == 400
    assert verdict.p_value < 0.05


def test_directional_book_with_few_trades_still_fails_the_gate():
    """Мало сделок у направленной книги — по-прежнему провал гейта."""
    case = _directional()
    trades = pd.Series(np.random.default_rng(44).normal(0.002, 0.01, 20))

    verdict = _verdict(case, trades)

    assert not verdict.alive
    assert any("недостаточно сделок: 20 < 100" in reason
               for reason in verdict.reasons)
    assert verdict.inapplicable == ()


def test_constant_nonzero_exposure_keeps_permutation_as_failure():
    """Постоянная НЕНУЛЕВАЯ позиция — направленная книга: вырожденный p-value
    остаётся провалом гейта, а не «неприменимо» (spec 6.4). Классификация не
    имеет права расширяться до «перестановка ничего не меняет»."""
    n = 4000
    rng = np.random.default_rng(45)
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    positions = pd.Series(np.ones(n))
    returns = positions * price_ret
    trades = pd.Series(rng.normal(0.0, 0.01, 200))

    verdict = _verdict((returns, price_ret, positions), trades, n_trials=1000)

    assert not verdict.alive
    assert verdict.p_value == 1.0
    assert verdict.inapplicable == ()
    assert any("p-value" in reason for reason in verdict.reasons)


def test_flat_book_without_pnl_keeps_trade_failure():
    """Книга не торговала вовсе (нулевые позиции и нулевой P&L): min_trades —
    настоящий провал (свидетельств нет), неприменим только permutation-тест,
    которому нечего перемешивать."""
    n = 300
    rng = np.random.default_rng(46)
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    positions = pd.Series(np.zeros(n))
    returns = pd.Series(np.zeros(n))

    verdict = _verdict((returns, price_ret, positions),
                       pd.Series(dtype="float64"))

    assert not verdict.alive
    assert any("недостаточно сделок: 0 < 100" in reason
               for reason in verdict.reasons)
    assert len(verdict.inapplicable) == 1
    assert "permutation" in verdict.inapplicable[0]


# --- 3. Вырождение permutation-теста измерено ---------------------------------


def test_permutation_pvalue_is_constant_one_for_zero_positions():
    """p-value не зависит ни от доходностей цены, ни от дохода стратегии.

    Тест перемешивает позиции; нулевой ряд инвариантен, поэтому p ≡ 1.0 при
    любой цене. Доходности стратегии в permutation_pvalue вообще не входят —
    именно поэтому гейт не может свидетельствовать против carry-книги.
    """
    rng = np.random.default_rng(47)
    zeros = pd.Series(np.zeros(1000))

    for scale in (1e-4, 1e-1):
        price_ret = pd.Series(rng.normal(0.0, scale, 1000))
        assert permutation_pvalue(price_ret, zeros) == 1.0
        assert permutation_pvalue(price_ret, zeros, n_permutations=25) == 1.0


def test_two_books_with_opposite_pnl_have_identical_permutation_pvalue():
    """Один и тот же нулевой ряд позиций с прибыльной и убыточной книгой даёт
    один и тот же p-value: тест не видит carry-доход и не различает их."""
    rng = np.random.default_rng(48)
    n = 3000
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    zeros = pd.Series(np.zeros(n))

    winner = _verdict((pd.Series(np.full(n, 1e-4)), price_ret, zeros),
                      pd.Series(dtype="float64"))
    loser = _verdict((pd.Series(np.full(n, -1e-4)), price_ret, zeros),
                     pd.Series(dtype="float64"))

    assert winner.p_value == loser.p_value == 1.0
    assert winner.inapplicable == loser.inapplicable
