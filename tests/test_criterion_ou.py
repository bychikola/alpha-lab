"""Критерий 4: MR на синтетическом OU с известными параметрами выживает в validate().

Земляная правда полигона: на OU-ряде с известной theta край существует по
построению, поэтому если validate() убивает стратегию — это баг валидатора,
а не рынка (spec 9.1). Прежний ближайший контроль (test_strong_edge_survives_validation)
проверял обобщённый синтетический край на случайных доходностях, а не саму
модель MeanReversionStrategy на процессе, под который она написана.

Параметры подобраны замером, а не подгонкой порогов:
  * theta=0.02 (полураспад 34.7 баров) — край снимает ГОРИЗОНТ УДЕРЖАНИЯ, а не
    окно входа: полураспад длиннее окна z-скора (20 баров), поэтому вход ловит
    лишь начало движения, а зарабатывает его удержание до max_bars=500 с
    геометрией 2×ATR стоп / 6×ATR тейк. Замер форвардной доходности лонга
    после входа при z ≤ −2: +1.32% за 35 баров (t=8.6) и +3.19% за 500 баров
    (t=14.7) при издержках круга ≈11 bps. На 10 seed'ах подряд вердикт
    alive=True, DSR 0.95..1.00, p 0.001..0.028, сделок 269..321;
  * theta=0.002 (полураспад ≈ 347 баров) — возврат на горизонте удержания
    почти нулевой: форвардная доходность лонга +0.01% (t=0.1) и не покрывает
    издержки; на 10 seed'ах подряд alive=False при 349..400 сделках, то есть
    убивает статистика, а не гейт min_trades.
ВАЖНО (замер): theta=0.01 (полураспад ≈ 69 баров) НЕ является контролем —
на 8 из 10 seed'ах он alive: даже медленный, но настоящий возврат стратегия
успевает отработать за max_bars=500. «Низкая theta» не синоним «нет края»;
честный контроль — theta <= 0.002.
"""
from __future__ import annotations

import numpy as np
from fixtures.synthetic import ou_bars

from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost
from alpha_lab.strategies.mean_reversion import MeanReversionStrategy
from alpha_lab.validation.validator import validate

PARAMS = {"window": 20, "k": 2.0}
N_BARS = 20000
SEED = 42          # дефолт фикстуры, не подобранный


def _run(theta: float):
    """MR на OU-барах через рабочий бэктест и валидатор. Все параметры явные."""
    bars = ou_bars(n=N_BARS, theta=theta, sigma=1.0, seed=SEED, freq="1h").copy()
    # Объём фикстуры (10 единиц ≈ 1000 котируемых) — синтетическая константа,
    # а не модель ликвидности: заявка на капитал 10 000 больше объёма бара,
    # RealisticCost клипует долю единицей и даёт ~1000 bps проскальзывания на
    # сделку — край убивает артефакт фикстуры, а не рынок (замер RED: Sharpe
    # -13.2, DSR ≈ 0). Приводим объём к ликвидности реального BTC-часа
    # (~1e9 котируемых, как quote_volume=1e8 в фикстуре CLI): комиссия и floor
    # проскальзывания остаются реальными (share ≈ 1e-5), capacity не горит.
    bars["volume"] *= 1e6
    bars["quote_volume"] *= 1e6
    strategy = MeanReversionStrategy(PARAMS)
    result = run_backtest(bars, strategy.generate(bars), RealisticCost())
    verdict = validate(
        returns=result.returns, trade_returns=trade_returns(result),
        equity=result.equity, config={}, n_trials=1, strategy_name="mr_ou",
        experiment_id=f"ou_theta_{theta}",
        price_returns=result.price_returns, positions=result.positions,
        periods_per_year=8760,
    )
    return verdict, result


def test_mr_survives_validation_on_mean_reverting_ou():
    """Настоящий возврат к среднему обязан пережить валидатор.

    Негативный контроль (следующий тест) не менее важен: валидатор, который
    режет всё подряд, бесполезен ровно так же, как всеядный.
    """
    verdict, result = _run(theta=0.02)

    assert verdict.alive, f"MR убита на OU с настоящим краем: {verdict.reasons}"
    assert verdict.reasons == ()
    assert verdict.trades >= 100
    assert verdict.dsr > 0.95 and verdict.p_value < 0.05
    assert not result.over_capacity, "фикстура не должна упираться в ёмкость"
    # Край не вырожденный: годовой Sharpe в реалистичном окне, а не сотни.
    assert 1.0 < verdict.sharpe < 3.0


def test_mr_is_dead_on_ou_without_reversion():
    """Медленный OU (theta=0.002) не даёт края в окне стратегии — вердикт мёртв.

    Сделок достаточно (сотни), поэтому смерть обязана прийти от статистики
    (DSR/p-value), а не от min_trades: иначе тест доказывал бы не отсутствие
    края, а нехватку выборки.
    """
    verdict, result = _run(theta=0.002)

    assert not verdict.alive, f"MR выжила без края: {verdict.reasons}"
    assert verdict.trades >= 100
    assert not any("недостаточно сделок" in r for r in verdict.reasons), verdict.reasons
    assert any("DSR" in r or "p-value" in r for r in verdict.reasons), verdict.reasons
    assert verdict.dsr < 0.95 or verdict.p_value >= 0.05
