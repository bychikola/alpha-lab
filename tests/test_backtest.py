import numpy as np
import pandas as pd
import pytest

from alpha_lab.engine.backtest import run_backtest
from alpha_lab.engine.costs import RealisticCost, ZeroCost


def _bars(close):
    close = pd.Series(close, dtype="float64")
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=len(close), freq="1min", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6, "quote_volume": 1e9, "trades": 100, "taker_buy_volume": 1e5,
    })


def test_buy_and_hold_zero_cost_matches_price_return():
    bars = _bars([100, 110, 121, 133.1])          # +10% за бар
    positions = pd.Series([1.0, 1.0, 1.0, 1.0])

    res = run_backtest(bars, positions, ZeroCost())

    # позиция удерживается со следующего бара: 3 перехода по +10%
    assert res.total_return == pytest.approx(1.1 ** 3 - 1, rel=1e-6)


def test_flat_positions_give_zero_return():
    bars = _bars([100, 110, 90, 95])
    positions = pd.Series([0.0, 0.0, 0.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert res.total_return == pytest.approx(0.0)
    assert res.turnover.sum() == pytest.approx(0.0)


def test_positions_are_shifted_no_lookahead():
    """Стратегия, идеально знающая бар t, не должна заработать на нём же."""
    bars = _bars([100, 200, 100, 200])
    # позиция 1 ровно в бары роста — если бы исполнялась в тот же бар, был бы огромный профит
    positions = pd.Series([1.0, 0.0, 1.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())

    # Сдвиг: позиция 1 держится на баре 1 (рост 100→200, профит),
    # позиция 0 на баре 2 (падение — не участвуем), позиция 1 на баре 3 (рост).
    assert res.total_return == pytest.approx(2.0 * 2.0 - 1.0, rel=1e-6)


def test_costs_reduce_return_and_create_turnover():
    bars = _bars([100] * 10 + [110] * 10)
    positions = pd.Series([1.0] * 20)

    free = run_backtest(bars, positions, ZeroCost())
    costly = run_backtest(bars, positions, RealisticCost(taker_fee_bps=100.0))

    assert costly.total_return < free.total_return
    assert costly.turnover.sum() > 0


def test_higher_fees_monotonically_lower_return():
    bars = _bars([100, 105, 95, 110, 100, 115])
    positions = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    returns = [
        run_backtest(bars, positions, RealisticCost(taker_fee_bps=f)).total_return
        for f in (0.0, 5.0, 20.0, 100.0)
    ]

    assert returns == sorted(returns, reverse=True)


def test_equal_timestamps_produce_no_pnl():
    bars = _bars([100, 100, 100])
    positions = pd.Series([1.0, 1.0, 1.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert res.total_return == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_is_negative_or_zero():
    bars = _bars([100, 120, 60, 80])
    positions = pd.Series([1.0, 1.0, 1.0, 1.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert res.max_drawdown <= 0
    assert res.max_drawdown == pytest.approx(60 / 120 - 1, rel=1e-6)


def test_funding_charged_on_held_position():
    bars = _bars([100] * 20)
    positions = pd.Series([1.0] * 20)
    funding = pd.Series(0.0, index=bars.index)
    funding.iloc[0] = 0.001      # заявка бара 0 ещё не удерживается — платить нечего
    funding.iloc[10] = 0.001     # 0.1% выплата
    # Изолируем funding: с дефолтным RealisticCost() вход добавляет к убытку
    # 5 bps комиссии + 0.51 bps проскальзывания = -5.51e-4, и ожидание -0.001
    # перестаёт быть верным (факт -0.001550449).
    funding_only = RealisticCost(taker_fee_bps=0.0, min_slippage_bps=0.0,
                                 impact_coef=0.0)

    res = run_backtest(bars, positions, funding_only, funding_rate=funding)

    assert res.total_return == pytest.approx(-0.001, rel=1e-3)
    # Позиция бара 0 не удерживалась: funding начисляется на held, а не на target
    assert res.costs["funding"].iloc[0] == 0.0


def test_result_lengths_match_input():
    bars = _bars([100, 101, 102, 103, 104])
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert len(res.equity) == len(bars)
    assert len(res.returns) == len(bars)
    assert len(res.positions) == len(bars)


def test_mismatched_lengths_raise():
    bars = _bars([100, 101, 102])
    positions = pd.Series([1.0, 1.0])

    with pytest.raises(ValueError, match="длин"):
        run_backtest(bars, positions, ZeroCost())


def test_capacity_diagnostic_flags_orders_above_volume_share():
    """Крошечный объём бара: заявка физически неисполнима — это видно в cap_hits."""
    bars = _bars([100.0] * 5)
    bars["quote_volume"] = 1_000.0        # порог 0.01 * 1000 = 10 << заявки 10_000
    positions = pd.Series([1.0] * 5)

    res = run_backtest(bars, positions, ZeroCost(), capital=10_000.0)

    assert res.cap_hits > 0
    assert res.over_capacity is True


def test_capacity_diagnostic_counts_only_bars_with_orders():
    """Одна заявка на входе — ровно один bar недоказанной ёмкости, не все бары."""
    bars = _bars([100.0] * 5)
    bars["quote_volume"] = 1_000.0
    positions = pd.Series([1.0] * 5)

    res = run_backtest(bars, positions, ZeroCost(), capital=10_000.0)

    assert res.cap_hits == 1


def test_large_volume_produces_no_capacity_hits():
    """На ликвидном рынке (quote_volume 1e9) диагностика молчит."""
    bars = _bars([100.0] * 5)
    positions = pd.Series([1.0] * 5)

    res = run_backtest(bars, positions, ZeroCost(), capital=10_000.0)

    assert res.cap_hits == 0
    assert res.over_capacity is False
