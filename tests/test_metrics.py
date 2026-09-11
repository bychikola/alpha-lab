import numpy as np
import pandas as pd
import pytest

from alpha_lab.validation.metrics import (
    calmar_ratio, max_drawdown, profit_factor, sharpe_ratio, sortino_ratio, summarize,
)


def test_sharpe_of_constant_returns_is_zero_std():
    r = pd.Series([0.001] * 100)

    assert sharpe_ratio(r) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))

    assert sharpe_ratio(r) > 0


def test_sharpe_scales_with_annualization():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))

    raw = sharpe_ratio(r, annualize=False)
    ann = sharpe_ratio(r, periods_per_year=365 * 24)

    assert ann == pytest.approx(raw * np.sqrt(365 * 24), rel=1e-9)


def test_sharpe_of_negated_returns_flips_sign():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))

    assert sharpe_ratio(-r) == pytest.approx(-sharpe_ratio(r))


def test_max_drawdown_known_value():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.5])

    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1.0)


def test_max_drawdown_monotonic_equity_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_sortino_ignores_upside_volatility():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.01, 3000))

    # Сортино ≥ Шарпа: в знаменателе только нисходящая волатильность
    assert sortino_ratio(r) >= sharpe_ratio(r) * 0.9


def test_profit_factor():
    trades = pd.Series([0.05, -0.02, 0.03, -0.01])

    assert profit_factor(trades) == pytest.approx(0.08 / 0.03)


def test_profit_factor_no_losses_is_inf():
    assert profit_factor(pd.Series([0.01, 0.02])) == float("inf")


def test_calmar_ratio():
    equity = pd.Series([1.0, 1.5, 1.2, 1.8])
    returns = equity.pct_change().fillna(0.0)

    c = calmar_ratio(returns, equity)

    assert c > 0


def test_summarize_returns_all_keys():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    equity = (1 + r).cumprod()
    trades = pd.Series(rng.normal(0.002, 0.01, 50))

    out = summarize(r, trades, equity)

    for key in ("sharpe", "sortino", "max_dd", "calmar", "profit_factor",
                "total_return", "trades", "win_rate", "avg_trade"):
        assert key in out


def test_summarize_handles_empty_trades():
    r = pd.Series([0.0] * 10)
    out = summarize(r, pd.Series([], dtype="float64"), (1 + r).cumprod())

    assert out["trades"] == 0
    assert np.isnan(out["profit_factor"]) or out["profit_factor"] == 0.0
