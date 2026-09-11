import numpy as np
import pandas as pd
import pytest

from alpha_lab.data.quality import periods_per_year
from alpha_lab.validation.metrics import (
    DEFAULT_PERIODS, calmar_ratio, max_drawdown, profit_factor, sharpe_ratio,
    sortino_ratio, summarize,
)


def test_periods_per_year_derived_from_timeframe():
    """Годовой множитель выводится из таймфрейма, а не захардкожен под часы.

    Крипта торгуется 24/7, поэтому год — 365 дней: 1d → 365, 1h → 8760,
    1m → 525600. Множитель общий для Sharpe, Sortino и Calmar.
    """
    assert periods_per_year("1d") == 365
    assert periods_per_year("1h") == 365 * 24
    assert periods_per_year("4h") == 365 * 6
    assert periods_per_year("15m") == 365 * 24 * 4
    assert periods_per_year("5m") == 365 * 24 * 12
    assert periods_per_year("1m") == 365 * 24 * 60


def test_periods_per_year_unknown_timeframe_raises():
    """Неизвестный таймфрейм — громкий ValueError, а не молчаливый дефолт.

    Неверный множитель невидим в отчёте: Sharpe отличается на sqrt(24),
    Calmar — на 24, и все метрики вердикта испорчены без единого признака
    в выводе.
    """
    with pytest.raises(ValueError, match="Неизвестный таймфрейм"):
        periods_per_year("2h")


def test_sharpe_and_calmar_scale_with_timeframe_periods():
    """Синтетика с известными моментами: годовой Sharpe на 1h и 1d сходится
    с формулой mean/std*sqrt(periods_per_year), а не с одним и тем же числом."""
    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))
    raw = float(r.mean() / r.std(ddof=1))

    s_1h = sharpe_ratio(r, periods_per_year=periods_per_year("1h"))
    s_1d = sharpe_ratio(r, periods_per_year=periods_per_year("1d"))

    assert s_1h == pytest.approx(raw * np.sqrt(8760), rel=1e-12)
    assert s_1d == pytest.approx(raw * np.sqrt(365), rel=1e-12)
    assert s_1h / s_1d == pytest.approx(np.sqrt(24), rel=1e-12)

    equity = (1.0 + r).cumprod()
    c_1h = calmar_ratio(r, equity, periods_per_year=periods_per_year("1h"))
    c_1d = calmar_ratio(r, equity, periods_per_year=periods_per_year("1d"))
    # Годовая доходность арифметическая: множитель входит ровно в 24 раза.
    assert c_1h / c_1d == pytest.approx(24.0, rel=1e-12)


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


def test_sortino_uses_only_downside_deviation():
    r = pd.Series([0.02, -0.01, 0.03, -0.02, 0.01, -0.015])
    downside = r[r < 0]
    expected = r.mean() / downside.std(ddof=1) * np.sqrt(DEFAULT_PERIODS)

    # Знаменатель — только нисходящее СКО, а не полное: в этом весь смысл
    # Сортино. Проверяем арифметикой, а не неравенством, которое прошло бы
    # и при полном СКО (тогда Сортино совпал бы с Шарпом).
    assert sortino_ratio(r) == pytest.approx(expected, rel=1e-9)
    assert sortino_ratio(r) != pytest.approx(sharpe_ratio(r), rel=1e-9)


def test_profit_factor():
    trades = pd.Series([0.05, -0.02, 0.03, -0.01])

    assert profit_factor(trades) == pytest.approx(0.08 / 0.03)


def test_profit_factor_no_losses_is_inf():
    assert profit_factor(pd.Series([0.01, 0.02])) == float("inf")


def test_calmar_ratio_hand_computed():
    equity = pd.Series([1.0, 1.5, 1.2, 1.8])
    returns = equity.pct_change().fillna(0.0)

    # returns = [0, 0.5, -0.2, 0.5], mean = 0.2, max_dd = -0.2.
    # Годовая доходность = 0.2 * 8760 = 1752, Calmar = 1752 / 0.2 = 8760.
    assert calmar_ratio(returns, equity) == pytest.approx(0.2 * 8760 / 0.2, rel=1e-9)


def test_calmar_is_frequency_invariant():
    daily = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    # Та же траектория в часовой дискретизации: доходность дня набирается
    # за один час, остальные 23 часа нулевые. Границы дней совпадают с
    # дневной серией, внутридневной путь монотонен, поэтому максимальная
    # просадка у серий одинаковая. В общем случае точного равенства быть не
    # может — просадка зависит от внутрипериодного пути, который при
    # агрегации теряется; здесь он сохранён конструкцией.
    hourly = np.zeros(len(daily) * 24)
    hourly[::24] = daily
    eq_daily = pd.Series(np.concatenate([[1.0], np.cumprod(1.0 + daily)]))
    eq_hourly = pd.Series(np.concatenate([[1.0], np.cumprod(1.0 + hourly)]))

    # Годовая доходность арифметическая и не зависит от частоты:
    # mean(d/24) * 8760 == mean(d) * 365.
    assert hourly.mean() * DEFAULT_PERIODS == pytest.approx(daily.mean() * 365, rel=1e-12)

    c_daily = calmar_ratio(daily, eq_daily, periods_per_year=365)
    c_hourly = calmar_ratio(hourly, eq_hourly, periods_per_year=DEFAULT_PERIODS)

    # Старая формула (mean/std * P) давала здесь расхождение в разы.
    assert c_hourly == pytest.approx(c_daily, rel=1e-9)


def test_calmar_positive_mean_small_drawdown_is_of_ordinary_magnitude():
    # 249 часов по +0.06% и один час -5%: mean = 0.0003976,
    # ann = mean * 365 = 0.1451, max_dd = 0.05, Calmar ≈ 2.90.
    returns = pd.Series([0.0006] * 249 + [-0.05])
    equity = (1.0 + returns).cumprod()

    c = calmar_ratio(returns, equity, periods_per_year=365)

    # Опубликованные Calmar живут в диапазоне ~0.5–3: при положительном
    # среднем и просадке 5% значение обязано быть больше 1, но оставаться
    # обычным (меньше 10). Старая формула давала здесь ~907, потому что
    # делила на СКО периода вместо годовой доходности.
    assert 1.0 < c < 10.0


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
