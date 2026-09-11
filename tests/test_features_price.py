import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_series, random_walk

from alpha_lab.features.price import (
    adf_pvalue, hurst_exponent, ou_params, zscore,
)


def test_ou_params_recovers_known_theta():
    """theta=0.05 → полужизнь ln(2)/0.05 ≈ 13.9 бар."""
    s = ou_series(n=20000, theta=0.05, sigma=1.0, seed=1)

    p = ou_params(s)

    assert p.theta == pytest.approx(0.05, rel=0.25)
    assert p.half_life == pytest.approx(np.log(2) / 0.05, rel=0.25)
    assert p.mu == pytest.approx(100.0, abs=0.5)


def test_ou_params_faster_mean_reversion_gives_shorter_half_life():
    slow = ou_params(ou_series(n=20000, theta=0.01, seed=2))
    fast = ou_params(ou_series(n=20000, theta=0.10, seed=2))

    assert fast.half_life < slow.half_life
    assert fast.theta > slow.theta


def test_ou_params_on_random_walk_has_low_r2():
    """Случайное блуждание не возвращается к среднему — R² должен быть мал."""
    p = ou_params(random_walk(n=20000, seed=3))

    assert p.r2 < 0.05


def test_adf_rejects_random_walk_and_accepts_ou():
    rw_p = adf_pvalue(random_walk(n=5000, seed=4))
    ou_p = adf_pvalue(ou_series(n=5000, theta=0.1, seed=4))

    assert rw_p > 0.05      # нестационарен
    assert ou_p < 0.05      # стационарен


def test_hurst_random_walk_near_half():
    h = hurst_exponent(random_walk(n=20000, seed=5))

    assert 0.4 < h < 0.6


def test_hurst_mean_reverting_below_half():
    h = hurst_exponent(ou_series(n=20000, theta=0.1, seed=6))

    assert h < 0.5


def test_zscore_is_standardized():
    """theta=0.5 → полужизнь ≈ 1.4 бара, ряд почти iid.

    Окно 100 бар покрывает маргинальное распределение, поэтому скользящий
    z-скор действительно стандартизован.
    """
    s = ou_series(n=5000, theta=0.5, seed=8)

    z = zscore(s, window=100).dropna()

    assert abs(z.mean()) < 0.15
    assert abs(z.std() - 1.0) < 0.15


def test_zscore_on_persistent_series_is_wider_than_one():
    """На устойчивом ряду (theta=0.05, полужизнь ≈ 14 бар) окно 100 бар — лишь ~5 τ.

    Скользящее среднее успевает следовать за ценой, поэтому скользящая σ внутри
    окна занижает маргинальную σ из-за автокорреляции, и z-скор выходит шире
    единицы. Практический смысл: порог в k сигм срабатывает реже, чем настоящее
    k-сигма событие на таком ряду.
    """
    s = ou_series(n=5000, theta=0.05, seed=8)

    z = zscore(s, window=100).dropna()

    assert z.std() > 1.1


def test_zscore_is_zero_on_constant_series():
    s = pd.Series([5.0] * 100)

    z = zscore(s, window=20)

    assert (z.fillna(0.0) == 0.0).all()
