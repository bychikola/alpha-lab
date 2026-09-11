import warnings

import numpy as np
import pytest
from scipy import stats

from alpha_lab.validation.significance import (
    _sharpe_raw, deflated_sharpe_ratio, pbo_cscv, permutation_pvalue,
)


def test_dsr_of_pure_noise_is_low_after_many_trials():
    """Шум при поправке на 1000 попыток обязан получить низкий DSR.

    Проверяем именно с n_trials > 1: при одной попытке DSR шума колеблется
    вокруг 0.5, и любой жёсткий порог здесь — флейки-тест.
    """
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, 5000)

    assert deflated_sharpe_ratio(r, n_trials=1000) < 0.1


def test_dsr_lower_for_more_trials():
    """Чем больше конфигураций перебрали, тем сильнее штраф."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0008, 0.01, 5000)

    few = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=1000)

    assert many < few


def test_dsr_high_for_strong_genuine_edge():
    rng = np.random.default_rng(3)
    r = rng.normal(0.003, 0.01, 20000)      # Sharpe ≈ 0.3 за период

    assert deflated_sharpe_ratio(r, n_trials=1) > 0.95


def test_dsr_handles_short_series():
    assert deflated_sharpe_ratio(np.array([0.01, 0.02]), n_trials=1) == 0.0


def test_dsr_constant_series_is_zero_and_silent():
    """Ненулевая константа: нулевая дисперсия → ровно 0.0 и ни одного warning.

    simplefilter("error") превращает любой warning scipy в исключение, поэтому
    регресс «Precision loss … catastrophic cancellation» снова уронит тест.
    """
    r = np.full(100, 0.001)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert deflated_sharpe_ratio(r, n_trials=10) == 0.0


def test_dsr_all_zero_series_is_zero_and_silent():
    """Все нули: нулевая дисперсия → ровно 0.0 и ни одного warning."""
    r = np.zeros(100)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert deflated_sharpe_ratio(r, n_trials=10) == 0.0


def test_dsr_rejects_bad_n_trials():
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, 1000)

    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(r, n_trials=0)


def test_dsr_explicit_sr_variance_deflates_more_and_matches_hand_formula():
    """Явная межтрековая дисперсия Sharpe должна использоваться в sr0 напрямую.

    Ручной пересчёт: z1 = Φ⁻¹(1 − 1/N), z2 = Φ⁻¹(1 − 1/(N·e)),
    sr0 = sqrt(var)·((1 − γ)·z1 + γ·z2), γ = 0.5772156649015329,
    DSR = Φ((sr − sr0)·sqrt(n − 1)/denom). Большая дисперсия даёт больший sr0
    и, значит, меньший DSR при тех же доходностях и n_trials.
    """
    rng = np.random.default_rng(12)
    r = rng.normal(0.001, 0.01, 2000)
    n_trials = 50
    small_var, large_var = 1e-4, 4e-4

    small = deflated_sharpe_ratio(r, n_trials=n_trials, sr_variance=small_var)
    large = deflated_sharpe_ratio(r, n_trials=n_trials, sr_variance=large_var)

    assert large < small

    gamma = 0.5772156649015329
    sr = _sharpe_raw(r)
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    sr0 = np.sqrt(large_var) * ((1.0 - gamma) * z1 + gamma * z2)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    denom = np.sqrt(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2)
    expected = stats.norm.cdf((sr - sr0) * np.sqrt(len(r) - 1) / denom)

    assert large == pytest.approx(expected, rel=1e-12)


def test_pbo_high_when_performance_is_random():
    """Случайные конфигурации: лучшая на train не лучше на test → PBO ≈ 0.5.

    Одной матрицы мало: сочетания CSCV делят одни и те же блоки и сильно
    зависимы, поэтому выборочное std PBO по одной матрице ≈ 0.20, а не
    sqrt(p(1-p)/252) ≈ 0.03. Усредняем по 20 независимым матрицам — тогда
    стандартная ошибка среднего ≈ 0.03, и полоса (0.35, 0.65) держится с
    запасом в несколько сигм. n_blocks=8 (70 сочетаний) даёт ту же дисперсию,
    что и 10 блоков, но заметно быстрее.
    """
    pbos = []
    for seed in range(100, 120):
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 0.01, size=(2000, 20))
        pbos.append(pbo_cscv(matrix, n_blocks=8))

    assert 0.35 < np.mean(pbos) < 0.65


def test_pbo_low_when_one_config_genuinely_better():
    """Одна конфигурация с реальным преимуществом → PBO низкий."""
    rng = np.random.default_rng(6)
    matrix = rng.normal(0.0, 0.01, size=(2000, 20))
    matrix[:, 0] += 0.002                    # устойчивое преимущество

    pbo = pbo_cscv(matrix, n_blocks=10)

    assert pbo < 0.2


def test_pbo_requires_enough_configs():
    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0, 0.01, size=(1000, 1))

    assert np.isnan(pbo_cscv(matrix))


def test_permutation_separates_signal_from_noise():
    """Сигнал, связанный с доходностью, обязан получить p-value ниже случайного."""
    rng = np.random.default_rng(8)
    n = 3000
    price_ret = rng.normal(0.0, 0.01, n)
    pos_signal = np.sign(price_ret)                          # идеальное предвидение
    pos_noise = rng.choice([-1.0, 0.0, 1.0], size=n)         # сигнала нет

    p_signal = permutation_pvalue(price_ret, pos_signal, n_permutations=500, seed=1)
    p_noise = permutation_pvalue(price_ret, pos_noise, n_permutations=500, seed=1)

    assert p_signal < 0.05
    assert p_noise > p_signal


def test_permutation_of_returns_alone_would_be_meaningless():
    """Регрессия: Sharpe инвариантен к перестановке доходностей.

    Этот тест фиксирует причину, по которой перемешиваются позиции, а не доходности.
    """
    rng = np.random.default_rng(9)
    r = rng.normal(0.001, 0.01, 1000)

    shuffled = rng.permutation(r)

    assert _sharpe_raw(r) == pytest.approx(_sharpe_raw(shuffled))


def test_permutation_is_reproducible_with_seed():
    rng = np.random.default_rng(10)
    price_ret = rng.normal(0.0, 0.01, 800)
    pos = rng.choice([-1.0, 0.0, 1.0], size=800)

    a = permutation_pvalue(price_ret, pos, n_permutations=300, seed=42)
    b = permutation_pvalue(price_ret, pos, n_permutations=300, seed=42)

    assert a == b


def test_permutation_rejects_flat_signal():
    rng = np.random.default_rng(11)
    price_ret = rng.normal(0.0, 0.01, 500)

    assert permutation_pvalue(price_ret, np.zeros(500)) == 1.0


def test_permutation_requires_matching_lengths():
    with pytest.raises(ValueError, match="Длины"):
        permutation_pvalue(np.zeros(100), np.zeros(50))
