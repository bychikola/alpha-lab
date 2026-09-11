"""Проверка статистической значимости результата.

Три механизма против трёх разных способов обмануть себя:

DSR  — против множественных сравнений. Ожидаемый максимум Sharpe у N случайных
       стратегий растёт как sqrt(2·ln N); при N=500 это ≈ 3.5. Сырой Sharpe,
       выбранный как лучший из многих, почти наверняка завышен.
PBO  — против «повезло на истории» (CSCV, Bailey et al. 2015).
Perm — против «а может, оно само так вышло»: перемешиваем, строим нулевое
       распределение, получаем честный p-value.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _sharpe_raw(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return 0.0 if sd < 1e-12 else float(r.mean() / sd)


def deflated_sharpe_ratio(returns, n_trials: int,
                          sr_variance: float | None = None) -> float:
    """Вероятность, что истинный Sharpe > 0 с поправкой на число попыток.

    Возвращает P ∈ [0, 1]. Порог вердикта: DSR > 0.95 (см. validator.py).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials должен быть ≥ 1, получено {n_trials}")

    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return 0.0

    sr = _sharpe_raw(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    if n_trials < 2:
        sr0 = 0.0
    else:
        if sr_variance is None or not np.isfinite(sr_variance) or sr_variance <= 0:
            # Дисперсия оценки Sharpe в нулевой гипотезе
            sr_variance = (1.0 + 0.5 * sr ** 2) / n
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))

    denom = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2, 1e-12))
    z_score = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z_score))


def pbo_cscv(returns_matrix, n_blocks: int = 10) -> float:
    """Probability of Backtest Overfitting через CSCV.

    returns_matrix: (T наблюдений × N конфигураций).
    Блоки делятся на все сочетания половин; для каждого сочетания ищем лучшую
    конфигурацию на train и смотрим её ранг на test. PBO — доля сочетаний,
    где лучшая на train оказалась ниже медианы на test.
    """
    m = np.asarray(returns_matrix, dtype="float64")
    if m.ndim != 2:
        raise ValueError("returns_matrix должен быть двумерным (T × N)")
    t, n_configs = m.shape
    if n_configs < 2 or t < n_blocks * 2 or n_blocks < 4 or n_blocks % 2 != 0:
        return float("nan")

    block_size = t // n_blocks
    blocks = m[:block_size * n_blocks].reshape(n_blocks, block_size, n_configs)
    n_train = n_blocks // 2

    logits = []
    for train_idx in combinations(range(n_blocks), n_train):
        test_idx = [i for i in range(n_blocks) if i not in train_idx]
        train = blocks[list(train_idx)].reshape(-1, n_configs)
        test = blocks[test_idx].reshape(-1, n_configs)

        sr_train = np.array([_sharpe_raw(train[:, j]) for j in range(n_configs)])
        sr_test = np.array([_sharpe_raw(test[:, j]) for j in range(n_configs)])

        best = int(np.argmax(sr_train))
        ranks = stats.rankdata(sr_test, method="average")
        omega = ranks[best] / (n_configs + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(np.log(omega / (1.0 - omega)))

    logits = np.asarray(logits)
    return float((logits < 0).mean())


def permutation_pvalue(price_returns, positions,
                       n_permutations: int = 1000, seed: int = 0) -> float:
    """Доля перемешанных версий, чей Sharpe не хуже наблюдаемого.

    Нулевая гипотеза: сигнал не связан с доходностями.

    ВАЖНО: перемешиваются ПОЗИЦИИ, а не доходности. Sharpe = mean/std инвариантен
    к перестановке доходностей, поэтому перемешивание самого ряда дало бы
    p-value ≡ 1.0 и тест не отклонял бы ничего. Смысл имеет только разрушение
    соответствия «сигнал ↔ доходность».
    """
    pr = np.asarray(price_returns, dtype="float64")
    pos = np.asarray(positions, dtype="float64")
    if len(pr) != len(pos):
        raise ValueError(
            f"Длины price_returns ({len(pr)}) и positions ({len(pos)}) не совпадают"
        )
    ok = np.isfinite(pr) & np.isfinite(pos)
    pr, pos = pr[ok], pos[ok]
    if len(pr) < 10 or not pos.any():
        return 1.0

    observed = _sharpe_raw(pr * pos)
    rng = np.random.default_rng(seed)
    better = 0
    for _ in range(n_permutations):
        if _sharpe_raw(pr * rng.permutation(pos)) >= observed:
            better += 1
    return float((better + 1) / (n_permutations + 1))
