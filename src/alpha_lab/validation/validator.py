"""Сборка вердикта. Слепой слой: знает только результаты, не стратегию.

Инвариант: этот модуль НЕ импортирует слой стратегий.
Иначе появляется соблазн «подкрутить» проверку под конкретную стратегию.
Проверяется тестом из tests/test_traps.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_lab.validation.metrics import max_drawdown, sharpe_ratio, summarize
from alpha_lab.validation.significance import (
    deflated_sharpe_ratio, permutation_pvalue,
)

DEFAULT_THRESHOLDS = {
    "min_trades": 100,
    "max_pbo": 0.5,
    "max_p_value": 0.05,
    "min_dsr": 0.95,
    "n_permutations": 1000,
}

# Итоговый годовой Sharpe считается для часовых баров крипты (24/7)
PERIODS_PER_YEAR = 365 * 24


@dataclass(frozen=True)
class Verdict:
    strategy_name: str
    experiment_id: str
    sharpe: float
    dsr: float
    p_value: float
    max_dd: float
    total_return: float
    trades: int
    n_configs_tried: int
    alive: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)
    pbo: float = float("nan")
    # Гейты, которые в этом прогоне НЕ проверялись (нет входных данных для
    # проверки). Это не причины смерти и не влияют на alive: отсутствие
    # матрицы конфигураций — свойство одиночного прогона, а не дефект
    # стратегии. Но молчать нельзя: иначе отчёт с пустым reasons читается
    # как «все гейты пройдены», хотя часть из них не запускалась.
    warnings: tuple[str, ...] = field(default_factory=tuple)


def validate(returns, trade_returns, equity, config: dict, n_trials: int,
             strategy_name: str, experiment_id: str,
             returns_matrix=None, price_returns=None, positions=None,
             warnings: tuple[str, ...] = ()) -> Verdict:
    """Выносит вердикт. Все пороги — из config, значения по умолчанию в DEFAULT_THRESHOLDS.

    price_returns и positions обязательны для permutation-теста: он перемешивает
    позиции относительно доходностей. Без них проверка невозможна, и вердикт
    выносится отрицательный — тихая деградация недопустима.

    warnings — внешние (собранные CLI) предупреждения о непроверенных гейтах;
    к ним добавляется предупреждение о неоценённом PBO. Предупреждения — plain
    strings, не зависят от NaN и не участвуют в alive.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(config or {})}

    r = np.asarray(pd.Series(returns), dtype="float64")
    r = r[np.isfinite(r)]
    t = np.asarray(pd.Series(trade_returns), dtype="float64")
    t = t[np.isfinite(t)]
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]

    n_trades = int(len(t))
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials)

    permutation_available = price_returns is not None and positions is not None
    if permutation_available:
        p_value = permutation_pvalue(
            price_returns, positions,
            n_permutations=int(thresholds["n_permutations"]), seed=0,
        )
    else:
        p_value = 1.0

    pbo = float("nan")
    warn: list[str] = list(warnings)
    if returns_matrix is not None:
        from alpha_lab.validation.significance import pbo_cscv
        pbo = pbo_cscv(returns_matrix)
    else:
        # Одиночный прогон: PBO физически не вычислим (нужна матрица
        # T × N конфигураций). Это не причина смерти — но и не «пройдено»:
        # условие spec 6.5 «pbo < 0.5» остаётся непроверенным, и вердикт
        # обязан сказать об этом явно, отдельным каналом warnings.
        warn.append(
            "PBO не оценён: матрица доходностей конфигураций не передана. "
            "Условие spec 6.5 «pbo < 0.5» для этого прогона не проверено; "
            "PBO требует многоконфигурационную матрицу (возможность фазы 2)."
        )

    reasons: list[str] = []
    if not permutation_available:
        reasons.append(
            "permutation-тест не выполнен: не переданы price_returns и positions"
        )
    if n_trades < thresholds["min_trades"]:
        reasons.append(
            f"недостаточно сделок: {n_trades} < {thresholds['min_trades']}"
        )
    if not np.isfinite(dsr) or dsr <= thresholds["min_dsr"]:
        reasons.append(
            f"DSR {dsr:.3f} ≤ {thresholds['min_dsr']} (с поправкой на {n_trials} попыток)"
        )
    if not np.isfinite(p_value) or p_value >= thresholds["max_p_value"]:
        reasons.append(
            f"p-value {p_value:.3f} ≥ {thresholds['max_p_value']} — неотличимо от случая"
        )
    if np.isfinite(pbo):
        if pbo >= thresholds["max_pbo"]:
            reasons.append(f"PBO {pbo:.2f} ≥ {thresholds['max_pbo']} — признак подгонки")
    elif returns_matrix is not None:
        # Матрицу передали, но CSCV её не осилил: молча пропустить проверку нельзя.
        reasons.append(
            f"PBO не вычислен для матрицы {np.asarray(returns_matrix).shape}: "
            f"нужно ≥ 2 конфигураций и ≥ 2·n_blocks наблюдений"
        )

    stats = summarize(r, t, eq, PERIODS_PER_YEAR) if len(r) else {}

    return Verdict(
        strategy_name=strategy_name,
        experiment_id=experiment_id,
        sharpe=sharpe_ratio(r, PERIODS_PER_YEAR),
        dsr=dsr,
        p_value=p_value,
        max_dd=max_drawdown(eq) if len(eq) else 0.0,
        total_return=float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0,
        trades=n_trades,
        n_configs_tried=n_trials,
        alive=len(reasons) == 0,
        reasons=tuple(reasons),
        metrics=stats,
        pbo=pbo,
        warnings=tuple(warn),
    )
