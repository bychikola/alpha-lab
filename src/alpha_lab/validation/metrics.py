"""Метрики эффективности.

sharpe_ratio с annualize=False возвращает значение за период — именно оно
требуется формуле Deflated Sharpe (см. significance.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 365 * 24 * 60
# Часовой дефолт крипты (24/7). Он сохранён для вызовов без таймфрейма, но
# рабочий путь обязан передавать множитель своего таймфрейма: единственный
# источник правды «таймфрейм → баров в году» — data.quality.periods_per_year.
DEFAULT_PERIODS = 365 * 24


def _clean(returns) -> np.ndarray:
    r = np.asarray(pd.Series(returns), dtype="float64")
    return r[np.isfinite(r)]


def sharpe_ratio(returns, periods_per_year: int = DEFAULT_PERIODS,
                 annualize: bool = True, risk_free: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    raw = (r.mean() - risk_free) / sd
    return float(raw * np.sqrt(periods_per_year) if annualize else raw)


def sortino_ratio(returns, periods_per_year: int = DEFAULT_PERIODS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd < 1e-12:
        return float("inf") if r.mean() > 0 else 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity) -> float:
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def calmar_ratio(returns, equity, periods_per_year: int = DEFAULT_PERIODS) -> float:
    """Калмар: годовая доходность к абсолютной максимальной просадке.

    Годовая доходность аннуализируется арифметически —
    mean(returns) * periods_per_year. Геометрическое компаундирование (CAGR)
    для коротких серий, которые оценивает polygon, не имеет смысла: на
    нескольких барах оно вырождается и даёт нестабильные значения.
    """
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    dd = abs(max_drawdown(equity))
    if dd < 1e-12:
        return 0.0
    ann_return = float(np.mean(r) * periods_per_year)
    return float(ann_return / dd)


def profit_factor(trade_returns) -> float:
    t = _clean(trade_returns)
    if len(t) == 0:
        return float("nan")
    gains = t[t > 0].sum()
    losses = abs(t[t < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize(returns, trade_returns, equity,
              periods_per_year: int = DEFAULT_PERIODS) -> dict[str, float]:
    r = _clean(returns)
    t = _clean(trade_returns)
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]

    total = float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0
    wins = t[t > 0]
    return {
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_dd": max_drawdown(eq),
        "calmar": calmar_ratio(r, eq, periods_per_year),
        "profit_factor": profit_factor(t),
        "total_return": total,
        "trades": float(len(t)),
        "win_rate": float(len(wins) / len(t)) if len(t) else 0.0,
        "avg_trade": float(t.mean()) if len(t) else 0.0,
    }
