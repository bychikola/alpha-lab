"""Статистические признаки цены.

ou_params реализует ту же МНК-регрессию ΔS = a + b·S, что и Pine Script
(index.html, блок «Ядро модели»), но на numpy и без ограничений скользящего окна.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class OUParams:
    # При отсутствии возврата theta <= 0 (theta = -b, b > 0), а nan только
    # у half_life. Проверять возврат надо по theta > 0 или по finite
    # half_life, а не через isnan(theta).
    theta: float        # скорость возврата к среднему; <= 0, если возврата нет
    mu: float           # долгосрочное среднее
    sigma: float        # волатильность остатка
    half_life: float    # ln(2)/theta, бар
    r2: float           # качество подгонки; малое значение = возврата нет
    n: int


def ou_params(series: pd.Series | np.ndarray) -> OUParams:
    """Оценка параметров OU-процесса МНК-регрессией ΔS на S.

    ΔS = a + b·S + ε,  theta = -b,  half_life = ln(2)/theta.
    Возврат к среднему есть только при b < 0 (theta > 0).
    """
    s = np.asarray(series, dtype="float64")
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 10:
        return OUParams(np.nan, np.nan, np.nan, np.nan, np.nan, n)

    y = np.diff(s)          # ΔS
    x = s[:-1]              # S[t]
    x_mean, y_mean = x.mean(), y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx < 1e-12:
        return OUParams(np.nan, float(x_mean), 0.0, np.nan, np.nan, n)

    b = ((x - x_mean) * (y - y_mean)).sum() / sxx
    a = y_mean - b * x_mean

    resid = y - (a + b * x)
    sigma = float(resid.std(ddof=2)) if n > 3 else float(resid.std())

    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 1e-12 else 0.0

    theta = -b
    if theta > 0:
        half_life = float(np.log(2) / theta)
        mu = float(-a / b)
    else:
        half_life, mu = np.nan, float(x_mean)

    return OUParams(theta=float(theta), mu=mu, sigma=sigma,
                    half_life=half_life, r2=r2, n=n)


def adf_pvalue(series: pd.Series | np.ndarray, maxlag: int | None = None) -> float:
    """p-value теста Дики–Фуллера. Малый p → ряд стационарен (возвращается к среднему)."""
    s = np.asarray(series, dtype="float64")
    s = s[np.isfinite(s)]
    if len(s) < 20 or np.std(s) < 1e-12:
        return 1.0
    try:
        # result_object=True — новый API statsmodels (>=0.15): tuple-индексация [1]
        # помечена FutureWarning, в 0.16 adfuller вернёт ADFullerResult по умолчанию.
        res = adfuller(s, maxlag=maxlag, autolag="AIC", result_object=True)
    except (ValueError, np.linalg.LinAlgError):
        return 1.0
    return float(res.pvalue)


def hurst_exponent(series: pd.Series | np.ndarray, min_lag: int = 2,
                   max_lag: int = 100) -> float:
    """Показатель Хёрста через масштабирование дисперсии лаговых разностей.

    H ≈ 0.5 — случайное блуждание; H < 0.5 — возврат к среднему; H > 0.5 — тренд.
    """
    s = np.asarray(series, dtype="float64")
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 4:
        return np.nan

    lags = np.arange(min_lag, max_lag)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    ok = tau > 1e-12
    if ok.sum() < 4:
        return np.nan

    slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
    return float(slope)


def rolling_sigma(series: pd.Series, window: int) -> pd.Series:
    """Скользящее стандартное отклонение (ddof=0). Первые window-1 значений — NaN."""
    s = pd.Series(series).astype("float64")
    return s.rolling(window, min_periods=window).std(ddof=0).rename("sigma")


def zscore(series: pd.Series, window: int) -> pd.Series:
    """z-скор относительно скользящего среднего и σ. σ≈0 → 0.

    Первые window-1 значений — 0.0 (прогрев), а не NaN: окно ещё не набрано.
    0.0 означает «сигнала нет», а не «ряд ровно на среднем»; отличить прогрев
    от истинного нуля по значению нельзя, поэтому использовать прогревочные
    нули как непрерывный признак нельзя. Для пороговой стратегии
    (z <= -k / z >= k) это безопасно: 0.0 порог не пересекает.
    """
    s = pd.Series(series).astype("float64")
    mean = s.rolling(window, min_periods=window).mean()
    std = rolling_sigma(s, window)
    return ((s - mean) / std.where(std > 1e-12, np.nan)).fillna(0.0).rename("z")
