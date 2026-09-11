"""OU-процесс с известными параметрами.

Это эталон для проверки: мы знаем истинные theta и half_life,
значит знаем, что движок обязан на этих данных получить.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ou_series(n=5000, mu=100.0, theta=0.05, sigma=1.0, seed=42,
              start_price=None) -> pd.Series:
    """Дискретизация OU: x[t+1] = x[t] + theta*(mu - x[t]) + sigma*eps."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype="float64")
    x[0] = mu if start_price is None else start_price
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * eps[t]
    return pd.Series(x, name="close")


def _bars_frame(close: pd.Series, start: str, freq: str, delta: float) -> pd.DataFrame:
    """OHLC-рамка вокруг готового close-ряда.

    Единственный источник правды о конверте: high/low обязаны охватывать и
    open, и close, иначе бар физически невозможен (цена вне [low, high]) и не
    пройдёт check_bars. `delta` — запас сверх диапазона open/close.
    """
    ts = pd.date_range(start, periods=len(close), freq=freq, tz="UTC")
    open_ = close.shift(1).fillna(close.iloc[0])
    both = pd.concat([open_, close], axis=1)
    return pd.DataFrame({
        "ts": ts,
        "open": open_,
        "high": both.max(axis=1) + delta,
        "low": both.min(axis=1) - delta,
        "close": close,
        "volume": 10.0,
        "quote_volume": close * 10.0,
        "trades": 5,
        "taker_buy_volume": 6.0,
    })


def ou_bars(n=5000, mu=100.0, theta=0.05, sigma=1.0, seed=42,
            start="2024-01-01", freq="1min") -> pd.DataFrame:
    close = ou_series(n, mu, theta, sigma, seed)
    return _bars_frame(close, start, freq, delta=sigma * 0.5)


def random_walk(n=5000, seed=7, start="2024-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.standard_normal(n)), name="close")


def random_walk_bars(n=5000, seed=7, start="2024-01-01",
                     freq="1min") -> pd.DataFrame:
    """Физически корректные бары вокруг случайного блуждания.

    close — ровно тот же ряд, что у random_walk(n, seed), а high/low строятся
    вокруг него, а не вокруг чужого OU-ряда. Прежний приём (подмена close у
    ou_bars) оставлял конверт от OU: close выходил за [low, high], ATR
    раздувался до ~33 вместо ~1.6, и все метрики стратегии становились
    артефактом фикстуры. delta=0.5 — половина типичного шага блуждания (1.0).
    """
    close = random_walk(n, seed, start=start)
    return _bars_frame(close, start, freq, delta=0.5)
