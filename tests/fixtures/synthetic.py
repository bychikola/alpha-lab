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


def ou_bars(n=5000, mu=100.0, theta=0.05, sigma=1.0, seed=42,
            start="2024-01-01", freq="1min") -> pd.DataFrame:
    close = ou_series(n, mu, theta, sigma, seed)
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    open_ = close.shift(1).fillna(close.iloc[0])
    # high/low обязаны охватывать и open, и close, иначе бар физически
    # невозможен (open вне [low, high]) и не пройдёт check_bars.
    both = pd.concat([open_, close], axis=1)
    return pd.DataFrame({
        "ts": ts,
        "open": open_,
        "high": both.max(axis=1) + sigma * 0.5,
        "low": both.min(axis=1) - sigma * 0.5,
        "close": close,
        "volume": 10.0,
        "quote_volume": close * 10.0,
        "trades": 5,
        "taker_buy_volume": 6.0,
    })


def random_walk(n=5000, seed=7, start="2024-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.standard_normal(n)), name="close")
