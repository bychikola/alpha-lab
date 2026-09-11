"""Каноническая схема бара.

Формат подтверждён на реальном файле архива Binance
(data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip):
12 колонок, заголовок присутствует в свежих файлах и отсутствует в старых.
"""
from __future__ import annotations

import pandas as pd

# Имена колонок в CSV архива Binance (USDT-M futures)
RAW_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

# Канонический вид после нормализации
KLINE_COLUMNS = [
    "ts", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume",
]

FUNDING_COLUMNS = ["ts", "rate", "interval_hours"]

_NUMERIC = ["open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_volume"]


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит бары к каноническому виду: UTC-индекс времени, сортировка, дедуп."""
    missing = [c for c in KLINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В барах отсутствуют колонки: {missing}")

    out = df.loc[:, KLINE_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col in _NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset="ts", keep="last")
    return out.sort_values("ts").reset_index(drop=True)
