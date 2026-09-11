"""Mean Reversion Quant Pro — перенос Pine Script (index.html) в Python.

Источник: блок pine-strategy, секция «ЯДРО МОДЕЛИ».
Логика входа: z ≤ −k → лонг, z ≥ +k → шорт.
Выход: стоп 2×ATR, тейк 6×ATR, принудительный выход по времени.
Опциональный фильтр: полужизнь OU должна попадать в [hl_min, hl_max].

ВАЖНО: это исследовательская копия, а не эталон. Расхождение с TradingView
на общем периоде — ожидаемо и измеряемо (см. критерий успеха 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.engine.exits import atr_brackets, simulate_bracket_exits
from alpha_lab.features.price import ou_params, zscore
from alpha_lab.features.volatility import atr as atr_series

DEFAULTS = {
    "window": 20,
    "k": 2.0,
    "atr_len": 14,
    "sl_atr": 2.0,
    "tp_atr": 6.0,
    "max_bars": 500,
    "use_hl_filter": False,
    "hl_min": 5.0,
    "hl_max": 100.0,
    "hl_window": 200,
}


class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, params: dict | None = None):
        cfg = {**DEFAULTS, **(params or {})}
        self.window = int(cfg["window"])
        self.k = float(cfg["k"])
        self.atr_len = int(cfg["atr_len"])
        self.sl_atr = float(cfg["sl_atr"])
        self.tp_atr = float(cfg["tp_atr"])
        self.max_bars = int(cfg["max_bars"])
        self.use_hl_filter = bool(cfg["use_hl_filter"])
        self.hl_min = float(cfg["hl_min"])
        self.hl_max = float(cfg["hl_max"])
        self.hl_window = int(cfg["hl_window"])
        self.params = cfg

    @property
    def history_bars(self) -> int:
        """Сколько хвостовых баров (включая текущий) нужно решению.

        Максимум из окон, которые решение реально использует: z-скор
        (window баров, включая текущий), ATR (atr_len) и — только при
        включённом фильтре — окно полужизни (hl_window: цикл _half_life_ok
        берёт values[i-w:i], то есть w баров строго до бара i). Выключенный
        фильтр в требование не входит.
        """
        required = max(self.window, self.atr_len)
        if self.use_hl_filter:
            required = max(required, self.hl_window)
        return required

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        z = zscore(close, self.window)
        atr_vals = atr_series(bars, self.atr_len)

        # copy=True: в pandas 3 to_numpy() отдаёт read-only массив (CoW),
        # а маски ниже дописываются на месте через &=
        long_entry = (z <= -self.k).to_numpy(copy=True)
        short_entry = (z >= self.k).to_numpy(copy=True)

        # Прогрев — явная граница входа, а не побочный эффект значений
        # признаков. zscore отдаёт 0.0 на первых window-1 барах (0.0 значит
        # «сигнала нет»), а ATR заводится SMA первых atr_len истинных
        # диапазонов и конечен только начиная с бара atr_len-1. До этой
        # границы входов нет ни при каком сигнале.
        bar_idx = np.arange(len(bars))
        ready = (bar_idx >= self.window - 1) & (bar_idx >= self.atr_len - 1)
        long_entry &= ready
        short_entry &= ready

        if self.use_hl_filter:
            allowed = self._half_life_ok(close).to_numpy()
            long_entry &= allowed
            short_entry &= allowed

        # Стоп и тейк для каждого направления; берём то, что соответствует входу
        sl_long, tp_long = atr_brackets(close, atr_vals, 1, self.sl_atr, self.tp_atr)
        sl_short, tp_short = atr_brackets(close, atr_vals, -1, self.sl_atr, self.tp_atr)

        # Никогда не входим без защиты. Для simulate_bracket_exits нефинитный
        # уровень — это «данной защиты нет»: позиция открылась бы без стопа и
        # тейка и молча держалась бы до тайм-стопа, записав неограниченный
        # убыток как обычную сделку. Позиция без стопа — не сделка этой
        # модели, поэтому сигнал на баре с нефинитным стопом или тейком
        # гасится, а не передаётся в симуляцию.
        finite_long = np.isfinite(sl_long.to_numpy()) & np.isfinite(tp_long.to_numpy())
        finite_short = (np.isfinite(sl_short.to_numpy())
                        & np.isfinite(tp_short.to_numpy()))
        long_entry &= finite_long
        short_entry &= finite_short

        entries = np.zeros(len(bars), dtype="float64")
        entries[long_entry] = 1.0
        entries[short_entry] = -1.0

        entries_series = pd.Series(entries, index=bars.index)
        sl = sl_long.where(entries_series > 0, sl_short)
        tp = tp_long.where(entries_series > 0, tp_short)

        pos = simulate_bracket_exits(bars, entries_series, sl, tp, self.max_bars)
        return pos.fillna(0.0)

    def _half_life_ok(self, close: pd.Series) -> pd.Series:
        """Полужизнь OU в скользящем окне должна лежать в допустимом диапазоне."""
        w = self.hl_window
        hl = np.full(len(close), np.nan)
        values = close.to_numpy()
        for i in range(w, len(close)):
            p = ou_params(values[i - w:i])
            hl[i] = p.half_life
        ok = (hl >= self.hl_min) & (hl <= self.hl_max)
        return pd.Series(ok, index=close.index)
