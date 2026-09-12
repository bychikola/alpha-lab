"""Mean Reversion Quant Pro — перенос Pine Script (index.html) в Python.

Источник: блок pine-strategy, секция «ЯДРО МОДЕЛИ».
Логика входа: z ≤ −k → лонг, z ≥ +k → шорт.
Выход: стоп 2×ATR, тейк 6×ATR, принудительный выход по времени.
Опциональный фильтр: полужизнь OU должна попадать в [hl_min, hl_max].

ВАЖНО: это исследовательская копия, а не эталон. Расхождение с TradingView
на общем периоде — ожидаемо и измеряемо (см. критерий успеха 0).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alpha_lab.engine.exits import atr_brackets, simulate_bracket_exits
from alpha_lab.features.price import ou_params, zscore
from alpha_lab.features.volatility import atr as atr_series

# Допуск остаточного влияния старого TR на ATR: 1e-3 (0.1 % от значения ATR).
# При типичном для 1h ATR/price ≈ 1 % это ≈0.1 bps цены — меньше минимального
# проскальзывания (0.5 bps) и на полтора порядка меньше комиссии тейкера
# (5 bps), то есть в масштабе исполнения решение уже не отличается от честного
# пересчёта. Ужесточение до 1e-4/1e-6 не меняло бы решения, но удлиняло маску
# разрыва (125/187 баров при atr_len=14).
ATR_DECAY_TOLERANCE = 1e-3


def atr_decay_bars(atr_len: int, tolerance: float = ATR_DECAY_TOLERANCE) -> int:
    """Горизонт, на котором память ATR Уайлдера падает ниже допуска.

    ATR — рекурсия RMA с одним seed:
    ``rma[i] = (1 − 1/L)·rma[i−1] + (1/L)·tr[i]``. Вклад TR бара, отстоящего
    на k шагов назад, равен ``(1 − 1/L)^k``: память бесконечна, а не L баров.
    Поэтому объявлять историей ATR его длину нельзя — через L баров в ATR
    остаётся ``(1 − 1/L)^L ≈ e^-1 = 37 %`` прежних данных (для L=14:
    ``(13/14)^14 = 35.5 %``), а через 20 баров — ещё 22.7 %.

    Граница k выбирается из ``(1 − 1/L)^k ≤ tolerance``:
    ``k = ceil(ln(tolerance) / ln(1 − 1/L))``. Для L=14 и допуска 1e-3 это 94
    бара (``(13/14)^94 = 9.4e-4``). L=1 — вырожденный случай: RMA совпадает
    с TR, память ровно один бар, формула неприменима.

    Значение выводится из параметра, а не подбирается: при росте atr_len
    требование растёт вместе с памятью рекурсии.
    """
    if atr_len <= 1:
        return 1
    return int(math.ceil(math.log(tolerance) / math.log1p(-1.0 / atr_len)))


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
    # Пространство параметров — для сетки гипотез (grid.py спрашивает его у
    # класса). Выводится из DEFAULTS, чтобы новый параметр не забывался в
    # списке: ключ дефолтов и есть принятое имя.
    PARAM_NAMES = frozenset(DEFAULTS)
    # Значения по умолчанию — атрибут класса, а не только модульная константа:
    # наследник расширяет набор параметров, и слияние обязано взять ИМЕННО его
    # набор. Иначе параметры наследника молча не заполнялись бы дефолтами и
    # он падал бы на KeyError — или, хуже, брал бы базовые значения.
    defaults = DEFAULTS

    def __init__(self, params: dict | None = None):
        cfg = {**type(self).defaults, **(params or {})}
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

        Максимум из всего, от чего решение реально зависит:

        * z-скор — window баров, включая текущий;
        * ATR — atr_decay_bars(atr_len), а не atr_len: рекурсия Уайлдера с
          одним seed помнит бесконечно, и остаточное влияние старого TR на
          объявленном горизонте обязано быть ниже ATR_DECAY_TOLERANCE (для
          atr_len=14 это 94 бара против прежних 20, где ещё сидело 22.7 %
          предразрывной волатильности);
        * окно полужизни — только при включённом фильтре (hl_window: цикл
          _half_life_ok берёт values[i-w:i], то есть w баров строго до бара i);
          выключенный фильтр в требование не входит.
        """
        required = max(self.window, self.atr_len, atr_decay_bars(self.atr_len))
        if self.use_hl_filter:
            required = max(required, self.hl_window)
        return required

    def _signal_thresholds(self, bars: pd.DataFrame, close: pd.Series,
                           z: pd.Series):
        """Пороги z: ниже нижнего — лонг, выше верхнего — шорт.

        По умолчанию константы ±k. Точка расширения для вариантов с
        адаптивным порогом: там возвращаются не числа, а ряды, потому что
        порог меняется от бара к бару. Возвращаются именно ПОРОГИ, а не
        готовые маски, — чтобы смысл знака («ниже нижнего — это лонг»)
        остался в одном месте и переопределение не могло его перепутать.
        """
        return -self.k, self.k

    def _entry_masks(self, bars: pd.DataFrame, close: pd.Series, z: pd.Series,
                     ready: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Маски входа (лонг, шорт) до страховки от нефинитных уровней.

        ready — прогревочная граница: до неё входов нет ни при каком сигнале.
        Маски возвращаются writable-копиями: фильтрующие варианты дописывают
        их на месте через &=, а to_numpy() в pandas 3 отдаёт read-only массив.
        """
        low, high = self._signal_thresholds(bars, close, z)
        long_entry = (z <= low).to_numpy(copy=True)
        short_entry = (z >= high).to_numpy(copy=True)
        long_entry &= ready
        short_entry &= ready

        if self.use_hl_filter:
            allowed = self._half_life_ok(close).to_numpy()
            long_entry &= allowed
            short_entry &= allowed

        return long_entry, short_entry

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        z = zscore(close, self.window)
        atr_vals = atr_series(bars, self.atr_len)

        # Прогрев — явная граница входа, а не побочный эффект значений
        # признаков. zscore отдаёт 0.0 на первых window-1 барах (0.0 значит
        # «сигнала нет»), а ATR заводится SMA первых atr_len истинных
        # диапазонов и конечен только начиная с бара atr_len-1. До этой
        # границы входов нет ни при каком сигнале.
        bar_idx = np.arange(len(bars))
        ready = (bar_idx >= self.window - 1) & (bar_idx >= self.atr_len - 1)
        long_entry, short_entry = self._entry_masks(bars, close, z, ready)

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
