"""MR с фильтрами режима: ADX, эмпирический квантиль, сжатие волатильности.

Откуда взялось. В чужом индикаторе (Dynamic Bollinger Bands) нашлись три идеи,
которые стоит не «внедрить», а проверить. Внедрять чужие фильтры без прогона —
ровно тот способ обмануть себя, против которого построен полигон: пять методов
× шесть множителей × периоды и пороги дают тысячи комбинаций без единого теста,
а из 500 вариантов лучший по Sharpe показывает √(2·ln 500) ≈ 3.5 даже на чистом
шуме. Поэтому здесь они лежат **параметрами**, а не улучшениями:

* ``adx_max`` / ``adx_min`` — вход только при слабом (или только при сильном)
  тренде. Проверяемая гипотеза H1.
* ``quantile_window`` / ``coverage`` — порог входа по эмпирическому квантилю
  распределения z вместо k·σ. Гипотеза H2.
* ``squeeze_ratio`` — вход только когда волатильность ниже своей нормы.
  Гипотеза H3.

Основание для H1 — измеренное, а не литературное. В фазе 2 замерено: вход
|z| ≥ 2 **не различает** OU-процесс и случайное блуждание (на блуждании сигналов
даже больше — 11.6 % против 9.3 %). То есть сам порог не является детектором
возврата к среднему, и включение MR только там, где рынок действительно
возвращается, — осмысленная попытка это починить.

Основание для H2 — тяжёлые хвосты. z считается в предположении нормальности:
«две сигмы» покрывают 95 % только у нормального распределения. Эмпирический
квантиль берёт настоящее распределение окна. В чужом коде метод сломан: там
эмпирический 90 %-диапазон сравнивается с теоретическим 2·k·σ, то есть с 4σ =
95.4 %, и порог занижен на 18 % даже на идеально нормальных данных. Здесь
теоретического плеча нет вовсе: порог И ЕСТЬ квантиль.

Инвариант: **выключенные фильтры не меняют ничего**. ``generate`` при пустых
настройках побитово равен MeanReversionStrategy — без этого разница в вердикте
не была бы свойством фильтра.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.features.price import rolling_sigma
from alpha_lab.features.volatility import adx, adx_decay_bars
from alpha_lab.strategies.mean_reversion import (
    ATR_DECAY_TOLERANCE, MeanReversionStrategy,
)

# Фильтры выключены значением None (у порогов в единицах индикатора) или нулём
# (у длин окон). Ноль как «выключено» выбран там, где окно нулевой длины не
# имеет смысла: отдельного флага не нужно, а сетка может перебирать длины, не
# заводя ось «включено/выключено».
FILTER_DEFAULTS = {
    "adx_period": 14,
    "adx_max": None,        # вход только при ADX < adx_max (None — не фильтровать)
    "adx_min": None,        # вход только при ADX > adx_min
    "quantile_window": 0,   # 0 — порог k·σ; иначе — эмпирический квантиль
    "coverage": 95.0,       # покрытие двустороннего интервала, %
    "squeeze_window": 20,
    "squeeze_ratio": None,  # σ < sma(σ, окно)·ratio (None — не фильтровать)
}


class MeanReversionFilteredStrategy(MeanReversionStrategy):
    """MR с фильтрами режима. Все фильтры выключены по умолчанию."""

    name = "mean_reversion_filtered"
    # Слияние параметров в __init__ идёт через type(self).defaults, поэтому
    # расширенный набор переопределяет именно этот атрибут.
    defaults = {**MeanReversionStrategy.defaults, **FILTER_DEFAULTS}
    PARAM_NAMES = frozenset(defaults)

    def __init__(self, params: dict | None = None):
        super().__init__(params)
        cfg = self.params
        self.adx_period = int(cfg["adx_period"])
        self.adx_max = None if cfg["adx_max"] is None else float(cfg["adx_max"])
        self.adx_min = None if cfg["adx_min"] is None else float(cfg["adx_min"])
        self.quantile_window = int(cfg["quantile_window"])
        self.coverage = float(cfg["coverage"])
        self.squeeze_window = int(cfg["squeeze_window"])
        self.squeeze_ratio = (None if cfg["squeeze_ratio"] is None
                              else float(cfg["squeeze_ratio"]))

    @property
    def history_bars(self) -> int:
        """Память базовой модели, расширенная включёнными фильтрами.

        Каждый фильтр добавляет своё требование, и только включённый:

        * ADX — двойная рекурсия Уайлдера, её горизонт длиннее однократного
          (см. adx_decay_bars): 125 баров против 94 для L=14;
        * квантиль — окно квантиля поверх window (прогревочные нули z в окно
          не берутся, поэтому нужен ещё window баров настоящих значений);
        * сжатие — sma по σ, а σ сама считается по окну window.
        """
        required = super().history_bars
        if self.adx_max is not None or self.adx_min is not None:
            required = max(
                required, adx_decay_bars(self.adx_period, ATR_DECAY_TOLERANCE))
        if self.quantile_window > 0:
            required = max(required, self.window + self.quantile_window - 1)
        if self.squeeze_ratio is not None:
            required = max(required, self.window + self.squeeze_window - 1)
        return required

    def _signal_thresholds(self, bars: pd.DataFrame, close: pd.Series,
                           z: pd.Series):
        """Порог входа: k·σ (числа) либо эмпирический квантиль (ряды).

        Квантиль считается по окну, оканчивающемуся ТЕКУЩИМ баром: z уже
        построен по информацию до t включительно, поэтому текущее значение —
        не подглядывание, а часть доступной выборки (так же устроен
        ta.percentile_nearest_rank в Pine, который включает текущий бар).

        Прогревочные нули zscore в выборку не берутся. Ноль там означает
        «окно ещё не набрано», а не «ряд ровно на среднем» (см. zscore), и
        окно, набранное из заглушек, сжало бы порог к нулю — стратегия
        торговала бы по несуществующему распределению. Поэтому до бара
        window + quantile_window − 2 порог равен NaN, а сравнение с NaN ложно:
        вход запрещён.
        """
        if self.quantile_window <= 0:
            return super()._signal_thresholds(bars, close, z)

        alpha = (1.0 - self.coverage / 100.0) / 2.0
        real = np.where(np.arange(len(z)) >= self.window - 1,
                        z.to_numpy(), np.nan)
        window = self.quantile_window
        rolling = pd.Series(real, index=z.index).rolling(
            window, min_periods=window)
        return rolling.quantile(alpha), rolling.quantile(1.0 - alpha)

    def _entry_masks(self, bars: pd.DataFrame, close: pd.Series, z: pd.Series,
                     ready: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        long_entry, short_entry = super()._entry_masks(bars, close, z, ready)
        if (self.adx_max is None and self.adx_min is None
                and self.squeeze_ratio is None):
            return long_entry, short_entry
        allowed = self._regime_allowed(bars, close).to_numpy()
        return long_entry & allowed, short_entry & allowed

    def _regime_allowed(self, bars: pd.DataFrame,
                        close: pd.Series) -> pd.Series:
        """Бары, на которых режим допускает вход. Недосчитанный фильтр — запрет.

        Сравнение с NaN ложно, поэтому прогрев любого фильтра автоматически
        становится запретом входа. Это безопасное направление: пропустить
        вход дёшево, открыть его по несуществующему фильтру — нет. Обратный
        выбор («пока не знаем — не фильтруем») тихо торговал бы ровно на тех
        барах, где фильтр ещё не работает.
        """
        allowed = pd.Series(True, index=bars.index)

        if self.adx_max is not None or self.adx_min is not None:
            strength = adx(bars, self.adx_period)
            if self.adx_max is not None:
                allowed &= strength < self.adx_max
            if self.adx_min is not None:
                allowed &= strength > self.adx_min

        if self.squeeze_ratio is not None:
            # σ — ровно та же величина, что стоит в полосе k·σ: та же функция,
            # что внутри zscore, поэтому ширина полосы и её средняя сравнимы.
            sigma = rolling_sigma(close, self.window)
            norm = sigma.rolling(self.squeeze_window,
                                 min_periods=self.squeeze_window).mean()
            allowed &= sigma < norm * self.squeeze_ratio

        return allowed
