"""Стратегия сбора funding: дельта-нейтральная книга, решение — режим ставки.

ПРАВИЛО ВХОДА (зафиксировано априори, до первого прогона на данных; по
результатам не подбиралось и не менялось).

    Пусть rate[t] — ставка funding бара t (сумма событий, попавших в бар;
    на 1h это обычно одно событие из трёх за сутки, остальные бары нулевые).
    Скользящее среднее mu[t] = mean(rate[t−window+1 .. t]) — только прошлое и
    текущий бар. Порог окупаемости:

        one_way   = (fee_bps + slippage_bps) · 1e-4   — стоимость единицы
                                                       оборота одной ноги;
        round_trip = 2 ноги · 2 направления · one_way = 4 · one_way;
        threshold  = round_trip / horizon_bars         — требуемая ставка НА БАР.

    Держим carry = −1 (шорт перпа против спота), пока mu[t] > threshold;
    иначе carry = 0. net ≡ 0 всегда.

Вывод порога. Вход в дельта-нейтральную книгу торгует gross = 2 (две ноги по
|carry| = 1), выход — ещё 2; суммарный оборот круглого рейса = 4 единицы, и
издержки круглого рейса равны 4·one_way. Если держать книгу H баров, ожидаемый
funding составит H·mu. Сделка окупается при H·mu > 4·one_way, то есть при
mu > 4·one_way/H. H — объявленный горизонт удержания (horizon_bars), а не
оценка по данным: структурная премия держится неделями, и горизонт — это
часть гипотезы, а не подогнанный параметр. При дефолтах (fee 5 bps, slippage
1 bp, H = 720 баров = 30 суток на 1h) threshold = 4·6e-4/720 = 3.33e-6 на бар
≈ 2.9 % годовых (8760 баров): правило держит книгу в нормальном
положительном режиме и стоит вне рынка в придавленном.

Знак ставки — явное решение, а не молчание. Порог положителен при любых
положительных издержках, поэтому условие mu > threshold равносильно
«mu положительно И окупает издержки»: при mu ≤ 0 книга закрывается, в
отрицательном режиме стратегия не удерживается. Отрицательные ставки в
BTCUSDT — не редкость (13 % событий 2022–2025), и «держать дальше» было бы
другой сделкой: платить funding самой. Граница строгая: при mu, равном порогу,
ожидаемая прибыль ровно нулевая, и мы не платим издержки за нулевое ожидание;
при нулевых издержках порог = 0 и правило требует строго положительной ставки.
Ранние бары окна могут быть отрицательными, пока среднее ещё положительно, —
это задокументированная память окна, а не удержание отрицательного режима:
как только среднее пересекло порог вниз, позиция закрыта.

Переопределение порога и предел «держать всегда». Параметр threshold_rate
задаёт порог решения напрямую (ставка на бар): None — априорный порог
окупаемости из формулы выше (поведение S2 не меняется ни на байте), конечное
число — явный порог, -inf — предел «держать всегда» (условие mu > -inf
выполнено для любой конечной ставки: окно не читается, прогрева нет,
carry ≡ -1), +inf — симметричный предел «не входить никогда». Пределы нужны
сетке S3: always-hold — базовая книга S1, с которой сравнивается всё
остальное, и внутрь правила её иначе не поместить. Сам порог окупаемости
ограничен снизу нулём (4·one_way/H ≥ 0 при H > 0 и неотрицательных
издержках), поэтому НИКАКОЕ конечное H и даже нулевые издержки не дают
always-hold: при неположительном среднем ставки правило выходит из книги.
Измерено на данных 2022–2025: минимум скользящего среднего (window=72, 1h)
отрицателен у BTCUSDT (-4.5e-5 на 7.8 % баров), ETHUSDT, BNBUSDT (-1.8e-4 на
69 %), DOTUSDT, SOLUSDT. Значит, -inf — не аппроксимация «очень большим H», а
точка, в которой условия выхода нет по построению; «приблизить» её конечным
порогом нельзя, и это проверено тестом
(test_always_hold_limit_is_exact_and_not_reachable_by_finite_params).

Контракт «вошёл — держи — вышел» (ограничение S2). Стратегия не ребалансирует
книгу и не разворачивает carry: целевые значения — только 0 и −1, gross —
только 0 и 2, поэтому каждая заявка — это ровно вход или выход. Ограничение
намеренное: magnitude-база издержек `|Δgross|` (S1) не видит встречного
движения ног (спот 1.5 → 1.2, перп −0.5 → −0.2: net и gross стоят, а 0.6
ноционала торгуется) и не заряжает прямой разворот carry −1 → +1 (4 единицы
оборота). Это известное ограничение МОДЕЛИ ДВИЖКА, а не свойство стратегии;
S2 обходит его тем, что никогда не ребалансирует, а закрепивший его тест
(`test_magnitude_gross_undercount_is_pinned`) обязан упасть, если движок
когда-нибудь «починят» иначе. Вариант стратегии с ребалансировкой обязан
сначала пересмотреть модель издержек.

Единица сделки — открытый вопрос S4. net ≡ 0, поэтому `trade_returns` не
видит ни одной направленной сделки, и `min_trades ≥ 100` провалит такой
прогон («недостаточно сделок: 0 < 100»). Это честно: число независимых ставок
в carry-режиме считает не число баров, и выдумывать «сделки» здесь, чтобы
пройти гейт, значило бы сдать защиту от самообмана. Своя единица (эпизод
удержания) — предмет S4; S2 лишь делает ситуацию видимой, а не прячет её.

Funding-ряд приходит колонкой `funding_rate` в барах: её прикрепляет CLI
(`run_config`) стратегиям с `needs_funding = True`, и тогда причинностный
harness проверяет решение ровно на тех же барах со ставками. Без колонки
`generate_legs` падает громко: решение по ставке без ставки невозможно, а
молчаливая книга вне рынка выдала бы отсутствие данных за отсутствие сигнала.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.strategies.base import (
    FUNDING_RATE_COLUMN, PositionLegs, TwoLegStrategy,
)

# Дефолты — часть априорного правила, а не результат подбора:
#   window = 72 (три суток на 1h, ≈9 событий funding по 8 ч) — среднее, которое
#     не дёргается от одного выброса, но успевает заметить смену режима;
#   horizon_bars = 720 (30 суток на 1h) — заявленный горизонт удержания
#     структурной премии: им задаётся требуемая ставка на бар;
#   fee_bps = 5.0 — тейкерская комиссия Binance USDT-M VIP0 (как в mr_base);
#   slippage_bps = 1.0 — консервативная оценка на единицу оборота: floor
#     0.5 bps из mr_base плюс запас 0.5 bps на impact при капитале 10 000
#     (фактический impact на часовом объёме BTC много меньше). Запас вверх
#     завышает порог и занижает число входов — ошибка в безопасную сторону.
DEFAULTS = {
    "window": 72,
    "horizon_bars": 720,
    "fee_bps": 5.0,
    "slippage_bps": 1.0,
    # None — априорный порог окупаемости; число задаёт порог напрямую,
    # ±inf — пределы «держать всегда» / «не входить никогда» (см. docstring).
    "threshold_rate": None,
}

# Оборот круглого рейса в единицах |carry|: вход gross=2 и выход gross=2.
ROUND_TRIP_TURNOVER = 4.0


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} должен быть целым, получено {value!r}")
    if int(value) < 1:
        raise ValueError(f"{name} должен быть ≥ 1, получено {value!r}")
    return int(value)


def _non_negative_float(value, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(
            f"{name} должен быть конечным неотрицательным числом, "
            f"получено {value!r}"
        )
    return number


def _threshold_override(value, name: str) -> float | None:
    """Порог решения: None (априорный) или число; ±inf — законные пределы.

    NaN отвергается: сравнение с ним ложно на всех барах, и книга молча
    стояла бы вне рынка — явное «никогда не входить» должно быть +inf, а не
    следствием NaN. bool отвергается: True — это флаг, а не ставка 1.0.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)):
        raise ValueError(
            f"{name} должен быть числом или None, получено {value!r}")
    number = float(value)
    if np.isnan(number):
        raise ValueError(
            f"{name} не может быть NaN: сравнение с NaN ложно на всех барах, "
            f"и книга молча стояла бы вне рынка. «Не входить никогда» — это "
            f"+inf, «держать всегда» — -inf"
        )
    return number


class FundingHarvestStrategy(TwoLegStrategy):
    """Дельта-нейтральный сбор funding по режиму ставки.

    Решение — общее на все три базы (см. модульный docstring): net ≡ 0,
    gross = 2·|carry|, carry = −1, пока скользящее среднее ставки выше порога
    окупаемости, и 0 иначе.
    """

    name = "funding_harvest"
    # CLI обязан прикрепить к барам колонку FUNDING_RATE_COLUMN, иначе решение
    # невозможно; флаг — точка расширения протокола, не меняющая одноногие пути.
    needs_funding = True
    PARAM_NAMES = frozenset(DEFAULTS)

    def __init__(self, params: dict | None = None):
        cfg = {**DEFAULTS, **(params or {})}
        self.window = _positive_int(cfg["window"], "window")
        self.horizon_bars = _positive_int(cfg["horizon_bars"], "horizon_bars")
        self.fee_bps = _non_negative_float(cfg["fee_bps"], "fee_bps")
        self.slippage_bps = _non_negative_float(cfg["slippage_bps"], "slippage_bps")
        self.threshold_rate = _threshold_override(
            cfg["threshold_rate"], "threshold_rate")
        self.params = cfg

    @property
    def history_bars(self) -> int:
        """Скользящее среднее требует window баров, включая текущий.

        У предельных порогов (±inf) решение не читает ставку вовсе, поэтому
        памяти нет: history_bars = 1. Это честная диагностика, а не
        оптимизация: harness и отчёт не должны приписывать решению окно,
        которого оно не видит.
        """
        if self.threshold_rate is not None and not np.isfinite(self.threshold_rate):
            return 1
        return self.window

    def cost_recovery_threshold(self) -> float:
        """Требуемая ставка на бар: 4·(fee+slippage)/H (см. модульный docstring).

        Метод публичный: тесты границы и разбор отчёта обязаны брать ровно то
        число, по которому принимается решение, а не пересчитывать его заново.
        threshold_rate это число не меняет — он переопределяет точку решения
        (entry_threshold), а не модель окупаемости.
        """
        one_way = (self.fee_bps + self.slippage_bps) * 1e-4
        return ROUND_TRIP_TURNOVER * one_way / self.horizon_bars

    def entry_threshold(self) -> float:
        """Порог, по которому реально принимается решение.

        threshold_rate=None → априорный cost_recovery_threshold(); иначе явное
        значение, включая пределы ±inf. Публичный по той же причине, что и
        cost_recovery_threshold: потребитель обязан видеть ровно то число, по
        которому принято решение.
        """
        if self.threshold_rate is None:
            return self.cost_recovery_threshold()
        return float(self.threshold_rate)

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        threshold = self.entry_threshold()
        if np.isneginf(threshold):
            # Предел «держать всегда»: условия выхода нет по построению,
            # ставка не читается, прогрев окна не нужен.
            in_book = np.ones(len(bars), dtype=bool)
        elif np.isposinf(threshold):
            # Симметричный предел «не входить никогда».
            in_book = np.zeros(len(bars), dtype=bool)
        else:
            rate = self._funding_rate(bars)
            # min_periods=window: до первого полного окна статистики нет —
            # стоим. rolling смотрит только назад, поэтому решение причинно.
            trailing = rate.rolling(self.window, min_periods=self.window).mean()
            in_book = (trailing > threshold).to_numpy()
        index = bars.index
        zero = pd.Series(0.0, index=index, name="net")
        return PositionLegs(
            net=zero,
            gross=pd.Series(
                np.where(in_book, 2.0, 0.0), index=index, name="gross"),
            carry=pd.Series(
                np.where(in_book, -1.0, 0.0), index=index, name="carry"),
        )

    def _funding_rate(self, bars: pd.DataFrame) -> pd.Series:
        """Ставка из колонки funding_rate; отсутствие и NaN — громкий отказ."""
        if FUNDING_RATE_COLUMN not in bars.columns:
            raise ValueError(
                f"Стратегии '{self.name}' нужна колонка '{FUNDING_RATE_COLUMN}' "
                f"со ставкой funding: без неё решение о входе невозможно. "
                f"Колонку прикрепляет CLI (run_config) стратегиям с "
                f"needs_funding=True; при прямом вызове её обязан передать "
                f"вызывающий."
            )
        rate = pd.Series(
            bars[FUNDING_RATE_COLUMN].to_numpy(dtype="float64"),
            index=bars.index, name=FUNDING_RATE_COLUMN,
        )
        if not np.isfinite(rate.to_numpy()).all():
            first_bad = float(rate.to_numpy()[~np.isfinite(rate.to_numpy())][0])
            raise ValueError(
                f"Колонка '{FUNDING_RATE_COLUMN}' содержит нефинитные значения "
                f"(первое — {first_bad!r}): NaN в скользящем среднем молча "
                f"обнулил бы решение"
            )
        return rate
