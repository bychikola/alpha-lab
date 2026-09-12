"""Признаки волатильности и funding.

ATR считается методом Уайлдера (RMA с alpha = 1/length), как ta.atr в Pine
Script: рекурсия заводится SMA первых length TR (а не первым TR, как pandas
ewm(adjust=False)), поэтому результаты сходятся с TradingView на общем периоде.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1)
    tr = ranges.max(axis=1)
    tr.iloc[0] = bars["high"].iloc[0] - bars["low"].iloc[0]
    return tr.rename("tr")


def wilder_rma(source: pd.Series | np.ndarray, length: int) -> pd.Series:
    """Сглаживание Уайлдера (ta.rma в Pine) — одна рекурсия на весь проект.

    Определение Pine буквально:

        sum := na(sum[1]) ? ta.sma(source, length)
                          : alpha * source + (1 - alpha) * sum[1]

    то есть рекурсия заводится **средним первых length конечных значений**, а
    не первым значением (как pandas ewm(adjust=False) без затравки). Разные
    затравки дают разные уровни: этой ловушкой уже ловились ATR-стопы, а
    теперь через неё проходят TR, ±DM и DX индикатора ADX.

    Ведущие пропуски сдвигают затравку: первый результат появляется на баре,
    где впервые набралось length конечных значений окна.

    **Пропуск внутри ряда не переносит состояние через разрыв.** Pine на баре с
    пропуском получает sum = NaN и на следующем баре заводит рекурсию заново —
    через ta.sma по полному чистому окну. Здесь то же самое. Соблазн отдать это
    pandas-овскому ewm недопустим: ewm пропускает NaN и продолжает рекурсию с
    затуханием по расстоянию, то есть **протаскивает состояние сквозь дыру** —
    ровно то, что запрещает инвариант «разрывы сегментируют ряд». В рабочем
    пути полигона пропусков внутри участка не бывает (движок сегментирует ряд),
    поэтому ветка с разрывом — страховка, а не рабочий режим, и она вынесена в
    цикл, чтобы не платить за неё на каждом баре.
    """
    s = pd.Series(source).astype("float64")
    if length < 1:
        raise ValueError(f"Длина сглаживания должна быть ≥ 1, получено {length}")
    empty = pd.Series(np.nan, index=s.index, name=s.name)
    if len(s) < length:
        return empty

    # Первое окно из length конечных значений подряд — там, где ta.sma впервые
    # отдаёт число (любой NaN в окне делает сумму NaN, поэтому нужен полный
    # отрезок без пропусков).
    full = s.notna().rolling(length).sum() == length
    starts = np.flatnonzero(full.to_numpy())
    if starts.size == 0:
        return empty
    start = int(starts[0])

    finite = s.notna().to_numpy()
    if finite[start:].all():
        # Обычный случай: дальше пропусков нет, и вся рекурсия — один ewm.
        seeded = s.to_numpy(copy=True)
        seeded[:start] = np.nan
        seeded[start] = s.iloc[start - length + 1: start + 1].mean()
        # ewm(adjust=False) продолжает ровно рекурсию Уайлдера:
        # rma[i] = (1 - 1/length) * rma[i-1] + (1/length) * source[i].
        return (pd.Series(seeded, index=s.index)
                .ewm(alpha=1.0 / length, adjust=False).mean()
                .rename(s.name))

    # Есть пропуски внутри ряда: рекурсия перезаводится после каждого, как в
    # Pine. Цикл здесь допустим — этот путь не встречается в рабочих данных.
    alpha = 1.0 / length
    values = s.to_numpy()
    out = np.full(len(values), np.nan)
    run: list[float] = []
    state = np.nan
    for i, x in enumerate(values):
        if not np.isfinite(x):
            state = np.nan
            run.clear()
            continue
        run.append(float(x))
        if not np.isfinite(state):
            if len(run) >= length:
                state = float(np.mean(run[-length:]))
                out[i] = state
            continue
        state = alpha * x + (1.0 - alpha) * state
        out[i] = state
    return pd.Series(out, index=s.index, name=s.name)


def atr(bars: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR методом Уайлдера: RMA с alpha = 1/length и SMA-затравкой первых length TR."""
    # Вырожденный случай (баров меньше окна) обрабатывает wilder_rma: SMA
    # первых length TR не существует, и она отдаёт ряд NaN, а не исключение.
    return wilder_rma(true_range(bars), length).rename("atr")


def atr_zscore(bars: pd.DataFrame, atr_len: int = 14, window: int = 200) -> pd.Series:
    """Насколько текущая волатильность отклоняется от своей нормы."""
    a = atr(bars, atr_len)
    mean = a.rolling(window, min_periods=window).mean()
    std = a.rolling(window, min_periods=window).std(ddof=0)
    return ((a - mean) / std.where(std > 1e-12, np.nan)).rename("atr_z")


def volatility_regime(bars: pd.DataFrame, atr_len: int = 14, window: int = 200,
                      z_low: float = -1.0, z_high: float = 1.0) -> pd.Series:
    """Режим волатильности: -1 (тихо), 0 (норма), +1 (бурно)."""
    z = atr_zscore(bars, atr_len, window)
    out = pd.Series(0.0, index=bars.index, name="vol_regime")
    out[z >= z_high] = 1.0
    out[z <= z_low] = -1.0
    out[z.isna()] = np.nan
    return out


# Допуск остаточного влияния старого бара на ADX: 1e-3. Та же величина, что у
# ATR_DECAY_TOLERANCE в стратегии, и по той же причине: решение не должно
# зависеть от данных, чей вклад в признак уже ниже 0.1 %.
ADX_DECAY_TOLERANCE = 1e-3


def adx_decay_bars(length: int, tolerance: float = ADX_DECAY_TOLERANCE) -> int:
    """Горизонт, на котором память ADX падает ниже допуска.

    ADX — композиция двух рекурсий Уайлдера: DI± = 100·rma(±DM, L)/rma(TR, L),
    затем ADX = rma(DX, L). Импульсный отклик двух последовательных RMA на
    единичный импульс есть a²(k+1)r^k (a = 1/L, r = 1 − 1/L) — распределение
    Паскаля: его сумма по k равна единице, то есть отклик **не затухает**, а
    распределяется. Поэтому «память» здесь — не плотность, а её хвост: влия-
    ние данных возрастом m баров и старше равно

        Σ_{k ≥ m} a²(k+1)r^k = (1 + m/L) · (1 − 1/L)^m.

    Граница — **наименьшее** m, при котором этот хвост уже ниже допуска;
    решается перебором, а не подобранным литералом, поэтому растёт вместе с
    length. Однократная граница ATR (r^m) — частный случай: у неё хвост
    геометрического ядра равен самому ядру.

    Для L=14 и допуска 1e-3 граница равна 125 барам против 94 у ATR — двойная
    рекурсия помнит дольше, как и должно быть.
    """
    if length <= 1:
        # Вырожденный случай: RMA(L=1) совпадает с входом, память ровно один
        # бар, формула хвоста неприменима.
        return 1
    r = 1.0 - 1.0 / length
    m = 1
    while (1.0 + m / length) * r ** m > tolerance:
        m += 1
    return m


def adx(bars: pd.DataFrame, length: int = 14,
        smoothing: int | None = None) -> pd.Series:
    """ADX Уайлдера — сила тренда без его направления, 0…100.

    Соответствует `ta.adx(length)` в Pine, то есть `ta.dmi(length, length)`:
    сглаживание DI и сглаживание DX одной длины. Порядок операций — как в
    ta.dmi, включая нюансы:

    * ±DM на первом баре не определены (нет предыдущего) и остаются NaN;
    * знаменатель DX — `sum == 0 ? 1 : sum`: при нулевой сумме DX равен нулю,
      а не бесконечности. Сравнение именно с нулём, а не `> 0`: NaN в
      знаменателе обязан остаться NaN, иначе прогрев превратился бы в
      «тренда нет» — правдоподобное, но неверное значение;
    * первое конечное значение — на баре 2·length − 1 (прогрев двух ступеней).

    Вырожденный бар (high == low == предыдущий close) даёт trur = 0 и уводит
    ряд в NaN. Это безопасное направление: фильтр по ADX при NaN не пропускает
    вход, а не пропускает его наугад.
    """
    smooth = length if smoothing is None else smoothing
    high = pd.Series(bars["high"]).astype("float64")
    low = pd.Series(bars["low"]).astype("float64")

    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    # Первый бар: ta.change не определён, поэтому и ±DM не определены.
    plus_dm[up.isna()] = np.nan
    minus_dm[down.isna()] = np.nan

    trur = wilder_rma(true_range(bars), length)
    plus = 100.0 * wilder_rma(plus_dm, length) / trur
    minus = 100.0 * wilder_rma(minus_dm, length) / trur

    total = plus + minus
    dx = 100.0 * (plus - minus).abs() / total.where(total != 0, 1.0)
    return wilder_rma(dx, smooth).rename("adx")


def funding_features(rate: pd.Series, window: int = 90) -> pd.DataFrame:
    """Скользящее среднее funding и его z-скор.

    Положительный funding означает, что лонги платят шортам — держать лонг дорого.
    """
    r = pd.Series(rate).astype("float64")
    ma = r.rolling(window, min_periods=1).mean()
    std = r.rolling(window, min_periods=window).std(ddof=0)
    z = ((r - ma) / std.where(std > 1e-12, np.nan)).fillna(0.0)
    return pd.DataFrame({"funding_ma": ma, "funding_z": z})
