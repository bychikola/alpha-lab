import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_bars

from alpha_lab.features.volatility import (
    ADX_DECAY_TOLERANCE, adx, adx_decay_bars, atr, atr_zscore, funding_features,
    true_range, volatility_regime, wilder_rma,
)


def test_atr_matches_manual_true_range():
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC"),
        "open": [10.0, 11.0, 12.0, 13.0],
        "high": [11.0, 13.0, 14.0, 15.0],
        "low": [9.0, 10.0, 11.0, 12.0],
        "close": [10.5, 12.5, 13.5, 14.5],
        "volume": 1.0, "quote_volume": 1.0, "trades": 1, "taker_buy_volume": 1.0,
    })

    a = atr(bars, length=3)
    tr = true_range(bars)

    # TR: [2.0, 3.0, 3.0, 3.0]. Уайлдер (ta.rma в Pine) заводит рекурсию не с
    # первого TR, как pandas ewm(adjust=False), а со SMA первых length значений:
    # первое значение — на баре length-1 (индекс 2) и равно mean(TR[:3]) = 8/3,
    # далее rma[i] = 2/3*rma[i-1] + 1/3*tr[i]: 8/3 -> 25/9.
    assert a.iloc[:2].isna().all()
    # Ключевое свойство ta.rma: первый валидный ATR — это среднее первых length TR.
    assert a.iloc[2] == pytest.approx(tr.iloc[:3].mean())
    assert a.iloc[2] == pytest.approx(8 / 3)
    assert a.iloc[3] == pytest.approx(25 / 9)
    # Контроль: это не SMA последних TR (она дала бы ровно 3.0 по построению
    # данных). Уайлдер сглаживает хвост, поэтому ATR(3) на баре 3 ниже.
    assert a.iloc[3] < 3.0 - 0.2


def test_atr_is_positive():
    a = atr(ou_bars(n=500), length=14).dropna()

    assert (a > 0).all()


def test_atr_shorter_than_length_is_all_nan():
    bars = ou_bars(n=10)

    a = atr(bars, length=14)

    # Вырожденный случай: баров меньше окна — не исключение, а весь ряд NaN
    # (SMA-затравку длины length построить нельзя).
    assert len(a) == len(bars)
    assert a.index.equals(bars.index)
    assert a.isna().all()


def test_atr_zscore_centered():
    z = atr_zscore(ou_bars(n=2000), atr_len=14, window=200).dropna()

    assert abs(z.mean()) < 0.3
    assert abs(z.std() - 1.0) < 0.4


def test_atr_zscore_constant_bars_is_nan():
    # sigma=0 -> все OHLC равны, TR и ATR тождественно нулевые, std(ATR) = 0.
    # Гвардия std > 1e-12 обязана отдать NaN, а не 0/0 или inf.
    atr_len, window = 3, 20
    bars = ou_bars(n=60, sigma=0.0)

    a = atr(bars, length=atr_len)
    # ATR за прогревом определён и ровно нулевой — значит, rolling std тоже
    # определён и равен нулю, то есть NaN даёт именно гвардия, а не прогрев.
    assert (a.iloc[atr_len - 1:] == 0.0).all()

    z = atr_zscore(bars, atr_len=atr_len, window=window)

    warmup_end = (atr_len - 1) + (window - 1)  # первый бар, где std посчитан
    assert z.iloc[warmup_end:].isna().all()
    assert z.notna().sum() == 0


def test_volatility_regime_values():
    bars = ou_bars(n=2000)

    reg = volatility_regime(bars, atr_len=14, window=200, z_low=-1.0, z_high=1.0)

    assert set(reg.dropna().unique()).issubset({-1.0, 0.0, 1.0})
    # На этом ряду встречаются все три режима: тождественный ноль прошёл бы
    # проверку подмножества, но не равенство.
    assert set(reg.dropna().unique()) == {-1.0, 0.0, 1.0}


def test_funding_features_zscore_of_constant_is_zero():
    rate = pd.Series([0.0001] * 100)

    out = funding_features(rate, window=20)

    # eq, а не fillna(0.0) == 0.0: последнее прошло бы и на ряде из одних NaN.
    assert out["funding_z"].eq(0.0).all()
    assert out["funding_z"].notna().all()
    assert out["funding_ma"].iloc[-1] == pytest.approx(0.0001)


def test_funding_features_detects_positive_bias():
    rate = pd.Series(np.r_[np.full(100, 0.0001), np.full(100, 0.001)])

    out = funding_features(rate, window=50)

    # Бар 100 — первый бар нового режима. Окно 51..100: 49 значений 1e-4 и одно
    # 1e-3, поэтому ma = 1.18e-4, sigma(ddof=0) = 1.26e-4,
    # z = (1e-3 - 1.18e-4) / 1.26e-4 = 7.0.
    assert out["funding_z"].iloc[100] == pytest.approx(7.0)
    # К концу нового блока окно целиком лежит в уровне 1e-3: sigma = 0 -> z = 0.
    # Это адаптация окна, а не потеря сигнала: скачок уже позади.
    assert out["funding_z"].iloc[-1] == 0.0


# ═════════════════════════ ADX (Уайлдер) ═══════════════════════════════════
#
# ADX — индикатор силы тренда: композиция ТРЁХ рекурсий Уайлдера
# (TR, ±DM — два RMA, затем сглаживание DX). Затравка у каждой — SMA первых
# length конечных значений (ta.rma в Pine). Разбирать эту цепочку нужно по
# частям: ошибка в любой из трёх даёт правдоподобный, но неверный индикатор.


def _frame(high, low, close) -> pd.DataFrame:
    high = np.asarray(high, dtype="float64")
    low = np.asarray(low, dtype="float64")
    close = np.asarray(close, dtype="float64")
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=len(close), freq="1min",
                            tz="UTC"),
        "open": np.r_[close[0], close[:-1]],
        "high": high, "low": low, "close": close,
        "volume": 1.0, "quote_volume": 1.0, "trades": 1, "taker_buy_volume": 1.0,
    })


def _one_way_bars(n=120, step=1.0) -> pd.DataFrame:
    """Ряд с односторонним движением: +DM > 0 и −DM ≡ 0 (или наоборот).

    close растёт на step, high — на тот же шаг выше close, low — на прошлом
    уровне. Тогда up = step > 0, а down = 0, значит максимум одно из
    направленных движений ненулевое: DI одного знака равен нулю, и
    DX = 100·|DI+ − DI−| / (DI+ + DI−) = 100 ровно, а с ним и ADX.
    """
    close = 100.0 + step * np.arange(n)
    high = close + step
    low = np.full(n, 99.0)
    return _frame(high, low, close)


def test_adx_is_exactly_100_on_a_one_way_trend():
    """Точная проверка всей цепочки: TR → ±DM → DI → DX → ADX.

    На одностороннем движении DI противоположного знака равен нулю, поэтому
    отношение в DX сокращается в единицу. Это не «примерно высоко»: значение
    обязано быть ровно 100 (с точностью сложения/деления), и любая ошибка в
    знаках ±DM, в порядке деления на trur или в сглаживании DX его сдвинет.
    """
    bars = _one_way_bars(n=120)
    a = adx(bars, length=14)

    assert a.iloc[:27].isna().all()          # прогрев: 2·14 − 1 = 27
    assert a.iloc[27:].notna().all()
    assert (a.iloc[27:] - 100.0).abs().max() < 1e-6


def test_adx_ignores_trend_direction():
    """ADX измеряет СИЛУ тренда, а не его знак: падение даёт те же 100.

    Зеркальный ряд с падающими low и неподвижным high даёт −DM > 0, +DM ≡ 0,
    то есть противоположный знак DI — а значение то же.
    """
    n = 120
    close = 200.0 - np.arange(n, dtype="float64")
    high = np.full(n, 201.0)
    low = close - 1.0
    bars = _frame(high, low, close)

    a = adx(bars, length=14)

    assert (a.iloc[27:] - 100.0).abs().max() < 1e-6


def test_adx_is_low_where_there_is_no_trend():
    """Обратная сторона: без направленного движения ADX мал.

    Односторонний ряд даёт 100 (тест выше); колеблющийся — заметно меньше.
    Без этой половины тест не отличал бы «силу тренда» от константы 100.
    """
    a = adx(ou_bars(n=5000, theta=0.10, seed=3), length=14).dropna()

    assert a.median() < 40.0


def test_adx_is_bounded():
    a = adx(ou_bars(n=3000, seed=5), length=14).dropna()

    assert (a >= 0.0).all() and (a <= 100.0).all()


def test_adx_matches_naive_reference():
    """Векторная реализация обязана совпасть с дословным циклом.

    Эталон ниже — построчная запись ta.dmi из документации Pine: наивный
    цикл, отдельная функция rma на каждую рекурсию. Он медленный и служит
    спецификацией: при расхождении права наивная версия (она повторяет Pine
    буквально), а не быстрая. Проверяется, что ускорение не изменило смысл.
    """
    bars = ou_bars(n=600, theta=0.05, seed=17)
    fast = adx(bars, length=14)
    slow = _adx_reference(bars, length=14, smoothing=14)

    diff = (fast - slow).abs().max()
    assert np.isfinite(slow.iloc[27:]).all()
    assert diff < 1e-12, f"расхождение с наивным эталоном: {diff}"


def _adx_reference(bars: pd.DataFrame, length: int, smoothing: int) -> pd.Series:
    """Дословная запись ta.dmi(length, smoothing) наивным циклом."""
    n = len(bars)
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")

    tr = np.full(n, np.nan)
    plus_dm = np.full(n, np.nan)
    minus_dm = np.full(n, np.nan)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        prev = close[i - 1]
        tr[i] = max(high[i] - low[i], abs(high[i] - prev), abs(low[i] - prev))

    def rma(x: np.ndarray, ln: int) -> np.ndarray:
        out = np.full(n, np.nan)
        seeded = False
        for i in range(n):
            if not seeded:
                window = x[max(0, i - ln + 1):i + 1]
                if i >= ln - 1 and np.isfinite(window).all():
                    out[i] = window.mean()
                    seeded = True
                continue
            out[i] = (1.0 - 1.0 / ln) * out[i - 1] + (1.0 / ln) * x[i]
        return out

    trur = rma(tr, length)
    plus = 100.0 * rma(plus_dm, length) / trur
    minus = 100.0 * rma(minus_dm, length) / trur
    dx = np.full(n, np.nan)
    for i in range(n):
        if not (np.isfinite(plus[i]) and np.isfinite(minus[i])):
            continue
        total = plus[i] + minus[i]
        dx[i] = 100.0 * abs(plus[i] - minus[i]) / (1.0 if total == 0 else total)
    return pd.Series(rma(dx, smoothing), index=bars.index)


def test_adx_is_truncation_invariant():
    """Причинность: хвост ряда не влияет на прошлые значения.

    Затравки всех трёх рекурсий берутся из НАЧАЛА ряда, поэтому обрезка справа
    их не меняет. Это ровно то свойство, которое требует harness причинности
    от стратегии: если ADX им не обладает, фильтр по нему невалидируем.
    """
    bars = ou_bars(n=400, seed=21)
    full = adx(bars, length=14)

    for k in (100, 200, 300, 399):
        part = adx(bars.iloc[:k], length=14)
        assert part.index.equals(bars.index[:k])
        both = pd.concat([part, full.iloc[:k]], axis=1)
        assert (both.iloc[:, 0] - both.iloc[:, 1]).abs().max() < 1e-15


def test_wilder_rma_matches_atr():
    """Затравка RMA — общая для ATR и ADX, и это не совпадение.

    Обе рекурсии обязаны заводиться одинаково (SMA первых length значений).
    Разные затравки — та самая ловушка, что стоила вечера при переносе ATR в
    Python: уровни стопа разъезжались, потому что seed был разным.
    """
    bars = ou_bars(n=400, seed=8)

    assert wilder_rma(true_range(bars), 14).equals(atr(bars, 14))


def test_wilder_rma_does_not_carry_state_across_a_gap():
    """Пропуск внутри ряда обрывает рекурсию, а не протаскивает её сквозь дыру.

    Соблазн отдать это pandas-овскому ewm недопустим: ewm пропускает NaN и
    продолжает с затуханием по расстоянию — состояние переносится через
    разрыв, ровно то, что запрещено инвариантом «разрывы сегментируют ряд».
    Pine же на баре с пропуском получает NaN и перезаводится по полному
    чистому окну. Проверяем второе.
    """
    x = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0])

    out = wilder_rma(x, 3)

    assert out.iloc[2] == pytest.approx(2.0)              # mean(1,2,3)
    assert out.iloc[3] == pytest.approx(2.0 * 2 / 3 + 4.0 / 3)

    # Окна [4, NaN, 6], [NaN, 6, 7] неполны — рекурсия не заведена.
    assert out.iloc[4:7].isna().all()

    # Первое полное чистое окно после дыры — [6, 7, 8]: рекурсия заводится
    # заново, старым состоянием не сдабривается.
    assert out.iloc[7] == pytest.approx(7.0)
    assert out.iloc[8] == pytest.approx(2.0 * 7 / 3 + 9.0 / 3)


def test_adx_decay_bars_is_the_smallest_solution():
    """Граница памяти выводится из рекурсии, а не подбирается литералом.

    Импульсный отклик двух RMA — распределение Паскаля a²(k+1)r^k с суммой 1,
    поэтому память задаётся его ХВОСТОМ (1 + m/L)·r^m, а не плотностью: брать
    «наименьшее m, где плотность мала» нельзя — у плотности нет предела
    малости при больших L. Проверяем минимальность хвостовой границы.
    """
    tol = ADX_DECAY_TOLERANCE
    assert 0.0 < tol < 0.01

    previous = 0
    for length in (2, 7, 14, 50):
        m = adx_decay_bars(length, tol)
        r = 1.0 - 1.0 / length
        assert (1.0 + m / length) * r ** m <= tol        # хвост уже под допуском
        assert (1.0 + (m - 1) / length) * r ** (m - 1) > tol   # m−1 ещё нет
        assert m >= 2 * length - 1                       # не короче прогрева
        assert m > previous                              # монотонно по длине
        previous = m

    assert adx_decay_bars(1) == 1                       # вырожденный случай
