import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_bars

from alpha_lab.features.volatility import (
    atr, atr_zscore, funding_features, true_range, volatility_regime,
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
