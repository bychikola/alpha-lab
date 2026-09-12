"""Проверка верности порта модели в Pine.

Pine запустить из Python нельзя, поэтому здесь построчно повторяется логика
pine/mr_quant_v2.pine — ровно в том порядке и с теми же формулами, что в
скрипте, — и результат сверяется с настоящей моделью
(alpha_lab.strategies.mean_reversion.MeanReversionStrategy).

Смысл проверки: индикатор, разошедшийся с моделью хотя бы на бар, показывает
не то, что проверялось в бэктесте. Расхождение здесь — дефект порта, а не
особенность реализации.

Запуск:
    uv run python scripts/verify_pine_port.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from fixtures.synthetic import ou_bars  # noqa: E402

from alpha_lab.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402


def pine_port(bars: pd.DataFrame, win_len: int = 20, k_entry: float = 2.0,
              atr_len: int = 14, sl_atr: float = 2.0, tp_atr: float = 6.0,
              max_bars: int = 500) -> pd.Series:
    """Построчный эквивалент pine/mr_quant_v2.pine.

    Порядок операций внутри бара повторяет Pine: сначала возможный вход,
    затем проверка стопа и тейка в ТОМ ЖЕ баре.
    """
    close = bars["close"].astype("float64")
    high = bars["high"].astype("float64")
    low = bars["low"].astype("float64")
    n = len(bars)

    # ── ② ЯДРО: μ и σ, генеральная дисперсия (ddof = 0) ──────────────────
    mean = close.rolling(win_len).mean()
    mean_sq = (close * close).rolling(win_len).mean()
    variance = mean_sq - mean * mean
    sigma = np.sqrt(np.maximum(variance, 0.0))
    z = np.where((~sigma.isna()) & (sigma > 1e-12), (close - mean) / sigma, 0.0)
    z = np.where(np.isnan(z), 0.0, z)

    # ── ATR Уайлдера: ta.atr = ta.rma(ta.tr(true), len) ──────────────────
    # Затравка ta.rma — СРЕДНЕЕ первых len истинных диапазонов, а не первый
    # TR. Это не деталь: разные начальные условия дают разные уровни стопа и
    # разные сделки. Та же ошибка ловилась при переносе ATR в Python (Task 7).
    prev_close = close.shift(1)
    tr = pd.concat([high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    tr.iloc[0] = high.iloc[0] - low.iloc[0]
    seeded = tr.copy()
    seeded.iloc[:atr_len - 1] = np.nan
    seeded.iloc[atr_len - 1] = tr.iloc[:atr_len].mean()
    atr = seeded.ewm(alpha=1.0 / atr_len, adjust=False).mean()

    # ── ③ УРОВНИ И СИГНАЛЫ ──────────────────────────────────────────────
    sl_long = close - sl_atr * atr
    tp_long = close + tp_atr * atr
    sl_short = close + sl_atr * atr
    tp_short = close - tp_atr * atr

    ready = np.arange(n) >= max(win_len, atr_len) - 1
    long_ok = ready & sl_long.notna().to_numpy() & tp_long.notna().to_numpy()
    short_ok = ready & sl_short.notna().to_numpy() & tp_short.notna().to_numpy()
    want_long = long_ok & (z <= -k_entry)
    want_short = short_ok & (z >= k_entry)

    c = close.to_numpy()
    h = high.to_numpy()
    lo = low.to_numpy()
    sll, tpl = sl_long.to_numpy(), tp_long.to_numpy()
    sls, tps = sl_short.to_numpy(), tp_short.to_numpy()

    # ── ④ ПОСЛЕДОВАТЕЛЬНАЯ СИМУЛЯЦИЯ ВЫХОДОВ ────────────────────────────
    pos = np.zeros(n, dtype="float64")
    g_dir, g_entry, g_stop, g_take, g_bar = 0, 0.0, 0.0, 0.0, -1

    for i in range(n):
        if g_dir == 0 and want_long[i]:
            g_dir, g_bar = 1, i
            g_entry, g_stop, g_take = c[i], sll[i], tpl[i]
        elif g_dir == 0 and want_short[i]:
            g_dir, g_bar = -1, i
            g_entry, g_stop, g_take = c[i], sls[i], tps[i]

        if g_dir != 0:
            hit_sl = lo[i] <= g_stop if g_dir > 0 else h[i] >= g_stop
            hit_tp = h[i] >= g_take if g_dir > 0 else lo[i] <= g_take
            time_up = (i - g_bar) >= max_bars
            if hit_sl or time_up:
                g_dir = 0
            elif hit_tp:
                g_dir = 0

        pos[i] = g_dir          # posNow = gDir в Pine

    return pd.Series(pos, index=bars.index, name="position")


def main() -> int:
    # Консоль Windows по умолчанию не в UTF-8 и падает на греческих буквах
    # (θ) в метках фикстур. Тот же приём, что в cli.py.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    print("Проверка порта Pine против Python-модели\n")

    cases = [
        ("OU θ=0.10 n=5000 seed=13", dict(n=5000, theta=0.10, sigma=1.0, seed=13)),
        ("OU θ=0.05 n=5000 seed=7", dict(n=5000, theta=0.05, sigma=1.0, seed=7)),
        ("OU θ=0.02 n=20000 seed=42", dict(n=20000, theta=0.02, sigma=1.0, seed=42)),
        ("OU θ=0.30 n=3000 seed=99", dict(n=3000, theta=0.30, sigma=1.0, seed=99)),
    ]

    all_ok = True
    for label, kw in cases:
        bars = ou_bars(**kw)
        truth = MeanReversionStrategy(
            {"window": 20, "k": 2.0, "atr_len": 14, "sl_atr": 2.0, "tp_atr": 6.0,
             "max_bars": 500, "use_hl_filter": False}
        ).generate(bars)
        port = pine_port(bars)

        t = truth.to_numpy()
        p = port.to_numpy()
        diff = int((t != p).sum())
        first = int(np.argmax(t != p)) if diff else -1
        n_trades = int((np.diff(np.concatenate([[0.0], t])) != 0).sum())

        status = "СОВПАЛО" if diff == 0 else f"РАСХОЖДЕНИЕ в {diff} барах (первый: {first})"
        print(f"  {label:<30} сделок={n_trades:<5} {status}")
        if diff:
            all_ok = False
            lo = max(0, first - 2)
            for i in range(lo, min(len(t), first + 3)):
                print(f"      бар {i}: модель={t[i]:+.0f} порт={p[i]:+.0f}")

    print()
    if all_ok:
        print("ИТОГ: порт верен — позиции совпадают побитово на всех фикстурах.")
        print("      Индикатор показывает ровно то, что проверялось в бэктесте.")
    else:
        print("ИТОГ: порт РАСХОДИТСЯ с моделью. Индикатор показывал бы не то,")
        print("      что тестировалось. Чинить pine/mr_quant_v2.pine.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
