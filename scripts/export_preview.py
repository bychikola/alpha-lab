"""Выгрузка данных модели для превью на полигоне.

Полигон показывает, что индикатор ДОЛЖЕН нарисовать. Считает это не Pine
(его нельзя исполнить в браузере), а проверенная Python-модель — та самая,
что совпадает с Pine-портом побитово (scripts/verify_pine_port.py).

Отсюда и ценность превью: это эталон, с которым потом сверяется индикатор
в TradingView, а не приблизительная картинка.

Запуск:
    uv run python scripts/export_preview.py --symbol BTCUSDT --timeframe 1h --bars 1200
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from alpha_lab.data.query import load_bars  # noqa: E402
from alpha_lab.strategies.mean_reversion import MeanReversionStrategy  # noqa: E402

DEFAULT_ROOT = Path(r"D:\alpha-lab\data")
DEFAULT_PARAMS = {
    "window": 20,
    "k": 2.0,
    "atr_len": 14,
    "sl_atr": 2.0,
    "tp_atr": 6.0,
    "max_bars": 500,
    "use_hl_filter": False,
}

TIMEFRAME_SECONDS = {
    "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400,
}


def build_trades(bars: pd.DataFrame, pos: pd.Series, atr: pd.Series,
                 sl_atr: float, tp_atr: float, max_bars: int) -> list[dict]:
    """Разбирает ряд позиций на сделки: вход, выход, причина, R.

    Риск — это 1R в цене, то есть slAtr·ATR на баре входа: ровно то же
    определение, что в индикаторе и в симуляторе выходов. Считать риск как
    расстояние до экстремума бара нельзя — это другая величина, и R разошёлся
    бы с тем, что показывает Pine.
    """
    p = pos.to_numpy()
    high = bars["high"].to_numpy(dtype="float64")
    low = bars["low"].to_numpy(dtype="float64")
    close = bars["close"].to_numpy(dtype="float64")
    a = atr.to_numpy(dtype="float64")
    ts = bars["ts"].tolist()

    trades: list[dict] = []
    i, n = 0, len(p)
    while i < n:
        if p[i] == 0:
            i += 1
            continue
        d = int(p[i])
        j = i
        while j + 1 < n and p[j + 1] == p[i]:
            j += 1

        entry_p = float(close[i])
        risk = sl_atr * a[i] if np.isfinite(a[i]) else np.nan
        sl = entry_p - d * risk if np.isfinite(risk) else np.nan
        tp = entry_p + d * tp_atr * a[i] if np.isfinite(a[i]) else np.nan

        # Модель ставит pos[бар_выхода] = 0, поэтому серия ненулевых позиций
        # кончается ЗА БАР до фактического выхода. Сканировать [i..j] нельзя:
        # бар, на котором сработал стоп, в диапазон не попадает, и все сделки
        # выглядят вышедшими по времени.
        exit_bar = j + 1 if j + 1 < n else j

        # Причина и цена выхода — порядок как в _simulate_py:
        # «стоп ИЛИ таймаут» старше тейка.
        kind, exit_p = "время", float(close[exit_bar])
        for k in range(i, exit_bar + 1):
            hit_sl = low[k] <= sl if d > 0 else high[k] >= sl
            hit_tp = high[k] >= tp if d > 0 else low[k] <= tp
            if not np.isfinite(sl):
                hit_sl = False
            if not np.isfinite(tp):
                hit_tp = False
            if hit_sl or (k - i) >= max_bars:
                kind = "стоп" if hit_sl else "время"
                exit_p = float(sl) if hit_sl else float(close[k])
                break
            if hit_tp:
                kind, exit_p = "тейк", float(tp)
                break

        r = (exit_p - entry_p) / risk if d > 0 else (entry_p - exit_p) / risk
        r = float(r) if np.isfinite(r) else 0.0

        trades.append({
            "dir": d,
            "entry_i": i, "exit_i": exit_bar,
            "entry_t": str(ts[i]), "exit_t": str(ts[exit_bar]),
            "entry_p": round(entry_p, 8), "exit_p": round(exit_p, 8),
            "sl": None if not np.isfinite(sl) else round(float(sl), 8),
            "tp": None if not np.isfinite(tp) else round(float(tp), 8),
            "bars": exit_bar - i,
            "kind": kind,
            "r": round(r, 4),
        })
        i = exit_bar + 1
    return trades


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--bars", type=int, default=1200,
                    help="Сколько последних баров выгружать")
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default="polygon/preview.js")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    bars = load_bars(Path(args.root), args.symbol, "1m",
                     start="2022-01-01", end="2025-12-31", resample=args.timeframe)
    bars = bars.tail(args.bars).reset_index(drop=True)
    print(f"загружено {len(bars)} баров {args.symbol} {args.timeframe}")

    strategy = MeanReversionStrategy(dict(DEFAULT_PARAMS))
    pos = strategy.generate(bars)

    # Среднее, полосы и z — те же формулы, что в индикаторе.
    w = DEFAULT_PARAMS["window"]
    mean = bars["close"].rolling(w).mean()
    mean_sq = (bars["close"] * bars["close"]).rolling(w).mean()
    sigma = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    z = ((bars["close"] - mean) / sigma.where(sigma > 1e-12)).fillna(0.0)

    def rnd(s, d=6):
        return [None if not np.isfinite(v) else round(float(v), d) for v in s]

    from alpha_lab.features.volatility import atr as atr_series
    atr_vals = atr_series(bars, DEFAULT_PARAMS["atr_len"])

    trades = build_trades(bars, pos, atr_vals, DEFAULT_PARAMS["sl_atr"],
                          DEFAULT_PARAMS["tp_atr"], DEFAULT_PARAMS["max_bars"])
    wins = [t for t in trades if t["r"] > 0]
    payload = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "params": DEFAULT_PARAMS,
        "bars": [
            {"t": str(t), "o": round(float(o), 8), "h": round(float(h), 8),
             "l": round(float(lo), 8), "c": round(float(c), 8)}
            for t, o, h, lo, c in zip(bars["ts"], bars["open"], bars["high"],
                                      bars["low"], bars["close"])
        ],
        "series": {
            "mean": rnd(mean, 8),
            "upper": rnd(mean + DEFAULT_PARAMS["k"] * sigma, 8),
            "lower": rnd(mean - DEFAULT_PARAMS["k"] * sigma, 8),
            "z": rnd(z, 4),
            "pos": [int(v) for v in pos],
        },
        "trades": trades,
        "stats": {
            "trades": len(trades),
            "wins": len(wins),
            "win_rate": round(100.0 * len(wins) / len(trades), 1) if trades else 0.0,
            "sum_r": round(sum(t["r"] for t in trades), 2),
            "avg_r": round(sum(t["r"] for t in trades) / len(trades), 3) if trades else 0.0,
            "avg_bars": round(sum(t["bars"] for t in trades) / len(trades), 1) if trades else 0.0,
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out.write_text(f"window.ALPHA_PREVIEW = {text};\n", encoding="utf-8")
    print(f"сделок: {len(trades)}, выигрышных: {len(wins)}, "
          f"сумма R: {payload['stats']['sum_r']}")
    print(f"записано: {out} ({out.stat().st_size / 1024:.0f} КБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
