"""Запись отчёта в report.json и report.js.

Два файла из одного объекта: json — для программ, js — для дашборда.
Дашборд открывается через file://, где fetch() к локальным файлам блокируется
CORS, поэтому данные подключаются как <script src="report.js">.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.report.schema import SCHEMA_VERSION
from alpha_lab.validation.validator import Verdict


def _round(values, digits: int = 6) -> list:
    out = []
    for v in values:
        f = float(v)
        out.append(None if not np.isfinite(f) else round(f, digits))
    return out


def build_report(verdict: Verdict, equity: pd.Series, close: pd.Series,
                 positions: pd.Series, costs: pd.DataFrame,
                 price_bars: pd.DataFrame | None = None,
                 extra: dict | None = None) -> dict:
    eq = pd.Series(equity).astype("float64")
    dd = eq / eq.cummax() - 1.0

    series = {
        "ts": [pd.Timestamp(t).isoformat() for t in eq.index],
        "equity": _round(eq.to_numpy(), 8),
        "drawdown": _round(dd.to_numpy(), 8),
        "close": _round(pd.Series(close).to_numpy(), 8),
        "position": _round(pd.Series(positions).to_numpy(), 6),
        "fee": _round(costs.get("fee", pd.Series(0.0, index=eq.index)).to_numpy(), 8),
        "slippage": _round(costs.get("slippage", pd.Series(0.0, index=eq.index)).to_numpy(), 8),
        "funding": _round(costs.get("funding", pd.Series(0.0, index=eq.index)).to_numpy(), 8),
    }
    if price_bars is not None:
        for col in ("open", "high", "low"):
            if col in price_bars.columns:
                series[col] = _round(price_bars[col].to_numpy(), 8)

    verdict_payload = {
        "strategy_name": verdict.strategy_name,
        "experiment_id": verdict.experiment_id,
        "alive": verdict.alive,
        "sharpe": _num(verdict.sharpe),
        "dsr": _num(verdict.dsr),
        "pbo": _num(verdict.pbo),
        "p_value": _num(verdict.p_value),
        "max_dd": _num(verdict.max_dd),
        "total_return": _num(verdict.total_return),
        "trades": verdict.trades,
        "n_configs_tried": verdict.n_configs_tried,
        "reasons": list(verdict.reasons),
        "metrics": {k: _num(v) for k, v in (verdict.metrics or {}).items()},
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict_payload,
        "series": series,
    }
    if extra:
        payload["extra"] = extra
    return payload


def _num(value):
    if value is None:
        return None
    f = float(value)
    return None if not np.isfinite(f) else round(f, 6)


def write_report(payload: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    js_path = out / "report.js"

    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    json_path.write_text(text, encoding="utf-8")
    js_path.write_text(f"window.ALPHA_REPORT = {text};\n", encoding="utf-8")
    return json_path, js_path
