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
from alpha_lab.validation.validator import DEFAULT_THRESHOLDS, Verdict

DEFAULT_N_PERMUTATIONS = DEFAULT_THRESHOLDS["n_permutations"]


def _round(values, digits: int = 6) -> list:
    out = []
    for v in values:
        f = float(v)
        out.append(None if not np.isfinite(f) else round(f, digits))
    return out


def _json_safe(value):
    """Рекурсивно приводит extra к JSON-совместимому виду.

    Нефинитные float (NaN, ±Inf) заменяются на None — так же, как в _round/_num.
    Иначе json.dumps(..., allow_nan=False) в write_report упал бы на ровном месте,
    хотя весь остальной payload такие значения уже не пропускает.
    """
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if not np.isfinite(f) else f
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _require_same_length(name: str, values, expected: int) -> None:
    """Проверяет, что ряд выровнен с equity и совпадает с ним по длине."""
    actual = len(values)
    if actual != expected:
        raise ValueError(
            f"длина {name} ({actual}) не совпадает с длиной equity ({expected}); "
            "все ряды должны быть выровнены и иметь одинаковую длину"
        )


def build_report(verdict: Verdict, equity: pd.Series, close: pd.Series,
                 positions: pd.Series, costs: pd.DataFrame,
                 price_bars: pd.DataFrame | None = None,
                 extra: dict | None = None) -> dict:
    """Собирает payload отчёта по схеме.

    Все ряды обязаны быть взаимно выровнены и одной длины с equity: дашборд
    сопоставляет их с series.ts по позиции в массиве, поэтому расхождение длин
    нарисовало бы правдоподобный, но неверный график. Требование проверяется:
    close, positions, любая колонка costs и переданные колонки price_bars при
    несовпадении длины дают ValueError. Перевыравнивание (reindex) не делается —
    выходы движка приходят позиционными, и reindex молча превратил бы их в NaN.
    """
    eq = pd.Series(equity).astype("float64")
    n = len(eq)

    _require_same_length("close", close, n)
    _require_same_length("positions", positions, n)
    for col in costs.columns:
        _require_same_length(f"costs['{col}']", costs[col], n)
    if price_bars is not None:
        for col in ("open", "high", "low"):
            if col in price_bars.columns:
                _require_same_length(f"price_bars['{col}']", price_bars[col], n)

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
        # int(): json.dumps не умеет numpy-скаляры, а валидатор не обязан быть
        # единственным источником вердикта.
        "trades": int(verdict.trades),
        "n_configs_tried": int(verdict.n_configs_tried),
        "reasons": list(verdict.reasons),
        # Предупреждения — отдельный от reasons канал: непроверенный гейт не
        # делает стратегию мёртвой, но и не должен теряться при сериализации.
        # Пустой список (не null): дашборд отличает «предупреждений нет» от
        # «поле не передано» без дополнительных догадок.
        "warnings": list(verdict.warnings or ()),
        # P5: грейд вердикта — элемент контракта, а не деталь. Отчёт, снятый
        # черновым прогоном, обязан нести это в payload: иначе precise и
        # rough вердикты неотличимы после закрытия терминала.
        "screening": bool(getattr(verdict, "screening", False)),
        "n_permutations": int(getattr(verdict, "n_permutations",
                                      DEFAULT_N_PERMUTATIONS)),
        "metrics": {k: _num(v) for k, v in (verdict.metrics or {}).items()},
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict_payload,
        "series": series,
    }
    if extra:
        payload["extra"] = _json_safe(extra)
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
