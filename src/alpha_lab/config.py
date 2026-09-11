"""Загрузка и валидация YAML-конфигов."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")


@dataclass(frozen=True)
class Universe:
    symbols: list[str]
    start: str
    end: str | None
    market: str
    include_delisted: bool = True


@dataclass(frozen=True)
class Experiment:
    name: str
    strategy: str
    params: dict[str, Any]
    timeframe: str
    start: str
    end: str | None
    costs: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Конфиг {path} должен быть YAML-словарём")
    return data


def _parse_include_delisted(data: dict[str, Any]) -> bool:
    """include_delisted — строго bool; отсутствие или null — безопасный true.

    bool(None) == False молча выключал бы делистингованные пары и вносил
    ошибку выживаемости, а строка "no"/"yes" вообще не проходила бы проверку
    смысла. Поэтому null трактуется как «не задано» (spec 5 требует включать
    делистингованные пары), а любое не-bool значение — ошибка конфига.
    """
    value = data.get("include_delisted")
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError(
            f"include_delisted должен быть true или false, получено {value!r}")
    return value


def load_universe(path: str | Path) -> Universe:
    data = _read_yaml(Path(path))
    symbols = data.get("symbols") or []
    if not symbols:
        raise ValueError("Юниверс должен содержать хотя бы один символ")
    if len(symbols) != len(set(symbols)):
        dupes = sorted({s for s in symbols if symbols.count(s) > 1})
        raise ValueError(f"В юниверсе дубликат символов: {dupes}")
    for field_name in ("market", "start"):
        if not data.get(field_name):
            raise ValueError(f"В юниверсе не задано поле '{field_name}'")
    return Universe(
        symbols=list(symbols),
        start=str(data["start"]),
        end=str(data["end"]) if data.get("end") else None,
        market=str(data["market"]),
        include_delisted=_parse_include_delisted(data),
    )


def load_experiment(path: str | Path) -> Experiment:
    data = _read_yaml(Path(path))
    for field_name in ("name", "strategy", "timeframe", "start"):
        if not data.get(field_name):
            raise ValueError(f"В эксперименте не задано поле '{field_name}'")
    tf = str(data["timeframe"])
    if tf not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Неизвестный таймфрейм '{tf}'. Допустимые: {', '.join(VALID_TIMEFRAMES)}"
        )
    return Experiment(
        name=str(data["name"]),
        strategy=str(data["strategy"]),
        params=dict(data.get("params") or {}),
        timeframe=tf,
        start=str(data["start"]),
        end=str(data["end"]) if data.get("end") else None,
        costs=dict(data.get("costs") or {}),
        validation=dict(data.get("validation") or {}),
    )
