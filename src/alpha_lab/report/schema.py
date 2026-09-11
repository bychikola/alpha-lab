"""Схема отчёта — контракт между движком и дашбордом.

Мажорная версия меняется при несовместимом изменении структуры.
Дашборд обязан отклонять неизвестную мажорную версию, а не рисовать пустые графики.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"
MAJOR_VERSION = int(SCHEMA_VERSION.split(".")[0])

REQUIRED_TOP_LEVEL = ("schema_version", "verdict", "series", "generated_at")
REQUIRED_SERIES = ("ts", "equity", "drawdown", "close", "position", "fee", "funding")


def is_compatible(version: str) -> bool:
    try:
        return int(str(version).split(".")[0]) == MAJOR_VERSION
    except (ValueError, AttributeError):
        return False
