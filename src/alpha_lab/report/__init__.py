"""Отчёт для дашборда: версионированный JSON-контракт и его запись."""
from __future__ import annotations

from alpha_lab.report.schema import (
    MAJOR_VERSION,
    REQUIRED_SERIES,
    REQUIRED_TOP_LEVEL,
    SCHEMA_VERSION,
    is_compatible,
)
from alpha_lab.report.writer import build_report, write_report

__all__ = [
    "MAJOR_VERSION",
    "REQUIRED_SERIES",
    "REQUIRED_TOP_LEVEL",
    "SCHEMA_VERSION",
    "build_report",
    "is_compatible",
    "write_report",
]
