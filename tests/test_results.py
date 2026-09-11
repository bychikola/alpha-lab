"""Тесты хранилища результатов P3 (Parquet + DuckDB).

Проверяются свойства, от которых зависит корректность возобновления свипа и
воронки отбора: круговой рейс, идемпотентность по config_id, громкое
отвержение несовместимой версии схемы, различимость строк-ошибок и
атомарность записи (обрыв не оставляет битое хранилище).
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from alpha_lab.results import (
    COLUMNS, RESULTS_SCHEMA_VERSION,
    completed_ids, error_rows, group_counts, is_error, query_runs, read_runs,
    verdict_rows, write_runs,
)


def _row(config_id: str, **overrides) -> dict:
    row = {
        "config_id": config_id,
        "experiment_id": f"exp-{config_id}",
        "index": 0,
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "strategy": "mean_reversion",
        "params_json": '{"k":2.0}',
        "start": "2022-01-01",
        "end": "2025-12-31",
        "error": "",
        "data_version": "dv1",
        "duration_s": 2.0,
        "sharpe": 1.0,
        "dsr": 0.99,
        "p_value": 0.01,
        "pbo": 0.2,
        "max_dd": -0.1,
        "total_return": 0.5,
        "trades": 200,
        "n_trials": 10,
        "n_permutations": 1000,
        "screening": False,
        "alive": True,
        "reasons": "",
        "warnings": "",
        "thresholds_json": json.dumps(
            {"min_trades": 100, "min_dsr": 0.95, "max_p_value": 0.05,
             "max_pbo": 0.5}, sort_keys=True),
        "cost_total": 0.02,
        "costs_json": '{"fee": 0.01}',
    }
    row.update(overrides)
    return row


def _store(tmp_path: Path) -> Path:
    return tmp_path / "results"


def test_round_trip_write_read(tmp_path):
    path = _store(tmp_path)
    rows = [_row("a"), _row("b", symbol="ETHUSDT", timeframe="4h", alive=False,
                         dsr=0.5, reasons="DSR 0.5 ≤ 0.95")]
    write_runs(path, rows)

    frame = read_runs(path)

    assert len(frame) == 2
    assert set(frame["config_id"]) == {"a", "b"}
    assert list(frame.columns) == list(COLUMNS)
    eth = frame[frame["config_id"] == "b"].iloc[0]
    assert eth["symbol"] == "ETHUSDT"
    assert bool(eth["alive"]) is False
    assert eth["dsr"] == 0.5
    assert eth["data_version"] == "dv1"
    # Пороги гейтов едут в строке: воронка обязана судить строку ими, а не
    # текущими дефолтами, которые сетка могла переопределить.
    assert json.loads(eth["thresholds_json"])["min_trades"] == 100

    # Пустой/отсутствующий store читается как пустая таблица контракта.
    empty = read_runs(tmp_path / "nowhere")
    assert len(empty) == 0
    assert list(empty.columns) == list(COLUMNS)


def test_write_is_idempotent_by_config_id(tmp_path):
    path = _store(tmp_path)
    write_runs(path, [_row("a", sharpe=1.0), _row("b", sharpe=2.0)])
    write_runs(path, [_row("a", sharpe=9.0, dsr=0.4)])

    frame = read_runs(path)

    assert len(frame) == 2
    assert sorted(frame["config_id"]) == ["a", "b"]
    a = frame[frame["config_id"] == "a"].iloc[0]
    assert a["sharpe"] == 9.0 and a["dsr"] == 0.4
    b = frame[frame["config_id"] == "b"].iloc[0]
    assert b["sharpe"] == 2.0


def test_incompatible_schema_version_is_rejected_loudly(tmp_path):
    """Схема 1.0 отвергается: без порогов по строкам воронка судила бы её
    текущими дефолтами, а сетка могла задавать другие пороги."""
    path = _store(tmp_path)
    path.mkdir(parents=True)
    table = pa.table({"config_id": ["x"], "data_version": ["dv"], "error": [""]})
    table = table.replace_schema_metadata({b"alpha_lab_schema_version": b"3.0"})
    pq.write_table(table, path / "runs.parquet")

    with pytest.raises(ValueError) as exc:
        read_runs(path)

    message = str(exc.value)
    assert "несовместим" in message.lower()
    assert "3.0" in message and "2" in message


def test_store_without_schema_version_is_rejected(tmp_path):
    path = _store(tmp_path)
    path.mkdir(parents=True)
    pq.write_table(pa.table({"config_id": ["x"]}), path / "runs.parquet")

    with pytest.raises(ValueError) as exc:
        read_runs(path)

    assert "версия" in str(exc.value).lower()


def test_error_rows_are_explicitly_distinguishable(tmp_path):
    path = _store(tmp_path)
    crash = _row("dead", error="ValueError: плохие параметры", alive=True,
                 trades=777, reasons="DSR 0.99", dsr=float("nan"))
    verdict = _row("ok", alive=False)
    write_runs(path, [crash, verdict])

    frame = read_runs(path)

    # Контракт строки-ошибки принудителен: крах не выглядит живым вердиктом.
    row = frame[frame["config_id"] == "dead"].iloc[0]
    assert row["error"] == "ValueError: плохие параметры"
    assert bool(row["alive"]) is False
    assert int(row["trades"]) == 0
    assert row["reasons"] == ""

    assert is_error(frame).tolist() == [True, False]
    assert set(error_rows(frame)["config_id"]) == {"dead"}
    assert set(verdict_rows(frame)["config_id"]) == {"ok"}


def test_query_helpers_filter_sort_and_group(tmp_path):
    path = _store(tmp_path)
    write_runs(path, [
        _row("a", symbol="BTCUSDT", timeframe="1h", alive=True, dsr=0.99),
        _row("b", symbol="BTCUSDT", timeframe="4h", alive=False, dsr=0.4),
        _row("c", symbol="ETHUSDT", timeframe="1h", alive=True, dsr=0.97),
        _row("e", symbol="ETHUSDT", timeframe="1h", error="boom"),
    ])

    assert set(query_runs(path, symbol="BTCUSDT")["config_id"]) == {"a", "b"}
    assert set(query_runs(path, timeframe="1h")["config_id"]) == {"a", "c"}
    assert set(query_runs(path, alive=True)["config_id"]) == {"a", "c"}
    # По умолчанию строки-ошибки не попадают в выборку: крах — не гипотеза.
    assert "e" not in set(query_runs(path)["config_id"])
    assert "e" in set(query_runs(path, status="error")["config_id"])
    assert "e" in set(query_runs(path, status="any")["config_id"])

    top = query_runs(path, order_by="dsr", descending=True, limit=1)
    assert top["config_id"].tolist() == ["a"]

    by_symbol = group_counts(path, by="symbol")
    counts = dict(zip(by_symbol["symbol"], by_symbol["n"]))
    assert counts == {"BTCUSDT": 2, "ETHUSDT": 1}

    with pytest.raises(ValueError, match="order_by"):
        query_runs(path, order_by="; DROP TABLE")


def test_completed_ids_respects_data_version_errors_and_grade(tmp_path):
    path = _store(tmp_path)
    write_runs(path, [
        _row("full", data_version="dv1", screening=False),
        _row("screen", data_version="dv1", screening=True, alive=False),
        _row("failed", data_version="dv1", error="boom"),
        _row("old", data_version="dv0"),
    ])

    # Полный прогон пересчитывает всё, что не подтверждено полным вердиктом.
    assert completed_ids(path, "dv1", screening=False) == {"full"}
    # Черновой прогон не пересчитывает уже подтверждённые полные вердикты.
    assert completed_ids(path, "dv1", screening=True) == {"full", "screen"}


def test_failed_write_does_not_corrupt_store(tmp_path, monkeypatch):
    path = _store(tmp_path)
    write_runs(path, [_row("a", sharpe=1.0)])

    import alpha_lab.results as results_mod
    real_write = results_mod.pq.write_table

    def boom(*args, **kwargs):
        raise OSError("диск переполнен")

    monkeypatch.setattr(results_mod.pq, "write_table", boom)
    with pytest.raises(OSError):
        write_runs(path, [_row("b", sharpe=2.0)])
    monkeypatch.setattr(results_mod.pq, "write_table", real_write)

    frame = read_runs(path)
    assert frame["config_id"].tolist() == ["a"]
    assert frame.iloc[0]["sharpe"] == 1.0

    # Повторная запись после сбоя проходит и не задваивает строки.
    write_runs(path, [_row("b", sharpe=2.0)])
    assert set(read_runs(path)["config_id"]) == {"a", "b"}


def test_unknown_column_is_rejected(tmp_path):
    path = _store(tmp_path)
    with pytest.raises(ValueError, match="колонк"):
        write_runs(path, [_row("a", sharpe_typo=1.0)])


def test_schema_version_constant(tmp_path):
    # Мажор 2: в строке появились пороги гейтов (thresholds_json), и старые
    # хранилища 1.0 обязаны отвергаться, а не читаться без порогов.
    assert RESULTS_SCHEMA_VERSION == "2.0"
    path = _store(tmp_path)
    write_runs(path, [_row("a")])
    meta = pq.read_metadata(path / "runs.parquet").metadata
    assert meta[b"alpha_lab_schema_version"].decode() == RESULTS_SCHEMA_VERSION
