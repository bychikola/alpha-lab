"""Тесты воронки отбора P4: точные счётчики гейтов, честность нуля, экспорт.

Синтетическое хранилище с известной смесью выживших, мёртвых и упавших строк:
воронка обязана дать ровно ожидаемые числа, а не «примерно». Ошибки — не
гипотезы: они исключены из знаменателей и посчитаны отдельно.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.cli as cli
from alpha_lab.funnel import (
    DEFAULT_THRESHOLDS, FUNNEL_SCHEMA_VERSION, build_funnel, format_funnel,
    write_funnel,
)
from alpha_lab.results import read_runs, write_runs

_DEFAULT_THRESHOLDS_JSON = json.dumps(DEFAULT_THRESHOLDS, sort_keys=True)


def _row(config_id: str, *, symbol="BTCUSDT", timeframe="1h", trades=200,
         dsr=0.99, p_value=0.01, pbo=0.2, alive=False, screening=False,
         error="", sharpe=1.0, max_dd=-0.2, total_return=0.3,
         cost_total=0.02, data_version="dv1", n_permutations=1000,
         thresholds=None) -> dict:
    return {
        "config_id": config_id, "experiment_id": f"exp-{config_id}",
        "index": 0, "symbol": symbol, "timeframe": timeframe,
        "strategy": "mean_reversion", "params_json": '{"k": 2.0}',
        "start": "2022-01-01", "end": "2025-12-31", "error": error,
        "data_version": data_version, "duration_s": 1.0, "sharpe": sharpe,
        "dsr": dsr, "p_value": p_value, "pbo": pbo, "max_dd": max_dd,
        "total_return": total_return, "trades": trades, "n_trials": 12,
        "n_permutations": n_permutations, "screening": screening, "alive": alive,
        "reasons": "" if alive else "DSR ниже порога",
        "warnings": "", "thresholds_json": _DEFAULT_THRESHOLDS_JSON
        if thresholds is None else json.dumps(thresholds, sort_keys=True),
        "cost_total": cost_total,
        "costs_json": '{"fee": 0.01}',
    }


def _mixed_rows() -> list[dict]:
    """12 вердиктов + 2 ошибки; воронка обязана дать 11→9→6→2→2."""
    rows = [
        _row("v1", trades=50),                                   # гейт сделок
        _row("v2", dsr=0.5), _row("v3", dsr=0.1),                # гейт DSR
        _row("v4", p_value=0.2), _row("v5", p_value=0.3),
        _row("v6", p_value=0.5),                                 # гейт p-value
        _row("v7", pbo=0.8), _row("v8", pbo=0.6),
        _row("v9", pbo=0.9), _row("v10", pbo=1.0),               # гейт PBO
        _row("v11", alive=True, dsr=0.999, sharpe=2.0),          # выживший
        _row("v12", alive=True, dsr=0.97, sharpe=1.5,
             symbol="ETHUSDT", timeframe="4h"),                  # выживший
        _row("e1", error="ValueError: нет данных", trades=999,
             dsr=float("nan"), alive=False, symbol="SOLUSDT"),
        _row("e2", error="FileNotFoundError: нет файла", symbol="BNBUSDT"),
    ]
    return rows


def test_funnel_counts_are_exact_and_sequential():
    frame = _frame(_mixed_rows())

    report = build_funnel(frame, top=5)
    funnel = report["funnel"]

    assert report["n_configs"] == 14
    assert report["errors"]["n"] == 2
    assert funnel["n"] == 12                       # ошибки исключены
    assert [step["n"] for step in funnel["steps"]] == [11, 9, 6, 2]
    assert [step["key"] for step in funnel["steps"]] == \
        ["trades", "dsr", "p_value", "pbo"]
    assert funnel["alive"]["n"] == 2
    assert [step["fraction_of_total"] for step in funnel["steps"]] == \
        pytest.approx([11 / 12, 9 / 12, 6 / 12, 2 / 12], abs=1e-6)
    assert [step["fraction_of_previous"] for step in funnel["steps"]] == \
        pytest.approx([11 / 12, 9 / 11, 6 / 9, 2 / 6], abs=1e-6)
    assert funnel["alive"]["fraction_of_total"] == pytest.approx(2 / 12, abs=1e-6)

    # Ошибки не влияют на воронку: строка e1 с trades=999 и dsr=NaN не в счёте.
    assert report["survivors"]["n"] == 2
    assert report["survivors"]["selected_from"]["n_verdicts"] == 12
    assert report["survivors"]["selected_from"]["n_errors"] == 2
    assert [s["config_id"] for s in report["survivors"]["top"]] == ["v11", "v12"]


def test_funnel_breakdown_by_timeframe_and_symbol():
    frame = _frame(_mixed_rows())

    report = build_funnel(frame)

    by_tf = {g["timeframe"]: g for g in report["by_timeframe"]}
    assert by_tf["1h"]["n"] == 11 and by_tf["1h"]["alive"] == 1
    assert by_tf["4h"]["n"] == 1 and by_tf["4h"]["alive"] == 1
    # Ошибки в разрезах посчитаны отдельно и не входят в n.
    by_sym = {g["symbol"]: g for g in report["by_symbol"]}
    assert by_sym["SOLUSDT"]["n"] == 0 and by_sym["SOLUSDT"]["errors"] == 1
    assert by_sym["BNBUSDT"]["n"] == 0 and by_sym["BNBUSDT"]["errors"] == 1
    assert by_sym["BTCUSDT"]["n"] == 11 and by_sym["BTCUSDT"]["alive"] == 1


def test_zero_survivors_is_reported_honestly():
    rows = [_row(f"d{i}", dsr=0.5 + 0.01 * i) for i in range(5)]
    rows.append(_row("dead-best", dsr=0.949, sharpe=9.9, p_value=0.001,
                     pbo=0.01))
    report = build_funnel(_frame(rows))

    assert report["funnel"]["alive"]["n"] == 0
    assert report["survivors"]["n"] == 0
    assert report["survivors"]["top"] == []
    assert "Лучший из мёртвых" in report["survivors"]["note"]

    text = format_funnel(report)
    assert "Выживших нет" in text
    assert "Топ выживших" not in text
    # Лучший из мёртвых не предъявляется как результат: ни его id, ни Sharpe.
    assert "dead-best" not in text
    assert "9.90" not in text


def test_alive_without_gate_intersection_is_not_a_survivor():
    """Хранимый alive вне пересечения гейтов не делает строку выжившей.

    Воспроизведение ревью: сетка, чья единственная ось — symbols, даёт по одной
    конфигурации на группу (символ, таймфрейм). Матрицы доходностей нет, PBO
    остаётся NaN, а валидатор честно пишет alive=True с предупреждением, что
    гейт PBO не проверялся. Воронка обязана показать ноль выживших и честный
    ноль, а не «Топ выживших» из строк, не прошедших пересечение гейтов.
    """
    rows = [
        _row("ghost-pbo-1", alive=True, pbo=float("nan")),
        _row("ghost-pbo-2", alive=True, pbo=float("nan"), symbol="ETHUSDT"),
        _row("ghost-dsr", alive=True, dsr=float("nan"), symbol="SOLUSDT"),
    ]
    report = build_funnel(_frame(rows))

    # alive-флаг есть у всех трёх, но пересечение гейтов пусто.
    assert report["funnel"]["alive"]["n"] == 0
    assert report["funnel"]["alive_without_pbo"] == 3
    assert report["survivors"]["n"] == 0
    assert report["survivors"]["top"] == []

    text = format_funnel(report)
    assert "Выживших нет" in text
    assert "Топ выживших" not in text
    for ghost in ("ghost-pbo-1", "ghost-pbo-2", "ghost-dsr"):
        assert ghost not in text


def test_survivors_are_exactly_the_alive_gate_intersection():
    """Список выживших — ровно то множество, по которому посчитан alive.

    Инвариант по построению: непустой список невозможен при нулевом счётчике
    alive, а мёртвая строка с лучшими метриками в список не попадает.
    """
    rows = _mixed_rows() + [
        _row("ghost", alive=True, pbo=float("nan")),
        _row("dead-best", alive=False, dsr=0.999, sharpe=9.9, pbo=0.01),
    ]
    report = build_funnel(_frame(rows))

    alive = report["funnel"]["alive"]["n"]
    assert alive == 2
    assert report["survivors"]["n"] == alive
    assert {s["config_id"] for s in report["survivors"]["top"]} == {"v11", "v12"}
    # Непустой список выживших подразумевает ненулевой alive — и наоборот.
    assert (report["survivors"]["n"] > 0) == (alive > 0)
    for entry in report["survivors"]["top"]:
        assert entry["dsr"] is not None and entry["pbo"] is not None
        assert entry["dsr"] > DEFAULT_THRESHOLDS["min_dsr"]
        assert entry["pbo"] < DEFAULT_THRESHOLDS["max_pbo"]


def test_funnel_judges_rows_by_thresholds_recorded_in_store(tmp_path):
    """Строка судится порогами, записанными сеткой, а не текущими дефолтами.

    Строка проходит все четыре гейта только при порогах сетки (min_trades=1,
    min_dsr=0.5, max_p_value=0.5, max_pbo=0.9). Записанная свипом строка несёт
    их в thresholds_json; воронка же, считающая по дефолтам (100 / 0.95 / 0.05 /
    0.5), показала бы ноль на всех шагах и потеряла бы выжившего.
    """
    from alpha_lab.batch import _row as sweep_row
    from alpha_lab.validation.validator import Verdict

    validation = {"min_trades": 1, "min_dsr": 0.5, "max_p_value": 0.5,
                  "max_pbo": 0.9}
    verdict = Verdict(
        strategy_name="mean_reversion", experiment_id="e1", sharpe=1.5,
        dsr=0.6, p_value=0.2, max_dd=-0.1, total_return=0.2, trades=5,
        n_configs_tried=1, alive=True, pbo=0.7,
    )
    row = sweep_row(_grid_cell(validation=validation), "dv1",
                    experiment_id="e1", n_trials=1, verdict=verdict)
    store = tmp_path / "results"
    write_runs(store, [row])

    report = build_funnel(read_runs(store))

    assert [step["n"] for step in report["funnel"]["steps"]] == [1, 1, 1, 1]
    assert report["funnel"]["alive"]["n"] == 1
    assert report["survivors"]["n"] == 1
    assert report["survivors"]["top"][0]["config_id"] == "c1"
    assert report["thresholds"]["min_trades"] == 1
    assert report["thresholds"]["min_dsr"] == pytest.approx(0.5)
    assert report["thresholds"]["max_p_value"] == pytest.approx(0.5)
    assert report["thresholds"]["max_pbo"] == pytest.approx(0.9)
    assert "сделок ≥ 1" in format_funnel(report)


def test_missing_thresholds_warn_and_fall_back_to_defaults():
    """Нет записанных порогов — громкое предупреждение, а не тихий дефолт.

    Схема 2.0 пишет пороги каждой строки; строка без них (собрана в обход
    свипа) судится порогами по умолчанию, но читатель обязан видеть, что
    сравнение неоднородно.
    """
    frame = _frame(_mixed_rows())
    frame["thresholds_json"] = ""
    report = build_funnel(frame)

    assert any("пороги" in w.lower() for w in report["warnings"])
    assert [step["n"] for step in report["funnel"]["steps"]] == [11, 9, 6, 2]


def _grid_cell(*, validation=None, config_id="c1", symbol="BTCUSDT"):
    """Ячейка сетки для проверки записи порогов свипом (batch._row)."""
    from alpha_lab.config import Experiment
    from alpha_lab.grid import GridCell

    return GridCell(
        index=0, config_id=config_id, symbol=symbol,
        experiment=Experiment(
            name="custom", strategy="mean_reversion",
            params={"window": 20, "k": 2.0}, timeframe="1h",
            start="2022-01-01", end="2025-12-31", costs={},
            validation=dict(validation or {})))


def test_screening_rows_are_candidates_not_alive():
    rows = _mixed_rows() + [
        _row("s1", screening=True, alive=False, n_permutations=200),
        _row("s2", screening=True, alive=False, trades=10, n_permutations=200),
    ]
    report = build_funnel(_frame(rows))

    assert report["funnel"]["n"] == 12            # черновые — отдельная воронка
    assert report["screening"]["n"] == 2
    assert report["screening"]["alive"]["n"] == 0
    assert report["screening"]["candidates"] == 1  # s1 прошёл все гейты
    assert report["grade"] == "mixed"
    assert report["survivors"]["n"] == 2
    assert {s["config_id"] for s in report["survivors"]["top"]} == {"v11", "v12"}
    # N выбора — только полные вердикты: черновая строка не может быть
    # выжившей, поэтому в знаменатель отбора она не входит, но видна отдельно.
    assert report["survivors"]["selected_from"]["n_verdicts"] == 12
    assert report["survivors"]["selected_from"]["n_screening"] == 2
    assert "черновых screening: 2" in report["survivors"]["note"]

    text = format_funnel(report)
    assert "чернов" in text.lower()
    assert "кандидат" in text.lower()


def test_screening_only_store_is_marked_rough():
    rows = [_row("s1", screening=True, alive=False, n_permutations=200),
            _row("s2", screening=True, alive=False, p_value=0.5,
                 n_permutations=200)]
    report = build_funnel(_frame(rows))

    assert report["grade"] == "screening"
    assert report["screening"]["candidates"] == 1
    assert report["survivors"]["n"] == 0
    assert report["screening"]["min_achievable_p"] == pytest.approx(1 / 201,
                                                                   abs=1e-6)
    text = format_funnel(report)
    assert "1/201" in text
    assert "Топ выживших" not in text


def test_export_round_trips(tmp_path):
    report = build_funnel(_frame(_mixed_rows()))
    json_path, js_path = write_funnel(report, tmp_path)

    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == FUNNEL_SCHEMA_VERSION
    text = js_path.read_text(encoding="utf-8").strip()
    assert text.startswith("window.ALPHA_FUNNEL = ")
    body = json.loads(text.removeprefix("window.ALPHA_FUNNEL = ")
                      .removesuffix(";"))
    assert body == saved
    # NaN в payload недопустим: JSON.parse на дашборде упал бы.
    assert "NaN" not in js_path.read_text(encoding="utf-8")


def test_cli_report_command(tmp_path, capsys):
    store = tmp_path / "results"
    write_runs(store, _mixed_rows())
    out = tmp_path / "dashboard"

    code = cli.main(["report", "--store", str(store), "--out", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "Воронка" in printed and "12" in printed and "2" in printed
    assert (out / "funnel.json").exists() and (out / "funnel.js").exists()
    payload = json.loads((out / "funnel.json").read_text(encoding="utf-8"))
    assert payload["funnel"]["alive"]["n"] == 2


def test_cli_report_zero_survivors_prints_honestly(tmp_path, capsys):
    store = tmp_path / "results"
    write_runs(store, [_row("d1", dsr=0.2), _row("d2", dsr=0.3)])
    out = tmp_path / "dashboard"

    code = cli.main(["report", "--store", str(store), "--out", str(out)])

    assert code == 0
    printed = capsys.readouterr().out
    assert "Выживших нет" in printed
    assert "Топ выживших" not in printed


def _frame(rows: list[dict]):
    import pandas as pd
    from alpha_lab.results import _normalize
    return _normalize(pd.DataFrame(rows), where="тест")
