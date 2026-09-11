import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.report.schema import (
    MAJOR_VERSION, REQUIRED_SERIES, REQUIRED_TOP_LEVEL, SCHEMA_VERSION, is_compatible,
)
from alpha_lab.report.writer import build_report, write_report
from alpha_lab.validation.validator import Verdict


def _verdict():
    return Verdict(
        strategy_name="mean_reversion", experiment_id="abc123",
        sharpe=1.4, dsr=0.97, p_value=0.01, max_dd=-0.22, total_return=0.5,
        trades=250, n_configs_tried=12, alive=True, reasons=(),
        metrics={"win_rate": 0.55}, pbo=0.21,
    )


def _series(n=100):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return ts, pd.Series(np.linspace(1.0, 1.5, n), index=ts)


def test_build_report_has_schema_version():
    ts, equity = _series()

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["verdict"]["alive"] is True


def test_build_report_serializes_timestamps_as_iso():
    ts, equity = _series(5)

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["series"]["ts"][0] == "2024-01-01T00:00:00+00:00"


def test_build_report_has_all_series():
    ts, equity = _series(10)

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(1.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.001}, index=ts))

    for key in ("ts", "equity", "drawdown", "close", "position", "fee", "funding"):
        assert key in payload["series"], key


def test_write_report_creates_both_files(tmp_path):
    ts, equity = _series(10)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    json_path, js_path = write_report(payload, tmp_path)

    assert json_path.exists() and js_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


def test_report_js_assigns_global(tmp_path):
    ts, equity = _series(5)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    _, js_path = write_report(payload, tmp_path)
    text = js_path.read_text(encoding="utf-8")

    assert text.startswith("window.ALPHA_REPORT = ")
    assert text.rstrip().endswith(";")


def test_report_js_payload_matches_json(tmp_path):
    ts, equity = _series(5)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    json_path, js_path = write_report(payload, tmp_path)
    from_json = json.loads(json_path.read_text(encoding="utf-8"))
    from_js = json.loads(
        js_path.read_text(encoding="utf-8")
        .split("=", 1)[1].strip().rstrip(";")
    )

    assert from_json == from_js


def test_nan_values_become_null(tmp_path):
    ts, equity = _series(3)
    v = Verdict(strategy_name="s", experiment_id="x", sharpe=0.0, dsr=0.0,
                p_value=1.0, max_dd=0.0, total_return=0.0, trades=0,
                n_configs_tried=1, alive=False, reasons=("нет",),
                metrics={}, pbo=float("nan"))

    payload = build_report(v, equity=equity, close=equity,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["verdict"]["pbo"] is None

    # NaN в report.js не распарсился бы как JSON вообще, поэтому проверяем оба файла.
    json_path, js_path = write_report(payload, tmp_path)
    js_text = js_path.read_text(encoding="utf-8")
    assert "NaN" not in js_text
    assert json.loads(json_path.read_text(encoding="utf-8"))["verdict"]["pbo"] is None
    from_js = json.loads(js_text.split("=", 1)[1].strip().rstrip(";"))
    assert from_js["verdict"]["pbo"] is None


# --- Усиление контракта: проверки, которых нет в кратком ТЗ, но без них
# --- дашборд может молча развалиться.

def _payload(n=7, verdict=None, extra=None):
    ts, equity = _series(n)
    return build_report(verdict or _verdict(), equity=equity, close=equity * 100,
                        positions=pd.Series(0.0, index=ts),
                        costs=pd.DataFrame({"fee": 0.0}, index=ts), extra=extra)


def test_report_js_is_valid_json_after_stripping_prefix(tmp_path):
    """Контракт дашборда: <script src="report.js"> даёт валидный JSON-объект."""
    ts, equity = _series(5)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    _, js_path = write_report(payload, tmp_path)
    text = js_path.read_text(encoding="utf-8").strip()
    prefix = "window.ALPHA_REPORT = "

    assert text.startswith(prefix)
    assert text.endswith(";")
    body = text.removeprefix(prefix).removesuffix(";")
    assert json.loads(body) == payload


def test_report_js_survives_equals_sign_in_payload(tmp_path):
    """Наивный split("=") без maxsplit сломался бы на строке со знаком "="."""
    payload = _payload(5, extra={"note": "a=b", "experiment": "id=x=y"})

    json_path, js_path = write_report(payload, tmp_path)
    from_json = json.loads(json_path.read_text(encoding="utf-8"))
    from_js = json.loads(
        js_path.read_text(encoding="utf-8").split("=", 1)[1].strip().rstrip(";")
    )

    assert from_json["extra"] == {"note": "a=b", "experiment": "id=x=y"}
    assert from_js == from_json


def test_all_series_have_same_length():
    """Несовпадение длин молча разъедет графики на дашборде."""
    ts, equity = _series(7)

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(1.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.001}, index=ts))

    for key in ("ts", "equity", "drawdown", "close", "position", "fee", "funding"):
        assert len(payload["series"][key]) == len(ts), key


def test_drawdown_is_computed_from_running_max():
    ts = pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC")
    equity = pd.Series([1.0, 2.0, 1.0, 3.0], index=ts)

    payload = build_report(_verdict(), equity=equity, close=equity,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["series"]["drawdown"] == [0.0, 0.0, -0.5, 0.0]


def test_missing_cost_columns_default_to_zero():
    ts, equity = _series(4)

    payload = build_report(_verdict(), equity=equity, close=equity,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame(index=ts))

    assert payload["series"]["fee"] == [0.0] * 4
    assert payload["series"]["funding"] == [0.0] * 4


def test_schema_constants_describe_the_payload():
    payload = _payload()

    for key in REQUIRED_TOP_LEVEL:
        assert key in payload, key
    for key in REQUIRED_SERIES:
        assert key in payload["series"], key


def test_is_compatible_checks_major_version():
    assert MAJOR_VERSION == 1
    assert is_compatible(SCHEMA_VERSION)
    assert is_compatible("1.7")
    assert not is_compatible("2.0")
    assert not is_compatible("мусор")
    assert not is_compatible(None)
