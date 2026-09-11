"""Тесты режима просеивания P5: черновой вердикт только отсеивает.

Ключевое свойство: прохождение screening НЕ выносит alive. Черновой вердикт
помечен как грубый (число перестановок, минимальный достижимый p-value), а
полный путь по умолчанию не меняется ни на цифру.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alpha_lab.cli as cli
from alpha_lab.report.writer import build_report
from alpha_lab.validation.validator import (
    DEFAULT_THRESHOLDS, SCREENING_PERMUTATIONS, Verdict, validate,
)


def _case(n=5000, seed=1, strength=0.8, drift=0.0):
    """Согласованный набор: доходности цены, позиции и доходность стратегии."""
    rng = np.random.default_rng(seed)
    price_ret = pd.Series(rng.normal(drift, 0.01, n))
    sign = np.sign(price_ret.to_numpy())
    random_side = rng.choice([-1.0, 1.0], size=n)
    positions = pd.Series(np.where(rng.random(n) < strength, sign, random_side))
    returns = positions * price_ret
    return returns, price_ret, positions


def _trades(n, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.001, 0.01, n))


def _run(case, trades, **overrides):
    r, pr, pos = case
    kwargs = {"config": {}, "n_trials": 1, "strategy_name": "s",
              "experiment_id": "x", "price_returns": pr, "positions": pos}
    kwargs.update(overrides)
    return validate(r, trades, (1 + r).cumprod(), **kwargs)


def test_screening_verdict_is_marked_rough():
    v = _run(_case(seed=1, strength=0.8), _trades(300, 3), screening=True)

    assert v.screening is True
    assert v.n_permutations == SCREENING_PERMUTATIONS == 200
    assert any("1/201" in w for w in v.warnings)
    assert any("чернов" in w.lower() for w in v.warnings)
    # Гейты пройдены, но alive не вынесен: кандидат — не «жива».
    assert v.reasons == ()
    assert v.alive is False


def test_full_verdict_path_is_unchanged():
    case = _case(seed=1, strength=0.8)
    trades = _trades(300, 3)

    full = _run(case, trades)
    again = _run(case, trades)

    assert full.screening is False
    assert full.n_permutations == DEFAULT_THRESHOLDS["n_permutations"] == 1000
    assert full.alive is True and full.reasons == ()
    # Полный путь детерминирован и не сдвинут (pbo здесь NaN — сравниваем
    # содержательные поля, а не nan != nan).
    for field in ("sharpe", "dsr", "p_value", "trades", "max_dd",
                  "total_return", "alive", "reasons", "warnings", "screening",
                  "n_permutations", "metrics"):
        assert getattr(full, field) == getattr(again, field), field

    # Явно заданное число перестановок конфига сохраняется на полном пути.
    configured = _run(case, trades, config={"n_permutations": 50})
    assert configured.n_permutations == 50
    assert configured.screening is False


def test_screening_overrides_config_permutations():
    v = _run(_case(seed=1, strength=0.8), _trades(300, 3),
             config={"n_permutations": 5000}, screening=True)

    assert v.n_permutations == 200


def test_screening_rejects_noise_but_does_not_accept():
    noise = _run(_case(seed=2, strength=0.0), _trades(300, 4), screening=True)

    assert noise.alive is False and noise.reasons
    assert noise.screening is True


def test_report_serializes_screening_grade(tmp_path):
    ts = pd.date_range("2024-01-01", periods=30, freq="1h", tz="UTC")
    equity = pd.Series(np.linspace(1.0, 1.1, 30), index=ts)
    verdict = Verdict(
        strategy_name="s", experiment_id="x", sharpe=1.0, dsr=0.99,
        p_value=0.01, max_dd=-0.1, total_return=0.1, trades=200,
        n_configs_tried=10, alive=False, reasons=(),
        warnings=("Черновой вердикт (screening): перестановок 200",),
        screening=True, n_permutations=200,
    )
    payload = build_report(verdict, equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["verdict"]["screening"] is True
    assert payload["verdict"]["n_permutations"] == 200

    from alpha_lab.report.writer import write_report
    json_path, js_path = write_report(payload, tmp_path)
    import json
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["verdict"]["screening"] is True
    assert saved["verdict"]["n_permutations"] == 200


def test_validate_cli_accepts_screening_flag(tmp_path, capsys):
    """Флаг есть у validate; без данных команда честно падает, но парсится."""
    journal = tmp_path / "trials.jsonl"

    code = cli.main(["validate", "--config", "configs/experiments/mr_base.yaml",
                     "--screening", "--data-root", str(tmp_path / "nope"),
                     "--journal", str(journal)])

    assert code == cli.EXIT_ERROR
    assert "Данных нет" in capsys.readouterr().err
