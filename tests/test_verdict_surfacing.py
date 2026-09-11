"""S4: неприменимость гейтов и слепые зоны доезжают до читателя вердикта.

Вердикт читают три потребителя: CLI (терминал), отчёт `report.json`/`report.js`
(дашборд) и строка хранилища свипа. Молчаливый пропуск гейта недопустим ни в
одном из них:

* `verdict.inapplicable` — структурное «гейт не проверялся и не может быть
  проверен», отдельно от `reasons` (провалы) и `warnings` (непроверенное в
  этом прогоне);
* `verdict.warnings` — слепые зоны дельта-нейтральной книги (ликвидация ноги,
  биржа, депег, займ, базис), которых нет ни в ценовом ряду, ни в ряде ставок;
* CLI печатает статус «НЕ ВЫНЕСЕН (неприменимые гейты)», а не «МЕРТВА»:
  иначе читатель примет неинформативный инструмент за приговор стратегии.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_lab.cli as cli
import alpha_lab.strategies.base as base
from alpha_lab.batch import _row as batch_row
from alpha_lab.config import Experiment
from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import write_bars, write_funding
from alpha_lab.grid import GridCell
from alpha_lab.report.writer import build_report
from alpha_lab.strategies.base import PositionLegs, Strategy, TwoLegStrategy
from alpha_lab.validation.validator import Verdict, validate


# --- 1. Отчёт: inapplicable — отдельный канал --------------------------------


def _neutral_verdict() -> Verdict:
    rng = np.random.default_rng(51)
    n = 2000
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    positions = pd.Series(np.zeros(n))
    returns = pd.Series(1e-4 + 2e-4 * rng.standard_normal(n))
    return validate(returns, pd.Series(dtype="float64"),
                    (1 + returns).cumprod(), config={}, n_trials=1,
                    strategy_name="neutral", experiment_id="s4",
                    price_returns=price_ret, positions=positions)


def _payload(verdict: Verdict) -> dict:
    n = 5
    idx = pd.RangeIndex(n)
    return build_report(
        verdict,
        equity=pd.Series(np.linspace(1.0, 1.1, n), index=idx),
        close=pd.Series(np.full(n, 100.0), index=idx),
        positions=pd.Series(np.zeros(n), index=idx),
        costs=pd.DataFrame({"fee": np.zeros(n), "slippage": np.zeros(n),
                            "funding": np.zeros(n)}, index=idx),
    )


def test_report_serializes_inapplicable_gates_separately_from_warnings():
    verdict = _neutral_verdict()

    payload = _payload(verdict)

    serialized = payload["verdict"]["inapplicable"]
    assert serialized == list(verdict.inapplicable)
    assert len(serialized) == 2
    assert any("min_trades" in text for text in serialized)
    assert any("permutation" in text for text in serialized)
    # Каналы не смешаны: причины пусты (провалов нет), а неприменимость видна.
    assert payload["verdict"]["reasons"] == []
    assert serialized != payload["verdict"]["warnings"]


def test_report_without_inapplicable_serializes_empty_list():
    """Пустой список, а не null: «неприменимых нет» отличимо от «поле забыли»."""
    rng = np.random.default_rng(52)
    n = 1000
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    positions = pd.Series(rng.choice([-1.0, 1.0], n))
    returns = positions * price_ret

    verdict = validate(returns, pd.Series(np.zeros(300)), (1 + returns).cumprod(),
                       config={"min_trades": 100}, n_trials=1,
                       strategy_name="dir", experiment_id="s4",
                       price_returns=price_ret, positions=positions)

    payload = _payload(verdict)

    assert payload["verdict"]["inapplicable"] == []


# --- 2. CLI: статус не путает «не вынесен» с «мертва» ------------------------


def _fake_verdict(**overrides) -> Verdict:
    fields = dict(
        strategy_name="s", experiment_id="x", sharpe=1.0, dsr=0.99,
        p_value=0.01, max_dd=-0.1, total_return=0.2, trades=200,
        n_configs_tried=1, alive=False, reasons=(), metrics={},
        pbo=0.1, warnings=(), inapplicable=(), screening=False,
        n_permutations=1000,
    )
    fields.update(overrides)
    return Verdict(**fields)


def test_verdict_status_distinguishes_not_rendered_from_dead():
    assert cli.verdict_status(_fake_verdict(alive=True)) == "ЖИВА"
    assert cli.verdict_status(_fake_verdict(inapplicable=("гейт X неприменим",))) \
        == "НЕ ВЫНЕСЕН (неприменимые гейты)"
    # Настоящий провал решает судьбу: статус МЕРТВА, ограничение печатается
    # отдельным блоком (смешивать их нельзя, но и прятать провал нельзя).
    assert cli.verdict_status(_fake_verdict(
        inapplicable=("гейт X неприменим",),
        reasons=("DSR 0.100 ≤ 0.95",))) == "МЕРТВА"
    assert cli.verdict_status(_fake_verdict(
        reasons=("недостаточно сделок: 5 < 100",))) == "МЕРТВА"
    assert cli.verdict_status(_fake_verdict(
        screening=True)) == "КАНДИДАТ (черновой вердикт)"
    # Черновой прогон не «повышает» структурную неприменимость до кандидата.
    assert cli.verdict_status(_fake_verdict(
        screening=True, inapplicable=("гейт X неприменим",))) \
        == "НЕ ВЫНЕСЕН (неприменимые гейты)"


# --- 3. Слепые зоны: канал warnings ------------------------------------------


class _NeutralCarry(TwoLegStrategy):
    """Постоянная дельта-нейтральная книга: net ≡ 0, gross = 2, carry = −1."""

    name = "test_neutral_carry_s4"
    history_bars = 1
    PARAM_NAMES = frozenset()

    def __init__(self, params=None):
        self.params = params or {}

    def generate_legs(self, bars: pd.DataFrame) -> PositionLegs:
        idx = bars.index
        return PositionLegs(
            net=pd.Series(0.0, index=idx),
            gross=pd.Series(2.0, index=idx),
            carry=pd.Series(-1.0, index=idx),
        )


class _LongOnly(Strategy):
    name = "test_long_only_s4"
    history_bars = 1

    def __init__(self, params=None):
        self.params = params or {}

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


def _bars(n, close=100.0):
    if np.isscalar(close):
        close = np.full(n, float(close))
    close = pd.Series(np.asarray(close, dtype="float64"))
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6, "quote_volume": 1e9, "trades": 100,
        "taker_buy_volume": 1e5,
    })


def _loaded_data(bars, funding_rate=None):
    from alpha_lab.cli import LoadedData
    return LoadedData(
        symbol="BTCUSDT", timeframe="1h", bars=bars,
        ts_index=pd.Index(bars["ts"].to_numpy(), name="ts"),
        tradable=np.ones(len(bars), dtype=bool), dirty=0, gaps=0,
        missing_bars=0, gap_masked=0, data_warnings=(), funding_rate=funding_rate,
        funding_available=funding_rate is not None, funding_events=0,
        funding_matched=0, ppy=8760,
    )


def _exp(strategy: str) -> Experiment:
    return Experiment(
        name="s4_probe", strategy=strategy, params={}, timeframe="1h",
        start="2024-01-01", end=None,
        costs={"taker_fee_bps": 5.0, "min_slippage_bps": 0.0,
               "impact_coef": 0.0},
        validation={},
    )


def test_delta_neutral_run_carries_blind_spots_and_inapplicable(monkeypatch):
    monkeypatch.setitem(base.REGISTRY, _NeutralCarry.name, _NeutralCarry)
    n = 12
    bars = _bars(n)
    rate = pd.Series(np.full(n, 1e-3), index=bars.index)

    outcome = cli.run_config(_loaded_data(bars, funding_rate=rate),
                             _exp(_NeutralCarry.name))
    verdict = cli.validate_config(outcome, _loaded_data(bars, funding_rate=rate),
                                  experiment_id="s4", n_trials=1)

    assert outcome.result.positions.tolist() == [0.0] * n
    assert len(verdict.inapplicable) == 2
    # Слепые зоны — в канале warnings (его рисует дашборд), дословно и полно.
    blind = [w for w in verdict.warnings if "Слепые зоны" in w]
    assert len(blind) == 1, verdict.warnings
    for keyword in ("ликвидация", "бирж", "депег", "займ", "базис"):
        assert keyword in blind[0].lower(), (keyword, blind[0])
    assert "не видит" in blind[0].lower()


def test_directional_run_has_no_delta_neutral_blind_spots(monkeypatch):
    monkeypatch.setitem(base.REGISTRY, _LongOnly.name, _LongOnly)
    bars = _bars(20, close=np.linspace(100.0, 110.0, 20))

    outcome = cli.run_config(_loaded_data(bars), _exp(_LongOnly.name))
    verdict = cli.validate_config(outcome, _loaded_data(bars),
                                  experiment_id="s4", n_trials=1)

    assert not any("Слепые зоны" in w for w in verdict.warnings)


# --- 4. Строка хранилища свипа не теряет ограничение -------------------------


def test_batch_row_keeps_inapplicable_in_warnings_not_reasons():
    verdict = _neutral_verdict()
    cell = GridCell(
        index=0, config_id="cfg0", symbol="BTCUSDT",
        experiment=Experiment(
            name="neutral", strategy="funding_harvest", params={},
            timeframe="1h", start="2024-01-01", end=None, costs={},
            validation={}),
    )

    row = batch_row(cell, "dv", experiment_id="e", n_trials=1, verdict=verdict)

    row_warnings = row["warnings"].split(" | ")
    assert any("неприменим" in w for w in row_warnings)
    # Неприменимость — не причина смерти: reasons остаются только провалами.
    assert row["reasons"] == ""
    assert row["alive"] is False


# --- 5. Полный путь CLI: терминал говорит «НЕ ВЫНЕСЕН» ------------------------


def test_cli_prints_not_rendered_status_for_funding_book(tmp_path, capsys):
    """Сеть не трогается: минутные бары и funding пишутся в tmp-хранилище."""
    root = tmp_path / "data"
    n = 6000
    rng = np.random.default_rng(53)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.001, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    df = normalize_bars(pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) * 1.0005,
        "low": np.minimum(open_, close) * 0.9995, "close": close,
        "volume": 1e5, "quote_volume": 1e8, "trades": 500,
        "taker_buy_volume": 5e4,
    }))
    write_bars(df, root, "BTCUSDT", "1m")
    hourly_ts = df["ts"].iloc[::480].reset_index(drop=True)
    write_funding(pd.DataFrame({"ts": hourly_ts, "rate": 1e-3,
                                "interval_hours": 8.0}), root, "BTCUSDT")

    universe = tmp_path / "universe.yaml"
    universe.write_text(
        "market: futures-um\nstart: '2024-01-01'\nend: '2024-01-05'\n"
        "symbols: [BTCUSDT]\n", encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "name: funding_verdict\nstrategy: funding_harvest\ntimeframe: 1h\n"
        "start: '2024-01-01'\nend: '2024-01-05'\n"
        # window=24 при ставке раз в 8 ч держит книгу открытой и собирает
        # funding: DSR проходит, и статус определяется неприменимостью, а не
        # провалом — ровно тот случай, ради которого S4 существует.
        "params: {window: 24, horizon_bars: 24}\n"
        "costs: {taker_fee_bps: 5.0, min_slippage_bps: 0.0, impact_coef: 0.0}\n"
        "validation: {min_trades: 100, n_permutations: 100}\n",
        encoding="utf-8")
    out = tmp_path / "out"

    code = cli.main([
        "validate", "--config", str(exp), "--universe", str(universe),
        "--data-root", str(root), "--out", str(out),
        "--journal", str(tmp_path / "trials.jsonl"),
    ])
    captured = capsys.readouterr()
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))

    assert code == 0
    assert "НЕ ВЫНЕСЕН (неприменимые гейты)" in captured.out
    assert "МЕРТВА" not in captured.out
    assert "min_trades" in captured.out and "неприменим" in captured.out
    assert len(payload["verdict"]["inapplicable"]) == 2
    assert any("Слепые зоны" in w for w in payload["verdict"]["warnings"])
