"""Тесты пакетного свипа: переиспользование баров, возобновляемость, изоляция отказов.

Сеть и реальное хранилище не трогаются: минутные бары пишутся в tmp_path,
хранилище результатов подменяется узким FakeStore (интерфейс P3) или
временным parquet-хранилищем самого batch.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_lab.cli as cli
import alpha_lab.strategies.base as strategies_base
from alpha_lab.batch import (
    ProgressReporter, SweepSummary, format_progress, open_result_store,
    run_sweep,
)
from alpha_lab.config import load_universe
from alpha_lab.data.query import data_version
from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import write_bars
from alpha_lab.grid import load_grid
from fixtures.traps import LookAheadStrategy

N_MINUTES = 4000
SYMBOLS = ("BTCUSDT", "ETHUSDT")


class FakeStore:
    """Узкий двойник хранилища P3: идемпотентность по config_id, error=успех."""

    def __init__(self, preset=()):
        self.rows: list[dict] = []
        self.preset = set(preset)

    def completed_ids(self, data_version: str) -> set[str]:
        done = {r["config_id"] for r in self.rows if not r.get("error")}
        return done | self.preset

    def write(self, rows: list[dict]) -> None:
        by_id = {r["config_id"]: r for r in self.rows}
        for row in rows:
            by_id[row["config_id"]] = row
        self.rows = list(by_id.values())


def _write_minute_bars(root: Path, symbol: str, n: int = N_MINUTES,
                       seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = normalize_bars(pd.DataFrame({
        "ts": ts, "open": open_,
        "high": np.maximum(open_, close) * 1.0005,
        "low": np.minimum(open_, close) * 0.9995,
        "close": close, "volume": 1e5, "quote_volume": 1e8,
        "trades": 500, "taker_buy_volume": 5e4,
    }))
    write_bars(df, root, symbol, "1m")


def _write_universe(tmp_path: Path, symbols=("BTCUSDT",)) -> Path:
    path = tmp_path / "universe.yaml"
    path.write_text(
        "market: futures-um\nstart: '2024-01-01'\nend: '2024-06-30'\n"
        f"symbols: {list(symbols)}\n", encoding="utf-8")
    return path


def _write_grid(tmp_path: Path, *, symbols=("BTCUSDT",), timeframes=("1h",),
                k=(1.5, 2.0), strategy="mean_reversion",
                extra_params="") -> Path:
    path = tmp_path / "grid.yaml"
    path.write_text(
        "experiment:\n"
        "  name: test_grid\n"
        f"  strategy: {strategy}\n"
        f"  timeframe: {timeframes[0]}\n"
        "  start: '2024-01-01'\n"
        "  end: '2024-06-30'\n"
        "  params: {window: 20, k: 2.0" + extra_params + "}\n"
        "  costs: {taker_fee_bps: 5.0}\n"
        "  validation: {min_trades: 1, n_permutations: 100}\n"
        "axes:\n"
        f"  symbols: {list(symbols)}\n"
        f"  timeframes: {list(timeframes)}\n"
        f"  params:\n    k: {list(k)}\n",
        encoding="utf-8")
    return path


def _sweep(grid_path: Path, root: Path, universe: Path, store, *, force=False,
           journal=None) -> SweepSummary:
    return run_sweep(
        load_grid(grid_path), data_root=root, universe=load_universe(universe),
        force=force, journal=journal, store=store)


def test_sweep_runs_every_cell_and_writes_rows(tmp_path):
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid = load_grid(_write_grid(tmp_path))
    store = FakeStore()

    summary = _sweep(grid_path=grid.path, root=root, universe=u, store=store,
                     journal=tmp_path / "trials.jsonl")

    assert summary.total == 2
    assert summary.executed == 2
    assert summary.skipped == 0
    assert summary.failed == 0
    assert {r["config_id"] for r in store.rows} == \
           {c.config_id for c in grid.expand()}
    for row in store.rows:
        assert row["symbol"] == "BTCUSDT"
        assert row["timeframe"] == "1h"
        assert row["error"] == ""
        assert isinstance(row["alive"], bool)
        assert np.isfinite(row["sharpe"])
        assert row["trades"] >= 0
        # Строка пригодна для воронки P4: есть причины/предупреждения и DSR.
        assert isinstance(row["reasons"], str)
        assert isinstance(row["warnings"], str)
        assert np.isfinite(row["dsr"])
        json.loads(row["params_json"])["k"] in (1.5, 2.0)


def test_sweep_loads_bars_once_per_symbol_timeframe(tmp_path, monkeypatch):
    """Загрузка баров — раз на (символ, таймфрейм), а не раз на конфигурацию."""
    root = tmp_path / "data"
    for symbol in SYMBOLS:
        _write_minute_bars(root, symbol)
    u = _write_universe(tmp_path, SYMBOLS)
    grid = load_grid(_write_grid(tmp_path, symbols=SYMBOLS,
                                 timeframes=("1h", "4h"), k=(1.5, 2.0)))

    calls: list[tuple] = []
    real_load = cli.load_bars

    def spy(root_, symbol, freq, start, end, resample=None):
        calls.append((symbol, resample))
        return real_load(root_, symbol, freq, start, end, resample=resample)

    monkeypatch.setattr(cli, "load_bars", spy)
    store = FakeStore()
    summary = _sweep(grid_path=grid.path, root=root, universe=u, store=store,
                     journal=tmp_path / "trials.jsonl")

    assert summary.total == 8 and summary.executed == 8
    assert sorted(calls) == [
        ("BTCUSDT", "1h"), ("BTCUSDT", "4h"),
        ("ETHUSDT", "1h"), ("ETHUSDT", "4h"),
    ]


def test_sweep_resume_skips_completed_and_force_reruns(tmp_path, monkeypatch):
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid = load_grid(_write_grid(tmp_path))
    cells = grid.expand()
    store = FakeStore(preset={cells[0].config_id})

    ran: list[str] = []
    real_run = cli.run_config

    def spy(data, exp, **kwargs):
        ran.append(exp.params["k"])
        return real_run(data, exp, **kwargs)

    monkeypatch.setattr(cli, "run_config", spy)
    summary = run_sweep(grid, data_root=root, universe=load_universe(u),
                        store=store, journal=tmp_path / "trials.jsonl")

    assert summary.skipped == 1 and summary.executed == 1
    assert ran == [2.0]                      # k=1.5 уже посчитан и не пересчитан
    assert {r["config_id"] for r in store.rows} == {cells[1].config_id}

    ran.clear()
    forced = run_sweep(grid, data_root=root, universe=load_universe(u),
                       force=True, store=store,
                       journal=tmp_path / "trials2.jsonl")
    assert forced.skipped == 0 and forced.executed == 2
    assert sorted(ran) == [1.5, 2.0]


def test_sweep_records_bad_config_as_failure_and_continues(tmp_path):
    """Ошибка одной конфигурации — строка с error, а не падение свипа.

    Неудачная конфигурация не считается выполненной: следующий запуск
    повторяет её (отказ бывает временным — данные, память), а не хоронит
    навсегда.
    """
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid_path = _write_grid(tmp_path, k=(2.0, "not-a-number"))
    grid = load_grid(grid_path)
    store = FakeStore()

    summary = run_sweep(grid, data_root=root, universe=load_universe(u),
                        store=store, journal=tmp_path / "trials.jsonl")

    assert summary.executed == 2 and summary.failed == 1
    assert len(store.rows) == 2
    failed = [r for r in store.rows if r["error"]]
    assert len(failed) == 1
    assert "not-a-number" in failed[0]["error"]
    assert failed[0]["alive"] is False
    assert failed[0]["sharpe"] != failed[0]["sharpe"]      # NaN в метриках
    assert store.completed_ids("dv") == {r["config_id"] for r in store.rows
                                         if not r["error"]}

    # Повторный запуск видит только неудачную конфигурацию и пробует её снова.
    second = run_sweep(grid, data_root=root, universe=load_universe(u),
                       store=store, journal=tmp_path / "trials2.jsonl")
    assert second.skipped == 1 and second.executed == 1 and second.failed == 1


def test_sweep_missing_data_for_one_symbol_does_not_abort_others(tmp_path):
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")     # ETHUSDT намеренно отсутствует
    u = _write_universe(tmp_path, SYMBOLS)
    grid = load_grid(_write_grid(tmp_path, symbols=SYMBOLS))
    store = FakeStore()

    summary = run_sweep(grid, data_root=root, universe=load_universe(u),
                        store=store, journal=tmp_path / "trials.jsonl")

    assert summary.executed == 4 and summary.failed == 2
    btc = [r for r in store.rows if r["symbol"] == "BTCUSDT"]
    eth = [r for r in store.rows if r["symbol"] == "ETHUSDT"]
    assert len(btc) == 2 and all(not r["error"] for r in btc)
    assert len(eth) == 2 and all("Нет данных" in r["error"] for r in eth)


def test_sweep_rejects_symbol_outside_universe(tmp_path):
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path, ("BTCUSDT",))
    grid = load_grid(_write_grid(tmp_path, symbols=("ETHUSDT",)))

    with pytest.raises(ValueError, match="ETHUSDT"):
        run_sweep(grid, data_root=root, universe=load_universe(u),
                  store=FakeStore(), journal=tmp_path / "trials.jsonl")


def test_sweep_causality_failure_is_recorded_not_raised(tmp_path, monkeypatch):
    """Harness причинности обязан работать в свипе и не ронять его."""
    class TrapStrategy(LookAheadStrategy):
        name = "trap_grid"
        PARAM_NAMES = frozenset({"k", "window"})

        def __init__(self, params=None):
            pass

    monkeypatch.setitem(strategies_base.REGISTRY, "trap_grid", TrapStrategy)
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid = load_grid(_write_grid(tmp_path, strategy="trap_grid"))
    store = FakeStore()

    summary = run_sweep(grid, data_root=root, universe=load_universe(u),
                        store=store, journal=tmp_path / "trials.jsonl")

    assert summary.executed == 2 and summary.failed == 2
    assert all("причинност" in r["error"] for r in store.rows)


def test_format_progress_has_position_elapsed_and_eta():
    text = format_progress(2, 8, elapsed=12.0, eta=36.0,
                           label="BTCUSDT 1h k=2.0")
    assert "[2/8]" in text
    assert "25.0%" in text
    assert "прошло" in text and "осталось" in text
    assert "BTCUSDT 1h k=2.0" in text


def test_progress_reporter_respects_cadence():
    """Строка не на каждую конфигурацию: 100 обновлений дают пару строк."""
    now = [0.0]
    buf = io.StringIO()
    reporter = ProgressReporter(total=100, stream=buf, interval=30.0,
                                clock=lambda: now[0])
    reporter.start()
    for i in range(1, 100):
        now[0] = i * 0.1                       # 9.9 с — интервал не истёк
        reporter.tick(i, "BTCUSDT", eta=1.0)
    assert len(buf.getvalue().splitlines()) == 1     # только стартовая строка
    now[0] = 31.0
    reporter.tick(99, "BTCUSDT", eta=1.0)
    reporter.finish(100)
    lines = buf.getvalue().splitlines()
    assert len(lines) == 3
    assert "[99/100]" in lines[1]
    assert "[100/100]" in lines[2]


def test_sweep_verdicts_match_cli_configs_sweep(tmp_path):
    """Числа свипа обязаны совпасть с production-путём CLI до последнего знака.

    Один и тот же перебор двумя путями: `sweep` по сетке и `validate --configs`
    по манифесту. Сравниваются experiment_id, Sharpe, DSR, p-value, PBO, сделки
    и вердикт каждой конфигурации. Расхождение означало бы, что свип считает
    вердикты другим маршрутом, — худший исход для исследовательского полигона.
    """
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid_path = _write_grid(tmp_path)
    grid = load_grid(grid_path)

    # --configs: два YAML-эксперимента, собранные из тех же ячеек сетки.
    params_list = [c.experiment.params for c in grid.expand()]
    exp_paths = []
    for i, params in enumerate(params_list):
        p = tmp_path / f"exp{i}.yaml"
        p.write_text(
            "name: test_grid\nstrategy: mean_reversion\ntimeframe: 1h\n"
            "start: '2024-01-01'\nend: '2024-06-30'\n"
            f"params: {json.dumps(params)}\n"
            "costs: {taker_fee_bps: 5.0}\n"
            "validation: {min_trades: 1, n_permutations: 100}\n",
            encoding="utf-8")
        exp_paths.append(p)
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("\n".join(str(p) for p in exp_paths), encoding="utf-8")
    cli_out = tmp_path / "cli_out"
    code = cli.main(["validate", "--configs", str(manifest), "--universe",
                     str(u), "--data-root", str(root), "--out", str(cli_out),
                     "--journal", str(tmp_path / "cli_journal.jsonl")])
    assert code == 0
    cli_rows = json.loads(
        (cli_out / "report.json").read_text(encoding="utf-8")
    )["extra"]["sweep"]["configs"]

    store = FakeStore()
    run_sweep(grid, data_root=root, universe=load_universe(u), store=store,
              journal=tmp_path / "batch_journal.jsonl")

    by_k = {json.loads(r["params_json"])["k"]: r for r in store.rows}
    by_index = {entry["index"]: entry for entry in cli_rows}
    assert len(by_k) == len(by_index) == 2
    for i, cell in enumerate(grid.expand()):
        batch = by_k[cell.experiment.params["k"]]
        assert batch["experiment_id"] == by_index[i]["experiment_id"]
        for key in ("sharpe", "dsr", "p_value", "pbo", "trades"):
            assert batch[key] == pytest.approx(by_index[i][key], rel=0,
                                               abs=0), key
        assert batch["alive"] == by_index[i]["alive"]
        assert batch["reasons"] == " | ".join(by_index[i]["reasons"])


def test_cli_sweep_command_runs_and_resumes(tmp_path, capsys):
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid_path = _write_grid(tmp_path)
    store_dir = tmp_path / "results"
    journal = tmp_path / "trials.jsonl"
    argv = ["sweep", "--grid", str(grid_path), "--out", str(store_dir),
            "--universe", str(u), "--data-root", str(root),
            "--journal", str(journal)]

    assert cli.main(argv) == 0
    first = capsys.readouterr().out
    assert "выполнено 2" in first and "пропущено 0" in first

    grid = load_grid(grid_path)
    sink = open_result_store(store_dir)
    assert sink.completed_ids(data_version(root)) == \
           {c.config_id for c in grid.expand()}

    # Второй запуск не трогает данные: все конфигурации уже посчитаны.
    assert cli.main(argv) == 0
    second = capsys.readouterr().out
    assert "выполнено 0" in second and "пропущено 2" in second


def test_cli_sweep_without_journal_uses_default(tmp_path, monkeypatch, capsys):
    """Документированная команда без --journal обязана работать.

    `sweep --grid ... --out ...` — ровно та строка, что записана в плане
    фазы 2. Журнал при этом берётся по умолчанию, как в одиночном validate:
    Path(None) до run_sweep не доходит.
    """
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid_path = _write_grid(tmp_path)
    store_dir = tmp_path / "results"
    journal = tmp_path / "default_journal.jsonl"
    monkeypatch.setattr(cli, "DEFAULT_JOURNAL", journal)

    argv = ["sweep", "--grid", str(grid_path), "--out", str(store_dir),
            "--universe", str(u), "--data-root", str(root)]
    assert cli.main(argv) == 0
    out = capsys.readouterr().out
    assert "выполнено 2" in out and "пропущено 0" in out
    # Журнал по умолчанию реально ведётся: обе попытки записаны.
    assert journal.exists()
    assert len(journal.read_text(encoding="utf-8").splitlines()) == 2


def test_cli_sweep_force_reruns_everything(tmp_path, capsys):
    root = tmp_path / "data"
    _write_minute_bars(root, "BTCUSDT")
    u = _write_universe(tmp_path)
    grid_path = _write_grid(tmp_path)
    store_dir = tmp_path / "results"
    journal = tmp_path / "trials.jsonl"
    argv = ["sweep", "--grid", str(grid_path), "--out", str(store_dir),
            "--universe", str(u), "--data-root", str(root),
            "--journal", str(journal)]

    assert cli.main(argv) == 0
    capsys.readouterr()
    assert cli.main([*argv, "--force"]) == 0
    assert "выполнено 2" in capsys.readouterr().out


def test_cli_sweep_warns_loudly_about_large_grid(tmp_path, monkeypatch, capsys):
    """Предупреждение о размере печатается до запуска, а не через час."""
    from alpha_lab import batch as batch_mod

    root = tmp_path / "data"
    u = _write_universe(tmp_path)
    k_values = [1.0 + 0.001 * i for i in range(1001)]
    grid_path = _write_grid(tmp_path, k=k_values)

    calls: list[dict] = []

    def stub(*args, **kwargs):
        calls.append(kwargs)
        return SweepSummary(total=1001, executed=0, skipped=0, failed=0,
                            store_path=tmp_path / "results")

    monkeypatch.setattr(batch_mod, "run_sweep", stub)
    code = cli.main(["sweep", "--grid", str(grid_path), "--out",
                     str(tmp_path / "results"), "--universe", str(u),
                     "--data-root", str(root), "--journal",
                     str(tmp_path / "trials.jsonl")])

    assert code == 0
    err = capsys.readouterr().err
    assert "ВНИМАНИЕ" in err
    assert "1 001" in err and "мин" in err
    # Предупреждение не блокирует осознанный запуск: свип вызван.
    assert calls
