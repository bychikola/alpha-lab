"""Тесты CLI: сквозной прогон на синтетическом parquet-хранилище.

Сеть не трогается: фикстура пишет минутные бары в tmp_path, CLI читает их
через --data-root, отчёты и журнал попыток тоже уходят в tmp_path.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import alpha_lab.cli as cli
from alpha_lab.cli import count_prior_trials, experiment_id, log_trial, main
from alpha_lab.data.query import load_bars
from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import write_bars
from alpha_lab.strategies.base import build_strategy
from alpha_lab.validation.significance import permutation_pvalue
from alpha_lab.validation.validator import Verdict

N_MINUTES = 30000
PARAMS = {"window": 20, "k": 2.0}


def _write_fixture_data(root: Path, n: int = N_MINUTES, seed: int = 1,
                        quote_volume: float = 1e8) -> None:
    """Кладёт минутные синтетические бары туда, откуда их читает load_bars.

    Пишем именно 1m: CLI читает базовый таймфрейм 1m и ресэмплит его в
    таймфрейм эксперимента. 30000 минут = 500 часовых баров, а прогрев
    стратегии — window=20 и atr_len=14, поэтому сделки гарантированно есть.

    Цена — геометрическое случайное блуждание: аддитивное с такой длиной
    уходит в отрицательные значения, а бары с неположительной ценой clean_mask
    справедливо считает грязными (в фикстуре из бриффа таких набиралось ~206
    часовых баров, и «чистые» тесты падали на счётчике грязных баров).
    """
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = normalize_bars(pd.DataFrame({
        "ts": ts, "open": open_,
        "high": np.maximum(open_, close) * 1.0005,
        "low": np.minimum(open_, close) * 0.9995,
        "close": close, "volume": 1e5, "quote_volume": quote_volume,
        "trades": 500, "taker_buy_volume": 5e4,
    }))
    write_bars(df, root, "BTCUSDT", "1m")


def _write_configs(tmp_path, root=None, symbols=("BTCUSDT",)):
    """root не используется: конфигам он не нужен, но так короче вызовы."""
    u = tmp_path / "universe.yaml"
    u.write_text(
        "market: futures-um\nstart: '2024-01-01'\nend: '2024-06-30'\n"
        f"symbols: {list(symbols)}\n", encoding="utf-8")
    e = tmp_path / "exp.yaml"
    e.write_text(
        "name: test\nstrategy: mean_reversion\ntimeframe: 1h\n"
        "start: '2024-01-01'\nend: '2024-06-30'\n"
        "params: {window: 20, k: 2.0}\n"
        "costs: {taker_fee_bps: 5.0}\n"
        "validation: {min_trades: 1}\n", encoding="utf-8")
    return u, e


def _hourly_bars(root: Path) -> pd.DataFrame:
    """Бары ровно в том виде, в каком их читает CLI (1m -> 1h)."""
    return load_bars(root, "BTCUSDT", "1m", "2024-01-01", "2024-06-30",
                     resample="1h")


def _raw_targets(bars: pd.DataFrame) -> np.ndarray:
    return build_strategy("mean_reversion", PARAMS).generate(bars).to_numpy(
        dtype="float64")


def _held_from_targets(raw: np.ndarray) -> np.ndarray:
    """Конвенция движка: held[t] = target[t-1], held[0] = 0."""
    held = np.empty_like(raw)
    held[0] = 0.0
    held[1:] = raw[:-1]
    return held


def _price_returns(bars: pd.DataFrame) -> np.ndarray:
    close = bars["close"].to_numpy(dtype="float64")
    out = np.zeros(len(close), dtype="float64")
    out[1:] = close[1:] / close[:-1] - 1.0
    return out


def _dirty_hours(root: Path, hours) -> None:
    """Обнуляет объём во всех минутных барах перечисленных часов.

    Нулевой объём — грязный бар по clean_mask, поэтому соответствующий
    часовой бар после ресэмплинга тоже становится грязным.
    """
    bars = load_bars(root, "BTCUSDT", "1m")
    ts = pd.to_datetime(bars["ts"], utc=True)
    wanted = {pd.Timestamp(h).floor("1h") for h in hours}
    bars.loc[ts.dt.floor("1h").isin(wanted), "volume"] = 0.0
    write_bars(bars, root, "BTCUSDT", "1m")


def _run_validate(root: Path, u: Path, e: Path, out: Path,
                  journal: Path) -> int:
    return main(["validate", "--config", str(e), "--universe", str(u),
                 "--data-root", str(root), "--out", str(out),
                 "--journal", str(journal)])


def test_experiment_id_is_deterministic():
    cfg = {"a": 1, "b": [1, 2]}
    assert experiment_id(cfg, "dv1", "g1") == experiment_id(cfg, "dv1", "g1")
    assert experiment_id(cfg, "dv1", "g1") != experiment_id(cfg, "dv2", "g1")


def test_experiment_id_ignores_key_order():
    assert experiment_id({"a": 1, "b": 2}, "d", "g") == \
           experiment_id({"b": 2, "a": 1}, "d", "g")


def test_validate_command_writes_report(tmp_path, capsys):
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"
    journal = tmp_path / "trials.jsonl"

    code = _run_validate(root, u, e, out, journal)

    assert code == 0
    assert (out / "report.json").exists()
    assert (out / "report.js").exists()
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "verdict" in payload
    assert payload["verdict"]["alive"] in (True, False)
    assert len(payload["series"]["equity"]) > 0
    assert len(payload["series"]["position"]) == len(payload["series"]["equity"])
    assert payload["extra"]["symbol"] == "BTCUSDT"
    assert payload["extra"]["timeframe"] == "1h"
    assert payload["extra"]["data_version"]
    assert payload["extra"]["dirty_bars"] == 0

    printed = capsys.readouterr().out
    assert "ВЕРДИКТ" in printed
    assert payload["verdict"]["experiment_id"] in printed
    # Пользователь обязан видеть, почему вердикт такой, а не только статус.
    for reason in payload["verdict"]["reasons"]:
        assert reason in printed


def test_validate_receives_engine_held_positions(tmp_path, monkeypatch):
    """Валидатору обязаны уходить позиции движка (held), а не сырые цели.

    Движок держит позицию бара t на баре t+1: held[t] = target[t-1]. Сырые
    цели смещены на бар, поэтому permutation-тест сравнил бы сигнал не с той
    доходностью и убил бы прогон за мнимый дефект, а не за реальный.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"
    journal = tmp_path / "trials.jsonl"

    captured: dict = {}
    real_validate = cli.validate

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(cli, "validate", spy)
    assert _run_validate(root, u, e, out, journal) == 0

    bars = _hourly_bars(root)
    raw = _raw_targets(bars)
    held = _held_from_targets(raw)
    assert np.abs(raw - held).max() > 0.0, \
        "фикстура обязана различать сырые цели и удержанные позиции"

    got = captured["positions"]
    assert isinstance(got, pd.Series)
    assert got.index.equals(bars.index)
    np.testing.assert_allclose(got.to_numpy(dtype="float64"), held)

    # Тот же p-value, посчитанный по сырым целям, отличается — значит проверка
    # p-value в отчёте действительно ловит подмену ряда.
    price_ret = _price_returns(bars)
    p_held = permutation_pvalue(price_ret, held, n_permutations=1000, seed=0)
    p_raw = permutation_pvalue(price_ret, raw, n_permutations=1000, seed=0)
    assert p_raw != pytest.approx(p_held, abs=1e-9)

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["verdict"]["p_value"] == pytest.approx(p_held, abs=1e-6)
    assert payload["verdict"]["p_value"] != pytest.approx(p_raw, abs=1e-6)


def test_repeated_runs_increment_n_trials(tmp_path):
    """Каждый прогон пишется в журнал; DSR обязан штрафовать за число попыток."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    journal = tmp_path / "trials.jsonl"

    trials = []
    for i in range(4):
        out = tmp_path / f"out{i}"
        assert _run_validate(root, u, e, out, journal) == 0
        payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
        trials.append(payload["verdict"]["n_configs_tried"])

    assert trials == [1, 2, 3, 4]


def test_count_prior_trials_counts_only_same_key(tmp_path):
    journal = tmp_path / "trials.jsonl"
    key = {"strategy": "mean_reversion", "params": PARAMS,
           "symbol": "BTCUSDT", "timeframe": "1h"}
    other = {**key, "params": {"window": 30, "k": 2.0}}

    log_trial(journal, key, "exp1", {"sharpe": 1.0})
    log_trial(journal, key, "exp2", {"sharpe": 1.1})
    log_trial(journal, other, "exp3", {"sharpe": 2.0})

    assert count_prior_trials(journal, key) == 2
    assert count_prior_trials(journal, other) == 1
    assert count_prior_trials(tmp_path / "missing.jsonl", key) == 0

    record = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert record["experiment_id"] == "exp1"
    assert record["sharpe"] == 1.0

    # Битая строка журнала не должна обнулять счётчик и ронять прогон.
    with journal.open("a", encoding="utf-8") as fh:
        fh.write("не json\n")
    assert count_prior_trials(journal, key) == 2


def test_dirty_bar_targets_are_forced_flat(tmp_path, capsys):
    """Бар с нулевым объёмом не торгуется (spec раздел 8).

    Проверяем именно применение маски: находим час, на котором стратегия
    держит позицию и держала бы её дальше, портим эти бары и убеждаемся, что
    удержанная позиция на следующем баре обнулилась.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)

    bars = _hourly_bars(root)
    raw = _raw_targets(bars)
    found = next((j for j in range(5, len(raw) - 3)
                  if raw[j] != 0.0 and raw[j + 1] != 0.0), None)
    assert found is not None, "фикстура обязана давать серию удержаний"
    i = found
    _dirty_hours(root, [bars["ts"].iloc[i], bars["ts"].iloc[i + 1]])

    out = tmp_path / "out"
    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["dirty_bars"] == 2
    position = payload["series"]["position"]
    # Предусловие: без маски на баре i+1 позиция была бы ненулевой.
    assert raw[i] != 0.0 and raw[i + 1] != 0.0
    assert position[i + 1] == 0.0   # цель грязного бара i обнулена
    assert position[i + 2] == 0.0   # цель грязного бара i+1 обнулена

    err = capsys.readouterr().err
    assert "грязн" in err and "2" in err


def test_over_capacity_prints_warning(tmp_path, capsys):
    """Ёмкость — диагностика: при превышении CLI обязан громко предупредить."""
    root = tmp_path / "data"
    # Ничтожный объём бара: заявка на 10k$ заведомо больше 1% объёма.
    _write_fixture_data(root, quote_volume=1e3)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    printed = capsys.readouterr().out
    assert "over_capacity=ДА" in printed
    assert "ПРЕДУПРЕЖДЕНИЕ" in printed

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    capacity = payload["extra"]["capacity"]
    assert capacity["over_capacity"] is True
    assert capacity["cap_hits"] > 0
    assert capacity["max_participation_observed"] > 0.01


def test_capacity_diagnostic_is_printed_when_clean(tmp_path, capsys):
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    printed = capsys.readouterr().out
    assert "cap_hits=0" in printed
    assert "over_capacity=нет" in printed
    assert "max_participation=" in printed
    assert "ПРЕДУПРЕЖДЕНИЕ" not in printed

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["capacity"]["cap_hits"] == 0
    assert payload["extra"]["capacity"]["over_capacity"] is False


def test_dead_verdict_is_still_exit_ok(tmp_path, monkeypatch, capsys):
    """Смерть стратегии — результат исследования, а не ошибка запуска."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"
    dead = Verdict(
        strategy_name="test", experiment_id="deadbeef", sharpe=0.1, dsr=0.2,
        p_value=0.9, max_dd=-0.5, total_return=-0.3, trades=10,
        n_configs_tried=1, alive=False,
        reasons=("DSR 0.200 ≤ 0.95 (с поправкой на 1 попыток)",
                 "p-value 0.900 ≥ 0.05 — неотличимо от случая"),
    )
    monkeypatch.setattr(cli, "validate", lambda **kwargs: dead)

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    printed = capsys.readouterr().out
    assert "МЕРТВА" in printed
    for reason in dead.reasons:
        assert reason in printed


def test_validate_missing_data_returns_error(tmp_path):
    root = tmp_path / "empty"
    u, e = _write_configs(tmp_path, root)

    code = main([
        "validate", "--config", str(e), "--universe", str(u),
        "--data-root", str(root), "--out", str(tmp_path / "out"),
    ])

    assert code == 2


def test_ingest_requires_universe(tmp_path):
    code = main(["ingest", "--config", str(tmp_path / "nope.yaml"),
                 "--data-root", str(tmp_path)])

    assert code == 2


def test_ingest_prints_one_line_per_result(tmp_path, monkeypatch, capsys):
    from alpha_lab.data import ingest as ingest_mod
    from alpha_lab.data.ingest import IngestResult

    u, _ = _write_configs(tmp_path, tmp_path)
    results = [
        IngestResult("BTCUSDT", "klines", 3, 1000, "OK: 1000 баров, проблем нет"),
        IngestResult("ETHUSDT", "funding", 0, 0, "", "нет файлов fundingRate"),
    ]
    monkeypatch.setattr(ingest_mod, "ingest_universe", lambda *a, **k: results)

    code = main(["ingest", "--config", str(u), "--data-root",
                 str(tmp_path / "d")])

    assert code == 1
    printed = capsys.readouterr().out
    assert "BTCUSDT" in printed and "ETHUSDT" in printed
    assert "Итого: 1 успешно, 1 с ошибками" in printed


def test_ingest_all_ok_returns_zero(tmp_path, monkeypatch, capsys):
    from alpha_lab.data import ingest as ingest_mod
    from alpha_lab.data.ingest import IngestResult

    u, _ = _write_configs(tmp_path, tmp_path)
    results = [IngestResult("BTCUSDT", "klines", 3, 1000, "OK")]
    monkeypatch.setattr(ingest_mod, "ingest_universe", lambda *a, **k: results)

    assert main(["ingest", "--config", str(u), "--data-root",
                 str(tmp_path / "d")]) == 0
    assert "Итого: 1 успешно, 0 с ошибками" in capsys.readouterr().out


def test_configure_stdio_forces_utf8(monkeypatch):
    """На Windows cp1251 превращает русский текст в мусор — поток обязан стать UTF-8."""
    calls = []

    class FakeStream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())

    cli._configure_stdio()

    assert calls == [{"encoding": "utf-8"}, {"encoding": "utf-8"}]


def test_configure_stdio_tolerates_streams_without_reconfigure(monkeypatch):
    monkeypatch.setattr(sys, "stdout", io.StringIO())
    monkeypatch.setattr(sys, "stderr", io.StringIO())

    cli._configure_stdio()  # не должно быть исключения


def test_main_configures_stdio(tmp_path, monkeypatch):
    """main обязан сам переключать потоки: иначе фикс легко потерять."""
    calls = []

    class FakeStream:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

        def write(self, text):
            pass

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", FakeStream())
    monkeypatch.setattr(sys, "stderr", FakeStream())

    assert main(["ingest", "--config", str(tmp_path / "nope.yaml"),
                 "--data-root", str(tmp_path)]) == 2
    assert calls == [{"encoding": "utf-8"}, {"encoding": "utf-8"}]
