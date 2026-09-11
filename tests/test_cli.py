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
from alpha_lab.data.query import (
    align_funding_to_bars, load_bars, load_funding,
)
from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import write_bars, write_funding
from alpha_lab.strategies.base import build_strategy
from alpha_lab.strategies.mean_reversion import atr_decay_bars
from alpha_lab.validation.significance import permutation_pvalue
from alpha_lab.validation.validator import Verdict
from fixtures.synthetic import random_walk_bars
from fixtures.traps import AlwaysLongStrategy, LookAheadStrategy

N_MINUTES = 30000
PARAMS = {"window": 20, "k": 2.0}


def _write_fixture_data(root: Path, n: int = N_MINUTES, seed: int = 1,
                        quote_volume: float = 1e8,
                        symbol: str = "BTCUSDT") -> None:
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
    write_bars(df, root, symbol, "1m")


def _write_configs(tmp_path, root=None, symbols=("BTCUSDT",), timeframe="1h",
                   params="{window: 20, k: 2.0}"):
    """root не используется: конфигам он не нужен, но так короче вызовы."""
    u = tmp_path / "universe.yaml"
    u.write_text(
        "market: futures-um\nstart: '2024-01-01'\nend: '2024-06-30'\n"
        f"symbols: {list(symbols)}\n", encoding="utf-8")
    e = tmp_path / "exp.yaml"
    e.write_text(
        f"name: test\nstrategy: mean_reversion\ntimeframe: {timeframe}\n"
        "start: '2024-01-01'\nend: '2024-06-30'\n"
        f"params: {params}\n"
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


def _remove_hours(root: Path, hours) -> None:
    """Вырезает из минутного хранилища все бары перечисленных часов.

    Дыра в минутных данных даёт разрыв уже на часовом таймфрейме: check_bars
    считает такие разрывы, но вердикт о них раньше молчал.
    """
    bars = load_bars(root, "BTCUSDT", "1m")
    ts = pd.to_datetime(bars["ts"], utc=True)
    wanted = {pd.Timestamp(h).floor("1h") for h in hours}
    write_bars(bars.loc[~ts.dt.floor("1h").isin(wanted)], root, "BTCUSDT", "1m")


def _crafted_gap_bars() -> pd.DataFrame:
    """Ряд, где до дыры открывается лонг и переносится сквозь неё.

    Цены подобраны детерминированно: провал на барах 35–39 даёт z-скор
    (окно 20, k=2) ниже −2 и лонг на баре 36; дальше close идёт внутри
    2×ATR-стопа и 6×ATR-тейка, поэтому полный ряд несёт эту сделку через
    вырезанный бар 40 и после него. Позиция на баре 40 (первый после дыры)
    в несегментированном прогоне ненулевая — это и есть перенос состояния,
    который обязана снять сегментация.
    """
    n = 80
    idx = np.arange(n)
    close = 100.0 + 0.5 * np.sin(idx / 5.0)
    close[35:40] = [99.4, 98.6, 98.8, 99.0, 99.2]
    close[40:] = 99.3 + 0.2 * np.sin(idx[40:] / 7.0)
    open_ = np.concatenate(([close[0]], close[:-1]))
    keep = idx != 40
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")[keep]
    return pd.DataFrame({
        "ts": ts,
        "open": open_[keep],
        "high": (np.maximum(open_, close) + 0.05)[keep],
        "low": (np.minimum(open_, close) - 0.05)[keep],
        "close": close[keep],
        "volume": 1e5,
        "quote_volume": 1e8,
    })


def _segmented_targets(bars: pd.DataFrame) -> np.ndarray:
    """Сырые цели MR с сегментацией по дырам — как их видит CLI."""
    strategy = build_strategy("mean_reversion", PARAMS)
    return cli._generate_segmented(strategy, bars, "1h").to_numpy(dtype="float64")


def _run_validate(root: Path, u: Path, e: Path, out: Path,
                  journal: Path) -> int:
    return main(["validate", "--config", str(e), "--universe", str(u),
                 "--data-root", str(root), "--out", str(out),
                 "--journal", str(journal)])


@pytest.fixture(autouse=True)
def causality_stub(request, monkeypatch):
    """Дешёвый шпион вместо причинностного harness в CLI-тестах не о нём.

    Настоящий harness на 500-баровой фикстуре прогоняет 498 точек усечения
    (~2 с на каждый validate). Это реальное покрытие дублируется выделенными
    тестами с маркером real_causality, а в остальных трёх десятках тестов
    является чистой платой за время. Шпион записывает факт вызова, поэтому
    проводка CLI («harness вызван до бэктеста на полном ряду») остаётся
    проверенной. Число точек шпион не эмулирует и возвращает 0: тест, которому
    важен счётчик causality_cuts, обязан быть помечен real_causality.
    """
    if request.node.get_closest_marker("real_causality") is not None:
        yield []
        return
    calls: list[dict] = []

    def stub(strategy, bars, cut_points=None):
        calls.append({"strategy": strategy, "n_bars": len(bars),
                      "cut_points": cut_points})
        return 0

    monkeypatch.setattr(cli, "assert_strategy_is_causal", stub)
    yield calls


def test_experiment_id_is_deterministic():
    cfg = {"a": 1, "b": [1, 2]}
    assert experiment_id(cfg, "dv1", "g1") == experiment_id(cfg, "dv1", "g1")
    assert experiment_id(cfg, "dv1", "g1") != experiment_id(cfg, "dv2", "g1")


def test_experiment_id_ignores_key_order():
    assert experiment_id({"a": 1, "b": 2}, "d", "g") == \
           experiment_id({"b": 2, "a": 1}, "d", "g")


def test_experiment_id_includes_validation(tmp_path, monkeypatch):
    """Пороги валидации — часть гипотезы, а не оформление.

    Иначе два прогона с разными min_trades делят experiment_id: второй молча
    перезапишет отчёт первого, а журнал сочтёт их одной попыткой.
    """
    _, e1 = _write_configs(tmp_path)
    e2 = tmp_path / "exp2.yaml"
    e2.write_text(
        e1.read_text(encoding="utf-8").replace(
            "validation: {min_trades: 1}", "validation: {min_trades: 999}"),
        encoding="utf-8")

    seen: list[dict] = []
    real = cli.experiment_id

    def spy(cfg, dv, gh):
        seen.append(cfg)
        return real(cfg, dv, gh)

    monkeypatch.setattr(cli, "experiment_id", spy)
    for config in (e1, e2):
        # Данных нет намеренно: experiment_id считается до чтения баров,
        # поэтому тесту не нужен полный бэктест.
        assert main(["validate", "--config", str(config), "--data-root",
                     str(tmp_path / "no_data"), "--out",
                     str(tmp_path / "out")]) == 2

    assert len(seen) == 2
    assert seen[0] != seen[1]              # RED без validation в нагрузке
    assert "validation" in seen[0]
    assert real(seen[0], "d", "g") != real(seen[1], "d", "g")


def test_validate_command_writes_report(tmp_path, capsys, causality_stub):
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
    # Чистый ряд: ни одного разрыва и ни одного бара, исключённого маской дыр.
    assert payload["extra"]["gaps"] == 0
    assert payload["extra"]["gap_masked_bars"] == 0
    # Проводка harness проверена и с дешёвым шпионом: CLI обязан позвать
    # причинностную проверку на полном ряду (500 часовых баров фикстуры).
    assert len(causality_stub) == 1
    assert causality_stub[0]["n_bars"] == 500

    printed = capsys.readouterr().out
    assert "ВЕРДИКТ" in printed
    assert payload["verdict"]["experiment_id"] in printed
    # Пользователь обязан видеть, почему вердикт такой, а не только статус.
    for reason in payload["verdict"]["reasons"]:
        assert reason in printed


def test_report_series_ts_matches_bar_timestamps(tmp_path):
    """series.ts обязан содержать реальные времена баров, а не номера позиций.

    Движок индексирует ряды bars.index, а load_bars заканчивается
    reset_index(drop=True) — это RangeIndex 0..n-1. Если отдать его в
    build_report, pd.Timestamp(0/1/...) превратит ось времени в наносекунды
    от эпохи 1970 года: графики дашборда станут бессмысленными, при этом
    значения выглядят правдоподобно и ничто не кричит об ошибке.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    # format="ISO8601": у эпохи 1970 дробная часть разной длины, и строгий
    # разбор падал бы ValueError ещё до сравнения — тест обязан показывать
    # именно несовпадение времён.
    ts = pd.to_datetime(payload["series"]["ts"], utc=True, format="ISO8601")
    bars = _hourly_bars(root)

    assert len(ts) == len(bars)
    # Именно сверка с фактической колонкой ts фикстуры, а не «год не 1970»:
    # проверка на год пропустила бы любой сдвиг внутри правильного года.
    assert ts[0] == pd.Timestamp(bars["ts"].iloc[0])
    assert ts[-1] == pd.Timestamp(bars["ts"].iloc[-1])
    assert ts.equals(pd.DatetimeIndex(bars["ts"]))


def test_report_series_ts_is_monotonic_and_unique(tmp_path):
    """Ось времени отчёта обязана строго возрастать и не иметь дублей.

    Фикстура чистая (ни одного грязного бара), поэтому любые повторы или
    перестановки ts означали бы, что ряды переиндексированы неверно.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    ts = pd.to_datetime(payload["series"]["ts"], utc=True, format="ISO8601")

    assert ts.is_monotonic_increasing
    assert ts.is_unique


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


def test_validate_survives_corrupt_journal(tmp_path):
    """Битый журнал не роняет прогон: валидные строки вокруг мусора считаются."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    journal = tmp_path / "trials.jsonl"
    key = {"strategy": "mean_reversion", "params": PARAMS,
           "symbol": "BTCUSDT", "timeframe": "1h"}
    record = json.dumps({"key": key, "experiment_id": "x"}).encode()
    # Первой идёт строка с невалидным UTF-8: она проверяет, что чтение файла
    # не падает ещё до разбора строк.
    journal.write_bytes(b"\n".join([
        record, b"\xff\xfe", b"null", b'{"key": "mean', b"[]", record,
    ]) + b"\n")

    out = tmp_path / "out"
    assert _run_validate(root, u, e, out, journal) == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    # Две валидные записи + текущий прогон; мусор пропущен.
    assert payload["verdict"]["n_configs_tried"] == 3


def test_failed_report_write_leaves_journal_unchanged(tmp_path, monkeypatch):
    """Журнал пишется после отчёта: иначе упавшая запись оставит фантомную попытку."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    journal = tmp_path / "trials.jsonl"

    def boom(*args, **kwargs):
        raise RuntimeError("диск полон")

    monkeypatch.setattr(cli, "write_report", boom)
    with pytest.raises(RuntimeError):
        _run_validate(root, u, e, tmp_path / "out", journal)

    assert not journal.exists()


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


def test_count_prior_trials_skips_malformed_lines(tmp_path):
    """Любой мусор в журнале пропускается, валидные строки вокруг — считаются.

    Журнал — append-only черновик: его может оборвать упавший процесс или
    правка руками. Валидный JSON не-объект (null/[]/123/"x"), обрыв объекта и
    невалидный UTF-8 не имеют права ни уронить прогон, ни обнулить счётчик.
    """
    journal = tmp_path / "trials.jsonl"
    key = {"strategy": "mean_reversion", "params": PARAMS,
           "symbol": "BTCUSDT", "timeframe": "1h"}
    other = {**key, "params": {"window": 30, "k": 2.0}}
    good_key = json.dumps({"key": key, "experiment_id": "e1"}).encode()
    good_other = json.dumps({"key": other, "experiment_id": "e2"}).encode()
    journal.write_bytes(b"\n".join([
        good_key,
        b"null",                          # валидный JSON, но не объект
        b"[]",
        b"123",
        b'"x"',
        b'{"key": "mean_reversion", "params":',   # обрыв объекта
        b"\xff\xfe",                      # не UTF-8
        b'{"key": "\xff"}',               # валидный JSON с битым байтом внутри
        good_other,
        good_key,
    ]) + b"\n")

    assert count_prior_trials(journal, key) == 2
    assert count_prior_trials(journal, other) == 1


def test_unreadable_journal_warns_and_counts_zero(tmp_path, capsys):
    """Нечитаемый журнал — не повод падать, но и не повод молчать.

    Молчаливый ноль занижает число попыток и тем завышает DSR, поэтому CLI
    обязан явно предупредить пользователя.
    """
    journal = tmp_path / "journal_is_a_dir"
    journal.mkdir()
    key = {"strategy": "mean_reversion", "params": PARAMS}

    assert count_prior_trials(journal, key) == 0

    err = capsys.readouterr().err
    assert "Предупреждение" in err
    assert "нечитаем" in err


def test_corrupt_journal_lines_warn_with_skip_count(tmp_path, capsys):
    """Пропущенная строка — потерянная попытка: об этом обязаны предупредить.

    Молчаливый пропуск занижает n_trials и тем завышает DSR, поэтому счётчик
    пропущенного печатается один раз на весь журнал, как и решение про
    нечитаемый файл целиком.
    """
    journal = tmp_path / "trials.jsonl"
    key = {"strategy": "mean_reversion", "params": PARAMS,
           "symbol": "BTCUSDT", "timeframe": "1h"}
    good = json.dumps({"key": key, "experiment_id": "e1"}).encode()
    journal.write_bytes(b"\n".join([
        good,
        b"null",          # не объект
        b"\xff\xfe",      # не UTF-8
        b"{}",            # объект без key
        good,
    ]) + b"\n")

    assert count_prior_trials(journal, key) == 2
    assert "пропущено повреждённых строк: 3" in capsys.readouterr().err


def test_unusable_journal_fails_closed_without_report(tmp_path, capsys):
    """Непригодный журнал останавливает прогон ДО бэктеста и отчёта.

    Иначе на диске остался бы отчёт с n_trials = 1: недодефлированный вердикт
    выглядит ЛУЧШЕ правды и толкает к ложному «жива» — ровно та ошибка, против
    которой и существует поправка на множественные сравнения.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)

    dir_journal = tmp_path / "journal_is_a_dir"
    dir_journal.mkdir()
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("файл вместо каталога", encoding="utf-8")

    for i, journal in enumerate((dir_journal, blocker / "trials.jsonl")):
        out = tmp_path / f"out{i}"
        assert _run_validate(root, u, e, out, journal) == 2
        assert not (out / "report.json").exists()

        err = capsys.readouterr().err
        assert "Ошибка" in err
        assert "журнал" in err
        assert str(journal) in err


def test_unreadable_but_writable_journal_fails_closed(tmp_path, monkeypatch,
                                                      capsys):
    """Журнал, который пишется, но не читается, обязан останавливать прогон.

    Проверять только дозапись мало: файл может открываться на запись и не
    открываться на чтение (POSIX 0200, ACL Windows). Тогда count_prior_trials
    ловит OSError, возвращает 0, и отчёт записывается с n_trials = 1 —
    недодефлированный вердикт выглядит ЛУЧШЕ правды.

    На Windows «запись разрешена, чтение запрещено» через chmod не собрать,
    поэтому нечитаемость имитируется: Path.open падает с OSError на режиме
    чтения ровно для файла журнала, а пробник дозаписи проходит.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    journal = tmp_path / "trials.jsonl"
    key = {"strategy": "mean_reversion", "params": PARAMS,
           "symbol": "BTCUSDT", "timeframe": "1h"}
    log_trial(journal, key, "old", {"sharpe": 1.0})

    real_open = Path.open

    def fake_open(self, mode="r", *args, **kwargs):
        if self == journal and "r" in mode:
            raise OSError(13, "Permission denied (read)")
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fake_open)

    # Функция обязана увидеть проблему чтения, хотя дозапись доступна.
    assert cli.journal_problem(journal) is not None

    out = tmp_path / "out"
    assert _run_validate(root, u, e, out, journal) == cli.EXIT_ERROR
    assert not (out / "report.json").exists()

    err = capsys.readouterr().err
    assert "Ошибка" in err
    assert "журнал" in err
    assert str(journal) in err


def test_ignore_journal_skips_read_and_write(tmp_path, capsys):
    """--ignore-journal — осознанный отказ от защиты: журнал не читается и не
    пишется, n_trials = 1, и об этом громко предупреждают."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)

    def run(journal: Path, out: Path) -> int:
        return main(["validate", "--config", str(e), "--universe", str(u),
                     "--data-root", str(root), "--symbol", "BTCUSDT",
                     "--out", str(out), "--journal", str(journal),
                     "--ignore-journal"])

    # Непригодный путь перестаёт быть препятствием: защита отключена явно.
    bad = tmp_path / "journal_is_a_dir"
    bad.mkdir()
    assert run(bad, tmp_path / "out_bad") == cli.EXIT_OK
    payload = json.loads((tmp_path / "out_bad" / "report.json")
                         .read_text(encoding="utf-8"))
    assert payload["verdict"]["n_configs_tried"] == 1
    err = capsys.readouterr().err
    assert "ignore-journal" in err
    assert "множественных сравнений" in err

    # Чтение тоже пропускается: две прошлые попытки не штрафуют прогон, а
    # запись не добавляет третью.
    seeded = tmp_path / "trials.jsonl"
    key = {"strategy": "mean_reversion", "params": PARAMS,
           "symbol": "BTCUSDT", "timeframe": "1h"}
    log_trial(seeded, key, "old1", {"sharpe": 1.0})
    log_trial(seeded, key, "old2", {"sharpe": 1.0})
    assert run(seeded, tmp_path / "out_seeded") == cli.EXIT_OK
    payload = json.loads((tmp_path / "out_seeded" / "report.json")
                         .read_text(encoding="utf-8"))
    assert payload["verdict"]["n_configs_tried"] == 1
    assert len(seeded.read_text(encoding="utf-8").splitlines()) == 2


def test_experiment_id_includes_symbol(tmp_path, monkeypatch):
    """Символ — часть гипотезы (журнал уже различает его через trial_key).

    Без символа в id прогон того же конфига по другому символу молча затирает
    reports/<exp_id> первого.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    _write_fixture_data(root, seed=7, symbol="ETHUSDT")
    u, e = _write_configs(tmp_path, symbols=("BTCUSDT", "ETHUSDT"))
    monkeypatch.chdir(tmp_path)

    for symbol in ("BTCUSDT", "ETHUSDT"):
        # --out не задан намеренно: проверяем именно каталог по умолчанию.
        assert main(["validate", "--config", str(e), "--universe", str(u),
                     "--data-root", str(root), "--symbol", symbol]) == 0

    reports = sorted((tmp_path / "reports").glob("*/report.json"))
    assert len(reports) == 2, "разные символы обязаны дать разные каталоги"
    ids = [json.loads(p.read_text(encoding="utf-8"))["verdict"]["experiment_id"]
           for p in reports]
    assert ids[0] != ids[1]
    assert reports[0].parent.name == ids[0]
    assert reports[1].parent.name == ids[1]


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


def test_ingest_summary_counts_delisted_and_excluded(tmp_path, monkeypatch,
                                                     capsys):
    """Делистингованные пары видны в итоге ingest, а не растворяются в «ок»."""
    from alpha_lab.data import ingest as ingest_mod
    from alpha_lab.data.ingest import IngestResult

    u, _ = _write_configs(tmp_path, tmp_path)
    results = [
        IngestResult("BTCUSDT", "klines", 3, 1000, "OK"),
        IngestResult("OLDUSDT", "klines", 0, 0,
                     "исключён: include_delisted=false — ошибка выживаемости",
                     delisted=True, excluded=True,
                     last_bar_ts=pd.Timestamp("2022-01-01", tz="UTC")),
    ]
    monkeypatch.setattr(ingest_mod, "ingest_universe", lambda *a, **k: results)

    code = main(["ingest", "--config", str(u), "--data-root",
                 str(tmp_path / "d")])

    assert code == 0
    printed = capsys.readouterr().out
    assert "делистингованных: 1" in printed
    assert "исключено: 1" in printed
    assert "[SKIP]" in printed


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


def test_missing_funding_is_disclosed_as_unavailable(tmp_path, capsys):
    """Нет файла funding — это не «funding = 0», а отсутствие данных.

    Иначе панель издержек не отличает «ставок не было» от «ставки не
    применялись», а вердикт выглядит лучше правды на величину съеденного
    финансирования. Предупреждение обязано быть и в stderr, и в вердикте.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    extra = payload["extra"]
    assert extra["funding_available"] is False
    assert extra["funding_events"] == 0
    assert extra["funding_matched"] == 0

    warnings = payload["verdict"]["warnings"]
    assert any("Funding" in w and "занижен" in w for w in warnings), warnings

    captured = capsys.readouterr()
    assert "Funding" in captured.err and "занижен" in captured.err
    # Предупреждение обязано быть видно и в самом блоке вердикта, а статус
    # funding — читаться отдельной строкой, а не только суммой издержек.
    assert "Предупреждения" in captured.out
    for warning in warnings:
        assert warning in captured.out
    assert "НЕДОСТУПЕН" in captured.out


def test_empty_funding_file_is_also_unavailable(tmp_path, capsys):
    """Существующий, но пустой parquet — та же недоступность, что и отсутствие.

    load_funding отдаёт по нему пустую рамку, align молча заливает нули;
    без явной проверки пустоты отчёт показывал бы «funding = 0» как факт.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"
    fund_dir = root / "funding" / "BTCUSDT"
    fund_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ts": pd.DatetimeIndex([], tz="UTC"), "rate": [],
                  "interval_hours": []}).to_parquet(
        fund_dir / "BTCUSDT-funding.parquet", index=False)

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["funding_available"] is False
    assert payload["extra"]["funding_events"] == 0
    assert any("Funding" in w for w in payload["verdict"]["warnings"])
    assert "Funding" in capsys.readouterr().err


def test_funding_events_are_counted_and_matched(tmp_path, capsys):
    """Есть ставки — в отчёт идут и число событий, и число привязанных к барам."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    bars = _hourly_bars(root)
    ts = bars["ts"].iloc[::8].reset_index(drop=True)
    write_funding(pd.DataFrame({"ts": ts, "rate": 1e-4,
                                "interval_hours": 8.0}), root, "BTCUSDT")
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    extra = payload["extra"]
    assert extra["funding_available"] is True
    assert extra["funding_events"] == len(ts)
    assert extra["funding_matched"] == len(ts)
    assert not any("Funding" in w for w in payload["verdict"]["warnings"])
    captured = capsys.readouterr()
    assert "Funding" not in captured.err
    assert f"Funding        доступен (событий {len(ts)}, " \
           f"привязано к барам {len(ts)})" in captured.out


def test_daily_experiment_annualizes_with_daily_periods(tmp_path, monkeypatch):
    """CLI обязан брать годовой множитель из таймфрейма эксперимента.

    На 1d это 365, а не 8760: перепутанный множитель завышает Sharpe в
    sqrt(24) раз, Calmar — в 24 раза, и в отчёте это никак не видно.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root, timeframe="1d",
                          params="{window: 5, k: 2.0, atr_len: 3}")

    captured: dict = {}
    real_validate = cli.validate

    def spy(*args, **kwargs):
        captured.update(kwargs)
        return real_validate(*args, **kwargs)

    monkeypatch.setattr(cli, "validate", spy)
    out = tmp_path / "out"
    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    assert captured["periods_per_year"] == 365
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["timeframe"] == "1d"


def test_daily_funding_is_fully_accounted(tmp_path):
    """На 1d сумма применённого funding равна всем выплатам периода.

    Решающая проверка W2: ставки 08:00 и 16:00 обязаны попасть в тот же
    дневной бар, что и 00:00. Старое точное совпадение оставляло 1/3 выплат,
    занижая издержки и льстя вердикту.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)          # 30000 минут ≈ 21 сутки
    u, e = _write_configs(tmp_path, root, timeframe="1d",
                          params="{window: 5, k: 2.0, atr_len: 3}")
    funding_ts = pd.date_range("2024-01-01", "2024-01-21 16:00", freq="8h",
                               tz="UTC")
    rates = np.linspace(1e-5, 2e-4, len(funding_ts))
    write_funding(pd.DataFrame({"ts": funding_ts, "rate": rates,
                                "interval_hours": 8.0}), root, "BTCUSDT")
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    extra = payload["extra"]
    assert extra["funding_available"] is True
    assert extra["funding_events"] == len(funding_ts)
    assert extra["funding_matched"] == len(funding_ts)

    bars = load_bars(root, "BTCUSDT", "1m", "2024-01-01", "2024-06-30",
                     resample="1d")
    funding = load_funding(root, "BTCUSDT")
    aligned = align_funding_to_bars(bars, funding, timeframe="1d")
    assert aligned.sum() == pytest.approx(float(funding["rate"].sum()))

    # Сумма издержек отчёта — это ровно Σ held[t]·rate[t]: проверяется не
    # только привязка события, но и то, что движок применил её к позиции бара.
    held = np.asarray(payload["series"]["position"], dtype="float64")
    assert len(held) == len(aligned)
    expected_cost = float(np.sum(held * aligned.to_numpy()))
    assert extra["costs_total"]["funding"] == pytest.approx(expected_cost,
                                                            abs=1e-12)


def test_data_gaps_are_surfaced_and_warned(tmp_path, capsys):
    """Разрыв в данных обязан быть виден, а торговля через него — остановлена.

    spec раздел 8: пропуск → не торговать. Разрыв не инвалидирует весь прогон
    (120 часов дыры на 4 годах сделали бы пару непригодной), но позиция,
    удерживаемая через дыру, обнуляется, и счётчики обязаны это показать.
    Маскируется ровно последний бар перед дырой: движок удерживает
    held[i] = target[i-1], и без обнуления этой цели позиция прошла бы сквозь
    неизвестное движение в пропуске. Вперёд маски нет — сегментация
    перезапускает историю стратегии с первого бара после дыры.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    bars = _hourly_bars(root)
    _remove_hours(root, [bars["ts"].iloc[10]])
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["gaps"] == 1
    assert payload["extra"]["missing_bars"] == 1
    # Один разрыв — ровно один маскированный бар: решение, которое иначе
    # исполнилось бы через дыру. history_bars остаётся в отчёте диагностикой
    # памяти стратегии, но маску вперёд больше не задаёт.
    history = build_strategy("mean_reversion", PARAMS).history_bars
    assert payload["extra"]["history_bars"] == history
    assert payload["extra"]["gap_masked_bars"] == 1

    warnings = payload["verdict"]["warnings"]
    assert any("разрыв" in w and "приостановлена" in w for w in warnings), warnings
    assert any("lookahead=0" in w and "сегментац" in w
               for w in warnings), warnings

    captured = capsys.readouterr()
    assert "разрыв" in captured.err
    assert "приостановлена" in captured.err
    for warning in warnings:
        assert warning in captured.out


def test_gap_mask_window_margins_are_configurable():
    """Окно маскирования — параметр: запас назад и вперёд задаётся явно.

    i — индекс первого бара после разрыва (здесь 04:00 после пропущенного
    03:00). Маскируется [i-lookback, i-1+lookahead]: при lookback=1,
    lookahead=1 — [i-1..i]; при lookback=2, lookahead=3 — [i-2..i+2].
    Дефолт lookahead=0: вперёд маскировать нечего, сегментация сама
    перезапускает историю стратегии.
    """
    ts = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC").delete(3)
    bars = pd.DataFrame({"ts": ts})

    narrow = cli._gap_mask(bars, "1h", lookback=1, lookahead=1)
    wide = cli._gap_mask(bars, "1h", lookback=2, lookahead=3)
    default = cli._gap_mask(bars, "1h")

    assert narrow.tolist() == [False, False, True, True, False, False, False]
    assert wide.tolist() == [False, True, True, True, True, True, False]
    assert default.tolist() == [False, False, True, False, False, False, False]


def test_gap_mask_has_no_forward_window_after_segmentation():
    """Вперёд маска не нужна: сегментация перезапускает историю стратегии.

    После разрыва generate вызывается на свежем участке (см.
    _generate_segmented), поэтому ни одно решение не опирается на окно,
    пересекающее пропуск. Маскируется только последний бар перед дырой —
    без этого движок удержал бы позицию через неизвестное движение
    (held[i] = target[i-1]). Прежняя маска в history_bars вперёд глушила
    достоверные сигналы нового участка.
    """
    ts = pd.date_range("2024-01-01", periods=200, freq="1h", tz="UTC").delete(9)
    bars = pd.DataFrame({"ts": ts})
    history = build_strategy("mean_reversion", PARAMS).history_bars
    # Требование выводится из рекурсии ATR: окно z-скора (20) меньше её
    # decay-горизонта, поэтому history_bars больше 20 — но маску он больше
    # не задаёт.
    assert history == max(PARAMS["window"], atr_decay_bars(14)) > 20

    mask = cli._gap_mask(bars, "1h")
    i = 9                       # первый бар после дыры
    assert mask.sum() == 1
    assert mask[i - 1]          # позиция не пройдёт сквозь дыру
    assert not mask[i - 2]      # бар до него контигуозен с i-1
    assert not mask[i:].any()   # вперёд маски нет: решения достоверны

    # Lookback=1 — несущий: без него бар i-1 не маскируется, и позиция
    # проходит сквозь дыру.
    assert not cli._gap_mask(bars, "1h", lookback=0).any()


def test_gap_blocks_carry_and_trading_resumes(tmp_path, capsys):
    """Сквозная проверка маски разрыва на часовом прогоне.

    Сравниваем весь ряд удержанных позиций с эталоном: обнулена ровно цель
    последнего бара перед дырой (иначе held[k] = target[k-1] прошла бы сквозь
    пропуск), все остальные цели — сегментированные. Заодно проверяется, что
    сигналы свежего участка после дыры маской не глушатся: торговля
    возобновляется сама, по мере прогрева стратегии.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)

    bars = _hourly_bars(root)
    raw = _segmented_targets(bars)          # дыр ещё нет — сегментация no-op
    # Час k: цель на k-1 и k-2 ненулевая — без маски позиция прошла бы сквозь
    # дыру, а бар k-1 ещё и торгуется (позиция из target[k-2]).
    k = next(j for j in range(6, len(raw) - 60)
             if raw[j - 1] != 0.0 and raw[j - 2] != 0.0)
    _remove_hours(root, [bars["ts"].iloc[k]])

    out = tmp_path / "out"
    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["gaps"] == 1
    assert payload["extra"]["missing_bars"] == 1
    assert payload["extra"]["gap_masked_bars"] == 1
    assert payload["extra"]["dirty_bars"] == 0

    # После вырезания часа k первый бар за дырой снова имеет индекс k.
    # Ожидание строится из сегментированных целей (generate вызывается на
    # непрерывных участках), а не из прогона по всему ряду.
    bars_after = _hourly_bars(root)
    raw_after = _segmented_targets(bars_after)

    # Предусловие: сегментация сама перенос НЕ снимает — позиция на баре k
    # по-прежнему равна цели k-1, решённой до дыры.
    assert _held_from_targets(raw_after)[k] != 0.0

    # Маска обнуляет ровно цель k-1; всё остальное — честные решения участков.
    targets = raw_after.copy()
    targets[k - 1] = 0.0
    expected = _held_from_targets(targets)

    assert expected[k] == 0.0                # позиция сквозь дыру не проходит
    assert expected[k - 1] != 0.0            # до дыры торговля идёт
    assert (expected[k + 1:] != 0.0).any()   # после дыры торговля возобновляется

    got = np.asarray(payload["series"]["position"], dtype="float64")
    np.testing.assert_allclose(got, expected, atol=1e-6)


def test_post_gap_signal_is_not_suppressed_by_mask():
    """Честный сигнал нового участка не глушится маской разрыва.

    На crafted-фикстуре сегментированная стратегия даёт ненулевые цели
    начиная с i+24 (i — первый бар после дыры). Прежняя маска
    lookahead=history_bars=94 накрывала весь остаток ряда и выбрасывала эти
    достоверные решения; новая маскирует только бар i-1.
    """
    bars = _crafted_gap_bars()
    i = int(cli._gap_starts(bars, "1h")[0])
    seg = _segmented_targets(bars)
    nz = np.flatnonzero(seg[i:] != 0.0)
    assert len(nz) > 0
    first = i + int(nz[0])
    assert first == i + 24, "сигнал свежего участка внутри прежнего окна маски"

    mask = cli._gap_mask(bars, "1h")
    assert mask.sum() == 1
    assert mask[i - 1]                       # перенос через дыру закрыт
    assert not mask[first]                   # а достоверный сигнал — нет
    assert not mask[first:].any()

    # Конвейер CLI: цель выживает после объединения с clean_mask и маской.
    tradable = ~mask
    targets = seg.copy()
    targets[~tradable] = 0.0
    assert targets[first] == seg[first] != 0.0


def test_gap_segmentation_resets_carried_strategy_state():
    """Дыра перезапускает состояние стратегии, а не только гасит выход.

    MR держит сделку внутри simulate_bracket_exits: direction, entry_bar,
    cur_sl и cur_tp переживают пропуск, и на первом немаскированном баре
    позиция всплывает с предразрывной ценой входа, стопом/тейком и часами
    max_bars, не считавшими пропущенные бары. Обнуление целей этого не лечит:
    маска гасит выход, но не память. Сегментация вызывает generate на
    непрерывных участках, поэтому состояние перезапускается по построению.
    """
    bars = _crafted_gap_bars()
    strategy = build_strategy("mean_reversion", PARAMS)

    full = strategy.generate(bars)
    # Предусловие: без сегментации сделка, открытая до дыры (бар 36), жива и
    # на первом баре после дыры (индекс 40) — именно этот перенос и неверен.
    assert full.iloc[40] != 0.0

    seg = cli._generate_segmented(strategy, bars, "1h")

    # Свежий участок: первые max(window, atr_len)−1 баров после дыры — прогрев,
    # сигналов нет ни при каком сигнале полного ряда, переносить сюда нечего.
    warmup = max(strategy.window, strategy.atr_len) - 1
    assert (seg.iloc[40:40 + warmup] == 0.0).all()
    assert len(seg) == len(bars)
    assert seg.index.equals(bars.index)


def test_gap_segmentation_aligns_to_original_index(tmp_path):
    """Склейка участков сохраняет исходный индекс, длину и раскладку.

    Позиция подписана баром: результат обязан быть длины len(bars), выровнен
    по bars.index, а значения каждого участка — совпадать с отдельным
    прогоном generate по этому участку (индекс среза bars.iloc сохраняется).
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    bars0 = _hourly_bars(root)
    _remove_hours(root, [bars0["ts"].iloc[100], bars0["ts"].iloc[300]])
    bars = _hourly_bars(root)

    strategy = build_strategy("mean_reversion", PARAMS)
    seg = cli._generate_segmented(strategy, bars, "1h")

    # Границы участков — те же дыры, что видит маска (один детектор разрыва).
    # Вырезаны часы с исходными индексами 100 и 300; после удаления двух баров
    # второй разрыв открывается баром 299 (300 − 2 + 1).
    starts = cli._gap_starts(bars, "1h")
    assert starts.tolist() == [100, 299]
    # Маска обнуляет ровно бар перед каждой дырой: 99 и 298; первые бары
    # после дыр (100, 299) маской не трогаются — их сдерживает только прогрев
    # свежего участка.
    mask = cli._gap_mask(bars, "1h")
    assert mask[99] and not mask[100] and mask[298] and not mask[299]
    assert mask.sum() == 2

    assert len(seg) == len(bars)
    assert seg.index.equals(bars.index)

    bounds = [0, *starts.tolist(), len(bars)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        np.testing.assert_array_equal(
            seg.iloc[a:b].to_numpy(dtype="float64"),
            strategy.generate(bars.iloc[a:b]).to_numpy(dtype="float64"))

    # Первый (догэповый) участок совпадает с прогоном по всему ряду: дыра
    # впереди не влияет на прошлое.
    raw = strategy.generate(bars)
    np.testing.assert_array_equal(seg.iloc[:100].to_numpy(dtype="float64"),
                                  raw.iloc[:100].to_numpy(dtype="float64"))


def test_gap_segmentation_is_noop_without_gaps():
    """Нет дыр — нет ни сегментации, ни маски: ряд считается ровно как раньше."""
    bars = random_walk_bars(n=500, freq="1h", seed=3)
    strategy = build_strategy("mean_reversion", PARAMS)
    assert len(cli._gap_starts(bars, "1h")) == 0
    assert not cli._gap_mask(bars, "1h").any()

    seg = cli._generate_segmented(strategy, bars, "1h")
    raw = strategy.generate(bars)

    pd.testing.assert_series_equal(seg, raw)


def _run_validate_extra(root: Path, u: Path, e: Path, out: Path,
                        journal: Path, *extra: str) -> int:
    return main(["validate", "--config", str(e), "--universe", str(u),
                 "--data-root", str(root), "--out", str(out),
                 "--journal", str(journal), *extra])


def test_same_journal_state_gives_same_verdict(tmp_path):
    """Критерий 7: воспроизводимость — при одинаковом состоянии журнала.

    Бит-в-бит report.json не воспроизводится и не может: generated_at — метка
    времени, а n_trials берётся из журнала и растёт с каждым прогоном, меняя
    DSR (у границы — и alive). Гарантия честнее и уже: при идентичных входах,
    версии данных и состоянии журнала совпадает ВЕРДИКТ (alive, метрики,
    причины, предупреждения) и все ряды; различается только generated_at.
    Оба прогона идут с чистым журналом — то есть в одном состоянии.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)

    payloads = []
    for i in range(2):
        out = tmp_path / f"out{i}"
        journal = tmp_path / f"trials{i}.jsonl"     # свежий журнал на прогон
        assert _run_validate(root, u, e, out, journal) == 0
        payloads.append(json.loads((out / "report.json").read_text(encoding="utf-8")))
    first, second = payloads

    # Метка времени различается by design — это и есть единственное отличие.
    assert first["generated_at"] != second["generated_at"]

    first.pop("generated_at")
    second.pop("generated_at")
    # Полное равенство после снятия метки: вердикт, extra и все ряды.
    assert first == second

    # Несущие поля перечислены явно, чтобы падение было локализуемым.
    for key in ("alive", "metrics", "reasons", "warnings", "sharpe", "dsr",
                "p_value", "max_dd", "total_return", "trades",
                "n_configs_tried"):
        assert first["verdict"][key] == second["verdict"][key], key
    for key, series in first["series"].items():
        assert len(series) == len(second["series"][key]), key


@pytest.mark.real_causality
def test_non_causal_strategy_blocks_verdict_before_backtest(tmp_path,
                                                            monkeypatch, capsys):
    """Стратегия без причинности не получает вердикта, и бэктест не запускается.

    Подглядывание статистикой по результатам не ловится (spec 8.1), поэтому
    единственная защита — harness над generate. Он обязан жить в рабочем пути
    CLI, а не только в pytest: иначе новая стратегия молча получит «ЖИВА» с
    отличными метриками. Проверка обязана стоять ДО бэктеста: подменённый
    run_backtest падает, если его всё-таки вызвали.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "build_strategy",
                        lambda name, params: LookAheadStrategy())

    def boom(*args, **kwargs):
        raise AssertionError("бэктест запущен до проверки причинности")

    monkeypatch.setattr(cli, "run_backtest", boom)

    code = _run_validate(root, u, e, out, tmp_path / "trials.jsonl")

    assert code == cli.EXIT_ERROR
    assert not (out / "report.json").exists(), "отчёт не должен быть записан"
    err = capsys.readouterr().err
    # Сообщение обязано называть стратегию и первую разошедшуюся точку усечения.
    assert "trap_lookahead" in err
    assert "не причинн" in err
    assert "k=" in err
    assert "Отчёт не записан" in err


@pytest.mark.real_causality
def test_causal_strategy_proceeds_through_validation(tmp_path, monkeypatch,
                                                     capsys):
    """Причинная стратегия проходит проверку и получает обычный вердикт.

    Контроль в обратную сторону: harness, блокирующий всё подряд, бесполезен.
    AlwaysLong причинна (игнорирует вход), поэтому CLI обязан записать отчёт.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "build_strategy",
                        lambda name, params: AlwaysLongStrategy())

    code = _run_validate(root, u, e, out, tmp_path / "trials.jsonl")

    assert code == cli.EXIT_OK
    assert (out / "report.json").exists()
    assert "не причинн" not in capsys.readouterr().err


@pytest.mark.real_causality
def test_skip_causality_flag_bypasses_check_with_loud_warning(tmp_path,
                                                              monkeypatch,
                                                              capsys):
    """--skip-causality — осознанный отказ от единственной защиты от look-ahead.

    Флаг существует для отладки, но обязан громко объяснять цену: без harness
    подглядывающая стратегия получает вердикт, и он оптимистичен по построению.
    Отказ обязан остаться следом в самом отчёте (extra + warnings), иначе
    готовый report.json с отключённой защитой неотличим от защищённого.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    monkeypatch.setattr(cli, "build_strategy",
                        lambda name, params: LookAheadStrategy())

    code = _run_validate_extra(root, u, e, out, tmp_path / "trials.jsonl",
                               "--skip-causality")

    assert code == cli.EXIT_OK
    assert (out / "report.json").exists()
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert payload["extra"]["causality_checked"] is False
    assert payload["extra"]["causality_cuts"] == 0

    warnings = payload["verdict"]["warnings"]
    assert any("ОТКЛЮЧЕНА" in w and "look-ahead" in w for w in warnings), warnings

    captured = capsys.readouterr()
    assert "skip-causality" in captured.err
    assert "ОТКЛЮЧЕНА" in captured.err
    assert "look-ahead" in captured.err.lower()
    # Дашборд читает verdict.warnings, а не stderr: предупреждение обязано быть
    # в обоих каналах, иначе после закрытия терминала след теряется.
    assert "Предупреждения" in captured.out
    for warning in warnings:
        assert warning in captured.out


@pytest.mark.real_causality
def test_causality_provenance_is_recorded_for_protected_run(tmp_path):
    """Защищённый прогон обязан нести в отчёте число оценённых точек усечения.

    Гарантия harness выборочная: 205 точек на 35 063 проверяемых позициях —
    это 0.58% покрытия. Без счётчика в отчёте покрытие выглядит полным, а
    утечка короче шага сетки (~175 баров) может остаться незамеченной.
    """
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"

    assert _run_validate(root, u, e, out, tmp_path / "trials.jsonl") == 0

    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    extra = payload["extra"]
    assert extra["causality_checked"] is True
    # 500-баровый ряд: политика n <= 600 — сплошное покрытие, k = 2..n-1.
    assert extra["causality_cuts"] == 498
    assert not any("ОТКЛЮЧЕНА" in w for w in payload["verdict"]["warnings"])
