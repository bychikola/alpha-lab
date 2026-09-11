import pytest
import yaml

from alpha_lab.config import load_experiment, load_universe


def test_load_universe(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text(yaml.safe_dump({
        "market": "futures-um",
        "start": "2022-01-01",
        "end": None,
        "symbols": ["BTCUSDT", "ETHUSDT"],
    }), encoding="utf-8")

    u = load_universe(p)

    assert u.symbols == ["BTCUSDT", "ETHUSDT"]
    assert u.start == "2022-01-01"
    assert u.end is None


def test_universe_rejects_empty_symbols(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text(yaml.safe_dump({"market": "futures-um", "start": "2022-01-01",
                                 "end": None, "symbols": []}), encoding="utf-8")

    with pytest.raises(ValueError, match="хотя бы один символ"):
        load_universe(p)


def test_universe_rejects_duplicate_symbols(tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text(yaml.safe_dump({"market": "futures-um", "start": "2022-01-01",
                                 "end": None, "symbols": ["BTCUSDT", "BTCUSDT"]}),
                 encoding="utf-8")

    with pytest.raises(ValueError, match="дубликат"):
        load_universe(p)


def test_load_experiment(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump({
        "name": "mr_base",
        "strategy": "mean_reversion",
        "params": {"window": 20, "k": 2.0},
        "timeframe": "1h",
        "start": "2022-01-01",
        "end": "2025-12-31",
        "costs": {"taker_fee_bps": 5.0},
        "validation": {"n_splits": 6},
    }), encoding="utf-8")

    e = load_experiment(p)

    assert e.name == "mr_base"
    assert e.params["window"] == 20
    assert e.timeframe == "1h"


def test_experiment_rejects_unknown_timeframe(tmp_path):
    p = tmp_path / "e.yaml"
    p.write_text(yaml.safe_dump({
        "name": "bad", "strategy": "mean_reversion", "params": {},
        "timeframe": "7m", "start": "2022-01-01", "end": "2025-12-31",
        "costs": {}, "validation": {},
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="таймфрейм"):
        load_experiment(p)
