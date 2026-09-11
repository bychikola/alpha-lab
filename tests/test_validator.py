from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from alpha_lab.validation.validator import Verdict, validate


def _case(n=5000, seed=1, strength=0.8, drift=0.0):
    """Согласованный набор: доходности цены, позиции и доходность стратегии.

    strength — доля баров, где позиция совпадает со знаком доходности.
    strength=0.8 даёт настоящий edge, strength=0.0 — чистый шум.
    """
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


def test_strong_strategy_survives():
    v = _run(_case(seed=1, strength=0.8), _trades(300, 3))

    assert isinstance(v, Verdict)
    assert v.alive


def test_pure_noise_is_killed():
    v = _run(_case(seed=2, strength=0.0), _trades(300, 4))

    assert not v.alive
    assert any("DSR" in reason or "p-value" in reason for reason in v.reasons)


def test_too_few_trades_kills():
    v = _run(_case(seed=1, strength=0.8), _trades(20, 5))

    assert not v.alive
    assert any("сделок" in reason for reason in v.reasons)


def test_missing_permutation_inputs_gives_negative_verdict():
    """Тихая деградация недопустима: нет данных для теста — нет вердикта «жива»."""
    r, _, _ = _case(seed=1, strength=0.8)
    v = validate(r, _trades(300, 11), (1 + r).cumprod(), config={}, n_trials=1,
                 strategy_name="s", experiment_id="x")

    assert not v.alive
    assert any(
        "permutation" in reason and "price_returns" in reason and "positions" in reason
        for reason in v.reasons
    )


def test_many_trials_deflate_and_can_kill():
    """Дефляция должна быть видна на числах и уметь убивать.

    С сильным edge (strength=0.8, n=5000) DSR упирается в 1.0 при любом
    n_trials, и утверждение выродилось бы в 1.0 < 1.0. Берём маргинальный
    случай, где дефляция различима: seed=6, n=200, strength=0.55 —
    DSR(1 попытка)=0.999989 (жива), DSR(5000 попыток)=0.790833 (мертва).
    """
    case = _case(n=200, seed=6, strength=0.55)
    v_few = _run(case, _trades(300, 6), n_trials=1)
    v_many = _run(case, _trades(300, 6), n_trials=5000)

    assert v_few.alive
    assert v_many.dsr < v_few.dsr
    assert not v_many.alive
    assert any("DSR" in reason for reason in v_many.reasons)


def test_verdict_is_frozen():
    v = _run(_case(seed=7, strength=0.8), _trades(300, 7))

    with pytest.raises(FrozenInstanceError):
        v.alive = False


def test_thresholds_come_from_config():
    v = _run(_case(seed=8, strength=0.8), _trades(300, 8),
             config={"min_trades": 100000})

    assert not v.alive
    assert any(
        "сделок" in reason and "100000" in reason for reason in v.reasons
    )


def test_reasons_empty_when_alive():
    v = _run(_case(seed=9, strength=0.8), _trades(300, 9))

    assert v.reasons == ()


def test_pbo_is_nan_without_matrix_and_threshold_skipped():
    """Без returns_matrix PBO не определён: порог пропускается явно, не случайно."""
    v = _run(_case(seed=10, strength=0.8), _trades(300, 10),
             config={"max_pbo": 0.0})

    assert np.isnan(v.pbo)
    assert v.alive
    assert not any("PBO" in reason for reason in v.reasons)


def test_pbo_degenerate_matrix_single_config_is_dead():
    """Матрица передана, но PBO невычислим: тихий пропуск недопустим."""
    rng = np.random.default_rng(14)
    matrix = rng.normal(0.0, 0.01, size=(200, 1))
    v = _run(_case(seed=14, strength=0.8), _trades(300, 14),
             returns_matrix=matrix)

    assert np.isnan(v.pbo)
    assert not v.alive
    assert any("PBO не вычислен" in reason for reason in v.reasons)


def test_pbo_too_short_matrix_is_dead():
    """Матрицы короче 2·n_blocks наблюдений для CSCV недостаточно."""
    rng = np.random.default_rng(15)
    matrix = rng.normal(0.0, 0.01, size=(15, 5))
    v = _run(_case(seed=15, strength=0.8), _trades(300, 15),
             returns_matrix=matrix)

    assert np.isnan(v.pbo)
    assert not v.alive
    assert any("PBO не вычислен" in reason for reason in v.reasons)


def test_pbo_computed_when_matrix_given():
    rng = np.random.default_rng(12)
    matrix = rng.normal(0.0, 0.01, size=(200, 5))
    v = _run(_case(seed=12, strength=0.8), _trades(300, 12),
             returns_matrix=matrix)

    assert np.isfinite(v.pbo)
    assert 0.0 <= v.pbo <= 1.0


def test_non_finite_dsr_produces_reason(monkeypatch):
    """NaN DSR не должен молча проходить порог: nan <= x и nan >= x — оба False."""
    import alpha_lab.validation.validator as validator_module

    monkeypatch.setattr(
        validator_module, "deflated_sharpe_ratio", lambda *args, **kwargs: float("nan")
    )
    v = _run(_case(seed=16, strength=0.8), _trades(300, 16))

    assert np.isnan(v.dsr)
    assert not v.alive
    assert any("DSR" in reason for reason in v.reasons)


def test_validation_block_from_experiment_config_is_tolerated():
    """validation: из experiment YAML передаётся целиком, лишние ключи не мешают."""
    v = _run(_case(seed=13, strength=0.8), _trades(300, 13),
             config={"n_splits": 6, "purge": 50, "embargo": 20,
                     "n_permutations": 1000, "min_trades": 100,
                     "max_pbo": 0.5, "max_p_value": 0.05})

    assert v.alive
