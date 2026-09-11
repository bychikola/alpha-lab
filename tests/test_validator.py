from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from alpha_lab.validation.metrics import sharpe_ratio
from alpha_lab.validation.validator import Verdict, build_returns_matrix, validate


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


def test_none_matrix_warns_that_pbo_was_not_evaluated():
    """Одиночный прогон: PBO не определён, но это НЕ причина смерти.

    Отсутствие матрицы — не дефект стратегии, а непроверенный гейт spec 6.5.
    Вердикт обязан остаться alive (если остальное чисто) и явно сказать, что
    условие pbo < 0.5 не проверялось; иначе читатель считает, что все гейты
    пройдены. Предупреждение живёт в отдельном канале и на alive не влияет.
    """
    v = _run(_case(seed=17, strength=0.8), _trades(300, 17))

    assert v.alive
    assert v.reasons == ()
    assert any("PBO" in w and "не провер" in w for w in v.warnings), v.warnings
    assert any("6.5" in w for w in v.warnings), v.warnings


def test_unusable_matrix_reason_is_not_duplicated_as_warning():
    """Непригодная матрица — это смерть с причиной, а не предупреждение.

    Причина уже объясняет пропуск гейта; дублировать её в warnings нельзя —
    иначе один дефект выглядел бы как два независимых сигнала.
    """
    rng = np.random.default_rng(18)
    matrix = rng.normal(0.0, 0.01, size=(200, 1))
    v = _run(_case(seed=18, strength=0.8), _trades(300, 18),
             returns_matrix=matrix)

    assert not v.alive
    assert any("PBO не вычислен" in reason for reason in v.reasons)
    assert not any("PBO" in w for w in v.warnings), v.warnings


def test_usable_matrix_has_no_pbo_warning():
    rng = np.random.default_rng(19)
    matrix = rng.normal(0.0, 0.01, size=(200, 5))
    v = _run(_case(seed=19, strength=0.8), _trades(300, 19),
             returns_matrix=matrix)

    assert np.isfinite(v.pbo)
    assert not any("PBO" in w for w in v.warnings), v.warnings


def test_validate_annualizes_with_passed_periods_per_year():
    """validate обязан считать Sharpe/Sortino/Calmar переданным множителем.

    Дефолт сохраняет часовое поведение; 1d-множитель даёт sqrt(24) раз меньше
    Sharpe и ровно в 24 раза меньше Calmar — иначе дневной прогон выглядел бы
    лучше правды, причём незаметно для читателя отчёта.
    """
    case = _case(seed=30, strength=0.8)
    r, _, _ = case
    trades = _trades(300, 30)
    raw = sharpe_ratio(r.to_numpy(), annualize=False)

    hourly = _run(case, trades)
    daily = _run(case, trades, periods_per_year=365)

    assert hourly.sharpe == pytest.approx(raw * np.sqrt(365 * 24), rel=1e-12)
    assert daily.sharpe == pytest.approx(raw * np.sqrt(365), rel=1e-12)
    assert daily.metrics["calmar"] == pytest.approx(
        hourly.metrics["calmar"] / 24.0, rel=1e-9)


def test_validate_rejects_nonpositive_periods_per_year():
    """Множитель ≤ 0 — ошибка вызывающего, а не «метрики в нуле».

    Ноль/отрицательное значение так же невидимо портит вердикт, как и
    неверный таймфрейм, поэтому падаем громко и здесь.
    """
    with pytest.raises(ValueError, match="periods_per_year"):
        _run(_case(seed=31, strength=0.8), _trades(300, 31), periods_per_year=0)


def test_external_warnings_are_preserved_and_do_not_kill():
    """Канал предупреждений принимает строки извне (funding, разрывы) и не
    участвует в вердикте: иначе disclosure-канал стал бы новым гейтом."""
    v = _run(_case(seed=20, strength=0.8), _trades(300, 20),
             warnings=("внешнее предупреждение о данных",))

    assert v.alive
    assert "внешнее предупреждение о данных" in v.warnings
    assert v.reasons == ()


def _matrix_columns(n=200, seed=40) -> dict[str, pd.Series]:
    """Выровненные ряды доходностей трёх «конфигураций» по общему индексу баров."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return {
        "cfg_a": pd.Series(rng.normal(0.0, 0.01, n), index=idx),
        "cfg_b": pd.Series(rng.normal(0.0, 0.01, n), index=idx),
        "cfg_c": pd.Series(rng.normal(0.0, 0.01, n), index=idx),
    }


def test_build_returns_matrix_aligns_columns_on_shared_index():
    columns = _matrix_columns()

    matrix = build_returns_matrix(columns)

    assert matrix.shape == (200, 3)
    assert list(matrix.columns) == ["cfg_a", "cfg_b", "cfg_c"]
    assert matrix.index.equals(columns["cfg_a"].index)
    np.testing.assert_allclose(matrix["cfg_b"].to_numpy(),
                               columns["cfg_b"].to_numpy())


def test_build_returns_matrix_rejects_length_mismatch():
    """Разная длина — это разные ряды, а не «общий кусок»: ValueError, не обрезка."""
    columns = _matrix_columns()
    columns["cfg_short"] = columns["cfg_a"].iloc[:-1]

    with pytest.raises(ValueError, match="cfg_short"):
        build_returns_matrix(columns)


def test_build_returns_matrix_rejects_index_mismatch():
    """Одинаковая длина при сдвинутом времени — молчаливая порча PBO.

    Доходности конфигураций относились бы к разным барам, и CSCV сравнивал бы
    несовместимые ряды. Выравнивать (reindex) нельзя: NaN или сдвиг внутри
    матрицы неотличимы от честных данных.
    """
    columns = _matrix_columns()
    shifted = columns["cfg_a"].copy()
    shifted.index = shifted.index + pd.Timedelta(hours=1)
    columns["cfg_b"] = shifted

    with pytest.raises(ValueError, match="cfg_b"):
        build_returns_matrix(columns)


def test_build_returns_matrix_rejects_single_config():
    """PBO на одной колонке неопределён: вызывающий обязан не делать вид, что нет."""
    columns = _matrix_columns()
    single = {"cfg_a": columns["cfg_a"]}

    with pytest.raises(ValueError, match="2"):
        build_returns_matrix(single)


def test_build_returns_matrix_rejects_identical_columns():
    """Идентичные ряды — одна гипотеза, а не свип: CSCV на них вырожден."""
    columns = _matrix_columns()
    columns["cfg_b"] = columns["cfg_a"].copy()

    with pytest.raises(ValueError, match="идентичн"):
        build_returns_matrix(columns)


def test_build_returns_matrix_rejects_identical_non_reference_pair():
    """Идентичность проверяется попарно, а не только с опорной колонкой.

    Матрица (A, B, B) — это две гипотезы, а не три: PBO на дублирующихся
    колонках вырожден, а штраф за перебор занижен. Сравнение лишь с первой
    колонкой пропускало такой дубликат, поэтому каждая пара обязана быть
    проверена.
    """
    columns = _matrix_columns()
    columns["cfg_c"] = columns["cfg_b"].copy()

    with pytest.raises(ValueError, match="cfg_c") as excinfo:
        build_returns_matrix(columns)

    message = str(excinfo.value)
    # Сообщение обязано называть обе конфигурации и их позиции в матрице,
    # иначе по тексту нельзя понять, какую пару чинить.
    assert "cfg_b" in message
    assert "[1]" in message
    assert "[2]" in message


def test_build_returns_matrix_rejects_nonfinite_values():
    columns = _matrix_columns()
    bad = columns["cfg_a"].copy()
    bad.iloc[7] = np.nan
    columns["cfg_b"] = bad

    with pytest.raises(ValueError, match="cfg_b"):
        build_returns_matrix(columns)


def test_validate_uses_precomputed_pbo_without_recomputing(monkeypatch):
    """CLI считает CSCV один раз на свип: готовый float не пересчитывается.

    Иначе на N конфигураций приходилось бы N одинаковых вызовов CSCV — тот же
    результат за N-кратную плату.
    """
    import alpha_lab.validation.significance as significance

    def boom(matrix, n_blocks=10):
        pytest.fail("pbo_cscv не должен вызываться при готовом pbo_value")

    monkeypatch.setattr(significance, "pbo_cscv", boom)
    rng = np.random.default_rng(22)
    matrix = rng.normal(0.0, 0.01, size=(200, 5))
    v = _run(_case(seed=22, strength=0.8), _trades(300, 22),
             config={"max_pbo": 0.5}, returns_matrix=matrix, pbo_value=0.75)

    assert v.pbo == 0.75
    assert not v.alive
    assert any("PBO 0.75" in reason for reason in v.reasons)


def test_validate_rejects_pbo_value_without_matrix():
    """Предвычисленный PBO без матрицы — ошибка вызывающего, а не тихий пропуск.

    Иначе гейт spec 6.5 «pbo < 0.5» не проверился бы, а вердикт молчал бы об
    этом: значение просто потерялось бы.
    """
    with pytest.raises(ValueError, match="pbo_value"):
        _run(_case(seed=23, strength=0.8), _trades(300, 23), pbo_value=0.3)


def test_pbo_above_threshold_kills_and_names_value(monkeypatch):
    """Гейт PBO обязан срабатывать по порогу из конфига и называть значение.

    Матрица здесь валидна, а число подменено: проверяется именно сравнение
    pbo >= max_pbo и текст причины, а не статистика CSCV.
    """
    import alpha_lab.validation.significance as significance

    monkeypatch.setattr(significance, "pbo_cscv", lambda matrix, n_blocks=10: 0.75)
    matrix = np.column_stack(
        [s.to_numpy(dtype="float64") for s in _matrix_columns().values()])
    v = _run(_case(seed=21, strength=0.8), _trades(300, 21),
             config={"max_pbo": 0.5}, returns_matrix=matrix)

    assert not v.alive
    assert any("PBO 0.75" in reason and "0.5" in reason for reason in v.reasons)
    assert not any("PBO" in w for w in v.warnings)
