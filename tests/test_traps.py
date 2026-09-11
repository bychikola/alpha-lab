"""Стратегии-ловушки: проверка самого валидатора.

Каждая ловушка обязана получить alive=False, и каждая — за СВОЙ дефект:
подгонка (DSR), неотличимость от случая (p-value), нехватка сделок
(min_trades). Если тест падает — чинить validator.py/significance.py,
а не подкручивать тест.

Два инварианта:
  1. Слепой слой: src/alpha_lab/validation/** не знает о alpha_lab.strategies.
     Скан статический и обязан прочитать хотя бы один файл.
  2. Ловушки, доказывающие детекцию, обязаны передавать price_returns и
     positions: без них validate() убивает вердикт отсутствием входов
     (Task 15), и тест «проходил» бы не по делу. Хелпер _assert_killed
     проверяет, что среди причин нет missing-input, а есть нужный дефект.

Ловушки подглядывания (LookAhead, PerfectForesight) — не про статистику:
в доходностях подглядывание неотличимо от настоящего edge, и validate() его
не ловит. Их тесты доказывают сам дефект (точное тождество gross[t] == |ret[t]|)
и то, что отбраковывает прогон не валидатор, а диагностика ёмкости движка.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from fixtures.synthetic import ou_bars
from fixtures.traps import (
    AlwaysLongStrategy, LookAheadStrategy, OverfitNoiseStrategy,
    PerfectForesightStrategy,
)

from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost, ZeroCost
from alpha_lab.validation.validator import validate

SRC = Path(__file__).resolve().parents[1] / "src"

# Task 15: validate() без price_returns/positions выносит alive=False с этой
# причиной. Увидеть её в тесте — значит доказать забытые аргументы, а не ловушку.
MISSING_INPUT_REASON = "permutation-тест не выполнен"


def _assert_killed(v, expected_reason: str) -> str:
    """Ловушка убита именно своим дефектом, а не отсутствием permutation-входов."""
    assert not v.alive, f"ловушка выжила: {v.reasons}"
    assert MISSING_INPUT_REASON not in " ".join(v.reasons), (
        f"вердикт отрицателен из-за отсутствия входов, а не из-за дефекта: {v.reasons}"
    )
    matches = [r for r in v.reasons if expected_reason in r]
    assert matches, f"среди причин нет {expected_reason!r}: {v.reasons}"
    return matches[0]


def test_validator_does_not_import_strategies():
    """Инвариант слоёв: validation слеп и не знает о стратегиях."""
    validation_dir = SRC / "alpha_lab" / "validation"
    scanned = 0
    for path in sorted(validation_dir.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        assert "alpha_lab.strategies" not in text, (
            f"{path.name} ссылается на strategies — нарушен инвариант слоёв"
        )
        scanned += 1
    # Пустой glob прошёл бы вхолостую: скан обязан реально прочитать файлы.
    assert scanned > 0, f"скан не прочитал ни одного файла в {validation_dir}"


def test_lookahead_strategy_is_rejected_as_too_good():
    """Подглядывание: сдвиг движка не блокирует его, а точно компенсирует.

    Движок держит held[t] = pos[t-1]. Ловушка кладёт
    pos[t] = sign(close[t+1] - close[t]), поэтому held[t] = sign(ret[t]) —
    сдвиг и заглядывание на один бар сокращаются, и gross[t] == |ret[t]|:
    каждая доходность бара берётся по модулю. Это точное тождество и есть
    доказательство подглядывания (замер: gross 23.67 против -0.06 у buy&hold).

    Утверждение брифа «даже с издержками total_return > 10» на дефолтном
    капитале 10 000 неверно: заявка (turnover ~ 1) в ~22 раза больше объёма
    минутного бара (quote_volume ~ 1000), движок помечает прогон как
    over_capacity, а издержки обнуляют эквити (total_return == -1.0).
    «Слишком хорошо» означает: при нулевых издержках сигнал даёт ~1.6e10,
    при реалистичных — неисполним и убыточен. Статистический валидатор
    подглядывание не ловит вовсе — защита в конвенции времени и диагностике
    ёмкости, поэтому здесь проверяем движок, а не validate().
    """
    bars = ou_bars(n=3000, seed=21)
    pos = LookAheadStrategy().generate(bars)
    close = bars["close"].to_numpy(dtype="float64")
    ret = np.zeros(len(close), dtype="float64")
    ret[1:] = close[1:] / close[:-1] - 1.0

    # Ручная проверка: движок держит ровно знак доходности текущего бара,
    # а gross — её модуль. Сдвиг не защитил: ловушка честно подглядывает.
    gross_only = run_backtest(bars, pos, ZeroCost())
    assert np.array_equal(gross_only.positions.to_numpy()[1:], np.sign(ret[1:]))
    assert np.allclose(gross_only.gross_returns.to_numpy()[1:], np.abs(ret[1:]))
    assert gross_only.gross_returns.sum() > 20.0      # 23.67 на 3000 барах
    assert gross_only.total_return > 10.0             # 1.6e10 — «слишком хорошо»

    # Реалистичные издержки при дефолтном капитале: прогон за пределом
    # ёмкости и убыточен. Отбраковывает его не DSR/p-value, а движок.
    realistic = run_backtest(bars, pos, RealisticCost())
    assert realistic.over_capacity
    assert realistic.max_participation_observed > 1.0
    assert realistic.total_return < 0.0


def test_perfect_foresight_is_impossible_in_practice():
    """Ловушка невозможна на практике: её сигнал в баре t требует бар t+1.

    Проверяем саму ловушку, а не валидатор: иначе kill-тесты доказывали бы
    убийство несуществующего дефекта. Возмущаем ТОЛЬКО будущий (последний)
    бар: одинаковое прошлое и разное будущее дают разные сигналы — значит
    generate читает то, чего на баре t ещё нет (протокол Strategy разрешает
    информацию только по бар t включительно).
    """
    bars = ou_bars(n=1000, seed=22)
    strategy = PerfectForesightStrategy()
    pos = strategy.generate(bars)

    # Последний бар не размечается: следующего бара для него не существует.
    assert pos.iloc[-1] == 0.0

    close_up = bars["close"].copy()
    close_up.iloc[-1] = bars["close"].iloc[-2] + 1.0
    close_down = bars["close"].copy()
    close_down.iloc[-1] = bars["close"].iloc[-2] - 1.0
    up = strategy.generate(bars.assign(close=close_up))
    down = strategy.generate(bars.assign(close=close_down))

    # Одно и то же прошлое, разное будущее — разные позиции бара t-1.
    assert up.iloc[-2] == 1.0
    assert down.iloc[-2] == -1.0
    pd.testing.assert_series_equal(pos.iloc[:-2], up.iloc[:-2])
    pd.testing.assert_series_equal(pos.iloc[:-2], down.iloc[:-2])


def _case(n, seed, strength):
    """Согласованный набор: доходности цены, позиции, доходность стратегии."""
    rng = np.random.default_rng(seed)
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    sign = np.sign(price_ret.to_numpy())
    random_side = rng.choice([-1.0, 1.0], size=n)
    positions = pd.Series(np.where(rng.random(n) < strength, sign, random_side))
    return positions * price_ret, price_ret, positions


def test_noise_strategy_fails_validation():
    """Чистый шум: p-value высокий, DSR низкий — убит за неотличимость от случая."""
    returns, pr, pos = _case(n=4000, seed=23, strength=0.0)
    trades = pd.Series(np.random.default_rng(23).normal(0.0, 0.01, 150))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1,
                 strategy_name="noise", experiment_id="t1",
                 price_returns=pr, positions=pos)

    _assert_killed(v, "p-value")
    assert any("DSR" in r for r in v.reasons)
    assert v.p_value >= 0.05 and v.dsr < 0.95


def test_always_long_on_flat_series_fails():
    """Лонг без альфы на ряде без тренда: постоянная позиция ничего не ловит."""
    bars = ou_bars(n=4000, seed=24)
    positions = AlwaysLongStrategy().generate(bars)
    price_ret = bars["close"] / bars["close"].shift(1) - 1.0
    price_ret = price_ret.fillna(0.0)
    returns = positions * price_ret
    trades = pd.Series(np.random.default_rng(24).normal(0.0, 0.01, 200))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1000,
                 strategy_name="always_long", experiment_id="t2",
                 price_returns=price_ret, positions=positions)

    _assert_killed(v, "p-value")
    # Перестановка постоянного вектора ничего не ломает: p-value вырожден в 1.0,
    # поэтому единственный честный сигнал — DSR (альфы нет).
    assert v.p_value == 1.0
    assert v.dsr < 0.95


def test_overfit_segment_is_killed_by_min_trades():
    """Подгонка под отрезок: 50 баров = 1 сделка — гейт min_trades рубит."""
    bars = ou_bars(n=4000, seed=29)
    positions = OverfitNoiseStrategy(start=1000, end=1050).generate(bars)
    res = run_backtest(bars, positions, RealisticCost())

    v = validate(res.returns, trade_returns(res), res.equity, config={},
                 n_trials=100, strategy_name="overfit", experiment_id="t4",
                 price_returns=res.price_returns, positions=res.positions)

    reason = _assert_killed(v, "недостаточно сделок")
    assert v.trades == 1
    assert "1 < 100" in reason


def test_strong_edge_survives_validation():
    """Контрольный случай: настоящая альфа НЕ должна убиваться.

    Не менее важен, чем тест на убийство ловушек: валидатор, который режет всё
    подряд, бесполезен ровно так же, как тот, что не режет ничего.
    """
    returns, pr, pos = _case(n=20000, seed=25, strength=0.8)
    # Оракул действительно сильный: позиция совпадает со знаком доходности
    # бара в ~90% случаев (0.8 подмешивания + половина случайных совпадений).
    assert (pos.to_numpy() == np.sign(pr.to_numpy())).mean() > 0.85
    trades = pd.Series(np.random.default_rng(25).normal(0.002, 0.01, 400))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1,
                 strategy_name="real", experiment_id="t3",
                 price_returns=pr, positions=pos)

    assert v.alive, f"Валидатор убил настоящий edge: {v.reasons}"
    assert v.reasons == ()
    assert v.trades == 400
    assert v.sharpe > 10.0
    assert v.dsr > 0.95
    assert v.p_value < 0.05


def test_all_traps_are_killed_by_validator():
    """Сводный тест: каждая ловушка обязана получить alive=False за свой дефект.

    Если падает — это баг валидатора, а не теста. Чинить validator.py/significance.py.
    """
    cases = {
        "noise":       dict(n=4000, seed=26, strength=0.0, trades=150, trials=1,
                            defect="p-value"),
        "tiny_sample": dict(n=4000, seed=27, strength=0.8, trades=5, trials=1,
                            defect="недостаточно сделок"),
        "many_trials": dict(n=4000, seed=28, strength=0.0, trades=150, trials=10000,
                            defect="10000 попыток"),
    }

    verdicts = {}
    for label, cfg in cases.items():
        returns, pr, pos = _case(cfg["n"], cfg["seed"], cfg["strength"])
        trades = pd.Series(
            np.random.default_rng(cfg["seed"]).normal(0.001, 0.01, cfg["trades"])
        )
        v = validate(returns, trades, (1 + returns).cumprod(), config={},
                     n_trials=cfg["trials"], strategy_name=label,
                     experiment_id=f"trap_{label}", price_returns=pr, positions=pos)
        _assert_killed(v, cfg["defect"])
        verdicts[label] = v

    # tiny_sample: статистика сама по себе отличная (DSR ~1, p < 0.05) —
    # убивает только гейт min_trades, и это видно по единственной причине.
    tiny = verdicts["tiny_sample"]
    assert tiny.reasons == ("недостаточно сделок: 5 < 100",)
    assert tiny.dsr > 0.95 and tiny.p_value < 0.05
