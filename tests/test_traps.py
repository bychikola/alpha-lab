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
не ловит — это фундаментальное ограничение, а не баг (spec 8.1). Регрессия
на слепоту валидатора зафиксирована ниже явно. Отбраковывает подглядывание
не статистика, а причинностный harness (tests/fixtures/causality.py):
generate(bars[:k]) обязана совпадать с generate(bars)[:k]. Тесты ниже
доказывают, что harness действительно ловит обе ловушки и не падает на
причинной AlwaysLong.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.causality import assert_strategy_is_causal
from fixtures.synthetic import ou_bars
from fixtures.traps import (
    AlwaysLongStrategy, LookAheadStrategy, OverfitNoiseStrategy,
    PerfectForesightStrategy,
)

from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost, ZeroCost
from alpha_lab.strategies.base import Strategy
from alpha_lab.validation.metrics import sharpe_ratio
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


def test_trap_strategies_satisfy_strategy_protocol():
    """Ловушки обязаны удовлетворять протоколу Strategy, включая history_bars.

    Протокол — контракт для рабочего пути CLI (`build_strategy`), и если
    фикстуры его не выполняют, тесты проверяют не тот интерфейс, что прод.
    """
    for trap in (LookAheadStrategy, OverfitNoiseStrategy, AlwaysLongStrategy,
                 PerfectForesightStrategy):
        instance = trap()
        assert isinstance(instance, Strategy), trap.__name__
        assert instance.history_bars >= 1


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


def test_lookahead_payoff_identity_and_capacity_rejection():
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
    при реалистичных — неисполнимо и убыточно. Статистический валидатор
    подглядывание не ловит вовсе (spec 8.1): причинность ловит harness, а
    здесь проверяются движок и его диагностика ёмкости.
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


class _MisalignedIndexStrategy:
    """Нарушает индексную конвенцию harness'а: подписывает позиции `ts`, не индексом баров.

    Значения причинны (константа), поэтому поймать нарушение может только
    явная проверка индекса: без неё reset_index/выравнивание по меткам молча
    «нормализовало» бы результат.
    """

    name = "trap_misaligned_index"
    history_bars = 1

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=pd.Index(bars["ts"], name="ts"))


class _SegmentLeak:
    """Подглядывает на один бар вперёд, но только на отрезке [start, end).

    Вне отрезка позиции нулевые, поэтому дефект сегментный: прежние четыре
    точки усечения (250/500/750/999) в него не попадали — при k=250 последняя
    позиция префикса 249 уже не подглядывает, и harness пропускал ловушку.
    При усечении до k последняя позиция k−1 теряет будущий бар и обнуляется,
    поэтому плотная сетка ловит расхождение внутри отрезка.
    """

    name = "trap_segment_leak"
    history_bars = 1

    def __init__(self, start: int = 100, end: int = 249):
        self.start = start
        self.end = end

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        fwd = np.sign(close.shift(-1) / close - 1.0).fillna(0.0)
        pos = pd.Series(0.0, index=bars.index)
        lo, hi = self.start, min(self.end, len(bars))
        if lo < hi:
            pos.iloc[lo:hi] = fwd.iloc[lo:hi]
        return pos


class _WarmupLeak:
    """Утечка нормализации: центрированное окно [t−1, t+1] в первых n_warmup барах.

    Нормировка на будущий бар — тот самый «признак, читающий t+1», который
    spec 8.1 называет незащищённым. Здесь дефект ограничен прогревом: при
    усечении до k у последнего бара префикса будущего нет, окно вырождается,
    и значение отличается от полного прогона. За пределами прогрева позиции
    нулевые, поэтому четыре прежние точки (k ≥ 250) дефекта не видели.
    """

    name = "trap_warmup_leak"
    history_bars = 1

    def __init__(self, n_warmup: int = 100):
        self.n_warmup = n_warmup

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        centered = (close.shift(-1) - close.shift(1)) / (2.0 * close)
        pos = pd.Series(0.0, index=bars.index)
        w = min(self.n_warmup, len(bars))
        pos.iloc[:w] = centered.iloc[:w].fillna(0.0)
        return pos


class _ParityLeak:
    """Подглядывает только на чётных позициях; нечётные точки усечения слепы.

    При n=1001 дефолтные k = 250/500/750/1000 все чётные, поэтому позиция k−1
    нечётна и не подглядывает — ловушка проходила целиком. При n=1000 её ловил
    только k=999 (нечётный): случайность чётности, а не политика harness'а.
    """

    name = "trap_parity_leak"
    history_bars = 1

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        fwd = np.sign(close.shift(-1) / close - 1.0).fillna(0.0).to_numpy()
        idx = np.arange(len(bars))
        vals = np.where(idx % 2 == 0, fwd, 0.0)
        return pd.Series(vals, index=bars.index)


def test_causality_harness_catches_lookahead_trap():
    """Harness обязан поймать подглядывание: усечение вскрывает чтение будущего.

    На усечённом ряде последний бар не имеет будущего, и ловушка ставит там 0;
    в полном прогоне на той же позиции стоит знак следующего бара. Расхождение
    ровно в одной позиции на каждую точку усечения — этого достаточно.

    Конкретное k не привязываем: его определяет политика плотности, и она
    обязана ловить тем раньше, чем плотнее сетка (сейчас — на первой же точке).
    """
    bars = ou_bars(n=1000, seed=22)

    with pytest.raises(AssertionError, match="не причинн"):
        assert_strategy_is_causal(LookAheadStrategy(), bars)


def test_causality_harness_catches_perfect_foresight_trap():
    """PerfectForesight читает весь ряд целиком — harness ловит и его."""
    bars = ou_bars(n=1000, seed=22)

    with pytest.raises(AssertionError, match="не причинн"):
        assert_strategy_is_causal(PerfectForesightStrategy(), bars)


def test_causality_harness_passes_always_long_and_custom_cut_points():
    """AlwaysLong причинна: она игнорирует вход, поэтому усечение ничего не меняет.

    Контроль в обратную сторону: harness, который валит всё подряд, бесполезен.
    Заодно проверяется путь caller-supplied cut_points.
    """
    bars = ou_bars(n=1000, seed=22)

    assert_strategy_is_causal(AlwaysLongStrategy(), bars, cut_points=[7, 333, 999])


def test_causality_harness_enforces_bar_index_convention():
    """Индексная конвенция проверяется, а не обходится reset_index'ом.

    Harness требует, чтобы позиция была подписана баром (индекс результата равен
    индексу входа). Стратегия, подписывающая позиции колонкой ts, — нарушение
    контракта, о котором сообщается явно: молчаливая нормализация скрыла бы
    misalignment, а значения здесь причинны, и без проверки индекса тест прошёл бы.
    """
    bars = ou_bars(n=200, seed=22)

    with pytest.raises(AssertionError, match="индексную конвенцию"):
        assert_strategy_is_causal(_MisalignedIndexStrategy(), bars)


def test_validator_cannot_detect_lookahead_documented_limitation():
    """ИЗВЕСТНОЕ ФУНДАМЕНТАЛЬНОЕ ограничение: validate() слеп к подглядыванию.

    Валидатор — статистика по реализованным доходностям. Подглядывание и
    настоящий edge дают одинаковое совместное распределение (r_t, h_t), поэтому
    никакая проверка по returns/price_returns/positions их не различит (spec 8.1).
    Замер: validate() на выходах движка с ZeroCost даёт alive=True, DSR 1.0,
    p 0.001, годовой Sharpe 125.1. Защита — не статистика, а причинностный
    harness (tests/fixtures/causality.py), которому обязана подвергаться каждая
    стратегия; здесь зафиксирована именно слепота валидатора.

    ЕСЛИ ЭТОТ ТЕСТ НАЧНЁТ ПАДАТЬ — значит в валидаторе появился механизм,
    который различает подглядывание; его нужно понять и осознанно принять,
    а не «починить» порогом или ослабить тест.
    """
    bars = ou_bars(n=3000, seed=21)
    target = LookAheadStrategy().generate(bars)
    res = run_backtest(bars, target, ZeroCost())

    v = validate(res.returns, trade_returns(res), res.equity, config={}, n_trials=1,
                 strategy_name="trap_lookahead", experiment_id="la_blind",
                 price_returns=res.price_returns, positions=res.positions)

    assert v.alive, f"валидатор внезапно поймал look-ahead: {v.reasons}"
    assert v.reasons == ()
    # Статистика «отличная» ровно потому, что ловушка выровнена с доходностью
    # по построению, — это и есть ловушка, а не доказательство мастерства.
    assert v.dsr > 0.95 and v.p_value < 0.05
    assert v.sharpe > 100.0
    # Причинностный harness на тех же данных обязан упасть — он и есть защита.
    with pytest.raises(AssertionError, match="не причинна"):
        assert_strategy_is_causal(LookAheadStrategy(), bars)


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


def test_positions_argument_must_be_engine_held_positions():
    """Контракт validate(positions=...): только УДЕРЖИВАЕМЫЕ движком позиции.

    permutation-тест спрашивает, выровнена ли удерживаемая экспозиция с
    доходностями цены. При ZeroCost выполняется точное тождество
    res.returns == res.positions * res.price_returns, поэтому именно
    result.positions — тот ряд, к которому относится наблюдаемый Sharpe.
    Сырые целевые позиции стратегии сдвинуты на бар (held[t] = target[t-1]),
    их произведение с price_returns — уже не реализованные доходности, и
    перемешивание разрушает связь, которой и так нет.

    Замер на ловушке LookAhead (ZeroCost, 3000 баров): result.positions дают
    p=0.001 (alive=True), strategy.generate(bars) — p=0.344 (kill по p-value),
    то есть подмена аргумента ложно убивает прогон. Task 18 обязан передавать
    result.positions.
    """
    bars = ou_bars(n=3000, seed=21)
    target = LookAheadStrategy().generate(bars)
    res = run_backtest(bars, target, ZeroCost())
    trades = trade_returns(res)
    common = dict(config={}, n_trials=1, strategy_name="trap_lookahead",
                  experiment_id="positions_contract")

    held = validate(res.returns, trades, res.equity,
                    price_returns=res.price_returns, positions=res.positions,
                    **common)
    raw = validate(res.returns, trades, res.equity,
                   price_returns=res.price_returns, positions=target, **common)

    assert held.p_value != raw.p_value
    assert held.p_value < 0.05           # корректный вход: экспозиция выровнена
    assert raw.p_value >= 0.05           # сырые цели: выравнивание потеряно
    assert held.alive
    _assert_killed(raw, "p-value")


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
    """Контрольный случай: РЕАЛИСТИЧНАЯ альфа у границы НЕ должна убиваться.

    Не менее важен, чем тест на убийство ловушек: валидатор, который режет всё
    подряд, бесполезен ровно так же, как тот, что не режет ничего. Прежний
    оракул (годовой Sharpe 78) проходил бы и всеядный валидатор, поэтому здесь
    край у самой границы: годовой Sharpe ≈ 1.7 при пороге ≈ 1.1 (DSR > 0.95
    при n_trials=1: per-bar Sharpe > 1.645/sqrt(n) ≈ 0.0116, годовой ≈ 1.09).
    Замер: per-bar 0.0184, годовой 1.717, DSR 0.9953, p 0.0060, alive=True.
    """
    returns, pr, pos = _case(n=20000, seed=25, strength=0.02)
    raw = sharpe_ratio(returns, annualize=False)
    annual = sharpe_ratio(returns)
    assert 0.015 < raw < 0.03, f"край не у границы: per-bar Sharpe {raw:.4f}"
    assert 1.5 < annual < 2.5, f"годовой Sharpe {annual:.3f} вне реалистичного окна"
    trades = pd.Series(np.random.default_rng(25).normal(0.002, 0.01, 400))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1,
                 strategy_name="real", experiment_id="t3",
                 price_returns=pr, positions=pos)

    assert v.alive, f"Валидатор убил настоящий edge: {v.reasons}"
    assert v.reasons == ()
    assert v.trades == 400
    assert v.dsr > 0.95
    assert v.p_value < 0.05


def test_edge_below_boundary_is_killed():
    """Негативный контроль: край слабее порога обязан погибнуть.

    Слабый, но настоящий сигнал (per-bar Sharpe ≈ 0.005, годовой ≈ 0.48)
    против порога ≈ 1.1. Замер: DSR 0.766 и p 0.251 — убит статистикой, а не
    отсутствием входов. В паре с тестом выше это и есть проверка порогов:
    всеядный валидатор валит первый тест, режущий всё — второй.
    """
    returns, pr, pos = _case(n=20000, seed=32, strength=0.02)
    raw = sharpe_ratio(returns, annualize=False)
    assert 0.0 < raw < 0.0116, f"контроль не ниже границы: per-bar Sharpe {raw:.4f}"
    trades = pd.Series(np.random.default_rng(32).normal(0.002, 0.01, 400))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1,
                 strategy_name="below_boundary", experiment_id="t3_neg",
                 price_returns=pr, positions=pos)

    assert not v.alive, f"край ниже порога выжил: {v.reasons}"
    assert MISSING_INPUT_REASON not in " ".join(v.reasons)
    assert any("p-value" in r or "DSR" in r for r in v.reasons), v.reasons
    assert v.sharpe < 1.1
    assert v.dsr < 0.95 and v.p_value >= 0.05


# Фикстура изоляции n_trials: слабый, но настоящий край у границы, который сам
# по себе (n_trials=1) выживает. Замер per-bar Sharpe 0.0525 на 4000 барах:
# n_trials=1 -> DSR 0.9995, p 0.002, alive=True; n_trials=10000 -> sr0 ≈ 0.061,
# DSR 0.294 — убивает ТОЛЬКО поправка на число попыток.
MANY_TRIALS_CASE = dict(n=4000, seed=28, strength=0.06)
MANY_TRIALS_TRADES = 150


def test_all_traps_are_killed_by_validator():
    """Сводный тест: каждая ловушка обязана получить alive=False за свой дефект.

    Если падает — это баг валидатора, а не теста. Чинить validator.py/significance.py.
    """
    cases = {
        "noise":       dict(n=4000, seed=26, strength=0.0, trades=150, trials=1,
                            defect="p-value"),
        "tiny_sample": dict(n=4000, seed=27, strength=0.8, trades=5, trials=1,
                            defect="недостаточно сделок"),
        "many_trials": {**MANY_TRIALS_CASE, "trades": MANY_TRIALS_TRADES,
                        "trials": 10000, "defect": "10000 попыток"},
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


def test_many_trials_penalty_is_isolated_via_dsr():
    """many_trials обязан убиваться ИМЕННО поправкой на число попыток.

    Прежняя фикстура (чистый шум) умирала от DSR и p-value уже при n_trials=1,
    поэтому реализация, полностью игнорирующая n_trials, проходила бы сводный
    тест. Здесь у стратегии есть слабый настоящий край: при n_trials=1 вердикт
    alive=True со всеми пройденными порогами, при n_trials=10000 те же returns,
    trades и p-value остаются прежними, а гибнет он ровно на DSR.
    """
    returns, pr, pos = _case(**MANY_TRIALS_CASE)
    trades = pd.Series(
        np.random.default_rng(MANY_TRIALS_CASE["seed"]).normal(0.001, 0.01,
                                                                MANY_TRIALS_TRADES)
    )

    def verdict(n_trials):
        return validate(returns, trades, (1 + returns).cumprod(), config={},
                        n_trials=n_trials, strategy_name="many_trials",
                        experiment_id=f"trials_{n_trials}",
                        price_returns=pr, positions=pos)

    one = verdict(1)
    many = verdict(10000)

    assert one.alive, f"край убит уже при n_trials=1: {one.reasons}"
    assert one.reasons == ()
    assert one.dsr > 0.95 and one.p_value < 0.05

    assert not many.alive
    # Единственная причина — DSR: значит, убила именно дефляция на 10000 попыток.
    assert len(many.reasons) == 1, f"убийство не изолировано: {many.reasons}"
    reason = _assert_killed(many, "10000 попыток")
    assert "DSR" in reason
    assert many.n_configs_tried == 10000
    assert many.dsr < 0.95
    # p-value и min_trades ПОБИТОВО те же, что при n_trials=1: значит, они не
    # могли убить many_trials — убила ровно дефляция. Проверяем равенство, а не
    # «< 0.05»: последнее лишь подразумевало бы неизменность.
    assert one.p_value == many.p_value
    assert one.trades == many.trades
    assert many.p_value < 0.05 and many.trades >= 100


@pytest.mark.parametrize("leak_cls", [_SegmentLeak, _WarmupLeak, _ParityLeak])
def test_causality_harness_catches_segment_confined_leaks(leak_cls):
    """Плотная сетка обязана ловить утечку, ограниченную отрезком.

    Четыре структурные точки (n//4, n//2, 3n//4, n−1) проверяли ровно четыре
    позиции, поэтому утечка, целиком лежащая ниже наименьшей из них, проходила:
    замерено на этих трёх классах. Политика _default_cut_points сделана плотной
    именно ради этого, и тест закрепляет, что она такой остаётся.
    """
    bars = ou_bars(n=1001, seed=22)   # нечётная длина: все структурные k чётны

    with pytest.raises(AssertionError, match="не причинн"):
        assert_strategy_is_causal(leak_cls(), bars)


def test_causality_harness_catches_parity_leak_on_even_length():
    """Та же ловушка на чётной длине: раньше её ловил только k=999 — по везению.

    При n=1000 из четырёх структурных точек нечётна лишь последняя, поэтому
    обнаружение зависело от чётности длины ряда, а не от политики harness'а.
    Плотная сетка убирает эту зависимость.
    """
    bars = ou_bars(n=1000, seed=22)

    with pytest.raises(AssertionError, match="не причинн"):
        assert_strategy_is_causal(_ParityLeak(), bars)


def test_default_cut_points_are_dense():
    """Политика разрезов обязана быть плотной, а не четырёхточечной.

    Регрессия на исходный дефект: четыре точки проверяли четыре позиции из n.
    """
    from fixtures.causality import EXHAUSTIVE_LIMIT, _default_cut_points

    short = _default_cut_points(EXHAUSTIVE_LIMIT)
    assert short == list(range(2, EXHAUSTIVE_LIMIT)), "короткий ряд — сплошное покрытие"

    long_cuts = _default_cut_points(2000)
    assert len(long_cuts) >= 200, f"длинный ряд покрыт редкой сеткой: {len(long_cuts)}"
    assert 2 <= min(long_cuts) and max(long_cuts) < 2000

