"""S2: стратегия сбора funding — решение по режиму ставки, а не по цене.

Дельта-нейтральная книга (net ≡ 0) не имеет направленных сделок: весь доход —
funding на ногу перпа, все издержки — оборот двух ног. Здесь проверяются:

* априорное правило входа: удержание, пока скользящее среднее ставки выше
  порога окупаемости круглого рейса, иначе — вне рынка;
* знак ставки: отрицательный режим не удерживается молча;
* строгая граница порога: ровно на пороге ожидаемая прибыль нулевая — стоим;
* причинность: harness проходит по всем трём базам;
* контракт «вошёл — держи — вышел»: carry ходит только 0 ↔ −1, оборот только
  0 ↔ 2, без ребалансировок и разворотов;
* закрепление известного ограничения magnitude-базиса S1 (встречное движение
  ног и прямой разворот carry не видны в |Δgross|) — тест обязан упасть, если
  движок когда-нибудь «починят» иначе, чтобы это не прошло незамеченным;
* синтетический контроль: на постоянной положительной ставке доход равен
  ровно сумме ставок минус издержки двух ног круглого рейса.

Сеть не трогается: все бары синтетические.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import alpha_lab.cli as cli
from alpha_lab.causality import assert_strategy_is_causal
from alpha_lab.config import Experiment
from alpha_lab.engine.backtest import run_backtest
from alpha_lab.engine.costs import RealisticCost
from alpha_lab.strategies.base import (
    FUNDING_RATE_COLUMN, build_strategy, strategy_param_names,
)
from alpha_lab.strategies.funding_harvest import DEFAULTS, FundingHarvestStrategy

# Издержки, при которых арифметика контроля точна: комиссия 10 bps за единицу
# оборота, проскальзывания нет. Круглый рейс двухногой книги — 4 единицы
# оборота (вход 2 + выход 2), то есть 4e-3.
_FEE_BPS = 10.0
_TWO_LEG_COST = RealisticCost(taker_fee_bps=_FEE_BPS, min_slippage_bps=0.0,
                              impact_coef=0.0)


def _bars(n, close=100.0, funding_rate=None, freq="1h", quote_volume=1e9):
    close = pd.Series(close, dtype="float64") if not np.isscalar(close) else \
        pd.Series([float(close)] * n, dtype="float64")
    n = len(close)
    df = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6, "quote_volume": quote_volume, "trades": 100,
        "taker_buy_volume": 1e5,
    })
    if funding_rate is not None:
        if np.isscalar(funding_rate):
            rate = np.full(n, float(funding_rate))
        else:
            rate = np.asarray(funding_rate, dtype="float64")
        assert len(rate) == n
        df[FUNDING_RATE_COLUMN] = rate
    return df


def _loaded_data(bars, funding_rate=None, tradable=None):
    from alpha_lab.cli import LoadedData
    ts_index = pd.Index(bars["ts"].to_numpy(), name="ts")
    if tradable is None:
        tradable = np.ones(len(bars), dtype=bool)
    return LoadedData(
        symbol="BTCUSDT", timeframe="1h", bars=bars, ts_index=ts_index,
        tradable=tradable, dirty=0, gaps=0, missing_bars=0, gap_masked=0,
        data_warnings=(), funding_rate=funding_rate,
        funding_available=funding_rate is not None, funding_events=0,
        funding_matched=0, ppy=8760,
    )


def _exp(params=None, costs=None, name="funding_test"):
    return Experiment(
        name=name, strategy="funding_harvest", params=params or {},
        timeframe="1h", start="2024-01-01", end=None,
        costs=costs if costs is not None else {
            "taker_fee_bps": _FEE_BPS, "min_slippage_bps": 0.0,
            "impact_coef": 0.0,
        },
        validation={},
    )


def _strategy(**params):
    return build_strategy("funding_harvest", params)


def _entries_exits(gross):
    g = np.asarray(gross, dtype="float64")
    prev = np.concatenate(([0.0], g[:-1]))
    entries = int(((prev == 0.0) & (g > 0.0)).sum())
    exits = int(((prev > 0.0) & (g == 0.0)).sum())
    return entries, exits


# --- 1. Регистрация и параметры ----------------------------------------------


def test_registered_in_build_strategy_with_declared_params():
    """Стратегия доступна через штатный build_strategy, параметры объявлены."""
    strategy = _strategy()
    assert isinstance(strategy, FundingHarvestStrategy)
    assert strategy.name == "funding_harvest"
    assert strategy_param_names("funding_harvest") == frozenset(DEFAULTS)
    assert strategy.history_bars == DEFAULTS["window"]


def test_parameters_are_validated():
    """Невалидные параметры — громкий отказ, а не тихо другой горизонт."""
    with pytest.raises(ValueError, match="window"):
        _strategy(window=0)
    with pytest.raises(ValueError, match="horizon_bars"):
        _strategy(horizon_bars=0)
    with pytest.raises(ValueError, match="fee_bps"):
        _strategy(fee_bps=-1.0)
    with pytest.raises(ValueError, match="slippage_bps"):
        _strategy(slippage_bps=-1.0)
    with pytest.raises(ValueError, match="threshold_rate"):
        _strategy(threshold_rate=float("nan"))
    with pytest.raises(ValueError, match="threshold_rate"):
        _strategy(threshold_rate="0")
    with pytest.raises(ValueError, match="threshold_rate"):
        _strategy(threshold_rate=True)


# --- 2. Синтетический контроль: точная арифметика ----------------------------

#window=72 (полное окно), horizon=720, fee=10 bps, slippage=0
# порог = 4·10e-4/720 = 5.5556e-6 на бар.
# Ставка 1e-3 держится бары 0..99, дальше 0. Первое полное окно — бар 71:
# цель −1 с бара 71, движок удерживает carry бары 72..171 (100 баров).
# Ненулевые ставки на удержанных барах — 72..99, это 28 баров: доход 28e-3.
# Вход и выход торгуют по gross=2, то есть 4 единицы оборота: издержки 4e-3.
_CONTROL_WINDOW = 72
_CONTROL_HORIZON = 720
_CONTROL_RATE = 1e-3
_CONTROL_POSITIVE_BARS = 100
_CONTROL_N = 200


def test_control_constant_positive_rate_earns_exact_sum_minus_round_trip():
    """Контроль модели: доход = ровно сумма ставок минус две ноги круглого рейса.

    Ставка известна, цена стоит, поэтому ценовой P&L обязан быть нулём, а
    итог — арифметикой, а не знаком: entry/exit по 2 единицы оборота каждый.
    Прогон идёт полным рабочим путём CLI (run_config), а не вызовом движка
    напрямую: так проверяется и проводка funding-колонки в решение.
    """
    rate = np.concatenate([
        np.full(_CONTROL_POSITIVE_BARS, _CONTROL_RATE),
        np.zeros(_CONTROL_N - _CONTROL_POSITIVE_BARS),
    ])
    # Бары — БЕЗ колонки funding: ставку обязан прикрепить CLI (needs_funding),
    # иначе этот тест не проверял бы проводку, а читал бы колонку напрямую.
    bars = _bars(_CONTROL_N)
    exp = _exp(params={"window": _CONTROL_WINDOW,
                       "horizon_bars": _CONTROL_HORIZON,
                       "fee_bps": _FEE_BPS, "slippage_bps": 0.0})
    outcome = cli.run_config(
        _loaded_data(bars, funding_rate=pd.Series(rate)), exp)
    res = outcome.result

    # Решение: дельта-нейтрально, 100 баров в книге (72..171), вход/выход 1/1.
    assert res.positions.tolist() == [0.0] * _CONTROL_N
    assert outcome.history == _CONTROL_WINDOW
    signal_bars = bars.assign(**{FUNDING_RATE_COLUMN: rate})
    legs = _strategy(window=_CONTROL_WINDOW, horizon_bars=_CONTROL_HORIZON,
                     fee_bps=_FEE_BPS, slippage_bps=0.0).generate_legs(signal_bars)
    assert legs.carry.tolist() == [0.0] * 71 + [-1.0] * 100 + [0.0] * 29
    assert _entries_exits(legs.gross) == (1, 1)
    in_book = float((legs.carry != 0.0).mean())
    assert in_book == pytest.approx(0.5, rel=0, abs=0)

    # Ценовой P&L строго нулевой: цена не двигалась, net ≡ 0.
    assert res.gross_returns.tolist() == [0.0] * _CONTROL_N

    # funding — доход ровно на удержанных барах с ненулевой ставкой.
    held_positive = _CONTROL_POSITIVE_BARS - _CONTROL_WINDOW  # 28 баров
    expected_funding = _CONTROL_RATE * held_positive
    assert -res.costs["funding"].sum() == pytest.approx(
        expected_funding, rel=1e-12, abs=0)
    # Издержки — две ноги на входе и две на выходе (4 единицы оборота).
    assert res.costs["fee"].sum() == pytest.approx(
        4.0 * _FEE_BPS * 1e-4, rel=1e-12, abs=0)
    assert res.costs["slippage"].sum() == 0.0
    assert float(res.returns.sum()) == pytest.approx(
        expected_funding - 4.0 * _FEE_BPS * 1e-4, rel=1e-12, abs=0)
    # total_return — произведение (1 + r), а не сумма: второй порядок мал.
    assert res.total_return == pytest.approx(
        expected_funding - 4.0 * _FEE_BPS * 1e-4, abs=1e-3)


# --- 3. Отрицательный режим ---------------------------------------------------


def test_negative_rate_regime_stays_flat_and_earns_nothing():
    """Отрицательная ставка — не «то же самое, но дешевле»: вне рынка.

    При положительном пороге окупаемости условие μ > порога не выполняется ни
    на одном баре, входов нет вовсе: ни издержек, ни funding.
    """
    n = 120
    bars = _bars(n, funding_rate=-1e-3)
    exp = _exp(params={"window": 24, "fee_bps": _FEE_BPS, "slippage_bps": 0.0})
    outcome = cli.run_config(_loaded_data(bars, funding_rate=pd.Series([-1e-3] * n)),
                             exp)
    res = outcome.result

    legs = _strategy(window=24, fee_bps=_FEE_BPS, slippage_bps=0.0).generate_legs(bars)
    assert legs.carry.tolist() == [0.0] * n
    assert legs.gross.tolist() == [0.0] * n
    assert res.turnover.tolist() == [0.0] * n
    assert res.costs.to_numpy().sum() == 0.0
    assert res.returns.tolist() == [0.0] * n


def test_exits_after_regime_turns_negative_within_window_memory():
    """После смены знака позиция закрывается, а не удерживается «молча».

    Правило: вне книги ровно там, где скользящее среднее ≤ порога. Ранние
    отрицательные бары окно ещё помнит (это задокументированная память, а не
    удержание режима), но как только среднее пересекло ноль, carry равен нулю
    до конца ряда — повторного входа в отрицательном режиме нет.
    """
    n = 200
    window = 72
    rate = np.concatenate([np.full(100, 1e-3), np.full(100, -1e-3)])
    bars = _bars(n, funding_rate=rate)
    strategy = _strategy(window=window, fee_bps=_FEE_BPS, slippage_bps=0.0)
    legs = strategy.generate_legs(bars)
    carry = legs.carry.to_numpy()
    ma = pd.Series(rate).rolling(window, min_periods=window).mean()
    expected = (ma > strategy.cost_recovery_threshold()).to_numpy()

    # Правило выполняется бар-в-бар: carry ≠ 0 ровно там, где среднее > порога.
    np.testing.assert_array_equal(carry != 0.0, expected)
    # Выход не позже window баров после смены режима (память окна конечна).
    held = legs.carry.shift(1).fillna(0.0).to_numpy()
    last_held = int(np.flatnonzero(held != 0.0)[-1])
    assert last_held <= 100 + window
    assert carry[last_held + 1:].tolist() == [0.0] * (n - last_held - 1)
    # Удержание не положительно никогда: это шорт перпа, а не разворот.
    assert not (carry > 0.0).any()


# --- 4. Строгая граница порога ------------------------------------------------


def test_threshold_boundary_is_strict():
    """Ровно на пороге ожидаемая прибыль нулевая — позиции нет.

    window=1, чтобы среднее совпадало со ставкой бит-в-бит и граница
    проверялась точно, а не «примерно рядом». Соседние float-значения порога
    снизу/сверху обязаны давать разные решения.
    """
    n = 6
    horizon = 200
    strategy = _strategy(window=1, horizon_bars=horizon,
                         fee_bps=5.0, slippage_bps=0.0)
    threshold = strategy.cost_recovery_threshold()
    assert threshold == pytest.approx(4.0 * 5e-4 / horizon, rel=1e-15)

    below = _strategy(window=1, horizon_bars=horizon, fee_bps=5.0,
                      slippage_bps=0.0).generate_legs(
        _bars(n, funding_rate=np.nextafter(threshold, -np.inf)))
    exact = _strategy(window=1, horizon_bars=horizon, fee_bps=5.0,
                      slippage_bps=0.0).generate_legs(
        _bars(n, funding_rate=threshold))
    above = _strategy(window=1, horizon_bars=horizon, fee_bps=5.0,
                      slippage_bps=0.0).generate_legs(
        _bars(n, funding_rate=np.nextafter(threshold, np.inf)))

    assert below.carry.tolist() == [0.0] * n
    assert exact.carry.tolist() == [0.0] * n
    assert above.carry.tolist() == [-1.0] * n
    assert above.gross.tolist() == [2.0] * n


# --- 5. Причинность -----------------------------------------------------------


def test_causality_harness_passes_on_all_three_legs():
    """Решение на баре t использует ставки до t включительно; harness судит
    все три базы (net/gross/carry)."""
    n = 300
    rng = np.random.default_rng(3)
    rate = 3e-4 * rng.standard_normal(n) + 2e-4
    bars = _bars(n, funding_rate=rate)
    strategy = _strategy(window=72)
    cuts = assert_strategy_is_causal(strategy, bars)
    assert cuts == n - 2  # сплошное покрытие: 2..n−1


# --- 6. Контракт «вошёл — держи — вышел» --------------------------------------


def test_no_flip_no_rebalance_turnover_is_only_two_units():
    """carry ходит только 0 ↔ −1, gross только 0 ↔ 2, оборот только 2.

    Осциллирующий режим даёт несколько круглых рейсов — это по-прежнему входы
    и выходы, а не ребалансировки: любое другое ненулевое значение оборота
    означало бы изменение размера книги, которого контракт не допускает.
    """
    n = 300
    window = 5
    block = 30
    rate = np.where((np.arange(n) // block) % 2 == 0, 1e-3, -1e-3)
    bars = _bars(n, funding_rate=rate)
    exp = _exp(params={"window": window, "horizon_bars": 100,
                       "fee_bps": 5.0, "slippage_bps": 0.0},
               costs={"taker_fee_bps": 5.0, "min_slippage_bps": 0.0,
                      "impact_coef": 0.0})
    outcome = cli.run_config(_loaded_data(bars, funding_rate=pd.Series(rate)), exp)
    legs = _strategy(window=window, horizon_bars=100, fee_bps=5.0,
                     slippage_bps=0.0).generate_legs(bars)
    carry = legs.carry.to_numpy()
    gross = legs.gross.to_numpy()

    assert set(np.unique(carry)) <= {0.0, -1.0}
    assert not (carry > 0.0).any()                      # никогда не лонг перпа
    np.testing.assert_array_equal(gross, 2.0 * np.abs(carry))
    assert float(np.abs(legs.net.to_numpy()).sum()) == 0.0
    # Ни одного бара, где carry сменил знак напрямую: −1 → +1 невозможен.
    assert ((np.sign(carry[1:]) * np.sign(carry[:-1])) < 0).sum() == 0

    turnover = outcome.result.turnover.to_numpy()
    nonzero = np.unique(turnover[turnover != 0.0])
    np.testing.assert_array_equal(nonzero, [2.0])
    entries, exits = _entries_exits(gross)
    assert entries >= 2 and exits >= 2                # фикстура не вырождена
    # Ни одной ребалансировки внутри книги: оборот меняется только на входе
    # и выходе, а не в барах удержания.
    assert entries == int(((turnover != 0.0) & (np.abs(carry) > 0.0)).sum())
    assert exits == int(((turnover != 0.0) & (np.abs(carry) == 0.0)).sum())


# --- 7. Закрепление ограничения magnitude-базиса S1 ---------------------------


def test_magnitude_gross_undercount_is_pinned():
    """Известное ограничение S1: |Δgross| не видит встречного движения ног.

    Оба сценария ниже торгуют реальный ноционал, но оба заряжают ноль:

    * ребалансировка книги без изменения net и gross (спот 1.5 → 1.2, перп
      −0.5 → −0.2: продано 0.3 спота, куплено 0.3 перпа, оборот 0.6);
    * прямой разворот carry −1 → +1: закрыть 2 единицы и открыть 2 — 4
      единицы оборота.

    Это НЕ свойство стратегии S2 (она никогда не ребалансирует и не
    разворачивает carry), а задокументированное ограничение модели движка.
    Тест намеренно фиксирует нулевой заряд: если magnitude-базис когда-нибудь
    «починят» (например, оборот начнут считать по ногам), он обязан упасть —
    иначе смена модели издержек пройдёт незамеченной, а числа S2 перестанут
    быть сопоставимыми.
    """
    # held[t] = target[t−1], поэтому вход (оборот 2) заряжается на баре 3, а
    # интересующее событие — последний бар ряда, где книга уже стоит.
    n = 5
    bars = _bars(n, close=[100.0] * n)

    # Встречное движение ног между барами 3 и 4: net=1 и gross=2 стоят, но
    # 0.6 ноционала торгуется.
    res_rebalance = run_backtest(
        bars, pd.Series([0.0, 0.0, 1.0, 1.0, 1.0]), _TWO_LEG_COST,
        gross_position=pd.Series([0.0, 0.0, 2.0, 2.0, 2.0]),
        carry_position=pd.Series([0.0, 0.0, -1.0, -1.0, -1.0]),
    )
    assert res_rebalance.turnover.tolist() == [0.0, 0.0, 0.0, 2.0, 0.0]
    # Бар 4: книга изменилась, а заряд ноль — это и есть закрепляемый дефект.
    assert res_rebalance.costs["fee"].iloc[4] == 0.0
    assert res_rebalance.costs["slippage"].iloc[4] == 0.0

    # Прямой разворот carry −1 → +1: закрыть 2 единицы и открыть 2 — 4 единицы
    # реального оборота, |Δgross| = 0, заряд ноль (событие видно на баре 4).
    res_flip = run_backtest(
        bars, pd.Series([0.0] * n), _TWO_LEG_COST,
        gross_position=pd.Series([0.0, 0.0, 2.0, 2.0, 2.0]),
        carry_position=pd.Series([0.0, 0.0, -1.0, 1.0, 1.0]),
    )
    assert res_flip.turnover.tolist() == [0.0, 0.0, 0.0, 2.0, 0.0]
    assert res_flip.costs["fee"].iloc[4] == 0.0
    assert res_flip.costs["slippage"].iloc[4] == 0.0


# --- 8. Отсутствие funding — отказ, а не молчаливый простой -------------------


def test_run_config_refuses_without_funding_data():
    """Без ставок решение невозможно: стратегия не имеет права молча стоять
    вне рынка, а вердикт — описывать отсутствие данных как отсутствие сигнала."""
    bars = _bars(40)  # колонки funding нет — как при недоступном файле
    with pytest.raises(cli.FundingRequiredError, match="funding"):
        cli.run_config(_loaded_data(bars, funding_rate=None), _exp())


def test_cli_reports_missing_funding_without_traceback(tmp_path, capsys):
    """Полный путь CLI: нет файла funding у funding-стратегии → чистый EXIT_ERROR."""
    from alpha_lab.cli import main
    from alpha_lab.data.schema import normalize_bars
    from alpha_lab.data.store import write_bars

    root = tmp_path / "data"
    n = 3000
    rng = np.random.default_rng(5)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.002, n)))
    open_ = np.concatenate(([close[0]], close[:-1]))
    df = normalize_bars(pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": open_, "high": np.maximum(open_, close) * 1.0005,
        "low": np.minimum(open_, close) * 0.9995, "close": close,
        "volume": 1e5, "quote_volume": 1e8, "trades": 500,
        "taker_buy_volume": 5e4,
    }))
    write_bars(df, root, "BTCUSDT", "1m")

    universe = tmp_path / "universe.yaml"
    universe.write_text(
        "market: futures-um\nstart: '2024-01-01'\nend: '2024-01-05'\n"
        "symbols: [BTCUSDT]\n", encoding="utf-8")
    exp = tmp_path / "exp.yaml"
    exp.write_text(
        "name: funding_no_data\nstrategy: funding_harvest\ntimeframe: 1h\n"
        "start: '2024-01-01'\nend: '2024-01-05'\n"
        "params: {window: 12, horizon_bars: 720}\n"
        "costs: {taker_fee_bps: 5.0}\n"
        "validation: {min_trades: 1}\n", encoding="utf-8")

    code = main([
        "validate", "--config", str(exp), "--universe", str(universe),
        "--data-root", str(root), "--out", str(tmp_path / "out"),
        "--journal", str(tmp_path / "trials.jsonl"),
    ])
    captured = capsys.readouterr()
    assert code == 2
    assert "funding" in captured.err.lower()
    assert "остановлен" in captured.err
    assert "Traceback" not in captured.err


# --- 9. Порог напрямую и предел «держать всегда» ------------------------------


def test_threshold_override_replaces_decision_threshold_only():
    """threshold_rate задаёт порог решения напрямую; априорный порог цел.

    cost_recovery_threshold() остаётся арифметикой издержек (4·one_way/H) —
    переопределяется только точка принятия решения, а не модель окупаемости.
    """
    n = 200
    window = 24
    rate = np.concatenate([np.full(100, 1e-3), np.full(100, -1e-3)])
    bars = _bars(n, funding_rate=rate)
    override = 1e-4
    strategy = _strategy(window=window, fee_bps=_FEE_BPS, slippage_bps=0.0,
                         threshold_rate=override)

    assert strategy.cost_recovery_threshold() == pytest.approx(
        4.0 * _FEE_BPS * 1e-4 / DEFAULTS["horizon_bars"], rel=1e-15)
    assert strategy.entry_threshold() == override
    legs = strategy.generate_legs(bars)
    ma = pd.Series(rate).rolling(window, min_periods=window).mean()
    np.testing.assert_array_equal(legs.carry.to_numpy() != 0.0,
                                  (ma > override).to_numpy())
    # Переопределение действительно меняет решение: порог выше априорного —
    # книга выходит из режима раньше, чем по правилу S2.
    default = _strategy(window=window, fee_bps=_FEE_BPS, slippage_bps=0.0)
    assert 0 < int((legs.carry != 0.0).sum()) < int(
        (default.generate_legs(bars).carry != 0.0).sum())


def test_always_hold_limit_is_exact_and_not_reachable_by_finite_params():
    """Предел −∞ — единственный способ держать книгу всегда; конечные
    параметры S2 его не достигают.

    Порог окупаемости ограничен снизу нулём: 4·one_way/H ≥ 0 при любом H и
    неотрицательных издержках, а правило требует μ > порога. При неположительном
    среднем ставки никакое конечное H (и даже нулевые издержки) не удерживает
    книгу — это измеренное свойство правила, а не деталь реализации. Отсюда
    явный предел threshold_rate = −∞: условие выполнено для любой конечной
    ставки, окно не читается, прогрева нет.
    """
    n = 50
    rate = np.full(n, -1e-3)
    bars = _bars(n, funding_rate=rate)

    # Конечный предел: H → ∞ даёт порог → 0, но не always-hold: книга стоит.
    finite = _strategy(window=1, horizon_bars=10 ** 12, fee_bps=0.0,
                       slippage_bps=0.0)
    assert finite.entry_threshold() == 0.0
    assert finite.generate_legs(bars).carry.tolist() == [0.0] * n

    # Явный предел: точная постоянная книга на любых ставках, включая
    # заведомо отрицательные, и без прогрева окна.
    limit = _strategy(threshold_rate=-np.inf)
    assert limit.history_bars == 1
    legs = limit.generate_legs(bars)
    assert legs.carry.tolist() == [-1.0] * n
    assert legs.gross.tolist() == [2.0] * n
    assert legs.net.tolist() == [0.0] * n


def test_always_hold_runs_through_cli_as_constant_book():
    """Предел проходит рабочий путь CLI и даёт ровно книгу S1: net ≡ 0,
    carry ≡ −1, один вход и ни одного выхода.

    Доход — вся сумма ставок на удержанных барах, издержки — ровно два
    ноционала входа (одна нога спота и одна нога перпа), без ребалансировок.
    Синтетика нарочно знакопеременная: правило S2 здесь вышло бы из книги, а
    предел держит.
    """
    n = 40
    rate = np.linspace(-1e-3, 1e-3, n)
    bars = _bars(n, funding_rate=rate)
    exp = _exp(params={"threshold_rate": -np.inf})
    outcome = cli.run_config(
        _loaded_data(bars, funding_rate=pd.Series(rate)), exp)
    res = outcome.result

    assert res.positions.tolist() == [0.0] * n            # ценовой экспозиции нет
    assert res.gross_positions.tolist() == [0.0] + [2.0] * (n - 1)
    assert res.turnover.tolist() == [0.0, 2.0] + [0.0] * (n - 2)
    # funding начисляется на удерживаемый carry, доход = сумма ставок баров 1..n−1
    assert -res.costs["funding"].sum() == pytest.approx(rate[1:].sum(), rel=1e-12)
    assert res.costs["fee"].sum() == pytest.approx(
        2.0 * _FEE_BPS * 1e-4, rel=1e-12)
    assert res.costs["slippage"].sum() == 0.0
    assert float(res.returns.sum()) == pytest.approx(
        rate[1:].sum() - 2.0 * _FEE_BPS * 1e-4, rel=1e-12)


def test_always_hold_limit_is_causal():
    """Решение предела не зависит от данных: harness проходит с history_bars=1."""
    n = 120
    bars = _bars(n, funding_rate=np.linspace(-1e-3, 1e-3, n))
    strategy = _strategy(threshold_rate=-np.inf)
    assert assert_strategy_is_causal(strategy, bars) == n - 2


def test_positive_infinity_threshold_means_never_hold():
    """+∞ — второй предел: ни одна конечная ставка порога не превышает."""
    n = 30
    bars = _bars(n, funding_rate=1.0)
    legs = _strategy(window=1, threshold_rate=np.inf).generate_legs(bars)
    assert legs.carry.tolist() == [0.0] * n
    assert legs.gross.tolist() == [0.0] * n


# --- 10. Фиксированный ноционал плеча: кросс-секционный портфель (E2) --------
#
# Кросс-секционный портфель — сумма независимых плеч равного размера. Чтобы
# вход/выход одного символа не перекладывал капитал других, ноционал каждого
# плеча фиксирован: целевые значения могут быть только 0 и −w (carry) при
# gross 0 и 2w. Изменение размера внутри книги движок оценить не может
# (magnitude-базис S1), поэтому правило обязано его не создавать — эти тесты
# закрепляют инвариант, на котором стоит весь портфельный замер E2.


def test_notional_scales_book_legs_without_changing_decision():
    """notional масштабирует carry/gross, но не решение: 0 ↔ −w, gross 2w."""
    n = 200
    window = 24
    rate = np.concatenate([np.full(100, 1e-3), np.full(100, -1e-3)])
    bars = _bars(n, funding_rate=rate)
    base = _strategy(window=window, fee_bps=_FEE_BPS, slippage_bps=0.0)
    scaled = _strategy(window=window, fee_bps=_FEE_BPS, slippage_bps=0.0,
                       notional=0.25)
    legs_base = base.generate_legs(bars)
    legs_scaled = scaled.generate_legs(bars)
    in_book = legs_base.carry.to_numpy() != 0.0

    assert not legs_scaled.net.to_numpy().any()
    np.testing.assert_array_equal(legs_scaled.carry.to_numpy() != 0.0, in_book)
    np.testing.assert_allclose(legs_scaled.carry.to_numpy(),
                               -0.25 * in_book)
    np.testing.assert_allclose(legs_scaled.gross.to_numpy(), 0.5 * in_book)
    assert not (legs_scaled.carry > 0.0).any()
    # Порог окупаемости — ставка на бар: и доход, и издержки масштабируются
    # ноционалом, поэтому порог от него не зависит (иначе размер плеча менял бы
    # само правило входа, чего априорная формулировка не допускает).
    assert scaled.cost_recovery_threshold() == pytest.approx(
        base.cost_recovery_threshold(), rel=0, abs=0)


def test_notional_is_validated():
    """Ноционал — конечное положительное число; bool/finf/NaN/ноль — отказ."""
    for bad in (0.0, -0.5, float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="notional"):
            _strategy(notional=bad)


def test_notional_control_arithmetic_scales_exactly():
    """Синтетический контроль w=0.25: funding = w·Σ ставок, издержки = 4w·fee."""
    n = 200
    w = 0.25
    rate = np.concatenate([np.full(100, 1e-3), np.zeros(100)])
    bars = _bars(n)                      # ставку прикрепляет путь CLI
    exp = _exp(params={"window": 72, "horizon_bars": 720,
                       "fee_bps": _FEE_BPS, "slippage_bps": 0.0,
                       "notional": w})
    outcome = cli.run_config(
        _loaded_data(bars, funding_rate=pd.Series(rate)), exp)
    res = outcome.result

    assert res.gross_positions.tolist() == [0.0] * 72 + [2 * w] * 100 + [0.0] * 28
    expected_funding = w * 1e-3 * 28     # ненулевые ставки на удержанных барах
    assert -res.costs["funding"].sum() == pytest.approx(
        expected_funding, rel=1e-12, abs=0)
    assert res.costs["fee"].sum() == pytest.approx(
        4 * w * _FEE_BPS * 1e-4, rel=1e-12, abs=0)
    assert float(res.returns.sum()) == pytest.approx(
        expected_funding - 4 * w * _FEE_BPS * 1e-4, rel=1e-12, abs=0)


def test_fixed_notional_sleeves_do_not_resize_on_membership_change():
    """Размер книги фиксирован: вход/выход соседнего плеча её не меняет.

    Плечо A держит книгу всё время (ставка положительна), плечо B входит и
    выходит блоками. По удержанным рядам обоих видно: gross ∈ {0, 2w} и любой
    ненулевой оборот равен ровно 2w — значит, смена состава портфеля нигде не
    меняет размер уже открытой книги. Движок ценит 0 ↔ 2w точно; изменение
    размера внутри книги он оценить не может (ограничение magnitude-базиса
    S1), поэтому правило обязано его не создавать.
    """
    n = 400
    w = 0.5
    window = 5
    block = 30
    rate_a = np.full(n, 1e-3)
    rate_b = np.where((np.arange(n) // block) % 2 == 0, 1e-3, -1e-3)
    params = {"window": window, "horizon_bars": 100, "fee_bps": 5.0,
              "slippage_bps": 0.0, "notional": w}
    costs = {"taker_fee_bps": 5.0, "min_slippage_bps": 0.0, "impact_coef": 0.0}
    results = {}
    for label, rate in (("A", rate_a), ("B", rate_b)):
        outcome = cli.run_config(
            _loaded_data(_bars(n), funding_rate=pd.Series(rate)),
            _exp(params=params, costs=costs))
        res = outcome.result
        held = res.gross_positions.to_numpy()
        turnover = res.turnover.to_numpy()

        assert res.positions.to_numpy().tolist() == [0.0] * n
        assert set(np.unique(held)) <= {0.0, 2 * w}
        nz = np.unique(turnover[turnover != 0.0])
        np.testing.assert_allclose(nz, [2 * w])
        # funding начисляется ровно на фиксированный carry: 0 или −w·rate.
        funding = res.costs["funding"].to_numpy()
        in_book = held > 0.0
        np.testing.assert_allclose(funding[in_book], -w * rate[in_book],
                                   rtol=0, atol=1e-18)
        assert np.allclose(funding[~in_book], 0.0, rtol=0, atol=0)
        # Оборот стоит только на входах и выходах: ребалансировок нет.
        entries, exits = _entries_exits(held)
        assert entries + exits == int((turnover != 0.0).sum())
        results[label] = (held, res.returns.to_numpy())

    held_a, returns_a = results["A"]
    entries_a, exits_a = _entries_exits(held_a)
    assert (entries_a, exits_a) == (1, 0)        # A не выходит и не мигает
    assert 2 * w in np.unique(held_a)
    held_b, returns_b = results["B"]
    entries_b, exits_b = _entries_exits(held_b)
    assert entries_b >= 2 and exits_b >= 2       # фикстура B не вырождена
    # Портфельная единица — равновесное среднее плеч равного бюджета, а не их
    # сумма: доходность портфеля на весь капитал = Σ P&L_i / (N·C/N).
    portfolio = 0.5 * (returns_a + returns_b)
    assert portfolio.shape == returns_a.shape
    assert np.isfinite(portfolio).all()
