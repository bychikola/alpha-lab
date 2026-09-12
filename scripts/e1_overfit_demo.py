"""E1: демонстрация подгонки под историю на эпохах «улучшения» модели.

Задача демонстрации — воспроизвести на реальных данных полигона процесс
«улучшать модель с каждой новой эпохой тестов, пока она не даст 50 % годовых
на всех монетах» и показать, чем этот процесс является на самом деле:
перебором, который максимизирует шум обучающего отрезка.

ПРОТОКОЛ ЗАФИКСИРОВАН ДО ПЕРВОГО ПРОГОНА. Ни один из пунктов ниже не
менялся по результатам; правила генерации кандидатов и метрика отбора
объявлены заранее и не подстраиваются.

1. Разбиение по времени (фиксировано):
   * обучение  — 2022-01-01 .. 2023-12-31;
   * вне выборки (OOS) — 2024-01-01 .. 2025-12-31.
   Прогоны на этих отрезках независимы: strategy.generate вызывается на баре
   каждого отрезка отдельно, окна и рекурсии не пересекают границу.

2. Панель: BTCUSDT, ETHUSDT, SOLUSDT, BNBUSDT, LINKUSDT, таймфрейм 1h.
   Все пять пар имеют полное покрытие 1m за 2022–2025 и ряд funding за тот же
   период. Модель обязана работать «на всех монетах», поэтому кандидат
   оценивается средним по панели, а не лучшим символом.

3. Эпоха: 10 эпох по 10 наборов параметров.
   * эпоха 1 — дефолт mean_reversion (mr_base) плюс равномерные случайные
     выборки из объявленного дискретного пространства;
   * эпохи 2..10 — инкамбент (лучший на обучении по прошлым эпохам) плюс
     8 локальных возмущений «±1 шаг по упорядоченному списку значений,
     каждый параметр независимо с вероятностью 0.5» плюс 1 равномерная
     случайная выборка (разведка). Локальные возмущения центрированы на
     инкамбенте, сам инкамбент входит в батч — правило «нести лучшее вперёд»
     из условия задачи.
   * Генератор детерминирован: random.Random(SEED + номер эпохи), SEED
     фиксирован. Дубликаты (внутри батча и против всех прошлых эпох)
     отбрасываются; лимит попыток на кандидата — 500.
   * Пространство параметров (только mean_reversion; funding_harvest не
     перебирается — бюджет прогона ограничен):
     window {10,15,20,30,40,60}, k {1.0,1.25,1.5,1.75,2.0,2.5,3.0},
     atr_len {7,10,14,20,28}, sl_atr {1.0,1.5,2.0,3.0},
     tp_atr {3.0,4.0,6.0,9.0}, max_bars {100,250,500,1000};
     use_hl_filter=false (фильтр полужизни не перебирается: он резко дороже
     генерации и не является предметом демонстрации).

4. Метрика отбора (фиксирована): средний годовой Sharpe на обучающем отрезке
   по панели (среднее арифметическое per-symbol annualized Sharpe,
   periods_per_year = 8760). Максимум по батчу становится инкамбентом.
   Инкамбент входит в каждый следующий батч, поэтому лучший обучающий
   показатель не может ухудшиться — это и есть видимая «эпоха улучшений».

5. Годовая доходность (для отчёта, не для отбора):
   ann = equity_end ** (ppy / n_bars) - 1 — геометрическая аннуализация кривой
   эквити отрезка (эквити стартует с 1.0). Средняя по панели — среднее
   арифметическое per-symbol ann.

6. Число испытаний (n_trials) для честного DSR — общее число реально
   выполненных прогонов движка за все эпохи: каждая уникальная тройка
   (параметры, символ, отрезок) считается один раз, повторный запрос берётся
   из кэша и второй попыткой не является. В это число входят и OOS-прогоны
   инкамбентов: они тоже запускались, и скрывать их от поправки нельзя.
   Итоговый вердикт продакшн-валидатора выносится с этим n_trials.

7. Издержки — как в configs/experiments/mr_base.yaml (тейкер Binance VIP0,
   funding из исторического ряда, impact_coef=0.1). Финальный вердикт — по
   якорному символу BTCUSDT (выбран до прогона), OOS-период первичен,
   обучающий вердикт приводится рядом и несёт PBO-матрицу перебора
   (CSCV по матрице доходностей всех уникальных по ряду конфигураций эпох;
   полностью совпавшие ряды — одна гипотеза и в матрицу не дублируются).

Запуск (полный протокол):
    uv run python scripts/e1_overfit_demo.py --out reports/e1_demo/results.json
Смоук-проверка механики (не протокол, помечается в JSON):
    uv run python scripts/e1_overfit_demo.py --smoke
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab import cli
from alpha_lab.config import Experiment, load_universe
from alpha_lab.data.query import data_version
from alpha_lab.strategies.mean_reversion import DEFAULTS as MR_DEFAULTS
from alpha_lab.validation.metrics import sharpe_ratio
from alpha_lab.validation.significance import pbo_cscv

# --- Зафиксированный протокол -------------------------------------------------

PROTOCOL_VERSION = "e1-2026-09-11"

PANEL = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "LINKUSDT")
ANCHOR = "BTCUSDT"
TIMEFRAME = "1h"
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31"
OOS_START, OOS_END = "2024-01-01", "2025-12-31"
N_EPOCHS = 10
BATCH_SIZE = 10
SEED = 20260911
MAX_DRAW_ATTEMPTS = 500

# Объявленное дискретное пространство поиска (порядок значений — шаг локального
# возмущения «±1 позиция»).
PARAM_GRID: dict[str, tuple[Any, ...]] = {
    "window": (10, 15, 20, 30, 40, 60),
    "k": (1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0),
    "atr_len": (7, 10, 14, 20, 28),
    "sl_atr": (1.0, 1.5, 2.0, 3.0),
    "tp_atr": (3.0, 4.0, 6.0, 9.0),
    "max_bars": (100, 250, 500, 1000),
}

# Издержки mr_base: тейкерские исполнения, funding из данных.
COSTS = {
    "taker_fee_bps": 5.0,
    "maker_fee_bps": 2.0,
    "maker_share": 0.0,
    "impact_coef": 0.1,
    "min_slippage_bps": 0.5,
}

PERIODS = {"train": (TRAIN_START, TRAIN_END), "oos": (OOS_START, OOS_END)}

BASE_PARAMS = {name: MR_DEFAULTS[name] for name in PARAM_GRID}


def canonical(params: dict[str, Any]) -> str:
    """Канонический ключ набора параметров (порядок ключей не важен)."""
    return json.dumps(params, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


@dataclass(frozen=True)
class RunMetrics:
    """Метрики одного прогона (параметры × символ × отрезок)."""

    symbol: str
    period: str
    sharpe: float
    annual_return: float
    total_return: float
    trades: int
    max_dd: float
    cap_hits: int


@dataclass(frozen=True)
class EpochRow:
    """Строка таблицы эпох."""

    epoch: int
    new_param_sets: int
    new_candidates: int
    new_runs: int
    cumulative_candidates: int
    cumulative_runs: int
    incumbent: dict[str, Any]
    train_sharpe: float
    train_annual_return: float
    oos_sharpe: float
    oos_annual_return: float
    train_annual_by_symbol: dict[str, float]
    oos_annual_by_symbol: dict[str, float]


class DemoEvaluator:
    """Кэш прогонов, бары один раз на (символ, отрезок), счётчик попыток."""

    def __init__(self, root: Path, panel: tuple[str, ...]):
        self.root = root
        self.panel = panel
        self.data: dict[tuple[str, str], cli.LoadedData] = {}
        self.metrics: dict[tuple[str, str, str], RunMetrics] = {}
        self.outcomes: dict[tuple[str, str, str], cli.RunOutcome] = {}
        # Ряды доходностей якоря на обучении — для PBO-матрицы перебора.
        self.anchor_train_returns: dict[str, np.ndarray] = {}
        self.total_runs = 0
        self.runs_by_period = {"train": 0, "oos": 0}
        self.run_seconds: list[float] = []

    def load(self, symbol: str, period: str) -> cli.LoadedData:
        key = (symbol, period)
        if key not in self.data:
            start, end = PERIODS[period]
            self.data[key] = cli.prepare_data(
                self.root, symbol, TIMEFRAME, start, end)
        return self.data[key]

    def preload(self) -> None:
        """Грузит все бары заранее: загрузка — разовая плата на пару."""
        for period in PERIODS:
            for symbol in self.panel:
                self.load(symbol, period)

    def evaluate(self, params: dict[str, Any], symbol: str,
                 period: str) -> RunMetrics:
        key = (canonical(params), symbol, period)
        if key in self.metrics:
            return self.metrics[key]
        data = self.load(symbol, period)
        start, end = PERIODS[period]
        exp = Experiment(
            name="e1_candidate", strategy="mean_reversion", params=dict(params),
            timeframe=TIMEFRAME, start=start, end=end, costs=dict(COSTS),
            validation={},
        )
        started = time.perf_counter()
        outcome = cli.run_config(data, exp)
        self.run_seconds.append(time.perf_counter() - started)
        result = outcome.result
        n = len(result.returns)
        equity_end = float(result.equity.iloc[-1])
        annual = (equity_end ** (data.ppy / n) - 1.0) if equity_end > 0 else -1.0
        metrics = RunMetrics(
            symbol=symbol, period=period,
            sharpe=sharpe_ratio(result.returns, data.ppy),
            annual_return=float(annual),
            total_return=float(result.total_return),
            trades=int(len(outcome.trades)),
            max_dd=float(result.max_drawdown),
            cap_hits=int(result.cap_hits),
        )
        self.metrics[key] = metrics
        if symbol == ANCHOR:
            # Исход якоря хранится для финального вердикта: продакшн-validate
            # получает уже посчитанный прогон, а не запускает второй (иначе
            # n_trials занизил бы число реально сделанных попыток).
            self.outcomes[key] = outcome
        if period == "train" and symbol == ANCHOR:
            self.anchor_train_returns[canonical(params)] = np.asarray(
                result.returns, dtype="float64")
        self.total_runs += 1
        self.runs_by_period[period] += 1
        return metrics

    def panel_metrics(self, params: dict[str, Any],
              period: str) -> tuple[float, float, dict[str, RunMetrics]]:
        """Средние по панели Sharpe и годовой доходности плюс per-symbol."""
        runs = {s: self.evaluate(params, s, period) for s in self.panel}
        sharpe = float(np.mean([m.sharpe for m in runs.values()]))
        annual = float(np.mean([m.annual_return for m in runs.values()]))
        return sharpe, annual, runs


def uniform_draw(rng: random.Random) -> dict[str, Any]:
    return {name: rng.choice(values) for name, values in PARAM_GRID.items()}


def perturb(rng: random.Random, current: dict[str, Any]) -> dict[str, Any]:
    """Локальное возмущение: ±1 позиция в списке значений параметра."""
    out = dict(current)
    moved = False
    for name, values in PARAM_GRID.items():
        if rng.random() < 0.5:
            i = values.index(out[name])
            j = min(max(i + rng.choice((-1, 1)), 0), len(values) - 1)
            if j != i:
                out[name] = values[j]
                moved = True
    if not moved:
        name = rng.choice(sorted(PARAM_GRID))
        values = PARAM_GRID[name]
        i = values.index(out[name])
        j = min(max(i + rng.choice((-1, 1)), 0), len(values) - 1)
        out[name] = values[j]
    return out


def propose_epoch(epoch: int, incumbent: dict[str, Any] | None,
                  seen: set[str], batch_size: int) -> list[dict[str, Any]]:
    """Батч эпохи: детерминированный генератор, без повторов с прошлым.

    Инкамбент разрешается повторно (allow_seen=True): он уже оценивался, но
    обязан участвовать в каждом батче — иначе «нести лучшее вперёд» теряет
    смысл, а лучший обучающий показатель мог бы ухудшиться. Остальные
    кандидаты обязаны быть новыми: повтор прошлой эпохи не несёт информации.
    """
    rng = random.Random(SEED + epoch)
    batch: list[dict[str, Any]] = []
    attempts = 0

    def try_add(params: dict[str, Any], *, allow_seen: bool = False) -> bool:
        key = canonical(params)
        if any(canonical(b) == key for b in batch):
            return False
        if not allow_seen and key in seen:
            return False
        batch.append(params)
        return True

    if epoch == 1 or incumbent is None:
        try_add(dict(BASE_PARAMS), allow_seen=True)
        while len(batch) < batch_size and attempts < MAX_DRAW_ATTEMPTS:
            attempts += 1
            try_add(uniform_draw(rng))
    else:
        try_add(dict(incumbent), allow_seen=True)
        while len(batch) < batch_size and attempts < MAX_DRAW_ATTEMPTS:
            attempts += 1
            params = (uniform_draw(rng) if len(batch) == 1
                      else perturb(rng, incumbent))
            try_add(params)
    return batch


def build_pbo_matrix(evaluator: DemoEvaluator) -> tuple[pd.DataFrame | None, str]:
    """PBO-матрица обучения якоря: уникальные по ряду конфигурации эпох.

    Полностью совпавшие ряды доходностей — одна и та же гипотеза на данных,
    и CSCV по дубликатам вырожден (build_returns_matrix отвергает дубликаты).
    Поэтому дубликаты по содержимому ряда отбрасываются; это объявлено в
    протоколе заранее, а не выбрано по результату.
    """
    data = evaluator.data[(ANCHOR, "train")]
    unique: dict[str, np.ndarray] = {}
    for key, values in evaluator.anchor_train_returns.items():
        digest = values.tobytes()
        if digest not in unique:
            unique[digest] = values
    if len(unique) < 2:
        return None, ("PBO не оценён: уникальных по ряду конфигураций на "
                      "обучении меньше двух")
    columns = {
        f"cfg_{i}": pd.Series(values).set_axis(data.ts_index)
        for i, values in enumerate(unique.values())
    }
    return cli.build_returns_matrix(columns), ""


def run_demo(root: Path, panel: tuple[str, ...], epochs: int, batch_size: int,
             protocol: bool) -> dict[str, Any]:
    started = time.perf_counter()
    evaluator = DemoEvaluator(root, panel)
    print(f"Загрузка баров: {len(panel)} символов × {len(PERIODS)} отрезков "
          f"({TIMEFRAME})...", flush=True)
    evaluator.preload()
    print(f"Загрузка завершена за {time.perf_counter() - started:.1f} с",
          flush=True)

    rows: list[EpochRow] = []
    seen: set[str] = set()
    incumbent: dict[str, Any] | None = None
    incumbent_metrics: dict[str, dict[str, RunMetrics]] | None = None

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        runs_before = dict(evaluator.runs_by_period)
        batch = propose_epoch(epoch, incumbent, seen, batch_size)
        new_params = [p for p in batch if canonical(p) not in seen]
        for p in batch:
            seen.add(canonical(p))

        scored: list[tuple[float, str, dict[str, Any]]] = []
        for params in batch:
            train_sharpe, _, _ = evaluator.panel_metrics(params, "train")
            scored.append((train_sharpe, canonical(params), params))
        # Тай-брейк по ключу: выбор детерминирован при равных метриках.
        best_sharpe, _, best_params = max(scored, key=lambda t: (t[0], t[1]))

        # Метрики лучшего батча — после отбора, не влияют на него.
        tr_sharpe, tr_annual, tr_runs = evaluator.panel_metrics(best_params, "train")
        oos_sharpe, oos_annual, oos_runs = evaluator.panel_metrics(best_params, "oos")
        incumbent = best_params
        incumbent_metrics = {"train": tr_runs, "oos": oos_runs}

        row = EpochRow(
            epoch=epoch,
            new_param_sets=len(new_params),
            new_candidates=(evaluator.runs_by_period["train"]
                            - runs_before["train"]),
            new_runs=evaluator.total_runs - sum(runs_before.values()),
            cumulative_candidates=evaluator.runs_by_period["train"],
            cumulative_runs=evaluator.total_runs,
            incumbent=dict(best_params),
            train_sharpe=tr_sharpe,
            train_annual_return=tr_annual,
            oos_sharpe=oos_sharpe,
            oos_annual_return=oos_annual,
            train_annual_by_symbol={
                s: tr_runs[s].annual_return for s in panel},
            oos_annual_by_symbol={
                s: oos_runs[s].annual_return for s in panel},
        )
        rows.append(row)
        print(
            f"эпоха {epoch:>2}/{epochs}: новых наборов {row.new_param_sets}, "
            f"кандидатов (обуч.) {row.new_candidates}, "
            f"всего прогонов {row.cumulative_runs} | "
            f"train Sharpe {tr_sharpe:+.2f}, train ann "
            f"{tr_annual:+.1%} | OOS Sharpe {oos_sharpe:+.2f}, OOS ann "
            f"{oos_annual:+.1%} | {canonical(best_params)} "
            f"({time.perf_counter() - epoch_started:.1f} с)",
            flush=True,
        )

    assert incumbent is not None and incumbent_metrics is not None

    # --- Вердикт продакшн-валидатора по финальному инкамбенту ---------------
    n_trials = evaluator.total_runs
    exp = Experiment(
        name="e1_incumbent", strategy="mean_reversion", params=dict(incumbent),
        timeframe=TIMEFRAME, start=TRAIN_START, end=OOS_END, costs=dict(COSTS),
        validation={},
    )
    universe_path = Path("configs/universe.yaml")
    try:
        universe = load_universe(universe_path)
    except (OSError, ValueError):
        universe = None
    if universe is not None:
        dv = data_version(root)
        exp_id = cli.experiment_id(
            cli._experiment_payload(exp, ANCHOR, universe), dv, cli.git_hash())
    else:
        exp_id = "no-universe"

    matrix, matrix_note = build_pbo_matrix(evaluator)
    pbo_value = pbo_cscv(matrix) if matrix is not None else None

    train_outcome = evaluator.outcomes.get((canonical(incumbent), ANCHOR, "train"))
    train_data = evaluator.data[(ANCHOR, "train")]
    train_verdict = None
    if train_outcome is not None:
        train_verdict = cli.validate_config(
            train_outcome, train_data, experiment_id=exp_id, n_trials=n_trials,
            returns_matrix=matrix, pbo_value=pbo_value,
            warnings=train_data.data_warnings)

    # OOS-прогон якоря берём из кэша (инкамбент уже оценивался на OOS): это
    # тот же прогон, что учтён в n_trials, а не новая попытка.
    oos_data = evaluator.data[(ANCHOR, "oos")]
    oos_outcome = evaluator.outcomes[(canonical(incumbent), ANCHOR, "oos")]
    oos_verdict = cli.validate_config(
        oos_outcome, oos_data, experiment_id=exp_id, n_trials=n_trials,
        warnings=oos_data.data_warnings)

    # Максимум годовой доходности обучения по всем прогонам поиска.
    train_runs = [m for m in evaluator.metrics.values() if m.period == "train"]
    best_run = max(train_runs, key=lambda m: (m.annual_return, m.symbol))

    def verdict_payload(v) -> dict[str, Any] | None:
        if v is None:
            return None
        return {
            "strategy": v.strategy_name, "experiment_id": v.experiment_id,
            "sharpe": v.sharpe, "dsr": v.dsr, "p_value": v.p_value,
            "pbo": None if not np.isfinite(v.pbo) else v.pbo,
            "trades": v.trades, "alive": v.alive,
            "total_return": v.total_return, "max_dd": v.max_dd,
            "n_configs_tried": v.n_configs_tried,
            "reasons": list(v.reasons), "warnings": list(v.warnings),
            "inapplicable": list(v.inapplicable),
        }

    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol": protocol,
        "panel": list(panel),
        "anchor": ANCHOR,
        "timeframe": TIMEFRAME,
        "train": [TRAIN_START, TRAIN_END],
        "oos": [OOS_START, OOS_END],
        "epochs": epochs,
        "batch_size": batch_size,
        "selection_metric": ("средний годовой Sharpe обучения по панели "
                             "(mean of per-symbol annualized Sharpe)"),
        "annual_return_definition": ("equity_end ** (ppy / n_bars) - 1, "
                                     "среднее по панели — mean per-symbol"),
        "n_trials_rule": ("число реально выполненных прогонов движка "
                          "(уникальные пары параметры×символ×отрезок)"),
        "n_trials": n_trials,
        "runs_by_period": dict(evaluator.runs_by_period),
        "candidates_evaluated": evaluator.runs_by_period["train"],
        "param_sets_evaluated": len(
            {key[0] for key in evaluator.metrics}),
        "wall_clock_seconds": None,
        "mean_config_seconds": (float(np.mean(evaluator.run_seconds))
                                if evaluator.run_seconds else None),
        "rows": [asdict(row) for row in rows],
        "max_train_annual_run": {
            "symbol": best_run.symbol, "period": best_run.period,
            "annual_return": best_run.annual_return,
            "sharpe": best_run.sharpe,
        },
        "final_incumbent": {
            "params": incumbent,
            "train_sharpe": rows[-1].train_sharpe,
            "train_annual_return": rows[-1].train_annual_return,
            "oos_sharpe": rows[-1].oos_sharpe,
            "oos_annual_return": rows[-1].oos_annual_return,
            "train_by_symbol": {
                s: asdict(incumbent_metrics["train"][s]) for s in panel},
            "oos_by_symbol": {
                s: asdict(incumbent_metrics["oos"][s]) for s in panel},
        },
        "pbo_matrix_note": matrix_note,
        "pbo_matrix_shape": (list(matrix.shape) if matrix is not None else None),
        "pbo_value": pbo_value,
        "verdict_train": verdict_payload(train_verdict),
        "verdict_oos": verdict_payload(oos_verdict),
    }
    summary["wall_clock_seconds"] = time.perf_counter() - started
    return summary


def print_table(summary: dict[str, Any]) -> None:
    print("\nэпоха | кандидатов за эпоху | лучший train Sharpe | его OOS Sharpe "
          "| накоплено кандидатов (всего прогонов)")
    for row in summary["rows"]:
        print(f"{row['epoch']:>5} | {row['new_candidates']:>19} | "
              f"{row['train_sharpe']:>19.2f} | {row['oos_sharpe']:>14.2f} | "
              f"{row['cumulative_candidates']:>10} "
              f"({row['cumulative_runs']})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="E1: демонстрация подгонки на эпохах улучшения")
    parser.add_argument("--out", default="reports/e1_demo/results.json")
    parser.add_argument("--data-root", default="D:/alpha-lab/data")
    parser.add_argument("--smoke", action="store_true",
                        help="Механика на одном символе и 2 эпохах; НЕ протокол")
    args = parser.parse_args(argv)

    cli._configure_stdio()
    if args.smoke:
        summary = run_demo(Path(args.data_root), ("BTCUSDT",), 2, 3,
                           protocol=False)
    else:
        summary = run_demo(Path(args.data_root), PANEL, N_EPOCHS, BATCH_SIZE,
                           protocol=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                              default=str), encoding="utf-8")
    print_table(summary)
    print(f"\nРезультат: {out}")
    print(f"n_trials (честное) = {summary['n_trials']}, "
          f"наборов параметров = {summary['param_sets_evaluated']}, "
          f"стена = {summary['wall_clock_seconds']:.0f} с")
    for label in ("verdict_train", "verdict_oos"):
        v = summary[label]
        if v is None:
            continue
        pbo = "n/a" if v["pbo"] is None else f"{v['pbo']:.3f}"
        print(f"{label}: DSR {v['dsr']:.4f}, p {v['p_value']:.4f}, PBO {pbo}, "
              f"alive={v['alive']}, сделок {v['trades']}")
        for reason in v["reasons"]:
            print(f"    причина: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
