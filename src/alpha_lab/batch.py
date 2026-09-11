"""Пакетный прогон сетки гипотез: бары один раз, вердикты в хранилище.

    alpha-lab sweep --grid configs/grids/mr.yaml --out results/

Устройство:

* **Бары загружаются один раз на (символ, таймфрейм)** и переиспользуются
  всеми конфигурациями группы: измеренная загрузка+ресемпл — 22.2 с, при
  тысячах конфигураций повторять её недопустимо. Группа — это все ячейки
  сетки с общим (символ, таймфрейм, период).
* **Вердикты считает тот же production-путь**, что и одиночный CLI: ячейка
  идёт через cli.run_config (harness причинности → сегментированный generate →
  маска грязных баров и разрывов → run_backtest) и cli.validate_config (тот же
  validate с позициями движка, тот же n_trials семьи). Свип не форкает
  последовательность: расхождение чисел с CLI было бы худшим исходом для
  исследовательского полигона.
* **PBO** считается по матрице доходностей внутри группы — только у
  конфигураций на одном символе/периоде/таймфрейме ряды выровнены (spec 6.5).
  Если в группе меньше двух посчитанных конфигураций или ряды вырождены
  (идентичные колонки), PBO не оценивается и это честно попадает в warnings
  вердикта, а не подменяется похожим числом. На возобновлённом прогоне
  недосчитанная часть группы даёт матрицу по меньшему числу колонок — это
  видно по warnings; полный свежий прогон совпадает с `validate --configs`.
* **Возобновляемость**: ключ — config_id ячейки из grid.py. Хранилище
  опрашивается через узкий интерфейс ResultSink.completed_ids(data_version);
  выполненные конфигурации не пересчитываются (повтор — force=True). Успешная
  строка — выполненная; строка с ошибкой — нет: отказ бывает временным, и
  следующий запуск обязан её повторить. Строки пишутся батчами по FLUSH_EVERY,
  поэтому обрыв стоит не больше нескольких конфигураций.
* **Изоляция отказов**: ошибка конфигурации (плохие параметры, отказ
  стратегии, проблема данных) записывается строкой с error и не прерывает
  свип; недоступные данные группы помечают все её ячейки и свип идёт дальше.
* **Параллелизма нет — по замеру, а не по интуиции.** BTCUSDT 1h 2022–2025
  (35 064 бара, 10 конфигураций): run_config последовательно 16.1–16.4 с,
  4 потока 17.4–19.3 с (хуже: harness держит GIL), 4 процесса 13.6–14.5 с
  (1.1x); validate последовательно 10.3–11.4 с, 4 потока 2.4–3.7 с (3–4x),
  4 процесса 3.7 с. Изолированная фаза ускоряется, но end-to-end на реальной
  сетке выигрыша нет: 12 конфигураций в одной группе — 52.1 с
  последовательно против 52.0 с на 4 потоках (машина делится с параллельными
  агентами), демо-сетка 2×2×3 — 110.5 против 129.8 с (загрузки баров
  доминируют). Поэтому свип идёт последовательно: сложность пула не
  окупается.

Интерфейс хранилища (P3, src/alpha_lab/results.py), на который опирается модуль:

    read_runs(store_path) -> DataFrame   # колонки config_id, data_version,
                                         # error (пусто/NaN у успешных)
    write_runs(store_path, rows) -> None # идемпотентно по config_id

Пока модуля нет, open_result_store отдаёт временное parquet-хранилище с тем же
контрактом (один файл runs.parquet в каталоге --out); граница узкая, поэтому
подключение P3 сводится к удалению этого фолбэка.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

import pandas as pd

from alpha_lab.config import Universe
from alpha_lab.data.query import data_version
from alpha_lab.grid import Grid, GridCell
from alpha_lab.validation.validator import build_returns_matrix

# Строк на батч записи в хранилище: обрыв свипа теряет не больше этого числа
# уже посчитанных конфигураций, а хранилище не переписывается на каждую строку.
FLUSH_EVERY = 16

# Каденция строк прогресса, секунды. Свип на часы не имеет права печатать
# строку на конфигурацию: 10 000 конфигураций — это 10 000 строк.
PROGRESS_INTERVAL = 30.0


class SweepError(ValueError):
    """Свип не может начаться: непригодный журнал, символ вне юниверса и т.п."""


class ResultSink(Protocol):
    """Узкий контракт хранилища результатов (P3)."""

    def completed_ids(self, data_version: str) -> set[str]:
        """id успешно посчитанных конфигураций для данной версии данных."""

    def write(self, rows: list[dict]) -> None:
        """Идемпотентно записывает строки, ключ — config_id."""


@dataclass(frozen=True)
class SweepSummary:
    total: int
    executed: int
    skipped: int
    failed: int
    store_path: Path | None


def _completed_from_frame(frame: pd.DataFrame | None,
                          data_version: str) -> set[str]:
    """id выполненных конфигураций из таблицы хранилища.

    Выполненная — строка без ошибки (error пуст/NaN). Строки с другой версией
    данных не считаются выполненными: данные изменились, вердикт устарел, и
    переиспользовать его молча нельзя. Колонка error отсутствует — считаем
    все строки успешными (совместимость с хранилищем без поля ошибок).
    """
    if frame is None or len(frame) == 0:
        return set()
    if "config_id" not in frame.columns:
        raise ValueError(
            "Хранилище результатов не содержит колонку 'config_id': "
            "возобновляемость по id конфигурации невозможна"
        )
    done = frame
    if "error" in done.columns:
        err = done["error"]
        done = done[err.isna() | (err.astype(str).str.strip() == "")]
    if "data_version" in done.columns:
        done = done[done["data_version"].astype(str) == str(data_version)]
    return set(done["config_id"].astype(str))


class _ResultsModuleSink:
    """Адаптер к alpha_lab.results (P3): read_runs/write_runs."""

    def __init__(self, module, path: Path):
        self._module = module
        self._path = path

    def completed_ids(self, data_version: str) -> set[str]:
        return _completed_from_frame(self._module.read_runs(self._path),
                                     data_version)

    def write(self, rows: list[dict]) -> None:
        self._module.write_runs(self._path, rows)


class _ParquetResultStore:
    """Временное хранилище до P3: один parquet-файл в каталоге --out.

    Контракт тот же, что у results.py (идемпотентность по config_id), поэтому
    замена фолбэка на P3 не меняет поведение свипа. Файл перезаписывается
    атомарно (tmp + os.replace): обрыв не оставляет половину таблицы.
    """

    FILE_NAME = "runs.parquet"

    def __init__(self, path: Path):
        self.path = Path(path)
        self.file = self.path / self.FILE_NAME

    def completed_ids(self, data_version: str) -> set[str]:
        if not self.file.exists():
            return set()
        return _completed_from_frame(pd.read_parquet(self.file), data_version)

    def write(self, rows: list[dict]) -> None:
        if not rows:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        fresh = pd.DataFrame(rows)
        if self.file.exists():
            old = pd.read_parquet(self.file)
            fresh = pd.concat([old, fresh], ignore_index=True)
        fresh = fresh.drop_duplicates(subset=["config_id"], keep="last")
        tmp = self.file.with_name(self.FILE_NAME + ".tmp")
        fresh.to_parquet(tmp, index=False)
        os.replace(tmp, self.file)


def open_result_store(store_path: str | Path) -> ResultSink:
    """Хранилище результатов: P3, если он есть, иначе временный parquet.

    Импорт results ленивый и единственный: пока модуля нет, свип работает на
    временном хранилище с тем же контрактом. Как только P3 появится, этот
    выбор подхватит его без правок свипа.
    """
    try:
        from alpha_lab import results as results_mod
    except ImportError:
        return _ParquetResultStore(Path(store_path))
    if not (hasattr(results_mod, "read_runs") and
            hasattr(results_mod, "write_runs")):
        raise RuntimeError(
            "alpha_lab.results не предоставляет read_runs/write_runs: "
            "интерфейс хранилища разошёлся с ожиданием batch.py"
        )
    return _ResultsModuleSink(results_mod, Path(store_path))


def _format_hms(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_progress(done: int, total: int, elapsed: float, eta: float,
                    label: str) -> str:
    """Строка прогресса: позиция в сетке, процент, прошло, осталось, ячейка."""
    pct = 100.0 * done / total if total else 100.0
    return (f"[{done}/{total}] {pct:5.1f}%  прошло {_format_hms(elapsed)}  "
            f"осталось ≈ {_format_hms(eta)}  {label}")


class ProgressReporter:
    """Печатает прогресс по каденции, а не по конфигурации.

    Между строками проходит interval секунд; старт и финиш печатаются всегда.
    Часы инжектируются, чтобы тест проверял каденцию без sleep.
    """

    def __init__(self, total: int, stream=None,
                 interval: float = PROGRESS_INTERVAL,
                 clock: Callable[[], float] = time.monotonic):
        self.total = total
        self.stream = sys.stderr if stream is None else stream
        self.interval = interval
        self.clock = clock
        self.started = self.clock()
        self.last = self.started
        self._announced = False

    def start(self, preamble: str = "") -> None:
        self.started = self.last = self.clock()
        self._announced = True
        suffix = f" {preamble}" if preamble else ""
        print(f"Свип: {self.total} конфигураций.{suffix}", file=self.stream)
        self.stream.flush()

    def tick(self, done: int, label: str, eta: float) -> None:
        now = self.clock()
        if not self._announced or now - self.last < self.interval:
            return
        self.last = now
        print(format_progress(done, self.total, now - self.started, eta, label),
              file=self.stream)
        self.stream.flush()

    def finish(self, done: int, label: str = "готово") -> None:
        print(format_progress(done, self.total, self.clock() - self.started, 0.0,
                              label), file=self.stream)
        self.stream.flush()


def _cli():
    """Ленивый доступ к production-пути CLI (избегает цикла импортов)."""
    from alpha_lab import cli
    return cli


def _params_json(params: dict) -> str:
    return json.dumps(params, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def _row(cell: GridCell, data_version: str, *, experiment_id: str,
         n_trials: int, verdict=None, error: str = "",
         duration_s: float = 0.0) -> dict:
    """Строка результата: успех несёт метрики вердикта, отказ — только error."""
    row: dict[str, Any] = {
        "config_id": cell.config_id,
        "experiment_id": experiment_id,
        "index": cell.index,
        "symbol": cell.symbol,
        "timeframe": cell.experiment.timeframe,
        "strategy": cell.experiment.strategy,
        "params_json": _params_json(cell.experiment.params),
        "start": cell.experiment.start,
        "end": cell.experiment.end or "",
        "error": error,
        "data_version": data_version,
        "duration_s": float(duration_s),
    }
    if verdict is None:
        row.update({
            "sharpe": float("nan"), "dsr": float("nan"),
            "p_value": float("nan"), "pbo": float("nan"),
            "max_dd": float("nan"), "total_return": float("nan"),
            "trades": 0, "n_trials": int(n_trials), "alive": False,
            "reasons": "", "warnings": "",
        })
    else:
        row.update({
            "sharpe": float(verdict.sharpe), "dsr": float(verdict.dsr),
            "p_value": float(verdict.p_value), "pbo": float(verdict.pbo),
            "max_dd": float(verdict.max_dd),
            "total_return": float(verdict.total_return),
            "trades": int(verdict.trades),
            "n_trials": int(verdict.n_configs_tried),
            "alive": bool(verdict.alive),
            "reasons": " | ".join(verdict.reasons),
            "warnings": " | ".join(verdict.warnings),
        })
    return row


def _trial_keys(cells: list[GridCell]) -> list[dict]:
    return [
        {"strategy": c.experiment.strategy, "params": c.experiment.params,
         "symbol": c.symbol, "timeframe": c.experiment.timeframe}
        for c in cells
    ]


def run_sweep(grid: Grid, *, data_root: Path, universe: Universe,
              store_path: str | Path | None = None, store: ResultSink | None = None,
              force: bool = False, skip_causality: bool = False,
              journal: str | Path | None = None,
              ignore_journal: bool = False,
              progress_stream=None,
              progress_interval: float = PROGRESS_INTERVAL) -> SweepSummary:
    """Прогоняет сетку, переиспользуя бары и не повторяя сделанное.

    store — тестовый/внешний ResultSink; иначе открывается open_result_store
    по store_path. force — пересчитать всё, даже лежащее в хранилище.
    Идёт последовательно: замеренные потоки/процессы end-to-end выигрыша не
    дали (докстринг модуля).
    """
    if store is None:
        if store_path is None:
            raise ValueError("run_sweep требует store_path или store")
        store = open_result_store(store_path)
    cli = _cli()
    root = Path(data_root)
    dv = data_version(root)
    cells = grid.expand()

    done = set() if force else set(store.completed_ids(dv))
    pending = [c for c in cells if c.config_id not in done]
    skipped = len(cells) - len(pending)

    # Юниверс — входной контракт: символ вне состава означал бы прогон не той
    # гипотезы. Проверка до загрузок, чтобы ошибка стоила секунды, а не часа.
    unknown = sorted({c.symbol for c in cells} - set(universe.symbols))
    if unknown:
        raise SweepError(
            f"символы {', '.join(unknown)} не входят в юниверс "
            f"(символов: {len(universe.symbols)}). Доступные: "
            f"{', '.join(universe.symbols)}. Добавьте символ в юниверс или "
            f"исправьте ось symbols сетки"
        )

    journal_path = (Path(journal) if journal is not None
                    else cli.DEFAULT_JOURNAL)
    keys = _trial_keys(cells)
    family = sorted(cli.trial_fingerprint(key) for key in keys)
    if ignore_journal:
        # Как и в одиночном CLI: осознанный отказ от защиты обязан быть
        # громким. n_trials = размер семьи, а не остаток: штраф DSR не имеет
        # права уменьшаться от того, что часть работы уже лежала в хранилище.
        n_trials = len(cells)
        print(f"Предупреждение: --ignore-journal: журнал не читается и не "
              f"пишется, n_trials = {n_trials} — защита от множественных "
              f"сравнений ОТКЛЮЧЕНА, DSR завышен, вердикт «жива» может быть "
              f"ложным", file=sys.stderr)
    else:
        problem = cli.journal_problem(journal_path)
        if problem is not None:
            raise SweepError(
                f"журнал попыток непригоден: {journal_path} ({problem}). "
                f"Свип остановлен: без журнала n_trials занижается, DSR "
                f"завышается, и вердикты выглядят лучше правды. Осознанный "
                f"отказ от защиты — --ignore-journal"
            )
        prior = cli.count_prior_trials_for_keys(journal_path, keys)
        # Попытки текущего запуска — только те, что реально считаются сейчас;
        # прошлые (в том числе пропущенные по хранилищу) уже лежат в журнале и
        # учтены prior. Так полный прогон даёт prior + N, а возобновлённый —
        # ровно число реально сделанных попыток, без двойного штрафа.
        n_trials = prior + len(pending)

    # Группировка по (символ, таймфрейм, период): внутри группы бары общие.
    groups: dict[tuple, list[GridCell]] = {}
    for cell in pending:
        key = (cell.symbol, cell.experiment.timeframe, cell.experiment.start,
               cell.experiment.end)
        groups.setdefault(key, []).append(cell)

    reporter = ProgressReporter(len(cells), stream=progress_stream,
                                interval=progress_interval)
    reporter.start(
        f"Пропущено по хранилищу: {skipped}. Загрузок баров: {len(groups)}."
        if skipped else f"Загрузок баров: {len(groups)}."
    )

    gh = cli.git_hash()
    buffer: list[dict] = []
    executed = 0
    failed = 0
    durations: list[float] = []
    processed = skipped

    def flush() -> None:
        if buffer:
            store.write(list(buffer))
            buffer.clear()

    for group_i, (key, group_cells) in enumerate(groups.items()):
        symbol, timeframe, start, end = key
        groups_remaining = len(groups) - group_i
        try:
            data = cli.prepare_data(root, symbol, timeframe, start, end)
        except Exception as exc:      # noqa: BLE001 — отказ группы не роняет свип
            error = f"{type(exc).__name__}: {exc}"
            for cell in group_cells:
                rows = _row(cell, dv, experiment_id="", n_trials=n_trials,
                            error=error)
                buffer.append(rows)
                executed += 1
                failed += 1
                processed += 1
                reporter.tick(processed, f"{symbol} {timeframe}", eta=0.0)
            flush()
            continue

        warnings = data.data_warnings
        if skip_causality:
            warnings = (*warnings, cli.CAUSALITY_DISABLED_WARNING)
        outcomes: list[tuple[GridCell, Any]] = []
        for cell in group_cells:
            started = time.monotonic()
            try:
                outcome = cli.run_config(data, cell.experiment,
                                         skip_causality=skip_causality)
            except Exception as exc:  # noqa: BLE001 — одна ячейка не роняет свип
                durations.append(time.monotonic() - started)
                buffer.append(_row(
                    cell, dv, experiment_id="", n_trials=n_trials,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_s=durations[-1]))
                executed += 1
                failed += 1
            else:
                durations.append(time.monotonic() - started)
                outcomes.append((cell, outcome))
                executed += 1
            processed += 1
            label = f"{symbol} {timeframe} {cli._params_summary(cell.experiment.params)}"
            mean = (sum(durations) / len(durations)) if durations else 0.0
            remaining_cells = len(cells) - processed
            eta = mean * remaining_cells + groups_remaining * 22.2
            reporter.tick(processed, label, eta=eta)

        # PBO — свойство группы выровненных рядов: только у ячеек одного
        # символа/периода/таймфрейма матрица T × N корректна.
        matrix = None
        pbo_value = None
        matrix_warning = None
        if len(outcomes) >= 2:
            try:
                matrix = build_returns_matrix({
                    f"[{cell.index}] {cell.experiment.name}":
                        outcome.result.returns.set_axis(data.ts_index)
                    for cell, outcome in outcomes
                })
                from alpha_lab.validation.significance import pbo_cscv
                pbo_value = pbo_cscv(matrix)
            except ValueError as exc:
                matrix = None
                pbo_value = None
                matrix_warning = (
                    f"PBO не оценён: матрица доходностей группы непригодна "
                    f"({exc})")
                print(f"Предупреждение: {matrix_warning}", file=sys.stderr)
        elif outcomes:
            matrix_warning = (
                f"PBO не оценён: в группе {symbol} {timeframe} посчитана "
                f"одна конфигурация (нужно ≥ 2 выровненных рядов)"
            )
        cell_warnings = warnings
        if matrix_warning:
            cell_warnings = (*warnings, matrix_warning)

        # Пул не используется: замер end-to-end на реальной сетке выигрыша не
        # дал (см. докстринг модуля), а сложность и недетерминизм порядка
        # готовности результатов не окупаются.
        for cell, outcome in outcomes:
            exp_id = cli.experiment_id(
                cli._experiment_payload(cell.experiment, cell.symbol, universe),
                dv, gh)
            started = time.monotonic()
            try:
                verdict = cli.validate_config(
                    outcome, data, experiment_id=exp_id, n_trials=n_trials,
                    returns_matrix=matrix, pbo_value=pbo_value,
                    warnings=cell_warnings)
            except Exception as exc:  # noqa: BLE001 — отказ ячейки изолирован
                failed += 1
                buffer.append(_row(
                    cell, dv, experiment_id=exp_id, n_trials=n_trials,
                    error=f"{type(exc).__name__}: {exc}",
                    duration_s=time.monotonic() - started))
            else:
                if not ignore_journal:
                    cli.log_trial(
                        journal_path,
                        {"strategy": cell.experiment.strategy,
                         "params": cell.experiment.params, "symbol": cell.symbol,
                         "timeframe": cell.experiment.timeframe},
                        exp_id,
                        {"sharpe": verdict.sharpe, "dsr": verdict.dsr,
                         "alive": verdict.alive},
                        family=family)
                buffer.append(_row(
                    cell, dv, experiment_id=exp_id, n_trials=n_trials,
                    verdict=verdict, duration_s=time.monotonic() - started))
            if len(buffer) >= FLUSH_EVERY:
                flush()
        flush()

    reporter.finish(processed, label=f"готово, ошибок: {failed}")
    return SweepSummary(total=len(cells), executed=executed, skipped=skipped,
                        failed=failed,
                        store_path=Path(store_path) if store_path else None)


def cmd_sweep(args) -> int:
    """CLI-команда sweep: разворачивает сетку и запускает пакетный прогон."""
    from alpha_lab.cli import EXIT_ERROR, EXIT_FAILED, EXIT_OK, load_universe
    from alpha_lab.grid import load_grid, size_warning

    try:
        grid = load_grid(args.grid)
        universe = load_universe(args.universe)
    except (OSError, ValueError) as exc:
        print(f"Ошибка конфига: {exc}", file=sys.stderr)
        return EXIT_ERROR

    # Предупреждение о размере — до первой загрузки баров: сетка на 200 000
    # конфигураций обязана сказать об этом сразу, а не через час.
    cells = grid.expand()
    n_loads = len({(c.symbol, c.experiment.timeframe) for c in cells})
    warning = size_warning(len(cells), n_loads)
    if warning:
        print(warning, file=sys.stderr)

    try:
        summary = run_sweep(
            grid, store_path=Path(args.out), data_root=Path(args.data_root),
            universe=universe, force=args.force,
            skip_causality=args.skip_causality, journal=Path(args.journal),
            ignore_journal=args.ignore_journal)
    except (SweepError, OSError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print(f"\nСвип завершён: всего {summary.total}, "
          f"выполнено {summary.executed}, пропущено {summary.skipped}, "
          f"ошибок {summary.failed}")
    print(f"Хранилище: {summary.store_path}")
    # Ошибки отдельных конфигураций — не сбой запуска (свип дошёл до конца),
    # но и не «всё хорошо»: код возврата обязан их показать.
    return EXIT_FAILED if summary.failed else EXIT_OK


def add_sweep_subparser(sub) -> None:
    """Регистрирует команду `alpha-lab sweep` в общем парсере CLI."""
    p_sweep = sub.add_parser(
        "sweep", help="Прогнать сетку гипотез из YAML, переиспользуя бары")
    p_sweep.add_argument("--grid", required=True,
                         help="Путь к YAML-сетке (experiment + axes)")
    from alpha_lab.data.store import DEFAULT_ROOT

    p_sweep.add_argument("--out", default="results",
                         help="Каталог хранилища результатов (P3)")
    p_sweep.add_argument("--data-root", default=str(DEFAULT_ROOT))
    p_sweep.add_argument("--universe", default="configs/universe.yaml",
                         help="Юниверс: все символы сетки обязаны в него "
                              "входить")
    p_sweep.add_argument("--journal", default=None,
                         help="Журнал экспериментов: из него берётся число "
                              "попыток для DSR")
    p_sweep.add_argument("--ignore-journal", action="store_true",
                         help="Сознательно не читать и не писать журнал: "
                              "n_trials = размер сетки, защита от "
                              "множественных сравнений отключается")
    p_sweep.add_argument("--skip-causality", action="store_true",
                         help="Сознательно отключить причинностную проверку "
                              "стратегии — единственную защиту от look-ahead "
                              "(spec 8.1)")
    p_sweep.add_argument("--force", action="store_true",
                         help="Пересчитать все конфигурации, даже уже "
                              "лежащие в хранилище")
    p_sweep.set_defaults(func=cmd_sweep)
