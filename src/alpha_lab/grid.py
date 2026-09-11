"""Сетка гипотез: YAML-описание осей → детерминированный список конфигураций.

Формат (пример — configs/grids/mr.yaml):

    experiment:                 # базовый эксперимент формы load_experiment
      name: mr_grid
      strategy: mean_reversion
      timeframe: 1h             # базовое значение; ось timeframes переопределяет
      start: "2022-01-01"
      end: "2025-12-31"
      params: {window: 20, k: 2.0}
      costs: {...}
      validation: {...}
    axes:
      symbols: [BTCUSDT, ETHUSDT]
      timeframes: [1h, 4h]
      params:
        k: [1.5, 2.0, 2.5]

Развёртка — декартово произведение: symbols × timeframes × (произведение
параметрических осей). Параметры осей накладываются поверх base.params, все
остальные поля базы наследуются без изменений.

**Порядок развёртки (стабилен и является частью контракта).**
1. symbols — внешняя ось, значения в порядке файла;
2. timeframes — в порядке файла;
3. параметрические оси — по алфавиту имён (k, window, …), последняя ось
   меняется быстрее всех; значения каждой оси — в порядке файла.

**Идентификатор конфигурации** — sha256 от канонического JSON гипотезы
(strategy, params, timeframe, start, end, costs, validation, symbol), 16 hex.
Имя базового эксперимента в нагрузку не входит: это ярлык, а не гипотеза,
поэтому переименование сетки не инвалидирует уже посчитанное хранилище.
Порядок ключей YAML на id не влияет; значения должны быть записаны одинаково
(2 и 2.0 — разные id).

Размер сетки проверяется до запуска: size_warning возвращает громкий текст с
числом конфигураций и оценкой машинного времени по измеренной стоимости
(см. SECONDS_PER_CONFIG), а не «узнаем через час».
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from alpha_lab.config import VALID_TIMEFRAMES, Experiment, parse_experiment
from alpha_lab.strategies.base import strategy_param_names

# Измеренная стоимость полного вердикта одной конфигурации: generate +
# run_backtest 47.6 мс + validate 662 мс (из них permutation-тест 646 мс) на
# BTCUSDT 1h 2022–2025, 35 064 бара. Оценка размера опирается на замер.
#
# ВНИМАНИЕ: в замер не входит причинностный harness, а он в рабочем пути CLI
# стоит ~1.6 с на конфигурацию (35 064 бара). Реальная стоимость прогона на
# 1h-ряде такой длины — ~2.7 с/конфигурацию (замерен полный свип), поэтому
# оценка размера оптимистична примерно в 2.6-3.9 раза. Числа brief'а фазы 2
# (0.710 с) сохранены как заявленный замер; расхождение зафиксировано явно.
SECONDS_PER_CONFIG = 0.710
# Загрузка + ресемпл баров: 22.2 с на (символ, таймфрейм), один раз.
SECONDS_PER_LOAD = 22.2
# Порог громкого предупреждения: 1000 конфигураций — это ≈ 12 минут только
# счёта (плюс загрузки). Ниже порога свип укладывается в короткую сессию.
LARGE_GRID_THRESHOLD = 1000

_AXIS_KEYS = ("symbols", "timeframes", "params")


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader, замечающий дубликаты ключей.

    Обычный YAML молча оставляет последнее значение дублирующегося ключа:
    ``k: [1.5, 2.0]`` и строкой ниже ``k: [2.5]`` дали бы сетку из одного
    значения, и пользователь не узнал бы, что ось потеряна. Дубликат —
    ошибка конфига, а не оформление.
    """

    _source: str = "<YAML>"


def _construct_unique_mapping(loader, node, deep=False):
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, (str, int, float, bool, type(None))):
            raise ValueError(
                f"Ключ {key!r} в {loader._source} (строка "
                f"{key_node.start_mark.line + 1}) — не скаляр: отображения "
                f"сетки обязаны иметь скалярные ключи"
            )
        if key in mapping:
            raise ValueError(
                f"Дубликат ключа '{key}' в {loader._source} (строка "
                f"{key_node.start_mark.line + 1}): YAML молча берёт последнее "
                f"значение, а потерянная ось — это не та сетка, которую "
                f"описал пользователь"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _read_grid_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Сетка не найдена: {path}")
    if path.is_dir():
        raise ValueError(f"Сетка {path} — это каталог, а не YAML-файл")
    loader_cls = type("_GridLoader", (_UniqueKeyLoader,), {"_source": str(path)})
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.load(fh, Loader=loader_cls)
    except yaml.YAMLError as exc:
        raise ValueError(f"Сетка {path}: некорректный YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Сетка {path} должна быть YAML-словарём")
    return data


def _scalar_axis(raw, where: str) -> tuple[Any, ...]:
    """Ось — непустой список скалярных значений без повторов."""
    if not isinstance(raw, list):
        raise ValueError(
            f"Ось {where} должна быть списком значений, получено "
            f"{type(raw).__name__}"
        )
    if not raw:
        raise ValueError(f"Ось {where} пуста: перебирать нечего")
    out: list[Any] = []
    for value in raw:
        if isinstance(value, (list, dict)):
            raise ValueError(
                f"Ось {where}: значение {value!r} — не скаляр. Вложенный "
                f"список/словарь неоднозначен (одно это значение или "
                f"несколько), поэтому такая ось отвергается"
            )
        if value in out:
            raise ValueError(
                f"Ось {where}: значение {value!r} указано дважды — это одна "
                f"и та же конфигурация, а не перебор. Дубликат оси вырождает "
                f"сетку"
            )
        out.append(value)
    return tuple(out)


@dataclass(frozen=True)
class GridCell:
    """Одна конфигурация развёрнутой сетки."""

    index: int
    config_id: str
    symbol: str
    experiment: Experiment


@dataclass(frozen=True)
class Grid:
    """Разобранная сетка. expand() детерминированно даёт список конфигураций."""

    path: Path
    name: str
    base: Experiment
    symbols: tuple[str, ...]
    timeframes: tuple[str, ...]
    # Пары (имя параметра, значения), отсортированы по имени — контракт порядка.
    param_axes: tuple[tuple[str, tuple[Any, ...]], ...]

    def __len__(self) -> int:
        total = len(self.symbols) * len(self.timeframes)
        for _, values in self.param_axes:
            total *= len(values)
        return total

    def config_id(self, experiment: Experiment, symbol: str) -> str:
        """Детерминированный id гипотезы: канонический JSON → sha256[:16]."""
        payload = {
            "strategy": experiment.strategy,
            "params": experiment.params,
            "timeframe": experiment.timeframe,
            "start": experiment.start,
            "end": experiment.end,
            "costs": experiment.costs,
            "validation": experiment.validation,
            "symbol": symbol,
        }
        # sort_keys — порядок ключей YAML не влияет на id. Без default=str:
        # молчаливое приведение склеило бы разные значения в один id.
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def expand(self) -> list[GridCell]:
        names = [name for name, _ in self.param_axes]
        value_lists = [values for _, values in self.param_axes]
        cells: list[GridCell] = []
        index = 0
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                for combo in itertools.product(*value_lists):
                    params = {**self.base.params, **dict(zip(names, combo))}
                    experiment = replace(self.base, params=params,
                                         timeframe=timeframe)
                    cells.append(GridCell(
                        index=index,
                        config_id=self.config_id(experiment, symbol),
                        symbol=symbol,
                        experiment=experiment,
                    ))
                    index += 1
        return cells


def load_grid(path: str | Path) -> Grid:
    """Читает и валидирует сетку. Ошибки — ValueError/FileNotFoundError."""
    path = Path(path)
    data = _read_grid_yaml(path)

    unknown = sorted(set(data) - {"experiment", "axes"})
    if unknown:
        raise ValueError(
            f"Сетка {path}: неизвестные разделы {unknown}. Допустимые: "
            f"'experiment' (базовый эксперимент) и 'axes' (оси перебора)"
        )
    base_raw = data.get("experiment")
    if not isinstance(base_raw, dict):
        raise ValueError(
            f"Сетка {path}: раздел 'experiment' обязателен и должен быть "
            f"словарём формы load_experiment"
        )
    base = parse_experiment(base_raw, path)

    axes_raw = data.get("axes")
    if not isinstance(axes_raw, dict) or not axes_raw:
        raise ValueError(
            f"Сетка {path}: раздел 'axes' обязателен и должен содержать хотя "
            f"бы одну ось из {list(_AXIS_KEYS)}"
        )
    unknown_axes = sorted(set(axes_raw) - set(_AXIS_KEYS))
    if unknown_axes:
        raise ValueError(
            f"Сетка {path}: неизвестные оси {unknown_axes}. Допустимые: "
            f"{list(_AXIS_KEYS)}"
        )

    if "symbols" not in axes_raw:
        raise ValueError(
            f"Сетка {path}: ось symbols обязательна: свип обязан знать, какие "
            f"символы перебирать"
        )
    symbols = _scalar_axis(axes_raw["symbols"], "symbols")
    for symbol in symbols:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError(
                f"Сетка {path}: символ {symbol!r} — не непустая строка")
    # Опущенная ось timeframes — базовый таймфрейм эксперимента. Дефолт "1h"
    # молча подменял бы гипотезу: сетка с base.timeframe=4h, варьирующая
    # только символы, уходила бы считать 1h — не тот вопрос, который описал
    # автор. Контракт «ось переопределяет базу, отсутствие оси наследует базу».
    timeframes = (base.timeframe,) if "timeframes" not in axes_raw else \
        _scalar_axis(axes_raw["timeframes"], "timeframes")
    for timeframe in timeframes:
        if timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"Сетка {path}: ось timeframes содержит неизвестный таймфрейм "
                f"'{timeframe}'. Допустимые: {', '.join(VALID_TIMEFRAMES)}"
            )

    # Пространство параметров спрашивается один раз и проверяет оба входа:
    # params базового эксперимента и params осей. Без проверки базы опечатка
    # (`windwo`) принималась бы молча, стратегия игнорировала бы ключ и брала
    # дефолт — сетка считала бы не ту конфигурацию, которую описал автор, а
    # валидация осей создавала бы впечатление проверенности.
    accepted = strategy_param_names(base.strategy)
    unknown_base = sorted(set(base.params) - accepted)
    if unknown_base:
        raise ValueError(
            f"Сетка {path}: базовый эксперимент содержит параметры, которых "
            f"стратегия '{base.strategy}' не принимает: "
            f"{', '.join(unknown_base)}. Стратегия молча игнорирует "
            f"неизвестные ключи и берёт значения по умолчанию — прогон был бы "
            f"не той конфигурацией, которую описал автор. Допустимые "
            f"параметры: {', '.join(sorted(accepted))}"
        )

    param_axes: list[tuple[str, tuple[Any, ...]]] = []
    if "params" in axes_raw:
        params_raw = axes_raw["params"]
        if not isinstance(params_raw, dict) or not params_raw:
            raise ValueError(
                f"Сетка {path}: ось params должна быть непустым словарём "
                f"«имя параметра → список значений»"
            )
        for name in sorted(params_raw):
            if name not in accepted:
                raise ValueError(
                    f"Сетка {path}: стратегия '{base.strategy}' не принимает "
                    f"параметр '{name}'. Допустимые параметры: "
                    f"{', '.join(sorted(accepted))}"
                )
            param_axes.append(
                (name, _scalar_axis(params_raw[name], f"params.{name}")))

    grid = Grid(path=path, name=base.name, base=base, symbols=tuple(symbols),
                timeframes=tuple(timeframes), param_axes=tuple(param_axes))
    total = len(grid)
    if total == 1:
        raise ValueError(
            f"Сетка {path} разворачивается в одну конфигурацию — это "
            f"--config, а не свип. Добавьте хотя бы одно значение в ось "
            f"symbols/timeframes/params (или запустите одиночный прогон)"
        )
    return grid


def estimate_wall_clock(n_configs: int, n_loads: int) -> float:
    """Оценка машинного времени свипа, секунды: счёт + разовые загрузки баров."""
    return n_configs * SECONDS_PER_CONFIG + n_loads * SECONDS_PER_LOAD


def _format_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours} ч {minutes:02d} мин"
    if minutes:
        return f"{minutes} мин {secs:02d} с"
    return f"{secs} с"


def size_warning(n_configs: int, n_loads: int) -> str | None:
    """Громкое предупреждение о размере сетки или None ниже порога.

    n_loads — число различных (символ, таймфрейм): бары загружаются один раз
    на пару. Оценка — по измеренной стоимости (SECONDS_PER_CONFIG/SECONDS_PER_LOAD),
    а не по интуиции; текст говорит, что уменьшить и как продолжить после
    прерывания.
    """
    if n_configs <= LARGE_GRID_THRESHOLD:
        return None
    total = estimate_wall_clock(n_configs, n_loads)
    count = f"{n_configs:,}".replace(",", " ")
    return (
        f"ВНИМАНИЕ: сетка разворачивается в {count} конфигураций — это "
        f"≈ {_format_duration(total)} машинного времени "
        f"({SECONDS_PER_CONFIG:.3f} с на конфигурацию по замеру + "
        f"{n_loads} загрузк(и) баров × {SECONDS_PER_LOAD:.1f} с). Это "
        f"надолго: если такой перебор не задуман, уменьшите оси "
        f"params/timeframes/symbols. Прогон можно прервать и продолжить — "
        f"завершённые конфигурации не пересчитываются; повтор уже "
        f"выполненной конфигурации — только флагом --force."
    )
