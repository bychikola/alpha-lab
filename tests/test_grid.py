"""Тесты сетки гипотез: детерминированная развёртка, пространство параметров, размер.

Сеть и реальное хранилище не трогаются: все сетки — временные YAML в tmp_path.
"""
from __future__ import annotations

import pytest

import alpha_lab.strategies.base as strategies_base
from alpha_lab.grid import (
    LARGE_GRID_THRESHOLD, SECONDS_PER_CONFIG, Grid, estimate_wall_clock,
    load_grid, size_warning,
)

GRID_YAML = """\
experiment:
  name: mr_grid
  strategy: mean_reversion
  timeframe: 1h
  start: "2022-01-01"
  end: "2025-12-31"
  params:
    window: 20
    k: 2.0
  costs: {taker_fee_bps: 5.0}
  validation: {min_trades: 100}
axes:
  symbols: [BTCUSDT, ETHUSDT]
  timeframes: [1h, 4h]
  params:
    k: [1.5, 2.0]
"""


def _write_grid(tmp_path, text: str = GRID_YAML):
    path = tmp_path / "grid.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class _FakeStrategy:
    """Стратегия вне штатного реестра: проверяет расширяемость без правки grid.py."""

    name = "fake"
    PARAM_NAMES = frozenset({"threshold", "window"})

    def __init__(self, params=None):
        self.params = params or {}


def test_grid_expands_to_cross_product_with_stable_ids(tmp_path):
    """k × timeframes × symbols — ровно 8 конфигураций, порядок и id устойчивы."""
    path = _write_grid(tmp_path)
    first = load_grid(path)
    second = load_grid(path)
    cells_a = first.expand()
    cells_b = second.expand()

    assert len(cells_a) == 8
    # Документированный порядок развёртки: symbols (внешняя ось, порядок файла),
    # затем timeframes (порядок файла), затем параметры по алфавиту имён.
    got = [(c.symbol, c.experiment.timeframe, c.experiment.params["k"])
           for c in cells_a]
    assert got == [
        ("BTCUSDT", "1h", 1.5), ("BTCUSDT", "1h", 2.0),
        ("BTCUSDT", "4h", 1.5), ("BTCUSDT", "4h", 2.0),
        ("ETHUSDT", "1h", 1.5), ("ETHUSDT", "1h", 2.0),
        ("ETHUSDT", "4h", 1.5), ("ETHUSDT", "4h", 2.0),
    ]
    # index — позиция в развёртке, нумерация с нуля и без дыр.
    assert [c.index for c in cells_a] == list(range(8))
    # Повторный разбор того же файла даёт побитово тот же результат.
    assert [(c.index, c.config_id, c.symbol) for c in cells_a] == \
           [(c.index, c.config_id, c.symbol) for c in cells_b]
    # id различают и параметры, и таймфрейм, и символ.
    assert len({c.config_id for c in cells_a}) == 8
    assert cells_a[0].config_id != cells_a[1].config_id   # разные значения k
    assert cells_a[0].config_id != cells_a[2].config_id   # разные таймфреймы
    assert cells_a[0].config_id != cells_a[4].config_id   # разные символы
    # База заполняет не перебираемые параметры, ось — переопределяет.
    assert cells_a[0].experiment.params["window"] == 20
    assert cells_a[0].experiment.params["k"] == 1.5
    assert cells_a[0].experiment.costs == {"taker_fee_bps": 5.0}
    assert cells_a[0].experiment.validation == {"min_trades": 100}


def test_grid_ids_ignore_yaml_key_order_and_list_order(tmp_path):
    """id — хеш содержания конфигурации, а не оформления YAML.

    Порядок ключей и порядок значений в списках — оформление: множество id
    обязано совпасть. Порядок развёртки следует за файлом (документирован).
    """
    shuffled = """\
axes:
  params:
    k: [2.0, 1.5]
  timeframes: [4h, 1h]
  symbols: [ETHUSDT, BTCUSDT]
experiment:
  validation: {min_trades: 100}
  costs: {taker_fee_bps: 5.0}
  params:
    k: 2.0
    window: 20
  end: "2025-12-31"
  start: "2022-01-01"
  timeframe: 1h
  strategy: mean_reversion
  name: mr_grid
"""
    base = load_grid(_write_grid(tmp_path)).expand()
    other = load_grid(_write_grid(tmp_path, shuffled)).expand()

    assert {c.config_id for c in base} == {c.config_id for c in other}
    # Развёртка идёт по порядку списка в файле, а не по алфавиту значений.
    assert other[0].symbol == "ETHUSDT"
    assert other[0].experiment.timeframe == "4h"
    assert other[0].experiment.params["k"] == 2.0


def test_grid_accepts_new_strategy_without_editing_grid_module(tmp_path,
                                                               monkeypatch):
    """Новая стратегия обязана работать через реестр, а не через правку grid.py."""
    monkeypatch.setitem(strategies_base.REGISTRY, "fake", _FakeStrategy)
    text = GRID_YAML.replace("strategy: mean_reversion", "strategy: fake") \
                    .replace("window: 20\n    k: 2.0", "threshold: 1.0") \
                    .replace("k: [1.5, 2.0]", "threshold: [1.0, 2.0]")
    cells = load_grid(_write_grid(tmp_path, text)).expand()

    assert len(cells) == 8
    assert {c.experiment.params["threshold"] for c in cells} == {1.0, 2.0}


def test_grid_omitted_timeframes_axis_falls_back_to_base(tmp_path):
    """Опущенная ось timeframes — базовый таймфрейм, а не молчаливый 1h.

    Сетка, варьирующая только символы, обязана считать гипотезу на том
    таймфрейме, который записан в эксперименте: подмена 4h на 1h — это
    молчаливый прогон другой гипотезы, а не «разумный дефолт».
    """
    text = GRID_YAML.replace("timeframe: 1h", "timeframe: 4h") \
                    .replace("  timeframes: [1h, 4h]\n", "")
    grid = load_grid(_write_grid(tmp_path, text))
    assert grid.timeframes == ("4h",)
    cells = grid.expand()
    assert len(cells) == 4                       # 2 символа × 2 значения k
    assert {c.experiment.timeframe for c in cells} == {"4h"}


def test_grid_rejects_unknown_parameter(tmp_path):
    """Параметр вне пространства стратегии — ошибка до единого бэктеста."""
    text = GRID_YAML.replace("k: [1.5, 2.0]", "nonexistent: [1, 2]")
    with pytest.raises(ValueError, match="nonexistent") as exc:
        load_grid(_write_grid(tmp_path, text)).expand()
    # Сообщение обязано перечислять принятые имена: без них ошибка не чинится.
    assert "k" in str(exc.value) and "window" in str(exc.value)


def test_grid_rejects_unknown_base_parameter(tmp_path):
    """Опечатка в params базового эксперимента — ошибка, а не тихий игнор.

    Стратегия молча игнорирует неизвестный ключ и берёт дефолт: свип ушёл бы
    считать конфигурацию, которую автор не писал, а валидация осей создавала
    бы впечатление проверенной сетки целиком.
    """
    text = GRID_YAML.replace("    window: 20\n", "    windwo: 50\n")
    with pytest.raises(ValueError, match="windwo") as exc:
        load_grid(_write_grid(tmp_path, text))
    # Сообщение обязано перечислять принятые имена — без них опечатка не чинится.
    message = str(exc.value)
    assert "window" in message and "k" in message


def test_grid_rejects_empty_axis(tmp_path):
    for broken in ("symbols: []", "timeframes: []", "k: []"):
        text = GRID_YAML.replace(
            {"symbols: []": "symbols: [BTCUSDT, ETHUSDT]",
             "timeframes: []": "timeframes: [1h, 4h]",
             "k: []": "k: [1.5, 2.0]"}[broken], broken)
        with pytest.raises(ValueError, match="пуст"):
            load_grid(_write_grid(tmp_path, text)).expand()


def test_grid_rejects_single_configuration(tmp_path):
    """Одна конфигурация — это --config: перебор из неё не состоит."""
    text = GRID_YAML.replace("symbols: [BTCUSDT, ETHUSDT]", "symbols: [BTCUSDT]") \
                    .replace("timeframes: [1h, 4h]", "timeframes: [1h]") \
                    .replace("k: [1.5, 2.0]", "k: [2.0]")
    with pytest.raises(ValueError, match="одн"):
        load_grid(_write_grid(tmp_path, text)).expand()


def test_grid_rejects_duplicate_axis_values(tmp_path):
    for broken, needle in (
        ("k: [1.5, 1.5]", "k"),
        ("symbols: [BTCUSDT, BTCUSDT]", "BTCUSDT"),
        ("timeframes: [1h, 1h]", "1h"),
    ):
        healthy = {"k: [1.5, 1.5]": "k: [1.5, 2.0]",
                   "symbols: [BTCUSDT, BTCUSDT]":
                       "symbols: [BTCUSDT, ETHUSDT]",
                   "timeframes: [1h, 1h]": "timeframes: [1h, 4h]"}[broken]
        with pytest.raises(ValueError, match=needle):
            load_grid(_write_grid(tmp_path,
                                  GRID_YAML.replace(healthy, broken))).expand()


def test_grid_rejects_duplicate_yaml_keys(tmp_path):
    """YAML молча берёт последний дубликат ключа; сетка обязана это заметить."""
    text = GRID_YAML.replace("k: [1.5, 2.0]", "k: [1.5, 2.0]\n    k: [2.5]")
    with pytest.raises(ValueError, match="k"):
        load_grid(_write_grid(tmp_path, text)).expand()


def test_grid_rejects_invalid_timeframe_and_unknown_axis(tmp_path):
    text = GRID_YAML.replace("timeframes: [1h, 4h]", "timeframes: [1h, 7h]")
    with pytest.raises(ValueError, match="7h"):
        load_grid(_write_grid(tmp_path, text)).expand()

    text = GRID_YAML.replace("  symbols:", "  periods:").replace(
        "  timeframes: [1h, 4h]", "  timeframes: [1h, 4h]")
    with pytest.raises(ValueError, match="periods"):
        load_grid(_write_grid(tmp_path, text)).expand()


def test_grid_rejects_non_list_and_nested_axis_values(tmp_path):
    text = GRID_YAML.replace("k: [1.5, 2.0]", "k: 2.0")
    with pytest.raises(ValueError, match="спис"):
        load_grid(_write_grid(tmp_path, text)).expand()

    text = GRID_YAML.replace("k: [1.5, 2.0]", "k: [[1.5, 2.0]]")
    with pytest.raises(ValueError, match="скаляр"):
        load_grid(_write_grid(tmp_path, text)).expand()


def test_size_warning_fires_above_threshold():
    """Порог и оценка времени: до запуска видно и число, и часы."""
    assert size_warning(LARGE_GRID_THRESHOLD, 1) is None
    warning = size_warning(200_000, 4)
    assert warning is not None
    assert "200 000" in warning
    # Оценка опирается на измеренную стоимость полного пути конфигурации, а не
    # на догадку: причинностный harness обязан входить в число (замер на 1h
    # 35 064 бара — ~2.3 с), иначе оценка оптимистична в разы.
    assert SECONDS_PER_CONFIG >= 2.0
    assert "ч" in warning
    assert str(int(200_000 * SECONDS_PER_CONFIG // 3600)) in warning
    assert estimate_wall_clock(200_000, 4) == pytest.approx(
        200_000 * SECONDS_PER_CONFIG + 4 * 22.2)
    # Текст называет, что именно входит в оценку, и что это оценка.
    assert "harness" in warning
    assert "validate" in warning
    assert "оценка" in warning.lower()
    assert "35 000" in warning          # названа база замера (длина ряда)


def test_size_warning_names_resume_and_axes():
    """Предупреждение обязано быть действенным: что уменьшить и как продолжить."""
    warning = size_warning(LARGE_GRID_THRESHOLD + 1, 2)
    assert warning is not None
    assert "--force" in warning
    assert "params" in warning and "timeframes" in warning and "symbols" in warning


def test_grid_carries_name_and_path(tmp_path):
    grid = load_grid(_write_grid(tmp_path))
    assert isinstance(grid, Grid)
    assert grid.name == "mr_grid"
    assert grid.path == tmp_path / "grid.yaml"
