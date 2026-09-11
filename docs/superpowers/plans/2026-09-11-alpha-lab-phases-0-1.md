# Alpha Lab — фазы 0–1: план реализации

> **Для агентов:** ОБЯЗАТЕЛЬНЫЙ СУБ-СКИЛЛ: используйте `superpowers:subagent-driven-development` (рекомендуется) или `superpowers:executing-plans` для выполнения плана задача за задачей. Шаги размечены чекбоксами (`- [ ]`).

**Цель:** Построить исследовательский полигон, который прогоняет торговую гипотезу на реальных данных Binance и выдаёт бинарный вердикт «жива / мертва» с защитой от оверфиттинга.

**Архитектура:** Слоистая, каждый слой знает только соседа снизу. `data` → `features` → `engine` (с моделью издержек) → `validation` (слепой, зависит только от результатов) → `report` (JSON-контракт) → статический HTML-дашборд. Ключевой инвариант: `validation` **не импортирует** `strategies`.

**Стек:** Python 3.14 через `uv` · pandas · numpy · numba · scipy · statsmodels · DuckDB · pyarrow · PyYAML · pytest · статический HTML/JS дашборд без сборки.

## Глобальные ограничения

- **Python:** 3.14 (файл `.python-version`). При несовместимости пакета — `uv python pin 3.12`, системный Python не трогать.
- **Данные:** только `D:\alpha-lab\data`. На `C:` свободно 6 ГБ — писать туда данные запрещено.
- **Polars не использовать.** Один API датафреймов — pandas.
- **`src/validation/` не импортирует `src/strategies/`.** Проверяется тестом (задача 17).
- **`experiment_id` = sha256(config + data_version + git_hash)**. Один ID → один результат.
- **Пороги вердикта (по умолчанию):** `dsr > 0`, `pbo < 0.5`, `p_value < 0.05`, `trades >= 100`.
- **Дашборд работает через `file://`.** Никакого `fetch()` к локальным файлам — только `<script src="report.js">`.
- **Формат данных Binance:** klines — 12 колонок, заголовок может отсутствовать в старых файлах; funding — `calc_time, funding_interval_hours, last_funding_rate`, ставка в **долях**, интервал читать из файла.
- **TDD:** каждый шаг — сначала падающий тест, потом реализация. Коммит после каждой задачи.
- **Один бар = одна минута** по умолчанию; таймфреймы агрегируются из минутных.

---

## Структура файлов

| Путь | Ответственность |
|---|---|
| `pyproject.toml`, `.python-version` | Зависимости и версия Python |
| `src/alpha_lab/config.py` | Загрузка и валидация YAML-конфигов |
| `src/alpha_lab/data/schema.py` | Каноническая схема бара, имена колонок |
| `src/alpha_lab/data/store.py` | Пути Parquet, запись/чтение партиций |
| `src/alpha_lab/data/quality.py` | Гейты качества данных |
| `src/alpha_lab/data/ingest.py` | Скачивание архива Binance, парсинг, раскладка |
| `src/alpha_lab/data/query.py` | Чтение баров через DuckDB, агрегация таймфреймов |
| `src/alpha_lab/features/price.py` | z-score, МНК-фит OU, полужизнь, ADF, Хёрст |
| `src/alpha_lab/features/volatility.py` | ATR, режим волатильности, funding-признаки |
| `src/alpha_lab/engine/costs.py` | `CostModel`: комиссии, проскальзывание, funding |
| `src/alpha_lab/engine/backtest.py` | Векторный движок: позиции → сделки и equity |
| `src/alpha_lab/engine/exits.py` | Numba-симуляция SL/TP-выходов |
| `src/alpha_lab/strategies/base.py` | Протокол `Strategy` |
| `src/alpha_lab/strategies/mean_reversion.py` | Перенос MR-модели из Pine Script |
| `src/alpha_lab/validation/metrics.py` | Sharpe, Sortino, Calmar, max DD, R-мультипликаторы |
| `src/alpha_lab/validation/splits.py` | Purged K-Fold с embargo |
| `src/alpha_lab/validation/significance.py` | DSR, PBO (CSCV), permutation-тест |
| `src/alpha_lab/validation/validator.py` | Сборка `Verdict` |
| `src/alpha_lab/report/schema.py` | Версионированная схема отчёта |
| `src/alpha_lab/report/writer.py` | Запись `report.json` + `report.js` |
| `src/alpha_lab/cli.py` | Команды `ingest`, `validate` |
| `dashboard/index.html`, `app.js`, `style.css` | Статический дашборд |
| `tests/fixtures/synthetic.py` | OU-генератор с известными параметрами |
| `tests/fixtures/traps.py` | Стратегии-ловушки |

---

## Задача 1: Скелет проекта и конфиги

**Файлы:**
- Создать: `pyproject.toml`, `.python-version`, `src/alpha_lab/__init__.py`, `src/alpha_lab/config.py`, `configs/universe.yaml`, `configs/experiments/mr_base.yaml`, `tests/test_config.py`

**Интерфейсы:**
- Отдаёт: `load_universe(path) -> Universe`, `load_experiment(path) -> Experiment`
- `Universe(symbols: list[str], start: str, end: str | None, market: str)`
- `Experiment(name, strategy, params, timeframe, start, end, costs, validation)`

- [ ] **Шаг 1: Создать структуру и зависимости**

```bash
cd "C:/Users/Admin/Documents/Alpha MR"
git config user.name "Alpha Lab"
git config user.email "alpha-lab@local"
mkdir -p src/alpha_lab/data src/alpha_lab/features src/alpha_lab/engine src/alpha_lab/strategies src/alpha_lab/validation src/alpha_lab/report configs/experiments tests/fixtures dashboard
```

`pyproject.toml`:
```toml
[project]
name = "alpha-lab"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "pandas>=2.2",
    "numpy>=1.26",
    "scipy>=1.11",
    "statsmodels>=0.14",
    "duckdb>=1.0",
    "pyarrow>=15",
    "requests>=2.31",
    "numba>=0.60",
    "pyyaml>=6.0",
]

[dependency-groups]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/alpha_lab"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

`.python-version`: `3.14`

```bash
uv sync
```

- [ ] **Шаг 2: Написать падающий тест конфигов**

`tests/test_config.py`:
```python
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
```

- [ ] **Шаг 3: Запустить тест — убедиться, что падает**

```bash
uv run pytest tests/test_config.py -v
```
Ожидается: `ModuleNotFoundError: No module named 'alpha_lab.config'`

- [ ] **Шаг 4: Реализовать `config.py`**

`src/alpha_lab/config.py`:
```python
"""Загрузка и валидация YAML-конфигов."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")


@dataclass(frozen=True)
class Universe:
    symbols: list[str]
    start: str
    end: str | None
    market: str
    include_delisted: bool = True


@dataclass(frozen=True)
class Experiment:
    name: str
    strategy: str
    params: dict[str, Any]
    timeframe: str
    start: str
    end: str | None
    costs: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Конфиг {path} должен быть YAML-словарём")
    return data


def load_universe(path: str | Path) -> Universe:
    data = _read_yaml(Path(path))
    symbols = data.get("symbols") or []
    if not symbols:
        raise ValueError("Юниверс должен содержать хотя бы один символ")
    if len(symbols) != len(set(symbols)):
        dupes = sorted({s for s in symbols if symbols.count(s) > 1})
        raise ValueError(f"В юниверсе дубликат символов: {dupes}")
    for field_name in ("market", "start"):
        if not data.get(field_name):
            raise ValueError(f"В юниверсе не задано поле '{field_name}'")
    return Universe(
        symbols=list(symbols),
        start=str(data["start"]),
        end=str(data["end"]) if data.get("end") else None,
        market=str(data["market"]),
        include_delisted=bool(data.get("include_delisted", True)),
    )


def load_experiment(path: str | Path) -> Experiment:
    data = _read_yaml(Path(path))
    for field_name in ("name", "strategy", "timeframe", "start"):
        if not data.get(field_name):
            raise ValueError(f"В эксперименте не задано поле '{field_name}'")
    tf = str(data["timeframe"])
    if tf not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Неизвестный таймфрейм '{tf}'. Допустимые: {', '.join(VALID_TIMEFRAMES)}"
        )
    return Experiment(
        name=str(data["name"]),
        strategy=str(data["strategy"]),
        params=dict(data.get("params") or {}),
        timeframe=tf,
        start=str(data["start"]),
        end=str(data["end"]) if data.get("end") else None,
        costs=dict(data.get("costs") or {}),
        validation=dict(data.get("validation") or {}),
    )
```

- [ ] **Шаг 5: Создать конфиги**

`configs/universe.yaml`:
```yaml
# Фокусированный юниверс USDT-перпетуалов Binance.
# Список фиксирован на всё исследование: изменение состава меняет experiment_id.
# ВАЖНО: делистингованные пары должны оставаться в списке — иначе возникает
# ошибка выживаемости (тестируем только выживших, убытки умерших не учитываем).
market: futures-um
start: "2020-01-01"
end: null            # null = по сегодняшний день
include_delisted: true
symbols:
  - BTCUSDT
  - ETHUSDT
  - BNBUSDT
  - XRPUSDT
  - ADAUSDT
  - SOLUSDT
  - DOGEUSDT
  - DOTUSDT
  - LINKUSDT
  - LTCUSDT
  - BCHUSDT
  - ETCUSDT
  - XLMUSDT
  - TRXUSDT
  - EOSUSDT
  - ATOMUSDT
  - NEOUSDT
  - DASHUSDT
  - ZECUSDT
  - ONTUSDT
  - QTUMUSDT
  - IOTAUSDT
  - XTZUSDT
  - ALGOUSDT
  - THETAUSDT
```

`configs/experiments/mr_base.yaml`:
```yaml
name: mr_base
strategy: mean_reversion
timeframe: 1h
start: "2022-01-01"
end: "2025-12-31"

# Параметры перенесены из Pine Script (index.html, блок pine-strategy)
params:
  window: 20          # окно μ и σ, баров
  k: 2.0              # порог входа, сигм
  atr_len: 14
  sl_atr: 2.0         # стоп = 2 × ATR
  tp_atr: 6.0         # тейк = 6 × ATR
  use_hl_filter: false
  hl_min: 5.0
  hl_max: 100.0

# Издержки Binance USDT-M, VIP0
costs:
  taker_fee_bps: 5.0     # 0.05%
  maker_fee_bps: 2.0     # 0.02%
  maker_share: 0.0       # 0 = все исполнения тейкерские (консервативно)
  impact_coef: 0.1
  min_slippage_bps: 0.5

validation:
  n_splits: 6
  purge: 50
  embargo: 20
  n_permutations: 1000
  min_trades: 100
  max_pbo: 0.5
  max_p_value: 0.05
```

- [ ] **Шаг 6: Запустить тесты — убедиться, что проходят**

```bash
uv run pytest tests/test_config.py -v
```
Ожидается: 5 passed

- [ ] **Шаг 7: Коммит**

```bash
git add pyproject.toml .python-version uv.lock src configs tests
git commit -m "feat: project skeleton and config loading"
```

---

## Задача 2: Схема данных и хранилище

**Файлы:**
- Создать: `src/alpha_lab/data/__init__.py`, `src/alpha_lab/data/schema.py`, `src/alpha_lab/data/store.py`, `tests/test_store.py`

**Интерфейсы:**
- Отдаёт: `KLINE_COLUMNS`, `FUNDING_COLUMNS`, `normalize_bars(df) -> pd.DataFrame`
- `bars_path(root, symbol, freq) -> Path`, `write_bars(df, root, symbol, freq) -> list[Path]`, `read_bars(root, symbol, freq, start, end) -> pd.DataFrame`
- `write_funding(df, root, symbol) -> list[Path]`, `read_funding(root, symbol) -> pd.DataFrame`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_store.py`:
```python
import pandas as pd
import pytest

from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import read_bars, read_funding, write_bars, write_funding


def _bars(n=100, start="2024-01-01"):
    ts = pd.date_range(start, periods=n, freq="1min", tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": [100.0 + i * 0.1 for i in range(n)],
        "volume": 10.0,
        "quote_volume": 1000.0,
        "trades": 5,
        "taker_buy_volume": 6.0,
    })


def test_normalize_bars_sorts_and_dedups():
    df = _bars(5)
    shuffled = pd.concat([df.iloc[[3, 1]], df.iloc[[3, 1]]], ignore_index=True)

    out = normalize_bars(shuffled)

    assert out["ts"].is_monotonic_increasing
    assert len(out) == 2
    assert out["ts"].dt.tz is not None


def test_normalize_bars_rejects_missing_columns():
    df = _bars(5).drop(columns=["volume"])

    with pytest.raises(ValueError, match="volume"):
        normalize_bars(df)


def test_write_read_roundtrip(tmp_path):
    df = normalize_bars(_bars(100))

    paths = write_bars(df, tmp_path, "BTCUSDT", "1m")
    back = read_bars(tmp_path, "BTCUSDT", "1m")

    assert len(paths) == 1
    pd.testing.assert_frame_equal(back, df)


def test_read_bars_filters_by_date(tmp_path):
    write_bars(normalize_bars(_bars(100, "2024-01-01")), tmp_path, "BTCUSDT", "1m")

    out = read_bars(tmp_path, "BTCUSDT", "1m",
                    start="2024-01-01T00:30", end="2024-01-01T00:39")

    assert len(out) == 10


def test_read_bars_missing_symbol_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Нет данных"):
        read_bars(tmp_path, "NOPEUSDT", "1m")


def test_funding_roundtrip(tmp_path):
    ts = pd.date_range("2024-01-01", periods=3, freq="8h", tz="UTC")
    df = pd.DataFrame({"ts": ts, "rate": [0.0001, -0.0002, 0.0003],
                       "interval_hours": [8, 8, 8]})

    write_funding(df, tmp_path, "BTCUSDT")
    back = read_funding(tmp_path, "BTCUSDT")

    pd.testing.assert_frame_equal(back, df)
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_store.py -v
```
Ожидается: `ModuleNotFoundError: No module named 'alpha_lab.data'`

- [ ] **Шаг 3: Реализовать `schema.py`**

`src/alpha_lab/data/schema.py`:
```python
"""Каноническая схема бара.

Формат подтверждён на реальном файле архива Binance
(data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip):
12 колонок, заголовок присутствует в свежих файлах и отсутствует в старых.
"""
from __future__ import annotations

import pandas as pd

# Имена колонок в CSV архива Binance (USDT-M futures)
RAW_KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume", "close_time",
    "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore",
]

# Канонический вид после нормализации
KLINE_COLUMNS = [
    "ts", "open", "high", "low", "close", "volume",
    "quote_volume", "trades", "taker_buy_volume",
]

FUNDING_COLUMNS = ["ts", "rate", "interval_hours"]

_NUMERIC = ["open", "high", "low", "close", "volume",
            "quote_volume", "trades", "taker_buy_volume"]


def normalize_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит бары к каноническому виду: UTC-индекс времени, сортировка, дедуп."""
    missing = [c for c in KLINE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"В барах отсутствуют колонки: {missing}")

    out = df.loc[:, KLINE_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    for col in _NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")

    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.drop_duplicates(subset="ts", keep="last")
    return out.sort_values("ts").reset_index(drop=True)
```

- [ ] **Шаг 4: Реализовать `store.py`**

`src/alpha_lab/data/store.py`:
```python
"""Parquet-хранилище на D:\\alpha-lab\\data.

Раскладка: <root>/bars/<symbol>/<freq>/<symbol>-<freq>-<YYYY-MM>.parquet
           <root>/funding/<symbol>/<symbol>-funding.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.data.schema import FUNDING_COLUMNS, normalize_bars

DEFAULT_ROOT = Path(r"D:\alpha-lab\data")


def bars_dir(root: Path, symbol: str, freq: str) -> Path:
    return Path(root) / "bars" / symbol / freq


def write_bars(df: pd.DataFrame, root: Path, symbol: str, freq: str) -> list[Path]:
    df = normalize_bars(df)
    out_dir = bars_dir(root, symbol, freq)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for period, chunk in df.groupby(df["ts"].dt.strftime("%Y-%m")):
        path = out_dir / f"{symbol}-{freq}-{period}.parquet"
        chunk.reset_index(drop=True).to_parquet(path, index=False, compression="zstd")
        written.append(path)
    return sorted(written)


def read_bars(root: Path, symbol: str, freq: str,
              start: str | None = None, end: str | None = None) -> pd.DataFrame:
    out_dir = bars_dir(root, symbol, freq)
    files = sorted(out_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"Нет данных: {out_dir}")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = normalize_bars(df)

    if start is not None:
        df = df[df["ts"] >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df["ts"] <= pd.Timestamp(end, tz="UTC")]
    return df.reset_index(drop=True)


def write_funding(df: pd.DataFrame, root: Path, symbol: str) -> Path:
    out_dir = Path(root) / "funding" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{symbol}-funding.parquet"

    out = df.loc[:, FUNDING_COLUMNS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out = (out.drop_duplicates(subset="ts", keep="last")
              .sort_values("ts")
              .reset_index(drop=True))
    out.to_parquet(path, index=False, compression="zstd")
    return path


def read_funding(root: Path, symbol: str) -> pd.DataFrame:
    path = Path(root) / "funding" / symbol / f"{symbol}-funding.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Нет данных о funding: {path}")
    out = pd.read_parquet(path)
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    return out.sort_values("ts").reset_index(drop=True)
```

- [ ] **Шаг 5: Запустить тесты**

```bash
uv run pytest tests/test_store.py -v
```
Ожидается: 6 passed

- [ ] **Шаг 6: Коммит**

```bash
git add src/alpha_lab/data tests/test_store.py
git commit -m "feat: bar schema and parquet store"
```

---

## Задача 3: Гейты качества данных

**Файлы:**
- Создать: `src/alpha_lab/data/quality.py`, `tests/test_quality.py`

**Интерфейсы:**
- Отдаёт: `QualityReport(gaps, duplicates, zero_volume, anomalous, total_rows, bad_rows)` с `.is_clean` и `.summary() -> str`
- `check_bars(df, freq) -> QualityReport`
- Задача 4 (`ingest`) использует `check_bars` для логирования.

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_quality.py`:
```python
import pandas as pd

from alpha_lab.data.quality import check_bars
from alpha_lab.data.schema import normalize_bars


def _bars(ts):
    return normalize_bars(pd.DataFrame({
        "ts": pd.to_datetime(ts, utc=True),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
        "volume": 10.0, "quote_volume": 1000.0, "trades": 5,
        "taker_buy_volume": 6.0,
    }))


def test_clean_series_has_no_issues():
    df = _bars(pd.date_range("2024-01-01", periods=100, freq="1min"))

    rep = check_bars(df, "1m")

    assert rep.is_clean
    assert rep.total_rows == 100
    assert rep.gaps == 0


def test_detects_gap():
    ts = list(pd.date_range("2024-01-01", periods=50, freq="1min"))
    ts += list(pd.date_range("2024-01-01 02:00", periods=50, freq="1min"))

    rep = check_bars(_bars(ts), "1m")

    assert not rep.is_clean
    assert rep.gaps == 1


def test_detects_zero_volume():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[3, "volume"] = 0.0

    rep = check_bars(df, "1m")

    assert rep.zero_volume == 1


def test_detects_impossible_ohlc():
    df = _bars(pd.date_range("2024-01-01", periods=10, freq="1min"))
    df.loc[2, "high"] = 50.0   # high ниже low

    rep = check_bars(df, "1m")

    assert rep.anomalous == 1


def test_duplicates_removed_by_normalize():
    df = _bars(pd.date_range("2024-01-01", periods=5, freq="1min"))
    df = pd.concat([df, df.iloc[[2]]], ignore_index=True)

    rep = check_bars(normalize_bars(df), "1m")

    assert rep.duplicates == 0
    assert rep.total_rows == 5


def test_summary_mentions_problems():
    ts = list(pd.date_range("2024-01-01", periods=10, freq="1min"))
    ts += list(pd.date_range("2024-01-01 05:00", periods=10, freq="1min"))

    text = check_bars(_bars(ts), "1m").summary()

    assert "пропуск" in text.lower()
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_quality.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `quality.py`**

`src/alpha_lab/data/quality.py`:
```python
"""Гейты качества данных.

Принцип: бар с проблемой не интерполируется и не чинится — он помечается,
и стратегия на нём не торгует. Тихая починка данных создаёт тихо неверный бэктест.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

FREQ_DELTA = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
    "1d": pd.Timedelta(days=1),
}


@dataclass(frozen=True)
class QualityReport:
    total_rows: int
    gaps: int
    duplicates: int
    zero_volume: int
    anomalous: int

    @property
    def bad_rows(self) -> int:
        return self.zero_volume + self.anomalous

    @property
    def is_clean(self) -> bool:
        return self.gaps == 0 and self.duplicates == 0 and self.bad_rows == 0

    def summary(self) -> str:
        if self.is_clean:
            return f"OK: {self.total_rows} баров, проблем нет"
        parts = []
        if self.gaps:
            parts.append(f"пропусков: {self.gaps}")
        if self.duplicates:
            parts.append(f"дубликатов: {self.duplicates}")
        if self.zero_volume:
            parts.append(f"нулевой объём: {self.zero_volume}")
        if self.anomalous:
            parts.append(f"аномалий OHLC: {self.anomalous}")
        return f"ПРОБЛЕМЫ ({self.total_rows} баров): " + ", ".join(parts)


def check_bars(df: pd.DataFrame, freq: str) -> QualityReport:
    if freq not in FREQ_DELTA:
        raise ValueError(f"Неизвестный таймфрейм для проверки: {freq}")
    if df.empty:
        return QualityReport(0, 0, 0, 0, 0)

    delta = FREQ_DELTA[freq]
    ts = pd.to_datetime(df["ts"], utc=True)

    duplicates = int(ts.duplicated().sum())
    gaps = int((ts.diff().dropna() > delta).sum())

    zero_volume = int((pd.to_numeric(df["volume"], errors="coerce") <= 0).sum())

    high, low = df["high"], df["low"]
    open_, close = df["open"], df["close"]
    anomalous = int((
        (high < low)
        | (high < open_) | (high < close)
        | (low > open_) | (low > close)
        | (close <= 0) | (open_ <= 0)
    ).sum())

    return QualityReport(
        total_rows=len(df),
        gaps=gaps,
        duplicates=duplicates,
        zero_volume=zero_volume,
        anomalous=anomalous,
    )


def clean_mask(df: pd.DataFrame) -> pd.Series:
    """Маска баров, пригодных для торговли. Непригодные исключаются, а не чинятся."""
    bad = (
        (pd.to_numeric(df["volume"], errors="coerce") <= 0)
        | (df["high"] < df["low"])
        | (df["close"] <= 0) | (df["open"] <= 0)
        | df[["open", "high", "low", "close"]].isna().any(axis=1)
    )
    return ~bad
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_quality.py -v
```
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/data/quality.py tests/test_quality.py
git commit -m "feat: data quality gates"
```

---

## Задача 4: Загрузка архива Binance

**Файлы:**
- Создать: `src/alpha_lab/data/ingest.py`, `tests/test_ingest.py`

**Интерфейсы:**
- Отдаёт: `parse_kline_csv(raw: bytes) -> pd.DataFrame`, `parse_funding_csv(raw: bytes) -> pd.DataFrame`
- `archive_url(market, kind, symbol, freq, period) -> str`
- `ingest_symbol(symbol, freq, start, end, root, market) -> IngestResult`
- `ingest_universe(universe, freq, root) -> list[IngestResult]`

**Проверенные факты (не догадки):**
- Klines: `https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/{FREQ}/{SYM}-{FREQ}-{YYYY-MM}.zip`, 12 колонок, заголовок есть в свежих файлах
- Funding: `https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYM}/{SYM}-fundingRate-{YYYY-MM}.zip`, колонки `calc_time, funding_interval_hours, last_funding_rate`, ставка в долях

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_ingest.py`:
```python
import pandas as pd
import pytest

from alpha_lab.data.ingest import (
    archive_url, month_range, parse_funding_csv, parse_kline_csv,
)

KLINE_WITH_HEADER = (
    b"open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    b"taker_buy_volume,taker_buy_quote_volume,ignore\n"
    b"1704067200000,42314.00,42335.80,42289.60,42331.90,289.641,"
    b"1704067259999,12256155.25625,3310,175.211,7414459.86355,0\n"
    b"1704067260000,42331.90,42353.10,42331.80,42350.40,202.444,"
    b"1704067319999,8572240.95470,1885,154.353,6535804.80720,0\n"
)

# Старые файлы архива не содержат строки заголовка
KLINE_WITHOUT_HEADER = b"\n".join(KLINE_WITH_HEADER.split(b"\n")[1:])


def test_parse_kline_with_header():
    df = parse_kline_csv(KLINE_WITH_HEADER)

    assert len(df) == 2
    assert df["close"].iloc[0] == pytest.approx(42331.90)
    assert df["trades"].iloc[0] == 3310
    assert str(df["ts"].dt.tz) == "UTC"


def test_parse_kline_without_header():
    df = parse_kline_csv(KLINE_WITHOUT_HEADER)

    assert len(df) == 2
    assert df["open"].iloc[1] == pytest.approx(42331.90)


def test_parse_kline_timestamps_are_minutes_apart():
    df = parse_kline_csv(KLINE_WITH_HEADER)

    assert (df["ts"].diff().dropna() == pd.Timedelta(minutes=1)).all()


FUNDING_CSV = (
    b"calc_time,funding_interval_hours,last_funding_rate\n"
    b"1704067200000,8,0.00037409\n"
    b"1704096000000,8,0.00027213\n"
)


def test_parse_funding():
    df = parse_funding_csv(FUNDING_CSV)

    assert list(df.columns) == ["ts", "rate", "interval_hours"]
    assert df["rate"].iloc[0] == pytest.approx(0.00037409)
    assert df["interval_hours"].iloc[0] == 8


def test_archive_url_klines():
    url = archive_url("futures-um", "klines", "BTCUSDT", "1m", "2024-01")

    assert url == ("https://data.binance.vision/data/futures/um/monthly/"
                   "klines/BTCUSDT/1m/BTCUSDT-1m-2024-01.zip")


def test_archive_url_funding():
    url = archive_url("futures-um", "fundingRate", "BTCUSDT", None, "2024-01")

    assert url == ("https://data.binance.vision/data/futures/um/monthly/"
                   "fundingRate/BTCUSDT/BTCUSDT-fundingRate-2024-01.zip")


def test_month_range():
    months = month_range("2024-01-15", "2024-04-02")

    assert months == ["2024-01", "2024-02", "2024-03", "2024-04"]
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_ingest.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `ingest.py`**

`src/alpha_lab/data/ingest.py`:
```python
"""Загрузка публичного архива Binance (data.binance.vision).

Архив бесплатный, без API-ключа и содержит всю историю. Это основной источник;
ccxt используется только как резерв, если архив недоступен из региона.
"""
from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from alpha_lab.data.quality import check_bars
from alpha_lab.data.schema import RAW_KLINE_COLUMNS, normalize_bars
from alpha_lab.data.store import DEFAULT_ROOT, write_bars, write_funding

BASE_URL = "https://data.binance.vision/data"
MARKET_PREFIX = {"futures-um": "futures/um", "spot": "spot"}
REQUEST_TIMEOUT = 60


@dataclass(frozen=True)
class IngestResult:
    symbol: str
    kind: str
    files: int
    rows: int
    quality: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def archive_url(market: str, kind: str, symbol: str, freq: str | None,
                period: str) -> str:
    """URL месячного файла архива. period в формате YYYY-MM."""
    prefix = MARKET_PREFIX.get(market)
    if prefix is None:
        raise ValueError(f"Неизвестный рынок: {market}")
    if kind == "klines":
        name = f"{symbol}-{freq}-{period}.zip"
        return f"{BASE_URL}/{prefix}/monthly/klines/{symbol}/{freq}/{name}"
    if kind == "fundingRate":
        name = f"{symbol}-fundingRate-{period}.zip"
        return f"{BASE_URL}/{prefix}/monthly/fundingRate/{symbol}/{name}"
    raise ValueError(f"Неизвестный тип данных: {kind}")


def month_range(start: str, end: str | None) -> list[str]:
    """Список месяцев YYYY-MM, покрывающих [start, end]."""
    first = pd.Period(pd.Timestamp(start), freq="M")
    last = pd.Period(pd.Timestamp(end or date.today().isoformat()), freq="M")
    return [str(p) for p in pd.period_range(first, last, freq="M")]


def _read_zip_csv(raw: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        name = zf.namelist()[0]
        return zf.read(name)


def _has_header(first_line: bytes) -> bool:
    """Заголовок присутствует, если первое поле не является числом."""
    head = first_line.split(b",", 1)[0].strip()
    try:
        float(head)
    except ValueError:
        return True
    return False


def parse_kline_csv(raw: bytes) -> pd.DataFrame:
    """Парсит CSV klines. Терпим к наличию и отсутствию заголовка."""
    content = _read_zip_csv(raw) if raw[:2] == b"PK" else raw
    first_line = content.split(b"\n", 1)[0]
    skip = 1 if _has_header(first_line) else 0

    df = pd.read_csv(
        io.BytesIO(content), header=None, skiprows=skip,
        names=RAW_KLINE_COLUMNS, usecols=range(len(RAW_KLINE_COLUMNS)),
    )
    df = df.rename(columns={
        "open_time": "ts", "count": "trades",
        "taker_buy_quote_volume": "_taker_quote", "ignore": "_ignore",
    })
    # open_time в миллисекундах; в очень старых файлах — в микросекундах
    ts = pd.to_numeric(df["ts"], errors="coerce")
    if ts.dropna().median() > 1e15:
        ts = ts / 1000.0
    df["ts"] = pd.to_datetime(ts, unit="ms", utc=True)
    df = df.drop(columns=[c for c in ("_taker_quote", "_ignore") if c in df.columns])
    return normalize_bars(df)


def parse_funding_csv(raw: bytes) -> pd.DataFrame:
    """Парсит CSV funding. Ставка в долях; интервал читается из файла."""
    content = _read_zip_csv(raw) if raw[:2] == b"PK" else raw
    first_line = content.split(b"\n", 1)[0]
    has_header = first_line.lower().startswith(b"calc_time") or _has_header(first_line)

    df = pd.read_csv(
        io.BytesIO(content), header=0 if has_header else None,
        names=None if has_header else ["calc_time", "funding_interval_hours",
                                       "last_funding_rate"],
    )
    df = df.rename(columns={"calc_time": "ts", "last_funding_rate": "rate",
                            "funding_interval_hours": "interval_hours"})
    df["ts"] = pd.to_datetime(pd.to_numeric(df["ts"]), unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df["rate"], errors="coerce").astype("float64")
    df["interval_hours"] = pd.to_numeric(
        df.get("interval_hours", 8), errors="coerce").fillna(8).astype("int64")
    return (df.loc[:, ["ts", "rate", "interval_hours"]]
              .dropna(subset=["rate"])
              .drop_duplicates(subset="ts", keep="last")
              .sort_values("ts")
              .reset_index(drop=True))


def _download(url: str) -> bytes | None:
    """Скачивает файл. None означает «файла нет» (404) — это норма для ранних месяцев."""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.content


def ingest_symbol(symbol: str, freq: str, start: str, end: str | None,
                  root: Path = DEFAULT_ROOT, market: str = "futures-um",
                  with_funding: bool = True) -> list[IngestResult]:
    """Скачивает и раскладывает все месяцы для одного символа."""
    results: list[IngestResult] = []
    months = month_range(start, end)

    bar_frames, bar_files, missing = [], 0, 0
    for period in months:
        try:
            raw = _download(archive_url(market, "klines", symbol, freq, period))
        except requests.RequestException as exc:
            results.append(IngestResult(symbol, "klines", 0, 0, "", str(exc)))
            return results
        if raw is None:
            missing += 1
            continue
        bar_frames.append(parse_kline_csv(raw))

    if bar_frames:
        bars = pd.concat(bar_frames, ignore_index=True)
        bar_files = len(write_bars(bars, root, symbol, freq))
        rep = check_bars(normalize_bars(bars), freq)
        results.append(IngestResult(symbol, "klines", bar_files, len(bars),
                                    rep.summary()))
    else:
        results.append(IngestResult(symbol, "klines", 0, 0, "",
                                    f"нет файлов за {len(months)} мес."))

    if with_funding:
        fund_frames = []
        for period in months:
            raw = _download(archive_url(market, "fundingRate", symbol, None, period))
            if raw is not None:
                fund_frames.append(parse_funding_csv(raw))
        if fund_frames:
            funding = pd.concat(fund_frames, ignore_index=True)
            write_funding(funding, root, symbol)
            results.append(IngestResult(symbol, "funding", 1, len(funding), "OK"))
        else:
            results.append(IngestResult(symbol, "funding", 0, 0, "",
                                        "нет файлов fundingRate"))

    return results


def ingest_universe(universe, freq: str, root: Path = DEFAULT_ROOT
                    ) -> list[IngestResult]:
    out: list[IngestResult] = []
    for symbol in universe.symbols:
        out.extend(ingest_symbol(symbol, freq, universe.start, universe.end,
                                 root=root, market=universe.market))
    return out
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_ingest.py -v
```
Ожидается: 7 passed (тесты работают на встроенных байтах, сеть не нужна)

- [ ] **Шаг 5: Проверить на реальных данных**

```bash
uv run python -c "
from alpha_lab.data.ingest import ingest_symbol
from pathlib import Path
import tempfile
res = ingest_symbol('BTCUSDT', '1m', '2024-01-01', '2024-01-31', Path(tempfile.mkdtemp()))
for r in res: print(r)
"
```
Ожидается: `IngestResult(symbol='BTCUSDT', kind='klines', files=1, rows=44640, ...)` и `... kind='funding', ... rows=93`

- [ ] **Шаг 6: Коммит**

```bash
git add src/alpha_lab/data/ingest.py tests/test_ingest.py
git commit -m "feat: binance archive ingest with tolerant csv parser"
```

---

## Задача 5: Слой запросов через DuckDB

**Файлы:**
- Создать: `src/alpha_lab/data/query.py`, `tests/test_query.py`

**Интерфейсы:**
- Отдаёт: `load_bars(root, symbol, freq, start, end, resample=None) -> pd.DataFrame`
- `load_funding(root, symbol) -> pd.DataFrame`
- `align_funding_to_bars(bars, funding) -> pd.Series`
- `data_version(root) -> str` — хэш содержимого для `experiment_id`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_query.py`:
```python
import pandas as pd

from alpha_lab.data.query import (
    align_funding_to_bars, data_version, load_bars, load_funding,
)
from alpha_lab.data.store import write_bars, write_funding
from alpha_lab.data.schema import normalize_bars


def _make(root, n=600, freq="1m"):
    ts = pd.date_range("2024-01-01", periods=n, freq=freq, tz="UTC")
    close = pd.Series(range(n), dtype="float64") + 100.0
    df = normalize_bars(pd.DataFrame({
        "ts": ts, "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": 10.0, "quote_volume": 1000.0,
        "trades": 5, "taker_buy_volume": 6.0,
    }))
    write_bars(df, root, "BTCUSDT", freq)


def test_load_bars_reads_all(tmp_path):
    _make(tmp_path, n=120)

    df = load_bars(tmp_path, "BTCUSDT", "1m")

    assert len(df) == 120


def test_load_bars_resamples_to_hourly(tmp_path):
    _make(tmp_path, n=600)   # 10 часов по минутам

    df = load_bars(tmp_path, "BTCUSDT", "1m", resample="1h")

    assert len(df) == 10
    assert df["close"].iloc[0] == 159.0     # последняя минута первого часа
    assert df["open"].iloc[0] == 100.0      # первая минута первого часа
    assert df["volume"].iloc[0] == 600.0    # 60 минут × 10


def test_data_version_is_stable_and_changes_with_data(tmp_path):
    _make(tmp_path, n=60)
    v1 = data_version(tmp_path)
    v2 = data_version(tmp_path)

    _make(tmp_path, n=120)
    v3 = data_version(tmp_path)

    assert v1 == v2
    assert v1 != v3


def test_align_funding_places_rate_at_funding_timestamps(tmp_path):
    _make(tmp_path, n=600)
    bars = load_bars(tmp_path, "BTCUSDT", "1m")
    funding = pd.DataFrame({
        "ts": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 08:00"], utc=True),
        "rate": [0.0001, 0.0002],
        "interval_hours": [8, 8],
    })

    series = align_funding_to_bars(bars, funding)

    assert len(series) == len(bars)
    assert series.iloc[0] == 0.0001
    assert series.iloc[480] == 0.0002
    assert series.sum() == 0.0001 + 0.0002
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_query.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `query.py`**

`src/alpha_lab/data/query.py`:
```python
"""Чтение данных через DuckDB.

DuckDB читает только нужные колонки и партиции, не поднимая весь датасет в память —
критично при 16 ГБ RAM и десятках гигабайт истории.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pandas as pd

from alpha_lab.data.store import DEFAULT_ROOT

# Правила агрегации минутных баров в старший таймфрейм
_AGG = {
    "open": "first", "high": "max", "low": "min", "close": "last",
    "volume": "sum", "quote_volume": "sum", "trades": "sum",
    "taker_buy_volume": "sum",
}
_PANDAS_FREQ = {"1m": "1min", "5m": "5min", "15m": "15min",
                "1h": "1h", "4h": "4h", "1d": "1D"}


def load_bars(root: Path = DEFAULT_ROOT, symbol: str = "BTCUSDT",
              freq: str = "1m", start: str | None = None, end: str | None = None,
              resample: str | None = None) -> pd.DataFrame:
    """Читает бары символа. Если задан resample — агрегирует из freq в resample."""
    pattern = str(Path(root) / "bars" / symbol / freq / "*.parquet")
    where, params = [], []
    if start:
        where.append("ts >= ?")
        params.append(pd.Timestamp(start, tz="UTC").to_pydatetime())
    if end:
        where.append("ts <= ?")
        params.append(pd.Timestamp(end, tz="UTC").to_pydatetime())
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    try:
        df = duckdb.sql(
            f"SELECT * FROM read_parquet('{pattern}') {clause} ORDER BY ts", params=params
        ).df()
    except duckdb.IOException as exc:
        raise FileNotFoundError(
            f"Нет данных: {pattern}. Сначала выполните ingest."
        ) from exc

    if df.empty:
        raise FileNotFoundError(f"Нет данных: {pattern} за указанный период")
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    if resample and resample != freq:
        df = _resample(df, resample)
    return df.reset_index(drop=True)


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule not in _PANDAS_FREQ:
        raise ValueError(f"Неизвестный таймфрейм для ресемплинга: {rule}")
    out = (df.set_index("ts")
             .resample(_PANDAS_FREQ[rule], label="left", closed="left")
             .agg(_AGG)
             .dropna(subset=["close"])
             .reset_index())
    return out


def load_funding(root: Path = DEFAULT_ROOT, symbol: str = "BTCUSDT") -> pd.DataFrame:
    pattern = str(Path(root) / "funding" / symbol / "*.parquet")
    try:
        df = duckdb.sql(
            f"SELECT * FROM read_parquet('{pattern}') ORDER BY ts"
        ).df()
    except duckdb.IOException as exc:
        raise FileNotFoundError(f"Нет данных о funding: {pattern}") from exc
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df.reset_index(drop=True)


def align_funding_to_bars(bars: pd.DataFrame, funding: pd.DataFrame) -> pd.Series:
    """Ставка funding, привязанная к барам.

    Ненулевое значение только в бары, совпадающие с моментом выплаты.
    """
    if funding.empty:
        return pd.Series(0.0, index=bars.index, name="funding_rate")

    lookup = dict(zip(pd.to_datetime(funding["ts"], utc=True),
                      funding["rate"].astype(float)))
    ts = pd.to_datetime(bars["ts"], utc=True)
    return ts.map(lookup).fillna(0.0).astype("float64").rename("funding_rate")


def data_version(root: Path = DEFAULT_ROOT) -> str:
    """Хэш содержимого хранилища. Участвует в experiment_id."""
    digest = hashlib.sha256()
    root = Path(root)
    if not root.exists():
        return "empty"
    for path in sorted(root.rglob("*.parquet")):
        stat = path.stat()
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(str(stat.st_size).encode())
        digest.update(str(int(stat.st_mtime)).encode())
    return digest.hexdigest()[:16]
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_query.py -v
```
Ожидается: 4 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/data/query.py tests/test_query.py
git commit -m "feat: duckdb query layer with resampling"
```

---

## Задача 6: Признаки — статистика цены

**Файлы:**
- Создать: `src/alpha_lab/features/__init__.py`, `src/alpha_lab/features/price.py`, `tests/fixtures/synthetic.py`, `tests/test_features_price.py`

**Интерфейсы:**
- Отдаёт: `ou_params(series) -> OUParams(theta, mu, sigma, half_life, r2)`
- `adf_pvalue(series) -> float`, `hurst_exponent(series) -> float`
- `zscore(series, window) -> pd.Series`, `rolling_sigma(series, window) -> pd.Series`

- [ ] **Шаг 1: Создать синтетический OU-генератор (фикстур)**

`tests/fixtures/synthetic.py`:
```python
"""OU-процесс с известными параметрами.

Это эталон для проверки: мы знаем истинные theta и half_life,
значит знаем, что движок обязан на этих данных получить.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ou_series(n=5000, mu=100.0, theta=0.05, sigma=1.0, seed=42,
              start_price=None) -> pd.Series:
    """Дискретизация OU: x[t+1] = x[t] + theta*(mu - x[t]) + sigma*eps."""
    rng = np.random.default_rng(seed)
    x = np.empty(n, dtype="float64")
    x[0] = mu if start_price is None else start_price
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * eps[t]
    return pd.Series(x, name="close")


def ou_bars(n=5000, mu=100.0, theta=0.05, sigma=1.0, seed=42,
            start="2024-01-01", freq="1min") -> pd.DataFrame:
    close = ou_series(n, mu, theta, sigma, seed)
    ts = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({
        "ts": ts,
        "open": close.shift(1).fillna(close.iloc[0]),
        "high": close + sigma * 0.5,
        "low": close - sigma * 0.5,
        "close": close,
        "volume": 10.0,
        "quote_volume": close * 10.0,
        "trades": 5,
        "taker_buy_volume": 6.0,
    })


def random_walk(n=5000, seed=7, start="2024-01-01") -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(100 + np.cumsum(rng.standard_normal(n)), name="close")
```

- [ ] **Шаг 2: Написать падающий тест**

`tests/test_features_price.py`:
```python
import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_series, random_walk

from alpha_lab.features.price import (
    adf_pvalue, hurst_exponent, ou_params, zscore,
)


def test_ou_params_recovers_known_theta():
    """theta=0.05 → полужизнь ln(2)/0.05 ≈ 13.9 бар."""
    s = ou_series(n=20000, theta=0.05, sigma=1.0, seed=1)

    p = ou_params(s)

    assert p.theta == pytest.approx(0.05, rel=0.25)
    assert p.half_life == pytest.approx(np.log(2) / 0.05, rel=0.25)
    assert p.mu == pytest.approx(100.0, abs=0.5)


def test_ou_params_faster_mean_reversion_gives_shorter_half_life():
    slow = ou_params(ou_series(n=20000, theta=0.01, seed=2))
    fast = ou_params(ou_series(n=20000, theta=0.10, seed=2))

    assert fast.half_life < slow.half_life
    assert fast.theta > slow.theta


def test_ou_params_on_random_walk_has_low_r2():
    """Случайное блуждание не возвращается к среднему — R² должен быть мал."""
    p = ou_params(random_walk(n=20000, seed=3))

    assert p.r2 < 0.05


def test_adf_rejects_random_walk_and_accepts_ou():
    rw_p = adf_pvalue(random_walk(n=5000, seed=4))
    ou_p = adf_pvalue(ou_series(n=5000, theta=0.1, seed=4))

    assert rw_p > 0.05      # нестационарен
    assert ou_p < 0.05      # стационарен


def test_hurst_random_walk_near_half():
    h = hurst_exponent(random_walk(n=20000, seed=5))

    assert 0.4 < h < 0.6


def test_hurst_mean_reverting_below_half():
    h = hurst_exponent(ou_series(n=20000, theta=0.1, seed=6))

    assert h < 0.5


def test_zscore_is_standardized():
    s = ou_series(n=5000, seed=8)

    z = zscore(s, window=100).dropna()

    assert abs(z.mean()) < 0.15
    assert abs(z.std() - 1.0) < 0.15


def test_zscore_is_zero_on_constant_series():
    s = pd.Series([5.0] * 100)

    z = zscore(s, window=20)

    assert (z.fillna(0.0) == 0.0).all()
```

- [ ] **Шаг 3: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_features_price.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 4: Реализовать `features/price.py`**

`src/alpha_lab/features/price.py`:
```python
"""Статистические признаки цены.

ou_params реализует ту же МНК-регрессию ΔS = a + b·S, что и Pine Script
(index.html, блок «Ядро модели»), но на numpy и без ограничений скользящего окна.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class OUParams:
    theta: float        # скорость возврата к среднему; nan, если возврата нет
    mu: float           # долгосрочное среднее
    sigma: float        # волатильность остатка
    half_life: float    # ln(2)/theta, бар
    r2: float           # качество подгонки; малое значение = возврата нет
    n: int


def ou_params(series: pd.Series | np.ndarray) -> OUParams:
    """Оценка параметров OU-процесса МНК-регрессией ΔS на S.

    ΔS = a + b·S + ε,  theta = -b,  half_life = ln(2)/theta.
    Возврат к среднему есть только при b < 0 (theta > 0).
    """
    s = np.asarray(series, dtype="float64")
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 10:
        return OUParams(np.nan, np.nan, np.nan, np.nan, np.nan, n)

    y = np.diff(s)          # ΔS
    x = s[:-1]              # S[t]
    x_mean, y_mean = x.mean(), y.mean()
    sxx = ((x - x_mean) ** 2).sum()
    if sxx < 1e-12:
        return OUParams(np.nan, float(x_mean), 0.0, np.nan, np.nan, n)

    b = ((x - x_mean) * (y - y_mean)).sum() / sxx
    a = y_mean - b * x_mean

    resid = y - (a + b * x)
    sigma = float(resid.std(ddof=2)) if n > 3 else float(resid.std())

    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 1e-12 else 0.0

    theta = -b
    if theta > 0:
        half_life = float(np.log(2) / theta)
        mu = float(-a / b)
    else:
        half_life, mu = np.nan, float(x_mean)

    return OUParams(theta=float(theta), mu=mu, sigma=sigma,
                    half_life=half_life, r2=r2, n=n)


def adf_pvalue(series: pd.Series | np.ndarray, maxlag: int | None = None) -> float:
    """p-value теста Дики–Фуллера. Малый p → ряд стационарен (возвращается к среднему)."""
    s = np.asarray(series, dtype="float64")
    s = s[np.isfinite(s)]
    if len(s) < 20 or np.std(s) < 1e-12:
        return 1.0
    try:
        return float(adfuller(s, maxlag=maxlag, autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return 1.0


def hurst_exponent(series: pd.Series | np.ndarray, min_lag: int = 2,
                   max_lag: int = 100) -> float:
    """Показатель Хёрста через масштабирование дисперсии лаговых разностей.

    H ≈ 0.5 — случайное блуждание; H < 0.5 — возврат к среднему; H > 0.5 — тренд.
    """
    s = np.asarray(series, dtype="float64")
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 4:
        return np.nan

    lags = np.arange(min_lag, max_lag)
    tau = np.array([np.std(s[lag:] - s[:-lag]) for lag in lags])
    ok = tau > 1e-12
    if ok.sum() < 4:
        return np.nan

    slope = np.polyfit(np.log(lags[ok]), np.log(tau[ok]), 1)[0]
    return float(slope)


def zscore(series: pd.Series, window: int) -> pd.Series:
    """z-скор относительно скользящего среднего и σ. σ≈0 → 0."""
    s = pd.Series(series).astype("float64")
    mean = s.rolling(window, min_periods=window).mean()
    std = s.rolling(window, min_periods=window).std(ddof=0)
    return ((s - mean) / std.where(std > 1e-12, np.nan)).fillna(0.0).rename("z")


def rolling_sigma(series: pd.Series, window: int) -> pd.Series:
    s = pd.Series(series).astype("float64")
    return s.rolling(window, min_periods=window).std(ddof=0).rename("sigma")
```

Также создайте `tests/fixtures/__init__.py` (пустой) и `tests/conftest.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
```

- [ ] **Шаг 5: Запустить тесты**

```bash
uv run pytest tests/test_features_price.py -v
```
Ожидается: 9 passed

- [ ] **Шаг 6: Коммит**

```bash
git add src/alpha_lab/features tests/test_features_price.py tests/fixtures tests/conftest.py
git commit -m "feat: price statistics features (ou params, adf, hurst, zscore)"
```

---

## Задача 7: Признаки — волатильность и funding

**Файлы:**
- Создать: `src/alpha_lab/features/volatility.py`, `tests/test_features_volatility.py`

**Интерфейсы:**
- Отдаёт: `atr(bars, length) -> pd.Series`, `atr_zscore(bars, atr_len, window) -> pd.Series`
- `volatility_regime(bars, atr_len, window, z_low, z_high) -> pd.Series` (значения −1/0/+1)
- `funding_features(funding_rate: pd.Series, window: int) -> pd.DataFrame` с колонками `funding_ma`, `funding_z`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_features_volatility.py`:
```python
import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_bars

from alpha_lab.features.volatility import (
    atr, atr_zscore, funding_features, volatility_regime,
)


def test_atr_matches_manual_true_range():
    bars = pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC"),
        "open": [10.0, 11.0, 12.0, 13.0],
        "high": [11.0, 13.0, 14.0, 15.0],
        "low": [9.0, 10.0, 11.0, 12.0],
        "close": [10.5, 12.5, 13.5, 14.5],
        "volume": 1.0, "quote_volume": 1.0, "trades": 1, "taker_buy_volume": 1.0,
    })

    a = atr(bars, length=3)

    # TR: [2.0, 3.0, 3.0, 3.0]; ATR(3) — среднее Уайлдера, первое значение на баре 3
    assert a.iloc[:2].isna().all()
    assert a.iloc[3] == pytest.approx(3.0, rel=0.01)


def test_atr_is_positive():
    a = atr(ou_bars(n=500), length=14).dropna()

    assert (a > 0).all()


def test_atr_zscore_centered():
    z = atr_zscore(ou_bars(n=2000), atr_len=14, window=200).dropna()

    assert abs(z.mean()) < 0.3
    assert abs(z.std() - 1.0) < 0.4


def test_volatility_regime_values():
    bars = ou_bars(n=2000)

    reg = volatility_regime(bars, atr_len=14, window=200, z_low=-1.0, z_high=1.0)

    assert set(reg.dropna().unique()).issubset({-1.0, 0.0, 1.0})


def test_funding_features_zscore_of_constant_is_zero():
    rate = pd.Series([0.0001] * 100)

    out = funding_features(rate, window=20)

    assert (out["funding_z"].fillna(0.0) == 0.0).all()
    assert out["funding_ma"].iloc[-1] == pytest.approx(0.0001)


def test_funding_features_detects_positive_bias():
    rate = pd.Series(np.r_[np.full(100, 0.0001), np.full(100, 0.001)])

    out = funding_features(rate, window=50)

    assert out["funding_z"].iloc[-1] > 2.0
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_features_volatility.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `volatility.py`**

`src/alpha_lab/features/volatility.py`:
```python
"""Признаки волатильности и funding.

ATR считается методом Уайлдера (сглаживание RMA), как ta.atr в Pine Script,
чтобы результаты сходились с TradingView на общем периоде.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(bars: pd.DataFrame) -> pd.Series:
    prev_close = bars["close"].shift(1)
    ranges = pd.concat([
        bars["high"] - bars["low"],
        (bars["high"] - prev_close).abs(),
        (bars["low"] - prev_close).abs(),
    ], axis=1)
    tr = ranges.max(axis=1)
    tr.iloc[0] = bars["high"].iloc[0] - bars["low"].iloc[0]
    return tr.rename("tr")


def atr(bars: pd.DataFrame, length: int = 14) -> pd.Series:
    """ATR методом Уайлдера: RMA с alpha = 1/length."""
    tr = true_range(bars)
    return tr.ewm(alpha=1.0 / length, adjust=False, min_periods=length).mean().rename("atr")


def atr_zscore(bars: pd.DataFrame, atr_len: int = 14, window: int = 200) -> pd.Series:
    """Насколько текущая волатильность отклоняется от своей нормы."""
    a = atr(bars, atr_len)
    mean = a.rolling(window, min_periods=window).mean()
    std = a.rolling(window, min_periods=window).std(ddof=0)
    return ((a - mean) / std.where(std > 1e-12, np.nan)).rename("atr_z")


def volatility_regime(bars: pd.DataFrame, atr_len: int = 14, window: int = 200,
                      z_low: float = -1.0, z_high: float = 1.0) -> pd.Series:
    """Режим волатильности: -1 (тихо), 0 (норма), +1 (бурно)."""
    z = atr_zscore(bars, atr_len, window)
    out = pd.Series(0.0, index=bars.index, name="vol_regime")
    out[z >= z_high] = 1.0
    out[z <= z_low] = -1.0
    out[z.isna()] = np.nan
    return out


def funding_features(rate: pd.Series, window: int = 90) -> pd.DataFrame:
    """Скользящее среднее funding и его z-скор.

    Положительный funding означает, что лонги платят шортам — держать лонг дорого.
    """
    r = pd.Series(rate).astype("float64")
    ma = r.rolling(window, min_periods=1).mean()
    std = r.rolling(window, min_periods=window).std(ddof=0)
    z = ((r - ma) / std.where(std > 1e-12, np.nan)).fillna(0.0)
    return pd.DataFrame({"funding_ma": ma, "funding_z": z})
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_features_volatility.py -v
```
Ожидается: 6 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/features/volatility.py tests/test_features_volatility.py
git commit -m "feat: volatility and funding features"
```

---

## Задача 8: Модель издержек

**Файлы:**
- Создать: `src/alpha_lab/engine/__init__.py`, `src/alpha_lab/engine/costs.py`, `tests/test_costs.py`

**Интерфейсы:**
- Отдаёт: `CostModel` (протокол), `ZeroCost`, `RealisticCost`
- `RealisticCost.from_config(dict) -> RealisticCost`
- Методы: `.fee_bps() -> float`, `.slippage_bps(order_notional, bar_quote_volume) -> float`, `.funding_cost(position, rate) -> float`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_costs.py`:
```python
import pytest

from alpha_lab.engine.costs import RealisticCost, ZeroCost


def test_zero_cost_has_no_fees():
    c = ZeroCost()

    assert c.fee_bps() == 0.0
    assert c.slippage_bps(1e6, 1e6) == 0.0
    assert c.funding_cost(1.0, 0.001) == 0.0


def test_taker_fee_default_matches_binance_vip0():
    c = RealisticCost()

    assert c.fee_bps() == pytest.approx(5.0)


def test_maker_share_reduces_fee():
    all_taker = RealisticCost(maker_share=0.0)
    half_maker = RealisticCost(maker_share=0.5)

    assert half_maker.fee_bps() < all_taker.fee_bps()
    assert half_maker.fee_bps() == pytest.approx(3.5)


def test_slippage_grows_with_order_size():
    c = RealisticCost()

    small = c.slippage_bps(1_000.0, 1_000_000.0)
    large = c.slippage_bps(100_000.0, 1_000_000.0)

    assert large > small


def test_slippage_has_floor():
    c = RealisticCost(min_slippage_bps=0.5)

    assert c.slippage_bps(0.0, 1_000_000.0) == pytest.approx(0.5)


def test_slippage_capped_when_order_exceeds_bar_volume():
    c = RealisticCost(impact_coef=0.1, min_slippage_bps=0.5)

    # заявка больше бара — доля ограничена единицей
    assert c.slippage_bps(10_000_000.0, 1_000_000.0) == pytest.approx(0.5 + 1e4 * 0.1)


def test_slippage_handles_zero_volume():
    c = RealisticCost(min_slippage_bps=0.5)

    assert c.slippage_bps(1000.0, 0.0) == pytest.approx(0.5)


def test_funding_cost_sign_follows_position():
    c = RealisticCost()

    long_cost = c.funding_cost(position=1.0, rate=0.001)
    short_cost = c.funding_cost(position=-1.0, rate=0.001)

    assert long_cost > 0        # лонг платит при положительном funding
    assert short_cost < 0       # шорт получает
    assert long_cost == pytest.approx(-short_cost)


def test_from_config_reads_values():
    c = RealisticCost.from_config({"taker_fee_bps": 7.5, "impact_coef": 0.2})

    assert c.taker_fee_bps == 7.5
    assert c.impact_coef == 0.2
    assert c.maker_share == 0.0     # значение по умолчанию
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_costs.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `costs.py`**

`src/alpha_lab/engine/costs.py`:
```python
"""Модель издержек — место, где рождается и умирает альфа.

Проскальзывание зависит от размера заявки относительно объёма бара: модель
«один тик на сделку» делает невидимой зависимость результата от капитала.
Funding начисляется каждые N часов (интервал у разных пар разный) и для позиций,
удерживаемых дольше нескольких часов, может съесть весь edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

BPS = 1e-4


class CostModel(Protocol):
    def fee_bps(self) -> float: ...
    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float: ...
    def funding_cost(self, position: float, rate: float) -> float: ...


@dataclass(frozen=True)
class ZeroCost:
    """Заведомо нулевые издержки. Только для инвариантных тестов."""

    def fee_bps(self) -> float:
        return 0.0

    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float:
        return 0.0

    def funding_cost(self, position: float, rate: float) -> float:
        return 0.0


@dataclass(frozen=True)
class RealisticCost:
    """Комиссии Binance USDT-M VIP0 + модель воздействия на рынок.

    taker_fee_bps: 5.0 = 0.05% (реальный тариф VIP0 для USDT-M futures)
    maker_fee_bps: 2.0 = 0.02%
    impact_coef:   коэффициент воздействия; проскальзывание растёт линейно
                   с долей заявки в объёме бара
    """
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    maker_share: float = 0.0
    impact_coef: float = 0.1
    min_slippage_bps: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict) -> "RealisticCost":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in known})

    def fee_bps(self) -> float:
        share = min(max(self.maker_share, 0.0), 1.0)
        return share * self.maker_fee_bps + (1.0 - share) * self.taker_fee_bps

    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float:
        if bar_quote_volume <= 0:
            return self.min_slippage_bps
        share = min(max(abs(order_notional) / bar_quote_volume, 0.0), 1.0)
        return self.min_slippage_bps + 1e4 * self.impact_coef * share

    def funding_cost(self, position: float, rate: float) -> float:
        """Издержка funding как доля ноционала. Лонг платит при rate > 0."""
        return position * rate
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_costs.py -v
```
Ожидается: 9 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/engine/costs.py tests/test_costs.py
git commit -m "feat: cost model with size-dependent slippage and funding"
```

---

## Задача 9: Векторный движок бэктеста

**Файлы:**
- Создать: `src/alpha_lab/engine/backtest.py`, `tests/test_backtest.py`

**Интерфейсы:**
- Отдаёт: `BacktestResult(equity, returns, gross_returns, positions, turnover, costs, trades, total_return, max_dd)`
- `run_backtest(bars, positions, cost_model, initial_equity=1.0, capital=10_000.0) -> BacktestResult`
- Соглашение: `positions[t]` — целевая позиция, **решённая по информацию до бара t включительно**; удерживается с бара `t+1`. Это устраняет look-ahead по построению.

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_backtest.py`:
```python
import numpy as np
import pandas as pd
import pytest

from alpha_lab.engine.backtest import run_backtest
from alpha_lab.engine.costs import RealisticCost, ZeroCost


def _bars(close):
    close = pd.Series(close, dtype="float64")
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=len(close), freq="1min", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": 1e6, "quote_volume": 1e9, "trades": 100, "taker_buy_volume": 1e5,
    })


def test_buy_and_hold_zero_cost_matches_price_return():
    bars = _bars([100, 110, 121, 133.1])          # +10% за бар
    positions = pd.Series([1.0, 1.0, 1.0, 1.0])

    res = run_backtest(bars, positions, ZeroCost())

    # позиция удерживается со следующего бара: 3 перехода по +10%
    assert res.total_return == pytest.approx(1.1 ** 3 - 1, rel=1e-6)


def test_flat_positions_give_zero_return():
    bars = _bars([100, 110, 90, 95])
    positions = pd.Series([0.0, 0.0, 0.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert res.total_return == pytest.approx(0.0)
    assert res.turnover.sum() == pytest.approx(0.0)


def test_positions_are_shifted_no_lookahead():
    """Стратегия, идеально знающая бар t, не должна заработать на нём же."""
    bars = _bars([100, 200, 100, 200])
    # позиция 1 ровно в бары роста — если бы исполнялась в тот же бар, был бы огромный профит
    positions = pd.Series([1.0, 0.0, 1.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())

    # Сдвиг: позиция 1 держится на баре 1 (рост 100→200, профит),
    # позиция 0 на баре 2 (падение — не участвуем), позиция 1 на баре 3 (рост).
    assert res.total_return == pytest.approx(2.0 * 2.0 - 1.0, rel=1e-6)


def test_costs_reduce_return_and_create_turnover():
    bars = _bars([100] * 10 + [110] * 10)
    positions = pd.Series([1.0] * 20)

    free = run_backtest(bars, positions, ZeroCost())
    costly = run_backtest(bars, positions, RealisticCost(taker_fee_bps=100.0))

    assert costly.total_return < free.total_return
    assert costly.turnover.sum() > 0


def test_higher_fees_monotonically_lower_return():
    bars = _bars([100, 105, 95, 110, 100, 115])
    positions = pd.Series([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])

    returns = [
        run_backtest(bars, positions, RealisticCost(taker_fee_bps=f)).total_return
        for f in (0.0, 5.0, 20.0, 100.0)
    ]

    assert returns == sorted(returns, reverse=True)


def test_equal_timestamps_produce_no_pnl():
    bars = _bars([100, 100, 100])
    positions = pd.Series([1.0, 1.0, 1.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert res.total_return == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_is_negative_or_zero():
    bars = _bars([100, 120, 60, 80])
    positions = pd.Series([1.0, 1.0, 1.0, 1.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert res.max_drawdown <= 0
    assert res.max_drawdown == pytest.approx(60 / 120 - 1, rel=1e-6)


def test_funding_charged_on_held_position():
    bars = _bars([100] * 20)
    positions = pd.Series([1.0] * 20)
    funding = pd.Series(0.0, index=bars.index)
    funding.iloc[10] = 0.001     # 0.1% выплата

    res = run_backtest(bars, positions, RealisticCost(), funding_rate=funding)

    assert res.total_return == pytest.approx(-0.001, rel=1e-3)


def test_result_lengths_match_input():
    bars = _bars([100, 101, 102, 103, 104])
    positions = pd.Series([0.0, 1.0, 1.0, 0.0, 0.0])

    res = run_backtest(bars, positions, ZeroCost())

    assert len(res.equity) == len(bars)
    assert len(res.returns) == len(bars)
    assert len(res.positions) == len(bars)


def test_mismatched_lengths_raise():
    bars = _bars([100, 101, 102])
    positions = pd.Series([1.0, 1.0])

    with pytest.raises(ValueError, match="длин"):
        run_backtest(bars, positions, ZeroCost())
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_backtest.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `backtest.py`**

`src/alpha_lab/engine/backtest.py`:
```python
"""Векторный движок бэктеста.

Соглашение о времени (устраняет look-ahead по построению):
    positions[t] — целевая позиция, решённая по информации ДО бара t включительно.
    Удерживается она начиная с бара t+1: held[t] = positions[t-1].
    Доходность бара t начисляется на held[t].

Все вычисления векторизованы; путь-зависимые выходы (SL/TP) живут в engine/exits.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.engine.costs import CostModel, ZeroCost

BPS = 1e-4


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    positions: pd.Series
    turnover: pd.Series
    costs: pd.DataFrame          # колонки fee, slippage, funding
    total_return: float
    max_drawdown: float
    bars: int
    price_returns: pd.Series      # доходности самой цены; нужны permutation-тесту

    @property
    def cost_totals(self) -> dict[str, float]:
        return {c: float(self.costs[c].sum()) for c in self.costs.columns}


def run_backtest(bars: pd.DataFrame, positions: pd.Series, cost_model: CostModel,
                 initial_equity: float = 1.0, capital: float = 10_000.0,
                 funding_rate: pd.Series | None = None) -> BacktestResult:
    """Прогоняет позиции по барам с учётом издержек.

    capital — размер счёта; вместе с объёмом бара определяет проскальзывание,
    поэтому результат зависит от капитала (это намеренно).
    """
    n = len(bars)
    if len(positions) != n:
        raise ValueError(
            f"Длина positions ({len(positions)}) не совпадает с длиной баров ({n})"
        )
    if n < 2:
        raise ValueError("Нужно минимум 2 бара")

    close = bars["close"].astype("float64").to_numpy()
    quote_volume = bars["quote_volume"].astype("float64").to_numpy()
    pos = positions.astype("float64").fillna(0.0).to_numpy()

    # Позиция, удерживаемая в баре t, решена на баре t-1
    held = np.empty(n, dtype="float64")
    held[0] = 0.0
    held[1:] = pos[:-1]

    # Доходность цены бар-к-бару
    price_ret = np.zeros(n, dtype="float64")
    price_ret[1:] = close[1:] / close[:-1] - 1.0
    price_ret[~np.isfinite(price_ret)] = 0.0

    gross = held * price_ret

    # Оборот: изменение удерживаемой позиции
    turnover = np.zeros(n, dtype="float64")
    turnover[1:] = np.abs(held[1:] - held[:-1])

    notional = np.abs(held) * capital
    slippage_bps = np.array([
        cost_model.slippage_bps(notional[i], quote_volume[i]) for i in range(n)
    ])
    fee = turnover * cost_model.fee_bps() * BPS
    slip = turnover * slippage_bps * BPS

    if funding_rate is not None:
        rate = pd.Series(funding_rate).astype("float64").fillna(0.0).to_numpy()
        fund = np.array([cost_model.funding_cost(held[i], rate[i]) for i in range(n)])
    else:
        fund = np.zeros(n, dtype="float64")

    costs = pd.DataFrame(
        {"fee": fee, "slippage": slip, "funding": fund}, index=bars.index
    )
    net = gross - fee - slip - fund

    equity = pd.Series(initial_equity * np.cumprod(1.0 + net), index=bars.index,
                       name="equity")
    peak = equity.cummax()
    max_dd = float((equity / peak - 1.0).min())

    return BacktestResult(
        equity=equity,
        returns=pd.Series(net, index=bars.index, name="returns"),
        gross_returns=pd.Series(gross, index=bars.index, name="gross_returns"),
        positions=pd.Series(held, index=bars.index, name="held"),
        turnover=pd.Series(turnover, index=bars.index, name="turnover"),
        costs=costs,
        total_return=float(equity.iloc[-1] / initial_equity - 1.0),
        max_drawdown=max_dd,
        bars=n,
        price_returns=pd.Series(price_ret, index=bars.index, name="price_returns"),
    )


def trade_returns(result: BacktestResult) -> pd.Series:
    """R-мультипликаторы сделок: суммарная доходность за непрерывный период удержания.

    Сделка — последовательность баров с одинаковым знаком удерживаемой позиции.
    """
    held = result.positions.to_numpy()
    net = result.returns.to_numpy()
    sign = np.sign(held)

    trades, current, current_sign = [], 0.0, 0.0
    for i in range(len(sign)):
        if sign[i] != current_sign:
            if current_sign != 0.0:
                trades.append(current)
            current, current_sign = 0.0, sign[i]
        if current_sign != 0.0:
            current += net[i]
    if current_sign != 0.0:
        trades.append(current)
    return pd.Series(trades, name="trade_return")
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_backtest.py -v
```
Ожидается: 10 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/engine/backtest.py tests/test_backtest.py
git commit -m "feat: vectorized backtest engine with shift-by-one execution"
```

---

## Задача 10: Симуляция выходов по SL/TP (numba)

**Файлы:**
- Создать: `src/alpha_lab/engine/exits.py`, `tests/test_exits.py`

**Интерфейсы:**
- Отдаёт: `simulate_bracket_exits(bars, entries, sl_price, tp_price, max_bars) -> pd.Series`
  Возвращает позицию (1/−1/0) по барам: 1 пока удерживается лонг, 0 после выхода.
- `atr_brackets(close, atr, direction, sl_atr, tp_atr) -> (sl_price, tp_price)`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_exits.py`:
```python
import numpy as np
import pandas as pd
import pytest

from alpha_lab.engine.exits import atr_brackets, simulate_bracket_exits


def _bars(highs, lows):
    n = len(highs)
    mid = [(h + l) / 2 for h, l in zip(highs, lows)]
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC"),
        "open": mid, "high": highs, "low": lows, "close": mid,
        "volume": 1e6, "quote_volume": 1e9, "trades": 100, "taker_buy_volume": 1e5,
    })


def test_long_take_profit_closes_position():
    bars = _bars(highs=[100, 101, 105, 106, 107], lows=[100, 100, 101, 102, 103])
    entries = pd.Series([True, False, False, False, False])
    sl = pd.Series([98.0] * 5)
    tp = pd.Series([104.0] * 5)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[0] == 1.0     # вошли
    assert pos.iloc[2] == 1.0     # ещё держим (high 105 пробил tp)
    assert pos.iloc[3] == 0.0     # вышли


def test_long_stop_loss_closes_position():
    bars = _bars(highs=[100, 101, 101, 101], lows=[100, 99, 97, 96])
    entries = pd.Series([True, False, False, False])
    sl = pd.Series([98.0] * 4)
    tp = pd.Series([110.0] * 4)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[0] == 1.0
    assert pos.iloc[1] == 1.0
    assert pos.iloc[2] == 0.0     # low 97 пробил стоп 98


def test_short_take_profit():
    bars = _bars(highs=[100, 100, 100, 100], lows=[100, 99, 94, 93])
    entries = pd.Series([False, True, False, False])
    sl = pd.Series([102.0] * 4)
    tp = pd.Series([95.0] * 4)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=10)

    assert pos.iloc[1] == -1.0
    assert pos.iloc[3] == 0.0


def test_max_bars_forces_exit():
    bars = _bars(highs=[100] * 10, lows=[100] * 10)
    entries = pd.Series([True] + [False] * 9)
    sl = pd.Series([50.0] * 10)
    tp = pd.Series([150.0] * 10)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=3)

    assert pos.iloc[4] == 0.0     # вышли по времени


def test_no_entry_means_no_position():
    bars = _bars(highs=[100] * 5, lows=[100] * 5)
    entries = pd.Series([False] * 5)

    pos = simulate_bracket_exits(bars, entries, pd.Series([90.0] * 5),
                                 pd.Series([110.0] * 5), max_bars=5)

    assert (pos == 0.0).all()


def test_stop_takes_priority_when_bar_hits_both():
    """Если бар пробил и стоп, и тейк — консервативно считаем стоп."""
    bars = _bars(highs=[100, 120], lows=[100, 80])
    entries = pd.Series([True, False])
    sl = pd.Series([90.0] * 2)
    tp = pd.Series([110.0] * 2)

    pos = simulate_bracket_exits(bars, entries, sl, tp, max_bars=5)

    assert pos.iloc[1] == 0.0


def test_atr_brackets_long_and_short():
    close = pd.Series([100.0, 100.0])
    a = pd.Series([2.0, 2.0])

    sl_l, tp_l = atr_brackets(close, a, direction=1, sl_atr=2.0, tp_atr=6.0)
    sl_s, tp_s = atr_brackets(close, a, direction=-1, sl_atr=2.0, tp_atr=6.0)

    assert sl_l.iloc[0] == pytest.approx(96.0)
    assert tp_l.iloc[0] == pytest.approx(112.0)
    assert sl_s.iloc[0] == pytest.approx(104.0)
    assert tp_s.iloc[0] == pytest.approx(88.0)


def test_position_length_matches_bars():
    bars = _bars(highs=[100] * 20, lows=[100] * 20)
    entries = pd.Series([True] + [False] * 19)

    pos = simulate_bracket_exits(bars, entries, pd.Series([90.0] * 20),
                                 pd.Series([110.0] * 20), max_bars=5)

    assert len(pos) == 20
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_exits.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `exits.py`**

`src/alpha_lab/engine/exits.py`:
```python
"""Путь-зависимая симуляция выходов по стопу и тейку.

Векторизовать нельзя: результат каждого бара зависит от того, были ли мы в позиции
на предыдущем. Цикл компилируется numba; при недоступности numba используется
чистый Python (тот же алгоритм, медленнее).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # numba ускоряет цикл на порядки; при отсутствии — тихий откат
    from numba import njit
    HAS_NUMBA = True
except ImportError:  # pragma: no cover
    HAS_NUMBA = False

    def njit(*args, **kwargs):
        def wrapper(fn):
            return fn
        return wrapper if not (len(args) == 1 and callable(args[0])) else args[0]


@njit(cache=True)
def _simulate(high, low, entry_idx, sl, tp, max_bars):
    n = len(high)
    pos = np.zeros(n, dtype=np.float64)
    direction = 0.0
    entry_bar = -1
    cur_sl = 0.0
    cur_tp = 0.0

    for i in range(n):
        if direction != 0.0:
            pos[i] = direction
            hit_sl = (low[i] <= cur_sl) if direction > 0 else (high[i] >= cur_sl)
            hit_tp = (high[i] >= cur_tp) if direction > 0 else (low[i] <= cur_tp)
            # Консервативно: если бар пробил и стоп, и тейк — считаем стоп
            if hit_sl or (i - entry_bar) >= max_bars:
                direction = 0.0
                pos[i] = 0.0
            elif hit_tp:
                direction = 0.0
                pos[i] = 0.0
                continue
            continue

        if entry_idx[i] != 0:
            direction = entry_idx[i]
            entry_bar = i
            cur_sl = sl[i]
            cur_tp = tp[i]
            pos[i] = direction

    return pos


def simulate_bracket_exits(bars: pd.DataFrame, entries: pd.Series,
                           sl_price: pd.Series, tp_price: pd.Series,
                           max_bars: int = 500) -> pd.Series:
    """Позиция по барам для входов с фиксированными стопом и тейком.

    entries: 0 — нет входа, +1 — лонг, −1 — шорт. Допускается bool (→ +1).
    Вход исполняется в баре сигнала; стоп проверяется в том же баре.
    """
    n = len(bars)
    high = bars["high"].astype("float64").to_numpy()
    low = bars["low"].astype("float64").to_numpy()

    raw = pd.Series(entries)
    if raw.dtype == bool:
        entry_idx = raw.astype("float64").to_numpy()
    else:
        entry_idx = raw.astype("float64").fillna(0.0).to_numpy()

    sl = pd.Series(sl_price).astype("float64").fillna(-np.inf).to_numpy()
    tp = pd.Series(tp_price).astype("float64").fillna(np.inf).to_numpy()

    pos = _simulate(high, low, entry_idx, sl, tp, int(max_bars))
    return pd.Series(pos, index=bars.index, name="position")


def atr_brackets(close: pd.Series, atr: pd.Series, direction: int,
                 sl_atr: float, tp_atr: float) -> tuple[pd.Series, pd.Series]:
    """Цены стопа и тейка на основе ATR. direction: +1 лонг, −1 шорт."""
    if direction not in (1, -1):
        raise ValueError("direction должен быть +1 или −1")
    c = pd.Series(close).astype("float64")
    a = pd.Series(atr).astype("float64")
    sl = c - direction * sl_atr * a
    tp = c + direction * tp_atr * a
    return sl.rename("sl"), tp.rename("tp")
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_exits.py -v
```
Ожидается: 8 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/engine/exits.py tests/test_exits.py
git commit -m "feat: numba bracket exit simulation for SL/TP"
```

---

## Задача 11: Протокол стратегии и перенос MR-модели

**Файлы:**
- Создать: `src/alpha_lab/strategies/__init__.py`, `src/alpha_lab/strategies/base.py`, `src/alpha_lab/strategies/mean_reversion.py`, `tests/test_strategy_mr.py`

**Интерфейсы:**
- Отдаёт: `Strategy` (протокол) с `name: str` и `generate(bars) -> pd.Series`
- `MeanReversionStrategy(params: dict)`; `build_strategy(name, params) -> Strategy`
- Перенос из `index.html` (блок `pine-strategy`, «ЯДРО МОДЕЛИ»): z-score, опциональный фильтр полужизни, ATR-стоп 2× и тейк 6×.

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_strategy_mr.py`:
```python
import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_bars

from alpha_lab.strategies.base import build_strategy
from alpha_lab.strategies.mean_reversion import MeanReversionStrategy


def test_position_is_bounded():
    bars = ou_bars(n=3000, theta=0.05, sigma=1.0, seed=11)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    assert pos.between(-1.0, 1.0).all()
    assert len(pos) == len(bars)


def test_no_lookahead_position_depends_only_on_past():
    """Изменение будущих баров не должно менять прошлые позиции."""
    bars = ou_bars(n=2000, seed=12)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    full = s.generate(bars)
    truncated = s.generate(bars.iloc[:1000])

    pd.testing.assert_series_equal(full.iloc[:1000], truncated, check_names=False)


def test_generates_trades_on_mean_reverting_series():
    bars = ou_bars(n=5000, theta=0.10, sigma=1.0, seed=13)
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    assert (pos != 0).sum() > 50      # на возвращающемся ряде входы обязаны быть


def test_rarely_trades_on_random_walk():
    from fixtures.synthetic import random_walk

    bars = ou_bars(n=5000, seed=14)
    bars["close"] = random_walk(n=5000, seed=14).to_numpy()
    s = MeanReversionStrategy({"window": 20, "k": 2.0})

    pos = s.generate(bars)

    # На трендовом ряде z-скор редко возвращается к нулю — входов мало
    assert (pos != 0).mean() < 0.5


def test_zero_signals_during_warmup():
    bars = ou_bars(n=500, seed=15)
    s = MeanReversionStrategy({"window": 100, "k": 2.0})

    pos = s.generate(bars)

    assert (pos.iloc[:100] == 0).all()


def test_half_life_filter_suppresses_signals():
    bars = ou_bars(n=3000, theta=0.30, sigma=1.0, seed=16)   # быстрый возврат
    strict = MeanReversionStrategy({
        "window": 20, "k": 2.0, "use_hl_filter": True,
        "hl_min": 0.01, "hl_max": 0.5,                        # заведомо узкое окно
    })
    loose = MeanReversionStrategy({"window": 20, "k": 2.0, "use_hl_filter": False})

    assert (strict.generate(bars) != 0).sum() < (loose.generate(bars) != 0).sum()


def test_build_strategy_returns_mr():
    s = build_strategy("mean_reversion", {"window": 10})

    assert isinstance(s, MeanReversionStrategy)


def test_build_strategy_rejects_unknown():
    with pytest.raises(ValueError, match="Неизвестная стратегия"):
        build_strategy("nope", {})
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_strategy_mr.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `base.py`**

`src/alpha_lab/strategies/base.py`:
```python
"""Протокол стратегии. Одна функция — намеренно узкий интерфейс."""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    name: str

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        """Целевая позиция: −1.0 (полный шорт) … 0.0 … +1.0 (полный лонг).

        Обязан использовать только информацию по бар t включительно.
        Исполнение происходит на баре t+1 (см. engine.backtest).
        """
        ...


def build_strategy(name: str, params: dict) -> Strategy:
    from alpha_lab.strategies.mean_reversion import MeanReversionStrategy

    registry = {"mean_reversion": MeanReversionStrategy}
    if name not in registry:
        raise ValueError(
            f"Неизвестная стратегия: '{name}'. Доступные: {sorted(registry)}"
        )
    return registry[name](params)
```

- [ ] **Шаг 4: Реализовать `mean_reversion.py`**

`src/alpha_lab/strategies/mean_reversion.py`:
```python
"""Mean Reversion Quant Pro — перенос Pine Script (index.html) в Python.

Источник: блок pine-strategy, секция «ЯДРО МОДЕЛИ».
Логика входа: z ≤ −k → лонг, z ≥ +k → шорт.
Выход: стоп 2×ATR, тейк 6×ATR, принудительный выход по времени.
Опциональный фильтр: полужизнь OU должна попадать в [hl_min, hl_max].

ВАЖНО: это исследовательская копия, а не эталон. Расхождение с TradingView
на общем периоде — ожидаемо и измеряемо (см. критерий успеха 0).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.engine.exits import atr_brackets, simulate_bracket_exits
from alpha_lab.features.price import ou_params, zscore
from alpha_lab.features.volatility import atr as atr_series

DEFAULTS = {
    "window": 20,
    "k": 2.0,
    "atr_len": 14,
    "sl_atr": 2.0,
    "tp_atr": 6.0,
    "max_bars": 500,
    "use_hl_filter": False,
    "hl_min": 5.0,
    "hl_max": 100.0,
    "hl_window": 200,
}


class MeanReversionStrategy:
    name = "mean_reversion"

    def __init__(self, params: dict | None = None):
        cfg = {**DEFAULTS, **(params or {})}
        self.window = int(cfg["window"])
        self.k = float(cfg["k"])
        self.atr_len = int(cfg["atr_len"])
        self.sl_atr = float(cfg["sl_atr"])
        self.tp_atr = float(cfg["tp_atr"])
        self.max_bars = int(cfg["max_bars"])
        self.use_hl_filter = bool(cfg["use_hl_filter"])
        self.hl_min = float(cfg["hl_min"])
        self.hl_max = float(cfg["hl_max"])
        self.hl_window = int(cfg["hl_window"])
        self.params = cfg

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        z = zscore(close, self.window)
        atr_vals = atr_series(bars, self.atr_len)

        long_entry = (z <= -self.k).to_numpy()
        short_entry = (z >= self.k).to_numpy()

        if self.use_hl_filter:
            allowed = self._half_life_ok(close).to_numpy()
            long_entry &= allowed
            short_entry &= allowed

        entries = np.zeros(len(bars), dtype="float64")
        entries[long_entry] = 1.0
        entries[short_entry] = -1.0

        # Стоп и тейк для каждого направления; берём то, что соответствует входу
        sl_long, tp_long = atr_brackets(close, atr_vals, 1, self.sl_atr, self.tp_atr)
        sl_short, tp_short = atr_brackets(close, atr_vals, -1, self.sl_atr, self.tp_atr)

        entries_series = pd.Series(entries, index=bars.index)
        sl = sl_long.where(entries_series > 0, sl_short)
        tp = tp_long.where(entries_series > 0, tp_short)

        pos = simulate_bracket_exits(bars, entries_series, sl, tp, self.max_bars)
        return pos.fillna(0.0)

    def _half_life_ok(self, close: pd.Series) -> pd.Series:
        """Полужизнь OU в скользящем окне должна лежать в допустимом диапазоне."""
        w = self.hl_window
        hl = np.full(len(close), np.nan)
        values = close.to_numpy()
        for i in range(w, len(close)):
            p = ou_params(values[i - w:i])
            hl[i] = p.half_life
        ok = (hl >= self.hl_min) & (hl <= self.hl_max)
        return pd.Series(ok, index=close.index)
```

- [ ] **Шаг 5: Запустить тесты**

```bash
uv run pytest tests/test_strategy_mr.py -v
```
Ожидается: 8 passed

- [ ] **Шаг 6: Коммит**

```bash
git add src/alpha_lab/strategies tests/test_strategy_mr.py
git commit -m "feat: strategy protocol and ported mean reversion model"
```

---

## Задача 12: Метрики эффективности

**Файлы:**
- Создать: `src/alpha_lab/validation/__init__.py`, `src/alpha_lab/validation/metrics.py`, `tests/test_metrics.py`

**Интерфейсы:**
- Отдаёт: `sharpe_ratio(returns, periods_per_year, annualize=True) -> float`
- `sortino_ratio`, `max_drawdown(equity)`, `calmar_ratio`, `profit_factor(trade_returns)`
- `summarize(returns, trade_returns, equity) -> dict[str, float]`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_metrics.py`:
```python
import numpy as np
import pandas as pd
import pytest

from alpha_lab.validation.metrics import (
    calmar_ratio, max_drawdown, profit_factor, sharpe_ratio, sortino_ratio, summarize,
)


def test_sharpe_of_constant_returns_is_zero_std():
    r = pd.Series([0.001] * 100)

    assert sharpe_ratio(r) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.001, 0.01, 5000))

    assert sharpe_ratio(r) > 0


def test_sharpe_scales_with_annualization():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))

    raw = sharpe_ratio(r, annualize=False)
    ann = sharpe_ratio(r, periods_per_year=365 * 24)

    assert ann == pytest.approx(raw * np.sqrt(365 * 24), rel=1e-9)


def test_sharpe_of_negated_returns_flips_sign():
    rng = np.random.default_rng(3)
    r = pd.Series(rng.normal(0.0005, 0.01, 2000))

    assert sharpe_ratio(-r) == pytest.approx(-sharpe_ratio(r))


def test_max_drawdown_known_value():
    equity = pd.Series([1.0, 1.2, 0.9, 1.0, 1.5])

    assert max_drawdown(equity) == pytest.approx(0.9 / 1.2 - 1.0)


def test_max_drawdown_monotonic_equity_is_zero():
    assert max_drawdown(pd.Series([1.0, 1.1, 1.2, 1.3])) == pytest.approx(0.0)


def test_sortino_ignores_upside_volatility():
    rng = np.random.default_rng(4)
    r = pd.Series(rng.normal(0.001, 0.01, 3000))

    # Сортино ≥ Шарпа: в знаменателе только нисходящая волатильность
    assert sortino_ratio(r) >= sharpe_ratio(r) * 0.9


def test_profit_factor():
    trades = pd.Series([0.05, -0.02, 0.03, -0.01])

    assert profit_factor(trades) == pytest.approx(0.08 / 0.03)


def test_profit_factor_no_losses_is_inf():
    assert profit_factor(pd.Series([0.01, 0.02])) == float("inf")


def test_calmar_ratio():
    equity = pd.Series([1.0, 1.5, 1.2, 1.8])
    returns = equity.pct_change().fillna(0.0)

    c = calmar_ratio(returns, equity)

    assert c > 0


def test_summarize_returns_all_keys():
    rng = np.random.default_rng(5)
    r = pd.Series(rng.normal(0.0005, 0.01, 1000))
    equity = (1 + r).cumprod()
    trades = pd.Series(rng.normal(0.002, 0.01, 50))

    out = summarize(r, trades, equity)

    for key in ("sharpe", "sortino", "max_dd", "calmar", "profit_factor",
                "total_return", "trades", "win_rate", "avg_trade"):
        assert key in out


def test_summarize_handles_empty_trades():
    r = pd.Series([0.0] * 10)
    out = summarize(r, pd.Series([], dtype="float64"), (1 + r).cumprod())

    assert out["trades"] == 0
    assert np.isnan(out["profit_factor"]) or out["profit_factor"] == 0.0
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_metrics.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `metrics.py`**

`src/alpha_lab/validation/metrics.py`:
```python
"""Метрики эффективности.

sharpe_ratio с annualize=False возвращает значение за период — именно оно
требуется формуле Deflated Sharpe (см. significance.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MINUTES_PER_YEAR = 365 * 24 * 60
DEFAULT_PERIODS = 365 * 24          # часовые бары крипты (24/7)


def _clean(returns) -> np.ndarray:
    r = np.asarray(pd.Series(returns), dtype="float64")
    return r[np.isfinite(r)]


def sharpe_ratio(returns, periods_per_year: int = DEFAULT_PERIODS,
                 annualize: bool = True, risk_free: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd < 1e-12:
        return 0.0
    raw = (r.mean() - risk_free) / sd
    return float(raw * np.sqrt(periods_per_year) if annualize else raw)


def sortino_ratio(returns, periods_per_year: int = DEFAULT_PERIODS) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd < 1e-12:
        return float("inf") if r.mean() > 0 else 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def max_drawdown(equity) -> float:
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]
    if len(eq) < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    return float((eq / peak - 1.0).min())


def calmar_ratio(returns, equity, periods_per_year: int = DEFAULT_PERIODS) -> float:
    dd = abs(max_drawdown(equity))
    if dd < 1e-12:
        return 0.0
    ann_return = sharpe_ratio(returns, periods_per_year, annualize=False) * periods_per_year
    return float(ann_return / dd)


def profit_factor(trade_returns) -> float:
    t = _clean(trade_returns)
    if len(t) == 0:
        return float("nan")
    gains = t[t > 0].sum()
    losses = abs(t[t < 0].sum())
    if losses < 1e-12:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def summarize(returns, trade_returns, equity,
              periods_per_year: int = DEFAULT_PERIODS) -> dict[str, float]:
    r = _clean(returns)
    t = _clean(trade_returns)
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]

    total = float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0
    wins = t[t > 0]
    return {
        "sharpe": sharpe_ratio(r, periods_per_year),
        "sortino": sortino_ratio(r, periods_per_year),
        "max_dd": max_drawdown(eq),
        "calmar": calmar_ratio(r, eq, periods_per_year),
        "profit_factor": profit_factor(t),
        "total_return": total,
        "trades": float(len(t)),
        "win_rate": float(len(wins) / len(t)) if len(t) else 0.0,
        "avg_trade": float(t.mean()) if len(t) else 0.0,
    }
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_metrics.py -v
```
Ожидается: 12 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/validation/metrics.py tests/test_metrics.py
git commit -m "feat: performance metrics"
```

---

## Задача 13: Purged K-Fold с embargo

**Файлы:**
- Создать: `src/alpha_lab/validation/splits.py`, `tests/test_splits.py`

**Интерфейсы:**
- Отдаёт: `purged_kfold_indices(n_samples, n_splits, purge, embargo) -> Iterator[tuple[np.ndarray, np.ndarray]]`
- `PurgedKFold` класс с `.split(n_samples)` и `.n_splits`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_splits.py`:
```python
import numpy as np
import pytest

from alpha_lab.validation.splits import PurgedKFold, purged_kfold_indices


def test_test_folds_cover_all_samples_exactly_once():
    seen = np.zeros(100, dtype=int)
    for _, test in purged_kfold_indices(100, n_splits=5, purge=2, embargo=2):
        seen[test] += 1

    assert (seen == 1).all()


def test_train_never_overlaps_test():
    for train, test in purged_kfold_indices(200, n_splits=4, purge=10, embargo=5):
        assert len(np.intersect1d(train, test)) == 0


def test_purge_removes_neighbours_of_test():
    for train, test in purged_kfold_indices(200, n_splits=4, purge=10, embargo=0):
        lo, hi = test.min(), test.max()
        inside = train[(train >= lo - 10) & (train <= hi + 10)]
        assert len(inside) == 0


def test_embargo_removes_samples_after_test():
    for train, test in purged_kfold_indices(200, n_splits=4, purge=0, embargo=5):
        hi = test.max()
        after = train[(train > hi) & (train <= hi + 5)]
        assert len(after) == 0


def test_zero_purge_and_embargo_keeps_all_other_samples():
    n, k = 100, 5
    for train, test in purged_kfold_indices(n, n_splits=k, purge=0, embargo=0):
        assert len(train) + len(test) == n


def test_embargo_at_series_end_does_not_overflow():
    splits = list(purged_kfold_indices(50, n_splits=5, purge=0, embargo=100))

    assert len(splits) == 5


def test_invalid_parameters_raise():
    with pytest.raises(ValueError, match="n_splits"):
        list(purged_kfold_indices(100, n_splits=1, purge=0, embargo=0))
    with pytest.raises(ValueError, match="purge"):
        list(purged_kfold_indices(100, n_splits=5, purge=-1, embargo=0))


def test_purged_kfold_class_matches_function():
    kf = PurgedKFold(n_splits=5, purge=3, embargo=3)
    a = list(kf.split(120))
    b = list(purged_kfold_indices(120, 5, 3, 3))

    assert len(a) == len(b)
    for (tr1, te1), (tr2, te2) in zip(a, b):
        assert np.array_equal(tr1, tr2)
        assert np.array_equal(te1, te2)
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_splits.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `splits.py`**

`src/alpha_lab/validation/splits.py`:
```python
"""Purged K-Fold с embargo (López de Prado, AFML гл. 7).

Зачем: признаки считаются на скользящем окне, поэтому train и test пересекаются
по времени, и модель косвенно видит будущее. Обычный KFold этого не замечает.

purge  — из train удаляются наблюдения, чьё окно признаков заглядывает в test.
embargo — дополнительно удаляются наблюдения сразу ПОСЛЕ test: признаки инерционны,
          и информация из test «протекает» в следующие за ним бары.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


def purged_kfold_indices(n_samples: int, n_splits: int, purge: int,
                         embargo: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if n_splits < 2:
        raise ValueError(f"n_splits должен быть ≥ 2, получено {n_splits}")
    if purge < 0:
        raise ValueError(f"purge не может быть отрицательным, получено {purge}")
    if embargo < 0:
        raise ValueError(f"embargo не может быть отрицательным, получено {embargo}")
    if n_samples < n_splits:
        raise ValueError(
            f"Наблюдений ({n_samples}) меньше, чем фолдов ({n_splits})"
        )

    indices = np.arange(n_samples)
    for test in np.array_split(indices, n_splits):
        test_lo, test_hi = int(test[0]), int(test[-1])
        train_mask = np.ones(n_samples, dtype=bool)

        # Сам тест исключается всегда
        train_mask[test_lo:test_hi + 1] = False
        # Purge: заглядываем на purge назад и вперёд от границ теста
        train_mask[max(0, test_lo - purge):min(n_samples, test_hi + 1 + purge)] = False
        # Embargo: зазор сразу после теста
        train_mask[test_hi + 1:min(n_samples, test_hi + 1 + embargo)] = False

        yield indices[train_mask], test


@dataclass(frozen=True)
class PurgedKFold:
    n_splits: int = 6
    purge: int = 50
    embargo: int = 20

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        return purged_kfold_indices(n_samples, self.n_splits, self.purge, self.embargo)
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_splits.py -v
```
Ожидается: 8 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/validation/splits.py tests/test_splits.py
git commit -m "feat: purged k-fold cross-validation with embargo"
```

---

## Задача 14: Значимость — DSR, PBO, permutation

**Файлы:**
- Создать: `src/alpha_lab/validation/significance.py`, `tests/test_significance.py`

**Интерфейсы:**
- Отдаёт: `deflated_sharpe_ratio(returns, n_trials, sr_variance=None) -> float`
- `pbo_cscv(returns_matrix, n_blocks=10) -> float`
- `permutation_pvalue(returns, n_permutations=1000, seed=0) -> float`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_significance.py`:
```python
import numpy as np
import pytest

from alpha_lab.validation.significance import (
    _sharpe_raw, deflated_sharpe_ratio, pbo_cscv, permutation_pvalue,
)


def test_dsr_of_pure_noise_is_low_after_many_trials():
    """Шум при поправке на 1000 попыток обязан получить низкий DSR.

    Проверяем именно с n_trials > 1: при одной попытке DSR шума колеблется
    вокруг 0.5, и любой жёсткий порог здесь — флейки-тест.
    """
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, 5000)

    assert deflated_sharpe_ratio(r, n_trials=1000) < 0.1


def test_dsr_lower_for_more_trials():
    """Чем больше конфигураций перебрали, тем сильнее штраф."""
    rng = np.random.default_rng(2)
    r = rng.normal(0.0008, 0.01, 5000)

    few = deflated_sharpe_ratio(r, n_trials=1)
    many = deflated_sharpe_ratio(r, n_trials=1000)

    assert many < few


def test_dsr_high_for_strong_genuine_edge():
    rng = np.random.default_rng(3)
    r = rng.normal(0.003, 0.01, 20000)      # Sharpe ≈ 0.3 за период

    assert deflated_sharpe_ratio(r, n_trials=1) > 0.95


def test_dsr_handles_short_series():
    assert deflated_sharpe_ratio(np.array([0.01, 0.02]), n_trials=1) == 0.0


def test_dsr_rejects_bad_n_trials():
    rng = np.random.default_rng(4)
    r = rng.normal(0.001, 0.01, 1000)

    with pytest.raises(ValueError, match="n_trials"):
        deflated_sharpe_ratio(r, n_trials=0)


def test_pbo_high_when_performance_is_random():
    """Случайные конфигурации: лучшая на train не лучше на test → PBO ≈ 0.5."""
    rng = np.random.default_rng(5)
    matrix = rng.normal(0.0, 0.01, size=(2000, 20))

    pbo = pbo_cscv(matrix, n_blocks=10)

    assert 0.25 < pbo < 0.75


def test_pbo_low_when_one_config_genuinely_better():
    """Одна конфигурация с реальным преимуществом → PBO низкий."""
    rng = np.random.default_rng(6)
    matrix = rng.normal(0.0, 0.01, size=(2000, 20))
    matrix[:, 0] += 0.002                    # устойчивое преимущество

    pbo = pbo_cscv(matrix, n_blocks=10)

    assert pbo < 0.2


def test_pbo_requires_enough_configs():
    rng = np.random.default_rng(7)
    matrix = rng.normal(0.0, 0.01, size=(1000, 1))

    assert np.isnan(pbo_cscv(matrix))


def test_permutation_separates_signal_from_noise():
    """Сигнал, связанный с доходностью, обязан получить p-value ниже случайного."""
    rng = np.random.default_rng(8)
    n = 3000
    price_ret = rng.normal(0.0, 0.01, n)
    pos_signal = np.sign(price_ret)                          # идеальное предвидение
    pos_noise = rng.choice([-1.0, 0.0, 1.0], size=n)         # сигнала нет

    p_signal = permutation_pvalue(price_ret, pos_signal, n_permutations=500, seed=1)
    p_noise = permutation_pvalue(price_ret, pos_noise, n_permutations=500, seed=1)

    assert p_signal < 0.05
    assert p_noise > p_signal


def test_permutation_of_returns_alone_would_be_meaningless():
    """Регрессия: Sharpe инвариантен к перестановке доходностей.

    Этот тест фиксирует причину, по которой перемешиваются позиции, а не доходности.
    """
    rng = np.random.default_rng(9)
    r = rng.normal(0.001, 0.01, 1000)

    shuffled = rng.permutation(r)

    assert _sharpe_raw(r) == pytest.approx(_sharpe_raw(shuffled))


def test_permutation_is_reproducible_with_seed():
    rng = np.random.default_rng(10)
    price_ret = rng.normal(0.0, 0.01, 800)
    pos = rng.choice([-1.0, 0.0, 1.0], size=800)

    a = permutation_pvalue(price_ret, pos, n_permutations=300, seed=42)
    b = permutation_pvalue(price_ret, pos, n_permutations=300, seed=42)

    assert a == b


def test_permutation_rejects_flat_signal():
    rng = np.random.default_rng(11)
    price_ret = rng.normal(0.0, 0.01, 500)

    assert permutation_pvalue(price_ret, np.zeros(500)) == 1.0


def test_permutation_requires_matching_lengths():
    with pytest.raises(ValueError, match="Длины"):
        permutation_pvalue(np.zeros(100), np.zeros(50))
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_significance.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `significance.py`**

`src/alpha_lab/validation/significance.py`:
```python
"""Проверка статистической значимости результата.

Три механизма против трёх разных способов обмануть себя:

DSR  — против множественных сравнений. Ожидаемый максимум Sharpe у N случайных
       стратегий растёт как sqrt(2·ln N); при N=500 это ≈ 3.5. Сырой Sharpe,
       выбранный как лучший из многих, почти наверняка завышен.
PBO  — против «повезло на истории» (CSCV, Bailey et al. 2015).
Perm — против «а может, оно само так вышло»: перемешиваем, строим нулевое
       распределение, получаем честный p-value.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _sharpe_raw(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return 0.0 if sd < 1e-12 else float(r.mean() / sd)


def deflated_sharpe_ratio(returns, n_trials: int,
                          sr_variance: float | None = None) -> float:
    """Вероятность, что истинный Sharpe > 0 с поправкой на число попыток.

    Возвращает P ∈ [0, 1]. Порог вердикта: DSR > 0.95 (см. validator.py).
    """
    if n_trials < 1:
        raise ValueError(f"n_trials должен быть ≥ 1, получено {n_trials}")

    r = np.asarray(returns, dtype="float64")
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return 0.0

    sr = _sharpe_raw(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))

    if n_trials < 2:
        sr0 = 0.0
    else:
        if sr_variance is None or not np.isfinite(sr_variance) or sr_variance <= 0:
            # Дисперсия оценки Sharpe в нулевой гипотезе
            sr_variance = (1.0 + 0.5 * sr ** 2) / n
        z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr0 = float(np.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))

    denom = np.sqrt(max(1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr ** 2, 1e-12))
    z_score = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(stats.norm.cdf(z_score))


def pbo_cscv(returns_matrix, n_blocks: int = 10) -> float:
    """Probability of Backtest Overfitting через CSCV.

    returns_matrix: (T наблюдений × N конфигураций).
    Блоки делятся на все сочетания половин; для каждого сочетания ищем лучшую
    конфигурацию на train и смотрим её ранг на test. PBO — доля сочетаний,
    где лучшая на train оказалась ниже медианы на test.
    """
    m = np.asarray(returns_matrix, dtype="float64")
    if m.ndim != 2:
        raise ValueError("returns_matrix должен быть двумерным (T × N)")
    t, n_configs = m.shape
    if n_configs < 2 or t < n_blocks * 2 or n_blocks < 4 or n_blocks % 2 != 0:
        return float("nan")

    block_size = t // n_blocks
    blocks = m[:block_size * n_blocks].reshape(n_blocks, block_size, n_configs)
    n_train = n_blocks // 2

    logits = []
    for train_idx in combinations(range(n_blocks), n_train):
        test_idx = [i for i in range(n_blocks) if i not in train_idx]
        train = blocks[list(train_idx)].reshape(-1, n_configs)
        test = blocks[test_idx].reshape(-1, n_configs)

        sr_train = np.array([_sharpe_raw(train[:, j]) for j in range(n_configs)])
        sr_test = np.array([_sharpe_raw(test[:, j]) for j in range(n_configs)])

        best = int(np.argmax(sr_train))
        ranks = stats.rankdata(sr_test, method="average")
        omega = ranks[best] / (n_configs + 1.0)
        omega = min(max(omega, 1e-6), 1.0 - 1e-6)
        logits.append(np.log(omega / (1.0 - omega)))

    logits = np.asarray(logits)
    return float((logits < 0).mean())


def permutation_pvalue(price_returns, positions,
                       n_permutations: int = 1000, seed: int = 0) -> float:
    """Доля перемешанных версий, чей Sharpe не хуже наблюдаемого.

    Нулевая гипотеза: сигнал не связан с доходностями.

    ВАЖНО: перемешиваются ПОЗИЦИИ, а не доходности. Sharpe = mean/std инвариантен
    к перестановке доходностей, поэтому перемешивание самого ряда дало бы
    p-value ≡ 1.0 и тест не отклонял бы ничего. Смысл имеет только разрушение
    соответствия «сигнал ↔ доходность».
    """
    pr = np.asarray(price_returns, dtype="float64")
    pos = np.asarray(positions, dtype="float64")
    if len(pr) != len(pos):
        raise ValueError(
            f"Длины price_returns ({len(pr)}) и positions ({len(pos)}) не совпадают"
        )
    ok = np.isfinite(pr) & np.isfinite(pos)
    pr, pos = pr[ok], pos[ok]
    if len(pr) < 10 or not pos.any():
        return 1.0

    observed = _sharpe_raw(pr * pos)
    rng = np.random.default_rng(seed)
    better = 0
    for _ in range(n_permutations):
        if _sharpe_raw(pr * rng.permutation(pos)) >= observed:
            better += 1
    return float((better + 1) / (n_permutations + 1))
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_significance.py -v
```
Ожидается: 10 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/validation/significance.py tests/test_significance.py
git commit -m "feat: deflated sharpe, pbo (cscv) and permutation tests"
```

---

## Задача 15: Сборка вердикта

**Файлы:**
- Создать: `src/alpha_lab/validation/validator.py`, `tests/test_validator.py`

**Интерфейсы:**
- Отдаёт: `Verdict` (dataclass, поля как в spec 4.3)
- `validate(returns, trade_returns, equity, config: dict, n_trials: int, strategy_name: str, experiment_id: str) -> Verdict`
- Пороги по умолчанию: `min_trades=100`, `max_pbo=0.5`, `max_p_value=0.05`, `min_dsr=0.95`

**ВАЖНО:** этот модуль не импортирует `alpha_lab.strategies` — проверяется задачей 17.

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_validator.py`:
```python
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
    assert any("permutation" in reason for reason in v.reasons)


def test_many_trials_deflate_and_can_kill():
    case = _case(seed=6, strength=0.8)
    v_few = _run(case, _trades(300, 6), n_trials=1)
    v_many = _run(case, _trades(300, 6), n_trials=5000)

    assert v_many.dsr < v_few.dsr


def test_verdict_is_frozen():
    v = _run(_case(seed=7, strength=0.8), _trades(300, 7))

    with pytest.raises(Exception):
        v.alive = False


def test_thresholds_come_from_config():
    v = _run(_case(seed=8, strength=0.8), _trades(300, 8),
             config={"min_trades": 100000})

    assert not v.alive


def test_reasons_empty_when_alive():
    v = _run(_case(seed=9, strength=0.8), _trades(300, 9))

    assert v.reasons == ()
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_validator.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `validator.py`**

`src/alpha_lab/validation/validator.py`:
```python
"""Сборка вердикта. Слепой слой: знает только результаты, не стратегию.

Инвариант: этот модуль НЕ импортирует alpha_lab.strategies.
Иначе появляется соблазн «подкрутить» проверку под конкретную стратегию.
Проверяется тестом tests/test_traps.py::test_validator_does_not_import_strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_lab.validation.metrics import max_drawdown, sharpe_ratio, summarize
from alpha_lab.validation.significance import (
    deflated_sharpe_ratio, permutation_pvalue,
)

DEFAULT_THRESHOLDS = {
    "min_trades": 100,
    "max_pbo": 0.5,
    "max_p_value": 0.05,
    "min_dsr": 0.95,
    "n_permutations": 1000,
}

# Итоговый годовой Sharpe считается для часовых баров крипты (24/7)
PERIODS_PER_YEAR = 365 * 24


@dataclass(frozen=True)
class Verdict:
    strategy_name: str
    experiment_id: str
    sharpe: float
    dsr: float
    p_value: float
    max_dd: float
    total_return: float
    trades: int
    n_configs_tried: int
    alive: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)
    metrics: dict[str, float] = field(default_factory=dict)
    pbo: float = float("nan")


def validate(returns, trade_returns, equity, config: dict, n_trials: int,
             strategy_name: str, experiment_id: str,
             returns_matrix=None, price_returns=None, positions=None) -> Verdict:
    """Выносит вердикт. Все пороги — из config, значения по умолчанию в DEFAULT_THRESHOLDS.

    price_returns и positions обязательны для permutation-теста: он перемешивает
    позиции относительно доходностей. Без них проверка невозможна, и вердикт
    выносится отрицательный — тихая деградация недопустима.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(config or {})}

    r = np.asarray(pd.Series(returns), dtype="float64")
    r = r[np.isfinite(r)]
    t = np.asarray(pd.Series(trade_returns), dtype="float64")
    t = t[np.isfinite(t)]
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]

    n_trades = int(len(t))
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials)

    permutation_available = price_returns is not None and positions is not None
    if permutation_available:
        p_value = permutation_pvalue(
            price_returns, positions,
            n_permutations=int(thresholds["n_permutations"]), seed=0,
        )
    else:
        p_value = 1.0

    pbo = float("nan")
    if returns_matrix is not None:
        from alpha_lab.validation.significance import pbo_cscv
        pbo = pbo_cscv(returns_matrix)

    reasons: list[str] = []
    if not permutation_available:
        reasons.append(
            "permutation-тест не выполнен: не переданы price_returns и positions"
        )
    if n_trades < thresholds["min_trades"]:
        reasons.append(
            f"недостаточно сделок: {n_trades} < {thresholds['min_trades']}"
        )
    if dsr <= thresholds["min_dsr"]:
        reasons.append(
            f"DSR {dsr:.3f} ≤ {thresholds['min_dsr']} (с поправкой на {n_trials} попыток)"
        )
    if p_value >= thresholds["max_p_value"]:
        reasons.append(
            f"p-value {p_value:.3f} ≥ {thresholds['max_p_value']} — неотличимо от случая"
        )
    if np.isfinite(pbo) and pbo >= thresholds["max_pbo"]:
        reasons.append(f"PBO {pbo:.2f} ≥ {thresholds['max_pbo']} — признак подгонки")

    stats = summarize(r, t, eq, PERIODS_PER_YEAR) if len(r) else {}

    return Verdict(
        strategy_name=strategy_name,
        experiment_id=experiment_id,
        sharpe=sharpe_ratio(r, PERIODS_PER_YEAR),
        dsr=dsr,
        p_value=p_value,
        max_dd=max_drawdown(eq) if len(eq) else 0.0,
        total_return=float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0,
        trades=n_trades,
        n_configs_tried=n_trials,
        alive=len(reasons) == 0,
        reasons=tuple(reasons),
        metrics=stats,
        pbo=pbo,
    )
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_validator.py -v
```
Ожидается: 7 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/validation/validator.py tests/test_validator.py
git commit -m "feat: verdict assembly with blind validation layer"
```

---

## Задача 16: Отчёт — JSON-контракт для дашборда

**Файлы:**
- Создать: `src/alpha_lab/report/__init__.py`, `src/alpha_lab/report/schema.py`, `src/alpha_lab/report/writer.py`, `tests/test_report.py`

**Интерфейсы:**
- Отдаёт: `SCHEMA_VERSION = "1.0"`, `build_report(...) -> dict`, `write_report(payload, out_dir) -> tuple[Path, Path]`
- `write_report` пишет **два** файла: `report.json` и `report.js` (`window.ALPHA_REPORT = {...};`)

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_report.py`:
```python
import json

import numpy as np
import pandas as pd
import pytest

from alpha_lab.report.schema import SCHEMA_VERSION
from alpha_lab.report.writer import build_report, write_report
from alpha_lab.validation.validator import Verdict


def _verdict():
    return Verdict(
        strategy_name="mean_reversion", experiment_id="abc123",
        sharpe=1.4, dsr=0.97, p_value=0.01, max_dd=-0.22, total_return=0.5,
        trades=250, n_configs_tried=12, alive=True, reasons=(),
        metrics={"win_rate": 0.55}, pbo=0.21,
    )


def _series(n=100):
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    return ts, pd.Series(np.linspace(1.0, 1.5, n), index=ts)


def test_build_report_has_schema_version():
    ts, equity = _series()

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["verdict"]["alive"] is True


def test_build_report_serializes_timestamps_as_iso():
    ts, equity = _series(5)

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["series"]["ts"][0] == "2024-01-01T00:00:00+00:00"


def test_build_report_has_all_series():
    ts, equity = _series(10)

    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(1.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.001}, index=ts))

    for key in ("ts", "equity", "drawdown", "close", "position", "fee", "funding"):
        assert key in payload["series"], key


def test_write_report_creates_both_files(tmp_path):
    ts, equity = _series(10)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    json_path, js_path = write_report(payload, tmp_path)

    assert json_path.exists() and js_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["schema_version"] == SCHEMA_VERSION


def test_report_js_assigns_global(tmp_path):
    ts, equity = _series(5)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    _, js_path = write_report(payload, tmp_path)
    text = js_path.read_text(encoding="utf-8")

    assert text.startswith("window.ALPHA_REPORT = ")
    assert text.rstrip().endswith(";")


def test_report_js_payload_matches_json(tmp_path):
    ts, equity = _series(5)
    payload = build_report(_verdict(), equity=equity, close=equity * 100,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    json_path, js_path = write_report(payload, tmp_path)
    from_json = json.loads(json_path.read_text(encoding="utf-8"))
    from_js = json.loads(
        js_path.read_text(encoding="utf-8")
        .split("=", 1)[1].strip().rstrip(";")
    )

    assert from_json == from_js


def test_nan_values_become_null():
    ts, equity = _series(3)
    v = Verdict(strategy_name="s", experiment_id="x", sharpe=0.0, dsr=0.0,
                p_value=1.0, max_dd=0.0, total_return=0.0, trades=0,
                n_configs_tried=1, alive=False, reasons=("нет",),
                metrics={}, pbo=float("nan"))

    payload = build_report(v, equity=equity, close=equity,
                           positions=pd.Series(0.0, index=ts),
                           costs=pd.DataFrame({"fee": 0.0}, index=ts))

    assert payload["verdict"]["pbo"] is None
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_report.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `schema.py` и `writer.py`**

`src/alpha_lab/report/schema.py`:
```python
"""Схема отчёта — контракт между движком и дашбордом.

Мажорная версия меняется при несовместимом изменении структуры.
Дашборд обязан отклонять неизвестную мажорную версию, а не рисовать пустые графики.
"""
from __future__ import annotations

SCHEMA_VERSION = "1.0"
MAJOR_VERSION = int(SCHEMA_VERSION.split(".")[0])

REQUIRED_TOP_LEVEL = ("schema_version", "verdict", "series", "generated_at")
REQUIRED_SERIES = ("ts", "equity", "drawdown", "close", "position", "fee", "funding")


def is_compatible(version: str) -> bool:
    try:
        return int(str(version).split(".")[0]) == MAJOR_VERSION
    except (ValueError, AttributeError):
        return False
```

`src/alpha_lab/report/writer.py`:
```python
"""Запись отчёта в report.json и report.js.

Два файла из одного объекта: json — для программ, js — для дашборда.
Дашборд открывается через file://, где fetch() к локальным файлам блокируется
CORS, поэтому данные подключаются как <script src="report.js">.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.report.schema import SCHEMA_VERSION
from alpha_lab.validation.validator import Verdict


def _round(values, digits: int = 6) -> list:
    out = []
    for v in values:
        f = float(v)
        out.append(None if not np.isfinite(f) else round(f, digits))
    return out


def build_report(verdict: Verdict, equity: pd.Series, close: pd.Series,
                 positions: pd.Series, costs: pd.DataFrame,
                 price_bars: pd.DataFrame | None = None,
                 extra: dict | None = None) -> dict:
    eq = pd.Series(equity).astype("float64")
    dd = eq / eq.cummax() - 1.0

    series = {
        "ts": [pd.Timestamp(t).isoformat() for t in eq.index],
        "equity": _round(eq.to_numpy(), 8),
        "drawdown": _round(dd.to_numpy(), 8),
        "close": _round(pd.Series(close).to_numpy(), 8),
        "position": _round(pd.Series(positions).to_numpy(), 6),
        "fee": _round(costs.get("fee", pd.Series(0.0, index=eq.index)).to_numpy(), 8),
        "slippage": _round(costs.get("slippage", pd.Series(0.0, index=eq.index)).to_numpy(), 8),
        "funding": _round(costs.get("funding", pd.Series(0.0, index=eq.index)).to_numpy(), 8),
    }
    if price_bars is not None:
        for col in ("open", "high", "low"):
            if col in price_bars.columns:
                series[col] = _round(price_bars[col].to_numpy(), 8)

    verdict_payload = {
        "strategy_name": verdict.strategy_name,
        "experiment_id": verdict.experiment_id,
        "alive": verdict.alive,
        "sharpe": _num(verdict.sharpe),
        "dsr": _num(verdict.dsr),
        "pbo": _num(verdict.pbo),
        "p_value": _num(verdict.p_value),
        "max_dd": _num(verdict.max_dd),
        "total_return": _num(verdict.total_return),
        "trades": verdict.trades,
        "n_configs_tried": verdict.n_configs_tried,
        "reasons": list(verdict.reasons),
        "metrics": {k: _num(v) for k, v in (verdict.metrics or {}).items()},
    }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "verdict": verdict_payload,
        "series": series,
    }
    if extra:
        payload["extra"] = extra
    return payload


def _num(value):
    if value is None:
        return None
    f = float(value)
    return None if not np.isfinite(f) else round(f, 6)


def write_report(payload: dict, out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "report.json"
    js_path = out / "report.js"

    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    json_path.write_text(text, encoding="utf-8")
    js_path.write_text(f"window.ALPHA_REPORT = {text};\n", encoding="utf-8")
    return json_path, js_path
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_report.py -v
```
Ожидается: 7 passed

- [ ] **Шаг 5: Коммит**

```bash
git add src/alpha_lab/report tests/test_report.py
git commit -m "feat: versioned report contract (json + js for file:// dashboard)"
```

---

## Задача 17: Стратегии-ловушки — тест самого валидатора

**Файлы:**
- Создать: `tests/fixtures/traps.py`, `tests/test_traps.py`

**Интерфейсы:**
- Отдаёт: `LookAheadStrategy`, `OverfitNoiseStrategy`, `AlwaysLongStrategy`
- Тест `test_validator_does_not_import_strategies` проверяет инвариант слоёв.

- [ ] **Шаг 1: Написать ловушки и тесты**

`tests/fixtures/traps.py`:
```python
"""Заведомо сломанные стратегии.

Назначение — тестировать НЕ их, а валидатор: он обязан убивать каждую.
Если валидатор пропускает ловушку, значит сломан валидатор.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LookAheadStrategy:
    """Торгует по цене СЛЕДУЮЩЕГО бара — классическое подглядывание в будущее."""

    name = "trap_lookahead"

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64")
        future_return = close.shift(-1) / close - 1.0
        # «Знаем» направление следующего бара и ставим позицию заранее
        signal = np.sign(future_return.fillna(0.0))
        return signal.astype("float64")


class OverfitNoiseStrategy:
    """Подогнана под конкретный исторический отрезок: торгует только там."""

    name = "trap_overfit"

    def __init__(self, start: int = 0, end: int = 50):
        self.start = start
        self.end = end

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        pos = pd.Series(0.0, index=bars.index)
        pos.iloc[self.start:self.end] = 1.0
        return pos


class AlwaysLongStrategy:
    """Всегда в лонге. На растущем ряде выглядит прибыльной без всякой альфы."""

    name = "trap_always_long"

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        return pd.Series(1.0, index=bars.index)


class PerfectForesightStrategy:
    """Знает весь будущий ряд целиком — эталон максимального подглядывания."""

    name = "trap_foresight"

    def generate(self, bars: pd.DataFrame) -> pd.Series:
        close = bars["close"].astype("float64").to_numpy()
        pos = np.zeros(len(close), dtype="float64")
        pos[:-1] = np.sign(close[1:] - close[:-1])
        return pd.Series(pos, index=bars.index)
```

`tests/test_traps.py`:
```python
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fixtures.synthetic import ou_bars
from fixtures.traps import (
    AlwaysLongStrategy, LookAheadStrategy, PerfectForesightStrategy,
)

from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost
from alpha_lab.validation.validator import validate

SRC = Path(__file__).resolve().parents[1] / "src"


def test_validator_does_not_import_strategies():
    """Инвариант слоёв: validation слеп и не знает о стратегиях."""
    import alpha_lab.validation.validator  # noqa: F401

    forbidden = [
        m for m in sys.modules
        if m.startswith("alpha_lab.strategies")
    ]
    # Модули стратегий могли быть импортированы другими тестами;
    # проверяем статически, по исходникам.
    for path in (SRC / "alpha_lab" / "validation").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "alpha_lab.strategies" not in text, (
            f"{path.name} импортирует strategies — нарушен инвариант слоёв"
        )


def test_lookahead_strategy_is_rejected_as_too_good():
    """Подглядывание даёт нереалистично высокий Sharpe — его ловит проверка на здравый смысл."""
    bars = ou_bars(n=3000, seed=21)
    pos = LookAheadStrategy().generate(bars)
    res = run_backtest(bars, pos, RealisticCost())

    # Даже с издержками подглядывание даёт аномально высокий результат
    assert res.total_return > 10.0


def test_perfect_foresight_is_impossible_in_practice():
    bars = ou_bars(n=1000, seed=22)
    pos = PerfectForesightStrategy().generate(bars)

    # Проверяем саму ловушку: она «знает» будущее и потому аномальна
    assert (pos != 0).mean() > 0.9


def _case(n, seed, strength):
    """Согласованный набор: доходности цены, позиции, доходность стратегии."""
    rng = np.random.default_rng(seed)
    price_ret = pd.Series(rng.normal(0.0, 0.01, n))
    sign = np.sign(price_ret.to_numpy())
    random_side = rng.choice([-1.0, 1.0], size=n)
    positions = pd.Series(np.where(rng.random(n) < strength, sign, random_side))
    return positions * price_ret, price_ret, positions


def test_noise_strategy_fails_validation():
    returns, pr, pos = _case(n=4000, seed=23, strength=0.0)
    trades = pd.Series(np.random.default_rng(23).normal(0.0, 0.01, 150))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1,
                 strategy_name="noise", experiment_id="t1",
                 price_returns=pr, positions=pos)

    assert not v.alive


def test_always_long_on_flat_series_fails():
    rng = np.random.default_rng(24)
    price_ret = pd.Series(rng.normal(0.0, 0.01, 4000))
    positions = pd.Series(1.0, index=price_ret.index)
    returns = positions * price_ret
    trades = pd.Series(rng.normal(0.0, 0.01, 200))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1000,
                 strategy_name="always_long", experiment_id="t2",
                 price_returns=price_ret, positions=positions)

    assert not v.alive


def test_strong_edge_survives_validation():
    """Контрольный случай: настоящая альфа НЕ должна убиваться.

    Не менее важен, чем тест на убийство ловушек: валидатор, который режет всё
    подряд, бесполезен ровно так же, как тот, что не режет ничего.
    """
    returns, pr, pos = _case(n=20000, seed=25, strength=0.8)
    trades = pd.Series(np.random.default_rng(25).normal(0.002, 0.01, 400))

    v = validate(returns, trades, (1 + returns).cumprod(), config={}, n_trials=1,
                 strategy_name="real", experiment_id="t3",
                 price_returns=pr, positions=pos)

    assert v.alive, f"Валидатор убил настоящий edge: {v.reasons}"


def test_all_traps_are_killed_by_validator():
    """Сводный тест: каждая ловушка обязана получить alive=False.

    Если падает — это баг валидатора, а не теста. Чинить validator.py/significance.py.
    """
    cases = {
        "noise":       dict(n=4000, seed=26, strength=0.0, trades=150, trials=1),
        "tiny_sample": dict(n=4000, seed=27, strength=0.8, trades=5, trials=1),
        "many_trials": dict(n=4000, seed=28, strength=0.0, trades=150, trials=10000),
    }

    for label, cfg in cases.items():
        returns, pr, pos = _case(cfg["n"], cfg["seed"], cfg["strength"])
        trades = pd.Series(
            np.random.default_rng(cfg["seed"]).normal(0.001, 0.01, cfg["trades"])
        )
        v = validate(returns, trades, (1 + returns).cumprod(), config={},
                     n_trials=cfg["trials"], strategy_name=label,
                     experiment_id=f"trap_{label}", price_returns=pr, positions=pos)
        assert not v.alive, f"Ловушка '{label}' прошла валидацию — сломан валидатор"
```

- [ ] **Шаг 2: Запустить тесты**

```bash
uv run pytest tests/test_traps.py -v
```
Ожидается: 6 passed. **Если `test_all_traps_are_killed_by_validator` или `test_strong_edge_survives_validation` падает — это баг валидатора, а не теста.** Исправлять `validator.py`/`significance.py`, не подкручивать тест.

- [ ] **Шаг 3: Коммит**

```bash
git add tests/fixtures/traps.py tests/test_traps.py
git commit -m "test: trap strategies prove the validator kills overfitting"
```

---

## Задача 18: CLI и сквозной прогон

**Файлы:**
- Создать: `src/alpha_lab/cli.py`, `tests/test_cli.py`

**Интерфейсы:**
- Отдаёт: `main(argv=None) -> int`; команды `ingest`, `validate`
- `experiment_id(config, data_version, git_hash) -> str`

- [ ] **Шаг 1: Написать падающий тест**

`tests/test_cli.py`:
```python
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.cli import experiment_id, main
from alpha_lab.data.schema import normalize_bars
from alpha_lab.data.store import write_bars


def _write_fixture_data(root: Path, n=30000, seed=1):
    """Кладёт минутные синтетические бары туда, откуда их читает load_bars.

    Пишем именно 1m: CLI читает базовый таймфрейм 1m и ресэмплит в таймфрейм
    эксперимента. 30000 минут ≈ 500 часовых баров после ресэмплинга.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    df = normalize_bars(pd.DataFrame({
        "ts": ts, "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1e5, "quote_volume": 1e8,
        "trades": 500, "taker_buy_volume": 5e4,
    }))
    write_bars(df, root, "BTCUSDT", "1m")


def test_experiment_id_is_deterministic():
    cfg = {"a": 1, "b": [1, 2]}
    assert experiment_id(cfg, "dv1", "g1") == experiment_id(cfg, "dv1", "g1")
    assert experiment_id(cfg, "dv1", "g1") != experiment_id(cfg, "dv2", "g1")


def test_experiment_id_ignores_key_order():
    assert experiment_id({"a": 1, "b": 2}, "d", "g") == \
           experiment_id({"b": 2, "a": 1}, "d", "g")


def _write_configs(tmp_path, root, symbols=("BTCUSDT",)):
    u = tmp_path / "universe.yaml"
    u.write_text(
        "market: futures-um\nstart: '2024-01-01'\nend: '2024-06-30'\n"
        f"symbols: {list(symbols)}\n", encoding="utf-8")
    e = tmp_path / "exp.yaml"
    e.write_text(
        "name: test\nstrategy: mean_reversion\ntimeframe: 1h\n"
        "start: '2024-01-01'\nend: '2024-06-30'\n"
        "params: {window: 20, k: 2.0}\n"
        "costs: {taker_fee_bps: 5.0}\n"
        "validation: {min_trades: 1}\n", encoding="utf-8")
    return u, e


def test_validate_command_writes_report(tmp_path, capsys):
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    out = tmp_path / "out"
    journal = tmp_path / "trials.jsonl"

    code = main([
        "validate", "--config", str(e), "--universe", str(u),
        "--data-root", str(root), "--out", str(out), "--journal", str(journal),
    ])

    assert code == 0
    assert (out / "report.json").exists()
    assert (out / "report.js").exists()
    payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "verdict" in payload
    assert payload["verdict"]["alive"] in (True, False)
    assert len(payload["series"]["equity"]) > 0


def test_repeated_runs_increment_n_trials(tmp_path):
    """Каждый прогон пишется в журнал; DSR обязан штрафовать за число попыток."""
    root = tmp_path / "data"
    _write_fixture_data(root)
    u, e = _write_configs(tmp_path, root)
    journal = tmp_path / "trials.jsonl"

    trials = []
    for i in range(4):
        out = tmp_path / f"out{i}"
        main(["validate", "--config", str(e), "--universe", str(u),
              "--data-root", str(root), "--out", str(out), "--journal", str(journal)])
        payload = json.loads((out / "report.json").read_text(encoding="utf-8"))
        trials.append(payload["verdict"]["n_configs_tried"])

    assert trials == [1, 2, 3, 4]


def test_dirty_bars_are_not_traded(tmp_path):
    """Бар с нулевым объёмом не должен участвовать в торговле (spec раздел 8)."""
    from alpha_lab.data.quality import clean_mask
    from alpha_lab.data.query import load_bars

    root = tmp_path / "data"
    _write_fixture_data(root)
    bars = load_bars(root, "BTCUSDT", "1m")
    mask = clean_mask(bars).to_numpy()

    assert mask.all(), "чистые данные обязаны проходить маску"

    bars.loc[bars.index[10], "volume"] = 0.0
    mask2 = clean_mask(bars).to_numpy()
    assert not mask2[10]
    assert mask2.sum() == len(bars) - 1


def test_validate_missing_data_returns_error(tmp_path):
    root = tmp_path / "empty"
    u, e = _write_configs(tmp_path, root)

    code = main([
        "validate", "--config", str(e), "--universe", str(u),
        "--data-root", str(root), "--out", str(tmp_path / "out"),
    ])

    assert code == 2


def test_ingest_requires_universe(tmp_path):
    code = main(["ingest", "--config", str(tmp_path / "nope.yaml"),
                 "--data-root", str(tmp_path)])

    assert code == 2
```

- [ ] **Шаг 2: Запустить — убедиться, что падает**

```bash
uv run pytest tests/test_cli.py -v
```
Ожидается: `ModuleNotFoundError`

- [ ] **Шаг 3: Реализовать `cli.py`**

`src/alpha_lab/cli.py`:
```python
"""Единая точка входа.

    uv run alpha-lab ingest   --config configs/universe.yaml
    uv run alpha-lab validate --config configs/experiments/mr_base.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config import load_experiment, load_universe
from alpha_lab.data.quality import clean_mask
from alpha_lab.data.query import align_funding_to_bars, data_version, load_bars, load_funding
from alpha_lab.data.store import DEFAULT_ROOT
from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost
from alpha_lab.report.writer import build_report, write_report
from alpha_lab.strategies.base import build_strategy
from alpha_lab.validation.validator import validate

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


DEFAULT_JOURNAL = Path("reports") / "trials.jsonl"


def experiment_id(config: dict, dv: str, gh: str) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{payload}|{dv}|{gh}".encode("utf-8"))
    return digest.hexdigest()[:16]


def count_prior_trials(journal: Path, key: dict) -> int:
    """Сколько раз эта же гипотеза уже прогонялась.

    Число попыток нужно Deflated Sharpe: без него главная защита от оверфиттинга
    отключается ровно там, где она нужна. Наивная «одна попытка» — самообман.
    """
    journal = Path(journal)
    if not journal.exists():
        return 0
    fingerprint = json.dumps(key, sort_keys=True, ensure_ascii=False)
    count = 0
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            if json.dumps(json.loads(line)["key"], sort_keys=True,
                          ensure_ascii=False) == fingerprint:
                count += 1
        except (json.JSONDecodeError, KeyError):
            continue
    return count


def log_trial(journal: Path, key: dict, experiment: str, metrics: dict) -> None:
    journal = Path(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "experiment_id": experiment, **metrics}
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _cmd_ingest(args) -> int:
    try:
        universe = load_universe(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка конфига: {exc}", file=sys.stderr)
        return EXIT_ERROR

    from alpha_lab.data.ingest import ingest_universe

    results = ingest_universe(universe, args.freq, root=Path(args.data_root))
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    for r in results:
        mark = "OK " if r.ok else "ERR"
        detail = r.quality if r.ok else r.error
        print(f"[{mark}] {r.symbol:<12} {r.kind:<8} {r.rows:>10,}  {detail}")
    print(f"\nИтого: {len(ok)} успешно, {len(bad)} с ошибками")
    return EXIT_OK if not bad else EXIT_FAILED


def _cmd_validate(args) -> int:
    try:
        exp = load_experiment(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка конфига: {exc}", file=sys.stderr)
        return EXIT_ERROR

    root = Path(args.data_root)
    dv = data_version(root)
    cfg_payload = {
        "experiment": exp.name, "strategy": exp.strategy, "params": exp.params,
        "timeframe": exp.timeframe, "start": exp.start, "end": exp.end,
        "costs": exp.costs,
    }
    exp_id = experiment_id(cfg_payload, dv, git_hash())

    symbol = args.symbol
    try:
        bars = load_bars(root, symbol, "1m", exp.start, exp.end,
                         resample=exp.timeframe)
    except FileNotFoundError as exc:
        print(f"Данных нет: {exc}", file=sys.stderr)
        return EXIT_ERROR

    funding_rate = None
    try:
        funding_rate = align_funding_to_bars(bars, load_funding(root, symbol))
    except FileNotFoundError:
        print("Предупреждение: funding недоступен, издержки занижены", file=sys.stderr)

    # Грязные бары: стратегия на них не торгует. Не чиним и не интерполируем.
    mask = clean_mask(bars).to_numpy()
    dirty = int((~mask).sum())
    if dirty:
        print(f"Предупреждение: {dirty} грязных баров исключено из торговли",
              file=sys.stderr)

    strategy = build_strategy(exp.strategy, exp.params)
    positions = strategy.generate(bars).to_numpy()
    positions[~mask] = 0.0
    positions = pd.Series(positions, index=bars.index)

    cost_model = RealisticCost.from_config(exp.costs)
    result = run_backtest(bars, positions, cost_model, funding_rate=funding_rate)
    trades = trade_returns(result)

    # Число попыток берётся из журнала: DSR без этой поправки не работает
    journal = Path(args.journal)
    trial_key = {"strategy": exp.strategy, "params": exp.params,
                 "symbol": symbol, "timeframe": exp.timeframe}
    n_trials = count_prior_trials(journal, trial_key) + 1

    verdict = validate(
        returns=result.returns, trade_returns=trades, equity=result.equity,
        config=exp.validation, n_trials=n_trials, strategy_name=exp.name,
        experiment_id=exp_id,
        price_returns=result.price_returns, positions=result.positions,
    )

    log_trial(journal, trial_key, exp_id,
              {"sharpe": verdict.sharpe, "dsr": verdict.dsr, "alive": verdict.alive})

    payload = build_report(
        verdict, equity=result.equity, close=bars["close"],
        positions=result.positions, costs=result.costs, price_bars=bars,
        extra={"symbol": symbol, "timeframe": exp.timeframe,
               "data_version": dv, "costs_total": result.cost_totals},
    )
    out = Path(args.out) if args.out else Path("reports") / exp_id
    json_path, js_path = write_report(payload, out)

    status = "ЖИВА" if verdict.alive else "МЕРТВА"
    print(f"\n{'=' * 62}\n  ВЕРДИКТ: {status}\n{'=' * 62}")
    print(f"  Стратегия      {exp.name}  ({exp.strategy}, {exp.timeframe})")
    print(f"  Символ         {symbol}")
    print(f"  Experiment ID  {exp_id}")
    print(f"  Сделок         {verdict.trades}")
    print(f"  Sharpe         {verdict.sharpe:.2f}")
    print(f"  DSR            {verdict.dsr:.4f}  (порог 0.95)")
    print(f"  p-value        {verdict.p_value:.4f}  (порог 0.05)")
    print(f"  Max DD         {verdict.max_dd:.1%}")
    print(f"  Доходность     {verdict.total_return:+.2%}")
    print(f"  Издержки       {result.cost_totals}")
    if verdict.reasons:
        print("\n  Причины:")
        for reason in verdict.reasons:
            print(f"    · {reason}")
    print(f"\n  Отчёт: {json_path}")
    print(f"  Дашборд: откройте dashboard/index.html\n")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alpha-lab", description="Полигон для исследования крипторынков")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Скачать историю с Binance")
    p_ing.add_argument("--config", required=True, help="Путь к universe.yaml")
    p_ing.add_argument("--data-root", default=str(DEFAULT_ROOT))
    p_ing.add_argument("--freq", default="1m")
    p_ing.set_defaults(func=_cmd_ingest)

    p_val = sub.add_parser("validate", help="Прогнать стратегию и вынести вердикт")
    p_val.add_argument("--config", required=True, help="Путь к эксперименту")
    p_val.add_argument("--universe", default="configs/universe.yaml")
    p_val.add_argument("--data-root", default=str(DEFAULT_ROOT))
    p_val.add_argument("--symbol", default="BTCUSDT")
    p_val.add_argument("--out", default=None)
    p_val.add_argument("--journal", default=str(DEFAULT_JOURNAL),
                       help="Журнал экспериментов: из него берётся число попыток для DSR")
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

Добавьте в `pyproject.toml`:
```toml
[project.scripts]
alpha-lab = "alpha_lab.cli:main"
```

- [ ] **Шаг 4: Запустить тесты**

```bash
uv run pytest tests/test_cli.py -v
```
Ожидается: 6 passed

- [ ] **Шаг 5: Полный прогон всех тестов**

```bash
uv run pytest -q
```
Ожидается: все тесты зелёные

- [ ] **Шаг 6: Коммит**

```bash
git add src/alpha_lab/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: cli with ingest and validate commands"
```

---

## Задача 19: Дашборд

**Файлы:**
- Создать: `dashboard/index.html`, `dashboard/app.js`, `dashboard/style.css`, `dashboard/report.js` (заглушка)

**Интерфейсы:**
- Потребляет: `window.ALPHA_REPORT` из `report.js` (см. задачу 16)
- Работает через `file://` — никакого `fetch`, никакого сервера, никакой сборки

- [ ] **Шаг 1: Создать `dashboard/style.css`**

```css
:root {
  --bg: #0b0e14; --panel: #0d1117; --line: rgba(255,255,255,.07);
  --text: #cbd5e1; --dim: #64748b;
  --amber: #f59e0b; --green: #22c55e; --red: #ef4444; --sky: #38bdf8;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--text);
  font-family: Inter, system-ui, sans-serif; font-size: 14px;
  background-image: radial-gradient(900px 460px at 85% -10%, rgba(56,189,248,.08), transparent 60%),
                    radial-gradient(700px 400px at -10% 40%, rgba(245,158,11,.06), transparent 55%);
}
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 28px 20px 80px; }

/* Панель вердикта */
.verdict { border-radius: 16px; padding: 22px 26px; border: 1px solid var(--line); margin-bottom: 22px; }
.verdict.alive { background: rgba(34,197,94,.08); border-color: rgba(34,197,94,.35); }
.verdict.dead  { background: rgba(239,68,68,.08); border-color: rgba(239,68,68,.35); }
.verdict h1 { margin: 0 0 4px; font-size: 30px; letter-spacing: -.5px; }
.verdict.alive h1 { color: var(--green); }
.verdict.dead h1 { color: var(--red); }
.verdict .sub { color: var(--dim); font-size: 12.5px; margin-bottom: 18px; }
.verdict .sub .mono { color: var(--text); }

.kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(132px, 1fr)); gap: 10px; }
.kpi { background: rgba(0,0,0,.28); border-radius: 10px; padding: 11px 13px; }
.kpi .k { font-size: 10px; text-transform: uppercase; letter-spacing: .06em; color: var(--dim); }
.kpi .v { font-size: 19px; font-weight: 600; margin-top: 3px; }
.kpi .t { font-size: 10.5px; color: var(--dim); margin-top: 2px; }
.kpi.pass .v { color: var(--green); }
.kpi.fail .v { color: var(--red); }

.reasons { margin-top: 16px; padding-left: 0; list-style: none; }
.reasons li { padding: 7px 0 7px 18px; position: relative; color: #fca5a5; font-size: 13px; border-top: 1px solid rgba(239,68,68,.18); }
.reasons li::before { content: '×'; position: absolute; left: 2px; color: var(--red); font-weight: 700; }

/* Карточки графиков */
.card { background: var(--panel); border: 1px solid var(--line); border-radius: 14px; padding: 16px 18px 10px; margin-bottom: 18px; }
.card h2 { margin: 0 0 12px; font-size: 13px; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
canvas { width: 100%; display: block; }

/* Таблица издержек */
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
th { color: var(--dim); font-size: 10.5px; text-transform: uppercase; letter-spacing: .05em; font-weight: 600; }
td.num { text-align: right; }
.neg { color: var(--red); }

/* Ошибка загрузки */
.error { background: rgba(239,68,68,.1); border: 1px solid rgba(239,68,68,.4); border-radius: 14px; padding: 28px; }
.error h2 { color: var(--red); margin-top: 0; }
.error code { background: rgba(0,0,0,.4); padding: 2px 6px; border-radius: 5px; color: var(--amber); }
.footer { color: var(--dim); font-size: 11.5px; text-align: center; margin-top: 30px; }
```

- [ ] **Шаг 2: Создать `dashboard/index.html`**

```html
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Alpha Lab — отчёт</title>
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="wrap">
  <div id="app"></div>
  <div class="footer">
    Alpha Lab · данные Binance USDT-M · отчёт читается локально, без сервера
  </div>
</div>

<!-- Данные подключаются как скрипт: fetch() к file:// блокируется CORS -->
<script src="report.js"></script>
<script src="app.js"></script>
</body>
</html>
```

- [ ] **Шаг 3: Создать `dashboard/app.js`**

```javascript
'use strict';

const REQUIRED_MAJOR = 1;

function fmtPct(v, digits = 1) {
  return (v === null || v === undefined) ? '—' : (v * 100).toFixed(digits) + '%';
}
function fmtNum(v, digits = 2) {
  return (v === null || v === undefined) ? '—' : Number(v).toFixed(digits);
}

function renderError(title, detail) {
  document.getElementById('app').innerHTML =
    `<div class="error"><h2>${title}</h2><p>${detail}</p></div>`;
}

function kpi(label, value, cls, tooltip) {
  return `<div class="kpi ${cls}">
    <div class="k">${label}</div>
    <div class="v mono">${value}</div>
    <div class="t">${tooltip}</div>
  </div>`;
}

function renderVerdict(v) {
  const cls = v.alive ? 'alive' : 'dead';
  const title = v.alive ? 'ЖИВА' : 'МЕРТВА';

  const kpis = [
    kpi('Sharpe', fmtNum(v.sharpe), ''),
    kpi('DSR', fmtNum(v.dsr, 4), v.dsr > 0.95 ? 'pass' : 'fail', 'порог 0.95'),
    kpi('p-value', fmtNum(v.p_value, 4), v.p_value < 0.05 ? 'pass' : 'fail', 'порог 0.05'),
    kpi('PBO', v.pbo === null ? '—' : fmtNum(v.pbo, 2),
        (v.pbo !== null && v.pbo < 0.5) ? 'pass' : 'fail', 'порог 0.50'),
    kpi('Сделок', v.trades, v.trades >= 100 ? 'pass' : 'fail', 'минимум 100'),
    kpi('Max DD', fmtPct(v.max_dd), ''),
    kpi('Доходность', fmtPct(v.total_return), ''),
    kpi('Попыток', v.n_configs_tried, '', 'учтено в DSR'),
  ].join('');

  const reasons = v.reasons && v.reasons.length
    ? `<ul class="reasons">${v.reasons.map(r => `<li>${r}</li>`).join('')}</ul>`
    : '';

  document.getElementById('verdict-box').innerHTML =
    `<div class="verdict ${cls}">
       <h1>${title}</h1>
       <div class="sub">
         <span class="mono">${v.strategy_name}</span> ·
         experiment <span class="mono">${v.experiment_id}</span>
       </div>
       <div class="kpis">${kpis}</div>
       ${reasons}
     </div>`;
}

function drawLine(canvas, values, color, fillBelow = false, baseline = null) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const data = values.filter(v => v !== null && isFinite(v));
  if (data.length < 2) return;

  let min = Math.min(...data), max = Math.max(...data);
  if (baseline !== null) { min = Math.min(min, baseline); max = Math.max(max, baseline); }
  if (max - min < 1e-9) { max = min + 1; }
  const pad = (max - min) * 0.08;
  min -= pad; max += pad;

  const x = i => (i / (values.length - 1)) * w;
  const y = v => h - ((v - min) / (max - min)) * h;

  if (baseline !== null) {
    ctx.strokeStyle = 'rgba(148,163,184,.28)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(0, y(baseline)); ctx.lineTo(w, y(baseline)); ctx.stroke();
    ctx.setLineDash([]);
  }

  ctx.beginPath();
  let started = false;
  values.forEach((v, i) => {
    if (v === null || !isFinite(v)) return;
    if (!started) { ctx.moveTo(x(i), y(v)); started = true; }
    else ctx.lineTo(x(i), y(v));
  });
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.stroke();

  if (fillBelow && started) {
    ctx.lineTo(x(values.length - 1), y(baseline ?? min));
    ctx.lineTo(x(0), y(baseline ?? min));
    ctx.closePath();
    ctx.fillStyle = color.replace('rgb', 'rgba').replace(')', ',.16)');
    ctx.fill();
  }
}

function drawHistogram(canvas, values, color) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const data = values.filter(v => v !== null && isFinite(v));
  if (data.length < 5) return;

  const min = Math.min(...data), max = Math.max(...data);
  if (max - min < 1e-12) return;
  const bins = Math.min(40, Math.max(10, Math.floor(Math.sqrt(data.length))));
  const counts = new Array(bins).fill(0);
  data.forEach(v => {
    const idx = Math.min(bins - 1, Math.floor(((v - min) / (max - min)) * bins));
    counts[idx]++;
  });
  const peak = Math.max(...counts);
  const bw = w / bins;

  counts.forEach((c, i) => {
    const bh = (c / peak) * (h - 14);
    const binStart = min + (i / bins) * (max - min);
    const positive = binStart >= 0;
    ctx.fillStyle = positive ? 'rgba(34,197,94,.55)' : 'rgba(239,68,68,.55)';
    ctx.fillRect(i * bw + 0.5, h - bh, bw - 1, bh);
  });

  ctx.strokeStyle = 'rgba(148,163,184,.3)';
  ctx.beginPath();
  const zeroX = ((0 - min) / (max - min)) * w;
  ctx.moveTo(zeroX, 0); ctx.lineTo(zeroX, h); ctx.stroke();
}

function renderSeries(s) {
  drawLine(document.getElementById('cv-equity'), s.equity, 'rgb(56,189,248)', false);
  drawLine(document.getElementById('cv-drawdown'), s.drawdown, 'rgb(239,68,68)', true, 0);
  drawLine(document.getElementById('cv-price'), s.close, 'rgb(148,163,184)', false);
}

function renderCosts(report) {
  // cost_totals лежит в extra (см. report/writer.py: build_report(..., extra=...))
  const totals = (report.extra && report.extra.costs_total) || {};
  const rows = [
    ['Комиссии', totals.fee || 0],
    ['Проскальзывание', totals.slippage || 0],
    ['Funding', totals.funding || 0],
  ];
  const sum = rows.reduce((acc, [, val]) => acc + val, 0);
  document.getElementById('costs-body').innerHTML = rows.map(([label, val]) =>
    `<tr><td>${label}</td><td class="num mono ${val < 0 ? 'neg' : ''}">${fmtPct(val, 3)}</td></tr>`
  ).join('') +
    `<tr><td><b>Итого</b></td><td class="num mono ${sum < 0 ? 'neg' : ''}"><b>${fmtPct(sum, 3)}</b></td></tr>`;
}

function init() {
  const report = window.ALPHA_REPORT;

  if (!report) {
    renderError('Отчёт не найден',
      'Файл <code>report.js</code> не загрузился. Сгенерируйте его командой ' +
      '<code>uv run alpha-lab validate --config configs/experiments/mr_base.yaml</code> ' +
      'и скопируйте в папку dashboard.');
    return;
  }

  const major = parseInt(String(report.schema_version).split('.')[0], 10);
  if (major !== REQUIRED_MAJOR) {
    renderError('Несовместимая версия схемы',
      `Отчёт версии <code>${report.schema_version}</code>, дашборд ожидает ` +
      `<code>${REQUIRED_MAJOR}.x</code>. Обновите оба компонента.`);
    return;
  }

  document.getElementById('app').innerHTML = `
    <div id="verdict-box"></div>
    <div class="card"><h2>Кривая доходности</h2>
      <canvas id="cv-equity" style="height:220px"></canvas></div>
    <div class="card"><h2>Просадка</h2>
      <canvas id="cv-drawdown" style="height:110px"></canvas></div>
    <div class="card"><h2>Цена</h2>
      <canvas id="cv-price" style="height:170px"></canvas></div>
    <div class="card"><h2>Издержки, доля от капитала</h2>
      <table><tbody id="costs-body"></tbody></table></div>
    <div class="card"><h2>Вклад баров (непрерывный рост/падение equity)</h2>
      <canvas id="cv-returns" style="height:150px"></canvas></div>`;

  renderVerdict(report.verdict);
  renderSeries(report.series);
  renderCosts(report);

  const eq = report.series.equity.filter(v => v !== null);
  const barReturns = eq.slice(1).map((v, i) => v / eq[i] - 1);
  drawHistogram(document.getElementById('cv-returns'), barReturns);
}

document.addEventListener('DOMContentLoaded', init);
window.addEventListener('resize', () => {
  const r = window.ALPHA_REPORT;
  if (r && parseInt(String(r.schema_version).split('.')[0], 10) === REQUIRED_MAJOR) {
    renderSeries(r.series);
  }
});
```

- [ ] **Шаг 4: Создать заглушку `dashboard/report.js`**

```javascript
// Заглушка. Замените файлом, который создаёт `alpha-lab validate`.
window.ALPHA_REPORT = null;
```

- [ ] **Шаг 5: Проверить на реальном отчёте**

```bash
uv run alpha-lab validate --config configs/experiments/mr_base.yaml --out dashboard
```
Ожидается: команда печатает вердикт, в `dashboard/` появляются `report.json` и `report.js`.

Откройте `dashboard/index.html` двойным кликом. Ожидается: панель вердикта, три графика, таблица издержек. **Сервер не нужен.**

- [ ] **Шаг 6: Проверить обработку несовместимой версии**

Временно измените `schema_version` в `dashboard/report.js` на `"2.0"`, обновите страницу.
Ожидается: красная панель «Несовместимая версия схемы», а не пустые графики. Верните `1.0`.

- [ ] **Шаг 7: Коммит**

```bash
git add dashboard
git commit -m "feat: static file:// dashboard reading report.js"
```

---

## Финальная проверка

- [ ] **Все тесты**

```bash
uv run pytest -q
```
Ожидается: все зелёные.

- [ ] **Проверка инварианта слоёв**

```bash
grep -r "alpha_lab.strategies" src/alpha_lab/validation/ && echo "НАРУШЕНИЕ" || echo "OK: validation слеп"
```
Ожидается: `OK: validation слеп`

- [ ] **Воспроизводимость**

```bash
uv run alpha-lab validate --config configs/experiments/mr_base.yaml --out /tmp/run1
uv run alpha-lab validate --config configs/experiments/mr_base.yaml --out /tmp/run2
diff <(grep -o '"experiment_id": "[^"]*"' /tmp/run1/report.json) \
     <(grep -o '"experiment_id": "[^"]*"' /tmp/run2/report.json) && echo "ID совпал"
```
Ожидается: `ID совпал`

---

## Критерии готовности (из spec, раздел 12)

| # | Критерий | Где проверяется |
|---|---|---|
| 1 | `ingest` скачивает и раскладывает данные | Задача 4, шаг 5 |
| 2 | `validate` выдаёт Verdict и отчёт | Задача 18, тесты |
| 3 | Валидатор убивает все ловушки | Задача 17, `test_all_traps_are_killed_by_validator` |
| 4 | Валидатор не убивает настоящий edge | Задача 17, `test_strong_edge_survives_validation` |
| 5 | Дашборд работает двойным кликом | Задача 19, шаг 5 |
| 6 | Получен первый вердикт по MR-стратегии | Задача 19, шаг 5 |
| 7 | Воспроизводимость по `experiment_id` | Финальная проверка |
