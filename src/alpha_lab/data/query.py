"""Чтение данных через DuckDB.

DuckDB читает только нужные колонки и партиции, не поднимая весь датасет в память —
критично при 16 ГБ RAM и десятках гигабайт истории.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from alpha_lab.data.quality import FREQ_DELTA
from alpha_lab.data.store import DEFAULT_ROOT, end_bound

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
    """Читает бары символа. Если задан resample — агрегирует из freq в resample.

    Границы периода: start включается с начала дня/момента; end-дата включает
    весь конечный день, явный момент времени — точная граница (см. store.end_bound).
    """
    # Только файлы своего символа и таймфрейма: глоб *.parquet подмешал бы в ряд
    # чужой parquet, случайно оказавшийся в каталоге.
    pattern = str(Path(root) / "bars" / symbol / freq / f"{symbol}-{freq}-*.parquet")
    where, params = [], []
    if start:
        where.append("ts >= ?")
        params.append(pd.Timestamp(start, tz="UTC").to_pydatetime())
    if end:
        bound, strict = end_bound(end)
        where.append("ts < ?" if strict else "ts <= ?")
        params.append(bound.to_pydatetime())
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


def _bar_step_ns(bar_ns: np.ndarray, timeframe: str | None) -> int | None:
    """Шаг бара в наносекундах: явный таймфрейм или медиана интервалов ряда.

    None — шаг неизвестен (меньше двух баров или все метки совпадают); тогда
    событие привязывается только при точном совпадении с меткой бара.
    """
    if timeframe is not None:
        if timeframe not in FREQ_DELTA:
            raise ValueError(
                f"Неизвестный таймфрейм: {timeframe!r}. "
                f"Допустимые: {', '.join(FREQ_DELTA)}"
            )
        return int(FREQ_DELTA[timeframe].value)
    if len(bar_ns) < 2:
        return None
    positive = np.diff(bar_ns)
    positive = positive[positive > 0]
    if positive.size == 0:
        return None
    # Медиана, а не минимум: дыры в ряде (большие интервалы) не должны
    # уменьшать шаг и растаскивать события по несуществующим барам.
    return int(np.median(positive))


def _funding_positions(bars: pd.DataFrame, funding: pd.DataFrame,
                       timeframe: str | None) -> tuple[np.ndarray, np.ndarray]:
    """Индексы баров-контейнеров для событий funding и признак попадания.

    Контейнер события — последний бар, открывающийся не позже события; бар
    занимает полуоткрытый интервал [t, t+шаг). Событие до первого бара или
    позже конца последнего бара (t+шаг) не принадлежит ни одному бару:
    valid=False, а не «приклеилось» к крайнему бару.
    """
    # as_unit("ns") обязателен: pandas 3 хранит datetime64 с разным разрешением
    # (у дневного ряда — микросекунды), и asi8 без приведения дал бы числа не в
    # наносекундах, а сравнение с шагом из FREQ_DELTA — молча неверную границу.
    bar_ns = pd.DatetimeIndex(
        pd.to_datetime(bars["ts"], utc=True)).as_unit("ns").asi8
    fund_ns = pd.DatetimeIndex(
        pd.to_datetime(funding["ts"], utc=True)).as_unit("ns").asi8
    if len(bar_ns) == 0:
        return np.zeros(0, dtype="int64"), np.zeros(len(fund_ns), dtype=bool)

    pos = np.searchsorted(bar_ns, fund_ns, side="right") - 1
    clipped = np.clip(pos, 0, len(bar_ns) - 1)
    valid = pos >= 0
    step = _bar_step_ns(bar_ns, timeframe)
    left = bar_ns[clipped]
    if step is None:
        valid &= fund_ns == left
    else:
        valid &= fund_ns < left + step
    return clipped, valid


def align_funding_to_bars(bars: pd.DataFrame, funding: pd.DataFrame,
                          timeframe: str | None = None) -> pd.Series:
    """Ставка funding, привязанная к барам.

    Каждое событие попадает в бар, который его СОДЕРЖИТ (полуоткрытый
    интервал [t, t+шаг)); несколько событий одного бара суммируются. Ставка
    ровно на границе принадлежит бару, который этой границей открывается:
    так же вёл себя точный матч на 1h, и так же считает движок — funding
    бара t начисляется на held[t], позицию, удерживаемую в этом баре.
    Поэтому часовое выравнивание не меняется, а на 1d перестают теряться
    выплаты 08:00 и 16:00 (две трети, занижавшие издержки).

    timeframe — шаг бара; None означает вывести шаг из медианы интервалов
    ряда (для регулярного ряда это тот же шаг). События вне диапазона баров
    не привязываются ни к какому бару.
    """
    if funding.empty:
        return pd.Series(0.0, index=bars.index, name="funding_rate")

    pos, valid = _funding_positions(bars, funding, timeframe)
    rates = funding["rate"].astype(float).to_numpy()
    sums = np.zeros(len(bars), dtype="float64")
    if valid.any():
        np.add.at(sums, pos[valid], rates[valid])
    return pd.Series(sums, index=bars.index, name="funding_rate")


def matched_funding_events(bars: pd.DataFrame, funding: pd.DataFrame,
                           timeframe: str | None = None) -> int:
    """Сколько событий funding попало в существующие бары (включая нулевые).

    Нулевая ставка — такое же учтённое событие, как ненулевая: суммой их не
    различить, а предупреждение «funding не привязан» должно срабатывать
    только когда события действительно не легли ни на один бар.
    """
    if funding.empty:
        return 0
    _, valid = _funding_positions(bars, funding, timeframe)
    return int(valid.sum())


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
