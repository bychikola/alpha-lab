"""Хранилище результатов прогонов: одна строка на конфигурацию (Parquet + DuckDB).

    write_runs(path, rows)                 # идемпотентно по config_id, атомарно
    frame = read_runs(path)                # вся таблица
    frame = query_runs(path, symbol=..., order_by="dsr", descending=True)
    counts = group_counts(path, by="timeframe")
    done = completed_ids(path, data_version, screening=False)

**Схема версионирована.** Версия лежит в метаданных Parquet под ключом
``alpha_lab_schema_version`` (константа RESULTS_SCHEMA_VERSION). Несовместимая
мажорная версия отвергается громко, с русским сообщением: прочитать чужой
формат как свою таблицу — значит молча получить мусор в воронке отбора.

**Идемпотентность по config_id.** Повторная запись строки с тем же id
перезаписывает её (keep="last"), а не добавляет вторую: на этом стоит
возобновляемость свипа, и дубликат конфигурации в воронке был бы двойным
счётом одной гипотезы.

**Атомарность.** Таблица пишется во временный файл и заменяется через
``os.replace``: обрыв между чтением старой таблицы и заменой не оставляет
половину строк, а читатель всегда видит либо старую версию целиком, либо новую.

**Контракт строки-ошибки.** ``error`` — пустая строка у вердикта и непустое
сообщение у упавшего прогона. Строка-ошибка принудительно несёт
``alive=False``, ``trades=0`` и пустые ``reasons``, а строки с ``screening=True``
не могут нести ``alive=True``: крах и черновой вердикт не имеют права
выглядеть как подтверждённый результат. Фильтровать их обязательно явно —
``error_rows(frame)`` / ``verdict_rows(frame)`` / ``is_error(frame)``; по
умолчанию ``query_runs`` отдаёт только вердикты, а ``group_counts`` тоже.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Версия схемы хранилища. Мажор меняется при несовместимом изменении набора
# или смысла колонок; минор — при совместимом добавлении.
RESULTS_SCHEMA_VERSION = "1.0"
RESULTS_MAJOR_VERSION = int(RESULTS_SCHEMA_VERSION.split(".")[0])

# Ключ метаданных Parquet, по которому читается версия. Метаданные едут внутри
# того же файла, поэтому версия не может «отстать» от данных, как отдельный
# sidecar-файл при частичной записи.
VERSION_KEY = b"alpha_lab_schema_version"

FILE_NAME = "runs.parquet"
TMP_SUFFIX = ".tmp"

# Порядок колонок — часть контракта: дашборд и тесты читают таблицу как есть.
COLUMNS = (
    "config_id", "experiment_id", "index", "symbol", "timeframe", "strategy",
    "params_json", "start", "end", "error", "data_version", "duration_s",
    "sharpe", "dsr", "p_value", "pbo", "max_dd", "total_return", "trades",
    "n_trials", "n_permutations", "screening", "alive", "reasons", "warnings",
    "cost_total", "costs_json",
)

_STRING_COLUMNS = (
    "config_id", "experiment_id", "symbol", "timeframe", "strategy",
    "params_json", "start", "end", "error", "data_version", "reasons",
    "warnings", "costs_json",
)
_INT_COLUMNS = ("index", "trades", "n_trials", "n_permutations")
_FLOAT_COLUMNS = ("duration_s", "sharpe", "dsr", "p_value", "pbo", "max_dd",
                  "total_return", "cost_total")
_BOOL_COLUMNS = ("screening", "alive")


def is_compatible(version) -> bool:
    """Совместима ли мажорная версия схемы хранилища с текущим кодом."""
    try:
        return int(str(version).split(".")[0]) == RESULTS_MAJOR_VERSION
    except (ValueError, AttributeError, TypeError):
        return False


def runs_file(path: str | Path) -> Path:
    """Файл таблицы: путь-каталог → runs.parquet, путь к файлу → он сам."""
    p = Path(path)
    return p if p.suffix == ".parquet" else p / FILE_NAME


def _empty_frame() -> pd.DataFrame:
    return _normalize(pd.DataFrame(), where="пустое хранилище")


def _default_value(column: str):
    if column in _INT_COLUMNS:
        return 0
    if column in _FLOAT_COLUMNS:
        return float("nan")
    if column in _BOOL_COLUMNS:
        return False
    return ""


def _normalize(frame: pd.DataFrame, *, where: str) -> pd.DataFrame:
    """Приводит таблицу к контракту схемы: колонки, типы, строка-ошибка.

    Неизвестная колонка — ошибка, а не молчаливое выбрасывание: подмена имени
    (`sharpe_typo`) иначе записала бы строку без метрики, и воронка увидела бы
    честную с виду конфигурацию с NaN. Отсутствующие колонки заполняются
    значениями по умолчанию — это путь для тестовых строк, а не для данных.
    """
    if frame is None:
        frame = pd.DataFrame()
    unknown = [c for c in frame.columns if c not in COLUMNS]
    if unknown:
        raise ValueError(
            f"Хранилище результатов ({where}): неизвестные колонки "
            f"{sorted(unknown)}. Схема {RESULTS_SCHEMA_VERSION} допускает "
            f"только: {', '.join(COLUMNS)}. Опечатка в имени колонки молча "
            f"потеряла бы метрику — запись отвергнута."
        )
    if "config_id" not in frame.columns and len(frame):
        raise ValueError(
            f"Хранилище результатов ({where}): нет колонки 'config_id' — "
            f"идемпотентность и возобновляемость по id конфигурации "
            f"невозможны"
        )
    frame = frame.reset_index(drop=True)
    n = len(frame)
    data: dict[str, pd.Series] = {}
    for column in COLUMNS:
        if column in frame.columns:
            data[column] = frame[column]
        else:
            data[column] = pd.Series([_default_value(column)] * n)
    out = pd.DataFrame(data)

    if n:
        ids = out["config_id"].fillna("").astype(str).str.strip()
        if (ids == "").any():
            raise ValueError(
                f"Хранилище результатов ({where}): пустой config_id — строка "
                f"без id не может быть ни перезаписана, ни возобновлена"
            )
        error = out["error"].fillna("").astype(str).str.strip()
        failed = error != ""
        # Контракт строки-ошибки принудителен: упавший прогон не является
        # гипотезой, которую отсеяли, и не имеет права выглядеть вердиктом.
        out.loc[failed, "alive"] = False
        out.loc[failed, "trades"] = 0
        out.loc[failed, "reasons"] = ""
        out["error"] = error
        # Черновой вердикт не может утверждать alive (P5): страховка на уровне
        # хранилища, даже если вызывающий ошибётся.
        screening = out["screening"].fillna(False).astype(bool)
        out.loc[screening, "alive"] = False

    for column in _STRING_COLUMNS:
        out[column] = out[column].fillna("").astype(str)
    for column in _INT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0).astype("int64")
    for column in _FLOAT_COLUMNS:
        out[column] = pd.to_numeric(out[column], errors="coerce").astype("float64")
    for column in _BOOL_COLUMNS:
        out[column] = out[column].fillna(False).astype(bool)
    return out[list(COLUMNS)]


def _check_version(file: Path) -> str:
    """Читает версию схемы из метаданных; несовместимую отвергает громко."""
    try:
        metadata = pq.read_metadata(file).metadata or {}
    except Exception as exc:  # noqa: BLE001 — любое нечтение файла = отказ
        raise ValueError(
            f"Хранилище результатов {file} не читается как Parquet: {exc}. "
            f"Восстановите файл или пересоздайте хранилище."
        ) from exc
    raw = metadata.get(VERSION_KEY)
    if raw is None:
        raise ValueError(
            f"Хранилище результатов {file}: версия схемы не указана (нет "
            f"метаданных '{VERSION_KEY.decode()}'). Файл создан не этим "
            f"хранилищем или повреждён; ожидается схема "
            f"{RESULTS_SCHEMA_VERSION}. Читать его как совместимый нельзя — "
            f"колонки могли бы разойтись молча."
        )
    version = raw.decode("utf-8", errors="replace")
    if not is_compatible(version):
        raise ValueError(
            f"Хранилище результатов {file}: несовместимая версия схемы "
            f"«{version}», ожидается мажор {RESULTS_MAJOR_VERSION}.x "
            f"({RESULTS_SCHEMA_VERSION}). Прочитанные как совместимые данные "
            f"дали бы правдоподобный, но неверный отбор."
        )
    return version


def read_runs(path: str | Path) -> pd.DataFrame:
    """Вся таблица прогонов. Отсутствующее хранилище — пустая таблица схемы."""
    file = runs_file(path)
    if not file.exists():
        return _empty_frame()
    _check_version(file)
    con = duckdb.connect()
    try:
        frame = con.execute(
            "SELECT * FROM read_parquet(?)", [str(file)]).fetchdf()
    finally:
        con.close()
    return _normalize(frame, where=str(file))


def write_runs(path: str | Path, rows: list[dict]) -> None:
    """Идемпотентно дописывает строки: тот же config_id заменяет прежнюю.

    Запись атомарна: временный файл + os.replace. Сбой на любом шаге до
    замены оставляет прежнее хранилище нетронутым.
    """
    if not rows:
        return
    file = runs_file(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    fresh = _normalize(pd.DataFrame(list(rows)), where=str(file))
    if file.exists():
        _check_version(file)
        old = read_runs(path)
        fresh = pd.concat([old, fresh], ignore_index=True)
        fresh = fresh.drop_duplicates(subset=["config_id"], keep="last")
    fresh = fresh.reset_index(drop=True)
    table = pa.Table.from_pandas(fresh, preserve_index=False)
    table = table.replace_schema_metadata({
        **(table.schema.metadata or {}),
        VERSION_KEY: RESULTS_SCHEMA_VERSION.encode("utf-8"),
    })
    tmp = file.with_name(file.name + TMP_SUFFIX)
    pq.write_table(table, tmp)
    os.replace(tmp, file)


def is_error(frame: pd.DataFrame) -> pd.Series:
    """True для строк-ошибок (непустой error). Отсутствие колонки — не ошибка."""
    if "error" not in frame.columns:
        return pd.Series(False, index=frame.index)
    error = frame["error"]
    return error.notna() & (error.astype(str).str.strip() != "")


def error_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Строки упавших прогонов: это не гипотезы, их нельзя мешать с вердиктами."""
    return frame[is_error(frame)]


def verdict_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Строки-вердикты (включая черновые): ошибки исключены."""
    return frame[~is_error(frame)]


def completed_ids(path: str | Path, data_version: str,
                  *, screening: bool = False) -> set[str]:
    """id подтверждённо выполненных конфигураций для данной версии данных.

    Ошибка не считается выполнением (отказ бывает временным). При
    ``screening=False`` (полный прогон) засчитываются только полные вердикты:
    черновая строка не имеет права остановить полную проверку финалиста. При
    ``screening=True`` засчитываются оба класса — полный вердикт строго
    сильнее чернового, пересчитывать его черновым незачем.
    """
    file = runs_file(path)
    if not file.exists():
        return set()
    _check_version(file)
    sql = (
        "SELECT config_id FROM read_parquet(?) "
        "WHERE data_version = ? "
        "AND (error IS NULL OR trim(error) = '')"
    )
    if not screening:
        sql += " AND coalesce(screening, FALSE) = FALSE"
    con = duckdb.connect()
    try:
        result = con.execute(sql, [str(file), str(data_version)]).fetchall()
    finally:
        con.close()
    return {str(row[0]) for row in result}


def query_runs(path: str | Path, *, symbol: str | None = None,
               timeframe: str | None = None, alive: bool | None = None,
               status: str = "verdict", order_by: str | None = None,
               descending: bool = False,
               limit: int | None = None) -> pd.DataFrame:
    """Тонкая выборка по хранилищу; работу делает DuckDB.

    ``status``: "verdict" (по умолчанию — только не упавшие), "error" (только
    упавшие), "any" (все). Строки-ошибки по умолчанию исключены намеренно:
    смешать крах с отсеянной гипотезой — значит посчитать отказ стратегии
    там, где стратегия не запускалась.
    """
    if status not in ("verdict", "error", "any"):
        raise ValueError(
            f"status должен быть 'verdict', 'error' или 'any', получено "
            f"{status!r}: неясно, считать ли строки-ошибки гипотезами")
    if order_by is not None and order_by not in COLUMNS:
        raise ValueError(
            f"order_by '{order_by}' — не колонка хранилища. Допустимые: "
            f"{', '.join(COLUMNS)}")
    file = runs_file(path)
    if not file.exists():
        return _empty_frame()
    _check_version(file)
    where: list[str] = []
    params: list = []
    if status == "verdict":
        where.append("(error IS NULL OR trim(error) = '')")
    elif status == "error":
        where.append("NOT (error IS NULL OR trim(error) = '')")
    if symbol is not None:
        where.append("symbol = ?")
        params.append(str(symbol))
    if timeframe is not None:
        where.append("timeframe = ?")
        params.append(str(timeframe))
    if alive is not None:
        where.append("alive = ?")
        params.append(bool(alive))
    sql = "SELECT * FROM read_parquet(?)"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if order_by is not None:
        sql += f' ORDER BY "{order_by}" {"DESC" if descending else "ASC"} NULLS LAST'
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    con = duckdb.connect()
    try:
        frame = con.execute(sql, [str(file), *params]).fetchdf()
    finally:
        con.close()
    return _normalize(frame, where=str(file))


def group_counts(path: str | Path, by: str, *,
                 status: str = "verdict") -> pd.DataFrame:
    """Число строк по значению колонки ``by`` (по умолчанию без ошибок)."""
    if by not in COLUMNS:
        raise ValueError(
            f"group_counts: '{by}' — не колонка хранилища. Допустимые: "
            f"{', '.join(COLUMNS)}")
    if status not in ("verdict", "error", "any"):
        raise ValueError(
            f"group_counts: status должен быть 'verdict', 'error' или 'any', "
            f"получено {status!r}")
    file = runs_file(path)
    if not file.exists():
        return pd.DataFrame({by: pd.Series(dtype="object"), "n": pd.Series(dtype="int64")})
    _check_version(file)
    where = ""
    if status == "verdict":
        where = " WHERE (error IS NULL OR trim(error) = '')"
    elif status == "error":
        where = " WHERE NOT (error IS NULL OR trim(error) = '')"
    sql = (f'SELECT "{by}" AS "{by}", count(*) AS n FROM read_parquet(?)'
           f'{where} GROUP BY 1 ORDER BY n DESC, 1')
    con = duckdb.connect()
    try:
        frame = con.execute(sql, [str(file)]).fetchdf()
    finally:
        con.close()
    return frame
