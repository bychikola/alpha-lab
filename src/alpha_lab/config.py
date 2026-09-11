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


def _parse_include_delisted(data: dict[str, Any]) -> bool:
    """include_delisted — строго bool; отсутствие или null — безопасный true.

    bool(None) == False молча выключал бы делистингованные пары и вносил
    ошибку выживаемости, а строка "no"/"yes" вообще не проходила бы проверку
    смысла. Поэтому null трактуется как «не задано» (spec 5 требует включать
    делистингованные пары), а любое не-bool значение — ошибка конфига.
    """
    value = data.get("include_delisted")
    if value is None:
        return True
    if not isinstance(value, bool):
        raise ValueError(
            f"include_delisted должен быть true или false, получено {value!r}")
    return value


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
        include_delisted=_parse_include_delisted(data),
    )


def load_manifest(path: str | Path) -> list[Path]:
    """Читает манифест свипа: один путь к эксперименту на строку.

    Формат выбран текстовым списком, а не перечислением через запятую:
    путь на Windows содержит двоеточие и обратные слэши, а запятая в имени
    файла сделала бы разбор неоднозначным; манифест к тому же переносится
    вместе с конфигами. Правила:

    * пустые строки игнорируются;
    * строки, начинающиеся с '#', — комментарии;
    * относительные пути разрешаются от каталога манифеста, а не от CWD;
    * дубликат пути отвергается: один файл — одна гипотеза, а PBO на
      идентичных колонках вырожден, поэтому «свип» из копий бессмыслен.

    Существование самих конфигов здесь не проверяется — это делает
    load_experiment с внятным сообщением. Каталог вместо файла отвергается
    явной ValueError: read_text на каталоге даёт ОС-зависимое исключение
    (IsADirectoryError на POSIX, PermissionError на Windows), и наружу уходил
    бы трейсбек вместо ошибки конфига.
    """
    path = Path(path)
    if path.is_dir():
        raise ValueError(
            f"Манифест {path} — это каталог, а не файл со списком конфигураций")
    if not path.exists():
        raise FileNotFoundError(f"Манифест не найден: {path}")
    entries: list[Path] = []
    for lineno, raw in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entry = Path(line)
        if not entry.is_absolute():
            entry = path.parent / entry
        if entry in entries:
            raise ValueError(
                f"Манифест {path}: путь {line} (строка {lineno}) указан дважды. "
                f"Дубликат — это одна гипотеза, а не две: PBO на идентичных "
                f"колонках вырожден."
            )
        entries.append(entry)
    if not entries:
        raise ValueError(
            f"Манифест {path} пуст: нужен хотя бы один путь к эксперименту "
            f"(для свипа с PBO — минимум два)."
        )
    return entries


def load_experiment(path: str | Path) -> Experiment:
    return parse_experiment(_read_yaml(Path(path)), Path(path))


def parse_experiment(data: dict[str, Any],
                     source: str | Path = "<словарь>") -> Experiment:
    """Собирает Experiment из уже разобранного словаря.

    Выделено из load_experiment ради сетки гипотез: базовый эксперимент сетки
    лежит вложенным словарём, и его форма обязана проверяться ровно тем же
    кодом, что и одиночный конфиг, — иначе два пути разъедутся.
    """
    for field_name in ("name", "strategy", "timeframe", "start"):
        if not data.get(field_name):
            raise ValueError(
                f"В эксперименте {source} не задано поле '{field_name}'")
    tf = str(data["timeframe"])
    if tf not in VALID_TIMEFRAMES:
        raise ValueError(
            f"Неизвестный таймфрейм '{tf}' в {source}. "
            f"Допустимые: {', '.join(VALID_TIMEFRAMES)}"
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
