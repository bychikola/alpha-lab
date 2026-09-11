"""Воронка отбора P4: «нашли ли мы что-нибудь» одним экраном.

    uv run alpha-lab report --store results/ --out dashboard

Воронка читает хранилище P3 и считает по порядку гейтов spec 6.5:

    N конфигураций → сделок ≥ min_trades → DSR > 0.95 → p-value < 0.05
                   → PBO < 0.5 → alive

Три правила честности, встроенные в устройство, а не в оформление:

* **Ошибки — не гипотезы.** Строки с непустым ``error`` исключены из всех
  знаменателей и показаны отдельно: крах прогона не является отсеянной
  гипотезой, и «доля выживших» без этого была бы занижена отказом железа.
* **Лучший из N мёртвых — всё ещё мёртвый.** Если выживших ноль, топ не
  формируется вовсе: ни строки «Топ выживших», ни метрик лучшего из мёртвых.
  Это тестируемое свойство (tests/test_funnel.py).
* **N виден рядом с победителем.** Секция выживших называет, из скольких
  вердиктов они выбраны и сколько строк-ошибок исключено; DSR уже дефлирован
  на число попыток, но читатель обязан видеть N, а не выводить его из контекста.
* **Грейд не смешивается.** Полные вердикты и черновые (screening, P5) считаются
  раздельно. Черновой кандидат не «жива»: он прошёл грубый гейт и требует
  полного прогона. Хранилище, смешавшее грейды, получает grade="mixed" и
  предупреждение в каждой воронке.
* **Строка судится своими порогами.** Сетка вправе переопределить
  min_trades/min_dsr/max_p_value/max_pbo, и это часть гипотезы (входит в
  config_id). Пороги записаны в каждой строке хранилища (``thresholds_json``,
  схема P3 2.0), поэтому воронка не подставляет сегодняшние дефолты в чужой
  прогон. Строка без записанных порогов честно называется в предупреждении.
* **Список выживших — то же множество, что счётчик alive.** Оба строятся по
  одной маске (пересечение гейтов ∧ хранимый флаг), поэтому «alive 0» рядом с
  непустым «Топ выживших» невозможен по построению, а не по договорённости.

Экспорт — по тому же контракту, что отчёт одиночного прогона: ``funnel.json``
для программ и ``funnel.js`` (``window.ALPHA_FUNNEL``) для дашборда, который
открывается через file:// и не может использовать fetch().
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.results import is_error, verdict_rows
from alpha_lab.validation.validator import (
    DEFAULT_THRESHOLDS as VALIDATOR_DEFAULT_THRESHOLDS,
    GATE_THRESHOLD_KEYS,
)

FUNNEL_SCHEMA_VERSION = "1.0"
FUNNEL_MAJOR_VERSION = int(FUNNEL_SCHEMA_VERSION.split(".")[0])

# Пороги по умолчанию — из валидатора (spec 6.5): единый источник, чтобы
# воронка и вердикт не разошлись в трактовке гейтов.
DEFAULT_THRESHOLDS = {key: VALIDATOR_DEFAULT_THRESHOLDS[key]
                      for key in GATE_THRESHOLD_KEYS}

SORTABLE = ("dsr", "sharpe", "p_value", "pbo", "trades", "total_return",
            "max_dd", "cost_total")

# Сколько сообщений об ошибках класть в payload: полный список ошибок бывает
# длинным, но счётчик и представители обязаны быть видны.
ERROR_MESSAGES_CAP = 20


def is_compatible(version) -> bool:
    """Совместима ли мажорная версия схемы воронки (контракт дашборда)."""
    try:
        return int(str(version).split(".")[0]) == FUNNEL_MAJOR_VERSION
    except (ValueError, AttributeError, TypeError):
        return False


def _finite_mask(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype="float64")
    return pd.Series(np.isfinite(values), index=series.index)


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("float64")


def _frac(part: int, whole: int) -> float | None:
    return None if whole == 0 else round(part / whole, 6)


def _safe(value):
    """JSON-совместимое число: NaN/Inf → None (json.dumps не терпит NaN)."""
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return None if not np.isfinite(number) else round(number, 6)
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def _screening_mask(frame: pd.DataFrame) -> pd.Series:
    if "screening" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["screening"].fillna(False).astype(bool)


def _split_grades(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    screening = _screening_mask(frame)
    return frame[~screening], frame[screening]


def _parse_thresholds(raw) -> dict | None:
    """Пороги гейтов из thresholds_json; None — записи нет/она битая/неполная.

    Неполная запись — тоже None: достроить недостающий порог значением по
    умолчанию значило бы судить строку порогом, которого у неё не было.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    out: dict[str, float] = {}
    for key in GATE_THRESHOLD_KEYS:
        try:
            number = float(raw[key])
        except (KeyError, TypeError, ValueError):
            return None
        if not np.isfinite(number):
            return None
        out[key] = number
    return out


def _row_thresholds(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Пороги по строкам когорты и число строк без записанных порогов.

    Схема P3 2.0 пишет пороги в каждую строку; строка без них (собрана в обход
    свипа или повреждена) судится порогами по умолчанию, но счётчик уходит в
    предупреждение воронки: молча подставить дефолты — значит повторить ровно
    тот дефект, ради которого пороги и записываются.
    """
    raw = (frame["thresholds_json"] if "thresholds_json" in frame.columns
           else pd.Series("", index=frame.index, dtype="object"))
    data = {key: np.empty(len(frame), dtype="float64")
            for key in GATE_THRESHOLD_KEYS}
    unknown = 0
    for i, value in enumerate(raw):
        parsed = _parse_thresholds(value)
        if parsed is None:
            unknown += 1
            parsed = DEFAULT_THRESHOLDS
        for key in GATE_THRESHOLD_KEYS:
            data[key][i] = parsed[key]
    return pd.DataFrame(data, index=frame.index), unknown


def _shown_threshold(values: pd.Series) -> float | None:
    """Единый порог когорты или None, если у строк разные пороги."""
    unique = pd.unique(pd.to_numeric(values, errors="coerce").dropna())
    return float(unique[0]) if len(unique) == 1 else None


def _threshold_out(key: str, value: float):
    """Порог для payload: min_trades — целое (как в конфиге), остальные float."""
    return int(value) if key == "min_trades" else float(value)


def _threshold_summary(table: pd.DataFrame) -> dict:
    """Сводка порогов хранилища: скаляр на гейт, None и mixed при разнобое."""
    mixed = False
    out: dict = {}
    for key in GATE_THRESHOLD_KEYS:
        shown = _shown_threshold(table[key]) if len(table) else None
        if shown is None and len(table):
            mixed = True
        if shown is None:
            shown = float(DEFAULT_THRESHOLDS[key])
        out[key] = _threshold_out(key, shown)
    if mixed:
        out["mixed"] = True
        out["note"] = (
            "В хранилище строки с разными порогами гейтов (или порог не "
            "записан): каждый гейт считается по порогу своей строки, поэтому "
            "null — единого значения нет, а доли таких строк между собой "
            "несравнимы."
        )
    return out


def _funnel_block(frame: pd.DataFrame, thresholds: pd.DataFrame
                  ) -> tuple[dict, pd.Series]:
    """Последовательные гейты по одной когорте вердиктов (ошибки уже исключены).

    thresholds — таблица порогов по строкам когорты (``_row_thresholds``):
    каждый вердикт судится порогами, записанными в его строке. Каждый следующий
    счётчик — подмножество предыдущего: доля шага считается и от исходной
    когорты, и от предыдущего шага. ``alive`` — пересечение хранимого флага с
    пройденными гейтами; живые без оценённого PBO (одиночный прогон без
    матрицы) считаются отдельно, чтобы не прятать непроверенный гейт.

    Возвращает (блок, маску alive). Маска — ровно то пересечение, по которому
    посчитан ``alive``; список выживших строится по ней же, поэтому счётчик и
    список не могут разойтись.
    """
    n = len(frame)
    if n == 0:
        empty = pd.Series(False, index=frame.index, dtype=bool)
        return ({"n": 0, "steps": [],
                 "alive": {"n": 0, "fraction_of_total": None,
                           "fraction_of_previous": None},
                 "alive_without_pbo": 0}, empty)

    trades_ok = _num(frame["trades"]).fillna(0) >= thresholds["min_trades"]
    dsr_values = _num(frame["dsr"])
    dsr_ok = (trades_ok & _finite_mask(frame["dsr"])
              & (dsr_values > thresholds["min_dsr"]))
    p_values = _num(frame["p_value"])
    p_ok = (dsr_ok & _finite_mask(frame["p_value"])
            & (p_values < thresholds["max_p_value"]))
    pbo_values = _num(frame["pbo"])
    pbo_ok = (p_ok & _finite_mask(frame["pbo"])
              & (pbo_values < thresholds["max_pbo"]))
    alive_col = frame["alive"].fillna(False).astype(bool)
    alive_ok = pbo_ok & alive_col

    specs = (
        ("trades", "min_trades", trades_ok,
         lambda v: f"сделок ≥ {int(v)}", "сделок ≥ порога строки"),
        ("dsr", "min_dsr", dsr_ok,
         lambda v: f"DSR > {v:g}", "DSR > порога строки"),
        ("p_value", "max_p_value", p_ok,
         lambda v: f"p-value < {v:g}", "p-value < порога строки"),
        ("pbo", "max_pbo", pbo_ok,
         lambda v: f"PBO < {v:g}", "PBO < порога строки"),
    )
    steps = []
    previous = n
    for key, threshold_key, mask, fmt, mixed_label in specs:
        count = int(mask.sum())
        shown = _shown_threshold(thresholds[threshold_key])
        steps.append({
            "key": key,
            "label": fmt(shown) if shown is not None else mixed_label,
            "n": count,
            "fraction_of_total": _frac(count, n),
            "fraction_of_previous": _frac(count, previous),
            "threshold": (None if shown is None
                          else _threshold_out(threshold_key, shown)),
        })
        previous = count
    alive_n = int(alive_ok.sum())
    block = {
        "n": n,
        "steps": steps,
        "alive": {
            "n": alive_n,
            "fraction_of_total": _frac(alive_n, n),
            "fraction_of_previous": _frac(alive_n, int(pbo_ok.sum())),
        },
        "alive_without_pbo": int((alive_col & ~pbo_ok).sum()),
    }
    return block, alive_ok


def _screening_block(frame: pd.DataFrame,
                     thresholds: pd.DataFrame) -> dict | None:
    if len(frame) == 0:
        return None
    block, _ = _funnel_block(frame, thresholds)
    candidates = block["steps"][-1]["n"] if block["steps"] else 0
    permutations = 0
    if "n_permutations" in frame.columns:
        values = pd.to_numeric(frame["n_permutations"], errors="coerce").fillna(0)
        if len(values) and int(values.max()) > 0:
            permutations = int(values.max())
    min_p = (1.0 / (permutations + 1)) if permutations else None
    block.update({
        "candidates": candidates,
        "alive": {"n": 0, "fraction_of_total": 0.0, "fraction_of_previous": 0.0},
        "n_permutations": permutations,
        "min_achievable_p": None if min_p is None else round(min_p, 6),
        "note": (
            "Черновой вердикт (screening) не выносит alive: кандидаты прошли "
            "грубый гейт и требуют полного прогона (без --screening, при "
            "необходимости --force). Смешивать их с полными вердиктами нельзя."
        ),
    })
    return block


def _breakdown(frame: pd.DataFrame, column: str) -> list[dict]:
    """Разрез по символу/таймфрейму: где выжившие, а где их нет.

    n — только вердикты; строки-ошибки посчитаны отдельной колонкой и в
    знаменатель не входят. Пороги берутся из строк группы: разрез не имеет
    права судить строки чужими порогами.
    """
    rows: list[dict] = []
    for value, group in frame.groupby(column, dropna=False, sort=False):
        clean = verdict_rows(group)
        full, screen = _split_grades(clean)
        full_thresholds, _ = _row_thresholds(full)
        screen_thresholds, _ = _row_thresholds(screen)
        full_block, _ = _funnel_block(full, full_thresholds)
        screen_block = _screening_block(screen, screen_thresholds)
        rows.append({
            column: str(value),
            "n": int(len(clean)),
            "alive": full_block["alive"]["n"],
            "screening": int(len(screen)),
            "screening_candidates": (screen_block["candidates"]
                                     if screen_block else 0),
            "errors": int(is_error(group).sum()),
        })
    return sorted(rows, key=lambda row: (-row["n"], row[column]))


def _top_entry(row: pd.Series) -> dict:
    def _params() -> dict:
        try:
            value = json.loads(row.get("params_json") or "{}")
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    def _costs() -> dict:
        try:
            value = json.loads(row.get("costs_json") or "{}")
            return value if isinstance(value, dict) else {}
        except (TypeError, ValueError):
            return {}

    def _lines(value) -> list[str]:
        text = str(value or "").strip()
        return [part.strip() for part in text.split("|") if part.strip()]

    return {
        "config_id": str(row["config_id"]),
        "experiment_id": str(row.get("experiment_id", "")),
        "symbol": str(row.get("symbol", "")),
        "timeframe": str(row.get("timeframe", "")),
        "strategy": str(row.get("strategy", "")),
        "params": _params(),
        "sharpe": _safe(row.get("sharpe")),
        "dsr": _safe(row.get("dsr")),
        "p_value": _safe(row.get("p_value")),
        "pbo": _safe(row.get("pbo")),
        "trades": int(row.get("trades") or 0),
        "max_dd": _safe(row.get("max_dd")),
        "total_return": _safe(row.get("total_return")),
        "cost_total": _safe(row.get("cost_total")),
        "costs": _costs(),
        "n_trials": int(row.get("n_trials") or 0),
        "n_permutations": int(row.get("n_permutations") or 0),
        "reasons": _lines(row.get("reasons")),
        "warnings": _lines(row.get("warnings")),
    }


def _survivors(frame: pd.DataFrame, alive_mask: pd.Series, *,
               n_configs: int, n_verdicts: int, n_screening: int,
               n_errors: int, top: int, sort_by: str) -> dict:
    """Топ выживших — ровно по маске alive из ``_funnel_block``.

    ``alive_mask`` — то же пересечение гейтов и хранимого флага, по которому
    посчитан ``funnel.alive.n``; передача готовой маски (а не повторный отбор
    по флагу) делает расхождение счётчика и списка невозможным.

    ``n_verdicts`` — только полные вердикты: черновые (screening) не могут быть
    выжившими, и включать их в N выбора значило бы завышать знаменатель отбора.
    Их число видно отдельно (``n_screening``).
    """
    if sort_by not in SORTABLE:
        raise ValueError(
            f"sort_by '{sort_by}' не сортируемый показатель. Допустимые: "
            f"{', '.join(SORTABLE)}")
    top = max(0, int(top))
    mask = alive_mask.reindex(frame.index, fill_value=False).astype(bool)
    alive = frame.loc[mask].copy()
    n_alive = int(mask.sum())
    entries: list[dict] = []
    if len(alive):
        alive["_sort"] = _num(alive[sort_by])
        alive = alive.sort_values("_sort", ascending=False,
                                  na_position="last", kind="mergesort")
        entries = [_top_entry(row) for _, row in alive.head(top).iterrows()]
    selected_from = {
        "n_configs": int(n_configs),
        "n_verdicts": int(n_verdicts),
        "n_screening": int(n_screening),
        "n_errors": int(n_errors),
        "sort_by": sort_by,
    }
    screening_note = (
        f"черновых screening: {n_screening} — они не могут быть выжившими; "
    )
    if not entries:
        if n_alive:
            # Живые есть, но лимит показа нулевой: честный ноль был бы ложью.
            note = (
                f"Выжившие есть: {n_alive}, но top={top} — список не "
                f"формируется, метрики не предъявляются. Увеличьте --top."
            )
        else:
            note = (
                f"Выживших нет: 0 из {n_verdicts} полных вердиктов "
                f"({screening_note}ошибок исключено: {n_errors}). Лучший из "
                f"мёртвых — всё ещё мёртвый: топ не формируется, метрики "
                f"лучшего из отсеянных не предъявляются как результат."
            )
    else:
        note = (
            f"Выжившие отобраны из {n_verdicts} полных вердиктов "
            f"({screening_note}всего строк {n_configs}, ошибок исключено: "
            f"{n_errors}); DSR каждого уже дефлирован на число попыток "
            f"n_trials, но выбор из N — часть результата, а не деталь "
            f"оформления."
        )
    return {"n": len(entries), "top": entries, "selected_from": selected_from,
            "note": note}


def build_funnel(frame: pd.DataFrame, *, top: int = 10, sort_by: str = "dsr",
                 store_path: str | Path | None = None) -> dict:
    """Строит payload воронки по таблице хранилища (включая строки-ошибки).

    Порогов-аргументов нет намеренно: каждая строка судится порогами,
    записанными в ней (``thresholds_json``), а не тем, что оператор помнит или
    что сегодня лежит в DEFAULT_THRESHOLDS. Хранилище без порогов честно
    предупреждает о подстановке дефолтов.
    """
    if frame is None:
        frame = pd.DataFrame()
    errors = frame[is_error(frame)] if len(frame) else frame
    clean = verdict_rows(frame) if len(frame) else frame
    full, screening = _split_grades(clean)
    thresholds, n_unknown_thresholds = _row_thresholds(clean)

    data_versions = sorted({
        str(v) for v in frame.get("data_version", pd.Series(dtype="object"))
        if str(v).strip()
    }) if len(frame) else []
    warnings: list[str] = []
    if len(data_versions) > 1:
        warnings.append(
            f"В хранилище {len(data_versions)} версий данных "
            f"({', '.join(data_versions)}): строки считались на разных данных, "
            f"и выжившие могли быть отобраны на устаревшем срезе."
        )
    if n_unknown_thresholds:
        warnings.append(
            f"У {n_unknown_thresholds} вердиктов пороги гейтов не записаны "
            f"(thresholds_json пуст или повреждён): их гейты сочтены порогами "
            f"по умолчанию, и с проверенными по своим порогам они несравнимы. "
            f"Пересчитайте такие конфигурации текущей версией (--force)."
        )
    if len(thresholds.drop_duplicates()) > 1:
        warnings.append(
            "В хранилище вердикты с разными порогами гейтов: воронка судит "
            "каждую строку её порогами (шаг без единого числа — «порога "
            "строки»), но доли строк с разными порогами между собой "
            "несравнимы. Для сравнимой воронки пересчитайте хранилище целиком."
        )
    grade = ("full" if len(screening) == 0
             else "screening" if len(full) == 0 else "mixed")
    if grade == "mixed":
        warnings.append(
            "В хранилище смешаны полные и черновые (screening) вердикты: "
            "воронки считаются раздельно, alive — только по полным."
        )
    elif grade == "screening":
        warnings.append(
            "Все вердикты черновые (screening): alive не вынесен ни по одному, "
            "показаны только кандидаты, требующие полного прогона."
        )

    error_messages = [
        {
            "config_id": str(row["config_id"]),
            "symbol": str(row.get("symbol", "")),
            "timeframe": str(row.get("timeframe", "")),
            "error": str(row.get("error", "")),
        }
        for _, row in errors.head(ERROR_MESSAGES_CAP).iterrows()
    ]

    funnel_block, alive_ok = _funnel_block(full, thresholds.loc[full.index])

    return {
        "schema_version": FUNNEL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {
            "store": None if store_path is None else str(store_path),
            "data_versions": data_versions,
        },
        "thresholds": _threshold_summary(thresholds),
        "n_configs": int(len(frame)),
        "errors": {
            "n": int(len(errors)),
            "by_symbol": _breakdown(errors, "symbol")
            if len(errors) else [],
            "messages": error_messages,
            "truncated": max(0, int(len(errors)) - len(error_messages)),
        },
        "grade": grade,
        "funnel": funnel_block,
        "screening": _screening_block(screening,
                                      thresholds.loc[screening.index]),
        "by_timeframe": _breakdown(frame, "timeframe")
        if len(frame) else [],
        "by_symbol": _breakdown(frame, "symbol")
        if len(frame) else [],
        # Список и счётчик alive строятся по одной маске: расхождение
        # «alive 0 рядом с топом» невозможно по построению.
        "survivors": _survivors(full, alive_ok, n_configs=len(frame),
                                n_verdicts=len(full), n_screening=len(screening),
                                n_errors=len(errors), top=top, sort_by=sort_by),
        "warnings": warnings,
    }


def _pct(fraction) -> str:
    if fraction is None:
        return "—"
    return f"{fraction * 100:.1f}%"


def _fmt_num(value, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def format_funnel(payload: dict) -> str:
    """Текстовая воронка для терминала: те же числа, что в funnel.json."""
    lines: list[str] = []
    n_errors = payload["errors"]["n"]
    lines.append(f"Воронка отбора: {payload['n_configs']} конфигураций в хранилище")
    lines.append(
        f"  вердиктов: {payload['funnel']['n'] + (payload['screening']['n'] if payload['screening'] else 0)}"
        f", строк-ошибок: {n_errors} (исключены из воронки)")
    for warning in payload.get("warnings", []):
        lines.append(f"  ! {warning}")

    def _block(title: str, block: dict, *, screening: bool = False) -> None:
        if block["n"] == 0:
            lines.append(f"\n{title}: 0 вердиктов")
            return
        lines.append(f"\n{title} ({block['n']} вердиктов):")
        total = block["n"]
        for step in block["steps"]:
            lines.append(
                f"  {step['label']:<22} {step['n']:>6}   "
                f"{_pct(step['fraction_of_total']):>7} от {total}   "
                f"{_pct(step['fraction_of_previous']):>7} от предыдущего гейта")
        if screening:
            lines.append(
                f"  {'кандидаты (не alive)':<22} {block['candidates']:>6}   "
                f"нужен полный вердикт")
            if block.get("min_achievable_p") is not None:
                lines.append(
                    f"  минимальный достижимый p-value: "
                    f"1/{block['n_permutations'] + 1} ≈ "
                    f"{block['min_achievable_p']:.4f} — грубее полного порога; "
                    f"черновой вердикт может только отсеять")
        else:
            alive = block["alive"]
            lines.append(
                f"  {'alive':<22} {alive['n']:>6}   "
                f"{_pct(alive['fraction_of_total']):>7} от {total}   "
                f"{_pct(alive['fraction_of_previous']):>7} от предыдущего гейта")
            if block.get("alive_without_pbo"):
                lines.append(
                    f"  ! alive без оценённого PBO: "
                    f"{block['alive_without_pbo']} — гейт PBO не проверялся")

    _block("Полная воронка", payload["funnel"])
    screening = payload["screening"]
    if screening is not None:
        _block("Черновая воронка (screening)", screening, screening=True)
        lines.append(f"  {screening['note']}")

    lines.append("\nРазрез по таймфреймам (вердикты/выжившие/черновые/ошибки):")
    for group in payload["by_timeframe"]:
        lines.append(
            f"  {group['timeframe']:<5} {group['n']:>5} / {group['alive']:>3} / "
            f"{group['screening']:>3} / {group['errors']:>3}")
    lines.append("\nРазрез по символам (вердикты/выжившие/черновые/ошибки):")
    for group in payload["by_symbol"]:
        lines.append(
            f"  {group['symbol']:<9} {group['n']:>5} / {group['alive']:>3} / "
            f"{group['screening']:>3} / {group['errors']:>3}")

    if n_errors:
        lines.append(f"\nОшибки ({n_errors} — не гипотезы, исключены из воронки):")
        for item in payload["errors"]["messages"]:
            lines.append(f"  {item['symbol']} {item['timeframe']} "
                         f"{item['config_id']}: {item['error']}")
        if payload["errors"]["truncated"]:
            lines.append(f"  … ещё {payload['errors']['truncated']}")

    survivors = payload["survivors"]
    if survivors["n"]:
        lines.append(f"\nТоп выживших — {survivors['note']}")
        lines.append(
            f"  {'#':>2}  {'config_id':<12} {'символ':<9} {'тайм':<4} "
            f"{'Sharpe':>7} {'DSR':>7} {'p':>7} {'PBO':>6} {'сделок':>7} "
            f"{'maxDD':>7} {'издержки':>9}")
        for i, entry in enumerate(survivors["top"], start=1):
            lines.append(
                f"  {i:>2}  {entry['config_id']:<12} {entry['symbol']:<9} "
                f"{entry['timeframe']:<4} {_fmt_num(entry['sharpe']):>7} "
                f"{_fmt_num(entry['dsr'], 4):>7} "
                f"{_fmt_num(entry['p_value'], 4):>7} "
                f"{_fmt_num(entry['pbo'], 2):>6} {entry['trades']:>7} "
                f"{_fmt_num(entry['max_dd'], 3):>7} "
                f"{_fmt_num(entry['cost_total'], 4):>9}")
    else:
        lines.append(f"\n{survivors['note']}")
    return "\n".join(lines)


def write_funnel(payload: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Пишет funnel.json и funnel.js (window.ALPHA_FUNNEL) из одного объекта.

    Тем же контрактом, что report.json/report.js: дашборд открывается через
    file://, fetch() к локальным файлам блокируется CORS, поэтому данные
    подключаются как <script src>.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / "funnel.json"
    js_path = out / "funnel.js"
    text = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    json_path.write_text(text, encoding="utf-8")
    js_path.write_text(f"window.ALPHA_FUNNEL = {text};\n", encoding="utf-8")
    return json_path, js_path


def cmd_report(args) -> int:
    """CLI-команда report: воронка по хранилищу свипа + экспорт для дашборда."""
    from alpha_lab.cli import EXIT_ERROR, EXIT_OK
    from alpha_lab.results import read_runs

    store = Path(args.store)
    try:
        frame = read_runs(store)
    except ValueError as exc:
        print(f"Ошибка хранилища: {exc}")
        return EXIT_ERROR
    if len(frame) == 0:
        print(f"Хранилище {store} пусто: сначала прогоните свип "
              f"(alpha-lab sweep --grid ... --out {store})")
        return EXIT_ERROR
    if args.data_version:
        frame = frame[frame["data_version"].astype(str) == str(args.data_version)]
        if len(frame) == 0:
            print(f"В хранилище {store} нет строк версии данных "
                  f"{args.data_version}")
            return EXIT_ERROR
    payload = build_funnel(frame, top=args.top, sort_by=args.sort,
                           store_path=store)
    print(format_funnel(payload))
    out = Path(args.out)
    json_path, js_path = write_funnel(payload, out)
    print(f"\nВоронка: {json_path}\nДашборд подхватит: {js_path}")
    return EXIT_OK


def add_report_subparser(sub) -> None:
    """Регистрирует команду `alpha-lab report` в общем парсере CLI."""
    p = sub.add_parser(
        "report", help="Воронка отбора по хранилищу свипа (P4)")
    p.add_argument("--store", default="results",
                   help="Каталог хранилища результатов (P3)")
    p.add_argument("--out", default="dashboard",
                   help="Каталог для funnel.json/funnel.js (по умолчанию "
                        "dashboard — дашборд подхватит воронку)")
    p.add_argument("--top", type=int, default=10,
                   help="Сколько выживших показать (по умолчанию 10)")
    p.add_argument("--sort", default="dsr", choices=SORTABLE,
                   help="Метрика сортировки выживших (по умолчанию dsr)")
    p.add_argument("--data-version", default=None,
                   help="Учитывать только строки этой версии данных "
                        "(data_version из experiment_id)")
    p.set_defaults(func=cmd_report)
