"""Сборка вердикта. Слепой слой: знает только результаты, не стратегию.

Инвариант: этот модуль НЕ импортирует слой стратегий.
Иначе появляется соблазн «подкрутить» проверку под конкретную стратегию.
Проверяется тестом из tests/test_traps.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_lab.validation.metrics import (
    DEFAULT_PERIODS, max_drawdown, sharpe_ratio, summarize,
)
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

# Пороги-гейты spec 6.5: их записывает хранилище P3 в thresholds_json, чтобы
# воронка судила строку порогами сетки, а не текущими дефолтами. Это ровно
# четыре гейта; n_permutations — параметр прогона, не гейт (screening его
# переопределяет, и в строке он лежит отдельной колонкой).
GATE_THRESHOLD_KEYS = ("min_trades", "min_dsr", "max_p_value", "max_pbo")

# --- P5: режим просеивания -------------------------------------------------
# Permutation-тест — 97% стоимости validate (замер: 646 из 662 мс). Широкий
# перебор просеивается 200 перестановками, финалисты проверяются полностью.
# Потеря точности документирована и записывается в вердикт: минимальный
# достижимый p-value = 1/(200+1) ≈ 0.0050 — всё ещё ниже порога 0.05, но
# грубее (полный прогон различает p ~ 0.001). Черновой вердикт может только
# отсеять: alive по нему не выносится, прохождение гейтов делает конфигурацию
# кандидатом, а не «живой».
SCREENING_PERMUTATIONS = 200
SCREENING_MIN_P = 1.0 / (SCREENING_PERMUTATIONS + 1)
SCREENING_WARNING = (
    f"Черновой вердикт (screening): перестановок {SCREENING_PERMUTATIONS} "
    f"вместо {DEFAULT_THRESHOLDS['n_permutations']}; минимальный достижимый "
    f"p-value = 1/{SCREENING_PERMUTATIONS + 1} ≈ {SCREENING_MIN_P:.4f} — "
    f"грубее полного. alive по черновому вердикту НЕ выносится: прогон может "
    f"только отсеять кандидатов; финалистам нужен полный вердикт "
    f"(без --screening, при необходимости --force)."
)


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
    # Гейты, которые в этом прогоне НЕ проверялись (нет входных данных для
    # проверки). Это не причины смерти и не влияют на alive: отсутствие
    # матрицы конфигураций — свойство одиночного прогона, а не дефект
    # стратегии. Но молчать нельзя: иначе отчёт с пустым reasons читается
    # как «все гейты пройдены», хотя часть из них не запускалась.
    warnings: tuple[str, ...] = field(default_factory=tuple)
    # P5: грейд вердикта. screening=True — черновой прогон: число перестановок
    # уменьшено, alive не выносится (может только отсеять). Поле обязано ехать
    # в отчёт/хранилище/воронку: смешать черновое с полным молча нельзя.
    screening: bool = False
    n_permutations: int = DEFAULT_THRESHOLDS["n_permutations"]


def build_returns_matrix(columns: dict[str, pd.Series]) -> pd.DataFrame:
    """Собирает матрицу доходностей конфигураций (T × N) для PBO/CSCV.

    columns — отображение «имя конфигурации → ряд доходностей БАРОВ» (не
    сделок): pbo_cscv ожидает периодические доходности, T — число баров одной
    и той же истории, N — число конфигураций. Порядок колонок сохраняется.

    Выравнивание проверяется, а не чинится: reindex или обрезка сдвинули бы
    доходности конфигураций друг относительно друга, и PBO посчитался бы по
    несогласованным рядам — правдоподобная тихая ложь, ровно тот класс ошибок,
    против которого существует полигон. Несовпадение длины или временного
    индекса, нефинитные значения и попарно идентичные колонки (одна гипотеза, а
    не свип) — ValueError.

    Идентичность проверяется для каждой пары колонок, а не только для пары с
    опорной: матрица (A, B, B) вырождена так же, как (A, A, B), и PBO по ней
    неотличим от честного перебора. Сравнение лишь с первой колонкой пропускало
    дубликат среди неопорных и занижало штраф за перебор.
    """
    if len(columns) < 2:
        raise ValueError(
            f"матрица доходностей требует минимум 2 конфигурации, получено "
            f"{len(columns)}: PBO на одной колонке неопределён"
        )
    items = list(columns.items())
    ref_name, ref = items[0]
    ref_index = pd.Index(ref.index)
    ref_values = np.asarray(ref, dtype="float64")
    if not np.isfinite(ref_values).all():
        raise ValueError(
            f"ряд '{ref_name}' содержит нефинитные доходности: PBO на таком "
            f"ряде неопределён"
        )
    values: list[np.ndarray] = [ref_values]
    for name, series in items[1:]:
        arr = np.asarray(series, dtype="float64")
        index = pd.Index(series.index)
        if len(arr) != len(ref_values) or not index.equals(ref_index):
            left = index[0] if len(index) else "—"
            right = index[-1] if len(index) else "—"
            ref_left = ref_index[0] if len(ref_index) else "—"
            ref_right = ref_index[-1] if len(ref_index) else "—"
            raise ValueError(
                f"конфигурация '{name}' не выровнена с '{ref_name}': длина "
                f"{len(arr)} против {len(ref_values)}, индекс {left}..{right} "
                f"против {ref_left}..{ref_right}. Матрица PBO строится только "
                f"из рядов на общем временном индексе; молча выравнивать нельзя."
            )
        if not np.isfinite(arr).all():
            raise ValueError(
                f"ряд '{name}' содержит нефинитные доходности: PBO на таком "
                f"ряде неопределён"
            )
        values.append(arr)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if np.array_equal(values[i], values[j]):
                raise ValueError(
                    f"конфигурации [{i}] '{items[i][0]}' и [{j}] "
                    f"'{items[j][0]}' дают идентичные ряды доходностей: это "
                    f"одна гипотеза, а не свип, и PBO на такой матрице вырожден."
                )
    return pd.DataFrame(
        {name: np.asarray(series, dtype="float64") for name, series in items},
        index=ref_index,
    )


def validate(returns, trade_returns, equity, config: dict, n_trials: int,
             strategy_name: str, experiment_id: str,
             returns_matrix=None, price_returns=None, positions=None,
             warnings: tuple[str, ...] = (),
             periods_per_year: int = DEFAULT_PERIODS,
             pbo_value: float | None = None,
             screening: bool = False) -> Verdict:
    """Выносит вердикт. Все пороги — из config, значения по умолчанию в DEFAULT_THRESHOLDS.

    price_returns и positions обязательны для permutation-теста: он перемешивает
    позиции относительно доходностей. Без них проверка невозможна, и вердикт
    выносится отрицательный — тихая деградация недопустима.

    returns_matrix — матрица (T × N) доходностей БАРОВ разных конфигураций на
    общем временном индексе (см. build_returns_matrix). Если она передана,
    PBO считается и гейт spec 6.5 «pbo < 0.5» срабатывает. Если нет (одиночный
    прогон), PBO остаётся NaN, а непроверенный гейт честно называется в
    warnings — не в reasons: отсутствие матрицы свойство прогона, а не дефект
    стратегии.

    warnings — внешние (собранные CLI) предупреждения о непроверенных гейтах;
    к ним добавляется предупреждение о неоценённом PBO. Предупреждения — plain
    strings, не зависят от NaN и не участвуют в alive.

    periods_per_year — годовой множитель Sharpe/Sortino/Calmar. CLI обязан
    передать множитель таймфрейма эксперимента (data.quality.periods_per_year):
    дефолт — часовой (8760) и сохраняет поведение прямых вызовов без
    таймфрейма, а неверный множитель невидимо портит все метрики вердикта.

    pbo_value — заранее посчитанный PBO переданной матрицы. CSCV детерминирован
    и считается по общей матрице свипа, поэтому CLI вычисляет его один раз и
    передаёт float, а не платит за тот же результат на каждой конфигурации.
    None — посчитать здесь (поведение прямых вызовов не изменилось). Передать
    pbo_value без матрицы нельзя: предвычисленному значению не к чему
    относиться, и гейт spec 6.5 молча остался бы непроверенным.

    screening=True — черновой вердикт (P5): permutation-тест идёт
    SCREENING_PERMUTATIONS перестановками вместо thresholds["n_permutations"],
    в warnings добавляется документированная потеря точности, а alive
    принудительно False. Прохождение всех гейтов делает конфигурацию
    кандидатом, но не «живой»: черновой вердикт может только отсеять.
    """
    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year должен быть конечным положительным числом "
            f"(получено {periods_per_year!r}): неверный годовой множитель "
            f"невидимо портит Sharpe/Sortino/Calmar"
        )
    thresholds = {**DEFAULT_THRESHOLDS, **(config or {})}

    r = np.asarray(pd.Series(returns), dtype="float64")
    r = r[np.isfinite(r)]
    t = np.asarray(pd.Series(trade_returns), dtype="float64")
    t = t[np.isfinite(t)]
    eq = np.asarray(pd.Series(equity), dtype="float64")
    eq = eq[np.isfinite(eq)]

    n_trades = int(len(t))
    dsr = deflated_sharpe_ratio(r, n_trials=n_trials)

    # Черновой режим переопределяет число перестановок конфига: его смысл —
    # «дешевле», и конфиг с n_permutations=1000 не должен его отменять.
    n_permutations = int(SCREENING_PERMUTATIONS if screening
                         else thresholds["n_permutations"])
    permutation_available = price_returns is not None and positions is not None
    if permutation_available:
        p_value = permutation_pvalue(
            price_returns, positions,
            n_permutations=n_permutations, seed=0,
        )
    else:
        p_value = 1.0

    pbo = float("nan")
    warn: list[str] = list(warnings)
    if screening:
        warn.append(SCREENING_WARNING)
    if returns_matrix is not None:
        if pbo_value is None:
            from alpha_lab.validation.significance import pbo_cscv
            pbo = pbo_cscv(returns_matrix)
        else:
            pbo = float(pbo_value)
    elif pbo_value is not None:
        # Предвычисленный PBO без матрицы: либо вызывающий перепутал аргументы,
        # либо рассчитывал на гейт, который молча не сработает. Fail closed.
        raise ValueError(
            "pbo_value передан без returns_matrix: предвычисленному PBO не к "
            "чему относиться, и условие spec 6.5 «pbo < 0.5» осталось бы "
            "непроверенным без единого следа в вердикте"
        )
    else:
        # Одиночный прогон: PBO физически не вычислим (нужна матрица
        # T × N конфигураций). Это не причина смерти — но и не «пройдено»:
        # условие spec 6.5 «pbo < 0.5» остаётся непроверенным, и вердикт
        # обязан сказать об этом явно, отдельным каналом warnings.
        warn.append(
            "PBO не оценён: матрица доходностей конфигураций не передана "
            "(одиночный прогон). Условие spec 6.5 «pbo < 0.5» для этого "
            "прогона не проверено; PBO требует многоконфигурационную матрицу "
            "— её строит свип по манифесту (--configs)."
        )

    reasons: list[str] = []
    if not permutation_available:
        reasons.append(
            "permutation-тест не выполнен: не переданы price_returns и positions"
        )
    if n_trades < thresholds["min_trades"]:
        reasons.append(
            f"недостаточно сделок: {n_trades} < {thresholds['min_trades']}"
        )
    if not np.isfinite(dsr) or dsr <= thresholds["min_dsr"]:
        reasons.append(
            f"DSR {dsr:.3f} ≤ {thresholds['min_dsr']} (с поправкой на {n_trials} попыток)"
        )
    if not np.isfinite(p_value) or p_value >= thresholds["max_p_value"]:
        reasons.append(
            f"p-value {p_value:.3f} ≥ {thresholds['max_p_value']} — неотличимо от случая"
        )
    if np.isfinite(pbo):
        if pbo >= thresholds["max_pbo"]:
            reasons.append(f"PBO {pbo:.2f} ≥ {thresholds['max_pbo']} — признак подгонки")
    elif returns_matrix is not None:
        # Матрицу передали, но CSCV её не осилил: молча пропустить проверку нельзя.
        reasons.append(
            f"PBO не вычислен для матрицы {np.asarray(returns_matrix).shape}: "
            f"нужно ≥ 2 конфигураций и ≥ 2·n_blocks наблюдений"
        )

    stats = summarize(r, t, eq, periods_per_year) if len(r) else {}

    return Verdict(
        strategy_name=strategy_name,
        experiment_id=experiment_id,
        sharpe=sharpe_ratio(r, periods_per_year),
        dsr=dsr,
        p_value=p_value,
        max_dd=max_drawdown(eq) if len(eq) else 0.0,
        total_return=float(eq[-1] / eq[0] - 1.0) if len(eq) > 1 else 0.0,
        trades=n_trades,
        n_configs_tried=n_trials,
        # Черновой вердикт не имеет права утверждать alive: прохождение
        # черновых гейтов — кандидатура, а не результат.
        alive=len(reasons) == 0 and not screening,
        reasons=tuple(reasons),
        metrics=stats,
        pbo=pbo,
        warnings=tuple(warn),
        screening=bool(screening),
        n_permutations=n_permutations,
    )
