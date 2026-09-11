"""Единая точка входа.

    uv run alpha-lab ingest   --config configs/universe.yaml
    uv run alpha-lab validate --config configs/experiments/mr_base.yaml
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.causality import assert_strategy_is_causal
from alpha_lab.config import Experiment, load_experiment, load_manifest, load_universe
from alpha_lab.data.quality import FREQ_DELTA, clean_mask, periods_per_year
from alpha_lab.data.query import (
    align_funding_to_bars, data_version, load_bars, load_funding,
    matched_funding_events,
)
from alpha_lab.data.store import DEFAULT_ROOT
from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost
from alpha_lab.report.writer import build_report, write_report
from alpha_lab.strategies.base import build_strategy, history_bars_of
from alpha_lab.validation.validator import (
    SCREENING_WARNING, build_returns_matrix, validate,
)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2

DEFAULT_JOURNAL = Path("reports") / "trials.jsonl"

# Доля объёма бара, выше которой заявка считается неисполнимой по смоделированной
# цене. Явно передаётся в движок, чтобы диагностика и её печать не разъехались
# при смене дефолта run_backtest.
MAX_PARTICIPATION = 0.01

# Запас маскирования разрыва назад, в барах. Один бар — это ровно то решение,
# которое иначе исполнилось бы через дыру: позиция бара i (первого после
# разрыва) равна target[i-1] (held[t] = target[t-1]), и доходность
# close[i]/close[i-1] − 1 получена сквозь неизвестное движение в пропуске.
# Память стратегии назад здесь ни при чём: все бары до дыры контигуозны, и
# решения на них достоверны. Меньше нельзя — перенос останется; больше не
# нужно — бар i-2 контигуозен с i-1.
GAP_MASK_LOOKBACK_BARS = 1


def git_hash() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "nogit"
    except (OSError, subprocess.SubprocessError):
        return "nogit"


def experiment_id(config: dict, dv: str, gh: str) -> str:
    payload = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    digest = hashlib.sha256(f"{payload}|{dv}|{gh}".encode("utf-8"))
    return digest.hexdigest()[:16]


def trial_fingerprint(key: dict) -> str:
    """Каноническая подпись trial-ключа для журнала и семейных меток.

    sort_keys/ensure_ascii убирают оформление (порядок ключей, экранирование),
    чтобы одна гипотеза не выглядела двумя. Запись обязана быть JSON-точной:
    default=str здесь не используется — молчаливое приведение типов склеило бы
    разные гипотезы.
    """
    return json.dumps(key, sort_keys=True, ensure_ascii=False)


def count_prior_trials_for_keys(journal: Path, keys: list[dict]) -> int:
    """Сколько попыток текущего поиска уже лежит в журнале.

    Попытка — одна запись журнала (один прогон одной конфигурации). ``keys`` —
    trial-ключи текущего прогона (для свипа — все конфигурации манифеста).
    Учитываются все различные записи, относящиеся к этому поиску:

    * запись, ключ которой совпадает с одним из ``keys``, — прямое прошлое
      испытание текущей гипотезы;
    * запись с семейной меткой ``family`` (отсортированный список отпечатков
      trial-ключей свипа), пересекающейся с ``keys``, — испытание, сделанное
      внутри того же перебора. Свип из N конфигураций — это N попыток, поэтому
      одиночный перезапуск его участника обязан нести штраф всей семьи, а не
      одной своей строки: иначе победителя свипа можно «отмыть» одиночным
      запуском и получить менее дефлированный DSR — ошибка в опасную сторону,
      ровно та, против которой существует поправка.

    Записи без поля ``family`` (журнал старых версий) учитываются только прямым
    совпадением ключа — как до появления семейных меток. Метка не-список или со
    не-строковыми элементами семейной информацией не считается: разбор журнала
    обязан быть устойчивым к мусору и не наказывать за него лишними попытками.

    Дедупликация по записям, а не сумма одиночных запросов по каждому ключу:
    если текущий прогон сам покрывает всю семью (повторный свип того же
    манифеста), каждая запись семьи обязана считаться один раз, иначе штраф
    раздуло бы в N раз.
    """
    journal = Path(journal)
    if not journal.exists():
        return 0
    fingerprints = {trial_fingerprint(key) for key in keys}
    try:
        # errors="replace": один битый байт (обрыв записи при падении процесса)
        # не должен ронять весь прогон — испорченная строка просто пропустится.
        text = journal.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        # Журнал нечитаем целиком (каталог на месте файла, нет прав). Падать
        # нельзя — это черновик, а не входные данные. Но и молчать нельзя: ноль
        # попыток завышает DSR, поэтому честно предупреждаем о занижении.
        print(f"Предупреждение: журнал попыток нечитаем ({exc}); "
              f"считаю, что попыток не было — DSR может быть завышен",
              file=sys.stderr)
        return 0
    count = 0
    skipped = 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            if not isinstance(record, dict):
                skipped += 1
                continue
            relevant = trial_fingerprint(record["key"]) in fingerprints
            if not relevant:
                family = record.get("family")
                if isinstance(family, list):
                    relevant = any(
                        isinstance(member, str) and member in fingerprints
                        for member in family
                    )
            if relevant:
                count += 1
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
            # Валидный JSON не-объект (null/[]/123/"x"), обрыв объекта или
            # запись без "key" — строку пропускаем, счёт остальных не теряем.
            skipped += 1
            continue
    if skipped:
        # Пропущенная запись — потерянная попытка: n_trials занижается, DSR
        # завышается. Молчать нельзя, как и в случае нечитаемого файла целиком.
        print(f"Предупреждение: в журнале {journal} пропущено повреждённых "
              f"строк: {skipped}; n_trials занижен — DSR может быть завышен",
              file=sys.stderr)
    return count


def count_prior_trials(journal: Path, key: dict) -> int:
    """Сколько раз эта гипотеза уже прогонялась — с учётом её семьи.

    Число попыток нужно Deflated Sharpe: без него главная защита от
    оверфиттинга отключается ровно там, где она нужна. Наивная «одна попытка» —
    самообман. Полная семантика семейных меток и устойчивости к мусору — в
    count_prior_trials_for_keys; эта обёртка отвечает про одну гипотезу.
    """
    return count_prior_trials_for_keys(journal, [key])


def journal_problem(journal: Path) -> str | None:
    """Причина, по которой журнал непригоден, или None, если он рабочий.

    Проверяются обе операции CLI над журналом — чтение и дозапись. Зовётся до
    бэктеста: непригодный журнал нельзя «пережить», потому что n_trials без
    него занижается, а недодефлированный вердикт выглядит ЛУЧШЕ правды.
    """
    journal = Path(journal)
    try:
        if journal.is_dir():
            return "это каталог, а не файл"
        journal.parent.mkdir(parents=True, exist_ok=True)
        if journal.exists():
            # Дозапись в существующий журнал проверяется напрямую.
            with journal.open("a", encoding="utf-8"):
                pass
            # Чтение — вторая операция CLI над журналом, и проверять только
            # запись мало: файл бывает доступен на дозапись, но не на чтение
            # (POSIX 0200, ACL Windows). Тогда count_prior_trials поймает
            # OSError, вернёт 0, и отчёт запишется с n_trials = 1 — заниженный
            # штраф DSR, то есть оптимистичный вердикт на диске.
            with journal.open("r", encoding="utf-8"):
                pass
        else:
            # Журнала ещё нет: проверяем, что каталог вообще доступен на
            # запись. Пробник удаляется сразу, чтобы проверка не оставляла
            # пустой журнал — файл без записи выглядел бы как попытка.
            with tempfile.NamedTemporaryFile(dir=journal.parent,
                                             prefix=".journal_probe_"):
                pass
    except OSError as exc:
        return str(exc)
    return None


def log_trial(journal: Path, key: dict, experiment: str, metrics: dict,
              family: list[str] | None = None) -> None:
    """Дозаписывает попытку в журнал.

    family — отсортированный список отпечатков trial-ключей свипа, если прогон
    был частью перебора (см. count_prior_trials_for_keys). Метка позволяет
    последующему одиночному прогону участника увидеть попытки всей семьи.
    Одиночный прогон family не пишет: он не перебор, и его запись обязана
    читаться ровно как раньше.
    """
    journal = Path(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "experiment_id": experiment, **metrics}
    if family is not None:
        record["family"] = list(family)
    with journal.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _configure_stdio() -> None:
    """Переводит потоки вывода в UTF-8.

    На Windows консоль по умолчанию работает в кодовой странице (cp866/cp1251),
    и русский текст превращается в мусор. Потоки могут быть подменены в тестах
    (например, capsys), поэтому reconfigure вызывается только при наличии метода.
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def _cmd_ingest(args) -> int:
    try:
        universe = load_universe(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка конфига: {exc}", file=sys.stderr)
        return EXIT_ERROR

    from alpha_lab.data.ingest import ingest_universe

    results = ingest_universe(universe, args.freq, root=Path(args.data_root))
    # Делистингованные пары не растворяются в «ок»: они посчитаны отдельно, а
    # исключённые явным include_delisted=false — тоже (молчаливый отказ от них
    # создавал бы ошибку выживаемости незаметно).
    ok = [r for r in results if r.ok and not r.excluded]
    bad = [r for r in results if not r.ok]
    delisted = [r for r in results if r.delisted]
    excluded = [r for r in results if r.excluded]
    for r in results:
        if r.excluded:
            mark = "SKIP"
        elif r.delisted:
            mark = "DELIST"
        elif r.ok:
            mark = "OK "
        else:
            mark = "ERR"
        detail = r.quality if r.ok else r.error
        print(f"[{mark}] {r.symbol:<12} {r.kind:<8} {r.rows:>10,}  {detail}")
    print(f"\nИтого: {len(ok)} успешно, {len(bad)} с ошибками, "
          f"делистингованных: {len(delisted)} "
          f"(исключено: {len(excluded)})")
    return EXIT_OK if not bad else EXIT_FAILED


def _gap_stats(bars: pd.DataFrame, timeframe: str) -> tuple[int, int]:
    """Число разрывов и суммарное число пропущенных баров на таймфрейме.

    Разрыв — интервал между соседними барами больше шага таймфрейма. Шаг
    берётся из FREQ_DELTA (тот же источник, что у check_bars), а масштаб
    пропуска — (интервал / шаг − 1): одна дыра в час и дыра в сутки не должны
    выглядеть одинаково. check_bars разрывы считает, но вердикт о них молчит,
    а стратегия торгует через них как через обычный бар.
    """
    delta = FREQ_DELTA[timeframe]
    diffs = pd.to_datetime(bars["ts"], utc=True).diff().dropna()
    holes = diffs[diffs > delta]
    if holes.empty:
        return 0, 0
    missing = int(round(float(((holes - delta) / delta).sum())))
    return int(len(holes)), missing


def _gap_starts(bars: pd.DataFrame, timeframe: str) -> np.ndarray:
    """Индексы первых баров после разрывов — единственное определение разрыва.

    Разрыв — интервал между соседними барами больше шага таймфрейма. Маска и
    сегментация обязаны видеть одни и те же дыры, поэтому детектор один:
    второй способ понимания разрыва неизбежно разъехался бы с первым.
    """
    if len(bars) < 2:
        return np.empty(0, dtype=np.int64)
    diffs = pd.to_datetime(bars["ts"], utc=True).diff()
    return np.flatnonzero((diffs > FREQ_DELTA[timeframe]).to_numpy())


def _gap_mask(bars: pd.DataFrame, timeframe: str,
              lookback: int = GAP_MASK_LOOKBACK_BARS,
              lookahead: int = 0) -> np.ndarray:
    """Маска баров, которые нельзя торговать из-за разрыва (True = исключён).

    Разрыв — интервал между соседними барами больше шага таймфрейма. Для
    разрыва перед баром i обнуляются цели на [i-lookback, i-1+lookahead]:

    * назад — движок удерживает позицию один бар (held[i] = target[i-1]),
      поэтому решение на последнем баре перед дырой принималось, когда о
      пропуске ещё не было известно, и без lookback=1 позиция прошла бы сквозь
      неизвестное движение и получила бы close[i]/close[i-1] − 1. Память
      стратегии назад ни при чём: все бары до дыры контигуозны, решения на них
      достоверны;
    * вперёд запас не нужен (lookahead=0): ряд сегментируется
      (_generate_segmented), generate на первом же баре после дыры стартует
      заново, и его окна не пересекают пропуск. Самая ранняя доходность,
      которую может заработать позиция нового участка, — close[i+1]/close[i] − 1,
      то есть полностью контигуозный бар. Маскирование вперёд лишь подавляло бы
      достоверные сигналы.

    Разрыв не инвалидирует весь прогон: 120 пропущенных часов на четырёх годах
    сделали бы пару непригодной, тогда как честный ответ — не торговать бар
    перед дырой и продолжить после неё. Границы клипуются по краям ряда.
    """
    n = len(bars)
    mask = np.zeros(n, dtype=bool)
    for i in _gap_starts(bars, timeframe):   # i — первый бар после разрыва
        lo = max(0, i - lookback)
        hi = min(n - 1, i - 1 + lookahead)
        mask[lo:hi + 1] = True
    return mask


def _generate_segmented(strategy, bars: pd.DataFrame,
                        timeframe: str) -> pd.Series:
    """Позиции стратегии по непрерывным участкам ряда, разделённым разрывами.

    Разрыв означает, что ряд неконтигуозен: что происходило в дыре — неизвестно.
    Стратегия не имеет права видеть сквозь неё, поэтому generate вызывается на
    каждом непрерывном участке отдельно, а результаты склеиваются позиционно.
    Это перезапускает внутреннее состояние стратегии на каждой дыре по
    построению — независимо от того, как стратегия написана, и без единого
    протокольного шва, который можно забыть.

    Обнуления целей на маскированных барах для этого НЕ достаточно: у MR
    состояние живёт внутри simulate_bracket_exits (direction, entry_bar,
    cur_sl, cur_tp), и сделка, открытая до дыры, на первом немаскированном
    баре переизлучается с предразрывной ценой входа, стопом/тейком и часами
    max_bars, не считавшими пропущенные бары. Маска гасит выход, но не память.

    Участок короче прогрева стратегии — не ошибка: generate на коротком входе
    не даёт сигналов, и это честный ответ для обрывка. Индекс исходного ряда
    сохраняется: результат равен длине bars и выровнен по bars.index.
    """
    starts = _gap_starts(bars, timeframe)
    if len(starts) == 0:
        # Ряд без дыр — ровно прежнее поведение, без лишних срезов и склейки.
        return strategy.generate(bars)
    bounds = [0, *starts.tolist(), len(bars)]
    parts = [
        np.asarray(strategy.generate(bars.iloc[a:b]), dtype="float64")
        for a, b in zip(bounds[:-1], bounds[1:])
        if a < b
    ]
    return pd.Series(np.concatenate(parts), index=bars.index, name="position")


def _universe_payload(universe) -> dict:
    """Состав юниверса для experiment_id.

    Порядок символов в файле — оформление, а не гипотеза: состав сравнивается
    как множество, поэтому список сортируется. Иначе перестановка строк в
    universe.yaml меняла бы experiment_id, хотя исследование то же.
    """
    return {
        "symbols": sorted(universe.symbols),
        "market": universe.market,
        "start": universe.start,
        "end": universe.end,
        "include_delisted": universe.include_delisted,
    }


def _experiment_payload(exp: Experiment, symbol: str, universe) -> dict:
    """Нагрузка experiment_id одной конфигурации.

    Пороги валидации — часть гипотезы: без них два прогона с разными
    min_trades делят id, отчёт второго молча затирает первый, а журнал считает
    их одной попыткой. Символ — тоже часть гипотезы (trial_key в журнале уже
    включает его): без него прогоны того же конфига по разным символам делят
    id. Состав юниверса — часть гипотезы (spec 5): смена состава обязана менять
    id, а порядок строк в файле — оформление, а не гипотеза.
    """
    return {
        "experiment": exp.name, "strategy": exp.strategy, "params": exp.params,
        "timeframe": exp.timeframe, "start": exp.start, "end": exp.end,
        "costs": exp.costs,
        "validation": exp.validation,
        "symbol": symbol,
        "universe": _universe_payload(universe),
    }


def _sweep_problem(configs: list[tuple[Path, Experiment]]) -> str | None:
    """Причина, по которой набор конфигураций не является свипом, или None.

    Проверки идут до единого бэктеста:

    * одинаковые условия прогона (timeframe/start/end): матрица доходностей
      выравнивается по времени, и конфиг на другом периоде дал бы другую длину
      или другой индекс. Это отвергается громко, а не выравнивается молча;
    * различные гипотезы: конфиги, совпадающие по strategy/params/costs/
      validation/периоду (имя — ярлык, а не гипотеза), не образуют перебор —
      PBO на идентичных колонках вырожден.
    """
    base_path, base = configs[0]
    base_label = f"[0] {base.name} ({base_path})"
    for i, (path, exp) in enumerate(configs[1:], start=1):
        for field_name in ("timeframe", "start", "end"):
            got, want = getattr(exp, field_name), getattr(base, field_name)
            if got != want:
                return (
                    f"конфигурация [{i}] {exp.name} ({path}) отличается от "
                    f"заголовочной {base_label} по {field_name}: {got!r} ≠ "
                    f"{want!r}. Свип обязан идти на одном символе, периоде и "
                    f"таймфрейме: иначе матрица доходностей конфигураций не "
                    f"выравнивается, и PBO считался бы по несогласованным "
                    f"рядам."
                )

    def hypothesis(exp: Experiment) -> dict:
        return {
            "strategy": exp.strategy, "params": exp.params,
            "timeframe": exp.timeframe, "start": exp.start, "end": exp.end,
            "costs": exp.costs, "validation": exp.validation,
        }

    seen: dict[str, tuple[int, str]] = {
        json.dumps(hypothesis(base), sort_keys=True, ensure_ascii=False,
                   default=str): (0, base.name)
    }
    for i, (path, exp) in enumerate(configs[1:], start=1):
        fingerprint = json.dumps(hypothesis(exp), sort_keys=True,
                                 ensure_ascii=False, default=str)
        if fingerprint in seen:
            first_i, first_name = seen[fingerprint]
            return (
                f"конфигурации [{first_i}] {first_name} и [{i}] {exp.name} "
                f"({path}) идентичны по strategy/params/периоду/costs/"
                f"validation (имя — ярлык, а не гипотеза): это одна гипотеза, "
                f"а не две. PBO на идентичных колонках вырожден — перебор не "
                f"состоялся."
            )
        seen[fingerprint] = (i, exp.name)
    return None


def _params_summary(params: dict) -> str:
    """Короткая подпись параметров для таблицы свипа."""
    return ",".join(f"{k}={v}" for k, v in sorted(params.items()))


# Единственный источник текста отказа от причинностной проверки: он уходит и в
# stderr одиночного прогона, и в warnings вердикта, и в строки пакетного свипа.
CAUSALITY_DISABLED_WARNING = (
    "Причинностная проверка ОТКЛЮЧЕНА (--skip-causality): вердикт не "
    "защищён от look-ahead. Подглядывание статистикой по результатам "
    "не ловится (spec 8.1), поэтому подглядывающая стратегия может "
    "получить «ЖИВА» с отличными метриками. Флаг — только для отладки."
)


@dataclass(frozen=True)
class LoadedData:
    """Бары одного (символ, таймфрейм) и всё, что считается из них один раз.

    Пакетный свип грузит данные на каждую пару (символ, таймфрейм) один раз и
    переиспользует их всеми конфигурациями группы: 22.2 с загрузки на
    конфигурацию сделали бы перебор тысяч гипотез нерабочим.
    """

    symbol: str
    timeframe: str
    bars: pd.DataFrame
    ts_index: pd.Index
    tradable: np.ndarray
    dirty: int
    gaps: int
    missing_bars: int
    gap_masked: int
    data_warnings: tuple[str, ...]
    funding_rate: pd.Series | None
    funding_available: bool
    funding_events: int
    funding_matched: int
    ppy: int


class CausalityError(RuntimeError):
    """Стратегия не прошла причинностный harness: бэктест не запускался."""

    def __init__(self, strategy_name: str, cause: BaseException):
        super().__init__(
            f"Стратегия '{strategy_name}' не прошла проверку причинности: "
            f"{cause}"
        )
        self.strategy_name = strategy_name
        self.cause = cause


@dataclass(frozen=True)
class RunOutcome:
    """Результат одной конфигурации до вердикта."""

    experiment: Experiment
    result: object
    trades: pd.Series
    history: int
    causality_cuts: int


def prepare_data(root: Path, symbol: str, timeframe: str, start: str,
                 end: str | None) -> LoadedData:
    """Грузит бары и фундамент издержек/масок для одного (символ, таймфрейм).

    Единая точка для одиночного прогона и пакетного свипа: маски, funding и
    предупреждения о данных обязаны быть одинаковыми в обоих путях, иначе
    вердикты разъедутся. FileNotFoundError пробрасывается: вызывающий решает,
    печатать «Данных нет» и завершаться (CLI) или записать отказ группы и
    продолжить (свип).
    """
    ppy = periods_per_year(timeframe)
    bars = load_bars(root, symbol, "1m", start, end, resample=timeframe)

    # Funding: недоступность — это не «ставка 0». Различаем отсутствие файла и
    # пустой файл и считаем, сколько событий реально легло на бары: панель
    # издержек обязана отличать «данных нет» от «funding был нулевым», иначе
    # вердикт «жива» может стоять на заниженных издержках.
    funding_available = False
    funding_events = 0
    funding_matched = 0
    funding_rate = None
    funding_warning = None
    try:
        funding = load_funding(root, symbol)
    except FileNotFoundError:
        funding = None
        funding_warning = (
            "Funding недоступен (файл не найден): ставки не применены, "
            "издержки занижены, вердикт оптимистичен"
        )
    if funding is not None:
        funding_events = int(len(funding))
        if funding_events == 0:
            # load_funding отдаёт все нули на пустой parquet без предупреждения,
            # а align_funding_to_bars заливает нули и в несовпавшие бары.
            funding_warning = (
                "Funding недоступен (файл пуст): ставки не применены, "
                "издержки занижены, вердикт оптимистичен"
            )
        else:
            funding_available = True
            # Событие привязывается к содержащему его бару и суммируется с
            # соседями по бару (на 1d 00:00+08:00+16:00 — один бар). Явный
            # timeframe задаёт шаг бара: события из пропущенных баров не
            # приклеиваются к соседним.
            funding_rate = align_funding_to_bars(bars, funding, timeframe)
            funding_matched = matched_funding_events(bars, funding, timeframe)
            if funding_matched == 0:
                funding_warning = (
                    f"Funding не привязан ни к одному бару "
                    f"({funding_events} событий): издержки занижены, "
                    f"вердикт оптимистичен"
                )
    if funding_warning:
        print(f"Предупреждение: {funding_warning}", file=sys.stderr)

    # Грязные бары: стратегия на них не торгует. Не чиним и не интерполируем —
    # иначе тихо неверный бэктест выглядел бы как честный.
    clean = clean_mask(bars).to_numpy()
    dirty = int((~clean).sum())
    if dirty:
        print(f"Предупреждение: {dirty} грязных баров исключено из торговли",
              file=sys.stderr)

    # Разрывы: через дыру цена шла неизвестно как, поэтому торговля через неё
    # не ведётся (spec 8: грязные данные → стратегия не торгует). Маска
    # компонуется с маской грязных баров; движок не меняется. Forward-маски
    # нет: ряд сегментируется (_generate_segmented), и на первом же баре после
    # дыры история стратегии перезапущена — окна сквозь пропуск не тянутся.
    # history_bars остаётся в отчёте диагностикой подлинной памяти стратегии.
    gaps, missing_bars = _gap_stats(bars, timeframe)
    gap_excluded = _gap_mask(bars, timeframe, lookahead=0)
    gap_masked = int(gap_excluded.sum())
    gap_warning = None
    if gaps:
        gap_warning = (
            f"В данных разрывов: {gaps}, пропущено баров: "
            f"{missing_bars} (таймфрейм {timeframe}) — торговля "
            f"приостановлена на {gap_masked} барах вокруг них "
            f"(lookback={GAP_MASK_LOOKBACK_BARS}: последнее решение перед "
            f"дырой, иначе позиция прошла бы сквозь неё; lookahead=0: "
            f"сегментация перезапускает историю стратегии с первого бара "
            f"после дыры); состояние стратегии перезапускается на каждом "
            f"непрерывном участке"
        )
        print(f"Предупреждение: {gap_warning}", file=sys.stderr)

    tradable = clean & ~gap_excluded
    data_warnings = tuple(
        w for w in (funding_warning, gap_warning) if w)
    # ВАЖНО: load_bars заканчивается reset_index(drop=True), поэтому бары и
    # выходы движка проиндексированы RangeIndex (0..n-1). build_report делает
    # из индекса series.ts через pd.Timestamp(t), а это наносекунды от эпохи
    # 1970 года — ось времени всех графиков дашборда становится бессмысленной.
    # Переиндексируем позиционные ряды реальными временами баров; set_axis не
    # меняет длину, поэтому проверка выравнивания в writer проходит.
    ts_index = pd.Index(bars["ts"].to_numpy(), name="ts")
    return LoadedData(
        symbol=symbol, timeframe=timeframe, bars=bars, ts_index=ts_index,
        tradable=tradable, dirty=dirty, gaps=gaps, missing_bars=missing_bars,
        gap_masked=gap_masked, data_warnings=data_warnings,
        funding_rate=funding_rate, funding_available=funding_available,
        funding_events=funding_events, funding_matched=funding_matched, ppy=ppy,
    )


def run_config(data: LoadedData, exp: Experiment, *,
               skip_causality: bool = False) -> RunOutcome:
    """Прогоняет одну конфигурацию: harness → сегментированный generate → бэктест.

    Общий кусок одиночного CLI и пакетного свипа. Причинностная проверка идёт
    до бэктеста каждой конфигурации: look-ahead статистикой по результатам не
    ловится (spec 8.1), поэтому единственная работающая защита — запрос к
    функции решения. Провал harness — CausalityError: CLI печатает «прогон
    остановлен», свип записывает отказ конфигурации и идёт дальше.
    """
    strategy = build_strategy(exp.strategy, exp.params)
    causality_cuts = 0
    if not skip_causality:
        try:
            causality_cuts = assert_strategy_is_causal(strategy, data.bars)
        except (AssertionError, ValueError) as exc:
            raise CausalityError(
                getattr(strategy, "name", exp.strategy), exc) from exc
    # Разрыв — не только «не торговать в окне»: ряд неконтигуозен, и
    # стратегия не имеет права видеть сквозь дыру. generate вызывается на
    # каждом непрерывном участке отдельно, поэтому внутреннее состояние
    # стратегии (у MR — direction/entry_bar/cur_sl/cur_tp внутри
    # simulate_bracket_exits) перезапускается на дыре по построению.
    # Обнуление целей этого не лечит: сделка, открытая до дыры, всплыла бы
    # на первом немаскированном баре с предразрывной ценой входа, стопом/
    # тейком и часами max_bars, которые не считали пропущенные бары. Маска
    # гасит ровно бар перед дырой — позицию, которая иначе прошла бы сквозь
    # пропуск; вперёд маски нет, потому что сегментация уже перезапустила
    # историю стратегии.
    #
    # copy=True: to_numpy() в pandas 3 отдаёт read-only массив, а маска
    # ниже пишет в него на месте. Обнуляются цели и грязных баров, и бара
    # перед каждым разрывом.
    targets = _generate_segmented(strategy, data.bars,
                                  exp.timeframe).to_numpy(
        dtype="float64", copy=True)
    targets[~data.tradable] = 0.0
    targets = pd.Series(targets, index=data.bars.index)

    cost_model = RealisticCost.from_config(exp.costs)
    result = run_backtest(data.bars, targets, cost_model,
                          funding_rate=data.funding_rate,
                          max_participation=MAX_PARTICIPATION)
    return RunOutcome(
        experiment=exp, result=result, trades=trade_returns(result),
        history=history_bars_of(strategy), causality_cuts=causality_cuts,
    )


def validate_config(outcome: RunOutcome, data: LoadedData, *,
                    experiment_id: str, n_trials: int,
                    returns_matrix=None, pbo_value=None,
                    warnings: tuple[str, ...] | None = None,
                    screening: bool = False):
    """Выносит вердикт по результату прогона — тот же validate, что и в CLI.

    Валидатору уходят позиции ДВИЖКА (result.positions — удержанные,
    held[t] = target[t-1]), а не сырые цели strategy.generate(): сырые цели
    смещены на бар, поэтому permutation-тест сравнил бы сигнал не с той
    доходностью (измерено 0.343656 у сырых против 0.000999 у позиций).

    screening=True — черновой вердикт P5: меньше перестановок, alive не
    выносится, грейд едет в отчёт и строку хранилища.
    """
    exp = outcome.experiment
    return validate(
        returns=outcome.result.returns, trade_returns=outcome.trades,
        equity=outcome.result.equity, config=exp.validation,
        n_trials=n_trials, strategy_name=exp.name,
        experiment_id=experiment_id,
        price_returns=outcome.result.price_returns,
        positions=outcome.result.positions,
        warnings=data.data_warnings if warnings is None else warnings,
        periods_per_year=data.ppy,
        returns_matrix=returns_matrix, pbo_value=pbo_value,
        screening=screening,
    )


def _cmd_validate(args) -> int:
    # Источник конфигураций ровно один: --config (одиночный прогон, PBO для
    # него невычислим) или --configs (манифест свипа, PBO вычислим). Молчаливое
    # предпочтение одного из них скрыло бы, какой режим реально отработал.
    if bool(args.config) == bool(args.configs):
        print(
            "Ошибка: укажите ровно один источник конфигураций: --config "
            "(одиночный прогон) или --configs (манифест свипа). Отчёт не "
            "записан.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    manifest_path: Path | None = None
    try:
        if args.configs:
            manifest_path = Path(args.configs)
            paths = load_manifest(manifest_path)
            if len(paths) < 2:
                print(
                    f"Ошибка: манифест {manifest_path} содержит {len(paths)} "
                    f"конфигурацию — для PBO нужно минимум 2: CSCV сравнивает "
                    f"конфигурации между собой, на одной колонке он "
                    f"неопределён. Отчёт не записан.",
                    file=sys.stderr,
                )
                return EXIT_ERROR
            configs: list[tuple[Path, Experiment]] = [
                (p, load_experiment(p)) for p in paths
            ]
        else:
            cfg_path = Path(args.config)
            configs = [(cfg_path, load_experiment(cfg_path))]
    except (OSError, ValueError) as exc:
        # OSError, а не только FileNotFoundError: каталог вместо манифеста
        # (IsADirectoryError на POSIX, PermissionError на Windows) и файл без
        # прав на чтение обязаны давать чистый EXIT_ERROR, а не трейсбек.
        print(f"Ошибка конфига: {exc}", file=sys.stderr)
        return EXIT_ERROR

    sweep_mode = len(configs) > 1
    if sweep_mode:
        problem = _sweep_problem(configs)
        if problem is not None:
            print(f"Ошибка манифеста: {problem} Отчёт не записан.",
                  file=sys.stderr)
            return EXIT_ERROR

    # Заголовочная — первая конфигурация манифеста. Правило «первая», а не
    # «лучшая по Sharpe»: выбор лучшей по тем же данным был бы ещё одним
    # отбором, который PBO и DSR обязаны штрафовать. Пользователь объявляет
    # гипотезу заранее порядком строк, и отчёт называет её явно.
    exp = configs[0][1]

    # Юниверс — входной контракт прогона, а не справка: прогон символа вне
    # зафиксированного состава исследовал бы не ту гипотезу, а состав обязан
    # менять experiment_id (spec 5). Поэтому отсутствующий или битый файл —
    # ошибка запуска. Молчаливое «проверять нечего» было бы тихой деградацией:
    # отчёт выглядел бы легитимным, не неся ни проверки членства, ни состава.
    try:
        universe = load_universe(args.universe)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Ошибка юниверса: {exc}", file=sys.stderr)
        return EXIT_ERROR

    if args.symbol not in universe.symbols:
        hints = difflib.get_close_matches(args.symbol, universe.symbols, n=3)
        hint = f" Ближайшее совпадение: {', '.join(hints)}." if hints else ""
        print(
            f"Ошибка: символ '{args.symbol}' не входит в юниверс "
            f"{args.universe} (символов: {len(universe.symbols)}).{hint} "
            f"Доступные символы: {', '.join(universe.symbols)}. "
            f"Отчёт не записан. Добавьте символ в юниверс или выберите "
            f"другой --universe.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    root = Path(args.data_root)
    dv = data_version(root)
    # git_hash запускает подпроцесс: на свип он считается один раз, а не по
    # разу на конфигурацию.
    gh = git_hash()
    exp_ids = [
        experiment_id(_experiment_payload(e, args.symbol, universe), dv, gh)
        for _, e in configs
    ]
    exp_id = exp_ids[0]

    # Годовой множитель метрик — из таймфрейма эксперимента, а не часовой
    # константы: на 1d часы завышают Sharpe в sqrt(24) раз, Calmar — в 24.
    # prepare_data берёт его из того же таймфрейма (periods_per_year).
    symbol = args.symbol
    try:
        data = prepare_data(root, symbol, exp.timeframe, exp.start, exp.end)
    except FileNotFoundError as exc:
        print(f"Данных нет: {exc}", file=sys.stderr)
        return EXIT_ERROR
    bars = data.bars

    # Причинностная проверка идёт до бэктеста каждой конфигурации: look-ahead
    # статистикой по результатам не ловится (spec 8.1), поэтому единственная
    # работающая защита — запрос к функции решения. Без неё подглядывающая
    # стратегия получает «ЖИВА» с отличными метриками, и это не гипотеза, а
    # измеренный факт (tests/test_traps.py).
    #
    # Политика разрезов — штатная плотная сетка harness'а (n // TARGET_CUTS =
    # 200 точек, для рядов <= 600 баров — сплошная). На реальном прогоне BTC
    # 35 064 бара это 205 точек и ~1.6 с — не material на фоне загрузки данных
    # (~22 с) и бэктеста, поэтому сплошное покрытие (cut_points="all", ~35 000
    # вызовов generate) не берётся. Плотная сетка ловит утечку в любой позиции
    # непосредственно перед точкой разреза; утечка, целиком лежащая между
    # соседними точками (окно <= n // 200 баров), теоретически может остаться
    # незамеченной — это зафиксировано в docstring harness'а.
    #
    # Причинностная проверка оставляет след в отчёте по обоим путям: и когда
    # она шла, и когда её отключили. Иначе готовый report.json, снятый с
    # единственной защиты от look-ahead, неотличим от защищённого — тихая
    # деградация, недопустимая по spec. Число точек важно и на защищённом
    # пути: гарантия harness выборочная (шаг сетки n // TARGET_CUTS), и без
    # счётчика покрытие выглядит полным.
    causality_checked = not args.skip_causality
    causality_warning = None
    if args.skip_causality:
        causality_warning = CAUSALITY_DISABLED_WARNING
        print(f"Предупреждение: {causality_warning}", file=sys.stderr)

    # Предупреждения о данных и отключённой проверке уходят тем же каналом,
    # что и неоценённый PBO: они не делают вердикт мёртвым (нет данных — не
    # дефект стратегии; отказ от проверки — осознанный флаг), но обязаны
    # попасть в отчёт и в блок вердикта — иначе после закрытия терминала
    # отключённая защита исчезает из артефакта.
    data_warnings = tuple(
        w for w in (*data.data_warnings, causality_warning) if w)

    # Судьба вердикта решается до бэктеста: журнал нужен для n_trials, а
    # непригодный журнал занижает n_trials и тем завышает DSR. Недодефлиро-
    # ванный вердикт выглядит ЛУЧШЕ правды и толкает к ложному «жива» —
    # поэтому плохой журнал останавливает прогон, а не молча льстит ему.
    journal = Path(args.journal)
    trial_keys = [
        {"strategy": e.strategy, "params": e.params, "symbol": symbol,
         "timeframe": e.timeframe}
        for _, e in configs
    ]
    if args.ignore_journal:
        n_trials = len(configs)
        print(f"Предупреждение: --ignore-journal: журнал не читается и не "
              f"пишется, n_trials = {n_trials} — защита от множественных "
              f"сравнений ОТКЛЮЧЕНА, DSR завышен, вердикт «жива» может быть "
              f"ложным",
              file=sys.stderr)
    else:
        problem = journal_problem(journal)
        if problem is not None:
            print(f"Ошибка: журнал попыток непригоден: {journal} ({problem}). "
                  f"Прогон остановлен: без журнала n_trials занижается, DSR "
                  f"завышается, и вердикт выглядит лучше правды. Отчёт не "
                  f"записан. Осознанный отказ от защиты — --ignore-journal",
                  file=sys.stderr)
            return EXIT_ERROR
        # n_trials свипа — суммарное число попыток всей семьи: прошлые записи
        # журнала, относящиеся к текущему поиску (включая семейные метки
        # прошлых свипов), плюс текущий прогон каждой конфигурации. Одна
        # попытка на манифест занизила бы DSR ровно там, где штраф нужен:
        # каждая конфигурация — проверенная гипотеза, даже если отчёт
        # показывает метрики одной (первой) из них. Подсчёт идёт набором ключей
        # с дедупликацией записей: сумма одиночных запросов раздула бы штраф,
        # когда текущий свип сам покрывает всю семью.
        n_trials = count_prior_trials_for_keys(journal, trial_keys) + len(configs)

    if args.screening:
        # Громко и до бэктеста: черновой вердикт не выносит alive, и молчаливо
        # получить «кандидата» вместо результата нельзя.
        print(f"Предупреждение: --screening: {SCREENING_WARNING}",
              file=sys.stderr)

    runs: list[dict] = []
    for i, (path, cfg) in enumerate(configs):
        try:
            outcome = run_config(data, cfg,
                                 skip_causality=args.skip_causality)
        except CausalityError as exc:
            where = (f" в конфигурации [{i}] {cfg.name} ({path})"
                     if sweep_mode else "")
            print(
                f"Ошибка: стратегия '{exc.strategy_name}' не прошла "
                f"проверку причинности{where}: {exc.cause} "
                f"Прогон остановлен до бэктеста: look-ahead статистикой по "
                f"результатам не ловится (spec 8.1), поэтому без harness "
                f"вердикт был бы оптимистичной ложью. Отчёт не записан. "
                f"Осознанный отказ от проверки — --skip-causality",
                file=sys.stderr,
            )
            return EXIT_ERROR
        runs.append({"path": path, "config": cfg, "outcome": outcome})

    # Матрица PBO — доходности БАРОВ (result.returns), не сделок: pbo_cscv
    # ожидает периодические доходности T × N. Конфигурации шли по одним и тем
    # же барам, поэтому индекс общий; build_returns_matrix проверяет это и
    # падает громко при любом расхождении, а не выравнивает молча.
    matrix = None
    pbo_value = None
    if sweep_mode:
        try:
            matrix = build_returns_matrix({
                f"[{i}] {run['config'].name}":
                    run["outcome"].result.returns.set_axis(data.ts_index)
                for i, run in enumerate(runs)
            })
        except ValueError as exc:
            # Несогласованная матрица — не повод для трейсбека и не повод
            # «посчитать как получится»: PBO по ней был бы правдоподобной
            # ложью, поэтому прогон останавливается без отчёта.
            print(f"Ошибка матрицы PBO: {exc} Отчёт не записан.",
                  file=sys.stderr)
            return EXIT_ERROR
        # CSCV детерминирован, а матрица у всего свипа одна: один вызов даёт
        # тот же float, что N вызовов, но без N-кратной платы. Импорт локальный
        # (как и в validate): тесты подменяют significance.pbo_cscv.
        from alpha_lab.validation.significance import pbo_cscv
        pbo_value = pbo_cscv(matrix)

    # n_trials у всех конфигураций один — суммарное число попыток семьи: свип
    # целиком был поиском, и каждая его строка обязана нести тот же штраф DSR,
    # что и заголовочный вердикт. Валидатор зовётся общей с пакетным свипом
    # функцией validate_config: она передаёт позиции ДВИЖКА (result.positions —
    # удержанные, held[t] = target[t-1]), а не сырые цели strategy.generate().
    # Сырые цели смещены на бар, поэтому permutation-тест сравнил бы сигнал не
    # с той доходностью: измерено 0.343656 у сырых целей против 0.000999 у
    # позиций движка — прогон убивался бы за мнимый дефект, а не за реальный.
    verdicts = [
        validate_config(run["outcome"], data, experiment_id=exp_ids[i],
                        n_trials=n_trials, returns_matrix=matrix,
                        pbo_value=pbo_value, warnings=data_warnings,
                        screening=args.screening)
        for i, run in enumerate(runs)
    ]
    run0 = runs[0]
    result0 = run0["outcome"].result
    verdict = verdicts[0]

    extra = {
        "symbol": symbol, "timeframe": exp.timeframe,
        "data_version": dv, "dirty_bars": data.dirty,
        # P5: грейд прогона виден в extra независимо от вердикта: отчёт,
        # снятый черновым прогоном, обязан быть отличим от полного.
        "screening": bool(verdict.screening),
        "n_permutations": int(verdict.n_permutations),
        "gaps": data.gaps, "missing_bars": data.missing_bars,
        "gap_masked_bars": data.gap_masked,
        "history_bars": run0["outcome"].history,
        "periods_per_year": data.ppy,
        # Провенанс причинности: checked=False — защита отключена флагом;
        # cuts — сколько точек усечения реально оценено (выборочная гарантия
        # harness, а не «все позиции»). Для свипа — по заголовочной
        # конфигурации; прогон каждой описан в extra.sweep.
        "causality_checked": causality_checked,
        "causality_cuts": run0["outcome"].causality_cuts,
        "funding_available": data.funding_available,
        "funding_events": data.funding_events,
        "funding_matched": data.funding_matched,
        "costs_total": result0.cost_totals,
        "capacity": {
            "cap_hits": result0.cap_hits,
            "over_capacity": result0.over_capacity,
            "max_participation_observed": result0.max_participation_observed,
            "max_participation_limit": MAX_PARTICIPATION,
        },
    }
    if sweep_mode:
        # Свип не выбрасывается: пользователь, перебиравший поле, видит все
        # конфигурации с их id, параметрами, DSR и p-value, а не только
        # победителя. Заголовочная — первая строка манифеста (объявлена
        # заранее); правило записано в отчёт, чтобы «чьи это метрики» не
        # приходилось угадывать.
        extra["sweep"] = {
            "rule": ("Заголовочная конфигурация — первая в манифесте: её "
                     "метрики и вердикт лежат в verdict. Выбор лучшей по "
                     "Sharpe был бы ещё одним отбором на тех же данных."),
            "manifest": str(manifest_path),
            "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
            "headline": {
                "index": 0, "name": exp.name, "path": str(configs[0][0]),
                "experiment_id": exp_ids[0],
            },
            "configs": [
                {
                    "index": i, "name": runs[i]["config"].name,
                    "path": str(runs[i]["path"]),
                    "params": runs[i]["config"].params,
                    "experiment_id": exp_ids[i],
                    "sharpe": verdicts[i].sharpe, "dsr": verdicts[i].dsr,
                    "p_value": verdicts[i].p_value, "pbo": verdicts[i].pbo,
                    "trades": verdicts[i].trades, "alive": verdicts[i].alive,
                    "reasons": list(verdicts[i].reasons),
                }
                for i in range(len(runs))
            ],
        }

    payload = build_report(
        verdict, equity=result0.equity.set_axis(data.ts_index),
        close=bars["close"].set_axis(data.ts_index),
        positions=result0.positions.set_axis(data.ts_index),
        costs=result0.costs.set_axis(data.ts_index), price_bars=bars.set_axis(data.ts_index),
        extra=extra,
    )
    out = Path(args.out) if args.out else Path("reports") / exp_id
    json_path, _ = write_report(payload, out)

    # Журнал — только после успешного отчёта: если запись отчёта упала,
    # незавершённый прогон не должен остаться в журнале и завысить n_trials
    # следующего запуска (это молча усилило бы штраф DSR). При
    # --ignore-journal запись не ведётся вовсе — отказ от защиты явный.
    # Каждая конфигурация свипа пишется отдельной записью: следующая попытка
    # любой из них увидит прошлые попытки и недодефлирует DSR. Свип вдобавок
    # помечает свои записи family-меткой всего манифеста, чтобы одиночный
    # перезапуск участника унаследовал штраф всей семьи, а не одной строки.
    # Одиночный прогон метку не пишет: его запись обязана читаться как раньше.
    if not args.ignore_journal:
        family = (sorted(trial_fingerprint(key) for key in trial_keys)
                  if sweep_mode else None)
        for i, key in enumerate(trial_keys):
            log_trial(journal, key, exp_ids[i],
                      {"sharpe": verdicts[i].sharpe, "dsr": verdicts[i].dsr,
                       "alive": verdicts[i].alive}, family=family)

    # Три исхода, а не два: черновое прохождение — кандидатура, а не «жива» и
    # не «мертва». Слить кандидата с МЕРТВА значило бы выдать непроверенное за
    # отвергнутое, а с ЖИВА — выдать черновое за результат.
    if verdict.alive:
        status = "ЖИВА"
    elif verdict.screening and not verdict.reasons:
        status = "КАНДИДАТ (черновой вердикт)"
    else:
        status = "МЕРТВА"
    print(f"\n{'=' * 62}")
    print(f"  ВЕРДИКТ: {status}")
    if verdict.screening:
        print(f"  Грейд: черновой (screening, {verdict.n_permutations} "
              f"перестановок) — alive не выносится, нужен полный прогон")
    print(f"{'=' * 62}")
    print(f"  Стратегия      {exp.name}  ({exp.strategy}, {exp.timeframe})")
    # Годовой множитель печатается явно: именно его неверное значение
    # (например, часовое на 1d-эксперименте) незаметно портит метрики.
    print(f"  Баров в году   {data.ppy}  (таймфрейм {exp.timeframe})")
    print(f"  Символ         {symbol}")
    print(f"  Experiment ID  {exp_id}")
    if sweep_mode:
        print(f"  Конфигураций   {len(configs)} (манифест {manifest_path})")
        print(f"  Заголовочная   [0] {exp.name} — первая конфигурация манифеста")
    print(f"  Попыток (DSR)  {n_trials}")
    print(f"  Сделок         {verdict.trades}")
    print(f"  Sharpe         {verdict.sharpe:.2f}")
    print(f"  DSR            {verdict.dsr:.4f}  (порог 0.95)")
    print(f"  p-value        {verdict.p_value:.4f}  (порог 0.05)")
    if sweep_mode:
        # PBO — свойство матрицы, общей для всего свипа; печатается явно,
        # чтобы четвёртый гейт не приходилось искать в причинах.
        print(f"  PBO            {verdict.pbo:.4f}  "
              f"(порог {exp.validation.get('max_pbo', 0.5)})")
        print(f"  PBO-матрица    {matrix.shape[0]}×{matrix.shape[1]} (T×N)")
    print(f"  Max DD         {verdict.max_dd:.1%}")
    print(f"  Доходность     {verdict.total_return:+.2%}")
    print(f"  Издержки       {result0.cost_totals}")
    # Статус funding печатается всегда: «нет данных» и «ставка была нулевой» —
    # разные вещи, и по одной сумме издержек их не различить.
    funding_state = "доступен" if data.funding_available else "НЕДОСТУПЕН"
    print(f"  Funding        {funding_state} (событий {data.funding_events}, "
          f"привязано к барам {data.funding_matched})")
    if not data.funding_available:
        print("                 издержки занижены, вердикт оптимистичен")
    print(f"  Ёмкость        cap_hits={result0.cap_hits}, "
          f"over_capacity={'ДА' if result0.over_capacity else 'нет'}, "
          f"max_participation={result0.max_participation_observed:.4%}  "
          f"(порог {MAX_PARTICIPATION:.1%})")
    if result0.over_capacity:
        # Диагностика ликвидности, а не детектор look-ahead: флаг привязан к
        # выбранному капиталу. Поэтому предупреждаем, но не блокируем вердикт —
        # от look-ahead защищает причинностная проверка выше (alpha_lab.causality).
        print("  !!! ПРЕДУПРЕЖДЕНИЕ: заявки превышают лимит участия в объёме")
        print("      бара — прогон оптимистичен, ёмкость не доказана.")
    if verdict.warnings:
        # Отдельный от «Причин» канал: эти пункты не убили вердикт, но и не
        # пройдены. Печатаются до причин, чтобы читатель не остановился на
        # «reasons пуст — значит всё проверено».
        print("\n  Предупреждения (на статус не влияют, но гейт не проверен):")
        for warning in verdict.warnings:
            print(f"    ! {warning}")
    if verdict.reasons:
        print("\n  Причины:")
        for reason in verdict.reasons:
            print(f"    · {reason}")
    if sweep_mode:
        # Поле свипа: видно не только победителя, но и весь перебор — иначе
        # пользователь не может судить, был ли выбор осмысленным.
        print(f"\n  Свип: {len(runs)} конфигураций, PBO по матрице "
              f"{matrix.shape[0]}×{matrix.shape[1]} (T×N)")
        print(f"  {'#':>2}  {'конфигурация':<16} {'Experiment ID':<16} "
              f"{'параметры':<26} {'Sharpe':>7} {'DSR':>7} {'p-value':>8} "
              f"{'сделок':>7}  статус")
        for i, (v_i, run_i) in enumerate(zip(verdicts, runs)):
            row_status = "ЖИВА" if v_i.alive else "МЕРТВА"
            print(f"  {i:>2}  {run_i['config'].name:<16} {exp_ids[i]:<16} "
                  f"{_params_summary(run_i['config'].params):<26} "
                  f"{v_i.sharpe:>7.2f} {v_i.dsr:>7.4f} {v_i.p_value:>8.4f} "
                  f"{v_i.trades:>7}  {row_status}")
    print(f"\n  Отчёт: {json_path}")
    print("  Дашборд: откройте dashboard/index.html\n")
    return EXIT_OK


def main(argv: list[str] | None = None) -> int:
    _configure_stdio()
    parser = argparse.ArgumentParser(
        prog="alpha-lab", description="Полигон для исследования крипторынков")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ing = sub.add_parser("ingest", help="Скачать историю с Binance")
    p_ing.add_argument("--config", required=True, help="Путь к universe.yaml")
    p_ing.add_argument("--data-root", default=str(DEFAULT_ROOT))
    p_ing.add_argument("--freq", default="1m")
    p_ing.set_defaults(func=_cmd_ingest)

    p_val = sub.add_parser("validate", help="Прогнать стратегию и вынести вердикт")
    p_val.add_argument("--config", default=None,
                       help="Путь к эксперименту (одиночный прогон). "
                            "Взаимоисключающ с --configs: PBO по одному "
                            "прогону невычислим")
    p_val.add_argument("--configs", default=None,
                       help="Манифест свипа: текстовый файл, по одному пути к "
                            "эксперименту на строку (# — комментарий, "
                            "относительные пути — от каталога манифеста). "
                            "Нужно ≥ 2 конфигурации на одном символе, периоде "
                            "и таймфрейме; первая — заголовочная. Даёт матрицу "
                            "доходностей для гейта PBO (spec 6.5)")
    p_val.add_argument("--universe", default="configs/universe.yaml",
                       help="Юниверс исследования: --symbol обязан входить в "
                            "его состав, а состав входит в experiment_id "
                            "(spec 5). Отсутствующий или битый файл — ошибка")
    p_val.add_argument("--data-root", default=str(DEFAULT_ROOT))
    p_val.add_argument("--symbol", default="BTCUSDT")
    p_val.add_argument("--out", default=None)
    p_val.add_argument("--journal", default=str(DEFAULT_JOURNAL),
                       help="Журнал экспериментов: из него берётся число "
                            "попыток для DSR")
    p_val.add_argument("--ignore-journal", action="store_true",
                       help="Сознательно не читать и не писать журнал: "
                            "n_trials = 1, защита от множественных сравнений "
                            "отключается")
    p_val.add_argument("--skip-causality", action="store_true",
                       help="Сознательно отключить причинностную проверку "
                            "стратегии. Это ЕДИНСТВЕННАЯ работающая защита от "
                            "look-ahead (spec 8.1): статистика его не ловит, "
                            "поэтому вердикт с этим флагом может быть ложным")
    p_val.add_argument("--screening", action="store_true",
                       help="Черновой прогон (P5): 200 перестановок вместо "
                            "1000, минимальный достижимый p-value 1/201. Alive "
                            "по черновому вердикту НЕ выносится: прохождение "
                            "гейтов — кандидатура, финалисту нужен полный "
                            "прогон без этого флага")
    p_val.set_defaults(func=_cmd_validate)

    # Пакетный свип и воронка регистрируются лениво: batch.py берёт
    # production-функции этого модуля, и импорт на верхнем уровне создал бы цикл.
    from alpha_lab.batch import add_sweep_subparser
    add_sweep_subparser(sub)
    from alpha_lab.funnel import add_report_subparser
    add_report_subparser(sub)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
