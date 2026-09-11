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
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.config import load_experiment, load_universe
from alpha_lab.data.quality import FREQ_DELTA, clean_mask, periods_per_year
from alpha_lab.data.query import (
    align_funding_to_bars, data_version, load_bars, load_funding,
    matched_funding_events,
)
from alpha_lab.data.store import DEFAULT_ROOT
from alpha_lab.engine.backtest import run_backtest, trade_returns
from alpha_lab.engine.costs import RealisticCost
from alpha_lab.report.writer import build_report, write_report
from alpha_lab.strategies.base import (
    DEFAULT_HISTORY_BARS, build_strategy, history_bars_of,
)
from alpha_lab.validation.validator import validate

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_ERROR = 2

DEFAULT_JOURNAL = Path("reports") / "trials.jsonl"

# Доля объёма бара, выше которой заявка считается неисполнимой по смоделированной
# цене. Явно передаётся в движок, чтобы диагностика и её печать не разъехались
# при смене дефолта run_backtest.
MAX_PARTICIPATION = 0.01

# Запас маскирования разрыва назад, в барах. Назад нужно ровно на один бар:
# позиция, удерживаемая через дыру, была решена за бар до неё, а решение на
# последнем баре перед дырой принималось, когда о разрыве ещё не было
# известно, — такие позиции обязаны быть закрыты. Память стратегии назад здесь
# ни при чём: все бары до дыры контигуозны и решение на них достоверно.
#
# Вперёд запас — это history_bars стратегии (см. strategies.base): первые
# history_bars − 1 решений после дыры опираются на окно, пересекающее пропуск.
# Дефолт для необъявившей стратегии — DEFAULT_HISTORY_BARS, консервативный.
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


def count_prior_trials(journal: Path, key: dict) -> int:
    """Сколько раз эта же гипотеза уже прогонялась.

    Число попыток нужно Deflated Sharpe: без него главная защита от оверфиттинга
    отключается ровно там, где она нужна. Наивная «одна попытка» — самообман.
    """
    journal = Path(journal)
    if not journal.exists():
        return 0
    fingerprint = json.dumps(key, sort_keys=True, ensure_ascii=False)
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
            if json.dumps(record["key"], sort_keys=True,
                          ensure_ascii=False) == fingerprint:
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


def log_trial(journal: Path, key: dict, experiment: str, metrics: dict) -> None:
    journal = Path(journal)
    journal.parent.mkdir(parents=True, exist_ok=True)
    record = {"key": key, "experiment_id": experiment, **metrics}
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
              lookahead: int = DEFAULT_HISTORY_BARS) -> np.ndarray:
    """Маска баров, которые нельзя торговать из-за разрыва (True = исключён).

    Разрыв — интервал между соседними барами больше шага таймфрейма. Для
    разрыва перед баром i обнуляются цели на [i-1-lookback, i-1+lookahead]:

    * назад — позиция, удерживаемая через дыру, решена раньше, и решение на
      последнем баре перед дырой принималось, когда о разрыве ещё не было
      известно; такие позиции обязаны быть закрыты, а не пройти сквозь дыру;
    * вперёд — первые решения после дыры опираются на скользящие окна,
      пересекающие пропуск, поэтому их сигнал недостоверен, пока не наберётся
      history_bars контигуозных баров. Вызывающий обязан передать требование
      истории своей стратегии (strategies.base.history_bars_of); дефолт
      консервативен (DEFAULT_HISTORY_BARS), а не 1: единица маскировала бы
      один бар и оставляла остальные решения на неконтигуозных данных.

    Разрыв не инвалидирует весь прогон: 120 пропущенных часов на четырёх годах
    сделали бы пару непригодной, тогда как честный ответ — не торговать короткое
    окно вокруг дыры и продолжить после него. Границы клипуются по краям ряда.
    """
    n = len(bars)
    mask = np.zeros(n, dtype=bool)
    for i in _gap_starts(bars, timeframe):   # i — первый бар после разрыва
        lo = max(0, i - 1 - lookback)
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
        # Пороги валидации — часть гипотезы: без них два прогона с разными
        # min_trades делят id, отчёт второго молча затирает первый, а журнал
        # считает их одной попыткой.
        "validation": exp.validation,
        # Символ — тоже часть гипотезы (trial_key в журнале уже включает его):
        # без него прогоны того же конфига по разным символам делят id, и
        # второй молча затирает reports/<exp_id> первого.
        "symbol": args.symbol,
    }
    exp_id = experiment_id(cfg_payload, dv, git_hash())

    # Годовой множитель метрик — из таймфрейма эксперимента, а не часовой
    # константы: на 1d часы завышают Sharpe в sqrt(24) раз, Calmar — в 24.
    # load_experiment уже отверг неизвестный таймфрейм; здесь он дал бы
    # ValueError, что тоже громко, а не молчаливо.
    ppy = periods_per_year(exp.timeframe)

    symbol = args.symbol
    try:
        bars = load_bars(root, symbol, "1m", exp.start, exp.end,
                         resample=exp.timeframe)
    except FileNotFoundError as exc:
        print(f"Данных нет: {exc}", file=sys.stderr)
        return EXIT_ERROR

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
            funding_rate = align_funding_to_bars(bars, funding, exp.timeframe)
            funding_matched = matched_funding_events(bars, funding, exp.timeframe)
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

    # Разрывы: через дыру цена шла неизвестно как, поэтому торговля вокруг неё
    # приостанавливается (spec 8: грязные данные → стратегия не торгует).
    # Маска компонуется с маской грязных баров; движок не меняется.
    # Стратегия строится здесь, до маски: lookahead маски — это объявленное
    # стратегией требование истории, и только она знает его точно.
    strategy = build_strategy(exp.strategy, exp.params)
    history = history_bars_of(strategy)
    gaps, missing_bars = _gap_stats(bars, exp.timeframe)
    gap_excluded = _gap_mask(bars, exp.timeframe, lookahead=history)
    gap_masked = int(gap_excluded.sum())
    gap_warning = None
    if gaps:
        gap_warning = (
            f"В данных разрывов: {gaps}, пропущено баров: {missing_bars} "
            f"(таймфрейм {exp.timeframe}) — торговля приостановлена на "
            f"{gap_masked} барах вокруг них (lookback="
            f"{GAP_MASK_LOOKBACK_BARS}, lookahead={history} = "
            f"history_bars стратегии '{exp.strategy}'); состояние стратегии "
            f"перезапускается на каждом непрерывном участке"
        )
        print(f"Предупреждение: {gap_warning}", file=sys.stderr)

    tradable = clean & ~gap_excluded

    # Предупреждения о данных уходят тем же каналом, что и неоценённый PBO:
    # они не делают вердикт мёртвым (нет данных — не дефект стратегии), но
    # обязаны попасть в отчёт и в блок вердикта.
    data_warnings = tuple(w for w in (funding_warning, gap_warning) if w)

    # Судьба вердикта решается до бэктеста: журнал нужен для n_trials, а
    # непригодный журнал занижает n_trials и тем завышает DSR. Недодефлиро-
    # ванный вердикт выглядит ЛУЧШЕ правды и толкает к ложному «жива» —
    # поэтому плохой журнал останавливает прогон, а не молча льстит ему.
    journal = Path(args.journal)
    trial_key = {"strategy": exp.strategy, "params": exp.params,
                 "symbol": symbol, "timeframe": exp.timeframe}
    if args.ignore_journal:
        n_trials = 1
        print("Предупреждение: --ignore-journal: журнал не читается и не "
              "пишется, n_trials = 1 — защита от множественных сравнений "
              "ОТКЛЮЧЕНА, DSR завышен, вердикт «жива» может быть ложным",
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
        n_trials = count_prior_trials(journal, trial_key) + 1

    # Разрыв — не только «не торговать в окне»: ряд неконтигуозен, и стратегия
    # не имеет права видеть сквозь дыру. generate вызывается на каждом
    # непрерывном участке отдельно, поэтому внутреннее состояние стратегии
    # (у MR — direction/entry_bar/cur_sl/cur_tp внутри simulate_bracket_exits)
    # перезапускается на дыре по построению. Обнуление целей этого не лечит:
    # сделка, открытая до дыры, всплыла бы на первом немаскированном баре с
    # предразрывной ценой входа, стопом/тейком и часами max_bars, которые не
    # считали пропущенные бары. Маска ниже по-прежнему гасит окно вокруг дыры
    # (в том числе бар перед ней — позицию, удержанную через пропуск).
    #
    # copy=True: to_numpy() в pandas 3 отдаёт read-only массив, а маска ниже
    # пишет в него на месте. Обнуляются цели и грязных баров, и окна разрывов.
    targets = _generate_segmented(strategy, bars, exp.timeframe).to_numpy(
        dtype="float64", copy=True)
    targets[~tradable] = 0.0
    targets = pd.Series(targets, index=bars.index)

    cost_model = RealisticCost.from_config(exp.costs)
    result = run_backtest(bars, targets, cost_model, funding_rate=funding_rate,
                          max_participation=MAX_PARTICIPATION)
    trades = trade_returns(result)

    # ВАЖНО: в валидатор уходят позиции ДВИЖКА (result.positions — удержанные,
    # held[t] = target[t-1]), а не сырые цели strategy.generate(). Сырые цели
    # смещены на бар, поэтому permutation-тест сравнил бы сигнал не с той
    # доходностью: измерено 0.343656 у сырых целей против 0.000999 у позиций
    # движка — прогон убивался бы за мнимый дефект, а не за реальный.
    verdict = validate(
        returns=result.returns, trade_returns=trades, equity=result.equity,
        config=exp.validation, n_trials=n_trials, strategy_name=exp.name,
        experiment_id=exp_id,
        price_returns=result.price_returns, positions=result.positions,
        warnings=data_warnings, periods_per_year=ppy,
    )

    # ВАЖНО: load_bars заканчивается reset_index(drop=True), поэтому бары и
    # выходы движка проиндексированы RangeIndex (0..n-1). build_report делает
    # из индекса series.ts через pd.Timestamp(t), а это наносекунды от эпохи
    # 1970 года — ось времени всех графиков дашборда становится бессмысленной.
    # Переиндексируем позиционные ряды реальными временами баров; set_axis не
    # меняет длину, поэтому проверка выравнивания в writer проходит.
    ts_index = pd.Index(bars["ts"].to_numpy(), name="ts")
    payload = build_report(
        verdict, equity=result.equity.set_axis(ts_index),
        close=bars["close"].set_axis(ts_index),
        positions=result.positions.set_axis(ts_index),
        costs=result.costs.set_axis(ts_index), price_bars=bars.set_axis(ts_index),
        extra={"symbol": symbol, "timeframe": exp.timeframe,
               "data_version": dv, "dirty_bars": dirty,
               "gaps": gaps, "missing_bars": missing_bars,
               "gap_masked_bars": gap_masked,
               "history_bars": history, "periods_per_year": ppy,
               "funding_available": funding_available,
               "funding_events": funding_events,
               "funding_matched": funding_matched,
               "costs_total": result.cost_totals,
               "capacity": {
                   "cap_hits": result.cap_hits,
                   "over_capacity": result.over_capacity,
                   "max_participation_observed": result.max_participation_observed,
                   "max_participation_limit": MAX_PARTICIPATION,
               }},
    )
    out = Path(args.out) if args.out else Path("reports") / exp_id
    json_path, _ = write_report(payload, out)

    # Журнал — только после успешного отчёта: если запись отчёта упала,
    # незавершённый прогон не должен остаться в журнале и завысить n_trials
    # следующего запуска (это молча усилило бы штраф DSR). При
    # --ignore-journal запись не ведётся вовсе — отказ от защиты явный.
    if not args.ignore_journal:
        log_trial(journal, trial_key, exp_id,
                  {"sharpe": verdict.sharpe, "dsr": verdict.dsr,
                   "alive": verdict.alive})

    status = "ЖИВА" if verdict.alive else "МЕРТВА"
    print(f"\n{'=' * 62}")
    print(f"  ВЕРДИКТ: {status}")
    print(f"{'=' * 62}")
    print(f"  Стратегия      {exp.name}  ({exp.strategy}, {exp.timeframe})")
    # Годовой множитель печатается явно: именно его неверное значение
    # (например, часовое на 1d-эксперименте) незаметно портит метрики.
    print(f"  Баров в году   {ppy}  (таймфрейм {exp.timeframe})")
    print(f"  Символ         {symbol}")
    print(f"  Experiment ID  {exp_id}")
    print(f"  Попыток (DSR)  {n_trials}")
    print(f"  Сделок         {verdict.trades}")
    print(f"  Sharpe         {verdict.sharpe:.2f}")
    print(f"  DSR            {verdict.dsr:.4f}  (порог 0.95)")
    print(f"  p-value        {verdict.p_value:.4f}  (порог 0.05)")
    print(f"  Max DD         {verdict.max_dd:.1%}")
    print(f"  Доходность     {verdict.total_return:+.2%}")
    print(f"  Издержки       {result.cost_totals}")
    # Статус funding печатается всегда: «нет данных» и «ставка была нулевой» —
    # разные вещи, и по одной сумме издержек их не различить.
    funding_state = "доступен" if funding_available else "НЕДОСТУПЕН"
    print(f"  Funding        {funding_state} (событий {funding_events}, "
          f"привязано к барам {funding_matched})")
    if not funding_available:
        print("                 издержки занижены, вердикт оптимистичен")
    print(f"  Ёмкость        cap_hits={result.cap_hits}, "
          f"over_capacity={'ДА' if result.over_capacity else 'нет'}, "
          f"max_participation={result.max_participation_observed:.4%}  "
          f"(порог {MAX_PARTICIPATION:.1%})")
    if result.over_capacity:
        # Диагностика ликвидности, а не детектор look-ahead: флаг привязан к
        # выбранному капиталу. Поэтому предупреждаем, но не блокируем вердикт —
        # от look-ahead защищает причинностный harness (тесты).
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
    p_val.add_argument("--config", required=True, help="Путь к эксперименту")
    p_val.add_argument("--universe", default="configs/universe.yaml")
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
    p_val.set_defaults(func=_cmd_validate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
