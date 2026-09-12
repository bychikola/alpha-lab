"""E2: кросс-секционный carry-портфель — премию собирает каждый символ, чей
собственный режим ставки её окупает; единица измерения — портфель.

ПРОТОКОЛ ЗАФИКСИРОВАН ДО ПЕРВОГО ПРОГОНА. Правило, юниверс, разбиение по
времени и модель капитала объявлены ниже; по результатам не менялись и не
подбирались. Ни один параметр не подбирался ни на обучении, ни вне выборки.

1. АПРИОРНОЕ ПРАВИЛО (то же правило S2, применённое НЕЗАВИСИМО к каждому
   символу; никакого ранжирования и никакого отбора символов):

       mu_i(t) — среднее ставки funding символа i за window=72 бара (3 суток
       на 1h), включая бар t (rolling назад, только прошлое и текущий бар);
       theta = 4*(fee_bps + slippage_bps)*1e-4 / horizon_bars = 3.3333e-6 на
       бар (fee=5 bps, slippage=1 bp, H=720 — дефолты S2, объявленные априори
       в её docstring);
       плечо i держит книгу (carry_i = −w, шорт перпа против спота), пока
       mu_i(t) > theta; иначе carry_i = 0. net_i ≡ 0 всегда.

   Порог theta — требуемая ставка НА БАР для окупаемости круглого рейса
   (4 единицы оборота за (H·mu) ставок); он не зависит от размера плеча w,
   потому что funding и издержки масштабируются им одинаково. Ставка
   отрицательна или не окупает издержки — книга символа закрыта; «держать
   всегда» (threshold_rate=−inf) в основной замер не входит (это отдельная
   справочная панель, см. п. 8).

2. ФИКСИРОВАННЫЙ РАЗМЕР ПЛЕЧА (ограничение движка S1: без ребалансировок и
   разворотов). У каждого символа собственный независимый счёт плеча размера
   C/N; книга плеча имеет фиксированный размер w·(C/N) на ногу (в единицах
   счёта плеча carry = −w, gross = 2w). Целевые значения принимают только
   0 и −w, поэтому вход/выход символа — это ровно 0 ↔ 2w по gross, что
   magnitude-базис движка ценит точно. Размер уже открытой книги не меняется
   никогда, и вход/выход одного символа не перекладывает капитал других:
   это и есть причина, по которой ноционал фиксирован, а не равен «1/N
   капитала, пересчитанному на число открытых книг». Встречное движение ног и
   прямой разворот carry, которых |Δgross| не видит, правило не создаёт.

3. МОДЕЛЬ КАПИТАЛА (что такое w и на что делится доходность). Капитал C
   делится на N равных плеч по C/N; перераспределения капитала между плечами
   нет. Книга плеча: спот-нога p·(C/N) (покупается на кэш) и перп-нога
   p·(C/N) (шорт; резервируется начальная маржа (p·(C/N))/L при плече перпа
   L). Размер p выбран так, что p·(1 + 1/L) = 1: капитала плеча хватает ровно
   на обе ноги при полной занятости. Тогда доходность портфеля — доход на ВЕСЬ
   капитал C, а не на ноционал книги. Основной замер — L=1 (перп без плеча:
   p = 1/2, консервативно, ликвидационный риск минимален), чувствительность —
   L=2 (p = 2/3) и L=5 (p = 5/6); «наивный» p = 1 (капитал = только спот-нога,
   маржа не зарезервирована) приведён верхней границей. Рост L повышает
   доходность на капитал и одновременно приближает ликвидацию перп-ноги —
   это слепая зона вердикта (spec 8.2), а не измеренная величина.

   Масштаб: счёт плеча — 10 000 (дефолт run_config), то есть при 20 символах
   портфель 200 000; проскальзывание зависит от абсолютного размера заявки,
   поэтому числа привязаны к этому масштабу, а не инвариантны к нему.

4. ПОРТФЕЛЬ — ЕДИНИЦА ИЗМЕРЕНИЯ. Доходность портфеля = равновесное среднее
   доходностей плеч: Σ P&L_i / (N·C/N). Издержки и funding складываются так же
   (среднее по плечам). Отчётные числа — портфельные; таблица по символам
   приводится только как диагностика состава, а не как результат.

5. РАЗБИЕНИЕ ПО ВРЕМЕНИ. 2022–2023 — обучение, 2024–2025 — вне выборки.
   Правило одно на весь период и не подбиралось ни на одной половине, поэтому
   вторая половина — честная проверка вне выборки. Ряды полупериодов — срезы
   одного непрерывного прогона (независимые перезапуски сдвинули бы книги,
   открытые через границу); пер-баровые доходности, издержки и funding при
   срезе не меняются.

6. ЮНИВЕРС — машинное правило (как в S3): символ входит, если в озере есть
   все 48 месячных партиций 1m-баров за 2022-01..2025-12 и funding-события в
   каждом из этих 48 месяцев. Список строится сканированием озера, а не задан
   руками; исключённые символы и причины печатаются. Ingest продолжается:
   срез смещён в сторону выживших и крупнейших пар, делистингованных в озере
   по-прежнему нет — это отдельная слепая зона, фиксируемая рядом с числами.

7. ЧЕСТНЫЕ ИЗДЕРЖКИ. Издержки — как в funding_base: тейкер 5 bps + модель
   воздействия по объёму бара (floor 0.5 bps). Вход и выход — по две ноги на
   символ (оборот 2w каждая), поэтому портфель платит за каждую смену состава.
   Funding и издержки печатаются раздельно, число входов/выходов — по факту
   удержанных рядов движка. Книги, открытые на конец периода, принудительно не
   закрываются (движок не знает будущего) — их число печатается явно.

8. СПРАВОЧНАЯ ПАНЕЛЬ (не гипотеза). Та же портфельная конструкция с
   threshold_rate=−inf (книга держится всегда; S1) — чтобы отделить свойство
   кросс-секции от эффекта тайминга. В вердикт и в основной вывод она не
   входит.

9. ВЕРДИКТ. Портфельный ряд прогоняется через продакшн-валидатор
   (cli.validate_config) как одна гипотеза: n_trials=1 (правило объявлено
   заранее, перебора нет; для прозрачности приведён и DSR при n_trials=N —
   если считать плечи отдельными попытками). По spec 6.6 у книги с net ≡ 0
   permutation-тест и min_trades неприменимы — ожидаемый статус «НЕ ВЫНЕСЕН
   (неприменимые гейты)», если применимый DSR не провален. Слепые зоны
   (spec 8.2) едут в warnings автоматически.

Запуск (полный протокол):
    uv run python scripts/e2_cross_sectional.py
Смоук-проверка механики (не протокол, помечается в JSON):
    uv run python scripts/e2_cross_sectional.py --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from alpha_lab import cli
from alpha_lab.config import Experiment, load_experiment
from alpha_lab.data.query import data_version
from alpha_lab.data.store import DEFAULT_ROOT
from alpha_lab.engine.backtest import (
    BacktestResult, position_episodes, trade_returns,
)
from alpha_lab.strategies.base import FUNDING_RATE_COLUMN, build_strategy
from alpha_lab.validation.metrics import max_drawdown, sharpe_ratio

PROTOCOL_VERSION = "e2-2026-09-12"

TIMEFRAME = "1h"
START = "2022-01-01"
END = "2025-12-31"
TRAIN_END = "2023-12-31"          # обучение: 2022-01-01 .. 2023-12-31
OOS_START = "2024-01-01"          # вне выборки: 2024-01-01 .. 2025-12-31
PPY = 8760                        # часов в году (24/7)

# Модель капитала: p = 1/(1 + 1/L). Основной замер — L=1 (p=1/2).
PRIMARY_LEVERAGE = 1.0
LEVERAGES = (1.0, 2.0, 5.0)
NAIVE_NOTIONAL = 1.0              # маржа не зарезервирована — верхняя граница

BASE_CONFIG = Path("configs/experiments/funding_base.yaml")
TIME_COLUMNS = ("whole", "train", "oos")


def notional_for_leverage(leverage: float) -> float:
    """Доля счёта плеча в одной ноге: p = 1/(1 + 1/L)."""
    if not math.isfinite(leverage) or leverage <= 0.0:
        raise ValueError(f"leverage должен быть положительным, {leverage!r}")
    return 1.0 / (1.0 + 1.0 / leverage)


def _months() -> list[str]:
    return [f"{year}-{month:02d}" for year in (2022, 2023, 2024, 2025)
            for month in range(1, 13)]


def discover_symbols(root: Path) -> tuple[list[str], list[dict]]:
    """Символы озера с полным покрытием 2022–2025; причины исключения — рядом.

    Правило машинное и объявлено заранее: 48/48 месячных партиций 1m-баров и
    funding-события в каждом из 48 месяцев. Ручной список не используется:
    он молча пропустил бы уже лежащий в озере символ (ошибка S3, исправлена
    в её отчёте), а состав юниверса — часть гипотезы, а не оформление.
    """
    months = _months()
    included: list[str] = []
    excluded: list[dict] = []
    for symbol_dir in sorted((root / "bars").iterdir()):
        if not symbol_dir.is_dir():
            continue
        symbol = symbol_dir.name
        files = {p.name for p in (symbol_dir / "1m").glob("*.parquet")}
        missing_bars = [m for m in months
                        if f"{symbol}-1m-{m}.parquet" not in files]
        funding_path = root / "funding" / symbol / f"{symbol}-funding.parquet"
        missing_fund: list[str] = []
        if funding_path.exists():
            frame = pd.read_parquet(funding_path, columns=["ts"])
            ts = pd.to_datetime(frame["ts"], utc=True)
            have = {f"{t.year}-{t.month:02d}" for t in ts}
            missing_fund = [m for m in months if m not in have]
        else:
            missing_fund = list(months)
        if not missing_bars and not missing_fund:
            included.append(symbol)
        else:
            excluded.append({
                "symbol": symbol,
                "missing_bar_months": len(missing_bars),
                "missing_funding_months": len(missing_fund),
            })
    return included, excluded


def _verify_sleeve(symbol: str, data, outcome, notional: float,
                   params: dict) -> dict:
    """Проверки инварианта фиксированного ноционала по рядам прогона.

    Смотрит и на решение (generate_legs), и на то, что из него сделал движок:
    carry/gross принимают только 0/−w и 0/2w, ненулевой оборот равен ровно 2w
    (вход ИЛИ выход, без ребалансировок), funding начислен ровно на
    удерживаемый carry, ценовой экспозиции нет. Нарушение — AssertionError:
    портфельная конструкция без этого инварианта недействительна.
    """
    bars = data.bars
    strategy = build_strategy("funding_harvest", params)
    signal_bars = bars.assign(
        **{FUNDING_RATE_COLUMN: data.funding_rate.to_numpy()})
    legs = strategy.generate_legs(signal_bars)
    carry = legs.carry.to_numpy()
    gross_target = legs.gross.to_numpy()
    in_book = carry != 0.0
    assert not np.any(legs.net.to_numpy() != 0.0), f"{symbol}: net ≠ 0"
    assert set(np.unique(carry)) <= {0.0, -notional}, f"{symbol}: carry {np.unique(carry)}"
    np.testing.assert_allclose(gross_target, 2.0 * notional * in_book,
                               rtol=0, atol=1e-15,
                               err_msg=f"{symbol}: gross ≠ 2w·in_book")

    res = outcome.result
    held_gross = res.gross_positions.to_numpy()
    turnover = res.turnover.to_numpy()
    assert not np.any(res.positions.to_numpy() != 0.0), f"{symbol}: net-позиция ≠ 0"
    assert set(np.unique(held_gross)) <= {0.0, 2.0 * notional}, \
        f"{symbol}: удержанный gross {np.unique(held_gross)}"
    nonzero_turnover = np.unique(turnover[turnover != 0.0])
    np.testing.assert_allclose(nonzero_turnover, [2.0 * notional],
                               rtol=0, atol=1e-15,
                               err_msg=f"{symbol}: оборот не 2w")

    rate = np.nan_to_num(data.funding_rate.to_numpy(), nan=0.0)
    funding = res.costs["funding"].to_numpy()
    live = held_gross > 0.0
    np.testing.assert_allclose(funding[live], -notional * rate[live],
                               rtol=0, atol=1e-15,
                               err_msg=f"{symbol}: funding ≠ −w·rate в книге")
    assert np.allclose(funding[~live], 0.0, rtol=0, atol=0), \
        f"{symbol}: funding вне книги"

    prev = np.concatenate(([0.0], held_gross[:-1]))
    entries = int(((prev == 0.0) & (held_gross > 0.0)).sum())
    exits = int(((prev > 0.0) & (held_gross == 0.0)).sum())
    assert entries + exits == int((turnover != 0.0).sum()), \
        f"{symbol}: оборот не совпал со входами/выходами"
    open_at_end = bool(held_gross[-1] > 0.0) if len(held_gross) else False
    return {
        "symbol": symbol,
        "entries": entries,
        "exits": exits,
        "open_at_end": open_at_end,
        "bars": int(len(bars)),
        "in_book_share": float(live.mean()) if len(live) else 0.0,
        "net_return": float(res.total_return),
        "funding_income": float(-res.costs["funding"].sum()),
        "fee": float(res.costs["fee"].sum()),
        "slippage": float(res.costs["slippage"].sum()),
        "max_dd": float(res.max_drawdown),
    }


class Portfolio:
    """Портфель равновесных плеч: ряды и разложение по компонентам."""

    def __init__(self, symbols: list[str], sleeves: dict[str, dict],
                 notional: float):
        self.symbols = list(symbols)
        self.notional = float(notional)
        self._sleeves = sleeves
        # Общий индекс портфеля — времена баров (ts_index), а не позиционный
        # RangeIndex движка: ряды разных символов выравниваются по времени, а
        # не по номеру бара (у символов разное число грязных баров и разрывов).
        idx = None
        for symbol in self.symbols:
            index = pd.Index(sleeves[symbol]["data"].ts_index)
            idx = index if idx is None else idx.union(index)
        self.index = idx

        def matrix(extract) -> np.ndarray:
            columns = []
            for symbol in self.symbols:
                res = sleeves[symbol]["outcome"].result
                values = np.asarray(extract(res), dtype="float64")
                columns.append(pd.Series(
                    values, index=sleeves[symbol]["data"].ts_index,
                ).reindex(idx).to_numpy(dtype="float64"))
            return np.column_stack(columns)

        # Плечи равного бюджета C/N: доходность портфеля на весь капитал —
        # среднее по плечам (Σ P&L_i)/(N·C/N), а не сумма.
        self.returns_by_symbol = matrix(lambda res: res.returns)
        self.returns = pd.Series(
            self.returns_by_symbol.mean(axis=1), index=idx,
            name="portfolio_returns")
        self.equity = (1.0 + self.returns).cumprod()
        self.costs = pd.DataFrame({
            name: matrix(lambda res, name=name: res.costs[name]).mean(axis=1)
            for name in ("fee", "slippage", "funding")
        }, index=idx)
        self.gross_by_symbol = matrix(lambda res: res.gross_positions)
        self.gross_positions = pd.Series(
            self.gross_by_symbol.mean(axis=1), index=idx, name="gross")
        self.turnover = pd.Series(
            matrix(lambda res: res.turnover).mean(axis=1), index=idx,
            name="turnover")
        self.price_returns = pd.Series(
            matrix(lambda res: res.price_returns).mean(axis=1), index=idx,
            name="price")
        self.positions = pd.Series(
            matrix(lambda res: res.positions).mean(axis=1), index=idx,
            name="positions")
        self.gross_returns = pd.Series(
            matrix(lambda res: res.gross_returns).mean(axis=1), index=idx,
            name="gross_ret")
        self.cap_hits = int(sum(
            sleeves[s]["outcome"].result.cap_hits for s in self.symbols))
        self.max_participation_observed = float(max(
            sleeves[s]["outcome"].result.max_participation_observed
            for s in self.symbols))
        self.in_book_count = pd.Series(
            (self.gross_by_symbol > 0.0).sum(axis=1), index=idx)
        self.sleeve_stats = [sleeves[s]["stats"] for s in self.symbols]

    @property
    def any_in_book(self) -> pd.Series:
        return self.in_book_count > 0

    def result(self) -> BacktestResult:
        """Портфель как BacktestResult — вход продакшн-валидатора."""
        eq = self.equity
        return BacktestResult(
            equity=eq,
            returns=self.returns,
            gross_returns=self.gross_returns,
            positions=self.positions,
            gross_positions=self.gross_positions,
            turnover=self.turnover,
            costs=self.costs,
            total_return=float(eq.iloc[-1] - 1.0) if len(eq) else 0.0,
            max_drawdown=max_drawdown(eq) if len(eq) else 0.0,
            bars=int(len(eq)),
            price_returns=self.price_returns,
            cap_hits=self.cap_hits,
            max_participation_observed=self.max_participation_observed,
        )


def curve_metrics(returns: pd.Series) -> dict:
    """Портфельные метрики периода: доход, CAGR, волатильность, просадка."""
    if len(returns) < 2:
        return {"bars": int(len(returns)), "total_return": 0.0, "cagr": 0.0,
                "vol": 0.0, "max_dd": 0.0, "sharpe": 0.0}
    eq = (1.0 + returns).cumprod()
    ts = pd.to_datetime(pd.Index(returns.index), utc=True)
    years = (ts[-1] - ts[0]).total_seconds() / (365.25 * 24 * 3600)
    total = float(eq.iloc[-1] - 1.0)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else 0.0
    vol = float(returns.std(ddof=1) * math.sqrt(PPY))
    return {
        "bars": int(len(returns)),
        "years": float(years),
        "total_return": total,
        "cagr": cagr,
        "vol": vol,
        "max_dd": float(max_drawdown(eq)),
        "sharpe": float(sharpe_ratio(returns, PPY)),
    }


def _period_mask(index, period: str) -> np.ndarray:
    ts = pd.to_datetime(pd.Index(index), utc=True)
    if period == "train":
        return np.asarray(ts <= pd.Timestamp(TRAIN_END, tz="UTC"))
    if period == "oos":
        return np.asarray(ts >= pd.Timestamp(OOS_START, tz="UTC"))
    return np.ones(len(ts), dtype=bool)


def _entries_exits(held_gross: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Входы/выходы (0 ↔ 2w) внутри маски периода по удержанному ряду."""
    values = held_gross[mask]
    if len(values) == 0:
        return 0, 0
    prev = np.concatenate(([0.0], values[:-1]))
    entries = int(((prev == 0.0) & (values > 0.0)).sum())
    exits = int(((prev > 0.0) & (values == 0.0)).sum())
    return entries, exits


def period_stats(portfolio: Portfolio, period: str) -> dict:
    mask = _period_mask(portfolio.index, period)
    stats = curve_metrics(portfolio.returns[mask])
    funding = portfolio.costs["funding"].to_numpy()[mask]
    fee = portfolio.costs["fee"].to_numpy()[mask]
    slip = portfolio.costs["slippage"].to_numpy()[mask]
    in_book = portfolio.any_in_book.to_numpy()[mask]
    counts = portfolio.in_book_count.to_numpy()[mask]
    # Входы/выходы — по каждому плечу отдельно: смена состава портфеля
    # платится дважды за символ (две ноги на входе и две на выходе).
    entries = exits = 0
    for held in portfolio.gross_by_symbol.T:
        sleeve_entries, sleeve_exits = _entries_exits(held, mask)
        entries += sleeve_entries
        exits += sleeve_exits
    return {
        **stats,
        "funding_income": float(-funding.sum()),
        "fee": float(fee.sum()),
        "slippage": float(slip.sum()),
        "entries": int(entries),
        "exits": int(exits),
        "in_book_share": float(in_book.mean()) if len(in_book) else 0.0,
        "avg_symbols_in_book": float(counts.mean()) if len(counts) else 0.0,
    }


def run_sleeves(root: Path, symbols: list[str], base: Experiment,
                params: dict[str, float], notional: float,
                cache: dict[str, object]) -> dict:
    """Прогон всех плеч через продакшн-путь: prepare_data → run_config.

    Бары грузятся один раз на символ и переиспользуются всеми вариантами
    ноционала: prepare_data — доминирующая стоимость прогона (1m → 1h), а
    варианты отличаются только размером плеча при том же решении.
    """
    sleeve_params = {**base.params, **params, "notional": notional}
    exp = Experiment(
        name="e2_cross_sectional_sleeve", strategy="funding_harvest",
        params=sleeve_params, timeframe=TIMEFRAME, start=START, end=END,
        costs=dict(base.costs), validation=dict(base.validation),
    )
    sleeves: dict[str, dict] = {}
    for symbol in symbols:
        if symbol not in cache:
            cache[symbol] = cli.prepare_data(root, symbol, TIMEFRAME, START, END)
        data = cache[symbol]
        outcome = cli.run_config(data, exp)
        stats = _verify_sleeve(symbol, data, outcome, notional, sleeve_params)
        sleeves[symbol] = {"data": data, "outcome": outcome, "stats": stats}
    return sleeves


def build_portfolio(symbols: list[str], sleeves: dict, notional: float) -> Portfolio:
    return Portfolio(symbols, sleeves, notional)


def verdict_payload(portfolio: Portfolio, base: Experiment, params: dict,
                    n_trials: int) -> dict:
    """Продакшн-вердикт по портфельному ряду как по одной гипотезе."""
    exp = Experiment(
        name="e2_cross_sectional", strategy="funding_harvest",
        params={**base.params, **params}, timeframe=TIMEFRAME, start=START,
        end=END, costs=dict(base.costs), validation=dict(base.validation),
    )
    result = portfolio.result()
    outcome = cli.RunOutcome(
        experiment=exp, result=result, trades=trade_returns(result),
        history=int(base.params.get("window", 72)), causality_cuts=0,
    )
    data_stub = SimpleNamespace(ppy=PPY, data_warnings=())
    verdict = cli.validate_config(
        outcome, data_stub, experiment_id=f"{PROTOCOL_VERSION}-n{n_trials}",
        n_trials=n_trials,
    )
    return {
        "n_trials": n_trials,
        "status": cli.verdict_status(verdict),
        "alive": bool(verdict.alive),
        "sharpe": float(verdict.sharpe),
        "dsr": float(verdict.dsr),
        "p_value": float(verdict.p_value),
        "trades": int(verdict.trades),
        "pbo": float(verdict.pbo) if np.isfinite(verdict.pbo) else None,
        "total_return": float(verdict.total_return),
        "max_dd": float(verdict.max_dd),
        "reasons": list(verdict.reasons),
        "inapplicable": list(verdict.inapplicable),
        "warnings": list(verdict.warnings),
    }


def summarize(portfolio: Portfolio, symbols: list[str], notional: float,
              base: Experiment, params: dict, excluded: list[dict],
              wall_seconds: float) -> dict:
    periods = {name: period_stats(portfolio, name) for name in TIME_COLUMNS}
    episodes = position_episodes(portfolio.result())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "rule": {
            "window": params.get("window"), "horizon_bars": params.get("horizon_bars"),
            "fee_bps": params.get("fee_bps"), "slippage_bps": params.get("slippage_bps"),
            "threshold_per_bar": None,  # заполняется из стратегии ниже
            "notional": notional,
            "leverage": None,
        },
        "symbols": symbols,
        "excluded": excluded,
        "n_symbols": len(symbols),
        "notional": notional,
        "bars": int(len(portfolio.index)),
        "wall_seconds": wall_seconds,
        "periods": periods,
        "sleeves": portfolio.sleeve_stats,
        "portfolio_episodes": {
            "count": int(len(episodes)),
            "worst_net_return": float(episodes["net_return"].min())
            if len(episodes) else None,
            "median_bars": float(episodes["bars"].median())
            if len(episodes) else None,
        },
        "open_at_end": int(sum(
            s["open_at_end"] for s in portfolio.sleeve_stats)),
        "capacity": {
            "cap_hits": portfolio.cap_hits,
            "max_participation_observed": portfolio.max_participation_observed,
        },
    }
    strategy = build_strategy("funding_harvest", {**base.params, **params})
    payload["rule"]["threshold_per_bar"] = float(strategy.entry_threshold())
    return payload


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def print_report(payload: dict, verdicts: dict[int, dict]) -> None:
    print("=" * 74)
    print(f"  E2: кросс-секционный портфель carry  ({PROTOCOL_VERSION})")
    print("=" * 74)
    print(f"  Символов: {payload['n_symbols']}  {', '.join(payload['symbols'])}")
    print(f"  Исключены: {len(payload['excluded'])} "
          f"({', '.join(e['symbol'] for e in payload['excluded'])})")
    print(f"  Баров: {payload['bars']:,} 1h;  решений: "
          f"{payload['bars'] * payload['n_symbols']:,} (символ × бар)")
    print(f"  Ноционал (L={payload['rule']['leverage']}): "
          f"{payload['notional']:.6f} на ногу, "
          f"порог {payload['rule']['threshold_per_bar']:.3e}/бар")
    print(f"  Wall-clock: {payload['wall_seconds']:.1f} c")
    print()
    header = (f"  {'период':<8} {'лет':>5} {'net':>10} {'CAGR':>9} "
              f"{'vol':>8} {'maxDD':>9} {'Sharpe':>7} {'в книге':>8} "
              f"{'вход/вых':>9}")
    print(header)
    for name in TIME_COLUMNS:
        s = payload["periods"][name]
        print(f"  {name:<8} {s['years']:>5.2f} {s['total_return']:>+10.4%} "
              f"{s['cagr']:>+9.2%} {s['vol']:>8.2%} {s['max_dd']:>+9.2%} "
              f"{s['sharpe']:>7.2f} {s['in_book_share']:>7.1%} "
              f"{s['entries']:>4}/{s['exits']:<4}")
    print()
    print("  Funding и издержки (доли капитала, простые суммы):")
    for name in TIME_COLUMNS:
        s = payload["periods"][name]
        print(f"    {name:<8} funding {s['funding_income']:>+9.4%}  "
              f"fee {s['fee']:>8.4%}  slippage {s['slippage']:>8.4%}  "
              f"net {s['funding_income'] - s['fee'] - s['slippage']:>+9.4%}")
    print()
    print("  Портфель вне книги никогда не был: "
          f"{payload['periods']['whole']['in_book_share']:.1%} баров хотя бы "
          f"одна книга; в среднем "
          f"{payload['periods']['whole']['avg_symbols_in_book']:.2f} из "
          f"{payload['n_symbols']}.")
    print(f"  Книг, открытых на конец периода (выход не заряжен): "
          f"{payload['open_at_end']}.")
    print()
    print("  Вердикт продакшн-валидатора (портфель = одна гипотеза):")
    for n_trials, v in verdicts.items():
        pbo = "—" if v["pbo"] is None else f"{v['pbo']:.4f}"
        print(f"    n_trials={n_trials}: {v['status']}; DSR={v['dsr']:.4f}; "
              f"p={v['p_value']:.4f}; сделок={v['trades']}; PBO={pbo}")
        for text in v["inapplicable"]:
            print(f"      ~ {text[:110]}…")
        for text in v["warnings"]:
            print(f"      ! {text[:110]}…")
        for text in v["reasons"]:
            print(f"      · {text}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--base-config", default=str(BASE_CONFIG))
    parser.add_argument("--out", default="reports/e2/portfolio.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="смоук: первые N символов, помечается в JSON")
    args = parser.parse_args(argv)

    # На Windows консоль по умолчанию в cp1251/cp866, и русский текст
    # превращается в мусор; рабочий путь CLI делает то же самое (cli._configure_stdio).
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    smoke = args.limit is not None
    started = time.perf_counter()
    root = Path(args.root)
    base = load_experiment(args.base_config)
    symbols, excluded = discover_symbols(root)
    if args.limit is not None:
        symbols = symbols[:args.limit]
    if not symbols:
        print("Ошибка: в озере нет символов с полным покрытием 2022–2025")
        return 2
    print(f"Озеро {root}: {len(symbols)} символов с полным покрытием "
          f"2022-01..2025-12; исключено {len(excluded)}")
    for item in excluded:
        print(f"  исключён {item['symbol']}: нет баров {item['missing_bar_months']} мес., "
              f"нет funding {item['missing_funding_months']} мес.")

    variants = {
        f"L{leverage:g}": notional_for_leverage(leverage)
        for leverage in LEVERAGES
    }
    variants["naive"] = NAIVE_NOTIONAL
    load_cache: dict[str, object] = {}
    portfolios: dict[str, Portfolio] = {}
    for label, notional in variants.items():
        t0 = time.perf_counter()
        sleeves = run_sleeves(root, symbols, base, {}, notional, load_cache)
        portfolios[label] = build_portfolio(symbols, sleeves, notional)
        print(f"  вариант {label}: p={notional:.6f}, "
              f"{time.perf_counter() - t0:.1f} c")

    # Справочная панель: always-hold (threshold_rate=−inf), не гипотеза.
    reference = run_sleeves(root, symbols, base,
                            {"threshold_rate": -np.inf}, variants["L1"],
                            load_cache)
    reference_portfolio = build_portfolio(symbols, reference, variants["L1"])

    primary = portfolios["L1"]
    payload = summarize(primary, symbols, variants["L1"], base,
                        {"window": base.params["window"],
                         "horizon_bars": base.params["horizon_bars"],
                         "fee_bps": base.params["fee_bps"],
                         "slippage_bps": base.params["slippage_bps"]},
                        excluded, time.perf_counter() - started)
    payload["rule"]["leverage"] = PRIMARY_LEVERAGE
    payload["variants"] = {
        label: {
            "notional": variants[label],
            "periods": {name: period_stats(portfolios[label], name)
                        for name in TIME_COLUMNS},
        }
        for label in variants
    }
    payload["reference_always_hold"] = {
        "periods": {name: period_stats(reference_portfolio, name)
                    for name in TIME_COLUMNS},
    }
    payload["smoke"] = bool(smoke)

    verdicts = {
        1: verdict_payload(primary, base, {"notional": variants["L1"]}, 1),
        len(symbols): verdict_payload(
            primary, base, {"notional": variants["L1"]}, len(symbols)),
    }
    payload["verdicts"] = verdicts

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print_report(payload, verdicts)
    print(f"  JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
