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

from alpha_lab.engine.costs import CostModel

BPS = 1e-4


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    gross_returns: pd.Series
    # Удержанный net (held[t] = net[t-1]) — ценовая экспозиция. По нему же
    # trade_returns сегментирует направленные сделки.
    positions: pd.Series
    # Оборот по базе издержек: у одноногой книги |Δnet|, у двухногой |Δgross|
    # (торгуемый ноционал обеих ног). Совпадает с базой комиссии и
    # проскальзывания, а не с изменением цены P&L.
    turnover: pd.Series
    costs: pd.DataFrame          # колонки fee, slippage, funding
    total_return: float
    max_drawdown: float
    bars: int
    price_returns: pd.Series      # доходности самой цены; нужны permutation-тесту
    # Число баров, где заявка превысила max_participation от объёма бара.
    # Диагностика, а не ограничение: движок не режет и не откладывает заявки,
    # поэтому прогон с cap_hits > 0 оптимистичен, а его ёмкость не доказана.
    # Поле обязательное: дефолт 0 молча объявлял бы любой прогон доказанным.
    cap_hits: int
    # Максимальное наблюдённое участие: max по барам trade_notional / quote_volume.
    # Мера запаса до лимита заполнения — показывает, насколько близко стратегия
    # подошла к порогу max_participation, даже когда cap_hits == 0 (флаг горит
    # только на самом пороге и легко теряется). 0.0, если заявок не было.
    # Бары с нулевым или нефинитным объёмом пропускаются (деление на ноль), но
    # заявка на таком баре всё равно попадает в cap_hits.
    max_participation_observed: float

    @property
    def cost_totals(self) -> dict[str, float]:
        return {c: float(self.costs[c].sum()) for c in self.costs.columns}

    @property
    def over_capacity(self) -> bool:
        """True, если хотя бы одна заявка превысила долю объёма бара.

        Это диагностический флаг, а не ограничение исполнения: движок не
        урезает заявки до объёма бара, поэтому такой прогон оптимистичен и
        его ёмкость не доказана. Проверять перед вердиктом.
        """
        return self.cap_hits > 0


def _shift_to_held(arr: np.ndarray) -> np.ndarray:
    """Позиция, удерживаемая в баре t: held[t] = arr[t-1], held[0] = 0.

    Единственное место сдвига для всех трёх баз: цена P&L, издержки и funding
    обязаны видеть одну и ту же временную границу, иначе бар исполнения
    разъедется между ними.
    """
    held = np.empty(len(arr), dtype="float64")
    held[0] = 0.0
    held[1:] = arr[:-1]
    return held


def _checked_basis_array(values, n: int, name: str) -> np.ndarray:
    """Ряд базы (gross/carry): длина как у баров и только конечные значения.

    NaN в carry отравил бы funding-арифметику, а NaN/inf в gross — издержки и
    диагностику ёмкости; оба отказа обязаны быть громкими и называть аргумент,
    а не превращаться в тихо неверный прогон. Для positions такого запрета нет:
    там NaN исторически означает «вне позиции» (fillna(0)).
    """
    arr = pd.Series(values).astype("float64").to_numpy()
    if len(arr) != n:
        raise ValueError(
            f"Длина {name} ({len(arr)}) не совпадает с длиной баров ({n})"
        )
    if not np.isfinite(arr).all():
        first_bad = float(arr[~np.isfinite(arr)][0])
        raise ValueError(
            f"{name} содержит нефинитные значения (NaN/inf, первое — "
            f"{first_bad!r}): такая база молча испортила бы издержки или "
            f"funding"
        )
    return arr


def run_backtest(bars: pd.DataFrame, positions: pd.Series, cost_model: CostModel,
                 initial_equity: float = 1.0, capital: float = 10_000.0,
                 funding_rate: pd.Series | None = None,
                 max_participation: float = 0.01,
                 gross_position: pd.Series | None = None,
                 carry_position: pd.Series | None = None) -> BacktestResult:
    """Прогоняет позиции по барам с учётом издержек.

    positions — net (спот + перп): ценовая экспозиция. Ценовой P&L и
    trade_returns считаются только по нему.

    gross_position — |спот| + |перп|, база комиссий и проскальзывания. None —
    одноногая книга: единственная торгуемая нога это net, и оборот равен
    |Δnet|. Через |Δ|net|| его выразить нельзя: разворот +1 -> −1 торгует две
    единицы, а изменение модуля даёт ноль, то есть занижает издержки ровно на
    самом дорогом баре. Поэтому дефолт считает оборот от знакового net и
    побитово повторяет прежний движок. Явно переданный gross — magnitude-база
    двухногой книги; изменение именно этой величины и есть заявка (вход
    дельта-нейтральной книги 0 -> 2 стоит две ноги, хотя net не меняется).

    carry_position — знаковый ноционал ноги перпа, база funding. None —
    одноногая книга: carry = net (прежнее поведение). Funding начисляется на
    удерживаемый carry, а не на net: у дельта-нейтральной книги доход живёт
    целиком здесь.

    capital — размер счёта; вместе с объёмом бара определяет проскальзывание,
    поэтому результат зависит от капитала (это намеренно).

    max_participation — доля объёма бара, выше которой заявка считается
    неисполнимой по смоделированной цене. Это ДИАГНОСТИКА, а не ограничение:
    движок не режет и не переносит заявки, он лишь считает бары, где модель
    издержек перестаёт быть доказанной (cap_hits/over_capacity). Прогон с
    cap_hits > 0 оптимистичен и не должен попадать в вердикт без оговорки.
    """
    n = len(bars)
    if len(positions) != n:
        raise ValueError(
            f"Длина positions ({len(positions)}) не совпадает с длиной баров ({n})"
        )
    gross_arr = None if gross_position is None else _checked_basis_array(
        gross_position, n, "gross_position")
    carry_arr = None if carry_position is None else _checked_basis_array(
        carry_position, n, "carry_position")
    if n < 2:
        raise ValueError("Нужно минимум 2 бара")
    # NaN делает любое сравнение False (cap_hits == 0 при любой заявке), а <= 0
    # инвертирует или перевозбуждает диагностику. Это не ограничение исполнения,
    # а защита самой диагностики: невалидный порог молча «доказывал» бы ёмкость.
    if not np.isfinite(max_participation) or max_participation <= 0.0:
        raise ValueError(
            "max_participation должен быть конечным положительным числом "
            f"(получено {max_participation!r})"
        )

    close = bars["close"].astype("float64").to_numpy()
    quote_volume = bars["quote_volume"].astype("float64").to_numpy()
    pos = positions.astype("float64").fillna(0.0).to_numpy()

    # Позиция, удерживаемая в баре t, решена на баре t-1
    held = _shift_to_held(pos)

    # Доходность цены бар-к-бару
    price_ret = np.zeros(n, dtype="float64")
    price_ret[1:] = close[1:] / close[:-1] - 1.0
    price_ret[~np.isfinite(price_ret)] = 0.0

    pnl = held * price_ret

    # Оборот: изменение удерживаемой базы издержек. Дефолт (gross=None) — это
    # знаковый held: одноногая книга торгует ровно net. Явный gross — его
    # изменение: magnitude-база двухногой книги, где заявка — это то, что
    # реально попадает на рынок (обе ноги), даже когда net не меняется.
    cost_basis = held if gross_arr is None else _shift_to_held(gross_arr)
    turnover = np.zeros(n, dtype="float64")
    turnover[1:] = np.abs(cost_basis[1:] - cost_basis[:-1])

    # База проскальзывания — размер ИСПОЛНЯЕМОЙ заявки (turnover * capital),
    # а не удерживаемой позиции: издержка исполнения возникает на сделке, а
    # заявка — это изменение позиции. На выходе (1 -> 0) held = 0, и базис по
    # позиции обнулил бы проскальзывание выхода (остался бы только floor); при
    # развороте (+1 -> -1) заявка вдвое больше позиции, и базис по held занизил
    # бы воздействие вдвое. Обе ошибки занижают издержки и завышают доходность,
    # сильнее всего — на высокооборотных стратегиях.
    trade_notional = turnover * capital
    slippage_bps = np.array([
        cost_model.slippage_bps(trade_notional[i], quote_volume[i]) for i in range(n)
    ])
    fee = turnover * cost_model.fee_bps() * BPS
    slip = turnover * slippage_bps * BPS

    # Funding — на удерживаемый carry (ногу перпа), а не на net: у
    # дельта-нейтральной книги net ≡ 0, и вся доходность живёт здесь.
    carry_held = held if carry_arr is None else _shift_to_held(carry_arr)
    if funding_rate is not None:
        rate = pd.Series(funding_rate).astype("float64").fillna(0.0).to_numpy()
        fund = np.array([
            cost_model.funding_cost(carry_held[i], rate[i]) for i in range(n)
        ])
    else:
        fund = np.zeros(n, dtype="float64")

    # Ёмкость: заявка на баре t — это изменение торгуемого ноционала (тот же
    # trade_notional, что и база проскальзывания: диагностика и модель издержек
    # должны одинаково понимать, что такое «заявка»). Для двухногой книги это
    # изменение gross — именно оно попадает на рынок, даже когда net стоит.
    # Порог не влияет на исполнение (движок не режет заявки), только на
    # диагностику.
    cap_hits = int(np.count_nonzero(trade_notional > max_participation * quote_volume))

    # Наблюдённое участие — насколько близко заявки подошли к лимиту заполнения.
    # Участие определено только там, где есть заявка и положительный конечный
    # объём бара; бары без объёма исключены из деления (заявка на них уже
    # посчитана в cap_hits), поэтому максимум не может стать inf/NaN.
    participation = np.zeros(n, dtype="float64")
    live = (trade_notional > 0.0) & np.isfinite(quote_volume) & (quote_volume > 0.0)
    participation[live] = trade_notional[live] / quote_volume[live]
    max_participation_observed = float(participation.max())

    costs = pd.DataFrame(
        {"fee": fee, "slippage": slip, "funding": fund}, index=bars.index
    )
    net_returns = pnl - fee - slip - fund

    equity = pd.Series(initial_equity * np.cumprod(1.0 + net_returns),
                       index=bars.index, name="equity")
    peak = equity.cummax()
    max_dd = float((equity / peak - 1.0).min())

    return BacktestResult(
        equity=equity,
        returns=pd.Series(net_returns, index=bars.index, name="returns"),
        gross_returns=pd.Series(pnl, index=bars.index, name="gross_returns"),
        positions=pd.Series(held, index=bars.index, name="held"),
        turnover=pd.Series(turnover, index=bars.index, name="turnover"),
        costs=costs,
        total_return=float(equity.iloc[-1] / initial_equity - 1.0),
        max_drawdown=max_dd,
        bars=n,
        price_returns=pd.Series(price_ret, index=bars.index, name="price_returns"),
        cap_hits=cap_hits,
        max_participation_observed=max_participation_observed,
    )


def trade_returns(result: BacktestResult) -> pd.Series:
    """Доходности сделок: от входа до полного закрытия, со всеми издержками.

    Сделка — непрерывный отрезок баров с одинаковым знаком удерживаемой позиции.
    Доходность бара — это net (gross - fee - slippage - funding), но издержки
    бара принадлежат не только бару удержания, поэтому атрибуция такая:
      * вход (0 -> ±) и добор/сокращение внутри сделки — целиком этой сделке;
      * выход (± -> 0) — закрываемой сделке: gross на баре выхода равен нулю,
        но заявка на выход оплачивается именно на нём (раньше эта издержка
        терялась и сделки выглядели лучше, чем были);
      * разворот (± -> ∓) — одна заявка закрывает старую позицию и открывает
        новую, поэтому издержка делится пропорционально |held[t-1]| : |held[t]|,
        а gross и funding бара относятся к новой позиции;
      * flat-бар между сделками (0 -> 0) — оборота и издержек нет.
    Инвариант: trade_returns(result).sum() == result.returns.sum() с точностью
    до ошибки сложения float64. Без него win-rate и profit factor (Task 12)
    видели бы только издержку входа и систематически завышали бы качество.

    **net ≡ 0 (дельта-нейтральная книга).** Направленных сделок нет, и
    результат — пустой float64-ряд: это честный ответ, а не потеря данных.
    Издержки и funding двухногой книги тогда не атрибутируются ни одной
    направленной сделке, и инвариант суммы сознательно не выполняется (пустой
    ряд против ненулевой доходности). Следствие для вердикта: min_trades
    провалит такую стратегию, и это консервативно правильно — число
    независимых ставок в carry-режиме не равно числу баров, а подменять его
    выдуманными «сделками» значило бы сдать гейт подгонкой. Собственная
    единица сделки для carry-класса — предмет S4, а не этого модуля.
    """
    held = np.nan_to_num(result.positions.to_numpy(dtype="float64"), nan=0.0)
    gross = result.gross_returns.to_numpy(dtype="float64")
    funding = result.costs["funding"].to_numpy(dtype="float64")
    fee_slip = (result.costs["fee"].to_numpy(dtype="float64")
                + result.costs["slippage"].to_numpy(dtype="float64"))

    # Номера сделок по барам (-1 — вне сделки). Новая сделка начинается там,
    # где знак ненулевой позиции отличается от знака предыдущего бара.
    sign = np.sign(held)
    prev_sign = np.concatenate(([0.0], sign[:-1]))
    starts = (sign != 0.0) & (sign != prev_sign)
    trade_id = np.cumsum(starts) - 1
    trade_id[sign == 0.0] = -1
    n_trades = int(trade_id.max()) + 1 if len(trade_id) else 0
    if n_trades == 0:
        # Ни одной направленной сделки (в том числе net ≡ 0 у двухногой
        # книги): издержкам и funding не к чьей сделке приписаться. Пустой
        # ряд — честный ответ; цикл ниже упал бы на trade_id = -1 при size 0.
        return pd.Series(dtype="float64", name="trade_return")
    totals = np.zeros(n_trades, dtype="float64")

    for i in range(len(held)):
        # gross и funding начисляются на удерживаемую позицию — её сделке.
        if trade_id[i] >= 0:
            totals[trade_id[i]] += gross[i] - funding[i]
        cost = fee_slip[i]
        if cost == 0.0:
            continue
        if sign[i] == prev_sign[i]:
            totals[trade_id[i]] -= cost              # добор внутри сделки
        elif prev_sign[i] == 0.0:
            totals[trade_id[i]] -= cost              # вход с плоского
        elif sign[i] == 0.0:
            totals[trade_id[i - 1]] -= cost          # выход в плоское
        else:
            # Разворот: заявка = закрытие |held[t-1]| + открытие |held[t]|.
            w_old = abs(held[i - 1]) / (abs(held[i - 1]) + abs(held[i]))
            close_cost = w_old * cost
            totals[trade_id[i - 1]] -= close_cost
            totals[trade_id[i]] -= cost - close_cost

    # dtype="float64" обязателен и для пустого результата: object-пустышка
    # ломает типы у потребителей (win-rate/profit factor в Task 12).
    return pd.Series(totals, dtype="float64", name="trade_return")
