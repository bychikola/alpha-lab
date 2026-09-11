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
    # Число баров, где заявка превысила max_participation от объёма бара.
    # Диагностика, а не ограничение: движок не режет и не откладывает заявки,
    # поэтому прогон с cap_hits > 0 оптимистичен, а его ёмкость не доказана.
    # Поле обязательное: дефолт 0 молча объявлял бы любой прогон доказанным.
    cap_hits: int

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


def run_backtest(bars: pd.DataFrame, positions: pd.Series, cost_model: CostModel,
                 initial_equity: float = 1.0, capital: float = 10_000.0,
                 funding_rate: pd.Series | None = None,
                 max_participation: float = 0.01) -> BacktestResult:
    """Прогоняет позиции по барам с учётом издержек.

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

    # Ёмкость: заявка на баре t — это изменение удерживаемой позиции. Порог
    # не влияет на исполнение (движок не режет заявки), только на диагностику.
    order_notional = turnover * capital
    cap_hits = int(np.count_nonzero(order_notional > max_participation * quote_volume))

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
        cap_hits=cap_hits,
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
