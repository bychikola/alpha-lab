"""Модель издержек — место, где рождается и умирает альфа.

Проскальзывание зависит от размера заявки относительно объёма бара: модель
«один тик на сделку» делает невидимой зависимость результата от капитала.
Funding начисляется каждые N часов (интервал у разных пар разный) и для позиций,
удерживаемых дольше нескольких часов, может съесть весь edge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

BPS = 1e-4


class CostModel(Protocol):
    def fee_bps(self) -> float: ...
    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float: ...
    def funding_cost(self, position: float, rate: float) -> float: ...


@dataclass(frozen=True)
class ZeroCost:
    """Заведомо нулевые издержки. Только для инвариантных тестов."""

    def fee_bps(self) -> float:
        return 0.0

    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float:
        return 0.0

    def funding_cost(self, position: float, rate: float) -> float:
        return 0.0


@dataclass(frozen=True)
class RealisticCost:
    """Комиссии Binance USDT-M VIP0 + модель воздействия на рынок.

    taker_fee_bps: 5.0 = 0.05% (реальный тариф VIP0 для USDT-M futures)
    maker_fee_bps: 2.0 = 0.02%
    impact_coef:   коэффициент воздействия; проскальзывание растёт линейно
                   с долей заявки в объёме бара
    """
    taker_fee_bps: float = 5.0
    maker_fee_bps: float = 2.0
    maker_share: float = 0.0
    impact_coef: float = 0.1
    min_slippage_bps: float = 0.5

    @classmethod
    def from_config(cls, cfg: dict) -> "RealisticCost":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in known})

    def fee_bps(self) -> float:
        share = min(max(self.maker_share, 0.0), 1.0)
        return share * self.maker_fee_bps + (1.0 - share) * self.taker_fee_bps

    def slippage_bps(self, order_notional: float, bar_quote_volume: float) -> float:
        if bar_quote_volume <= 0:
            return self.min_slippage_bps
        share = min(max(abs(order_notional) / bar_quote_volume, 0.0), 1.0)
        return self.min_slippage_bps + 1e4 * self.impact_coef * share

    def funding_cost(self, position: float, rate: float) -> float:
        """Издержка funding как доля ноционала. Лонг платит при rate > 0."""
        return position * rate
