import pytest

from alpha_lab.engine.costs import RealisticCost, ZeroCost


def test_zero_cost_has_no_fees():
    c = ZeroCost()

    assert c.fee_bps() == 0.0
    assert c.slippage_bps(1e6, 1e6) == 0.0
    assert c.funding_cost(1.0, 0.001) == 0.0


def test_taker_fee_default_matches_binance_vip0():
    c = RealisticCost()

    assert c.fee_bps() == pytest.approx(5.0)


def test_maker_share_reduces_fee():
    all_taker = RealisticCost(maker_share=0.0)
    half_maker = RealisticCost(maker_share=0.5)

    assert half_maker.fee_bps() < all_taker.fee_bps()
    assert half_maker.fee_bps() == pytest.approx(3.5)


def test_slippage_grows_with_order_size():
    c = RealisticCost()

    small = c.slippage_bps(1_000.0, 1_000_000.0)
    large = c.slippage_bps(100_000.0, 1_000_000.0)

    assert large > small


def test_slippage_has_floor():
    c = RealisticCost(min_slippage_bps=0.5)

    assert c.slippage_bps(0.0, 1_000_000.0) == pytest.approx(0.5)


def test_slippage_capped_when_order_exceeds_bar_volume():
    c = RealisticCost(impact_coef=0.1, min_slippage_bps=0.5)

    # заявка больше бара — доля ограничена единицей
    assert c.slippage_bps(10_000_000.0, 1_000_000.0) == pytest.approx(0.5 + 1e4 * 0.1)


def test_slippage_handles_zero_volume():
    c = RealisticCost(min_slippage_bps=0.5)

    assert c.slippage_bps(1000.0, 0.0) == pytest.approx(0.5)


def test_funding_cost_sign_follows_position():
    c = RealisticCost()

    long_cost = c.funding_cost(position=1.0, rate=0.001)
    short_cost = c.funding_cost(position=-1.0, rate=0.001)

    assert long_cost > 0        # лонг платит при положительном funding
    assert short_cost < 0       # шорт получает
    assert long_cost == pytest.approx(-short_cost)


def test_from_config_reads_values():
    c = RealisticCost.from_config({"taker_fee_bps": 7.5, "impact_coef": 0.2})

    assert c.taker_fee_bps == 7.5
    assert c.impact_coef == 0.2
    assert c.maker_share == 0.0     # значение по умолчанию


def test_from_config_ignores_unknown_keys():
    # конфиг может нести пояснительные записи — они не должны ломать загрузку
    c = RealisticCost.from_config({"taker_fee_bps": 6.0, "comment": "VIP1"})

    assert c.taker_fee_bps == 6.0
    assert c.maker_fee_bps == 2.0


def test_from_config_empty_and_none_use_defaults():
    assert RealisticCost.from_config({}) == RealisticCost()
    assert RealisticCost.from_config(None) == RealisticCost()
