from math import isclose

import pytest

from research.execution_costs import (
    FillPolicy,
    LiquidityProxy,
    OptionExecution,
    execution_cost,
    margin_proxy,
    rates_on,
    settlement_cost,
)


def test_dated_stt_and_exchange_ipft_boundaries():
    assert rates_on("2023-03-31").option_sale_stt == 0.0005
    assert rates_on("2023-04-01").option_sale_stt == 0.000625
    assert rates_on("2024-09-30").option_sale_stt == 0.000625
    assert rates_on("2024-10-01").option_sale_stt == 0.001
    assert rates_on("2026-04-01").option_sale_stt == 0.0015
    assert rates_on("2026-03-01").option_exercise_stt == 0.00125
    assert rates_on("2026-04-01").option_exercise_stt == 0.0015
    assert rates_on("2026-02-28").exchange == 0.0003503
    assert rates_on("2026-02-28").ipft == 0.000005
    assert rates_on("2026-03-01").exchange == 0.0003552
    assert rates_on("2026-03-01").ipft == 0.000000001


def test_execution_cost_components_and_tax_sides_are_explicit():
    orders = [
        OptionExecution("sell", premium=100, lots=1, lot_size=30),
        OptionExecution("buy", premium=80, lots=1, lot_size=30),
    ]
    costs = execution_cost(orders, "2025-01-02")
    rates = rates_on("2025-01-02")
    assert costs.brokerage == 40
    assert isclose(costs.stt, 3000 * rates.option_sale_stt)
    assert isclose(costs.stamp, 2400 * rates.stamp_on_buy)
    assert isclose(costs.exchange, 5400 * rates.exchange)
    assert isclose(costs.bid_ask_crossing, 5400 * 15 / 10_000)
    assert isclose(costs.impact_slippage, 5400 * 10 / 10_000)
    assert costs.total > sum((costs.brokerage, costs.stt, costs.stamp))


def test_two_x_scenario_doubles_every_component_without_mutation():
    base = execution_cost([OptionExecution("sell", 100, 1, 30)], "2025-01-02")
    stress = base.stressed()
    assert isclose(stress.total, 2 * base.total)
    assert stress.stt == 2 * base.stt
    assert stress.bid_ask_crossing == 2 * base.bid_ask_crossing


def test_liquidity_proxy_scales_with_volatility_and_size():
    proxy = LiquidityProxy()
    calm = proxy.bps(1.0, 1)
    volatile = proxy.bps(2.0, 1)
    larger = proxy.bps(1.0, 4)
    assert volatile[0] > calm[0]
    assert volatile[1] > calm[1]
    assert larger[1] > calm[1]


def test_fill_policy_is_deterministic_for_unknown_oi_partial_and_rejection():
    policy = FillPolicy(max_primary_lots=2, unknown_oi_capacity_lots=1)
    assert policy.decide(1, open_interest_contracts=None).status == "filled"
    partial = policy.decide(2, open_interest_contracts=None, allow_partial=True)
    assert (partial.status, partial.filled_lots) == ("partial", 1)
    rejected = policy.decide(2, open_interest_contracts=None, allow_partial=False)
    assert (rejected.status, rejected.filled_lots) == ("rejected", 0)
    missing = policy.decide(1, open_interest_contracts=10_000, snapshot_available=False)
    assert missing.status == "rejected"


def test_margin_proxy_applies_buffer_cap_and_carry():
    estimate = margin_proxy(
        capital=1_000_000,
        spot=50_000,
        lot_size=30,
        lots=1,
        holding_hours=6,
        has_uncovered_short=True,
    )
    assert estimate.base_requirement == 300_000
    assert estimate.buffered_requirement == 360_000
    assert estimate.capital_limit == 350_000
    assert not estimate.permitted
    assert estimate.carry_cost > 0
    assert estimate.unvalidated


def test_settlement_path_charges_exercise_and_short_gap_reserve():
    result = settlement_cost(
        trade_date="2026-04-02",
        long_intrinsic_points=100,
        spot_settlement=50_000,
        lot_size=30,
        lots=1,
        has_unclosed_short=True,
    )
    assert isclose(result.exercise_stt, 100 * 30 * 0.0015)
    assert isclose(result.conservative_gap_reserve, 50_000 * 30 * 0.01)


def test_invalid_execution_is_rejected():
    with pytest.raises(ValueError):
        OptionExecution("buy", -1, 1, 30)
