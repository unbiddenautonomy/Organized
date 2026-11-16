"""Conservative BANKNIFTY option execution, fee, and margin proxies.

This module deliberately does not calculate strategy P&L.  Historical bid/ask,
depth, fills, and SPAN files are absent, so spread, impact, fill, and margin
outputs are explicitly model estimates rather than reconstructed executions.

Rate schedule reviewed 2026-10-13.  See ``EXECUTION_MODEL.md`` for sources and
the limitations that must accompany every research result using this module.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date, datetime
from math import floor, sqrt
from typing import Iterable, Literal


Side = Literal["buy", "sell"]


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


@dataclass(frozen=True)
class RegulatoryRates:
    """Rates expressed as fractions of applicable premium turnover."""

    option_sale_stt: float
    option_exercise_stt: float
    exchange: float
    sebi: float
    ipft: float
    stamp_on_buy: float
    gst: float = 0.18


def rates_on(value: date | datetime | str) -> RegulatoryRates:
    """Return the conservative NSE equity-option schedule for a trade date."""
    day = _as_date(value)

    if day < date(2023, 4, 1):
        sale_stt = 0.0005
    elif day < date(2024, 10, 1):
        sale_stt = 0.000625
    elif day < date(2026, 4, 1):
        sale_stt = 0.001
    else:
        sale_stt = 0.0015

    exercise_stt = 0.00125 if day < date(2026, 4, 1) else 0.0015

    # The pre-2024 values intentionally use the highest published NSE member
    # slab as a conservative retail pass-through proxy.  They are not a
    # reconstruction of the unknown broker's actual monthly slab.
    if day < date(2023, 4, 1):
        exchange, ipft = 0.00053, 0.000000001
    elif day < date(2024, 4, 1):
        exchange, ipft = 0.00050, 0.000005
    elif day < date(2024, 10, 1):
        exchange, ipft = 0.000495, 0.000005
    elif day < date(2026, 3, 1):
        exchange, ipft = 0.0003503, 0.000005
    else:
        exchange, ipft = 0.0003552, 0.000000001

    # A uniform 0.003% buyer-side proxy is also used before national stamp-duty
    # harmonisation on 2020-07-01.  The historical state/broker detail needed
    # to reconstruct that earlier period is not in the repository.
    return RegulatoryRates(
        option_sale_stt=sale_stt,
        option_exercise_stt=exercise_stt,
        exchange=exchange,
        sebi=0.000001,
        ipft=ipft,
        stamp_on_buy=0.00003,
    )


@dataclass(frozen=True)
class OptionExecution:
    """One filled option order; premium is in index points per unit."""

    side: Side
    premium: float
    lots: int
    lot_size: int
    order_count: int = 1

    def __post_init__(self) -> None:
        if self.side not in ("buy", "sell"):
            raise ValueError("side must be 'buy' or 'sell'")
        if self.premium < 0 or self.lots <= 0 or self.lot_size <= 0:
            raise ValueError("premium must be nonnegative; lots and lot_size must be positive")
        if self.order_count <= 0:
            raise ValueError("order_count must be positive")

    @property
    def turnover(self) -> float:
        return float(self.premium * self.lots * self.lot_size)


@dataclass(frozen=True)
class LiquidityProxy:
    """Uncalibrated spread/impact assumptions necessitated by missing quotes."""

    base_half_spread_bps: float = 15.0
    base_impact_bps: float = 10.0
    volatility_impact_bps: float = 15.0
    size_impact_bps: float = 10.0

    def bps(self, volatility_ratio: float, lots: int) -> tuple[float, float]:
        if volatility_ratio <= 0 or lots <= 0:
            raise ValueError("volatility_ratio and lots must be positive")
        vol = max(1.0, float(volatility_ratio))
        spread = self.base_half_spread_bps * sqrt(vol)
        impact = (
            self.base_impact_bps
            + self.volatility_impact_bps * (vol - 1.0)
            + self.size_impact_bps * sqrt(max(0, lots - 1))
        )
        return spread, impact


@dataclass(frozen=True)
class CostBreakdown:
    brokerage: float = 0.0
    exchange: float = 0.0
    stt: float = 0.0
    gst: float = 0.0
    sebi: float = 0.0
    ipft: float = 0.0
    stamp: float = 0.0
    bid_ask_crossing: float = 0.0
    impact_slippage: float = 0.0
    margin_carry: float = 0.0

    @property
    def total(self) -> float:
        return float(sum(getattr(self, f.name) for f in fields(self)))

    def stressed(self, multiplier: float = 2.0) -> "CostBreakdown":
        """Return the mandatory all-in cost stress (2x by default)."""
        if multiplier < 1.0:
            raise ValueError("cost stress multiplier must be at least one")
        return CostBreakdown(**{f.name: getattr(self, f.name) * multiplier for f in fields(self)})


def execution_cost(
    executions: Iterable[OptionExecution],
    trade_date: date | datetime | str,
    *,
    volatility_ratio: float = 1.0,
    liquidity: LiquidityProxy = LiquidityProxy(),
    brokerage_per_order: float = 20.0,
    margin_carry: float = 0.0,
) -> CostBreakdown:
    """Price fees and proxy execution friction for already-filled orders."""
    orders = tuple(executions)
    if not orders:
        return CostBreakdown(margin_carry=margin_carry)
    if brokerage_per_order < 0 or margin_carry < 0:
        raise ValueError("brokerage_per_order and margin_carry must be nonnegative")

    rates = rates_on(trade_date)
    buy_turnover = sum(x.turnover for x in orders if x.side == "buy")
    sell_turnover = sum(x.turnover for x in orders if x.side == "sell")
    turnover = buy_turnover + sell_turnover
    brokerage = brokerage_per_order * sum(x.order_count for x in orders)
    exchange = turnover * rates.exchange
    sebi = turnover * rates.sebi
    ipft = turnover * rates.ipft
    gst = rates.gst * (brokerage + exchange + sebi + ipft)

    crossing = impact = 0.0
    for order in orders:
        spread_bps, impact_bps = liquidity.bps(volatility_ratio, order.lots)
        crossing += order.turnover * spread_bps / 10_000.0
        impact += order.turnover * impact_bps / 10_000.0

    return CostBreakdown(
        brokerage=brokerage,
        exchange=exchange,
        stt=sell_turnover * rates.option_sale_stt,
        gst=gst,
        sebi=sebi,
        ipft=ipft,
        stamp=buy_turnover * rates.stamp_on_buy,
        bid_ask_crossing=crossing,
        impact_slippage=impact,
        margin_carry=margin_carry,
    )


@dataclass(frozen=True)
class FillDecision:
    requested_lots: int
    filled_lots: int
    status: Literal["filled", "partial", "rejected"]
    reason: str


@dataclass(frozen=True)
class FillPolicy:
    """Deterministic capacity proxy; OI is not displayed executable size."""

    max_primary_lots: int = 1
    oi_participation: float = 0.001
    unknown_oi_capacity_lots: int = 1
    minimum_completion_ratio: float = 0.5

    def decide(
        self,
        requested_lots: int,
        *,
        open_interest_contracts: int | None,
        snapshot_available: bool = True,
        valid_premium: bool = True,
        allow_partial: bool = False,
    ) -> FillDecision:
        if requested_lots <= 0:
            raise ValueError("requested_lots must be positive")
        if not snapshot_available or not valid_premium:
            reason = "missing eligible next snapshot" if not snapshot_available else "invalid premium"
            return FillDecision(requested_lots, 0, "rejected", reason)
        if open_interest_contracts is not None and open_interest_contracts < 0:
            raise ValueError("open_interest_contracts cannot be negative")

        oi_capacity = (
            self.unknown_oi_capacity_lots
            if open_interest_contracts is None
            else floor(open_interest_contracts * self.oi_participation)
        )
        capacity = min(self.max_primary_lots, oi_capacity)
        possible = min(requested_lots, max(0, capacity))
        if possible == requested_lots:
            return FillDecision(requested_lots, possible, "filled", "within deterministic proxy capacity")
        if (
            allow_partial
            and possible > 0
            and possible / requested_lots >= self.minimum_completion_ratio
        ):
            return FillDecision(requested_lots, possible, "partial", "capacity-limited proxy fill")
        return FillDecision(requested_lots, 0, "rejected", "insufficient proxy capacity")


@dataclass(frozen=True)
class MarginEstimate:
    base_requirement: float
    buffered_requirement: float
    carry_cost: float
    capital_limit: float
    permitted: bool
    method: str
    unvalidated: bool = True


def margin_proxy(
    *,
    capital: float,
    spot: float,
    lot_size: int,
    lots: int,
    holding_hours: float,
    has_uncovered_short: bool,
    premium_debit: float = 0.0,
    defined_max_loss: float | None = None,
    naked_notional_rate: float = 0.20,
    buffer_rate: float = 0.20,
    capital_fraction_limit: float = 0.35,
    annual_carry_rate: float = 0.08,
) -> MarginEstimate:
    """Conservative broker-style proxy; never represents historical SPAN."""
    values = (capital, spot, premium_debit, holding_hours)
    if any(x < 0 for x in values) or lot_size <= 0 or lots <= 0:
        raise ValueError("monetary/time inputs must be nonnegative and sizes positive")
    if defined_max_loss is not None and defined_max_loss < 0:
        raise ValueError("defined_max_loss cannot be negative")

    if has_uncovered_short:
        base = max(premium_debit, spot * lot_size * lots * naked_notional_rate)
        method = "uncovered-short notional proxy"
    elif defined_max_loss is not None:
        base = max(premium_debit, defined_max_loss)
        method = "defined-risk maximum-loss proxy"
    else:
        base = premium_debit
        method = "long-premium debit proxy"

    buffered = base * (1.0 + buffer_rate)
    cap = capital * capital_fraction_limit
    carry = buffered * annual_carry_rate * holding_hours / (365.0 * 24.0)
    return MarginEstimate(base, buffered, carry, cap, buffered <= cap, method)


@dataclass(frozen=True)
class SettlementCost:
    exercise_stt: float
    conservative_gap_reserve: float

    @property
    def total(self) -> float:
        return self.exercise_stt + self.conservative_gap_reserve


def settlement_cost(
    *,
    trade_date: date | datetime | str,
    long_intrinsic_points: float,
    spot_settlement: float,
    lot_size: int,
    lots: int,
    has_unclosed_short: bool,
    gap_bps: float = 100.0,
) -> SettlementCost:
    """Charge long exercise STT and a disclosed missing-exit short-gap reserve."""
    if min(long_intrinsic_points, spot_settlement, gap_bps) < 0 or lot_size <= 0 or lots <= 0:
        raise ValueError("settlement values must be nonnegative and sizes positive")
    exercise = long_intrinsic_points * lot_size * lots * rates_on(trade_date).option_exercise_stt
    gap = spot_settlement * lot_size * lots * gap_bps / 10_000.0 if has_unclosed_short else 0.0
    return SettlementCost(exercise, gap)
