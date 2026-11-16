"""Pure, strategy-agnostic controls for the pre-registered risk policy.

The functions in this module do not load market data, estimate profitability,
or authorize live trading.  Currency inputs are Indian rupees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RiskLimits:
    capital: float = 1_000_000.0
    margin_fraction: float = 0.35
    margin_buffer: float = 0.20
    trade_stress_loss_fraction: float = 0.0075
    daily_loss_fraction: float = 0.01
    weekly_loss_fraction: float = 0.025
    drawdown_fraction: float = 0.05
    max_lots: int = 1
    max_open_positions: int = 1
    max_new_positions_per_day: int = 1

    @property
    def buffered_margin_cap(self) -> float:
        return self.capital * self.margin_fraction

    @property
    def raw_margin_cap(self) -> float:
        return self.buffered_margin_cap / (1.0 + self.margin_buffer)

    @property
    def trade_stress_loss_cap(self) -> float:
        return self.capital * self.trade_stress_loss_fraction

    @property
    def daily_loss_limit(self) -> float:
        return self.capital * self.daily_loss_fraction

    @property
    def weekly_loss_limit(self) -> float:
        return self.capital * self.weekly_loss_fraction

    @property
    def drawdown_limit(self) -> float:
        return self.capital * self.drawdown_fraction


@dataclass(frozen=True)
class EntryRisk:
    lots: int
    raw_margin: float
    stressed_loss: float
    open_positions: int = 0
    trades_today: int = 0
    day_pnl: float = 0.0
    week_pnl: float = 0.0
    current_drawdown: float = 0.0
    event_day: bool = False
    # Evidence-bearing flags fail closed.  Callers must opt in explicitly after
    # checking each condition; omission can never authorize a trade.
    inputs_fresh: bool = False
    margin_verified: bool = False
    executable_quotes: bool = False
    defined_max_loss: bool = False


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reasons: tuple[str, ...]
    buffered_margin: float


def entry_gate(entry: EntryRisk, limits: RiskLimits = RiskLimits(), *, live: bool = False) -> GateDecision:
    """Apply hard pre-trade limits; equality at a loss limit is a rejection."""
    if not np.isfinite(entry.raw_margin) or entry.raw_margin < 0:
        raise ValueError("raw_margin must be finite and nonnegative")
    if not np.isfinite(entry.stressed_loss) or entry.stressed_loss < 0:
        raise ValueError("stressed_loss must be finite and nonnegative")
    if not all(np.isfinite(value) for value in (entry.day_pnl, entry.week_pnl)):
        raise ValueError("day_pnl and week_pnl must be finite")
    if not np.isfinite(entry.current_drawdown) or entry.current_drawdown < 0:
        raise ValueError("current_drawdown must be finite and nonnegative")

    buffered_margin = entry.raw_margin * (1.0 + limits.margin_buffer)
    reasons: list[str] = []
    if entry.lots < 1 or entry.lots > limits.max_lots:
        reasons.append("lot_limit")
    if entry.open_positions >= limits.max_open_positions:
        reasons.append("position_already_open")
    if entry.trades_today >= limits.max_new_positions_per_day:
        reasons.append("daily_trade_limit")
    if buffered_margin > limits.buffered_margin_cap:
        reasons.append("buffered_margin_cap")
    if entry.stressed_loss > limits.trade_stress_loss_cap:
        reasons.append("trade_stress_loss_cap")
    if entry.day_pnl <= -limits.daily_loss_limit:
        reasons.append("daily_loss_limit")
    if entry.week_pnl <= -limits.weekly_loss_limit:
        reasons.append("weekly_loss_limit")
    if entry.current_drawdown >= limits.drawdown_limit:
        reasons.append("drawdown_suspension")
    if entry.event_day:
        reasons.append("event_day_no_entry")
    if not entry.inputs_fresh:
        reasons.append("stale_or_async_inputs")
    if live and not entry.margin_verified:
        reasons.append("unverified_margin")
    if live and not entry.executable_quotes:
        reasons.append("nonexecutable_quotes")
    if live and not entry.defined_max_loss:
        reasons.append("unbounded_loss")
    return GateDecision(not reasons, tuple(reasons), buffered_margin)


def max_drawdown(pnl: Iterable[float]) -> float:
    values = np.asarray(tuple(pnl), dtype=float)
    if values.size == 0:
        return 0.0
    if not np.isfinite(values).all():
        raise ValueError("pnl must be finite")
    equity = np.r_[0.0, np.cumsum(values)]
    peaks = np.maximum.accumulate(equity)
    return float(np.max(peaks - equity))


def loss_cvar(pnl: Iterable[float], level: float = 0.95) -> float:
    """Return positive expected loss in the empirical worst (1-level) tail."""
    values = np.asarray(tuple(pnl), dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("pnl must be nonempty and finite")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be between zero and one")
    # Subtract a tiny tolerance so, for example, 5% of 40 is exactly two
    # observations rather than three due to 0.95's binary representation.
    tail_count = max(1, int(np.ceil((1.0 - level) * values.size - 1e-12)))
    return float(max(0.0, -np.sort(values)[:tail_count].mean()))
