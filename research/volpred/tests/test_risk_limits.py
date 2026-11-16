from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from risk_limits import EntryRisk, RiskLimits, entry_gate, loss_cvar, max_drawdown  # noqa: E402


def test_margin_buffer_is_inside_the_35_percent_cap() -> None:
    limits = RiskLimits()
    assert limits.buffered_margin_cap == 350_000
    assert limits.raw_margin_cap == pytest.approx(291_666.6666666667)
    at_cap = entry_gate(EntryRisk(
        lots=1, raw_margin=limits.raw_margin_cap, stressed_loss=7_500,
        inputs_fresh=True,
    ))
    assert at_cap.allowed
    over = entry_gate(EntryRisk(
        lots=1, raw_margin=limits.raw_margin_cap + 1, stressed_loss=7_500,
        inputs_fresh=True,
    ))
    assert not over.allowed
    assert "buffered_margin_cap" in over.reasons


def test_hard_limits_reject_new_entry() -> None:
    decision = entry_gate(EntryRisk(
        lots=2,
        raw_margin=300_000,
        stressed_loss=7_501,
        open_positions=1,
        trades_today=1,
        day_pnl=-10_000,
        week_pnl=-25_000,
        current_drawdown=50_000,
        event_day=True,
        inputs_fresh=False,
    ))
    assert not decision.allowed
    assert set(decision.reasons) == {
        "lot_limit", "position_already_open", "daily_trade_limit", "buffered_margin_cap",
        "trade_stress_loss_cap", "daily_loss_limit", "weekly_loss_limit",
        "drawdown_suspension", "event_day_no_entry", "stale_or_async_inputs",
    }


def test_live_gate_requires_verified_executable_bounded_risk() -> None:
    paper = EntryRisk(
        lots=1,
        raw_margin=200_000,
        stressed_loss=5_000,
        inputs_fresh=True,
        margin_verified=False,
        executable_quotes=False,
        defined_max_loss=False,
    )
    assert entry_gate(paper, live=False).allowed
    live = entry_gate(paper, live=True)
    assert not live.allowed
    assert set(live.reasons) == {"unverified_margin", "nonexecutable_quotes", "unbounded_loss"}


def test_drawdown_includes_decline_from_initial_capital() -> None:
    assert max_drawdown([-3_000, 1_000, -4_000, 2_000]) == 6_000


def test_loss_cvar_uses_the_empirical_worst_tail_as_positive_loss() -> None:
    pnl = [-20_000, -10_000] + [1_000] * 38
    assert loss_cvar(pnl, 0.95) == 15_000


def test_invalid_risk_inputs_fail_closed() -> None:
    with pytest.raises(ValueError):
        entry_gate(EntryRisk(lots=1, raw_margin=float("nan"), stressed_loss=1_000))
    with pytest.raises(ValueError):
        loss_cvar([], 0.95)
    with pytest.raises(ValueError):
        entry_gate(EntryRisk(
            lots=1, raw_margin=1_000, stressed_loss=1_000,
            day_pnl=float("nan"), inputs_fresh=True,
        ))
    with pytest.raises(ValueError):
        entry_gate(EntryRisk(
            lots=1, raw_margin=1_000, stressed_loss=1_000,
            current_drawdown=-1, inputs_fresh=True,
        ))


def test_evidence_flags_fail_closed_by_default() -> None:
    paper = entry_gate(EntryRisk(lots=1, raw_margin=1_000, stressed_loss=1_000))
    assert not paper.allowed
    assert "stale_or_async_inputs" in paper.reasons

    live = entry_gate(
        EntryRisk(lots=1, raw_margin=1_000, stressed_loss=1_000, inputs_fresh=True),
        live=True,
    )
    assert not live.allowed
    assert set(live.reasons) == {
        "unverified_margin", "nonexecutable_quotes", "unbounded_loss"
    }
