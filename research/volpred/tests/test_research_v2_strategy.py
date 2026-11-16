from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import research.research_v2_strategy as strategy

from research.research_v2_strategy import (
    DECISION_CLOCKS, RiskState, _fixed_exit, atomic_fill, build_session_iv_cache,
    build_structure, evaluate_policy, first_complete_entry_snapshot_after,
    first_snapshot_after, historical_lot_size, load_event_calendar,
    index_option_sessions, load_approved_predictions, nearest_monthly_chain,
    open_pseudo_strategy_once, prepare_option_rows,
    prior_session_atm_ivs, select_entry_snapshot, uncertainty_cost, verify_strategy_spec,
    pretrade_state_rejection, proposed_loss_rejection, update_risk_state,
)
from research.execution_costs import CostBreakdown
from research.research_v2_evaluation import summarize_ledgers


TZ = "Asia/Kolkata"


def option_snapshot(day: str, observed: str, tick: str, *, exit_shift: float = 0.0,
                    oi: float = 5000.0) -> list[dict]:
    strikes = [9800, 9900, 10000, 10100, 10200]
    rows = []
    for strike in strikes:
        distance = abs(strike - 10000) / 100
        rows.append({
            "datetime": f"{day} {observed}", "Date": day, "tick_time": tick,
            "expiry": "2023-02-23", "is_monthly": True, "strike": strike,
            "spot_price": 10000.0, "call_price": 100 + distance + exit_shift,
            "put_price": 100 + distance + exit_shift, "call_oi": oi, "put_oi": oi,
            "iv_call": 0.20, "iv_put": 0.20,
            "iv_call_converged": True, "iv_put_converged": True,
        })
    # Farther recorded monthly expiry must not be selected.
    far = dict(rows[2]); far["expiry"] = "2023-03-30"; far["strike"] = 10000
    rows.append(far)
    return rows


def synthetic_options(include_exit: bool = True, low_oi: bool = False) -> pd.DataFrame:
    rows = []
    # Twenty prior eligible sessions establish the causal IV median.
    for day in pd.bdate_range("2023-01-02", periods=20):
        rows += option_snapshot(str(day.date()), "10:15:00", "10:15:01")
    rows += option_snapshot("2023-01-30", "10:15:00", "10:15:01", oi=10 if low_oi else 5000)
    if include_exit:
        rows += option_snapshot("2023-01-30", "12:15:01", "12:15:02", exit_shift=-10)
    return pd.DataFrame(rows)


def predictions() -> pd.DataFrame:
    # IV variance is about 5.08e-5; this forecast makes short_ratio exceed 1.25.
    return pd.DataFrame({
        "decision_time": [pd.Timestamp("2023-01-30 10:15:00", tz=TZ),
                          pd.Timestamp("2023-01-30 11:45:00", tz=TZ)],
        "p2_forecast": [0.00002, 0.00002],
    })


def test_strategy_spec_hash_and_tamper_failure(tmp_path: Path):
    assert len(verify_strategy_spec()) == 64
    spec = tmp_path / "spec.md"; spec.write_text("x", encoding="utf-8")
    checksum = tmp_path / "spec.sha256"; checksum.write_text("0" * 64 + " *spec.md\n")
    with pytest.raises(PermissionError, match="checksum mismatch"):
        verify_strategy_spec(spec, checksum)


@pytest.mark.parametrize("segment", ["development", "validation"])
def test_approved_prediction_loader_projects_actual_schema_without_outcomes(segment: str):
    frame = load_approved_predictions(segment)
    assert list(frame.columns) == ["decision_time", "p2_forecast"]
    assert not frame.empty
    assert str(frame.decision_time.dtype) == "datetime64[us, Asia/Kolkata]"
    assert frame.p2_forecast.gt(0).all()


def test_available_at_and_strict_next_snapshot_use_later_tick():
    options = prepare_option_rows(synthetic_options())
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    snapshot = first_snapshot_after(options, decision, strict=True)
    assert snapshot.available_at.min() == pd.Timestamp("2023-01-30 10:15:01", tz=TZ)
    chain = nearest_monthly_chain(snapshot, decision)
    assert set(chain.expiry) == {date(2023, 2, 23)}


def test_entry_scans_to_first_complete_chain_without_consulting_exit_rows():
    partial = option_snapshot("2023-01-30", "10:15:00", "10:15:01")
    partial = [row for row in partial if row["strike"] != 10200]
    complete = option_snapshot("2023-01-30", "10:15:01", "10:15:02")
    future_exit = option_snapshot("2023-01-30", "12:15:01", "12:15:02")
    options = prepare_option_rows(pd.DataFrame(partial + complete + future_exit))
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    selected = first_complete_entry_snapshot_after(options, decision)
    assert selected.available_at.min() == pd.Timestamp("2023-01-30 10:15:02", tz=TZ)
    without_exit = options[options.available_at.lt(pd.Timestamp("2023-01-30 12:00", tz=TZ))]
    selected_without_exit = first_complete_entry_snapshot_after(without_exit, decision)
    assert selected_without_exit.available_at.min() == selected.available_at.min()


def test_long_signal_uses_first_atm_complete_snapshot_without_unused_wings():
    first = option_snapshot("2023-01-30", "10:15:00", "10:15:01")
    first = [row for row in first if row["strike"] == 10000]
    later = option_snapshot("2023-01-30", "10:15:01", "10:15:02")
    options = prepare_option_rows(pd.DataFrame(first + later))
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    snapshot, _, structure, _, reason = select_entry_snapshot(
        options, decision, forecast=0.0002, threshold=1.25,
    )
    assert reason == "selected" and structure == "long_straddle"
    assert snapshot.available_at.min() == pd.Timestamp("2023-01-30 10:15:01", tz=TZ)


def test_asynchronous_entry_snapshot_is_skipped_fail_closed():
    asynchronous = option_snapshot("2023-01-30", "10:15:02", "10:15:01")
    asynchronous[0]["datetime"] = "2023-01-30 10:15:00"
    asynchronous[0]["tick_time"] = "10:15:02"  # same available_at, conflicting raw clocks
    complete = option_snapshot("2023-01-30", "10:15:02", "10:15:03")
    options = prepare_option_rows(pd.DataFrame(asynchronous + complete))
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    snapshot, _, structure, _, _ = select_entry_snapshot(
        options, decision, forecast=0.0002, threshold=1.25,
    )
    assert structure == "long_straddle"
    assert snapshot.available_at.min() == pd.Timestamp("2023-01-30 10:15:03", tz=TZ)


def test_session_index_entry_and_exit_are_equivalent_to_full_panel_filtering():
    options = prepare_option_rows(synthetic_options())
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    indexed = index_option_sessions(options)[decision.date()]
    full_entry = first_complete_entry_snapshot_after(options, decision)
    indexed_entry = first_complete_entry_snapshot_after(indexed, decision)
    assert indexed_entry.available_at.tolist() == full_entry.available_at.tolist()
    chain = nearest_monthly_chain(full_entry, decision)
    legs, _ = build_structure(chain, "iron_fly")
    full_exit = _fixed_exit(options, decision, full_entry.available_at.min(), date(2023, 2, 23), legs)
    indexed_exit = _fixed_exit(indexed, decision, indexed_entry.available_at.min(), date(2023, 2, 23), legs)
    assert indexed_exit == full_exit


def test_fixed_recorded_strikes_and_atomic_oi_rejection():
    options = prepare_option_rows(synthetic_options())
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    chain = nearest_monthly_chain(first_snapshot_after(options, decision, strict=True), decision)
    legs, _ = build_structure(chain, "iron_fly")
    assert [(x.option_type, x.strike, x.qty) for x in legs] == [
        ("call", 10000.0, -1), ("put", 10000.0, -1),
        ("call", 10200.0, 1), ("put", 9800.0, 1),
    ]
    assert atomic_fill(legs)[0]
    low = prepare_option_rows(synthetic_options(low_oi=True))
    low_chain = nearest_monthly_chain(first_snapshot_after(low, decision, strict=True), decision)
    assert not atomic_fill(build_structure(low_chain, "iron_fly")[0])[0]


def test_first_signal_per_day_and_future_exit_deletion_does_not_change_entry_count():
    observed = evaluate_policy(
        predictions(), synthetic_options(include_exit=True), set(), threshold=1.25,
        lot_mode="constant15", segment="development",
    )
    missing = evaluate_policy(
        predictions(), synthetic_options(include_exit=False), set(), threshold=1.25,
        lot_mode="constant15", segment="development",
    )
    assert observed.selected_signal.sum() == 1
    assert missing.selected_signal.sum() == 1
    assert len(observed) == len(missing) == len(predictions())
    observed_trade = observed[observed.selected_signal].iloc[0]
    missing_trade = missing[missing.selected_signal].iloc[0]
    assert not bool(observed_trade.missing_exit)
    assert bool(missing_trade.missing_exit)
    assert missing_trade.exit_treatment == "contractual_adverse_missing_exit"
    assert observed_trade.entry_strikes == missing_trade.entry_strikes
    assert observed.iloc[1].status == missing.iloc[1].status == "no_trade"
    assert observed.expected_decision_count.eq(len(predictions())).all()
    assert observed.decision_universe_sha256.nunique() == 1
    assert set(observed.decision_id) == set(missing.decision_id)
    assert observed_trade.turnover_rupees > 0
    assert observed_trade.margin_status == "unverified_margin_proxy"
    assert not bool(observed_trade.verified_margin_pass)
    assert not bool(observed_trade.promotion_eligible)


def test_runner_ledger_summarizes_end_to_end_with_mixed_statuses():
    ledger = evaluate_policy(
        predictions(), synthetic_options(), set(), threshold=1.25,
        lot_mode="constant15", segment="development",
    )
    ledger["trial_id"] = "synthetic-trial"
    summary = summarize_ledgers(ledger, trial_count=1)
    assert summary.iloc[0].eligible_decisions == len(predictions())
    assert summary.iloc[0].filled == 1 and summary.iloc[0].no_trade == 1


def test_exit_target_is_decision_plus_120_minutes_not_entry_plus_120():
    entry_rows = option_snapshot("2023-01-30", "10:19:59", "10:20:00")
    exit_rows = option_snapshot("2023-01-30", "12:15:01", "12:15:02", exit_shift=-10)
    options = prepare_option_rows(pd.DataFrame(entry_rows + exit_rows))
    decision = pd.Timestamp("2023-01-30 10:15:00", tz=TZ)
    entry = pd.Timestamp("2023-01-30 10:20:00", tz=TZ)
    chain = nearest_monthly_chain(options[options.available_at.eq(entry)], decision)
    legs, _ = build_structure(chain, "iron_fly")
    exit_time, prices = _fixed_exit(options, decision, entry, date(2023, 2, 23), legs)
    assert exit_time == pd.Timestamp("2023-01-30 12:15:02", tz=TZ)
    assert prices is not None


def test_session_iv_cache_is_causal_and_bounded():
    options = prepare_option_rows(synthetic_options())
    cache = build_session_iv_cache(options)
    history = prior_session_atm_ivs(cache, date(2023, 1, 30))
    assert len(history) == 20
    assert all(value == pytest.approx(0.20) for value in history)
    assert date(2023, 1, 30) not in sorted(cache)[:20]


def test_event_calendar_requires_exact_audited_schema_and_filters_primary(tmp_path: Path):
    calendar = tmp_path / "events.csv"
    calendar.write_text(
        "date,event_type,event_name,source_url,announced_at,primary_no_entry\n"
        "2023-01-30,policy,RBI,https://example.test/rbi,2022-12-01T10:00:00+05:30,true\n"
        "2023-02-01,data,PMI,https://example.test/pmi,2022-12-02T10:00:00+05:30,false\n",
        encoding="utf-8",
    )
    events, digest = load_event_calendar(calendar)
    assert events.primary_no_entry_dates == {date(2023, 1, 30)} and len(digest) == 64
    calendar.write_text("event_date,source,frozen_at\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="audited schema"):
        load_event_calendar(calendar)


def test_frozen_calendar_retains_unscheduled_halt_as_incident_not_hindsight_exclusion():
    events, digest = load_event_calendar(Path("EVENT_CALENDAR_V2.csv"))
    halt_day = date(2021, 2, 24)
    assert digest == strategy.EVENT_CALENDAR_SHA256
    assert halt_day in events.incident_stress_dates
    assert halt_day not in events.primary_no_entry_dates
    assert events.metadata_by_date[halt_day][0]["event_type"] == "nse_exchange_halt"


def test_event_and_risk_rejections_are_retained():
    event = evaluate_policy(
        predictions(), synthetic_options(), {date(2023, 1, 30)}, threshold=1.25,
        lot_mode="constant15", segment="development",
    )
    assert event.iloc[0].status == "rejected"
    assert event.iloc[0].reason == "event_day_no_entry"
    low_oi = evaluate_policy(
        predictions(), synthetic_options(low_oi=True), set(), threshold=1.25,
        lot_mode="constant15", segment="development",
    )
    assert low_oi.iloc[0].status == "rejected"
    assert "capacity" in low_oi.iloc[0].reason


def test_lot_regimes_and_cost_uncertainty_multipliers():
    assert historical_lot_size(date(2024, 12, 24)) == 15
    assert historical_lot_size(date(2024, 12, 25)) == 30
    base = CostBreakdown(brokerage=10, exchange=2, bid_ask_crossing=3,
                         impact_slippage=4, margin_carry=5)
    stressed = uncertainty_cost(base)
    assert stressed.brokerage == 20 and stressed.exchange == 4
    assert stressed.bid_ask_crossing == 9 and stressed.impact_slippage == 12
    assert stressed.margin_carry == 5


def test_weekly_and_drawdown_state_rejects_without_resetting_history():
    state = RiskState()
    update_risk_state(state, date(2023, 1, 2), -25_000)
    assert pretrade_state_rejection(state, date(2023, 1, 3)) == "weekly_loss_limit"
    # A new ISO week resets only the weekly counter; drawdown persists.
    assert pretrade_state_rejection(state, date(2023, 1, 9)) is None
    update_risk_state(state, date(2023, 1, 9), -25_000)
    assert pretrade_state_rejection(state, date(2023, 1, 10)) == "weekly_loss_limit"
    assert state.drawdown == 50_000


def test_proposed_severe_loss_checks_remaining_daily_and_weekly_budgets():
    state = RiskState(current_day=date(2023, 1, 3), day_pnl=-4_000,
                      iso_week=(2023, 1), week_pnl=-20_000)
    assert proposed_loss_rejection(state, date(2023, 1, 3), 6_000) == "remaining_daily_loss_budget"
    state.day_pnl = 0
    assert proposed_loss_rejection(state, date(2023, 1, 3), 5_000) == "remaining_weekly_loss_budget"
    state.week_pnl = 0
    state.cumulative_pnl = -45_000
    state.high_water_pnl = 0
    assert proposed_loss_rejection(state, date(2023, 1, 3), 5_000) == "remaining_drawdown_budget"


def test_pseudo_open_requires_frozen_sidecars_before_consuming_marker(tmp_path: Path):
    marker = tmp_path / "PSEUDO_STRATEGY_OPENED.json"
    with pytest.raises(PermissionError):
        open_pseudo_strategy_once(
            frozen_manifest=tmp_path / "manifest.json",
            authorization=tmp_path / "auth.json",
            event_calendar=tmp_path / "events.csv", marker=marker,
        )
    assert not marker.exists()


def test_pseudo_one_shot_marks_and_registers_all_trials_before_outcome_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    root = tmp_path
    p2 = root / "p2"; p2.mkdir()
    out = root / "out"
    files = {
        "spec": root / "spec.md", "addendum": root / "addendum.md",
        "calendar": root / "events.csv", "option": root / "options.csv",
        "pseudo": p2 / "pseudo_holdout_once_predictions.parquet",
        "gate": root / "STRATEGY_DEV_VALIDATION_GATE_V2.json",
        "cost": root / "research/execution_costs.py",
        "test": root / "tests/test_research_v2_strategy.py",
    }
    files["cost"].parent.mkdir(); files["test"].parent.mkdir()
    for key, path in files.items():
        path.write_text("x", encoding="utf-8")
    runner_name = Path(strategy.__file__).name
    files["gate"].write_text(json.dumps({
        "development_strategy_gate_pass": True, "validation_strategy_gate_pass": True,
        "strategy_logic_frozen": True, "strategy_runner_sha256": "runnerhash",
    }), encoding="utf-8")
    monkeypatch.setattr(strategy, "ROOT", root)
    monkeypatch.setattr(strategy, "P2_DIR", p2)
    monkeypatch.setattr(strategy, "OUT", out)
    monkeypatch.setattr(strategy, "SPEC", files["spec"])
    monkeypatch.setattr(strategy, "RISK_ADDENDUM", files["addendum"])
    monkeypatch.setattr(strategy, "EVENT_CALENDAR", files["calendar"])
    monkeypatch.setattr(strategy, "OPTION_SOURCE", files["option"])
    monkeypatch.setattr(strategy, "_relative", lambda path: Path(path).name)

    def fake_hash(path):
        path = Path(path)
        if path == files["calendar"]:
            return strategy.EVENT_CALENDAR_SHA256
        if path.name == runner_name:
            return "runnerhash"
        return "h"

    inventory = {Path(path).name: fake_hash(path) for path in [
        files["spec"], files["addendum"], files["calendar"], Path(strategy.__file__),
        files["cost"], files["test"], files["pseudo"], files["option"], files["gate"],
    ]}
    manifest = {"scope": "frozen_research_v2_pseudo_strategy",
                "thresholds": list(strategy.THRESHOLDS),
                "lot_modes": ["historical", "constant15"], "inventory": inventory}
    auth = {"approved": True, "scope": "research_v2_pseudo_strategy_once",
            "pseudo_authorized": True, "manifest_sha256": "manifesthash"}
    monkeypatch.setattr(strategy, "sha256_file", fake_hash)
    monkeypatch.setattr(strategy, "_checked_json_with_sidecar",
                        lambda path: (manifest, "manifesthash") if "manifest" in Path(path).name else (auth, "authhash"))
    monkeypatch.setattr(strategy, "verify_strategy_spec", lambda: "spec-hash")
    calls: list[str] = []
    monkeypatch.setattr(strategy, "register_strategy_trial",
                        lambda **kwargs: calls.append("register") or f"trial-{len(calls)}")
    monkeypatch.setattr(strategy, "complete_strategy_trial",
                        lambda *args, **kwargs: calls.append("complete"))
    marker = root / "opened.json"

    def fake_predictions(path):
        assert marker.exists() and calls == ["register"] * 7
        calls.append("prediction_read")
        return predictions().iloc[:1]

    monkeypatch.setattr(strategy, "_load_pseudo_predictions", fake_predictions)
    monkeypatch.setattr(strategy, "load_event_calendar",
                        lambda path: (strategy.EventCalendar(frozenset(), frozenset(), {}), "eventhash"))
    monkeypatch.setattr(strategy, "read_option_rows",
                        lambda *args, **kwargs: calls.append("option_read") or prepare_option_rows(synthetic_options()))
    monkeypatch.setattr(strategy, "evaluate_policy", lambda *args, **kwargs: pd.DataFrame({"status": ["no_trade"]}))
    open_pseudo_strategy_once(
        frozen_manifest=root / "manifest.json", authorization=root / "auth.json",
        event_calendar=files["calendar"], marker=marker,
    )
    assert calls[:7] == ["register"] * 7
    assert calls[7:9] == ["prediction_read", "option_read"]
    with pytest.raises(FileExistsError):
        open_pseudo_strategy_once(
            frozen_manifest=root / "manifest.json", authorization=root / "auth.json",
            event_calendar=files["calendar"], marker=marker,
        )
