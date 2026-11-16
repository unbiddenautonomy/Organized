from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from research.research_v2_evaluation import (
    annualized_daily_sharpe,
    block_bootstrap_cvar95_ucb,
    development_validation_gate,
    decision_universe_sha256,
    estimate_cscv_pbo,
    full_trial_count,
    load_authorized_ledgers,
    plateau_gate,
    probabilistic_and_deflated_sharpe,
    PSEUDO_SAMPLE_LABEL,
    PSEUDO_SEGMENT,
    primary_tail_gate,
    summarize_ledgers,
    tail_sample_sufficiency,
    trial_registry_counts,
    validate_decision_coverage,
    validate_ledger,
)


def ledger_frame(*, trials=("t1",), segments=("development",), days=24) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2022-01-03", periods=days, tz="Asia/Kolkata")
    for segment in segments:
        for trial_number, trial in enumerate(trials):
            for day_number, day in enumerate(dates):
                rejected = day_number % 7 == 0
                pnl = float((day_number % 5) - 1 + trial_number)
                row = {
                    "segment": segment,
                    "sample_label": (
                        PSEUDO_SAMPLE_LABEL if segment == PSEUDO_SEGMENT else
                        ("walk-forward validation" if segment == "validation" else "development")
                    ),
                    "trial_id": trial, "decision_id": f"{segment}-{day.isoformat()}",
                    "decision_time": day + pd.Timedelta(hours=10), "eligible_session": True,
                    "status": "rejected" if rejected else "filled", "event_day": day_number % 11 == 0,
                    "diagnostic_only": False,
                    "missing_exit": (not rejected and day_number == 9), "gross_pnl": pnl + 1,
                    "ordinary_net": pnl, "net_2x": pnl - 0.5, "uncertainty_net": pnl - 1,
                    "ordinary_cost": 1.0, "cost_2x": 2.0, "uncertainty_cost": 2.5,
                    "buffered_margin": 1000 + day_number, "entry_count": 1,
                    "turnover_rupees": 5000 + day_number, "exit_treatment": "observed_fixed_strikes",
                    "cost_brokerage": 0.2, "cost_stt": 0.1,
                    "daily_loss_breach": False, "weekly_loss_breach": False, "drawdown_breach": False,
                }
                rows.append(row)
    frame = pd.DataFrame(rows)
    for segment, index in frame.groupby("segment").groups.items():
        identifiers = frame.loc[index, "decision_id"].unique()
        frame.loc[index, "expected_decision_count"] = len(identifiers)
        frame.loc[index, "decision_universe_sha256"] = decision_universe_sha256(identifiers)
    return frame


def test_summary_reports_required_economics_risk_and_event_fields():
    summary = summarize_ledgers(ledger_frame(), trial_count=100)
    row = summary.iloc[0]
    assert row["filled"] + row["rejected"] + row["no_trade"] == row["eligible_decisions"] == 24
    assert row["missing_exits"] == 1
    assert np.isfinite(row["worst_day"]) and np.isfinite(row["worst_week"])
    assert row["max_drawdown"] >= 0 and row["historical_loss_cvar95"] is not None
    assert 0 <= row["win_rate"] <= 1
    assert row["turnover_rupees"] > 0 and row["filled_entry_count"] > 0
    assert row["max_margin"] > 0 and row["cost_brokerage_total"] > 0
    assert row["event_selected"] > 0 and row["full_trial_count"] == 100


def test_sharpe_convention_and_deflation_penalize_more_trials():
    daily = np.array([1.0, -0.2, 0.8, 0.3, 1.2, -0.1] * 20)
    sharp = annualized_daily_sharpe(daily)
    expected = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
    assert sharp["annualized_sharpe"] == pytest.approx(expected)
    few = probabilistic_and_deflated_sharpe(daily, 2)
    many = probabilistic_and_deflated_sharpe(daily, 1000)
    assert few["probabilistic_sharpe"] == pytest.approx(many["probabilistic_sharpe"])
    assert many["deflated_sharpe"] < few["deflated_sharpe"]
    assert annualized_daily_sharpe([1.0])["annualized_sharpe"] is None


def test_full_trial_count_rejects_duplicate_registry_ids(tmp_path):
    path = tmp_path / "TRIALS.csv"
    path.write_text("trial_id,status,notes\na,ok,\nb,ok,\n", encoding="utf-8")
    assert full_trial_count(path) == 2
    path.write_text("trial_id,status,notes\na,ok,\na,ok,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        full_trial_count(path)


def test_trial_count_excludes_completion_administration_but_validates_ids(tmp_path):
    path = tmp_path / "TRIALS.csv"
    path.write_text(
        "trial_id,status,notes\n"
        "strategy-a,running,substantive\n"
        "strategy-a-COMPLETION,complete,completion_for=strategy-a\n"
        "strategy-b,running,substantive\n"
        "admin-unusual,complete,completion_for=strategy-b\n",
        encoding="utf-8",
    )
    counts = trial_registry_counts(path)
    assert counts == {"substantive_trials": 2, "administrative_records": 2, "total_records": 4}
    assert full_trial_count(path) == 2


def test_cscv_pbo_estimates_with_complete_variants_and_explains_when_not():
    ledger = ledger_frame(trials=("good", "bad", "mixed"), days=40)
    mask = ledger["trial_id"].eq("bad") & ledger["status"].eq("filled")
    ledger.loc[mask, "ordinary_net"] *= -1
    result = estimate_cscv_pbo(ledger, segment="development", blocks=8)
    assert result["estimable"] is True and result["combinations"] == 70
    unavailable = estimate_cscv_pbo(ledger[ledger["trial_id"].eq("good")], segment="development")
    assert unavailable["estimable"] is False and "two" in unavailable["reason"]


def plateau_summary() -> pd.DataFrame:
    rows = []
    for segment in ("development", "validation"):
        for threshold, total in ((1.15, 950.0), (1.25, 1000.0), (1.35, 1050.0)):
            rows.append({
                "segment": segment, "trial_id": f"{segment}-{threshold}", "threshold": threshold,
                "lot_mode": "historical", "event_counterfactual": False, "ordinary_total": total,
                "net_2x_total": total - 100, "filled": 10,
                "max_positive_month_profit_concentration": 0.30,
                "daily_loss_breaches": 0, "weekly_loss_breaches": 0, "drawdown_breaches": 0,
                "worst_day": -1000, "worst_week": -2000, "max_drawdown": 3000,
                "historical_loss_cvar95": 1200, "missing_exits": 0, "unrecorded_exits": 0,
                "historical_size_500_sessions_pass": True, "historical_size_100_fills_pass": True,
                "cvar_ucb_20k_pass": True, "cvar95_block_bootstrap_estimable": True,
                "required_new_12m_forward_sample_satisfied": False,
            })
    return pd.DataFrame(rows)


def test_plateau_and_development_validation_gate_fail_closed():
    summary = plateau_summary()
    assert plateau_gate(summary, segment="development")["pass"] is True
    assert primary_tail_gate(summary, segment="development")["pass"] is True
    sample = tail_sample_sufficiency(summary, segment="development")
    assert sample["historical_tail_review_sufficient"] is True
    assert sample["forward_sample_satisfied"] is False
    assert development_validation_gate(summary)["pass"] is True
    broken = summary[~((summary.segment == "validation") & np.isclose(summary.threshold, 1.35))]
    result = development_validation_gate(broken)
    assert result["pass"] is False and result["verdict"] == "research_failed"
    concentrated = summary.copy()
    concentrated.loc[np.isclose(concentrated.threshold, 1.25), "max_positive_month_profit_concentration"] = 0.5
    assert development_validation_gate(concentrated)["pass"] is False


def test_validation_rejects_nonfinite_filled_economics_and_unknown_status():
    ledger = ledger_frame(days=4)
    ledger.loc[1, "ordinary_net"] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        validate_ledger(ledger)
    ledger = ledger_frame(days=4)
    ledger.loc[0, "status"] = "dropped"
    with pytest.raises(ValueError, match="filled, rejected, or no_trade"):
        validate_ledger(ledger)


def test_loader_fails_before_ledger_read_without_exact_scope_and_segments(tmp_path):
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        '{"authorized": true, "scope": "development_validation_strategy_evaluation", '
        '"allowed_segments": ["development"], "ledger_sha256": {"x": "bad"}}',
        encoding="utf-8",
    )
    with pytest.raises(PermissionError, match="allowed_segments"):
        load_authorized_ledgers([tmp_path / "does-not-exist.parquet"], authorization)


def test_no_trade_and_complete_decision_coverage_are_required():
    ledger = ledger_frame(trials=("a", "b"), days=8)
    ledger.loc[ledger.index[0], "status"] = "no_trade"
    validated = validate_ledger(ledger)
    assert validated.status.eq("no_trade").sum() == 1
    validate_decision_coverage(validated)
    incomplete = ledger.drop(ledger[(ledger.trial_id == "b")].index[-1])
    with pytest.raises(ValueError, match="decision coverage"):
        validate_decision_coverage(incomplete)
    tampered = ledger.copy()
    tampered["expected_decision_count"] += 1
    with pytest.raises(ValueError, match="absolute decision coverage"):
        validate_decision_coverage(tampered)


def test_block_bootstrap_cvar_is_deterministic_and_tail_gate_fails_closed():
    daily = np.array([1000.0, -500.0, 200.0, -100.0] * 10)
    first = block_bootstrap_cvar95_ucb(daily, replications=200)
    second = block_bootstrap_cvar95_ucb(daily, replications=200)
    assert first["estimable"] is True and first["ucb95"] == second["ucb95"]
    summary = plateau_summary()
    summary.loc[(summary.segment == "development") & np.isclose(summary.threshold, 1.25), "worst_day"] = -25_001
    gate = primary_tail_gate(summary, segment="development")
    assert gate["pass"] is False and gate["checks"]["worst_day_at_least_minus_25000"] is False


def test_cvar_uses_exact_worst_five_percent_without_zero_tie_dilution():
    from research.research_v2_evaluation import _historical_loss_cvar95

    daily = pd.Series([0.0] * 95 + [-100.0] * 5)
    var95, cvar95 = _historical_loss_cvar95(daily)
    assert var95 == 100.0 and cvar95 == 100.0
    bootstrap = block_bootstrap_cvar95_ucb(daily, replications=200)
    assert bootstrap["estimable"] is True and bootstrap["ucb95"] == pytest.approx(100.0)


def test_diagnostic_flag_must_match_eventdiag_trial_identity():
    ledger = ledger_frame(days=4)
    ledger["trial_id"] = "strategy-eventdiag-run"
    ledger["diagnostic_only"] = True
    validated = validate_ledger(ledger)
    assert validated["event_counterfactual"].all()
    ledger["diagnostic_only"] = False
    with pytest.raises(ValueError, match="eventdiag"):
        validate_ledger(ledger)
    primary = ledger_frame(days=4)
    primary["diagnostic_only"] = np.nan
    assert not validate_ledger(primary)["event_counterfactual"].any()


def test_primary_ledger_may_omit_diagnostic_column_but_eventdiag_may_not():
    primary = ledger_frame(days=4).drop(columns="diagnostic_only")
    validated = validate_ledger(primary)
    assert not validated["diagnostic_only"].any()
    assert not validated["event_counterfactual"].any()
    eventdiag = primary.copy()
    eventdiag["trial_id"] = "strategy-eventdiag-run"
    with pytest.raises(ValueError, match="explicit diagnostic_only"):
        validate_ledger(eventdiag)


def test_recorded_contractual_missing_exit_is_disclosed_but_does_not_fail_tail_gate():
    summary = plateau_summary()
    primary = (summary.segment == "development") & np.isclose(summary.threshold, 1.25)
    summary.loc[primary, "missing_exits"] = 3
    summary.loc[primary, "unrecorded_exits"] = 0
    gate = primary_tail_gate(summary, segment="development")
    assert gate["pass"] is True
    assert "zero_missing_exits" not in gate["checks"]
    summary.loc[primary, "unrecorded_exits"] = 1
    assert primary_tail_gate(summary, segment="development")["pass"] is False


def test_missing_exit_requires_exact_frozen_adverse_treatment_in_summary():
    ledger = ledger_frame(days=24)
    missing = ledger["missing_exit"]
    ledger.loc[missing, "exit_treatment"] = "contractual_adverse_missing_exit"
    summary = summarize_ledgers(ledger, trial_count=10).iloc[0]
    assert summary["missing_exits"] == 1 and summary["unrecorded_exits"] == 0
    ledger.loc[missing, "exit_treatment"] = "observed_fixed_strikes"
    summary = summarize_ledgers(ledger, trial_count=10).iloc[0]
    assert summary["missing_exits"] == 1 and summary["unrecorded_exits"] == 1


def test_pseudo_segment_and_visible_sample_label_are_canonical():
    ledger = ledger_frame(segments=(PSEUDO_SEGMENT,), days=4)
    validated = validate_ledger(ledger)
    assert validated.segment.eq(PSEUDO_SEGMENT).all()
    assert validated.sample_label.eq(PSEUDO_SAMPLE_LABEL).all()
    wrong_segment = ledger.copy()
    wrong_segment["segment"] = "pseudo"
    with pytest.raises(ValueError, match="noncanonical pseudo segment"):
        validate_ledger(wrong_segment)
    wrong_label = ledger.copy()
    wrong_label["sample_label"] = "pseudo"
    with pytest.raises(ValueError, match="sample_label exactly"):
        validate_ledger(wrong_label)
