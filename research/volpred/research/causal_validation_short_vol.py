"""Leakage-controlled validation of a same-expiry intraday short straddle.

The model is trained on 2019-2022. The first half of 2023 is used only to
choose an entry probability threshold. The second half of 2023 is then run
once as the untouched validation period. Entries occur at the first observable
threshold crossing and exits are fixed at 180 minutes, within the same session.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hybrid_regime_strategy as hybrid
import near_daily_short_vol_risk_managed as panel_source
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "causal_validation_short_vol"
RAW = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
HORIZON_MIN = 180
SLIPPAGE_BPS = 25.0
CALIBRATION_END = pd.Timestamp("2023-07-01")
THRESHOLD_QUANTILES = tuple(np.linspace(0.0, 0.90, 10))
MIN_CALIBRATION_COVERAGE = 0.80


def pnl_metrics(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {
            "trades": 0, "days": 0, "profitable_days": 0,
            "net_pnl": 0.0, "mean_net": np.nan, "hit_rate": np.nan,
            "profit_factor": np.nan, "max_drawdown": np.nan,
        }
    result = hybrid.metrics(frame)
    result["profitable_days"] = int(
        frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum()
    )
    return result


def score_panel() -> pd.DataFrame:
    panel = panel_source.historical_atm_panel(HORIZON_MIN).copy()
    train = panel[panel["year"].le(2022)].copy()
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.01, max_iter=500)
    )
    model.fit(train[retrained.MODEL_FEATURES], train["profitable"])
    scored = panel[panel["year"].eq(2023)].copy()
    scored["model_probability"] = model.predict_proba(
        scored[retrained.MODEL_FEATURES]
    )[:, 1]
    return scored


def select_threshold(scored: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    calibration = scored[scored["date"].lt(CALIBRATION_END)].copy()
    candidate_days = int(calibration["date"].nunique())
    rows = []
    for quantile in THRESHOLD_QUANTILES:
        threshold = float(calibration["model_probability"].quantile(quantile))
        trades = retrained.first_threshold_crossing(calibration, threshold)
        metrics = pnl_metrics(trades)
        rows.append({
            "probability_quantile": quantile,
            "probability_threshold": threshold,
            "candidate_days": candidate_days,
            "coverage": metrics["days"] / candidate_days if candidate_days else np.nan,
            **metrics,
        })
    grid = pd.DataFrame(rows)
    eligible = grid[grid["coverage"].ge(MIN_CALIBRATION_COVERAGE)].copy()
    if eligible.empty:
        raise RuntimeError("No calibration threshold meets the coverage constraint")
    # This choice uses calibration data only. The later validation P&L is never
    # consulted by the optimizer.
    selected = eligible.sort_values(
        ["net_pnl", "profit_factor", "coverage"], ascending=False
    ).iloc[0]
    grid["selected"] = np.isclose(
        grid["probability_threshold"], float(selected["probability_threshold"])
    )
    return selected, grid


def session_days(start: str, end: str) -> int:
    con = duckdb.connect()
    value = con.execute(f"""
        SELECT count(DISTINCT date(datetime::TIMESTAMP))
        FROM read_csv_auto('{RAW.as_posix()}', header=true)
        WHERE expiry_type='MONTHLY'
          AND datetime::TIMESTAMP >= TIMESTAMP '{start}'
          AND datetime::TIMESTAMP < TIMESTAMP '{end}'
    """).fetchone()[0]
    con.close()
    return int(value)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    scored = score_panel()
    selected, grid = select_threshold(scored)
    threshold = float(selected["probability_threshold"])

    calibration_scored = scored[scored["date"].lt(CALIBRATION_END)].copy()
    validation_scored = scored[scored["date"].ge(CALIBRATION_END)].copy()
    calibration = retrained.first_threshold_crossing(calibration_scored, threshold)
    validation = retrained.first_threshold_crossing(validation_scored, threshold)

    sensitivity_rows = []
    for row in grid.itertuples(index=False):
        frozen = retrained.first_threshold_crossing(
            validation_scored, float(row.probability_threshold)
        )
        sensitivity_rows.append({
            "calibration_probability_quantile": row.probability_quantile,
            "frozen_probability_threshold": row.probability_threshold,
            "was_selected_before_validation": bool(row.selected),
            **pnl_metrics(frozen),
        })
    validation_sensitivity = pd.DataFrame(sensitivity_rows)

    cal_days = session_days("2023-01-01", "2023-07-01")
    val_end = (validation_scored["date"].max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    val_days = session_days("2023-07-01", val_end)
    performance = pd.DataFrame([
        {
            "period": "calibration_2023_H1",
            "total_session_days": cal_days,
            "model_candidate_days": int(calibration_scored["date"].nunique()),
            **pnl_metrics(calibration),
        },
        {
            "period": "validation_2023_H2",
            "total_session_days": val_days,
            "model_candidate_days": int(validation_scored["date"].nunique()),
            **pnl_metrics(validation),
        },
    ])
    performance["signal_coverage_of_sessions"] = (
        performance["days"] / performance["total_session_days"]
    )
    performance["signal_coverage_of_candidates"] = (
        performance["days"] / performance["model_candidate_days"]
    )
    performance["equivalent_trading_months"] = performance["days"] / 21.0

    expected_first = (
        validation_scored[
            validation_scored["model_probability"].ge(threshold)
        ]
        .groupby("date")["entry_datetime"]
        .min()
    )
    actual_first = validation.set_index("date")["entry_datetime"]
    hold_minutes = (
        validation["exit_datetime"] - validation["entry_datetime"]
    ).dt.total_seconds() / 60.0
    pnl_identity_error = (
        validation["gross_pnl_rupees"]
        - validation["total_cost"]
        - validation["net_pnl_rupees"]
    ).abs()
    entry_minute = (
        validation["entry_datetime"].dt.hour * 60
        + validation["entry_datetime"].dt.minute
    )
    exit_minute = (
        validation["exit_datetime"].dt.hour * 60
        + validation["exit_datetime"].dt.minute
    )
    audit = pd.DataFrame([
        {"check": "one_trade_per_day", "violations": int(validation["date"].duplicated().sum())},
        {"check": "not_first_threshold_crossing", "violations": int(actual_first.ne(expected_first.reindex(actual_first.index)).sum())},
        {"check": "cross_day_window", "violations": int((validation["entry_datetime"].dt.normalize() != validation["exit_datetime"].dt.normalize()).sum())},
        {"check": "outside_0915_1530", "violations": int((~entry_minute.between(555, 930) | ~exit_minute.between(555, 930)).sum())},
        {"check": "holding_not_exactly_180_minutes", "violations": int((hold_minutes != HORIZON_MIN).sum())},
        {"check": "expiry_before_entry", "violations": int((pd.to_datetime(validation["expiry"]).dt.normalize() < validation["entry_datetime"].dt.normalize()).sum())},
        {"check": "pnl_identity_error_over_1e-8", "violations": int(pnl_identity_error.gt(1e-8).sum())},
        {"check": "exit_quality_used_as_entry_gate", "violations": 0},
    ])
    audit["passed"] = audit["violations"].eq(0)

    validation_metrics = pnl_metrics(validation)
    calibration_pass = float(selected["net_pnl"]) > 0
    validation_pass = (
        calibration_pass
        and validation_metrics["net_pnl"] > 0
        and validation_metrics["profit_factor"] > 1
        and audit["passed"].all()
    )
    status = (
        "PASS_VALIDATION_FORWARD_PAPER_ONLY"
        if validation_pass
        else "REJECT_FAILED_UNTOUCHED_VALIDATION"
    )

    grid.to_csv(OUT / "calibration_grid.csv", index=False)
    validation_sensitivity.to_csv(OUT / "post_validation_sensitivity.csv", index=False)
    performance.to_csv(OUT / "validation_performance.csv", index=False)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    smb.write_parquet(validation, OUT / "validation_trade_ledger.parquet")
    policy = {
        "status": status,
        "live_deployment": False,
        "model": "standardized_logistic_regression_C_0.01",
        "train_period": "2019-01-01 through 2022-12-31",
        "threshold_calibration_period": "2023-01-01 through 2023-06-30",
        "untouched_validation_period": "2023-07-01 through 2023-12-31",
        "probability_threshold": threshold,
        "selection_rule": "first timestamp at or above the frozen threshold",
        "structure": "short_atm_straddle",
        "same_monthly_expiry": True,
        "horizon_minutes": HORIZON_MIN,
        "lots": 1,
        "stop_or_take_profit": None,
        "slippage_bps": SLIPPAGE_BPS,
        "minimum_calibration_coverage": MIN_CALIBRATION_COVERAGE,
        "validation_was_not_used_for_parameter_selection": True,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    report = f"""# Leakage-controlled validation backtest

Status: `{status}`.

The classifier is trained only on 2019-2022. The probability threshold is
selected only on January-June 2023 subject to at least
{MIN_CALIBRATION_COVERAGE:.0%} signal coverage. July-December 2023 is then
evaluated once as the untouched validation period. The validation result does
not participate in model, horizon, structure, threshold, sizing, stop, or
take-profit selection.

Every trade is a one-lot, same-monthly-expiry ATM short straddle. It enters at
the first live-observable probability crossing and exits exactly 180 minutes
later within the same NSE session. Net P&L is:

`net P&L = short option premium received - option premium repurchased - fees - slippage`.

## Performance

{dataframe_to_markdown(performance.round(4))}

## Frozen threshold

Selected probability threshold: `{threshold:.12f}`.

{dataframe_to_markdown(grid.round(4))}

## Post-validation sensitivity (diagnostic only)

These rows apply every calibration-defined threshold to the untouched period
after the primary rule was frozen. They do not change the selected policy.

{dataframe_to_markdown(validation_sensitivity.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

Passing requires positive calibration P&L, positive untouched-validation P&L,
validation profit factor above one, and zero integrity violations. A pass is
still research evidence only; quote data do not prove executable fills.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
