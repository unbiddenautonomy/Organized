"""Apply the frozen causal short-volatility rule to 2024-2025 test data.

No model, feature, threshold, structure, horizon, sizing, or exit parameter is
selected with 2024-2025 outcomes. This is a diagnostic test because those years
were inspected in earlier research and are no longer a pristine holdout.
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

import causal_validation_short_vol as validation
import near_daily_short_vol_risk_managed as panel_source
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "causal_validation_short_vol"
RAW = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
POLICY = OUT / "policy.json"


def frozen_test_trades() -> tuple[pd.DataFrame, pd.DataFrame, float]:
    policy = json.loads(POLICY.read_text(encoding="utf-8"))
    threshold = float(policy["probability_threshold"])
    panel = panel_source.historical_atm_panel(validation.HORIZON_MIN).copy()
    train = panel[panel["year"].le(2022)].copy()
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.01, max_iter=500)
    )
    model.fit(train[retrained.MODEL_FEATURES], train["profitable"])
    scored = panel[panel["year"].ge(2024)].copy()
    scored["model_probability"] = model.predict_proba(
        scored[retrained.MODEL_FEATURES]
    )[:, 1]
    trades = retrained.first_threshold_crossing(scored, threshold)
    return trades, scored, threshold


def raw_session_counts(last_date: pd.Timestamp) -> dict[int, int]:
    con = duckdb.connect()
    rows = con.execute(f"""
        SELECT year(datetime::TIMESTAMP) AS sample_year,
               count(DISTINCT date(datetime::TIMESTAMP)) AS session_days
        FROM read_csv_auto('{RAW.as_posix()}', header=true)
        WHERE expiry_type='MONTHLY'
          AND datetime::TIMESTAMP >= TIMESTAMP '2024-01-01'
          AND date(datetime::TIMESTAMP) <= DATE '{last_date.date()}'
        GROUP BY 1
        ORDER BY 1
    """).fetchall()
    con.close()
    return {int(year): int(days) for year, days in rows}


def performance_row(
    frame: pd.DataFrame,
    period: str,
    total_session_days: int,
    candidate_days: int,
) -> dict:
    result = {
        "period": period,
        "total_session_days": total_session_days,
        "model_candidate_days": candidate_days,
        **validation.pnl_metrics(frame),
    }
    result["signal_coverage_of_sessions"] = (
        result["days"] / total_session_days if total_session_days else np.nan
    )
    result["signal_coverage_of_candidates"] = (
        result["days"] / candidate_days if candidate_days else np.nan
    )
    result["equivalent_trading_months"] = result["days"] / 21.0
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    trades, scored, threshold = frozen_test_trades()
    last_date = pd.Timestamp(scored["date"].max())
    session_counts = raw_session_counts(last_date)
    candidate_counts = scored.groupby("year")["date"].nunique().to_dict()

    rows = []
    for year, period in ((2024, "test_2024"), (2025, "test_2025_partial")):
        sample = trades[trades["year"].eq(year)].copy()
        rows.append(performance_row(
            sample,
            period,
            int(session_counts.get(year, 0)),
            int(candidate_counts.get(year, 0)),
        ))
    rows.append(performance_row(
        trades,
        "combined_2024_2025_partial",
        int(sum(session_counts.values())),
        int(sum(candidate_counts.values())),
    ))
    performance = pd.DataFrame(rows)

    monthly = (
        trades.assign(month=trades["date"].dt.to_period("M").astype(str))
        .groupby("month", as_index=False)
        .agg(
            trading_days=("date", "nunique"),
            profitable_days=("net_pnl_rupees", lambda x: int(x.gt(0).sum())),
            gross_pnl=("gross_pnl_rupees", "sum"),
            costs=("total_cost", "sum"),
            net_pnl=("net_pnl_rupees", "sum"),
        )
    )
    monthly["hit_rate"] = monthly["profitable_days"] / monthly["trading_days"]

    expected_first = (
        scored[scored["model_probability"].ge(threshold)]
        .groupby("date")["entry_datetime"]
        .min()
    )
    actual_first = trades.set_index("date")["entry_datetime"]
    hold_minutes = (
        trades["exit_datetime"] - trades["entry_datetime"]
    ).dt.total_seconds() / 60.0
    entry_minute = trades["entry_datetime"].dt.hour * 60 + trades["entry_datetime"].dt.minute
    exit_minute = trades["exit_datetime"].dt.hour * 60 + trades["exit_datetime"].dt.minute
    pnl_error = (
        trades["gross_pnl_rupees"] - trades["total_cost"] - trades["net_pnl_rupees"]
    ).abs()
    audit = pd.DataFrame([
        {"check": "one_trade_per_day", "violations": int(trades["date"].duplicated().sum())},
        {"check": "not_first_threshold_crossing", "violations": int(actual_first.ne(expected_first.reindex(actual_first.index)).sum())},
        {"check": "cross_day_window", "violations": int((trades["entry_datetime"].dt.normalize() != trades["exit_datetime"].dt.normalize()).sum())},
        {"check": "outside_0915_1530", "violations": int((~entry_minute.between(555, 930) | ~exit_minute.between(555, 930)).sum())},
        {"check": "holding_not_exactly_180_minutes", "violations": int((hold_minutes != validation.HORIZON_MIN).sum())},
        {"check": "expiry_before_entry", "violations": int((pd.to_datetime(trades["expiry"]).dt.normalize() < trades["entry_datetime"].dt.normalize()).sum())},
        {"check": "pnl_identity_error_over_1e-8", "violations": int(pnl_error.gt(1e-8).sum())},
        {"check": "outside_requested_test_years", "violations": int((~trades["year"].isin([2024, 2025])).sum())},
    ])
    audit["passed"] = audit["violations"].eq(0)

    validation_policy = json.loads(POLICY.read_text(encoding="utf-8"))
    combined_net = float(performance.loc[
        performance["period"].eq("combined_2024_2025_partial"), "net_pnl"
    ].iloc[0])
    test_status = (
        "TEST_PROFITABLE_BUT_VALIDATION_REJECTION_UNCHANGED"
        if combined_net > 0
        else "TEST_UNPROFITABLE_AND_VALIDATION_REJECTION_UNCHANGED"
    )
    test_policy = {
        "status": test_status,
        "live_deployment": False,
        "original_validation_status": validation_policy["status"],
        "threshold_frozen_from_2023_H1": threshold,
        "test_period": f"2024-01-01 through {last_date.date()}",
        "test_data_used_for_retuning": False,
        "not_pristine_holdout_warning": "2024-2025 outcomes were inspected in earlier research iterations",
        "structure": "one-lot short ATM straddle",
        "same_monthly_expiry": True,
        "horizon_minutes": validation.HORIZON_MIN,
        "slippage_bps": validation.SLIPPAGE_BPS,
    }

    performance.to_csv(OUT / "test_performance_2024_2025.csv", index=False)
    monthly.to_csv(OUT / "test_monthly_breakdown_2024_2025.csv", index=False)
    audit.to_csv(OUT / "test_integrity_audit_2024_2025.csv", index=False)
    smb.write_parquet(trades, OUT / "test_trade_ledger_2024_2025.parquet")
    (OUT / "test_policy.json").write_text(
        json.dumps(test_policy, indent=2), encoding="utf-8"
    )

    report = f"""# Frozen-rule test on 2024-2025

Status: `{test_status}`.

This run applies the model trained on 2019-2022 and the probability threshold
`{threshold:.12f}` selected on 2023 H1. Nothing was retuned with 2024-2025
outcomes. The trade remains a one-lot same-monthly-expiry ATM short straddle,
entered at the first threshold crossing and closed exactly 180 minutes later.

## Test performance

{dataframe_to_markdown(performance.round(4))}

## Monthly results

{dataframe_to_markdown(monthly.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

Net P&L equals premium received minus premium repurchased, fees, and 25 bps
slippage. The clean 2023 validation previously failed, so a profitable later
test would not reverse the rejection or authorize live deployment. In
addition, earlier research inspected 2024-2025 repeatedly, so this period is a
diagnostic test rather than a pristine final holdout.
"""
    (OUT / "TEST_REPORT_2024_2025.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
