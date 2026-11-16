"""Causal re-test of the previously reported near-daily short-volatility rule.

The prior backtest selected the maximum model score after seeing every
timestamp in a completed session.  This audit freezes all economic settings
and instead enters on the first timestamp whose score crosses the frozen
threshold.  That is the signal that could have been acted on live.
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
import near_daily_short_vol_risk_managed as prior
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "near_daily_short_vol_causal"
RAW = ROOT / "data" / "raw" / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
PATH_CACHE = OUT / "causal_atm_straddle_paths_180m.parquet"

HORIZON_MIN = 180
ENTRY_THRESHOLD = 0.5069686488117613
SIZE_THRESHOLD = 0.6382219825441634
HIGH_CONFIDENCE_LOTS = 3
STOP_FRACTION = 0.50
TAKE_FRACTION = 1.00
SLIPPAGE_BPS = 25.0


def causal_candidates() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score each timestamp, then retain the first observable crossing per day."""
    panel = prior.historical_atm_panel(HORIZON_MIN)
    training = panel[panel["year"].le(2022)].copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    model.fit(training[retrained.MODEL_FEATURES], training["profitable"])

    scored = panel[panel["year"].ge(2023)].copy()
    scored["model_probability"] = model.predict_proba(
        scored[retrained.MODEL_FEATURES]
    )[:, 1]
    eligible = scored[
        scored["model_probability"].ge(ENTRY_THRESHOLD)
        & scored["days_to_expiry"].ge(0)
    ].copy()
    daily = (
        eligible.sort_values(["date", "entry_datetime"])
        .groupby("date", as_index=False, group_keys=False)
        .head(1)
        .copy()
    )
    daily["trade_id"] = np.arange(len(daily), dtype=np.int64) + 180_000_000
    return daily, scored


def load_paths(daily: pd.DataFrame) -> pd.DataFrame:
    """Load the fixed entry strike path through the scheduled 180-minute exit."""
    if PATH_CACHE.exists():
        return smb.read_parquet(PATH_CACHE)
    keys = daily[["trade_id", "entry_datetime", "exit_datetime", "expiry"]].copy()
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.register("trade_keys", keys)
    con.execute(f"""
        CREATE TEMP TABLE raw_small AS
        SELECT datetime::TIMESTAMP AS quote_datetime, expiry::DATE AS expiry,
               strike_position::INTEGER AS strike_position, strike::DOUBLE AS strike,
               call_price::DOUBLE AS call_price, put_price::DOUBLE AS put_price
        FROM read_csv_auto('{RAW.as_posix()}', header=true)
        WHERE expiry_type='MONTHLY' AND datetime >= TIMESTAMP '2023-01-01'
    """)
    paths = con.execute("""
        WITH entries AS (
          SELECT k.*, r.strike,
                 r.call_price + r.put_price AS entry_premium
          FROM trade_keys k
          JOIN raw_small r ON r.quote_datetime=k.entry_datetime
                          AND r.expiry=k.expiry AND r.strike_position=0
        )
        SELECT e.trade_id, e.entry_datetime, e.exit_datetime AS scheduled_exit,
               e.expiry, e.strike, e.entry_premium, r.quote_datetime,
               r.call_price + r.put_price AS premium
        FROM entries e
        JOIN raw_small r ON r.expiry=e.expiry AND r.strike=e.strike
                        AND r.quote_datetime > e.entry_datetime
                        AND r.quote_datetime <= e.exit_datetime
        ORDER BY e.trade_id, r.quote_datetime
    """).fetchdf()
    con.close()
    smb.write_parquet(paths, PATH_CACHE)
    return paths


def metrics(frame: pd.DataFrame) -> dict:
    result = hybrid.metrics(frame)
    result["profitable_days"] = int(frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum())
    return result


def period_row(
    frame: pd.DataFrame, period: str, total_session_days: int, candidate_days: int
) -> dict:
    row = {
        "period": period,
        "total_session_days": total_session_days,
        "model_candidate_days": candidate_days,
        **metrics(frame),
    }
    row["signal_coverage_of_sessions"] = (
        row["days"] / total_session_days if total_session_days else np.nan
    )
    row["signal_coverage_of_candidates"] = (
        row["days"] / candidate_days if candidate_days else np.nan
    )
    row["equivalent_trading_months"] = row["days"] / 21.0
    return row


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    daily, scored = causal_candidates()
    paths = load_paths(daily)
    exited = prior.apply_exits(daily, paths, STOP_FRACTION, TAKE_FRACTION)
    sized = prior.apply_sizing(exited, SIZE_THRESHOLD, HIGH_CONFIDENCE_LOTS, SLIPPAGE_BPS)

    validation = sized[sized["year"].eq(2023)].copy()
    test = sized[sized["year"].eq(2024)].copy()
    holdout = sized[sized["year"].eq(2025)].copy()
    evaluation = sized[sized["year"].ge(2024)].copy()
    h1 = validation[validation["date"].lt("2023-07-01")]
    h2 = validation[validation["date"].ge("2023-07-01")]

    validation_pass = (
        validation["net_pnl_rupees"].sum() > 0
        and h1["net_pnl_rupees"].sum() > 0
        and h2["net_pnl_rupees"].sum() > 0
    )
    status = (
        "PASS_CAUSAL_VALIDATION_FORWARD_PAPER_ONLY"
        if validation_pass
        else "REJECT_CAUSAL_RULE_FAILED_VALIDATION"
    )

    available = scored.groupby("year")["date"].nunique().to_dict()
    con = duckdb.connect()
    raw_days = dict(con.execute(f"""
        SELECT year(datetime::TIMESTAMP) AS sample_year,
               count(DISTINCT date(datetime::TIMESTAMP)) AS session_days
        FROM read_csv_auto('{RAW.as_posix()}', header=true)
        WHERE expiry_type='MONTHLY'
          AND datetime::TIMESTAMP >= TIMESTAMP '2023-01-01'
          AND date(datetime::TIMESTAMP) <= DATE '{scored['date'].max().date()}'
        GROUP BY 1
    """).fetchall())
    con.close()
    performance = pd.DataFrame([
        period_row(validation, "validation_2023", int(raw_days.get(2023, 0)), int(available.get(2023, 0))),
        period_row(test, "test_2024", int(raw_days.get(2024, 0)), int(available.get(2024, 0))),
        period_row(holdout, "holdout_2025_partial", int(raw_days.get(2025, 0)), int(available.get(2025, 0))),
        period_row(evaluation, "combined_2024_2025_partial", int(raw_days.get(2024, 0) + raw_days.get(2025, 0)), int(available.get(2024, 0) + available.get(2025, 0))),
    ])
    halves = pd.DataFrame([
        {"period": "validation_2023_H1", **metrics(h1)},
        {"period": "validation_2023_H2", **metrics(h2)},
    ])

    one_lot = prior.apply_sizing(exited, float("inf"), HIGH_CONFIDENCE_LOTS, SLIPPAGE_BPS)
    one_lot_performance = pd.DataFrame([
        period_row(one_lot[one_lot["year"].eq(2023)], "validation_2023", int(raw_days.get(2023, 0)), int(available.get(2023, 0))),
        period_row(one_lot[one_lot["year"].eq(2024)], "test_2024", int(raw_days.get(2024, 0)), int(available.get(2024, 0))),
        period_row(one_lot[one_lot["year"].eq(2025)], "holdout_2025_partial", int(raw_days.get(2025, 0)), int(available.get(2025, 0))),
        period_row(one_lot[one_lot["year"].ge(2024)], "combined_2024_2025_partial", int(raw_days.get(2024, 0) + raw_days.get(2025, 0)), int(available.get(2024, 0) + available.get(2025, 0))),
    ])

    cost_rows = []
    for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        stressed = prior.apply_sizing(exited, SIZE_THRESHOLD, HIGH_CONFIDENCE_LOTS, bps)
        for year, period in ((2023, "validation_2023"), (2024, "test_2024"), (2025, "holdout_2025_partial")):
            cost_rows.append({"slippage_bps": bps, "period": period, **metrics(stressed[stressed["year"].eq(year)])})
    cost_sensitivity = pd.DataFrame(cost_rows)

    entry_minutes = daily["entry_datetime"].dt.strftime("%H:%M").value_counts().sort_index()
    entry_distribution = entry_minutes.rename_axis("entry_time").reset_index(name="days")
    exit_mix = sized.groupby(["year", "exit_reason"], as_index=False).agg(
        trades=("date", "size"),
        profitable_days=("net_pnl_rupees", lambda x: int(x.gt(0).sum())),
        net_pnl=("net_pnl_rupees", "sum"),
    )

    expected_first = (
        scored[scored["model_probability"].ge(ENTRY_THRESHOLD)]
        .groupby("date")["entry_datetime"].min()
    )
    actual_first = daily.set_index("date")["entry_datetime"]
    first_crossing_violations = int(
        actual_first.ne(expected_first.reindex(actual_first.index)).sum()
    )
    path_trade_counts = paths.groupby("trade_id")["strike"].nunique()
    audit = pd.DataFrame([
        {"check": "one_trade_per_day", "violations": int(sized["date"].duplicated().sum())},
        {"check": "cross_day_window", "violations": int((sized["entry_datetime"].dt.normalize() != sized["exit_datetime"].dt.normalize()).sum())},
        {"check": "outside_0915_1530", "violations": int((
            ~(sized["entry_datetime"].dt.hour * 60 + sized["entry_datetime"].dt.minute).between(555, 930)
            | ~(sized["exit_datetime"].dt.hour * 60 + sized["exit_datetime"].dt.minute).between(555, 930)
        ).sum())},
        {"check": "holding_exceeds_180_minutes", "violations": int((((sized["exit_datetime"] - sized["entry_datetime"]).dt.total_seconds() / 60) > HORIZON_MIN).sum())},
        {"check": "expiry_before_entry", "violations": int((pd.to_datetime(sized["expiry"]).dt.normalize() < sized["entry_datetime"].dt.normalize()).sum())},
        {"check": "not_first_threshold_crossing", "violations": first_crossing_violations},
        {"check": "missing_trade_path", "violations": int(len(daily) - paths["trade_id"].nunique())},
        {"check": "multiple_strikes_in_trade_path", "violations": int(path_trade_counts.gt(1).sum())},
    ])
    audit["passed"] = audit["violations"].eq(0)

    old_performance = pd.read_csv(
        ROOT / "model" / "regimeUnderstanding" / "near_daily_short_vol_risk_managed" / "performance.csv"
    )
    old_combined = float(old_performance.loc[old_performance["period"].eq("combined"), "net_pnl"].iloc[0])
    corrected_combined = float(evaluation["net_pnl_rupees"].sum())
    comparison = pd.DataFrame([
        {"version": "invalid_daily_max_selection", "combined_2024_2025_partial_net_pnl": old_combined},
        {"version": "causal_execution_frozen_old_sizing", "combined_2024_2025_partial_net_pnl": corrected_combined},
        {"version": "deployable_after_validation_gate", "combined_2024_2025_partial_net_pnl": corrected_combined if validation_pass else 0.0},
    ])

    smb.write_parquet(sized, OUT / "all_period_trade_ledger.parquet")
    smb.write_parquet(evaluation, OUT / "evaluation_trade_ledger.parquet")
    performance.to_csv(OUT / "performance.csv", index=False)
    halves.to_csv(OUT / "validation_halves.csv", index=False)
    one_lot_performance.to_csv(OUT / "one_lot_performance.csv", index=False)
    cost_sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)
    entry_distribution.to_csv(OUT / "entry_time_distribution.csv", index=False)
    exit_mix.to_csv(OUT / "exit_mix.csv", index=False)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    comparison.to_csv(OUT / "leakage_comparison.csv", index=False)

    policy = {
        "status": status,
        "live_deployment": False,
        "deployable_pnl_rupees": corrected_combined if validation_pass else 0.0,
        "selection_rule": "first timestamp at or above frozen probability threshold",
        "entry_probability_threshold": ENTRY_THRESHOLD,
        "high_confidence_probability_threshold": SIZE_THRESHOLD,
        "ordinary_lots": 1,
        "high_confidence_lots": HIGH_CONFIDENCE_LOTS,
        "same_monthly_expiry": True,
        "structure": "short_atm_straddle",
        "horizon_minutes": HORIZON_MIN,
        "stop_fraction": STOP_FRACTION,
        "take_fraction": TAKE_FRACTION,
        "slippage_bps": SLIPPAGE_BPS,
        "model_train_period": "2019-2022",
        "validation_period": "2023",
        "test_period": "2024",
        "partially_observed_period": "2025 through 2025-05-09",
        "warning": "2025 has been repeatedly inspected and is not pristine for future optimization",
        "parameter_selection_warning": "entry and sizing thresholds are frozen from the superseded search; reported P&L is diagnostic and the causal validation gate rejects it",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    report = f"""# Causal correction of the near-daily short-volatility backtest

Status: `{status}`.

The previously reported result is withdrawn. It selected the highest model
score from every timestamp after the whole trading day was known. A live
system cannot know in advance which later timestamp will be the day's maximum.

This correction freezes the model, thresholds, same-expiry ATM straddle,
180-minute horizon, 50% stop, one/three-lot sizing, fees, and 25 bps slippage.
It changes only the entry selection: take the first timestamp at which
`model_probability >= {ENTRY_THRESHOLD:.12f}`. The fixed ATM strike selected
at entry is retained through the exit path.

The one/three-lot overlay still uses the sizing threshold inherited from the
superseded search, so its P&L is a diagnostic measurement rather than a new
validated strategy. The one-lot table removes that sizing overlay. Both fail
the causal 2023 acceptance gate.

## Corrected performance

{dataframe_to_markdown(performance.round(4))}

## Validation stability

{dataframe_to_markdown(halves.round(4))}

The acceptance gate requires positive P&L for all of 2023 and independently
for both halves of 2023. A failed validation gate makes deployable P&L zero,
regardless of a later positive period.

## Profit correction

{dataframe_to_markdown(comparison.round(4))}

## One-lot benchmark

{dataframe_to_markdown(one_lot_performance.round(4))}

## Entry-time distribution

{dataframe_to_markdown(entry_distribution)}

## Exit attribution

{dataframe_to_markdown(exit_mix.round(4))}

## Cost sensitivity

{dataframe_to_markdown(cost_sensitivity[["slippage_bps", "period", "days", "net_pnl", "profit_factor", "max_drawdown"]].round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

The 2025 sample ends on 2025-05-09 and has been examined repeatedly. It must
not be treated as a pristine holdout in subsequent strategy optimization.
Mid-price data and modeled costs also do not establish live fill quality.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(policy, indent=2))
    print(performance.to_string(index=False))
    print(halves.to_string(index=False))
    print(comparison.to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
