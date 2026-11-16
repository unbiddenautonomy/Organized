from __future__ import annotations

import heapq
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INFERENCE = Path(__file__).resolve().parent.parent / "families" / "ivrv" / 'engine' / 'Inference'
OUTPUT = Path(__file__).resolve().parent / 'analysis_outputs' / 'three_iv_rv_conditions'
FEATURE_ROOT = Path(os.environ.get("VOLTR_V2_FEATURE_ROOT", Path(__file__).resolve().parents[2] / "research" / "v2" / "research"))
COMBINED_FEATURES = FEATURE_ROOT / "combined_atm_iv_rv_convergence_v2" / "artifacts" / "combined_atm_iv_convergence_v2_features.csv"
RV_FEATURES = FEATURE_ROOT / "fixed_strike_iv_rv_convergence" / "artifacts" / "fixed_strike_convergence_features.csv"
LEG_FEATURES = FEATURE_ROOT / "fixed_strike_iv_120m" / "artifacts" / "fixed_strike_iv_120m_dataset.csv"
sys.path.insert(0, str(INFERENCE))

import backtest_current_pipeline as source  # noqa: E402
from inference_pipeline import FixedStrikeInferencePipeline  # noqa: E402

import backtest_user_csv_current_paper_rules as trading  # noqa: E402
import optimize_confidence_threshold_cap3_fast as fast  # noqa: E402


OPTIMIZED_CONFIDENCE = 0.8005355169472045


def rebuild_predictions() -> pd.DataFrame:
    pipeline = FixedStrikeInferencePipeline()
    paths = pd.read_csv(
        source.PATH_OUTCOMES,
        parse_dates=["datetime", "exit_datetime", "stop_only_exit_datetime", "bracket_exit_datetime"],
        low_memory=False,
    )
    candidates = paths.loc[
        paths["datetime"].between(source.COMMON_TEST_START, source.COMMON_TEST_END, inclusive="both")
    ].copy()
    candidates = candidates.sort_values("row_id", kind="stable").reset_index(drop=True)
    candidates["_candidate_order"] = np.arange(len(candidates))

    combined_features: set[str] = set()
    leg_features: set[str] = set()
    for name, payload in pipeline.artifacts.items():
        if name.startswith("combined_"):
            combined_features.update(payload.get("features", []))
        elif name.startswith(("call_", "put_")):
            leg_features.update(payload.get("features", []))
    combined_features -= {"rv_pred_120m", "entry_gap_to_rv_pred_combined"}
    rv_features = set(pipeline.artifacts["rv_xgboost"]["features"])

    combined = source.read_and_align(
        COMBINED_FEATURES,
        combined_features | {"entry_atm_iv_combined", "rv_trailing_60m"},
        candidates,
        "combined",
    )
    rv = source.read_and_align(
        RV_FEATURES, source.source_columns(rv_features), candidates, "RV"
    )
    legs = source.read_and_align(
        LEG_FEATURES,
        leg_features | {"entry_iv_call", "entry_iv_put"},
        candidates,
        "leg",
    )
    predictions = pipeline.predict(combined, legs, rv)
    attach = candidates[["quality_trainable", "lot_size", "row_id"]].reset_index(drop=True)
    predictions = pd.concat([predictions.reset_index(drop=True), attach], axis=1)
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])
    predictions["expiry"] = pd.to_datetime(predictions["expiry"])
    nearest = predictions.groupby("datetime")["expiry"].transform("min")
    predictions = predictions.loc[predictions["expiry"].eq(nearest)].copy()
    minute = predictions["datetime"].dt.hour * 60 + predictions["datetime"].dt.minute
    predictions["base_eligible"] = (
        predictions["quality_trainable"].fillna(False).astype(bool)
        & minute.ge(9 * 60 + 15 + 120)
        & predictions["datetime"].dt.time.lt(trading.SQUARE_OFF)
    )
    predictions["Date"] = predictions["datetime"].dt.strftime("%Y-%m-%d")
    return predictions.reset_index(drop=True)


def select_with_cap(frame: pd.DataFrame, mask: pd.Series, label: str) -> tuple[pd.DataFrame, dict]:
    candidates = frame.loc[mask].sort_values("signal_datetime", kind="stable")
    heap: list[int] = []
    accepted = []
    for row in candidates.itertuples():
        signal_ns = int(pd.Timestamp(row.signal_datetime).value)
        while heap and heap[0] <= signal_ns:
            heapq.heappop(heap)
        if len(heap) >= 3:
            continue
        heapq.heappush(heap, int(pd.Timestamp(row.exit_datetime).value))
        accepted.append(row)
    ledger = pd.DataFrame([row._asdict() for row in accepted])
    pnl = ledger["pnl"].astype(float) if not ledger.empty else pd.Series(dtype=float)
    gains, losses = float(pnl[pnl > 0].sum()), float(-pnl[pnl < 0].sum())
    if ledger.empty:
        drawdown = 0.0
    else:
        equity = ledger.groupby("exit_datetime", sort=True)["pnl"].sum().cumsum()
        drawdown = float((equity - equity.cummax().clip(lower=0)).min())
    summary = {
        "scenario": label,
        "qualified_signals": int(len(candidates)),
        "executed_trades": int(len(ledger)),
        "signals_skipped_at_three_position_cap": int(len(candidates) - len(ledger)),
        "active_trade_days": int(pd.to_datetime(ledger["signal_datetime"]).dt.date.nunique()) if len(ledger) else 0,
        "gross_pnl_inr": float(pnl.sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "profit_factor": gains / losses if losses else None,
        "maximum_drawdown_inr": drawdown,
    }
    return ledger, summary


def main() -> None:
    trading.TARGET_INR = 1500.0
    trading.STOP_INR = -750.0
    predictions = rebuild_predictions()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT / "three_rule_predictions.csv.gz", index=False, compression="gzip")
    prices = trading.load_relevant_prices(predictions)
    outcomes = fast.precompute_outcomes(predictions, prices)
    joined = outcomes.merge(
        predictions[[
            "datetime",
            "pred_convergence_probability_classifier",
            "pred_strategy_rule_iv_above_rv_iv_declines_to_rv",
            "pred_strategy_rule_iv_rv_within_half_point",
            "pred_strategy_rule_iv_below_rv_fast_rv_decline",
            "pred_no120_gate_iv_minus_rv",
        ]],
        left_on="signal_datetime",
        right_on="datetime",
        how="left",
        validate="one_to_one",
    )
    near = joined["pred_strategy_rule_iv_rv_within_half_point"].astype(bool)
    above = (~near) & joined["pred_strategy_rule_iv_above_rv_iv_declines_to_rv"].astype(bool)
    below = (~near) & joined["pred_strategy_rule_iv_below_rv_fast_rv_decline"].astype(bool)
    joined["condition"] = np.select(
        [near, above, below],
        ["IV_current_near_RV_current", "IV_current_above_RV_current", "IV_current_below_RV_current"],
        default="NO_THREE_RULE_SIGNAL",
    )
    union = near | above | below
    confidence = joined["pred_convergence_probability_classifier"].astype(float) > OPTIMIZED_CONFIDENCE

    summaries = []
    ledgers = {}
    for condition, mask in (
        ("IV_current_above_RV_current", above),
        ("IV_current_near_RV_current", near),
        ("IV_current_below_RV_current", below),
        ("aggregated_three_conditions", union),
        ("aggregated_three_conditions_and_confidence_gt_80.0536pct", union & confidence),
    ):
        ledger, summary = select_with_cap(joined, mask, condition)
        ledger.to_csv(OUTPUT / f"trades_{condition}.csv", index=False)
        summaries.append(summary)
        ledgers[condition] = ledger

    aggregate = ledgers["aggregated_three_conditions"]
    attribution = (
        aggregate.groupby("condition")["pnl"].agg(["count", "sum"]).reset_index().rename(
            columns={"count": "executed_trades", "sum": "gross_pnl_inr"}
        ).to_dict(orient="records")
    )
    payload = {
        "evaluation_window": {"start": trading.START_DATE, "end": trading.END_DATE},
        "eligible_trading_days": int(predictions.loc[predictions["base_eligible"], "datetime"].dt.date.nunique()),
        "risk": {
            "profit_target_inr": trading.TARGET_INR,
            "stop_loss_inr": trading.STOP_INR,
            "maximum_open_positions": 3,
            "lot_size": trading.LOT_SIZE,
            "maximum_holding_minutes": trading.HORIZON_MINUTES,
        },
        "condition_definition": {
            "near_equal": "abs(IV_current - RV_current) <= 0.005; assigned first to keep branches mutually exclusive",
            "above": "IV_current > RV_current plus predicted IV decline and movement toward predicted RV",
            "below": "IV_current < RV_current plus no-120m sharp-RV-decline model signal",
        },
        "summaries": summaries,
        "aggregated_trade_pnl_attribution": attribution,
        "warning": "The three-rule output is diagnostic in the current live engine; this is a requested counterfactual backtest.",
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
