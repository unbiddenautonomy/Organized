from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PREDICTIONS_CSV = ARTIFACTS / "test_inference_predictions_with_pnl.csv"
TRADES_CSV = ARTIFACTS / "leakage_safe_classifier_trades_current_costs_no_brokerage.csv"
SUMMARY_CSV = ARTIFACTS / "current_cost_no_brokerage_pnl_summary.csv"
OUTPUT_JSON = ARTIFACTS / "inference_test_predictions_and_costs.json"

SAFE_START = pd.Timestamp("2024-09-19 12:56:59")
SAFE_END = pd.Timestamp("2025-05-09 13:21:59")


def records(frame: pd.DataFrame) -> list[dict]:
    """Convert through pandas JSON so NaN/NaT become strict JSON null values."""
    return json.loads(frame.to_json(orient="records", date_format="iso"))


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS_CSV)
    prediction_times = pd.to_datetime(predictions["datetime"], errors="raise")
    safe_mask = prediction_times.between(SAFE_START, SAFE_END, inclusive="both")
    predictions = predictions.loc[safe_mask].copy()
    predictions["pred_spot_reference"] = (
        predictions["pred_spot_range_1sigma_low"]
        * predictions["pred_spot_range_1sigma_high"]
    ).pow(0.5)

    trades = pd.read_csv(TRADES_CSV)
    summary = pd.read_csv(SUMMARY_CSV)

    if len(predictions) != 7_423:
        raise ValueError(f"Expected 7,423 leakage-safe predictions, found {len(predictions):,}")
    if len(trades) != 324:
        raise ValueError(f"Expected 324 executed trades, found {len(trades):,}")
    if not (trades["current_brokerage"] == 0).all():
        raise ValueError("Current-cost trade ledger contains non-zero brokerage")

    extrema = []
    for column in predictions.columns:
        if not column.startswith("pred_") or not pd.api.types.is_numeric_dtype(
            predictions[column]
        ):
            continue
        valid = predictions[column].dropna()
        if valid.empty:
            continue
        minimum_index = valid.idxmin()
        maximum_index = valid.idxmax()
        extrema.append(
            {
                "prediction_field": column,
                "minimum": predictions.at[minimum_index, column],
                "minimum_datetime": predictions.at[minimum_index, "datetime"],
                "minimum_entry_strike": predictions.at[minimum_index, "entry_strike"],
                "maximum": predictions.at[maximum_index, column],
                "maximum_datetime": predictions.at[maximum_index, "datetime"],
                "maximum_entry_strike": predictions.at[maximum_index, "entry_strike"],
            }
        )

    payload = {
        "schema_version": "1.0",
        "generated_at": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(),
        "description": (
            "Leakage-safe fixed-strike 120-minute inference predictions and "
            "classifier-selected short-straddle trades with current costs."
        ),
        "test_window": {
            "start": str(SAFE_START),
            "end": str(SAFE_END),
            "prediction_rows": int(len(predictions)),
            "executed_trade_rows": int(len(trades)),
            "selection_rule": "pred_convergence_probability_classifier >= 0.5",
            "position_rule": "one open trade at a time; fixed entry strike",
            "exit_rule": "750 INR minute stop or exact 120-minute exit",
        },
        "current_cost_assumptions": {
            "as_of_date": "2026-10-17",
            "brokerage_inr_per_trade": 0.0,
            "stt_sell_premium_rate": 0.0015,
            "nse_options_transaction_rate": 0.0003553,
            "sebi_turnover_rate": 0.000001,
            "stamp_buy_premium_rate": 0.00003,
            "gst_rate": 0.18,
            "gst_basis": "NSE transaction charges plus SEBI charges; brokerage is zero",
            "additional_clearing_charge": 0.0,
            "rate_application": "Current rates applied uniformly to historical test trades",
        },
        "spot_prediction_interpretation": {
            "pred_spot_reference": "Spot at prediction entry, reconstructed as the geometric midpoint of the range",
            "pred_spot_range_1sigma_low_high": "Nondirectional one-standard-deviation terminal range",
            "pred_spot_range_90pct_low_high": "Nondirectional approximate 90% terminal range",
            "note": "The RV model predicts movement magnitude, not spot direction or a directional point target",
        },
        "prediction_extrema": records(pd.DataFrame(extrema)),
        "cost_and_pnl_summary": records(summary),
        "predictions": records(predictions),
        "executed_trades_with_current_costs": records(trades),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("\n")

    print(
        f"Wrote {OUTPUT_JSON} with {len(predictions):,} predictions and "
        f"{len(trades):,} costed trades ({OUTPUT_JSON.stat().st_size:,} bytes)."
    )


if __name__ == "__main__":
    main()
