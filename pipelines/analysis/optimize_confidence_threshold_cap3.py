from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_user_csv_current_paper_rules as engine


OUTPUT = Path(__file__).resolve().parent / 'analysis_outputs' / 'confidence_threshold_optimization_cap3'


def evaluate(
    threshold: float, predictions: pd.DataFrame, prices: pd.DataFrame
) -> dict[str, float | int | str | None]:
    engine.CONFIDENCE_THRESHOLD = float(threshold)
    ledger, summary = engine.simulate(
        predictions, prices, use_confidence=True, maximum_open_positions=3
    )
    summary["confidence_threshold_strict_gt"] = float(threshold)
    summary["active_trade_days"] = (
        int(pd.to_datetime(ledger["entry_datetime"]).dt.date.nunique())
        if not ledger.empty else 0
    )
    return summary


def main() -> None:
    engine.TARGET_INR = 1500.0
    engine.STOP_INR = -750.0
    engine.MAX_OPEN_POSITIONS = 3
    predictions = engine.load_established_test_predictions()
    predictions["Date"] = predictions["datetime"].dt.strftime("%Y-%m-%d")
    prices = engine.load_relevant_prices(predictions)

    evaluated: dict[float, dict] = {}
    for threshold in np.arange(0.0, 0.951, 0.01):
        key = round(float(threshold), 12)
        evaluated[key] = evaluate(key, predictions, prices)

    coarse_best = max(
        evaluated.values(), key=lambda row: (float(row["gross_pnl_inr"]), float(row["confidence_threshold_strict_gt"]))
    )
    center = float(coarse_best["confidence_threshold_strict_gt"])
    low, high = max(0.0, center - 0.02), min(1.0, center + 0.02)
    probabilities = predictions.loc[
        predictions["base_eligible"], "pred_convergence_probability_classifier"
    ].astype(float)
    exact_candidates = sorted(set(probabilities.loc[probabilities.between(low, high)].tolist()))
    for threshold in exact_candidates:
        key = round(float(threshold), 12)
        if key not in evaluated:
            evaluated[key] = evaluate(float(threshold), predictions, prices)

    results = pd.DataFrame(evaluated.values()).sort_values(
        ["gross_pnl_inr", "confidence_threshold_strict_gt"], ascending=[False, False]
    ).reset_index(drop=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "threshold_results.csv", index=False)
    top = results.head(20).copy()
    top.to_csv(OUTPUT / "top_20_thresholds.csv", index=False)
    best = top.iloc[0].to_dict()
    payload = {
        "objective": "maximize historical gross P&L with maximum three concurrent positions",
        "evaluation_status": "threshold tuned on a reused held-out diagnostic; requires new out-of-sample validation",
        "evaluation_window": {"start": engine.START_DATE, "end": engine.END_DATE},
        "eligible_trading_days": int(predictions.loc[predictions["base_eligible"], "datetime"].dt.date.nunique()),
        "risk": {
            "profit_target_inr": engine.TARGET_INR,
            "stop_loss_inr": engine.STOP_INR,
            "maximum_open_positions": 3,
            "lot_size": engine.LOT_SIZE,
            "maximum_holding_minutes": engine.HORIZON_MINUTES,
        },
        "thresholds_evaluated": int(len(results)),
        "best": best,
        "top_10": top.head(10).to_dict(orient="records"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
