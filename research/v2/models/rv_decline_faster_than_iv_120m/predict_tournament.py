from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from tournament import regression_event_score
from train import numeric_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the selected classifier or two-target regression model")
    parser.add_argument("input_csv")
    parser.add_argument("--model", default="artifacts_tournament/winner.joblib")
    parser.add_argument("--output", default="tournament_predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(Path(args.model))
    frame = pd.read_csv(args.input_csv, low_memory=False)
    missing = sorted(set(bundle["features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required features; first missing columns: {missing[:10]}")
    matrix = numeric_matrix(frame, bundle["features"])
    output = frame[[column for column in ("datetime", "entry_strike", "expiry") if column in frame]].copy()

    if bundle["model_type"] == "direct_classifier":
        score = bundle["model"].predict_proba(matrix)[:, 1]
        output["model_probability"] = score
    else:
        component_prediction = bundle["model"].predict(matrix)
        target = bundle["target_definition"]
        score = regression_event_score(
            component_prediction,
            float(target["minimum_rv_drop"]),
            float(target["minimum_rv_drop_beyond_iv"]),
        )
        output["predicted_rv_change_120m"] = component_prediction[:, 0]
        output["predicted_iv_change_120m"] = component_prediction[:, 1]
        output["event_margin_score"] = score
    output["rv_declines_sharper_than_iv_signal_120m"] = score >= bundle["threshold"]
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
