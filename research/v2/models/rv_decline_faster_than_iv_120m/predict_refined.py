from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from refine import add_engineered_features, model_score
from train import numeric_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the refined 120-minute relative RV-decline model")
    parser.add_argument("input_csv")
    parser.add_argument("--model", default="artifacts_refined/winner.joblib")
    parser.add_argument("--output", default="refined_predictions.csv")
    args = parser.parse_args()
    bundle = joblib.load(Path(args.model))
    frame = pd.read_csv(args.input_csv, low_memory=False)
    frame, _ = add_engineered_features(frame)
    missing = sorted(set(bundle["features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required features; first missing columns: {missing[:10]}")
    score, components = model_score(
        bundle["model"], bundle["model_type"], numeric_matrix(frame, bundle["features"]), bundle["target_definition"]
    )
    output = frame[[column for column in ("datetime", "entry_strike", "expiry") if column in frame]].copy()
    output["model_score"] = score
    output["rv_declines_sharper_than_iv_signal_120m"] = score >= bundle["threshold"]
    if components is not None:
        output["predicted_rv_change_120m"] = components[:, 0]
        output["predicted_iv_change_120m"] = components[:, 1]
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
