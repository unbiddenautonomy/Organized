from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(frame: pd.DataFrame, model_path: str | Path = "model.joblib") -> pd.DataFrame:
    artifact = joblib.load(model_path)
    missing = [name for name in artifact["features"] if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    values = frame[artifact["features"]]
    probability = artifact["classifier"].predict_proba(values)[:, 1]
    predicted_edge = artifact["regressor"].predict(values)
    enter = (
        (probability >= float(artifact["probability_threshold"]))
        & (predicted_edge >= float(artifact["predicted_net_edge_threshold"]))
    )
    return pd.DataFrame({
        "win_probability": probability,
        "predicted_net_edge_points": predicted_edge,
        "enter_trade": enter,
    }, index=frame.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(pd.read_csv(args.features_csv), args.model).to_csv(args.output, index=False)
