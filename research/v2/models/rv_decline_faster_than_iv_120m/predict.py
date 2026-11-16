from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the 120-minute sharp RV-decline event")
    parser.add_argument("input_csv")
    parser.add_argument("--model", default="artifacts/winner.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()

    bundle = joblib.load(Path(args.model))
    frame = pd.read_csv(args.input_csv, low_memory=False)
    missing = sorted(set(bundle["features"]) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required features; first missing columns: {missing[:10]}")
    probability = bundle["model"].predict_proba(numeric_matrix(frame, bundle["features"]))[:, 1]
    output = frame[[column for column in ("datetime", "entry_strike", "expiry") if column in frame]].copy()
    output["rv_declines_sharper_than_iv_probability_120m"] = probability
    output["rv_declines_sharper_than_iv_signal_120m"] = probability >= bundle["decision_threshold"]
    output.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

