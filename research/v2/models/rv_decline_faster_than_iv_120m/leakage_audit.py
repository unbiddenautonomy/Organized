from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from train import purged_chronological_split


FORBIDDEN_EXACT = {
    "exit_datetime", "matched_exit_timestamp", "spot_window_120m_complete",
    "quality_exact_exit", "quality_exit_iv_valid", "quality_exact_spot_window",
    "quality_same_strike", "quality_exact_120m_timestamp", "label_available",
    "training_row_eligible", "rv_prediction_provenance",
}
FORBIDDEN_PREFIXES = ("target_", "exit_fixed_", "exit_gap_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the refined model for temporal and feature leakage")
    parser.add_argument("--model", default="artifacts_refined/winner.joblib")
    parser.add_argument(
        "--historical-features",
        default="../../research/combined_atm_iv_rv_convergence_v2/artifacts/combined_atm_iv_convergence_v2_features.csv",
    )
    parser.add_argument("--output", default="artifacts_refined/leakage_audit.json")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = list(bundle["features"])
    forbidden = sorted(
        name for name in features
        if name in FORBIDDEN_EXACT or name.startswith(FORBIDDEN_PREFIXES)
    )
    frame = pd.read_csv(
        args.historical_features,
        usecols=["datetime", "exit_datetime", "training_row_eligible"],
        parse_dates=["datetime", "exit_datetime"],
    )
    frame = frame[frame["training_row_eligible"].fillna(False)].copy()
    split = purged_chronological_split(frame, 0.70, 0.15, 120)
    train, validation, test = split["train"], split["validation"], split["test"]
    validation_start = validation["datetime"].min()
    test_start = test["datetime"].min()

    checks = {
        "no_forbidden_model_features": len(forbidden) == 0,
        "train_labels_end_before_validation": bool((train["exit_datetime"] < validation_start).all()),
        "validation_labels_end_before_test": bool((validation["exit_datetime"] < test_start).all()),
        "no_train_validation_timestamp_overlap": len(set(train["datetime"]).intersection(validation["datetime"])) == 0,
        "no_validation_test_timestamp_overlap": len(set(validation["datetime"]).intersection(test["datetime"])) == 0,
        "saved_model_has_embedded_preprocessing": all(
            hasattr(model, "named_steps") and "imputer" in model.named_steps and "scale" in model.named_steps
            for model in (bundle["model"].rv_model_, bundle["model"].iv_model_)
        ),
    }
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "forbidden_features_found": forbidden,
        "model_feature_count": len(features),
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "split_boundaries": {
            "validation_start": str(validation_start), "test_start": str(test_start),
            "latest_train_label_end": str(train["exit_datetime"].max()),
            "latest_validation_label_end": str(validation["exit_datetime"].max()),
        },
        "evaluation_finding": (
            "The historical test period was reused across iterative experiments. It is retained as development data; "
            "a separate 2026 out-of-time dataset is required for the final evaluation."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
