from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from train import TARGETS, future_from_log_ratio, load_config, score_predictions


BASE = Path(__file__).resolve().parent
SOURCE = BASE / "artifacts"
OUTPUT = BASE / "artifacts_inference_60m"


def main() -> None:
    config = load_config(str(BASE / "config.yaml"))
    manifest = json.loads(Path(config["data"]["feature_manifest_json"]).read_text(encoding="utf-8"))
    features = [
        column
        for column in manifest["feature_columns"]
        if not (column.endswith("120m") or "_120m_" in column)
    ]
    if len(features) != 143:
        raise AssertionError(f"Expected 143 one-hour features, found {len(features)}")

    frame = pd.read_csv(
        config["data"]["features_csv"], usecols=["row_id", *features], low_memory=False
    ).replace([np.inf, -np.inf], np.nan)
    targets = pd.read_csv(
        config["data"]["targets_csv"], parse_dates=["datetime", "exit_datetime"], low_memory=False
    )
    data = targets.merge(frame, on="row_id", validate="one_to_one", suffixes=("_audit", ""))
    data = data.sort_values("datetime").reset_index(drop=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "feature_count": len(features),
        "maximum_history_minutes": 60,
        "hyperparameters": "previously validation-selected; estimators refit on train plus validation",
        "sides": {},
    }
    clip = float(config["training"]["prediction_clip_log_ratio"])
    for side, (current_column, future_column, _change_column) in TARGETS.items():
        side_data = data[(data[current_column] > 0) & (data[future_column] > 0)].copy()
        side_data["model_target"] = np.log(side_data[future_column] / side_data[current_column])
        development = side_data[side_data["split"].isin(["train", "validation"])].copy()
        test = side_data[side_data["split"].eq("test")].copy()
        side_report: dict[str, object] = {}

        point_name = f"{side}_hist_gradient_boosting.joblib"
        point_payload = joblib.load(SOURCE / point_name)
        point_model = clone(point_payload["estimator"])
        point_model.fit(development[features], development["model_target"])
        point_future = future_from_log_ratio(
            test[current_column].to_numpy(), point_model.predict(test[features]), clip
        )
        point_metrics = score_predictions(
            test[future_column].to_numpy(), test[current_column].to_numpy(), point_future
        )
        joblib.dump(
            {
                "estimator": point_model,
                "features": features,
                "side": side,
                "target_transform": "log_ratio",
                "maximum_history_minutes": 60,
                "test_metrics": point_metrics,
            },
            OUTPUT / point_name,
        )
        side_report["hist_gradient_boosting"] = point_metrics

        quantile_report = {}
        for quantile in (10, 50, 90):
            name = f"{side}_lightgbm_q{quantile:02d}.joblib"
            payload = joblib.load(SOURCE / name)
            model = clone(payload["estimator"])
            model.fit(development[features], development["model_target"])
            future = future_from_log_ratio(
                test[current_column].to_numpy(), model.predict(test[features]), clip
            )
            metrics = score_predictions(
                test[future_column].to_numpy(), test[current_column].to_numpy(), future
            )
            joblib.dump(
                {
                    "estimator": model,
                    "features": features,
                    "side": side,
                    "target_transform": "log_ratio",
                    "quantile": quantile / 100.0,
                    "maximum_history_minutes": 60,
                    "test_metrics": metrics,
                },
                OUTPUT / name,
            )
            quantile_report[f"q{quantile:02d}"] = metrics
        side_report["quantiles"] = quantile_report
        report["sides"][side] = side_report
        print(
            json.dumps(
                {
                    "side": side,
                    "features": len(features),
                    "hgb_test_mae": point_metrics["mae"],
                }
            ),
            flush=True,
        )

    (OUTPUT / "inference_retraining_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({"artifacts": str(OUTPUT), "feature_count": len(features)}, indent=2))


if __name__ == "__main__":
    main()
