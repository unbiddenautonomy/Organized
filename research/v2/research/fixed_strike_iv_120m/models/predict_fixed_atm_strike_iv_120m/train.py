from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import ParameterSampler

from model_suite import ResidualHybrid, har_features, model_specs


TARGETS = {
    "call": ("entry_iv_call", "label_exit_iv_call", "label_iv_change_call"),
    "put": ("entry_iv_put", "label_exit_iv_put", "label_iv_change_put"),
}


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    for key in ("features_csv", "targets_csv", "feature_manifest_json"):
        config["data"][key] = str((config_path.parent / config["data"][key]).resolve())
    config["output"]["artifacts_dir"] = str((config_path.parent / config["output"]["artifacts_dir"]).resolve())
    return config


def future_from_log_ratio(current: np.ndarray, prediction: np.ndarray, clip: float) -> np.ndarray:
    return current * np.exp(np.clip(prediction, -clip, clip))


def score_predictions(actual_future: np.ndarray, current: np.ndarray, predicted_future: np.ndarray) -> dict:
    actual_change = actual_future - current
    predicted_change = predicted_future - current
    actual_direction = (actual_change > 0).astype(int)
    predicted_direction = (predicted_change > 0).astype(int)
    return {
        "mae": float(mean_absolute_error(actual_future, predicted_future)),
        "mae_vol_points": float(mean_absolute_error(actual_future, predicted_future) * 100),
        "rmse": float(mean_squared_error(actual_future, predicted_future) ** 0.5),
        "r2": float(r2_score(actual_future, predicted_future)),
        "direction_accuracy": float(np.mean(actual_direction == predicted_direction)),
        "balanced_direction_accuracy": float(balanced_accuracy_score(actual_direction, predicted_direction)),
        "correlation": float(np.corrcoef(actual_future, predicted_future)[0, 1]),
    }


def purged_folds(frame: pd.DataFrame, splits: int):
    n = len(frame)
    start_fraction = 0.5
    edges = np.linspace(start_fraction, 1.0, splits + 1)
    for index in range(splits):
        validation_start_pos = int(n * edges[index])
        validation_end_pos = int(n * edges[index + 1])
        validation_start = frame.iloc[validation_start_pos]["datetime"]
        fit = frame[(frame["datetime"] < validation_start) & (frame["exit_datetime"] < validation_start)]
        validation = frame.iloc[validation_start_pos:validation_end_pos]
        if len(fit) and len(validation):
            yield fit, validation


def sample_chronologically(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def tune_family(name, spec, train, feature_columns, har_columns, target_column, current_column, future_column, settings):
    scope = har_columns if spec.feature_scope == "har" else feature_columns
    sampled = sample_chronologically(train, int(settings["tuning_max_rows"]))
    rows = []
    candidates = list(ParameterSampler(spec.parameter_space, n_iter=int(settings["search_iterations"]),
                                       random_state=int(settings["random_seed"])))
    for candidate_index, params in enumerate(candidates):
        fold_metrics = []
        for fit, validation in purged_folds(sampled, int(settings["cv_splits"])):
            estimator = clone(spec.estimator).set_params(**params)
            estimator.fit(fit[scope], fit[target_column])
            raw_prediction = estimator.predict(validation[scope])
            predicted_future = future_from_log_ratio(validation[current_column].to_numpy(), raw_prediction,
                                                      float(settings["prediction_clip_log_ratio"]))
            fold_metrics.append(mean_absolute_error(validation[future_column], predicted_future))
        rows.append({"family": name, "candidate": candidate_index, "params": params,
                     "cv_mae": float(np.mean(fold_metrics)), "cv_fold_mae": fold_metrics})
    return min(rows, key=lambda item: item["cv_mae"]), rows


def fit_hybrid(train, validation, test, feature_columns, har_columns, target_column, current_column, future_column,
               best_ridge_params, best_lgb_params, clip):
    from lightgbm import LGBMRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    base = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", Ridge(**{key.replace("model__", ""): value for key, value in best_ridge_params.items()}))])
    residual_params = dict(best_lgb_params)
    residual_params.update({"objective": "huber", "verbosity": -1, "random_state": 42, "n_jobs": -1})
    hybrid = ResidualHybrid(base, LGBMRegressor(**residual_params), har_columns, feature_columns)
    hybrid.fit(train[feature_columns], train[target_column])
    validation_raw = hybrid.predict(validation[feature_columns])
    validation_future = future_from_log_ratio(validation[current_column].to_numpy(), validation_raw, clip)
    validation_metrics = score_predictions(validation[future_column].to_numpy(), validation[current_column].to_numpy(), validation_future)
    development = pd.concat([train, validation]).sort_values("datetime")
    hybrid.fit(development[feature_columns], development[target_column])
    test_raw = hybrid.predict(test[feature_columns])
    test_future = future_from_log_ratio(test[current_column].to_numpy(), test_raw, clip)
    return hybrid, validation_metrics, test_future


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exclude-over-60m", action="store_true")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["training"]
    exclude_over_60m = args.exclude_over_60m or int(settings.get("exclude_features_over_minutes", 0)) == 60
    artifacts = Path(args.artifacts_dir).resolve() if args.artifacts_dir else Path(config["output"]["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(config["data"]["feature_manifest_json"]).read_text(encoding="utf-8"))
    feature_columns = manifest["feature_columns"]
    if exclude_over_60m:
        feature_columns = [
            column for column in feature_columns
            if not (column.endswith("120m") or "_120m_" in column)
        ]
    forbidden = [column for column in feature_columns if column.startswith(("exit_", "label_", "quality_"))]
    if forbidden:
        raise AssertionError(f"Leakage columns in manifest: {forbidden}")
    features = pd.read_csv(config["data"]["features_csv"]).replace([np.inf, -np.inf], np.nan)
    targets = pd.read_csv(config["data"]["targets_csv"], parse_dates=["datetime", "exit_datetime"])
    # Audit metadata intentionally repeats entry_strike/atm_strike. Keep the
    # manifest-approved feature names unsuffixed and suffix only target copies.
    data = targets.merge(features, on="row_id", validate="one_to_one", suffixes=("_audit", ""))
    data = data.sort_values("datetime").reset_index(drop=True)
    har_columns = har_features(feature_columns)
    specs = model_specs(int(settings["random_seed"]), int(settings["n_jobs"]))
    leaderboard = []
    tuning_history = {}
    predictions = data.loc[data["split"] == "test", ["row_id", "datetime", "exit_datetime", "entry_strike"]].copy()

    for side, (current_column, future_column, change_column) in TARGETS.items():
        side_data = data[(data[current_column] > 0) & (data[future_column] > 0)].copy()
        side_data["model_target"] = np.log(side_data[future_column] / side_data[current_column])
        train = side_data[side_data["split"] == "train"]
        validation = side_data[side_data["split"] == "validation"]
        test = side_data[side_data["split"] == "test"]
        predictions = predictions.merge(
            test[["row_id", current_column, future_column, change_column]], on="row_id", how="left", validate="one_to_one"
        )
        family_best = {}
        for family, spec in specs.items():
            started = time.time()
            best, history = tune_family(family, spec, train, feature_columns, har_columns, "model_target",
                                        current_column, future_column, settings)
            tuning_history[f"{side}:{family}"] = history
            scope = har_columns if spec.feature_scope == "har" else feature_columns
            estimator = clone(spec.estimator).set_params(**best["params"])
            estimator.fit(train[scope], train["model_target"])
            validation_raw = estimator.predict(validation[scope])
            validation_future = future_from_log_ratio(validation[current_column].to_numpy(), validation_raw,
                                                       float(settings["prediction_clip_log_ratio"]))
            validation_metrics = score_predictions(validation[future_column].to_numpy(),
                                                   validation[current_column].to_numpy(), validation_future)
            development = pd.concat([train, validation]).sort_values("datetime")
            estimator.fit(development[scope], development["model_target"])
            test_raw = estimator.predict(test[scope])
            test_future = future_from_log_ratio(test[current_column].to_numpy(), test_raw,
                                                float(settings["prediction_clip_log_ratio"]))
            test_metrics = score_predictions(test[future_column].to_numpy(), test[current_column].to_numpy(), test_future)
            family_best[family] = {"params": best["params"], "validation": validation_metrics,
                                   "test": test_metrics, "scope": spec.feature_scope}
            joblib.dump({"estimator": estimator, "features": scope, "side": side, "target_transform": "log_ratio"},
                        artifacts / f"{side}_{family}.joblib")
            predictions.loc[predictions["row_id"].isin(test["row_id"]), f"pred_{side}_{family}"] = test_future
            leaderboard.append({"side": side, "family": family, "cv_mae": best["cv_mae"],
                                "validation_mae": validation_metrics["mae"], "test_mae": test_metrics["mae"],
                                "test_r2": test_metrics["r2"], "test_balanced_direction_accuracy": test_metrics["balanced_direction_accuracy"],
                                "seconds": time.time() - started})
            print(side, family, "validation_mae", validation_metrics["mae"], "test_mae", test_metrics["mae"], flush=True)

        # Residual hybrid uses the tuned HAR Ridge and LightGBM components.
        hybrid, hybrid_validation, hybrid_test = fit_hybrid(
            train, validation, test, feature_columns, har_columns, "model_target", current_column, future_column,
            family_best["har_ridge"]["params"], family_best["lightgbm"]["params"],
            float(settings["prediction_clip_log_ratio"]),
        )
        hybrid_test_metrics = score_predictions(test[future_column].to_numpy(), test[current_column].to_numpy(), hybrid_test)
        joblib.dump({"estimator": hybrid, "features": feature_columns, "side": side, "target_transform": "log_ratio"},
                    artifacts / f"{side}_har_lightgbm_hybrid.joblib")
        predictions.loc[predictions["row_id"].isin(test["row_id"]), f"pred_{side}_har_lightgbm_hybrid"] = hybrid_test
        family_best["har_lightgbm_hybrid"] = {"validation": hybrid_validation, "test": hybrid_test_metrics}
        leaderboard.append({"side": side, "family": "har_lightgbm_hybrid", "cv_mae": np.nan,
                            "validation_mae": hybrid_validation["mae"], "test_mae": hybrid_test_metrics["mae"],
                            "test_r2": hybrid_test_metrics["r2"],
                            "test_balanced_direction_accuracy": hybrid_test_metrics["balanced_direction_accuracy"], "seconds": np.nan})

        # Family winner is selected on validation only, before consulting test metrics.
        winner = min(family_best, key=lambda family: family_best[family]["validation"]["mae"])
        family_best["winner_selected_on_validation"] = winner

        # Conditional quantiles use the tuned LightGBM structure.
        from lightgbm import LGBMRegressor
        development = pd.concat([train, validation]).sort_values("datetime")
        lgb_params = dict(family_best["lightgbm"]["params"])
        for quantile in (0.10, 0.50, 0.90):
            quantile_model = LGBMRegressor(**lgb_params, objective="quantile", alpha=quantile,
                                          verbosity=-1, random_state=int(settings["random_seed"]), n_jobs=int(settings["n_jobs"]))
            quantile_model.fit(development[feature_columns], development["model_target"])
            raw = quantile_model.predict(test[feature_columns])
            future = future_from_log_ratio(test[current_column].to_numpy(), raw,
                                           float(settings["prediction_clip_log_ratio"]))
            predictions.loc[predictions["row_id"].isin(test["row_id"]), f"pred_{side}_q{int(quantile*100):02d}"] = future
            joblib.dump({"estimator": quantile_model, "features": feature_columns, "side": side,
                         "target_transform": "log_ratio", "quantile": quantile},
                        artifacts / f"{side}_lightgbm_q{int(quantile*100):02d}.joblib")

        actual_future = test[future_column].to_numpy()
        q10 = predictions.loc[predictions["row_id"].isin(test["row_id"]), f"pred_{side}_q10"].to_numpy()
        q90 = predictions.loc[predictions["row_id"].isin(test["row_id"]), f"pred_{side}_q90"].to_numpy()
        family_best["quantile_interval"] = {
            "nominal_coverage": 0.80,
            "test_coverage": float(np.mean((actual_future >= q10) & (actual_future <= q90))),
            "test_mean_width": float(np.mean(q90 - q10)),
            "crossing_rate": float(np.mean(q10 > q90)),
        }

        # Honest persistence and median-ratio baselines.
        median_ratio = float(train["model_target"].median())
        baseline_zero = test[current_column].to_numpy()
        baseline_median = future_from_log_ratio(test[current_column].to_numpy(), np.repeat(median_ratio, len(test)),
                                                float(settings["prediction_clip_log_ratio"]))
        family_best["persistence_baseline"] = {"test": score_predictions(test[future_column].to_numpy(), test[current_column].to_numpy(), baseline_zero)}
        family_best["median_ratio_baseline"] = {"test": score_predictions(test[future_column].to_numpy(), test[current_column].to_numpy(), baseline_median)}
        (artifacts / f"{side}_metrics.json").write_text(json.dumps(family_best, indent=2, default=str), encoding="utf-8")

    leaderboard_frame = pd.DataFrame(leaderboard).sort_values(["side", "validation_mae"])
    leaderboard_frame.to_csv(artifacts / "leaderboard.csv", index=False)
    predictions.to_csv(artifacts / "test_predictions.csv", index=False)
    (artifacts / "tuning_history.json").write_text(json.dumps(tuning_history, indent=2, default=str), encoding="utf-8")

    lines = ["# Fixed-strike IV model results", "", "Models predict the exact entry-ATM strike IV 120 minutes later.", ""]
    for side in TARGETS:
        metrics = json.loads((artifacts / f"{side}_metrics.json").read_text(encoding="utf-8"))
        winner = metrics["winner_selected_on_validation"]
        lines.extend([f"## {side.title()}", "", f"Validation-selected winner: **{winner}**.",
                      f"Test MAE: {metrics[winner]['test']['mae']:.6f} ({metrics[winner]['test']['mae_vol_points']:.3f} vol points).",
                      f"Test R²: {metrics[winner]['test']['r2']:.4f}.",
                      f"Persistence baseline MAE: {metrics['persistence_baseline']['test']['mae']:.6f}.",
                      f"Median-ratio baseline MAE: {metrics['median_ratio_baseline']['test']['mae']:.6f}.",
                      f"Nominal 80% interval coverage: {metrics['quantile_interval']['test_coverage']:.2%}.", ""])
    (artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(leaderboard_frame.to_string(index=False))


if __name__ == "__main__":
    main()

