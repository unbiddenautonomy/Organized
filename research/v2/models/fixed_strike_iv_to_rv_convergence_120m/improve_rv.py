from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "research" / "fixed_strike_iv_rv_convergence"
sys.path.insert(0, str(PROJECT))
from feature_pipeline import assert_leakage_safe_features, purged_chronological_split  # noqa: E402


TARGET = "target_rv_forward_120m"


def qlike(actual_vol, predicted_vol) -> float:
    actual_var = np.maximum(np.asarray(actual_vol, dtype=float) ** 2, 1e-12)
    predicted_var = np.maximum(np.asarray(predicted_vol, dtype=float) ** 2, 1e-12)
    ratio = actual_var / predicted_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def metrics(actual, predicted) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    error = predicted - actual
    return {
        "qlike": qlike(actual, predicted),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mean_prediction": float(np.mean(predicted)),
        "mean_actual": float(np.mean(actual)),
    }


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("feature_csv", "feature_manifest"):
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    config["output"]["artifact_directory"] = str(
        (path.parent / config["output"]["artifact_directory"]).resolve()
    )
    return config


def matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def add_rv_features(frame: pd.DataFrame, base_features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    engineered: list[str] = []
    for name in base_features:
        if name.startswith(("rv_trailing_", "rq_trailing_")):
            new_name = f"log_{name}"
            result[new_name] = np.log(np.maximum(pd.to_numeric(result[name], errors="coerce"), 1e-8))
            engineered.append(new_name)
        elif name.startswith("spot_return_"):
            new_name = f"abs_{name}"
            result[new_name] = pd.to_numeric(result[name], errors="coerce").abs()
            engineered.append(new_name)
    if "entry_iv_weighted" in result and "rv_trailing_120m" in result:
        result["iv_minus_rv_120m"] = result["entry_iv_weighted"] - result["rv_trailing_120m"]
        result["iv_to_rv_120m_ratio"] = result["entry_iv_weighted"] / result["rv_trailing_120m"].clip(lower=1e-8)
        engineered += ["iv_minus_rv_120m", "iv_to_rv_120m_ratio"]
    return result, base_features + engineered


def cap_rows(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    # Preserve the entire history rather than retaining only the most recent regime.
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def expanding_folds(frame: pd.DataFrame, splits: int):
    ordered = frame.sort_values("datetime")
    stamps = np.sort(ordered["datetime"].unique())
    boundaries = np.linspace(0.40, 1.0, splits + 1)
    for index, (left, right) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        start = pd.Timestamp(stamps[min(int(len(stamps) * left), len(stamps) - 1)])
        end = pd.Timestamp(stamps[min(int(len(stamps) * right), len(stamps) - 1)])
        fit = ordered[(ordered.datetime < start) & (ordered.exit_datetime < start)]
        # Half-open intervals avoid scoring a boundary timestamp twice.
        score = ordered[(ordered.datetime >= start) & (ordered.datetime < end)] if index < splits - 1 else ordered[(ordered.datetime >= start) & (ordered.datetime <= end)]
        if len(fit) and len(score):
            yield fit, score


def model_specs(seed: int, jobs: int, quick: bool):
    har_features = None
    ridge = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", RobustScaler()),
        ("model", Ridge(solver="lsqr")),
    ])
    specs = {
        "log_har": [
            ({"model__alpha": alpha}, ridge)
            for alpha in ([0.1, 10.0] if quick else [0.01, 0.1, 1.0, 10.0, 100.0])
        ],
        "lightgbm": [],
        "xgboost": [],
    }
    lgb_grid = [
        (300, 0.03, 15, 100, 0.7, 0.0, 0.0, "regression_l1"),
        (500, 0.025, 31, 80, 0.8, 0.1, 0.0, "regression_l1"),
        (700, 0.02, 31, 120, 0.7, 1.0, 0.1, "huber"),
        (500, 0.025, 63, 120, 0.65, 1.0, 0.1, "huber"),
        (800, 0.015, 31, 160, 0.8, 3.0, 0.2, "regression"),
        (600, 0.02, 63, 200, 0.6, 3.0, 0.2, "regression"),
        (450, 0.03, 15, 160, 0.9, 0.1, 0.1, "fair"),
        (700, 0.015, 63, 100, 0.75, 5.0, 0.3, "fair"),
    ]
    if quick:
        lgb_grid = lgb_grid[:3]
    for n, lr, leaves, child, cols, l1, l2, objective in lgb_grid:
        params = dict(n_estimators=n, learning_rate=lr, num_leaves=leaves,
                      min_child_samples=child, colsample_bytree=cols,
                      reg_alpha=l1, reg_lambda=l2, objective=objective)
        specs["lightgbm"].append((params, lgb.LGBMRegressor(
            random_state=seed, n_jobs=jobs, verbosity=-1, subsample_freq=1,
        )))
    xgb_grid = [
        (400, 0.03, 3, 20, 0.8, 0.8, 1.0, 0.0, "reg:squarederror"),
        (600, 0.02, 4, 30, 0.8, 0.7, 3.0, 0.0, "reg:squarederror"),
        (500, 0.025, 5, 40, 0.75, 0.7, 5.0, 0.1, "reg:squarederror"),
        (700, 0.015, 3, 50, 0.9, 0.9, 5.0, 0.1, "reg:squarederror"),
        (500, 0.025, 4, 30, 0.8, 0.8, 3.0, 0.1, "reg:pseudohubererror"),
        (700, 0.015, 5, 50, 0.75, 0.7, 5.0, 0.2, "reg:pseudohubererror"),
    ]
    if quick:
        xgb_grid = xgb_grid[:2]
    for n, lr, depth, child, rows, cols, l2, l1, objective in xgb_grid:
        params = dict(n_estimators=n, learning_rate=lr, max_depth=depth,
                      min_child_weight=child, subsample=rows, colsample_bytree=cols,
                      reg_lambda=l2, reg_alpha=l1, objective=objective)
        specs["xgboost"].append((params, XGBRegressor(
            random_state=seed, n_jobs=jobs, tree_method="hist", verbosity=0,
        )))
    return specs, har_features


def inverse_log_variance(prediction) -> np.ndarray:
    return np.sqrt(np.exp(np.clip(np.asarray(prediction, dtype=float), -20.0, 5.0)))


def calibration(actual, prediction) -> float:
    ratio = np.maximum(np.asarray(actual), 1e-8) ** 2 / np.maximum(np.asarray(prediction), 1e-8) ** 2
    return float(np.sqrt(np.mean(ratio)))


def tune_family(name, specs, train, features, settings):
    results = []
    selected_features = features
    if name == "log_har":
        selected_features = [f for f in features if f.startswith(("log_rv_trailing_", "log_rq_trailing_", "spot_return_", "abs_spot_return_")) or f in {
            "entry_iv_weighted", "surface_atm_iv_call", "surface_atm_iv_put",
            "day_of_week", "time_sin", "time_cos", "days_to_expiry",
            "iv_minus_rv_120m", "iv_to_rv_120m_ratio",
        }]
    for number, (params, estimator) in enumerate(specs, start=1):
        scores = []
        for fit, score in expanding_folds(train, int(settings["cv_splits"])):
            fit = cap_rows(fit[fit[TARGET].notna()], int(settings["max_training_rows"]))
            score = score[score[TARGET].notna()]
            model = clone(estimator).set_params(**params)
            model.fit(matrix(fit, selected_features), np.log(np.maximum(fit[TARGET].to_numpy(), 1e-8) ** 2))
            prediction = inverse_log_variance(model.predict(matrix(score, selected_features)))
            scores.append(qlike(score[TARGET], prediction))
        row = {"family": name, "candidate": number, "params": params, "fold_qlike": scores, "cv_qlike": float(np.mean(scores))}
        results.append(row)
        print(json.dumps({"tuned": row}), flush=True)
    return sorted(results, key=lambda row: row["cv_qlike"]), selected_features


def optimize_blend(actual, predictions: np.ndarray) -> np.ndarray:
    count = predictions.shape[1]
    result = minimize(
        lambda weights: qlike(actual, predictions @ weights),
        np.repeat(1.0 / count, count), method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 500, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Blend optimization failed: {result.message}")
    return result.x


def main():
    parser = argparse.ArgumentParser(description="Leakage-safe improvement study for the 120-minute RV forecast")
    parser.add_argument("--config", default="rv_improvement_config.yaml")
    parser.add_argument("--quick", action="store_true", help="Run a reduced candidate grid for a smoke test")
    parser.add_argument("--exclude-over-60m", action="store_true")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    config = load_config(Path(args.config).resolve())
    settings = config["training"]
    exclude_over_60m = args.exclude_over_60m or int(settings.get("exclude_features_over_minutes", 0)) == 60
    artifacts = Path(args.artifacts_dir).resolve() if args.artifacts_dir else Path(config["output"]["artifact_directory"])
    artifacts.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(config["data"]["feature_csv"], parse_dates=["datetime", "exit_datetime", "expiry"], low_memory=False)
    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    base_features = manifest["features"]
    assert_leakage_safe_features(base_features)
    frame, features = add_rv_features(frame, base_features)
    if exclude_over_60m:
        features = [name for name in features if not (name.endswith("120m") or "_120m_" in name)]
    split = purged_chronological_split(frame, float(settings["train_fraction"]), float(settings["validation_fraction"]), int(settings["horizon_minutes"]))
    train = split["train"].query("quality_entry_ok and target_rv_forward_120m == target_rv_forward_120m").copy()
    validation = split["validation"].query("quality_entry_ok and target_rv_forward_120m == target_rv_forward_120m").copy()
    test = split["test"].query("quality_entry_ok and target_rv_forward_120m == target_rv_forward_120m").copy()
    print(json.dumps({"rows": {"train": len(train), "validation": len(validation), "test": len(test)}, "features": len(features)}), flush=True)

    specs, _ = model_specs(int(settings["random_seed"]), int(settings["n_jobs"]), args.quick)
    searches, fitted = {}, {}
    validation_predictions, names = [], []
    for family, family_specs in specs.items():
        search, family_features = tune_family(family, family_specs, train, features, settings)
        searches[family] = search
        best = search[0]
        params, estimator = family_specs[best["candidate"] - 1]
        model = clone(estimator).set_params(**params)
        fit = cap_rows(train, int(settings["max_training_rows"]))
        model.fit(matrix(fit, family_features), np.log(np.maximum(fit[TARGET].to_numpy(), 1e-8) ** 2))
        raw = inverse_log_variance(model.predict(matrix(validation, family_features)))
        scale = calibration(validation[TARGET], raw)
        calibrated = raw * scale
        validation_predictions.append(calibrated)
        names.append(family)
        fitted[family] = {"model": model, "features": family_features, "params": params, "calibration": scale}
        best["validation_raw"] = metrics(validation[TARGET], raw)
        best["validation_calibrated"] = metrics(validation[TARGET], calibrated)

    prediction_matrix = np.column_stack(validation_predictions)
    weights = optimize_blend(validation[TARGET].to_numpy(), prediction_matrix)
    validation_blend = prediction_matrix @ weights

    development = pd.concat([train, validation]).sort_values("datetime")
    test_predictions = []
    for name in names:
        payload = fitted[name]
        final_model = clone(payload["model"])
        fit = cap_rows(development, int(settings["max_training_rows"]))
        final_model.fit(matrix(fit, payload["features"]), np.log(np.maximum(fit[TARGET].to_numpy(), 1e-8) ** 2))
        prediction = inverse_log_variance(final_model.predict(matrix(test, payload["features"]))) * payload["calibration"]
        payload["model"] = final_model
        payload["test_metrics"] = metrics(test[TARGET], prediction)
        test_predictions.append(prediction)
        joblib.dump(payload, artifacts / f"{name}.joblib")

    test_matrix = np.column_stack(test_predictions)
    test_blend = test_matrix @ weights
    persistence = test["rv_trailing_120m"].fillna(development[TARGET].median()).to_numpy()
    report = {
        "protocol": {
            "target": TARGET, "target_transform": "log_variance", "cv": "expanding_walk_forward",
            "held_out_test_used_for_tuning": False, "max_training_rows": int(settings["max_training_rows"]),
            "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test),
        },
        "searches": searches,
        "blend": {"model_names": names, "weights": weights.tolist(), "validation_metrics": metrics(validation[TARGET], validation_blend)},
        "test": {
            "persistence": metrics(test[TARGET], persistence),
            **{name: fitted[name]["test_metrics"] for name in names},
            "blend": metrics(test[TARGET], test_blend),
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    output = test[["datetime", "expiry", "dte_flag", TARGET]].copy()
    output["rv_trailing_120m"] = persistence
    for name, prediction in zip(names, test_predictions):
        output[f"rv_pred_120m_{name}"] = prediction
    output["rv_pred_120m"] = test_blend
    output.to_csv(artifacts / "held_out_rv_predictions.csv", index=False)
    lines = [
        "# Improved 120-minute RV forecast", "",
        "The held-out test block was not used for hyperparameter or blend selection.", "",
        "| Model | QLIKE | MAE | RMSE |", "|---|---:|---:|---:|",
    ]
    for name, values in report["test"].items():
        lines.append(f"| {name} | {values['qlike']:.6f} | {values['mae']:.6f} | {values['rmse']:.6f} |")
    lines += ["", "## Blend", "", *[f"- {name}: {weight:.4f}" for name, weight in zip(names, weights)]]
    (artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"artifacts": str(artifacts), "test": report["test"], "blend_weights": dict(zip(names, weights))}, indent=2), flush=True)


if __name__ == "__main__":
    main()
