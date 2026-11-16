from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from tournament import regression_event_score
from model_components import SeparateTargetRegressor
from train import (
    IV_CHANGE,
    RV_CURRENT,
    RV_FUTURE,
    TARGET,
    add_target,
    approved_features,
    load_config,
    non_overlapping_rows,
    numeric_matrix,
    purged_chronological_split,
)


RV_CHANGE = "target_rv_change_120m"
IV_CHANGE_COPY = "target_iv_change_120m"


def add_engineered_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = frame.copy()
    created = []

    def add(name: str, values):
        result[name] = values.replace([np.inf, -np.inf], np.nan) if hasattr(values, "replace") else values
        created.append(name)

    iv = pd.to_numeric(result.get("entry_atm_iv_combined"), errors="coerce")
    for horizon in (5, 15, 30, 60, 120):
        rv_name = f"rv_trailing_{horizon}m"
        if rv_name not in result:
            continue
        rv = pd.to_numeric(result[rv_name], errors="coerce")
        add(f"engineered_iv_minus_rv_{horizon}m", iv - rv)
        add(f"engineered_iv_to_rv_ratio_{horizon}m", iv / rv.where(rv.abs() > 1e-8))
        add(f"engineered_log_rv_{horizon}m", np.log(rv.clip(lower=1e-8)))
    add("engineered_log_entry_iv", np.log(iv.clip(lower=1e-8)))

    available = [h for h in (5, 15, 30, 60, 120) if f"rv_trailing_{h}m" in result]
    for short, long in zip(available[:-1], available[1:]):
        add(
            f"engineered_rv_term_slope_{short}m_{long}m",
            pd.to_numeric(result[f"rv_trailing_{short}m"], errors="coerce")
            - pd.to_numeric(result[f"rv_trailing_{long}m"], errors="coerce"),
        )
    if "entry_straddle_price" in result and "spot_price" in result:
        spot = pd.to_numeric(result["spot_price"], errors="coerce")
        add(
            "engineered_straddle_to_spot",
            pd.to_numeric(result["entry_straddle_price"], errors="coerce") / spot.where(spot.abs() > 1e-8),
        )
    if "call_oi" in result and "put_oi" in result:
        call_oi = pd.to_numeric(result["call_oi"], errors="coerce")
        put_oi = pd.to_numeric(result["put_oi"], errors="coerce")
        add("engineered_put_call_oi_ratio", put_oi / call_oi.where(call_oi.abs() > 1e-8))
        add("engineered_put_call_oi_imbalance", (put_oi - call_oi) / (put_oi + call_oi).where((put_oi + call_oi).abs() > 1e-8))
    return result, created


def linear_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def candidate_specs(seed: int, jobs: int) -> list[dict]:
    specs = []
    for lookback in (60, 120):
        prefix = f"lookback_{lookback}m"
        regression_bases = {
            "ordinary_linear": linear_pipeline(LinearRegression(n_jobs=1)),
            "ridge_a0.1": linear_pipeline(Ridge(alpha=0.1, solver="lsqr")),
            "ridge_a1": linear_pipeline(Ridge(alpha=1.0, solver="lsqr")),
            "ridge_a10": linear_pipeline(Ridge(alpha=10.0, solver="lsqr")),
            "elastic_a0.0001_l10.1": linear_pipeline(ElasticNet(alpha=0.0001, l1_ratio=0.1, max_iter=5000, random_state=seed)),
            "elastic_a0.0005_l10.2": linear_pipeline(ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=5000, random_state=seed)),
            "elastic_a0.001_l10.5": linear_pipeline(ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=seed)),
        }
        for name, base in regression_bases.items():
            specs.append({
                "name": f"{prefix}_{name}", "lookback": lookback, "kind": "two_target_regression",
                "model": SeparateTargetRegressor(base, base),
            })
        lgb_params = dict(
            n_estimators=350, learning_rate=0.025, num_leaves=15, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.75, random_state=seed, n_jobs=jobs, verbosity=-1,
        )
        specs.append({
            "name": f"{prefix}_lightgbm_two_target", "lookback": lookback, "kind": "two_target_regression",
            "model": SeparateTargetRegressor(
                lgb.LGBMRegressor(objective="regression_l1", **lgb_params),
                lgb.LGBMRegressor(objective="regression_l1", **lgb_params),
            ),
        })
        specs.append({
            "name": f"{prefix}_lightgbm_classifier", "lookback": lookback, "kind": "direct_classifier",
            "model": lgb.LGBMClassifier(
                objective="binary", class_weight="balanced", n_estimators=300, learning_rate=0.025,
                num_leaves=15, min_child_samples=100, subsample=0.8, colsample_bytree=0.75,
                random_state=seed, n_jobs=jobs, verbosity=-1,
            ),
        })
    return specs


def choose_threshold(actual: pd.Series, score: np.ndarray) -> tuple[float, dict]:
    y = actual.astype(int).to_numpy()
    candidates = np.unique(np.quantile(score[np.isfinite(score)], np.linspace(0.01, 0.99, 197)))
    rows = []
    for threshold in candidates:
        predicted = score >= threshold
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
        })
    best = max(rows, key=lambda row: (row["balanced_accuracy"], row["f1"]))
    return best["threshold"], best


def evaluate(actual: pd.Series, score: np.ndarray, threshold: float) -> dict:
    y = actual.astype(int).to_numpy()
    predicted = score >= threshold
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, score)),
        "average_precision": float(average_precision_score(y, score)),
    }


def model_score(model, kind: str, matrix: pd.DataFrame, target_settings: dict) -> tuple[np.ndarray, np.ndarray | None]:
    if kind == "direct_classifier":
        return model.predict_proba(matrix)[:, 1], None
    components = model.predict(matrix)
    score = regression_event_score(
        components, float(target_settings["minimum_rv_drop"]),
        float(target_settings["minimum_rv_drop_beyond_iv"]),
    )
    return score, components


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine the 120-minute relative RV-decline model")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts_refined")
    parser.add_argument("--reuse-validation-leaderboard", action="store_true")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    settings, target_settings = config["training"], config["target"]
    artifacts = (config_path.parent / args.artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    base_features_120 = approved_features(manifest, 120)
    required = list(dict.fromkeys([
        "datetime", "exit_datetime", "training_row_eligible", RV_CURRENT, RV_FUTURE, IV_CHANGE,
        *base_features_120,
    ]))
    frame = pd.read_csv(
        config["data"]["feature_csv"], usecols=required, parse_dates=["datetime", "exit_datetime"], low_memory=False
    )
    frame, engineered = add_engineered_features(frame)
    frame = add_target(
        frame, float(target_settings["minimum_rv_drop"]), float(target_settings["minimum_rv_drop_beyond_iv"])
    )
    frame = frame[frame["training_row_eligible"].fillna(False) & frame[TARGET].notna()].copy()
    frame[TARGET] = frame[TARGET].astype(int)
    split = purged_chronological_split(
        frame, float(settings["train_fraction"]), float(settings["validation_fraction"]),
        int(target_settings["horizon_minutes"]),
    )
    train, validation, test = split["train"], split["validation"], split["test"]
    development = pd.concat([train, validation]).sort_values("datetime")
    y_components = [RV_CHANGE, IV_CHANGE_COPY]
    leaderboard_path = artifacts / "validation_leaderboard.csv"
    if args.reuse_validation_leaderboard and leaderboard_path.exists():
        validation_frame = pd.read_csv(leaderboard_path)
    else:
        validation_results = []
        for spec in candidate_specs(int(settings["random_seed"]), int(settings["n_jobs"])):
            lookback = spec["lookback"]
            features = approved_features(manifest, lookback) + [
                name for name in engineered if not ("120m" in name and lookback < 120)
            ]
            matrix_train = numeric_matrix(train, features)
            matrix_validation = numeric_matrix(validation, features)
            target = train[TARGET] if spec["kind"] == "direct_classifier" else train[y_components]
            model = clone(spec["model"]).fit(matrix_train, target)
            score, components = model_score(model, spec["kind"], matrix_validation, target_settings)
            threshold, threshold_result = choose_threshold(validation[TARGET], score)
            row = {
                "name": spec["name"], "kind": spec["kind"], "lookback_minutes": lookback,
                "feature_count": len(features), "validation_roc_auc": float(roc_auc_score(validation[TARGET], score)),
                "validation_average_precision": float(average_precision_score(validation[TARGET], score)),
                **{f"validation_{key}": value for key, value in threshold_result.items()},
            }
            if components is not None:
                row["validation_rv_change_mae"] = float(mean_absolute_error(validation[RV_CHANGE], components[:, 0]))
                row["validation_iv_change_mae"] = float(mean_absolute_error(validation[IV_CHANGE_COPY], components[:, 1]))
            validation_results.append(row)
            print(f"validated {spec['name']}: AUC={row['validation_roc_auc']:.4f}", flush=True)
        validation_frame = pd.DataFrame(validation_results).sort_values(
            ["validation_roc_auc", "validation_balanced_accuracy"], ascending=False
        )
        validation_frame.to_csv(leaderboard_path, index=False)
    selected_name = str(validation_frame.iloc[0]["name"])
    selected_spec = next(spec for spec in candidate_specs(int(settings["random_seed"]), int(settings["n_jobs"])) if spec["name"] == selected_name)
    selected_validation = validation_frame.iloc[0]
    selected_lookback = int(selected_spec["lookback"])
    selected_features = approved_features(manifest, selected_lookback) + [
        name for name in engineered if not ("120m" in name and selected_lookback < 120)
    ]
    final_target = development[TARGET] if selected_spec["kind"] == "direct_classifier" else development[y_components]
    final_model = clone(selected_spec["model"]).fit(numeric_matrix(development, selected_features), final_target)
    test_score, test_components = model_score(final_model, selected_spec["kind"], numeric_matrix(test, selected_features), target_settings)
    test_metrics = evaluate(test[TARGET], test_score, float(selected_validation["validation_threshold"]))
    if test_components is not None:
        test_metrics["rv_change_mae"] = float(mean_absolute_error(test[RV_CHANGE], test_components[:, 0]))
        test_metrics["iv_change_mae"] = float(mean_absolute_error(test[IV_CHANGE_COPY], test_components[:, 1]))
    test_scored = test.copy()
    test_scored["selected_model_score"] = test_score
    non_overlapping = non_overlapping_rows(test_scored, int(target_settings["horizon_minutes"]))
    non_overlapping_metrics = evaluate(
        non_overlapping[TARGET], non_overlapping["selected_model_score"].to_numpy(),
        float(selected_validation["validation_threshold"]),
    )
    quarterly_metrics = {}
    for period, part in test_scored.groupby(test_scored["datetime"].dt.to_period("Q")):
        if part[TARGET].nunique() == 2:
            quarterly_metrics[str(period)] = evaluate(
                part[TARGET], part["selected_model_score"].to_numpy(),
                float(selected_validation["validation_threshold"]),
            )

    predictions = test[["datetime", TARGET, RV_CHANGE, IV_CHANGE_COPY]].copy()
    predictions["selected_model_score"] = test_score
    predictions["predicted_event"] = test_score >= float(selected_validation["validation_threshold"])
    if test_components is not None:
        predictions["predicted_rv_change_120m"] = test_components[:, 0]
        predictions["predicted_iv_change_120m"] = test_components[:, 1]
    predictions.to_csv(artifacts / "held_out_predictions.csv", index=False)

    bundle = {
        "model": final_model, "model_type": selected_spec["kind"], "features": selected_features,
        "engineered_features": engineered, "lookback_minutes": selected_lookback,
        "threshold": float(selected_validation["validation_threshold"]), "target_definition": target_settings,
        "name": selected_name,
    }
    joblib.dump(bundle, artifacts / "winner.joblib")
    metrics = {
        "selection_rule": "highest validation ROC AUC, tie-broken by validation balanced accuracy",
        "selected_model": selected_name,
        "selected_validation": selected_validation.to_dict(),
        "test": test_metrics,
        "test_non_overlapping_120m": non_overlapping_metrics,
        "test_by_quarter": quarterly_metrics,
        "rows": {name: int(len(part)) for name, part in split.items()},
        "periods": {
            name: {"start": str(part["datetime"].min()), "end": str(part["datetime"].max())}
            for name, part in split.items()
        },
        "versions": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    lines = [
        "# Refined 120-minute RV decline model", "",
        f"Selected on validation data: **{selected_name}**.", "",
        f"Held-out test accuracy: {test_metrics['accuracy']:.2%}.",
        f"Held-out balanced accuracy: {test_metrics['balanced_accuracy']:.2%}.",
        f"Held-out ROC AUC: {test_metrics['roc_auc']:.4f}.",
        f"Held-out average precision: {test_metrics['average_precision']:.4f}.",
        f"Held-out precision: {test_metrics['precision']:.2%}; recall: {test_metrics['recall']:.2%}.", "",
        f"Non-overlapping 120-minute test observations: {non_overlapping_metrics['rows']:,}; accuracy {non_overlapping_metrics['accuracy']:.2%}; ROC AUC {non_overlapping_metrics['roc_auc']:.4f}.", "",
        f"This model requires {selected_lookback} minutes of trailing history and uses {len(selected_features)} entry-time features.", "",
        "The validation leaderboard contains every tested feature scope and model specification. The held-out test set was evaluated only after validation selected the final specification.",
    ]
    (artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_model": selected_name, "test": test_metrics, "artifacts": str(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
