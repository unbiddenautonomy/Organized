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
import yaml
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, MultiTaskElasticNet, Ridge
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
from sklearn.preprocessing import RobustScaler, StandardScaler

from train import (
    IV_CHANGE,
    RV_CURRENT,
    RV_FUTURE,
    TARGET,
    add_target,
    approved_features,
    load_config,
    numeric_matrix,
    purged_chronological_split,
)


RV_CHANGE = "target_rv_change_120m"
IV_CHANGE_COPY = "target_iv_change_120m"


def classifier_models(seed: int, jobs: int) -> dict[str, object]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", LogisticRegression(
                C=0.05, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed
            )),
        ]),
        "random_forest_classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=350, min_samples_leaf=30, max_features="sqrt", class_weight="balanced_subsample",
                random_state=seed, n_jobs=jobs,
            )),
        ]),
        "extra_trees_classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=350, min_samples_leaf=20, max_features="sqrt", class_weight="balanced",
                random_state=seed, n_jobs=jobs,
            )),
        ]),
        "hist_gradient_boosting_classifier": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=300, max_leaf_nodes=15, min_samples_leaf=80,
                l2_regularization=1.0, class_weight="balanced", random_state=seed,
            )),
        ]),
        "lightgbm_classifier": lgb.LGBMClassifier(
            objective="binary", n_estimators=250, learning_rate=0.03, num_leaves=15,
            min_child_samples=80, subsample=0.8, colsample_bytree=0.7, class_weight="balanced",
            random_state=seed, n_jobs=jobs, verbosity=-1,
        ),
    }


def regression_models(seed: int) -> dict[str, object]:
    return {
        "ordinary_linear_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression(n_jobs=1)),
        ]),
        "ridge_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=1.0, solver="lsqr")),
        ]),
        "elastic_net_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MultiTaskElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=5000, random_state=seed)),
        ]),
    }


def select_threshold(y: np.ndarray, score: np.ndarray) -> tuple[float, dict]:
    finite = score[np.isfinite(score)]
    candidates = np.unique(np.quantile(finite, np.linspace(0.02, 0.98, 193)))
    best = None
    for threshold in candidates:
        prediction = score >= threshold
        row = {
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
        }
        if best is None or (row["balanced_accuracy"], row["f1"]) > (best["balanced_accuracy"], best["f1"]):
            best = row
    return best["threshold"], best


def score_metrics(y: pd.Series, score: np.ndarray, threshold: float) -> dict:
    actual = y.astype(int).to_numpy()
    prediction = score >= threshold
    return {
        "rows": int(len(actual)),
        "positive_rate": float(actual.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(actual, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "precision": float(precision_score(actual, prediction, zero_division=0)),
        "recall": float(recall_score(actual, prediction, zero_division=0)),
        "f1": float(f1_score(actual, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, score)),
        "average_precision": float(average_precision_score(actual, score)),
    }


def regression_event_score(prediction: np.ndarray, minimum_rv_drop: float, relative_margin: float) -> np.ndarray:
    predicted_rv_change = prediction[:, 0]
    predicted_iv_change = prediction[:, 1]
    rv_condition_margin = -minimum_rv_drop - predicted_rv_change
    relative_condition_margin = predicted_iv_change - predicted_rv_change - relative_margin
    return np.minimum(rv_condition_margin, relative_condition_margin)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classifiers and linear regression models for the RV/IV event")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--artifacts-dir", default="artifacts_tournament")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    settings = config["training"]
    target_settings = config["target"]
    artifacts = (config_path.parent / args.artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    features = approved_features(manifest, int(settings["max_feature_lookback_minutes"]))
    required = list(dict.fromkeys([
        "datetime", "exit_datetime", "training_row_eligible", RV_CURRENT, RV_FUTURE, IV_CHANGE, *features,
    ]))
    frame = pd.read_csv(
        config["data"]["feature_csv"], usecols=required, parse_dates=["datetime", "exit_datetime"], low_memory=False
    )
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
    predictions = test[["datetime", TARGET, RV_CHANGE, IV_CHANGE_COPY]].copy()
    leaderboard = []
    model_details = {}

    for name, template in classifier_models(int(settings["random_seed"]), int(settings["n_jobs"])).items():
        validation_model = clone(template).fit(numeric_matrix(train, features), train[TARGET])
        validation_score = validation_model.predict_proba(numeric_matrix(validation, features))[:, 1]
        threshold, validation_metrics = select_threshold(validation[TARGET].to_numpy(), validation_score)
        validation_metrics["roc_auc"] = float(roc_auc_score(validation[TARGET], validation_score))
        validation_metrics["average_precision"] = float(average_precision_score(validation[TARGET], validation_score))
        final_model = clone(template).fit(numeric_matrix(development, features), development[TARGET])
        test_score = final_model.predict_proba(numeric_matrix(test, features))[:, 1]
        result = score_metrics(test[TARGET], test_score, threshold)
        result.update({
            "model": name, "model_type": "direct_classifier",
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_average_precision": validation_metrics["average_precision"],
        })
        leaderboard.append(result)
        predictions[f"score_{name}"] = test_score
        model_details[name] = {"validation": validation_metrics, "test": result}
        joblib.dump({
            "model": final_model, "model_type": "direct_classifier", "features": features,
            "threshold": threshold, "target_definition": target_settings,
        }, artifacts / f"{name}.joblib")
        print(f"finished {name}: AUC={result['roc_auc']:.4f}", flush=True)

    y_columns = [RV_CHANGE, IV_CHANGE_COPY]
    for name, template in regression_models(int(settings["random_seed"])).items():
        validation_model = clone(template).fit(numeric_matrix(train, features), train[y_columns])
        validation_component_prediction = validation_model.predict(numeric_matrix(validation, features))
        validation_score = regression_event_score(
            validation_component_prediction,
            float(target_settings["minimum_rv_drop"]),
            float(target_settings["minimum_rv_drop_beyond_iv"]),
        )
        threshold, validation_metrics = select_threshold(validation[TARGET].to_numpy(), validation_score)
        validation_metrics["roc_auc"] = float(roc_auc_score(validation[TARGET], validation_score))
        validation_metrics["average_precision"] = float(average_precision_score(validation[TARGET], validation_score))
        final_model = clone(template).fit(numeric_matrix(development, features), development[y_columns])
        test_component_prediction = final_model.predict(numeric_matrix(test, features))
        test_score = regression_event_score(
            test_component_prediction,
            float(target_settings["minimum_rv_drop"]),
            float(target_settings["minimum_rv_drop_beyond_iv"]),
        )
        result = score_metrics(test[TARGET], test_score, threshold)
        result.update({
            "model": name,
            "model_type": "two_target_regression",
            "validation_roc_auc": validation_metrics["roc_auc"],
            "validation_average_precision": validation_metrics["average_precision"],
            "rv_change_mae": float(mean_absolute_error(test[RV_CHANGE], test_component_prediction[:, 0])),
            "iv_change_mae": float(mean_absolute_error(test[IV_CHANGE_COPY], test_component_prediction[:, 1])),
        })
        leaderboard.append(result)
        predictions[f"score_{name}"] = test_score
        predictions[f"predicted_rv_change_{name}"] = test_component_prediction[:, 0]
        predictions[f"predicted_iv_change_{name}"] = test_component_prediction[:, 1]
        model_details[name] = {"validation": validation_metrics, "test": result}
        joblib.dump({
            "model": final_model, "features": features, "threshold": threshold,
            "model_type": "two_target_regression", "target_definition": target_settings,
            "outputs": y_columns, "score_definition": "minimum margin across RV-drop and relative-drop conditions",
        }, artifacts / f"{name}.joblib")
        print(f"finished {name}: AUC={result['roc_auc']:.4f}", flush=True)

    leaderboard_frame = pd.DataFrame(leaderboard).sort_values(
        ["roc_auc", "balanced_accuracy"], ascending=False
    )
    leaderboard_frame.to_csv(artifacts / "leaderboard.csv", index=False)
    predictions.to_csv(artifacts / "held_out_model_predictions.csv", index=False)
    winner_name = str(max(leaderboard, key=lambda row: row["validation_roc_auc"])["model"])
    winner_source = artifacts / f"{winner_name}.joblib"
    joblib.dump(joblib.load(winner_source), artifacts / "winner.joblib")

    summary = {
        "target_definition": target_settings,
        "feature_count": len(features),
        "rows": {key: int(len(value)) for key, value in split.items()},
        "winner": winner_name,
        "leaderboard": leaderboard_frame.to_dict(orient="records"),
        "details": model_details,
        "versions": {
            "python": sys.version, "pandas": pd.__version__, "numpy": np.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_lines = [
        "# 120-minute RV decline model tournament", "",
        "All models use the same leakage-safe features, chronological split, purged boundaries, and held-out test period.", "",
        markdown_table(leaderboard_frame[[
            "model", "model_type", "roc_auc", "average_precision", "balanced_accuracy", "precision", "recall", "f1"
        ]]), "",
        f"Selected winner by validation ROC AUC before inspecting final test performance: **{winner_name}**.", "",
        "Linear, Ridge, and Elastic Net regressions predict RV change and IV change separately. Their event score is the smaller of the two predicted margins required by the target definition.",
    ]
    (artifacts / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(leaderboard_frame[["model", "model_type", "roc_auc", "average_precision", "balanced_accuracy", "precision", "recall"]].to_string(index=False))


if __name__ == "__main__":
    main()
