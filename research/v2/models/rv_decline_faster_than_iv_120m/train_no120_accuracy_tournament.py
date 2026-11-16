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
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_components import SeparateTargetRegressor
from refine import IV_CHANGE_COPY, RV_CHANGE, add_engineered_features, evaluate, model_score
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


def scaled(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def candidates(seed: int, jobs: int) -> list[dict]:
    """Models chosen to cover linear, bagged-tree, and boosted-tree families."""
    common_lgb = dict(random_state=seed, n_jobs=jobs, verbosity=-1)
    specs = [
        {
            "name": "logistic_unweighted_c0.2",
            "kind": "direct_classifier",
            "model": scaled(LogisticRegression(C=0.2, max_iter=2500, solver="liblinear", random_state=seed)),
        },
        {
            "name": "logistic_balanced_c0.2",
            "kind": "direct_classifier",
            "model": scaled(LogisticRegression(
                C=0.2, class_weight="balanced", max_iter=2500, solver="liblinear", random_state=seed
            )),
        },
        {
            "name": "random_forest_classifier",
            "kind": "direct_classifier",
            "model": RandomForestClassifier(
                n_estimators=300, min_samples_leaf=30, max_features=0.65,
                class_weight="balanced_subsample", random_state=seed, n_jobs=jobs,
            ),
        },
        {
            "name": "extra_trees_classifier_leaf20",
            "kind": "direct_classifier",
            "model": ExtraTreesClassifier(
                n_estimators=350, min_samples_leaf=20, max_features=0.75,
                class_weight="balanced", random_state=seed, n_jobs=jobs,
            ),
        },
        {
            "name": "extra_trees_classifier_leaf60",
            "kind": "direct_classifier",
            "model": ExtraTreesClassifier(
                n_estimators=350, min_samples_leaf=60, max_features=0.75,
                class_weight="balanced", random_state=seed, n_jobs=jobs,
            ),
        },
        {
            "name": "lightgbm_classifier_leaf15",
            "kind": "direct_classifier",
            "model": lgb.LGBMClassifier(
                objective="binary", n_estimators=450, learning_rate=0.02, num_leaves=15,
                min_child_samples=80, subsample=0.85, colsample_bytree=0.75, **common_lgb,
            ),
        },
        {
            "name": "lightgbm_classifier_leaf31",
            "kind": "direct_classifier",
            "model": lgb.LGBMClassifier(
                objective="binary", n_estimators=400, learning_rate=0.025, num_leaves=31,
                min_child_samples=100, subsample=0.85, colsample_bytree=0.75, **common_lgb,
            ),
        },
    ]

    regressors = {
        "ordinary_linear_two_target": scaled(LinearRegression(n_jobs=1)),
        "ridge_a1_two_target": scaled(Ridge(alpha=1.0, solver="lsqr")),
        "ridge_a10_two_target": scaled(Ridge(alpha=10.0, solver="lsqr")),
        "elastic_a0.001_l10.5_two_target": scaled(ElasticNet(
            alpha=0.001, l1_ratio=0.5, max_iter=5000, random_state=seed
        )),
        "extra_trees_regression_two_target": ExtraTreesRegressor(
            n_estimators=300, min_samples_leaf=25, max_features=0.75,
            random_state=seed, n_jobs=jobs,
        ),
        "lightgbm_regression_two_target": lgb.LGBMRegressor(
            objective="regression_l1", n_estimators=450, learning_rate=0.02, num_leaves=15,
            min_child_samples=80, subsample=0.85, colsample_bytree=0.75, **common_lgb,
        ),
    }
    specs.extend({
        "name": name,
        "kind": "two_target_regression",
        "model": SeparateTargetRegressor(model, model),
    } for name, model in regressors.items())
    return specs


def select_accuracy_threshold(actual: pd.Series, score: np.ndarray, kind: str) -> tuple[float, dict]:
    """Select the threshold on historical validation only, optimizing plain accuracy."""
    finite = np.asarray(score, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise ValueError("No finite validation scores")
    candidates_ = np.unique(np.quantile(finite, np.linspace(0.0, 1.0, 1001)))
    defaults = np.array([0.5] if kind == "direct_classifier" else [0.0])
    candidates_ = np.unique(np.concatenate([candidates_, defaults]))
    y = actual.astype(int).to_numpy()
    rows = []
    for threshold in candidates_:
        predicted = np.asarray(score) >= threshold
        rows.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
        })
    return max(rows, key=lambda row: (row["accuracy"], row["balanced_accuracy"], row["f1"]))["threshold"], max(
        rows, key=lambda row: (row["accuracy"], row["balanced_accuracy"], row["f1"])
    )


def strict_no120_features(manifest: dict, engineered: list[str]) -> list[str]:
    features = approved_features(manifest, 60) + [name for name in engineered if "120m" not in name.lower()]
    features = list(dict.fromkeys(name for name in features if "120m" not in name.lower()))
    forbidden = [name for name in features if "120m" in name.lower()]
    if forbidden:
        raise AssertionError(f"120-minute inputs found: {forbidden}")
    if any(name.startswith(("target_", "exit_fixed_", "exit_gap_")) for name in features):
        raise AssertionError("Future-derived input found")
    return features


def prepare(frame: pd.DataFrame, target_settings: dict) -> pd.DataFrame:
    frame = add_target(
        frame,
        float(target_settings["minimum_rv_drop"]),
        float(target_settings["minimum_rv_drop_beyond_iv"]),
    )
    frame = frame[frame["training_row_eligible"].fillna(False) & frame[TARGET].notna()].copy()
    frame[TARGET] = frame[TARGET].astype(int)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="No-120m-input accuracy tournament and 2026 evaluation")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--holdout-csv",
        default="../../research/combined_atm_iv_rv_convergence_v2/artifacts_2026_holdout/combined_atm_iv_convergence_v2_features_2026.csv",
    )
    parser.add_argument("--artifacts-dir", default="artifacts_no120_accuracy")
    parser.add_argument("--reuse-leaderboard", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    settings, target_settings = config["training"], config["target"]
    artifacts = (config_path.parent / args.artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    holdout_path = (config_path.parent / args.holdout_csv).resolve()
    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))

    base = approved_features(manifest, 60)
    required = list(dict.fromkeys([
        "datetime", "exit_datetime", "training_row_eligible", RV_CURRENT, RV_FUTURE, IV_CHANGE, *base,
    ]))
    historical = pd.read_csv(
        config["data"]["feature_csv"], usecols=required,
        parse_dates=["datetime", "exit_datetime"], low_memory=False,
    )
    holdout = pd.read_csv(
        holdout_path, usecols=required, parse_dates=["datetime", "exit_datetime"], low_memory=False,
    )
    historical, engineered = add_engineered_features(historical)
    holdout, engineered_holdout = add_engineered_features(holdout)
    if engineered != engineered_holdout:
        raise AssertionError("Historical and 2026 engineered-feature schemas differ")
    features = strict_no120_features(manifest, engineered)
    historical = prepare(historical, target_settings)
    holdout = prepare(holdout, target_settings)
    if historical["datetime"].max() >= holdout["datetime"].min():
        raise AssertionError("2026 holdout overlaps historical data")

    split = purged_chronological_split(
        historical,
        float(settings["train_fraction"]),
        float(settings["validation_fraction"]),
        int(target_settings["horizon_minutes"]),
    )
    train, validation = split["train"], split["validation"]
    y_components = [RV_CHANGE, IV_CHANGE_COPY]
    leaderboard_path = artifacts / "validation_leaderboard.csv"
    if args.reuse_leaderboard and leaderboard_path.exists():
        leaderboard_frame = pd.read_csv(leaderboard_path).sort_values(
            ["validation_accuracy", "validation_balanced_accuracy", "validation_roc_auc"], ascending=False
        )
    else:
        leaderboard = []
        for spec in candidates(int(settings["random_seed"]), int(settings["n_jobs"])):
            target = train[TARGET] if spec["kind"] == "direct_classifier" else train[y_components]
            model = clone(spec["model"]).fit(numeric_matrix(train, features), target)
            score, components = model_score(model, spec["kind"], numeric_matrix(validation, features), target_settings)
            threshold, threshold_metrics = select_accuracy_threshold(validation[TARGET], score, spec["kind"])
            row = {
                "name": spec["name"],
                "kind": spec["kind"],
                "feature_count": len(features),
                **{f"validation_{key}": value for key, value in threshold_metrics.items()},
                "validation_roc_auc": float(roc_auc_score(validation[TARGET], score)),
                "validation_average_precision": float(average_precision_score(validation[TARGET], score)),
            }
            if components is not None:
                row["validation_rv_change_mae"] = float(mean_absolute_error(validation[RV_CHANGE], components[:, 0]))
                row["validation_iv_change_mae"] = float(mean_absolute_error(validation[IV_CHANGE_COPY], components[:, 1]))
            leaderboard.append(row)
            print(
                f"{spec['name']}: accuracy={row['validation_accuracy']:.4f}, "
                f"balanced={row['validation_balanced_accuracy']:.4f}, AUC={row['validation_roc_auc']:.4f}",
                flush=True,
            )
        leaderboard_frame = pd.DataFrame(leaderboard).sort_values(
            ["validation_accuracy", "validation_balanced_accuracy", "validation_roc_auc"], ascending=False
        )
        leaderboard_frame.to_csv(leaderboard_path, index=False)
    selected = leaderboard_frame.iloc[0]
    selected_name = str(selected["name"])
    selected_spec = next(spec for spec in candidates(int(settings["random_seed"]), int(settings["n_jobs"]))
                         if spec["name"] == selected_name)
    historical_accuracy_threshold = float(selected["validation_threshold"])
    # A two-target regression score is a signed margin relative to both target
    # conditions. Zero therefore has a stable, period-independent meaning. A
    # fitted offset is retained for diagnosis but is not deployed.
    threshold = 0.0 if selected_spec["kind"] == "two_target_regression" else 0.5

    validation_target = train[TARGET] if selected_spec["kind"] == "direct_classifier" else train[y_components]
    validation_model = clone(selected_spec["model"]).fit(numeric_matrix(train, features), validation_target)
    validation_score, _ = model_score(
        validation_model, selected_spec["kind"], numeric_matrix(validation, features), target_settings
    )
    validation_deployment_metrics = evaluate(validation[TARGET], validation_score, threshold)

    final_target = historical[TARGET] if selected_spec["kind"] == "direct_classifier" else historical[y_components]
    final_model = clone(selected_spec["model"]).fit(numeric_matrix(historical, features), final_target)
    holdout_score, holdout_components = model_score(
        final_model, selected_spec["kind"], numeric_matrix(holdout, features), target_settings
    )
    holdout_metrics = evaluate(holdout[TARGET], holdout_score, threshold)
    holdout_historical_tuned_metrics = evaluate(
        holdout[TARGET], holdout_score, historical_accuracy_threshold
    )
    if holdout_components is not None:
        holdout_metrics["rv_change_mae"] = float(mean_absolute_error(holdout[RV_CHANGE], holdout_components[:, 0]))
        holdout_metrics["iv_change_mae"] = float(mean_absolute_error(holdout[IV_CHANGE_COPY], holdout_components[:, 1]))
    tn, fp, fn, tp = confusion_matrix(holdout[TARGET], holdout_score >= threshold).ravel()
    holdout_metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    scored = holdout.copy()
    scored["model_score"] = holdout_score
    nonoverlap = non_overlapping_rows(scored, int(target_settings["horizon_minutes"]))
    nonoverlap_metrics = evaluate(nonoverlap[TARGET], nonoverlap["model_score"].to_numpy(), threshold)

    predictions = holdout[["datetime", TARGET, RV_CHANGE, IV_CHANGE_COPY]].copy()
    predictions["model_score"] = holdout_score
    predictions["predicted_event"] = holdout_score >= threshold
    if holdout_components is not None:
        predictions["predicted_rv_change_120m"] = holdout_components[:, 0]
        predictions["predicted_iv_change_120m"] = holdout_components[:, 1]
    predictions.to_csv(artifacts / "oot_2026_predictions.csv", index=False)

    bundle = {
        "model": final_model,
        "model_type": selected_spec["kind"],
        "features": features,
        "engineered_features": [name for name in engineered if name in features],
        "lookback_minutes": 60,
        "threshold": threshold,
        "threshold_selection": (
            "fixed semantic zero boundary for signed two-target event margin"
            if selected_spec["kind"] == "two_target_regression" else "fixed classifier probability 0.5"
        ),
        "historical_accuracy_threshold_diagnostic": historical_accuracy_threshold,
        "target_definition": target_settings,
        "name": selected_name,
        "training_period": {
            "start": str(historical["datetime"].min()),
            "end": str(historical["datetime"].max()),
        },
    }
    if any("120m" in name.lower() for name in bundle["features"]):
        raise AssertionError("Saved model bundle contains a 120m input")
    joblib.dump(bundle, artifacts / "winner.joblib")

    majority_accuracy = float(max(validation[TARGET].mean(), 1.0 - validation[TARGET].mean()))
    metrics = {
        "strict_no_120m_inputs": True,
        "feature_count": len(features),
        "features_containing_120m": [name for name in features if "120m" in name.lower()],
        "target_horizon_minutes": int(target_settings["horizon_minutes"]),
        "target_definition": (
            "Predict whether, over the next 120 minutes, RV declines by at least "
            f"{float(target_settings['minimum_rv_drop']):.4f} and declines at least "
            f"{float(target_settings['minimum_rv_drop_beyond_iv']):.4f} more than fixed-strike combined ATM IV."
        ),
        "selection_data": {
            "train_start": str(train["datetime"].min()), "train_end": str(train["datetime"].max()),
            "validation_start": str(validation["datetime"].min()),
            "validation_end": str(validation["datetime"].max()),
            "validation_rows": int(len(validation)),
            "validation_positive_rate": float(validation[TARGET].mean()),
            "validation_majority_accuracy": majority_accuracy,
        },
        "winner": selected.to_dict(),
        "winner_at_deployment_threshold_on_validation": validation_deployment_metrics,
        "holdout_2026": holdout_metrics,
        "holdout_2026_at_historical_accuracy_threshold": holdout_historical_tuned_metrics,
        "holdout_2026_nonoverlapping_120m": nonoverlap_metrics,
        "versions": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "lightgbm": lgb.__version__,
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = [
        "# No-120-minute-input accuracy tournament", "",
        "The prediction horizon remains 120 minutes, but no model input feature contains `120m`.", "",
        f"- Models compared: {len(leaderboard_frame)}.",
        f"- Input features: {len(features)}; names containing `120m`: 0.",
        f"- Winner selected on historical validation only: {selected_name}.",
        f"- Best historical validation accuracy after threshold fitting: {selected['validation_accuracy']:.2%}.",
        f"- Historical validation accuracy at the stable semantic boundary: {validation_deployment_metrics['accuracy']:.2%}.",
        f"- Historical validation ROC AUC: {selected['validation_roc_auc']:.4f}.",
        f"- Saved semantic threshold: {threshold:.6f}.",
        f"- Period-specific historical threshold retained only as a diagnostic: {historical_accuracy_threshold:.6f}.", "",
        "## 2026 out-of-time comparison", "",
        f"- Rows: {holdout_metrics['rows']:,}.",
        f"- Accuracy: {holdout_metrics['accuracy']:.2%}.",
        f"- Balanced accuracy: {holdout_metrics['balanced_accuracy']:.2%}.",
        f"- Precision: {holdout_metrics['precision']:.2%}.",
        f"- Recall: {holdout_metrics['recall']:.2%}.",
        f"- F1: {holdout_metrics['f1']:.2%}.",
        f"- ROC AUC: {holdout_metrics['roc_auc']:.4f}.",
        f"- Average precision: {holdout_metrics['average_precision']:.4f}.",
        f"- Non-overlapping 120-minute sample accuracy: {nonoverlap_metrics['accuracy']:.2%}.", "",
        "The 2026 period was used in earlier project comparisons, so these figures are comparative out-of-time results rather than a newly untouched final test. Model and threshold selection in this run used historical validation only.",
    ]
    (artifacts / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({
        "winner": selected_name,
        "threshold": threshold,
        "historical_accuracy_threshold_diagnostic": historical_accuracy_threshold,
        "feature_count": len(features),
        "validation_accuracy": float(selected["validation_accuracy"]),
        "holdout_2026": holdout_metrics,
        "artifacts": str(artifacts),
    }, indent=2))


if __name__ == "__main__":
    main()
