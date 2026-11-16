from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[2]
FEATURE_PROJECT = ROOT / "research" / "combined_atm_iv_rv_convergence_v2"
sys.path.insert(0, str(FEATURE_PROJECT))
from feature_pipeline import assert_leakage_safe_features, purged_chronological_split  # noqa: E402


RV_CURRENT = "rv_trailing_120m"
RV_FUTURE = "target_rv_forward_120m"
IV_CHANGE = "target_fixed_combined_iv_change_120m"
TARGET = "target_rv_declines_sharper_than_iv_120m"


def load_config(path: str | Path) -> tuple[dict, Path]:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("feature_csv", "feature_manifest"):
        config["data"][key] = str((config_path.parent / config["data"][key]).resolve())
    return config, config_path


def feature_lookback(name: str) -> int:
    matches = re.findall(r"(?:lag_|change_|trailing_|return_)(\d+)m", name)
    return max((int(value) for value in matches), default=0)


def approved_features(manifest: dict, maximum_lookback: int) -> list[str]:
    features = [name for name in manifest["features"] if feature_lookback(name) <= maximum_lookback]
    assert_leakage_safe_features(features)
    forbidden = {RV_FUTURE, IV_CHANGE, "exit_fixed_atm_iv_combined"}
    overlap = forbidden.intersection(features)
    if overlap:
        raise ValueError(f"Future-derived columns found in features: {sorted(overlap)}")
    return features


def add_target(frame: pd.DataFrame, minimum_rv_drop: float, relative_margin: float) -> pd.DataFrame:
    result = frame.copy()
    result["target_rv_change_120m"] = result[RV_FUTURE] - result[RV_CURRENT]
    result["target_iv_change_120m"] = result[IV_CHANGE]
    result["target_rv_excess_decline_vs_iv_120m"] = result[IV_CHANGE] - result["target_rv_change_120m"]
    complete = result[[RV_CURRENT, RV_FUTURE, IV_CHANGE]].notna().all(axis=1)
    event = (
        (result["target_rv_change_120m"] <= -minimum_rv_drop)
        & (result["target_rv_excess_decline_vs_iv_120m"] >= relative_margin)
    )
    result[TARGET] = event.where(complete)
    return result


def numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def expanding_folds(frame: pd.DataFrame, splits: int):
    ordered = frame.sort_values("datetime")
    stamps = np.sort(ordered["datetime"].dropna().unique())
    boundaries = np.linspace(0.40, 1.0, splits + 1)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = pd.Timestamp(stamps[min(int(len(stamps) * left), len(stamps) - 1)])
        end = pd.Timestamp(stamps[min(int(len(stamps) * right), len(stamps) - 1)])
        fit = ordered[(ordered["datetime"] < start) & (ordered["exit_datetime"] < start)]
        score = ordered[(ordered["datetime"] >= start) & (ordered["datetime"] <= end)]
        if fit[TARGET].nunique() == 2 and score[TARGET].nunique() == 2:
            yield fit, score


def candidate_models(seed: int, jobs: int):
    return {
        "logistic": [
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
                ("model", LogisticRegression(
                    C=c, class_weight="balanced", max_iter=2000, random_state=seed, solver="liblinear"
                )),
            ])
            for c in (0.05, 0.2, 1.0)
        ],
        "lightgbm": [
            lgb.LGBMClassifier(
                objective="binary", class_weight="balanced", random_state=seed, n_jobs=jobs, verbosity=-1, **params
            )
            for params in (
                {"n_estimators": 250, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80,
                 "subsample": 0.8, "colsample_bytree": 0.7},
                {"n_estimators": 400, "learning_rate": 0.025, "num_leaves": 31, "min_child_samples": 80,
                 "subsample": 0.8, "colsample_bytree": 0.7},
                {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 120,
                 "subsample": 1.0, "colsample_bytree": 0.8},
            )
        ],
    }


def tune_models(train: pd.DataFrame, features: list[str], models: dict, splits: int) -> list[dict]:
    results = []
    for family, estimators in models.items():
        for index, estimator in enumerate(estimators):
            fold_rows = []
            for fit, score in expanding_folds(train, splits):
                model = clone(estimator)
                model.fit(numeric_matrix(fit, features), fit[TARGET].astype(int))
                probability = np.clip(model.predict_proba(numeric_matrix(score, features))[:, 1], 1e-6, 1 - 1e-6)
                fold_rows.append({
                    "log_loss": float(log_loss(score[TARGET].astype(int), probability)),
                    "roc_auc": float(roc_auc_score(score[TARGET].astype(int), probability)),
                    "average_precision": float(average_precision_score(score[TARGET].astype(int), probability)),
                })
            results.append({
                "family": family,
                "candidate_index": index,
                "mean_log_loss": float(np.mean([row["log_loss"] for row in fold_rows])),
                "mean_roc_auc": float(np.mean([row["roc_auc"] for row in fold_rows])),
                "mean_average_precision": float(np.mean([row["average_precision"] for row in fold_rows])),
                "folds": fold_rows,
            })
    return sorted(results, key=lambda row: row["mean_log_loss"])


def choose_threshold(actual: pd.Series, probability: np.ndarray) -> tuple[float, dict]:
    candidates = np.linspace(0.15, 0.85, 141)
    scores = []
    y = actual.astype(int).to_numpy()
    for threshold in candidates:
        predicted = probability >= threshold
        scores.append({
            "threshold": float(threshold),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "f1": float(f1_score(y, predicted, zero_division=0)),
            "precision": float(precision_score(y, predicted, zero_division=0)),
            "recall": float(recall_score(y, predicted, zero_division=0)),
        })
    best = max(scores, key=lambda row: (row["balanced_accuracy"], row["f1"]))
    return best["threshold"], best


def metrics(actual: pd.Series, probability: np.ndarray, threshold: float) -> dict:
    y = actual.astype(int).to_numpy()
    p = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    predicted = p >= threshold
    return {
        "rows": int(len(y)),
        "positive_rate": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "log_loss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "confusion_matrix": confusion_matrix(y, predicted).tolist(),
        "classification_report": classification_report(y, predicted, output_dict=True, zero_division=0),
    }


def top_features(model, features: list[str], limit: int = 30) -> list[dict]:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return []
    order = np.argsort(values)[::-1][:limit]
    return [{"feature": features[i], "importance": float(values[i])} for i in order]


def non_overlapping_rows(frame: pd.DataFrame, horizon_minutes: int) -> pd.DataFrame:
    selected = []
    for _, day in frame.sort_values("datetime").groupby(frame["datetime"].dt.date):
        next_allowed = None
        for index, row in day.iterrows():
            if next_allowed is None or row["datetime"] >= next_allowed:
                selected.append(index)
                next_allowed = row["datetime"] + pd.Timedelta(minutes=int(horizon_minutes))
    return frame.loc[selected].sort_values("datetime")


def ranked_operating_points(actual: pd.Series, probability: np.ndarray) -> list[dict]:
    ranked = pd.DataFrame({"actual": actual.astype(int).to_numpy(), "probability": probability}).sort_values(
        "probability", ascending=False
    )
    base_rate = float(ranked["actual"].mean())
    rows = []
    for fraction in (0.10, 0.20, 0.30, 0.50):
        selected = ranked.head(max(1, int(np.ceil(len(ranked) * fraction))))
        precision = float(selected["actual"].mean())
        rows.append({
            "top_fraction": fraction,
            "rows": int(len(selected)),
            "minimum_probability": float(selected["probability"].min()),
            "precision": precision,
            "lift_vs_test_base_rate": float(precision / base_rate),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a sharp 120-minute RV decline relative to fixed-strike ATM IV")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    target_settings = config["target"]
    settings = config["training"]
    artifacts = Path(args.artifacts_dir).resolve() if args.artifacts_dir else config_path.parent / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    features = approved_features(manifest, int(settings["max_feature_lookback_minutes"]))
    required = list(dict.fromkeys(["datetime", "exit_datetime", "training_row_eligible", RV_CURRENT, RV_FUTURE, IV_CHANGE] + features))
    frame = pd.read_csv(config["data"]["feature_csv"], usecols=required, parse_dates=["datetime", "exit_datetime"], low_memory=False)
    frame = add_target(frame, float(target_settings["minimum_rv_drop"]), float(target_settings["minimum_rv_drop_beyond_iv"]))
    frame = frame[frame["training_row_eligible"].fillna(False) & frame[TARGET].notna()].copy()
    frame[TARGET] = frame[TARGET].astype(int)

    split = purged_chronological_split(
        frame,
        float(settings["train_fraction"]),
        float(settings["validation_fraction"]),
        int(target_settings["horizon_minutes"]),
    )
    train, validation, test = split["train"], split["validation"], split["test"]
    models = candidate_models(int(settings["random_seed"]), int(settings["n_jobs"]))
    search = tune_models(train, features, models, int(settings["cv_splits"]))
    winner_row = search[0]
    winner_template = models[winner_row["family"]][winner_row["candidate_index"]]

    validation_model = clone(winner_template).fit(numeric_matrix(train, features), train[TARGET])
    validation_probability = validation_model.predict_proba(numeric_matrix(validation, features))[:, 1]
    decision_threshold, threshold_metrics = choose_threshold(validation[TARGET], validation_probability)

    development = pd.concat([train, validation]).sort_values("datetime")
    final_model = clone(winner_template).fit(numeric_matrix(development, features), development[TARGET])
    test_probability = final_model.predict_proba(numeric_matrix(test, features))[:, 1]
    test_metrics = metrics(test[TARGET], test_probability, decision_threshold)

    test_scored = test.copy()
    test_scored["predicted_probability"] = test_probability
    test_non_overlapping = non_overlapping_rows(test_scored, int(target_settings["horizon_minutes"]))
    non_overlapping_metrics = metrics(
        test_non_overlapping[TARGET], test_non_overlapping["predicted_probability"].to_numpy(), decision_threshold
    )
    quarterly_metrics = {}
    for period, part in test_scored.groupby(test_scored["datetime"].dt.to_period("Q")):
        if part[TARGET].nunique() == 2:
            quarterly_metrics[str(period)] = metrics(
                part[TARGET], part["predicted_probability"].to_numpy(), decision_threshold
            )

    baseline_probability = np.full(len(test), development[TARGET].mean())
    baseline_metrics = metrics(test[TARGET], baseline_probability, 0.5)
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(test[TARGET], test_probability)

    predictions = test[[
        "datetime", "exit_datetime", RV_CURRENT, RV_FUTURE, IV_CHANGE,
        "target_rv_change_120m", "target_rv_excess_decline_vs_iv_120m", TARGET,
    ]].copy()
    predictions["predicted_probability"] = test_probability
    predictions["predicted_event"] = test_probability >= decision_threshold
    predictions.to_csv(artifacts / "held_out_predictions.csv", index=False)

    importance = top_features(final_model, features)
    pd.DataFrame(importance).to_csv(artifacts / "feature_importance.csv", index=False)
    pd.DataFrame({
        "threshold": np.append(pr_thresholds, np.nan),
        "precision": precision_curve,
        "recall": recall_curve,
    }).to_csv(artifacts / "precision_recall_curve.csv", index=False)

    bundle = {
        "model": final_model,
        "features": features,
        "decision_threshold": decision_threshold,
        "target_definition": {
            "horizon_minutes": int(target_settings["horizon_minutes"]),
            "minimum_rv_drop": float(target_settings["minimum_rv_drop"]),
            "minimum_rv_drop_beyond_iv": float(target_settings["minimum_rv_drop_beyond_iv"]),
            "rv_current_column": RV_CURRENT,
            "rv_future_column": RV_FUTURE,
            "iv_change_column": IV_CHANGE,
        },
    }
    joblib.dump(bundle, artifacts / "winner.joblib")

    report = {
        "target_definition": bundle["target_definition"],
        "feature_count": len(features),
        "rows": {name: int(len(part)) for name, part in split.items()},
        "periods": {
            name: {"start": str(part["datetime"].min()), "end": str(part["datetime"].max())}
            for name, part in split.items()
        },
        "class_rates": {name: float(part[TARGET].mean()) for name, part in split.items()},
        "model_search": search,
        "winner": {"family": winner_row["family"], "candidate_index": winner_row["candidate_index"]},
        "validation_threshold_selection": threshold_metrics,
        "test": test_metrics,
        "test_non_overlapping_120m": non_overlapping_metrics,
        "test_by_quarter": quarterly_metrics,
        "ranked_operating_points": ranked_operating_points(test[TARGET], test_probability),
        "constant_probability_baseline": baseline_metrics,
        "top_features": importance,
        "versions": {
            "python": sys.version,
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# RV decline faster than IV over 120 minutes", "",
        "## Target", "",
        f"Positive when forward 120-minute RV falls by at least {100 * float(target_settings['minimum_rv_drop']):.1f} volatility points and falls at least {100 * float(target_settings['minimum_rv_drop_beyond_iv']):.1f} volatility points more than fixed-strike combined ATM IV.", "",
        "All model inputs are known at entry. Future RV and fixed-strike IV are labels only. The chronological split purges observations whose 120-minute label crosses a split boundary.", "",
        "## Held-out result", "",
        f"- Test period: {test['datetime'].min()} to {test['datetime'].max()}.",
        f"- Test rows: {len(test):,}; positive rate: {test[TARGET].mean():.2%}.",
        f"- Winner: {winner_row['family']}.",
        f"- ROC AUC: {test_metrics['roc_auc']:.4f}; average precision: {test_metrics['average_precision']:.4f}.",
        f"- Validation-selected threshold: {decision_threshold:.3f}.",
        f"- At that threshold: {test_metrics['precision']:.2%} precision, {test_metrics['recall']:.2%} recall, {test_metrics['balanced_accuracy']:.2%} balanced accuracy.",
        f"- Log loss: {test_metrics['log_loss']:.4f}; Brier score: {test_metrics['brier']:.4f}.", "",
        f"- Non-overlapping 120-minute sample: {non_overlapping_metrics['rows']:,} rows; ROC AUC {non_overlapping_metrics['roc_auc']:.4f}; precision {non_overlapping_metrics['precision']:.2%}; recall {non_overlapping_metrics['recall']:.2%}.", "",
        "## Interpretation", "",
        "The output is a probability of the defined event, not a forecast of the exact RV or IV level. Use `predicted_probability` for ranking and choose an operating threshold from the held-out precision-recall curve according to the cost of false positives.", "",
        "The held-out period is the only basis for the reported final performance. Retrain and test on newer untouched data before live use.",
    ]
    (artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"winner": report["winner"], "test": test_metrics, "artifacts": str(artifacts)}, indent=2))


if __name__ == "__main__":
    main()

