from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from ngboost import NGBRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


PROJECT = Path(__file__).resolve().parents[2] / "research" / "fixed_strike_iv_rv_convergence"
sys.path.insert(0, str(PROJECT))
from feature_pipeline import (  # noqa: E402
    add_rv_prediction_convergence,
    assert_leakage_safe_features,
    purged_chronological_split,
    safe_feature_columns,
)


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("feature_csv", "feature_manifest"):
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    config["artifacts"] = str((path.parent / "artifacts").resolve())
    return config


def qlike(actual_vol: np.ndarray, predicted_vol: np.ndarray) -> float:
    actual_var = np.maximum(np.asarray(actual_vol, float) ** 2, 1e-12)
    predicted_var = np.maximum(np.asarray(predicted_vol, float) ** 2, 1e-12)
    ratio = actual_var / predicted_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def regression_metrics(y, prediction) -> dict:
    y = np.asarray(y, float); prediction = np.asarray(prediction, float)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(prediction))),
    }


def classification_metrics(y, probability) -> dict:
    y = np.asarray(y, int); probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    prediction = probability >= 0.5
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
    }


def numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    result = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan)


def cap_rows(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def expanding_folds(frame: pd.DataFrame, splits: int, horizon: int):
    ordered = frame.sort_values("datetime")
    stamps = np.sort(ordered["datetime"].unique())
    boundaries = np.linspace(0.40, 1.0, splits + 1)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        validation_start = pd.Timestamp(stamps[min(int(len(stamps) * left), len(stamps) - 1)])
        validation_end = pd.Timestamp(stamps[min(int(len(stamps) * right), len(stamps) - 1)])
        fit = ordered[(ordered.datetime < validation_start) & (ordered.exit_datetime < validation_start)]
        score = ordered[(ordered.datetime >= validation_start) & (ordered.datetime <= validation_end)]
        if len(fit) and len(score):
            yield fit, score


def tune_regressors(name_to_specs, frame, features, target, settings, transform=None, loss="mae"):
    rows = []
    for model_name, specs in name_to_specs.items():
        for params, estimator in specs:
            fold_scores = []
            for fit, score in expanding_folds(frame, int(settings["cv_splits"]), int(settings["horizon_minutes"])):
                fit = cap_rows(fit.dropna(subset=[target]), int(settings["max_training_rows"]))
                score = score.dropna(subset=[target])
                model = clone(estimator).set_params(**params)
                y_fit = transform(fit[target].to_numpy()) if transform else fit[target]
                model.fit(numeric_matrix(fit, features), y_fit)
                pred = model.predict(numeric_matrix(score, features))
                if transform:
                    pred = np.sqrt(np.exp(pred))
                value = qlike(score[target], pred) if loss == "qlike" else mean_absolute_error(score[target], pred)
                fold_scores.append(float(value))
            rows.append({"model": model_name, "params": params, "cv_loss": float(np.mean(fold_scores))})
    return sorted(rows, key=lambda item: item["cv_loss"])


def tune_classifier(specs, frame, features, target, settings):
    rows = []
    for params, estimator in specs:
        scores = []
        for fit, score in expanding_folds(frame, int(settings["cv_splits"]), int(settings["horizon_minutes"])):
            fit = cap_rows(fit.dropna(subset=[target]), int(settings["max_training_rows"]))
            score = score.dropna(subset=[target])
            model = clone(estimator).set_params(**params)
            model.fit(numeric_matrix(fit, features), fit[target].astype(int))
            probability = model.predict_proba(numeric_matrix(score, features))[:, 1]
            scores.append(log_loss(score[target].astype(int), probability))
        rows.append({"params": params, "cv_log_loss": float(np.mean(scores))})
    return sorted(rows, key=lambda item: item["cv_log_loss"])


def tune_ngboost(param_grid, frame, features, target, settings, seed):
    """Tune NGBoost without sklearn Pipeline fitted-state compatibility checks."""
    rows = []
    for params in param_grid:
        fold_scores = []
        for fit, score in expanding_folds(frame, int(settings["cv_splits"]), int(settings["horizon_minutes"])):
            fit = cap_rows(fit.dropna(subset=[target]), int(settings["max_training_rows"]))
            score = score.dropna(subset=[target])
            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(numeric_matrix(fit, features))
            x_score = imputer.transform(numeric_matrix(score, features))
            model = NGBRegressor(random_state=seed, verbose=False, **params)
            model.fit(x_fit, fit[target].to_numpy())
            pred = model.predict(x_score)
            fold_scores.append(float(mean_absolute_error(score[target], pred)))
        rows.append({"model": "ngboost", "params": params, "cv_loss": float(np.mean(fold_scores))})
    return sorted(rows, key=lambda item: item["cv_loss"])


def harq_features(columns: list[str]) -> list[str]:
    preferred = [
        name for name in columns
        if name.startswith(("rv_trailing_", "rq_trailing_", "spot_return_"))
        or name in {"entry_iv_weighted", "surface_atm_iv_call", "surface_atm_iv_put", "day_of_week", "time_sin", "time_cos", "days_to_expiry"}
    ]
    return preferred


def rv_model_specs(seed: int, jobs: int):
    harq = []
    for alpha in (0.01, 0.1, 1.0, 10.0):
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("model", Ridge(solver="lsqr"))])
        harq.append(({"model__alpha": alpha}, estimator))
    lightgbm = []
    for params in [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.6},
    ]:
        estimator = lgb.LGBMRegressor(objective="regression_l1", random_state=seed, n_jobs=jobs, verbosity=-1)
        lightgbm.append((params, estimator))
    return harq, lightgbm


def fit_rv_pair(train, validation, test, features, settings, artifacts):
    seed, jobs = int(settings["random_seed"]), int(settings["n_jobs"])
    target = "target_rv_forward_120m"
    train_rv = train[train.quality_entry_ok & train[target].notna()].copy()
    val_rv = validation[validation.quality_entry_ok & validation[target].notna()].copy()
    test_rv = test[test.quality_entry_ok & test[target].notna()].copy()
    harq_cols = harq_features(features)
    harq_specs, lgb_specs = rv_model_specs(seed, jobs)
    transform = lambda y: np.log(np.maximum(y, 1e-8) ** 2)
    print("[1/8] Tuning Log-HARQ RV model", flush=True)
    harq_search = tune_regressors({"log_harq": harq_specs}, train_rv, harq_cols, target, settings, transform, "qlike")
    print("[2/8] Tuning LightGBM RV model", flush=True)
    lgb_search = tune_regressors({"lightgbm_rv": lgb_specs}, train_rv, features, target, settings, transform, "qlike")
    best_harq = clone(harq_specs[[x[0] for x in harq_specs].index(harq_search[0]["params"])][1]).set_params(**harq_search[0]["params"])
    best_lgb = clone(lgb_specs[[x[0] for x in lgb_specs].index(lgb_search[0]["params"])][1]).set_params(**lgb_search[0]["params"])
    fit_train = cap_rows(train_rv, int(settings["max_training_rows"]))
    best_harq.fit(numeric_matrix(fit_train, harq_cols), transform(fit_train[target].to_numpy()))
    best_lgb.fit(numeric_matrix(fit_train, features), transform(fit_train[target].to_numpy()))
    val_h = np.sqrt(np.exp(best_harq.predict(numeric_matrix(val_rv, harq_cols))))
    val_l = np.sqrt(np.exp(best_lgb.predict(numeric_matrix(val_rv, features))))
    weights = np.linspace(0, 1, 21)
    weight = min(weights, key=lambda w: qlike(val_rv[target], w * val_h + (1-w) * val_l))

    development = pd.concat([train_rv, val_rv]).sort_values("datetime")
    development_fit = cap_rows(development, int(settings["max_training_rows"]))
    final_harq = clone(best_harq).fit(numeric_matrix(development_fit, harq_cols), transform(development_fit[target].to_numpy()))
    final_lgb = clone(best_lgb).fit(numeric_matrix(development_fit, features), transform(development_fit[target].to_numpy()))
    test_h = np.sqrt(np.exp(final_harq.predict(numeric_matrix(test_rv, harq_cols))))
    test_l = np.sqrt(np.exp(final_lgb.predict(numeric_matrix(test_rv, features))))
    test_e = weight * test_h + (1-weight) * test_l
    persistence = test_rv["rv_trailing_120m"].fillna(development[target].median()).to_numpy()
    metrics = {
        "harq_search": harq_search, "lightgbm_search": lgb_search, "ensemble_harq_weight": float(weight),
        "test_harq_qlike": qlike(test_rv[target], test_h), "test_lightgbm_qlike": qlike(test_rv[target], test_l),
        "test_ensemble_qlike": qlike(test_rv[target], test_e), "test_persistence_qlike": qlike(test_rv[target], persistence),
        "test_ensemble_mae": float(mean_absolute_error(test_rv[target], test_e)), "test_rows": len(test_rv),
    }
    joblib.dump({"model": final_harq, "features": harq_cols}, artifacts / "rv_log_harq.joblib")
    joblib.dump({"model": final_lgb, "features": features}, artifacts / "rv_lightgbm.joblib")

    # Create strictly chronological RV predictions for downstream convergence models.
    result_parts = []
    print("[3/8] Generating walk-forward RV predictions", flush=True)
    folds = list(expanding_folds(train, max(4, int(settings["cv_splits"])), int(settings["horizon_minutes"])))
    for fit, score in folds:
        fit_rv = fit[fit.quality_entry_ok & fit[target].notna()]
        if len(fit_rv) < 100:
            continue
        fit_rv = cap_rows(fit_rv, int(settings["max_training_rows"]))
        mh = clone(best_harq).fit(numeric_matrix(fit_rv, harq_cols), transform(fit_rv[target].to_numpy()))
        ml = clone(best_lgb).fit(numeric_matrix(fit_rv, features), transform(fit_rv[target].to_numpy()))
        scored = score.copy()
        scored["rv_pred_120m"] = weight*np.sqrt(np.exp(mh.predict(numeric_matrix(scored, harq_cols)))) + (1-weight)*np.sqrt(np.exp(ml.predict(numeric_matrix(scored, features))))
        scored["rv_prediction_provenance"] = "oof"
        result_parts.append(scored)
    validation_pred = validation.copy()
    validation_pred["rv_pred_120m"] = weight*np.sqrt(np.exp(best_harq.predict(numeric_matrix(validation_pred, harq_cols)))) + (1-weight)*np.sqrt(np.exp(best_lgb.predict(numeric_matrix(validation_pred, features))))
    validation_pred["rv_prediction_provenance"] = "walk_forward"
    test_pred = test.copy()
    test_pred["rv_pred_120m"] = weight*np.sqrt(np.exp(final_harq.predict(numeric_matrix(test_pred, harq_cols)))) + (1-weight)*np.sqrt(np.exp(final_lgb.predict(numeric_matrix(test_pred, features))))
    test_pred["rv_prediction_provenance"] = "held_out"
    train_oof = pd.concat(result_parts).drop_duplicates(["datetime", "dte_flag", "expiry"], keep="last")
    return add_rv_prediction_convergence(train_oof), add_rv_prediction_convergence(validation_pred), add_rv_prediction_convergence(test_pred), metrics


def convergence_model_specs(seed: int, jobs: int, feature_count: int, gap_index: int):
    ridge_specs = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        estimator = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", RobustScaler()), ("model", Ridge(solver="lsqr"))])
        ridge_specs.append(({"model__alpha": alpha}, estimator))
    constraints = [0] * feature_count; constraints[gap_index] = -1
    xgb_specs = []
    for params in [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.02, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.7},
    ]:
        estimator = XGBRegressor(objective="reg:pseudohubererror", random_state=seed, n_jobs=jobs, monotone_constraints=tuple(constraints))
        xgb_specs.append((params, estimator))
    return ridge_specs, xgb_specs


def train_side(side, train, validation, test, features, settings, artifacts, exclude_over_60m=False):
    seed, jobs = int(settings["random_seed"]), int(settings["n_jobs"])
    delta_target = f"target_fixed_iv_change_{side}_120m"
    class_target = f"target_converged_to_rv_pred_{side}"
    gap_feature = f"entry_gap_to_rv_pred_{side}"
    eligible = lambda x: x[x.training_row_eligible & x[delta_target].notna() & x[class_target].notna()].copy()
    tr, va, te = eligible(train), eligible(validation), eligible(test)
    model_features = safe_feature_columns(tr)
    if exclude_over_60m:
        model_features = [
            name for name in model_features
            if not (
                (name.endswith("120m") or "_120m_" in name)
                and name != "rv_pred_120m"
                and not name.startswith("entry_gap_to_rv_pred_")
            )
        ]
    assert_leakage_safe_features(model_features)
    ridge_specs, xgb_specs = convergence_model_specs(seed, jobs, len(model_features), model_features.index(gap_feature))
    print(f"[4/8] Tuning {side} error-correction and monotonic XGBoost models", flush=True)
    ridge_search = tune_regressors({"error_correction_ridge": ridge_specs}, tr, model_features, delta_target, settings)
    xgb_search = tune_regressors({"monotonic_xgboost": xgb_specs}, tr, model_features, delta_target, settings)
    best_ridge = clone(ridge_specs[[x[0] for x in ridge_specs].index(ridge_search[0]["params"])][1]).set_params(**ridge_search[0]["params"])
    best_xgb = clone(xgb_specs[[x[0] for x in xgb_specs].index(xgb_search[0]["params"])][1]).set_params(**xgb_search[0]["params"])
    tr_fit = cap_rows(tr, int(settings["max_training_rows"]))
    best_ridge.fit(numeric_matrix(tr_fit, model_features), tr_fit[delta_target])
    best_xgb.fit(numeric_matrix(tr_fit, model_features), tr_fit[delta_target])
    pred_ridge_val = best_ridge.predict(numeric_matrix(va, model_features)); pred_xgb_val = best_xgb.predict(numeric_matrix(va, model_features))
    weights = np.linspace(0, 1, 21)
    blend_weight = min(weights, key=lambda w: mean_absolute_error(va[delta_target], w*pred_ridge_val+(1-w)*pred_xgb_val))

    classifier_specs = []
    for params in [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100},
    ]:
        classifier_specs.append((params, lgb.LGBMClassifier(random_state=seed, n_jobs=jobs, class_weight="balanced", verbosity=-1)))
    print(f"[5/8] Tuning {side} convergence classifier", flush=True)
    classifier_search = tune_classifier(classifier_specs, tr, model_features, class_target, settings)
    best_classifier = clone(classifier_specs[[x[0] for x in classifier_specs].index(classifier_search[0]["params"])][1]).set_params(**classifier_search[0]["params"])

    quantile_params = [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100},
    ]
    median_specs = [(p, lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=seed, n_jobs=jobs, verbosity=-1)) for p in quantile_params]
    print(f"[6/8] Tuning {side} quantile LightGBM", flush=True)
    quantile_search = tune_regressors({"lightgbm_quantile_median": median_specs}, tr, model_features, delta_target, settings)
    best_quantile_params = quantile_search[0]["params"]

    ng_params = [
        {"n_estimators": 75, "learning_rate": 0.03, "minibatch_frac": 0.8},
        {"n_estimators": 125, "learning_rate": 0.02, "minibatch_frac": 0.8},
    ]
    ng_settings = dict(settings); ng_settings["max_training_rows"] = int(settings["max_ngboost_rows"])
    print(f"[7/8] Tuning {side} NGBoost", flush=True)
    ng_search = tune_ngboost(ng_params, tr, model_features, delta_target, ng_settings, seed)

    development = pd.concat([tr, va]).sort_values("datetime")
    dev_fit = cap_rows(development, int(settings["max_training_rows"]))
    final_ridge = clone(best_ridge).fit(numeric_matrix(dev_fit, model_features), dev_fit[delta_target])
    final_xgb = clone(best_xgb).fit(numeric_matrix(dev_fit, model_features), dev_fit[delta_target])
    final_classifier = clone(best_classifier).fit(numeric_matrix(dev_fit, model_features), dev_fit[class_target].astype(int))
    quantile_models = {}
    for alpha in (0.1, 0.5, 0.9):
        model = lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=seed, n_jobs=jobs, verbosity=-1, **best_quantile_params)
        model.fit(numeric_matrix(dev_fit, model_features), dev_fit[delta_target]); quantile_models[str(alpha)] = model
    ng_fit = cap_rows(development, int(settings["max_ngboost_rows"]))
    ng_imputer = SimpleImputer(strategy="median")
    ng_x_fit = ng_imputer.fit_transform(numeric_matrix(ng_fit, model_features))
    best_ng = NGBRegressor(random_state=seed, verbose=False, **ng_search[0]["params"])
    best_ng.fit(ng_x_fit, ng_fit[delta_target].to_numpy())

    pr = final_ridge.predict(numeric_matrix(te, model_features)); px = final_xgb.predict(numeric_matrix(te, model_features))
    blend = blend_weight*pr+(1-blend_weight)*px
    probability = final_classifier.predict_proba(numeric_matrix(te, model_features))[:, 1]
    quantiles = {q: m.predict(numeric_matrix(te, model_features)) for q, m in quantile_models.items()}
    ng_pred = best_ng.predict(ng_imputer.transform(numeric_matrix(te, model_features)))
    baseline = np.zeros(len(te))
    print(f"[8/8] Evaluating and saving {side} models", flush=True)
    report = {
        "rows": {"train_oof": len(tr), "validation": len(va), "test": len(te)},
        "error_correction_search": ridge_search, "xgboost_search": xgb_search,
        "classifier_search": classifier_search, "quantile_search": quantile_search, "ngboost_search": ng_search,
        "blend_ridge_weight": float(blend_weight),
        "test": {
            "zero_change_baseline": regression_metrics(te[delta_target], baseline),
            "error_correction": regression_metrics(te[delta_target], pr),
            "monotonic_xgboost": regression_metrics(te[delta_target], px),
            "ensemble": regression_metrics(te[delta_target], blend),
            "ngboost_mean": regression_metrics(te[delta_target], ng_pred),
            "convergence_classifier": classification_metrics(te[class_target].astype(int), probability),
            "quantile_80pct_coverage": float(np.mean((te[delta_target] >= quantiles["0.1"]) & (te[delta_target] <= quantiles["0.9"]))),
        },
    }
    joblib.dump({"model": final_ridge, "features": model_features}, artifacts/f"{side}_error_correction.joblib")
    joblib.dump({"model": final_xgb, "features": model_features}, artifacts/f"{side}_monotonic_xgboost.joblib")
    joblib.dump({"model": final_classifier, "features": model_features}, artifacts/f"{side}_convergence_classifier.joblib")
    joblib.dump({"models": quantile_models, "features": model_features}, artifacts/f"{side}_quantile_lightgbm.joblib")
    joblib.dump({"model": best_ng, "imputer": ng_imputer, "features": model_features}, artifacts/f"{side}_ngboost.joblib")
    predictions = te[["datetime", "dte_flag", "expiry", "entry_strike", delta_target, class_target, "rv_pred_120m", gap_feature]].copy()
    predictions[f"pred_delta_{side}_ensemble"] = blend
    predictions[f"pred_convergence_probability_{side}"] = probability
    predictions[f"pred_delta_{side}_q10"] = quantiles["0.1"]
    predictions[f"pred_delta_{side}_q50"] = quantiles["0.5"]
    predictions[f"pred_delta_{side}_q90"] = quantiles["0.9"]
    return report, predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exclude-over-60m", action="store_true")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    config = load_config(args.config); settings = config["training"]
    exclude_over_60m = args.exclude_over_60m or int(settings.get("exclude_features_over_minutes", 0)) == 60
    artifacts = Path(args.artifacts_dir).resolve() if args.artifacts_dir else Path(config["artifacts"]); artifacts.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(config["data"]["feature_csv"], parse_dates=["datetime", "exit_datetime", "expiry"], low_memory=False)
    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    features = manifest["features"]
    if exclude_over_60m:
        features = [name for name in features if not (name.endswith("120m") or "_120m_" in name)]
    assert_leakage_safe_features(features)
    splits = purged_chronological_split(frame, float(settings["train_fraction"]), float(settings["validation_fraction"]), int(settings["horizon_minutes"]))
    train, validation, test = splits["train"], splits["validation"], splits["test"]
    train_rv, val_rv, test_rv, rv_report = fit_rv_pair(train, validation, test, features, settings, artifacts)
    call_report, call_predictions = train_side("call", train_rv, val_rv, test_rv, features, settings, artifacts, exclude_over_60m)
    put_report, put_predictions = train_side("put", train_rv, val_rv, test_rv, features, settings, artifacts, exclude_over_60m)
    predictions = call_predictions.merge(put_predictions, on=["datetime", "dte_flag", "expiry", "entry_strike", "rv_pred_120m"], how="outer")
    predictions.to_csv(artifacts/"held_out_predictions.csv", index=False)
    report = {"rv": rv_report, "call": call_report, "put": put_report}
    (artifacts/"metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    qlike_gain = 1 - rv_report["test_ensemble_qlike"] / rv_report["test_persistence_qlike"]
    lines = [
        "# 120-minute fixed-strike IV → RV model results", "", "## Experiment", "",
        f"- Test period: {predictions['datetime'].min()} through {predictions['datetime'].max()}.",
        f"- Held-out rows: {len(predictions):,}.",
        f"- Three expanding walk-forward folds with a {settings['horizon_minutes']}-minute purge.",
        f"- Compute caps: {settings['max_training_rows']:,} standard-model rows and {settings['max_ngboost_rows']:,} NGBoost rows.",
        "", "## RV forecast", "",
        "| Model | Test QLIKE |", "|---|---:|",
        f"| Persistence | {rv_report['test_persistence_qlike']:.6f} |",
        f"| Log-HARQ | {rv_report['test_harq_qlike']:.6f} |",
        f"| LightGBM | {rv_report['test_lightgbm_qlike']:.6f} |",
        f"| Ensemble | {rv_report['test_ensemble_qlike']:.6f} |", "",
        f"The RV ensemble reduces QLIKE by {qlike_gain:.2%} versus persistence.",
    ]
    model_labels = (("zero_change_baseline", "Zero-change baseline"), ("error_correction", "Error-correction Ridge"), ("monotonic_xgboost", "Monotonic XGBoost"), ("ensemble", "Ridge/XGBoost ensemble"), ("ngboost_mean", "NGBoost mean"))
    for side, result in (("Call", call_report), ("Put", put_report)):
        lines += ["", f"## Fixed-strike {side.lower()} IV change", "", "| Model | MAE | RMSE | R² | Direction accuracy |", "|---|---:|---:|---:|---:|"]
        for key, label in model_labels:
            metric = result["test"][key]
            direction = "n/a" if key == "zero_change_baseline" else f"{metric['direction_accuracy']:.2%}"
            lines.append(f"| {label} | {metric['mae']:.6f} | {metric['rmse']:.6f} | {metric['r2']:.4f} | {direction} |")
        baseline = result["test"]["zero_change_baseline"]["mae"]
        best_key = min((key for key, _ in model_labels), key=lambda key: result["test"][key]["mae"])
        best_label = dict(model_labels)[best_key]
        gain = 1 - result["test"][best_key]["mae"] / baseline
        classifier = result["test"]["convergence_classifier"]
        lines += ["", f"- Best test MAE: {best_label}, {gain:.2%} improvement versus zero change.", f"- Convergence classifier: {classifier['balanced_accuracy']:.2%} balanced accuracy; {classifier['roc_auc']:.4f} ROC AUC.", f"- Nominal 80% quantile coverage: {result['test']['quantile_80pct_coverage']:.2%}."]
    lines += ["", "## Interpretation", "", "A model is not production-ready merely because one metric beats a baseline. Check MAE, RMSE, R², direction, interval calibration, stability across regimes, and trading costs. NGBoost emitted numerical scale-overflow warnings on this dataset; prefer the quantile models for uncertainty until that specification is stabilized."]
    (artifacts/"REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"artifacts": str(artifacts), "report": report}, indent=2, default=str))


if __name__ == "__main__":
    main()
