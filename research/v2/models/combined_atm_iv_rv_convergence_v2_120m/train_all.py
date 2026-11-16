from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import ngboost
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from ngboost import NGBRegressor
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor


PROJECT = Path(__file__).resolve().parents[2] / "research" / "combined_atm_iv_rv_convergence_v2"
sys.path.insert(0, str(PROJECT))
from feature_pipeline import (  # noqa: E402
    add_rv_prediction_convergence,
    assert_leakage_safe_features,
    purged_chronological_split,
    safe_feature_columns,
)


DELTA_TARGET = "target_fixed_combined_iv_change_120m"
CLASS_TARGET = "target_converged_to_rv_pred_combined"
GAP_FEATURE = "entry_gap_to_rv_pred_combined"
RV_TARGET = "target_rv_forward_120m"


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key in ("feature_csv", "feature_manifest"):
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    config["artifacts"] = str((path.parent / "artifacts").resolve())
    return config


def qlike(actual_vol, predicted_vol) -> float:
    actual_var = np.maximum(np.asarray(actual_vol, float) ** 2, 1e-12)
    predicted_var = np.maximum(np.asarray(predicted_vol, float) ** 2, 1e-12)
    ratio = actual_var / predicted_var
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def regression_metrics(y, prediction) -> dict:
    y = np.asarray(y, float)
    prediction = np.asarray(prediction, float)
    return {
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(mean_squared_error(y, prediction) ** 0.5),
        "r2": float(r2_score(y, prediction)),
        "direction_accuracy": float(np.mean(np.sign(y) == np.sign(prediction))),
    }


def classification_metrics(y, probability) -> dict:
    y = np.asarray(y, int)
    probability = np.clip(np.asarray(probability, float), 1e-6, 1 - 1e-6)
    prediction = probability >= 0.5
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "log_loss": float(log_loss(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan"),
    }


def numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def cap_rows(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    if len(frame) <= maximum:
        return frame
    return frame.iloc[np.linspace(0, len(frame) - 1, maximum, dtype=int)]


def expanding_folds(frame: pd.DataFrame, splits: int):
    ordered = frame.sort_values("datetime")
    stamps = np.sort(ordered["datetime"].dropna().unique())
    boundaries = np.linspace(0.40, 1.0, splits + 1)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        start = pd.Timestamp(stamps[min(int(len(stamps) * left), len(stamps) - 1)])
        end = pd.Timestamp(stamps[min(int(len(stamps) * right), len(stamps) - 1)])
        fit = ordered[(ordered["datetime"] < start) & (ordered["exit_datetime"] < start)]
        score = ordered[(ordered["datetime"] >= start) & (ordered["datetime"] <= end)]
        if len(fit) and len(score):
            yield fit, score


def tune_regressors(name_to_specs, frame, features, target, settings, transform=None, loss="mae"):
    results = []
    for model_name, specs in name_to_specs.items():
        for params, estimator in specs:
            fold_scores = []
            for fit, score in expanding_folds(frame, int(settings["cv_splits"])):
                fit = cap_rows(fit.dropna(subset=[target]), int(settings["max_training_rows"]))
                score = score.dropna(subset=[target])
                model = clone(estimator).set_params(**params)
                y_fit = transform(fit[target].to_numpy()) if transform else fit[target].to_numpy()
                model.fit(numeric_matrix(fit, features), y_fit)
                prediction = model.predict(numeric_matrix(score, features))
                if transform:
                    prediction = np.sqrt(np.exp(prediction))
                value = qlike(score[target], prediction) if loss == "qlike" else mean_absolute_error(score[target], prediction)
                fold_scores.append(float(value))
            results.append({"model": model_name, "params": params, "cv_loss": float(np.mean(fold_scores))})
    return sorted(results, key=lambda row: row["cv_loss"])


def tune_classifier(specs, frame, features, target, settings):
    results = []
    for params, estimator in specs:
        fold_scores = []
        for fit, score in expanding_folds(frame, int(settings["cv_splits"])):
            fit = cap_rows(fit.dropna(subset=[target]), int(settings["max_training_rows"]))
            score = score.dropna(subset=[target])
            model = clone(estimator).set_params(**params)
            model.fit(numeric_matrix(fit, features), fit[target].astype(int))
            probability = model.predict_proba(numeric_matrix(score, features))[:, 1]
            fold_scores.append(float(log_loss(score[target].astype(int), probability)))
        results.append({"params": params, "cv_log_loss": float(np.mean(fold_scores))})
    return sorted(results, key=lambda row: row["cv_log_loss"])


def tune_ngboost(param_grid, frame, features, target, settings, seed):
    """Tune NGBoost on standardized fold targets to stabilize distribution scale."""
    results = []
    for params in param_grid:
        fold_scores = []
        for fit, score in expanding_folds(frame, int(settings["cv_splits"])):
            fit = cap_rows(fit.dropna(subset=[target]), int(settings["max_training_rows"]))
            score = score.dropna(subset=[target])
            imputer = SimpleImputer(strategy="median")
            x_fit = imputer.fit_transform(numeric_matrix(fit, features))
            x_score = imputer.transform(numeric_matrix(score, features))
            center = float(fit[target].mean())
            scale = max(float(fit[target].std()), 1e-8)
            model = NGBRegressor(random_state=seed, verbose=False, **params)
            model.fit(x_fit, (fit[target].to_numpy() - center) / scale)
            prediction = center + scale * model.predict(x_score)
            fold_scores.append(float(mean_absolute_error(score[target], prediction)))
        results.append({"model": "ngboost", "params": params, "cv_loss": float(np.mean(fold_scores))})
    return sorted(results, key=lambda row: row["cv_loss"])


def harq_features(features: list[str]) -> list[str]:
    return [
        name for name in features
        if name.startswith(("rv_trailing_", "rq_trailing_", "spot_return_"))
        or name in {
            "entry_atm_iv_combined", "surface_atm_iv_combined", "surface_skew_combined",
            "surface_smile_combined", "day_of_week", "time_sin", "time_cos", "days_to_expiry",
        }
    ]


def rv_specs(seed: int, jobs: int):
    harq = []
    for alpha in (0.01, 0.1, 1.0, 10.0):
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", Ridge(solver="lsqr")),
        ])
        harq.append(({"model__alpha": alpha}, estimator))
    lightgbm = []
    for params in [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 500, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 50, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 63, "min_child_samples": 80, "subsample": 0.8, "colsample_bytree": 0.6},
    ]:
        lightgbm.append((params, lgb.LGBMRegressor(objective="regression_l1", random_state=seed, n_jobs=jobs, verbosity=-1)))
    return harq, lightgbm


def best_from_specs(search, specs):
    params = search[0]["params"]
    estimator = next(estimator for candidate, estimator in specs if candidate == params)
    return clone(estimator).set_params(**params)


def fit_rv_models(train, validation, test, features, settings, artifacts):
    seed = int(settings["random_seed"])
    jobs = int(settings["n_jobs"])
    eligible = lambda part: part[part["quality_entry_ok"] & part[RV_TARGET].notna()].copy()
    train_rv, val_rv, test_rv = eligible(train), eligible(validation), eligible(test)
    harq_cols = harq_features(features)
    harq_grid, lightgbm_grid = rv_specs(seed, jobs)
    transform = lambda y: np.log(np.maximum(y, 1e-8) ** 2)
    print("[1/9] Tuning Log-HARQ RV", flush=True)
    harq_search = tune_regressors({"log_harq": harq_grid}, train_rv, harq_cols, RV_TARGET, settings, transform, "qlike")
    print("[2/9] Tuning LightGBM RV", flush=True)
    lightgbm_search = tune_regressors({"lightgbm_rv": lightgbm_grid}, train_rv, features, RV_TARGET, settings, transform, "qlike")
    best_harq = best_from_specs(harq_search, harq_grid)
    best_lightgbm = best_from_specs(lightgbm_search, lightgbm_grid)
    train_fit = cap_rows(train_rv, int(settings["max_training_rows"]))
    best_harq.fit(numeric_matrix(train_fit, harq_cols), transform(train_fit[RV_TARGET].to_numpy()))
    best_lightgbm.fit(numeric_matrix(train_fit, features), transform(train_fit[RV_TARGET].to_numpy()))
    val_harq = np.sqrt(np.exp(best_harq.predict(numeric_matrix(val_rv, harq_cols))))
    val_lgb = np.sqrt(np.exp(best_lightgbm.predict(numeric_matrix(val_rv, features))))
    candidate_weights = np.linspace(0, 1, 21)
    harq_weight = min(candidate_weights, key=lambda w: qlike(val_rv[RV_TARGET], w * val_harq + (1 - w) * val_lgb))

    development = pd.concat([train_rv, val_rv]).sort_values("datetime")
    development_fit = cap_rows(development, int(settings["max_training_rows"]))
    final_harq = clone(best_harq).fit(numeric_matrix(development_fit, harq_cols), transform(development_fit[RV_TARGET].to_numpy()))
    final_lightgbm = clone(best_lightgbm).fit(numeric_matrix(development_fit, features), transform(development_fit[RV_TARGET].to_numpy()))
    test_harq = np.sqrt(np.exp(final_harq.predict(numeric_matrix(test_rv, harq_cols))))
    test_lgb = np.sqrt(np.exp(final_lightgbm.predict(numeric_matrix(test_rv, features))))
    test_ensemble = harq_weight * test_harq + (1 - harq_weight) * test_lgb
    persistence = test_rv["rv_trailing_120m"].fillna(development[RV_TARGET].median()).to_numpy()
    report = {
        "harq_search": harq_search,
        "lightgbm_search": lightgbm_search,
        "ensemble_harq_weight": float(harq_weight),
        "test_harq_qlike": qlike(test_rv[RV_TARGET], test_harq),
        "test_lightgbm_qlike": qlike(test_rv[RV_TARGET], test_lgb),
        "test_ensemble_qlike": qlike(test_rv[RV_TARGET], test_ensemble),
        "test_persistence_qlike": qlike(test_rv[RV_TARGET], persistence),
        "test_ensemble_mae": float(mean_absolute_error(test_rv[RV_TARGET], test_ensemble)),
        "test_rows": int(len(test_rv)),
    }
    joblib.dump({"model": final_harq, "features": harq_cols}, artifacts / "rv_log_harq.joblib")
    joblib.dump({"model": final_lightgbm, "features": features}, artifacts / "rv_lightgbm.joblib")

    print("[3/9] Generating out-of-sample RV predictions", flush=True)
    oof_parts = []
    for fit, score in expanding_folds(train, max(4, int(settings["cv_splits"]))):
        fit_rv = fit[fit["quality_entry_ok"] & fit[RV_TARGET].notna()]
        if len(fit_rv) < 100:
            continue
        fit_rv = cap_rows(fit_rv, int(settings["max_training_rows"]))
        model_h = clone(best_harq).fit(numeric_matrix(fit_rv, harq_cols), transform(fit_rv[RV_TARGET].to_numpy()))
        model_l = clone(best_lightgbm).fit(numeric_matrix(fit_rv, features), transform(fit_rv[RV_TARGET].to_numpy()))
        scored = score.copy()
        scored["rv_pred_120m"] = (
            harq_weight * np.sqrt(np.exp(model_h.predict(numeric_matrix(scored, harq_cols))))
            + (1 - harq_weight) * np.sqrt(np.exp(model_l.predict(numeric_matrix(scored, features))))
        )
        scored["rv_prediction_provenance"] = "oof"
        oof_parts.append(scored)
    validation_pred = validation.copy()
    validation_pred["rv_pred_120m"] = (
        harq_weight * np.sqrt(np.exp(best_harq.predict(numeric_matrix(validation_pred, harq_cols))))
        + (1 - harq_weight) * np.sqrt(np.exp(best_lightgbm.predict(numeric_matrix(validation_pred, features))))
    )
    validation_pred["rv_prediction_provenance"] = "walk_forward"
    test_pred = test.copy()
    test_pred["rv_pred_120m"] = (
        harq_weight * np.sqrt(np.exp(final_harq.predict(numeric_matrix(test_pred, harq_cols))))
        + (1 - harq_weight) * np.sqrt(np.exp(final_lightgbm.predict(numeric_matrix(test_pred, features))))
    )
    test_pred["rv_prediction_provenance"] = "held_out"
    train_oof = pd.concat(oof_parts).drop_duplicates(["datetime", "dte_flag", "expiry"], keep="last")
    return (
        add_rv_prediction_convergence(train_oof),
        add_rv_prediction_convergence(validation_pred),
        add_rv_prediction_convergence(test_pred),
        report,
    )


def point_model_specs(seed: int, jobs: int, feature_count: int, gap_index: int):
    ridge = []
    for alpha in (0.1, 1.0, 10.0, 100.0):
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", RobustScaler()),
            ("model", Ridge(solver="lsqr")),
        ])
        ridge.append(({"model__alpha": alpha}, estimator))
    constraints = [0] * feature_count
    constraints[gap_index] = -1
    xgboost_grid = []
    for params in [
        {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "min_child_weight": 20, "subsample": 0.8, "colsample_bytree": 0.7},
        {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.02, "min_child_weight": 10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.03, "min_child_weight": 30, "subsample": 0.8, "colsample_bytree": 0.7},
    ]:
        xgboost_grid.append((params, XGBRegressor(
            objective="reg:pseudohubererror", random_state=seed, n_jobs=jobs,
            monotone_constraints=tuple(constraints),
        )))
    lightgbm_grid = []
    for params in [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100},
    ]:
        lightgbm_grid.append((params, lgb.LGBMRegressor(
            objective="regression_l1", random_state=seed, n_jobs=jobs, verbosity=-1,
        )))
    return ridge, xgboost_grid, lightgbm_grid


def select_blend_weights(y, ridge_pred, xgb_pred, lgb_pred):
    best = None
    for ridge_weight in np.linspace(0, 1, 11):
        for xgb_weight in np.linspace(0, 1 - ridge_weight, int(round((1 - ridge_weight) * 10)) + 1):
            lgb_weight = 1 - ridge_weight - xgb_weight
            prediction = ridge_weight * ridge_pred + xgb_weight * xgb_pred + lgb_weight * lgb_pred
            loss = mean_absolute_error(y, prediction)
            candidate = (float(loss), float(ridge_weight), float(xgb_weight), float(lgb_weight))
            if best is None or candidate[0] < best[0]:
                best = candidate
    return {"ridge": best[1], "xgboost": best[2], "lightgbm": best[3], "validation_mae": best[0]}


def train_combined_models(train, validation, test, settings, artifacts, exclude_over_60m=False):
    seed = int(settings["random_seed"])
    jobs = int(settings["n_jobs"])
    eligible = lambda part: part[
        part["training_row_eligible"] & part[DELTA_TARGET].notna() & part[CLASS_TARGET].notna()
    ].copy()
    train_model, validation_model, test_model = eligible(train), eligible(validation), eligible(test)
    model_features = safe_feature_columns(train_model)
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
    if GAP_FEATURE not in model_features:
        raise AssertionError("Out-of-sample IV-to-RV gap feature is missing")
    ridge_grid, xgb_grid, lgb_grid = point_model_specs(
        seed, jobs, len(model_features), model_features.index(GAP_FEATURE)
    )
    print("[4/9] Tuning Ridge, monotonic XGBoost, and direct LightGBM", flush=True)
    ridge_search = tune_regressors({"error_correction_ridge": ridge_grid}, train_model, model_features, DELTA_TARGET, settings)
    xgb_search = tune_regressors({"monotonic_xgboost": xgb_grid}, train_model, model_features, DELTA_TARGET, settings)
    lgb_search = tune_regressors({"direct_lightgbm": lgb_grid}, train_model, model_features, DELTA_TARGET, settings)
    best_ridge = best_from_specs(ridge_search, ridge_grid)
    best_xgb = best_from_specs(xgb_search, xgb_grid)
    best_lgb = best_from_specs(lgb_search, lgb_grid)
    train_fit = cap_rows(train_model, int(settings["max_training_rows"]))
    best_ridge.fit(numeric_matrix(train_fit, model_features), train_fit[DELTA_TARGET])
    best_xgb.fit(numeric_matrix(train_fit, model_features), train_fit[DELTA_TARGET])
    best_lgb.fit(numeric_matrix(train_fit, model_features), train_fit[DELTA_TARGET])
    validation_matrix = numeric_matrix(validation_model, model_features)
    validation_ridge = best_ridge.predict(validation_matrix)
    validation_xgb = best_xgb.predict(validation_matrix)
    validation_lgb = best_lgb.predict(validation_matrix)
    blend_weights = select_blend_weights(
        validation_model[DELTA_TARGET], validation_ridge, validation_xgb, validation_lgb
    )

    classifier_grid = []
    for params in [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100},
    ]:
        classifier_grid.append((params, lgb.LGBMClassifier(
            random_state=seed, n_jobs=jobs, class_weight="balanced", verbosity=-1,
        )))
    print("[5/9] Tuning convergence classifier", flush=True)
    classifier_search = tune_classifier(classifier_grid, train_model, model_features, CLASS_TARGET, settings)
    best_classifier = best_from_specs(classifier_search, classifier_grid)

    quantile_params = [
        {"n_estimators": 300, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 80},
        {"n_estimators": 500, "learning_rate": 0.02, "num_leaves": 31, "min_child_samples": 50},
        {"n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100},
    ]
    quantile_grid = [
        (params, lgb.LGBMRegressor(
            objective="quantile", alpha=0.5, random_state=seed, n_jobs=jobs, verbosity=-1,
        ))
        for params in quantile_params
    ]
    print("[6/9] Tuning quantile LightGBM", flush=True)
    quantile_search = tune_regressors(
        {"lightgbm_quantile_median": quantile_grid}, train_model, model_features, DELTA_TARGET, settings
    )
    best_quantile_params = quantile_search[0]["params"]

    ng_params = [
        {"n_estimators": 75, "learning_rate": 0.03, "minibatch_frac": 0.8},
        {"n_estimators": 125, "learning_rate": 0.02, "minibatch_frac": 0.8},
    ]
    ng_settings = dict(settings)
    ng_settings["max_training_rows"] = int(settings["max_ngboost_rows"])
    print("[7/9] Tuning standardized NGBoost", flush=True)
    ngboost_search = tune_ngboost(ng_params, train_model, model_features, DELTA_TARGET, ng_settings, seed)

    print("[8/9] Refitting selected models on train plus validation", flush=True)
    development = pd.concat([train_model, validation_model]).sort_values("datetime")
    development_fit = cap_rows(development, int(settings["max_training_rows"]))
    final_ridge = clone(best_ridge).fit(numeric_matrix(development_fit, model_features), development_fit[DELTA_TARGET])
    final_xgb = clone(best_xgb).fit(numeric_matrix(development_fit, model_features), development_fit[DELTA_TARGET])
    final_lgb = clone(best_lgb).fit(numeric_matrix(development_fit, model_features), development_fit[DELTA_TARGET])
    final_classifier = clone(best_classifier).fit(
        numeric_matrix(development_fit, model_features), development_fit[CLASS_TARGET].astype(int)
    )
    quantile_models = {}
    for alpha in (0.1, 0.5, 0.9):
        model = lgb.LGBMRegressor(
            objective="quantile", alpha=alpha, random_state=seed, n_jobs=jobs,
            verbosity=-1, **best_quantile_params,
        )
        model.fit(numeric_matrix(development_fit, model_features), development_fit[DELTA_TARGET])
        quantile_models[str(alpha)] = model
    ng_fit = cap_rows(development, int(settings["max_ngboost_rows"]))
    ng_imputer = SimpleImputer(strategy="median")
    ng_x_fit = ng_imputer.fit_transform(numeric_matrix(ng_fit, model_features))
    ng_center = float(ng_fit[DELTA_TARGET].mean())
    ng_scale = max(float(ng_fit[DELTA_TARGET].std()), 1e-8)
    final_ngboost = NGBRegressor(random_state=seed, verbose=False, **ngboost_search[0]["params"])
    final_ngboost.fit(ng_x_fit, (ng_fit[DELTA_TARGET].to_numpy() - ng_center) / ng_scale)

    print("[9/9] Evaluating untouched test period and saving artifacts", flush=True)
    test_matrix = numeric_matrix(test_model, model_features)
    pred_ridge = final_ridge.predict(test_matrix)
    pred_xgb = final_xgb.predict(test_matrix)
    pred_lgb = final_lgb.predict(test_matrix)
    pred_ensemble = (
        blend_weights["ridge"] * pred_ridge
        + blend_weights["xgboost"] * pred_xgb
        + blend_weights["lightgbm"] * pred_lgb
    )
    probability = final_classifier.predict_proba(test_matrix)[:, 1]
    quantiles = {key: model.predict(test_matrix) for key, model in quantile_models.items()}
    pred_ngboost = ng_center + ng_scale * final_ngboost.predict(ng_imputer.transform(test_matrix))
    baseline = np.zeros(len(test_model))
    report = {
        "rows": {
            "train_oof": int(len(train_model)),
            "validation": int(len(validation_model)),
            "test": int(len(test_model)),
        },
        "feature_count": int(len(model_features)),
        "ridge_search": ridge_search,
        "xgboost_search": xgb_search,
        "direct_lightgbm_search": lgb_search,
        "classifier_search": classifier_search,
        "quantile_search": quantile_search,
        "ngboost_search": ngboost_search,
        "ensemble_weights": blend_weights,
        "test": {
            "zero_change_baseline": regression_metrics(test_model[DELTA_TARGET], baseline),
            "error_correction_ridge": regression_metrics(test_model[DELTA_TARGET], pred_ridge),
            "monotonic_xgboost": regression_metrics(test_model[DELTA_TARGET], pred_xgb),
            "direct_lightgbm": regression_metrics(test_model[DELTA_TARGET], pred_lgb),
            "three_model_ensemble": regression_metrics(test_model[DELTA_TARGET], pred_ensemble),
            "ngboost_mean": regression_metrics(test_model[DELTA_TARGET], pred_ngboost),
            "convergence_classifier": classification_metrics(test_model[CLASS_TARGET].astype(int), probability),
            "quantile_80pct_coverage": float(np.mean(
                (test_model[DELTA_TARGET] >= quantiles["0.1"])
                & (test_model[DELTA_TARGET] <= quantiles["0.9"])
            )),
        },
    }
    joblib.dump({"model": final_ridge, "features": model_features}, artifacts / "combined_error_correction_ridge.joblib")
    joblib.dump({"model": final_xgb, "features": model_features}, artifacts / "combined_monotonic_xgboost.joblib")
    joblib.dump({"model": final_lgb, "features": model_features}, artifacts / "combined_direct_lightgbm.joblib")
    joblib.dump({"models": {"ridge": final_ridge, "xgboost": final_xgb, "lightgbm": final_lgb}, "weights": blend_weights, "features": model_features}, artifacts / "combined_three_model_ensemble.joblib")
    joblib.dump({"model": final_classifier, "features": model_features}, artifacts / "combined_convergence_classifier.joblib")
    joblib.dump({"models": quantile_models, "features": model_features}, artifacts / "combined_quantile_lightgbm.joblib")
    joblib.dump({"model": final_ngboost, "imputer": ng_imputer, "target_center": ng_center, "target_scale": ng_scale, "features": model_features}, artifacts / "combined_ngboost.joblib")
    predictions = test_model[[
        "datetime", "dte_flag", "expiry", "entry_strike", "entry_atm_iv_combined",
        "exit_fixed_atm_iv_combined", DELTA_TARGET, CLASS_TARGET, "rv_pred_120m", GAP_FEATURE,
    ]].copy()
    predictions["pred_delta_zero"] = baseline
    predictions["pred_delta_ridge"] = pred_ridge
    predictions["pred_delta_xgboost"] = pred_xgb
    predictions["pred_delta_lightgbm"] = pred_lgb
    predictions["pred_delta_ensemble"] = pred_ensemble
    predictions["pred_delta_ngboost"] = pred_ngboost
    predictions["pred_convergence_probability"] = probability
    predictions["pred_delta_q10"] = quantiles["0.1"]
    predictions["pred_delta_q50"] = quantiles["0.5"]
    predictions["pred_delta_q90"] = quantiles["0.9"]
    return report, predictions


def create_report(rv_report, model_report, predictions, settings) -> str:
    qlike_gain = 1 - rv_report["test_ensemble_qlike"] / rv_report["test_persistence_qlike"]
    test_metrics = model_report["test"]
    model_labels = [
        ("zero_change_baseline", "Zero-change baseline"),
        ("error_correction_ridge", "Error-correction Ridge"),
        ("monotonic_xgboost", "Monotonic XGBoost"),
        ("direct_lightgbm", "Direct LightGBM"),
        ("three_model_ensemble", "Three-model ensemble"),
        ("ngboost_mean", "NGBoost mean"),
    ]
    best_key, best_label = min(model_labels, key=lambda pair: test_metrics[pair[0]]["mae"])
    baseline_mae = test_metrics["zero_change_baseline"]["mae"]
    best_gain = 1 - test_metrics[best_key]["mae"] / baseline_mae
    lines = [
        "# V2 combined ATM IV to RV model results", "", "## Experiment", "",
        f"- Test period: {predictions['datetime'].min()} through {predictions['datetime'].max()}.",
        f"- Held-out rows: {len(predictions):,}.",
        f"- Point-model features: {model_report['feature_count']:,}.",
        f"- Three expanding walk-forward folds with a {settings['horizon_minutes']}-minute purge.",
        f"- Compute caps: {settings['max_training_rows']:,} ordinary-model rows and {settings['max_ngboost_rows']:,} NGBoost rows.",
        "", "## RV forecast", "", "| Model | Test QLIKE |", "|---|---:|",
        f"| Persistence | {rv_report['test_persistence_qlike']:.6f} |",
        f"| Log-HARQ | {rv_report['test_harq_qlike']:.6f} |",
        f"| LightGBM | {rv_report['test_lightgbm_qlike']:.6f} |",
        f"| Ensemble | {rv_report['test_ensemble_qlike']:.6f} |", "",
        f"The RV ensemble reduces QLIKE by {qlike_gain:.2%} versus persistence.",
        "", "## Combined fixed-strike IV change", "",
        "| Model | MAE | RMSE | R² | Direction accuracy |", "|---|---:|---:|---:|---:|",
    ]
    for key, label in model_labels:
        metric = test_metrics[key]
        direction = "n/a" if key == "zero_change_baseline" else f"{metric['direction_accuracy']:.2%}"
        lines.append(f"| {label} | {metric['mae']:.6f} | {metric['rmse']:.6f} | {metric['r2']:.4f} | {direction} |")
    classifier = test_metrics["convergence_classifier"]
    lines += [
        "",
        f"- Best held-out MAE: {best_label}; {best_gain:.2%} improvement versus zero change.",
        f"- Validation-selected ensemble weights: Ridge {model_report['ensemble_weights']['ridge']:.0%}, XGBoost {model_report['ensemble_weights']['xgboost']:.0%}, LightGBM {model_report['ensemble_weights']['lightgbm']:.0%}.",
        f"- Convergence classifier: {classifier['balanced_accuracy']:.2%} balanced accuracy and {classifier['roc_auc']:.4f} ROC AUC.",
        f"- Nominal 80% quantile coverage: {test_metrics['quantile_80pct_coverage']:.2%}.",
        "", "A lower MAE alone does not establish production readiness. Compare RMSE, R², interval calibration, regime stability, and trading P&L before deployment.",
        "", "Hyperparameter trials and full-precision metrics are stored in `metrics.json`.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V2 combined ATM-IV/RV models")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--exclude-over-60m", action="store_true")
    parser.add_argument("--artifacts-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["training"]
    exclude_over_60m = args.exclude_over_60m or int(settings.get("exclude_features_over_minutes", 0)) == 60
    artifacts = Path(args.artifacts_dir).resolve() if args.artifacts_dir else Path(config["artifacts"])
    artifacts.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(
        config["data"]["feature_csv"],
        parse_dates=["datetime", "exit_datetime", "expiry"],
        low_memory=False,
    )
    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    features = manifest["features"]
    if exclude_over_60m:
        features = [name for name in features if not (name.endswith("120m") or "_120m_" in name)]
    assert_leakage_safe_features(features)
    split = purged_chronological_split(
        frame,
        float(settings["train_fraction"]),
        float(settings["validation_fraction"]),
        int(settings["horizon_minutes"]),
    )
    train_rv, validation_rv, test_rv, rv_report = fit_rv_models(
        split["train"], split["validation"], split["test"], features, settings, artifacts
    )
    model_report, predictions = train_combined_models(
        train_rv, validation_rv, test_rv, settings, artifacts, exclude_over_60m
    )
    predictions.to_csv(artifacts / "held_out_predictions.csv", index=False)
    report = {"rv": rv_report, "combined_iv": model_report}
    (artifacts / "metrics.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    (artifacts / "REPORT.md").write_text(
        create_report(rv_report, model_report, predictions, settings), encoding="utf-8"
    )
    run_manifest = {
        "goal": "V2 combined call-put fixed-strike ATM IV to RV convergence",
        "data": config["data"],
        "training": settings,
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lgb.__version__,
            "xgboost": xgboost.__version__,
            "ngboost": ngboost.__version__,
        },
        "test_period": {
            "start": str(predictions["datetime"].min()),
            "end": str(predictions["datetime"].max()),
            "rows": int(len(predictions)),
        },
    }
    (artifacts / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    print(json.dumps({"artifacts": str(artifacts), "report": report}, indent=2, default=str))


if __name__ == "__main__":
    main()
