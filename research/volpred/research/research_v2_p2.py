"""Preregistered P2 predictive evaluation with no pseudo-holdout access.

Only development and 2023-01..2024-06 walk-forward rows are selected from the
common panel.  This module never computes strategy P&L.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import sys
import uuid
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import duckdb
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.research_v2_splits import (
    DEVELOPMENT,
    WALK_FORWARD,
    assert_calendar_partition,
    assert_purge_embargo,
    cpcv_8_choose_2,
    purged_embargoed_train,
    read_restricted_parquet,
    split_protocol_hash,
)


PANEL = ROOT / "outputs" / "research_v2" / "common" / "predictive_dataset_120m.parquet"
COMMON_MANIFEST = ROOT / "outputs" / "research_v2" / "common" / "manifest.json"
COMMON_HASHES = ROOT / "outputs" / "research_v2" / "common" / "SHA256SUMS"
SPLITS_PATH = ROOT / "SPLITS.md"
OUT = ROOT / "outputs" / "research_v2" / "p2"
MODEL_DIR = ROOT / "model" / "research_v2_p2"
MODEL_PATH = MODEL_DIR / "p2_hist_gradient_boosting_120m.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "artifact_metadata.json"
P0_MODEL_PATH = MODEL_DIR / "p0_ridge_120m.joblib"
P0_METADATA_PATH = MODEL_DIR / "p0_artifact_metadata.json"
FROZEN_SPEC_PATH = ROOT / "FROZEN_PREDICTIVE_SPEC_V2.json"
MODEL_HASHES_PATH = MODEL_DIR / "SHA256SUMS"
PSEUDO_MARKER_PATH = MODEL_DIR / "PSEUDO_HOLDOUT_OPENED.json"
PSEUDO_PREDICTIONS_PATH = OUT / "pseudo_holdout_once_predictions.parquet"
PSEUDO_RESULT_PATH = OUT / "pseudo_holdout_once_result.json"
ROOT_TRIALS = ROOT / "TRIALS.csv"
ROOT_TRIAL_LOCK = ROOT / "TRIALS.csv.lock"
SEED = 7319
EPS = 1e-12
HAC_LAG = 120
PSEUDO_START = pd.Timestamp("2024-07-01", tz="Asia/Kolkata")
APPROVED_SPLITS_SHA256 = "4409030241bf656af200c811a50a39387c49a8a4e3a4b01700b06fde1c79b48c"
APPROVED_DATASET_SHA256 = "02850acfebd8442bfb7c518f174abc94402e88181a72975f2db4986d35d8ded0"
APPROVED_MANIFEST_SHA256 = "3778739fa77c6470fbb788788c3ed93571cc7c1b4f5b90ad230a955d82ca18b6"
ROOT_TRIAL_COLUMNS = [
    "trial_id", "hypothesis", "registered_at", "status", "structure",
    "horizon_min", "decision_time", "dte_band", "parameter_grid",
    "outcome_opened", "notes",
]

P0_FEATURES = ["p0_log_rv_5", "p0_log_rv_15", "p0_log_rv_60"]
P2_FEATURES = P0_FEATURES + [
    "p1_log_rq_15", "p1_log_rq_60",
    "p1_downside_rv_15", "p1_upside_rv_15",
    "p1_downside_rv_60", "p1_upside_rv_60",
    "p1_jump_rv_60", "p1_abs_return", "p1_tod_sin", "p1_tod_cos",
]
P0_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
P2_GRID = tuple(
    {"max_depth": depth, "learning_rate": rate, "l2_regularization": l2}
    for depth, rate, l2 in itertools.product((2, 3), (0.03, 0.05), (1.0, 10.0))
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_approved_inputs() -> dict[str, str]:
    """Fail unless the frozen split, manifest, and dataset hashes all match."""
    recorded: dict[str, str] = {}
    for line in COMMON_HASHES.read_text(encoding="utf-8").splitlines():
        digest, name = line.strip().split(" *", maxsplit=1)
        recorded[name] = digest.lower()
    splits_hash = split_protocol_hash(SPLITS_PATH)
    manifest_hash = file_sha256(COMMON_MANIFEST)
    dataset_hash = file_sha256(PANEL)
    manifest = json.loads(COMMON_MANIFEST.read_text(encoding="utf-8"))
    checks = {
        "splits_sha256": (splits_hash, APPROVED_SPLITS_SHA256),
        "manifest_sha256": (manifest_hash, APPROVED_MANIFEST_SHA256),
        "dataset_sha256": (dataset_hash, APPROVED_DATASET_SHA256),
        "manifest_splits_sha256": (str(manifest.get("splits_sha256", "")), APPROVED_SPLITS_SHA256),
        "manifest_dataset_sha256": (str(manifest.get("dataset_sha256", "")), APPROVED_DATASET_SHA256),
        "inventory_manifest_sha256": (recorded.get(COMMON_MANIFEST.name, ""), APPROVED_MANIFEST_SHA256),
        "inventory_dataset_sha256": (recorded.get(PANEL.name, ""), APPROVED_DATASET_SHA256),
    }
    failures = {name: values for name, values in checks.items() if values[0].lower() != values[1].lower()}
    if failures:
        raise ValueError(f"unapproved research-v2 provenance: {failures}")
    return {name: actual for name, (actual, _) in checks.items()}


def load_permitted_panel() -> pd.DataFrame:
    """Read only approved development and walk-forward rows via shared guards."""
    verify_approved_inputs()
    columns = [
        "decision_time", "decision_date", "label_end_at", "split", "rv_fwd_120", "target",
        *P2_FEATURES,
    ]
    frame = read_restricted_parquet(
        PANEL, allowed_splits=[DEVELOPMENT, WALK_FORWARD], columns=columns,
        context="P2 model research",
    )
    if frame.empty:
        raise ValueError("approved panel view is empty")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True).dt.tz_convert("Asia/Kolkata")
    frame["label_end_at"] = pd.to_datetime(frame["label_end_at"], utc=True).dt.tz_convert("Asia/Kolkata")
    frame["decision_date"] = frame["decision_time"].dt.normalize()
    assert_calendar_partition(frame)
    if not (frame["decision_time"] < PSEUDO_START).all():
        raise AssertionError("unauthorized timestamp reached P2 namespace")
    required = ["rv_fwd_120", "target", *P2_FEATURES]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("permitted panel contains nonfinite model inputs")
    return frame.sort_values("decision_time", kind="stable").reset_index(drop=True)


def development_groups(frame: pd.DataFrame) -> tuple[list[np.ndarray], np.ndarray]:
    dates = frame.loc[frame["split"].eq("development"), "decision_date"].drop_duplicates().sort_values().to_numpy()
    groups, _ = cpcv_8_choose_2(dates)
    return [np.asarray(group) for group in groups], dates


def cpcv_pairs() -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(8), 2))


def purged_train_test(
    frame: pd.DataFrame,
    all_sessions: Iterable[pd.Timestamp | np.datetime64],
    test_dates: Iterable[pd.Timestamp | np.datetime64],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    sessions = pd.DatetimeIndex(pd.to_datetime(list(all_sessions)))
    tests = pd.DatetimeIndex(pd.to_datetime(list(test_dates)))
    if sessions.tz is not None:
        sessions = sessions.tz_convert("Asia/Kolkata").tz_localize(None)
    if tests.tz is not None:
        tests = tests.tz_convert("Asia/Kolkata").tz_localize(None)
    sessions = sessions.normalize()
    tests = tests.normalize()
    if not sessions.isin(tests).any():
        raise ValueError("test dates do not occur in the session calendar")
    frame_dates = frame["decision_date"].dt.tz_localize(None).dt.normalize()
    test_mask = frame_dates.isin(tests)
    test = frame.loc[test_mask].copy()
    if test.empty:
        raise ValueError("test dates do not occur in the frame")
    train = purged_embargoed_train(frame, test_dates=tests, all_sessions=sessions)
    if train.empty or test.empty:
        raise ValueError("purge/embargo produced an empty train or test set")
    assert_purge_embargo(train, test, all_sessions=sessions)
    removed = frame.loc[~test_mask & ~frame.index.isin(train.index)]
    removed_dates = removed["decision_date"].drop_duplicates().sort_values()
    return train, test, {
        "purged_or_embargoed_rows": int(len(removed)),
        "removed_dates": "|".join(str(x.date()) for x in removed_dates),
    }


def qlike(actual: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    ratio = np.clip(actual, EPS, None) / np.clip(forecast, EPS, None)
    return ratio - np.log(ratio) - 1.0


def log_mse(log_actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean((log_actual - np.log(np.clip(forecast, EPS, None))) ** 2))


def smear_from_training(log_actual: np.ndarray, fitted_log: np.ndarray) -> float:
    residual = np.asarray(log_actual) - np.asarray(fitted_log)
    return float(0.5 * np.var(residual, ddof=1)) if len(residual) > 1 else 0.0


def positive_forecast(log_forecast: np.ndarray, smear: float) -> tuple[np.ndarray, int]:
    raw_log = np.asarray(log_forecast, dtype=float) + smear
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        raw = np.exp(raw_log)
    failed = ~np.isfinite(raw_log) | ~np.isfinite(raw) | (raw <= 0)
    # Replacement occurs only after the raw failure mask is retained.
    safe_log = np.where(np.isfinite(raw_log), np.clip(raw_log, -50.0, 50.0), math.log(EPS))
    safe = np.exp(safe_log)
    return safe, int(failed.sum())


def fit_p0(train: pd.DataFrame, alpha: float) -> tuple[Any, float]:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(train[P0_FEATURES], train["target"])
    smear = smear_from_training(train["target"].to_numpy(), model.predict(train[P0_FEATURES]))
    return model, smear


def fit_p2(train: pd.DataFrame, params: dict[str, Any]) -> tuple[HistGradientBoostingRegressor, float]:
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=200, early_stopping=False,
        max_depth=int(params["max_depth"]), learning_rate=float(params["learning_rate"]),
        l2_regularization=float(params["l2_regularization"]), random_state=SEED,
    )
    model.fit(train[P2_FEATURES], train["target"])
    smear = smear_from_training(train["target"].to_numpy(), model.predict(train[P2_FEATURES]))
    return model, smear


def frozen_p2_params() -> dict[str, Any]:
    return {"max_depth": 3, "learning_rate": 0.05, "l2_regularization": 1.0}


def fit_frozen_p2(frame: pd.DataFrame) -> tuple[HistGradientBoostingRegressor, float]:
    if frame.empty:
        raise ValueError("authorized training frame is empty")
    allowed = {DEVELOPMENT, WALK_FORWARD}
    if not set(frame["split"].astype(str).unique()).issubset(allowed):
        raise PermissionError("frozen P2 fit received an unauthorized split")
    if not frame["label_end_at"].gt(frame["decision_time"]).all():
        raise ValueError("invalid label_end_at in frozen training frame")
    return fit_p2(frame, frozen_p2_params())


def metric_record(test: pd.DataFrame, forecast: np.ndarray, failures: int) -> dict[str, Any]:
    actual = test["rv_fwd_120"].to_numpy(float)
    return {
        "qlike": float(np.mean(qlike(actual, forecast))),
        "log_mse": log_mse(test["target"].to_numpy(float), forecast),
        "calibration_failures": int(failures),
        "mean_actual_forecast_ratio": float(np.mean(actual / np.clip(forecast, EPS, None))),
    }


def inner_select_p0(
    train: pd.DataFrame,
    context: str,
    trials: list[dict[str, Any]],
) -> float:
    dates = train["decision_date"].drop_duplicates().sort_values().to_numpy()
    split_at = max(1, int(math.floor(0.8 * len(dates))))
    if split_at >= len(dates):
        raise ValueError("insufficient dates for inner P0 selection")
    inner_train, inner_test, metadata = purged_train_test(train, dates, dates[split_at:])
    candidates: list[tuple[float, float, float]] = []
    for alpha in P0_ALPHAS:
        trial = {
            "trial_id": f"{context}-P0-alpha-{alpha:g}", "stage": "inner_selection",
            "fold_id": context, "model": "P0_Ridge", "params": json.dumps({"alpha": alpha}),
            "train_rows": len(inner_train), "test_rows": len(inner_test), **metadata,
        }
        try:
            model, smear = fit_p0(inner_train, alpha)
            forecast, failures = positive_forecast(model.predict(inner_test[P0_FEATURES]), smear)
            metrics = metric_record(inner_test, forecast, failures)
            trial.update(status="ok", error="", smear=smear, **metrics)
            candidates.append((metrics["qlike"], metrics["log_mse"], -alpha))
        except Exception as exc:  # every failure remains in the trial inventory
            trial.update(status="failed", error=repr(exc), smear=np.nan, qlike=np.nan,
                         log_mse=np.nan, calibration_failures=np.nan,
                         mean_actual_forecast_ratio=np.nan)
        trials.append(trial)
    if not candidates:
        raise RuntimeError(f"all P0 candidates failed in {context}")
    return float(-min(candidates)[2])


def _candidate_id(params: dict[str, Any]) -> str:
    return f"d{params['max_depth']}_lr{params['learning_rate']:.2f}_l2{params['l2_regularization']:g}"


def run_cpcv(
    frame: pd.DataFrame, run_id: str, exported_local_ids: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    dev = frame[frame["split"].eq("development")].copy()
    sessions = dev["decision_date"].drop_duplicates().sort_values().to_numpy()
    groups, fold_specs = cpcv_8_choose_2(sessions)
    trials: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for fold_number, fold_spec in enumerate(fold_specs, start=1):
        pair = fold_spec.test_groups
        fold_id = f"CPCV-{fold_number:02d}-g{pair[0]}g{pair[1]}"
        test_dates = fold_spec.test_dates
        train, test, metadata = purged_train_test(dev, sessions, test_dates)
        alpha = inner_select_p0(train, fold_id, trials)
        try:
            p0, smear = fit_p0(train, alpha)
            p0_forecast, failures = positive_forecast(p0.predict(test[P0_FEATURES]), smear)
            p0_metrics = metric_record(test, p0_forecast, failures)
            trials.append({
                "trial_id": f"{fold_id}-P0-selected", "stage": "outer_cpcv", "fold_id": fold_id,
                "model": "P0_Ridge", "params": json.dumps({"alpha": alpha}),
                "train_rows": len(train), "test_rows": len(test), **metadata,
                "status": "ok", "error": "", "smear": smear, **p0_metrics,
            })
        except Exception as exc:
            trials.append({
                "trial_id": f"{fold_id}-P0-selected", "stage": "outer_cpcv", "fold_id": fold_id,
                "model": "P0_Ridge", "params": json.dumps({"alpha": alpha}),
                "train_rows": len(train), "test_rows": len(test), **metadata,
                "status": "failed", "error": repr(exc), "smear": np.nan, "qlike": np.nan,
                "log_mse": np.nan, "calibration_failures": np.nan,
                "mean_actual_forecast_ratio": np.nan,
            })
            raise
        for params in P2_GRID:
            candidate = _candidate_id(params)
            trial = {
                "trial_id": f"{fold_id}-P2-{candidate}", "stage": "outer_cpcv", "fold_id": fold_id,
                "model": "P2_HistGradientBoosting", "candidate": candidate,
                "params": json.dumps(params, sort_keys=True), "train_rows": len(train),
                "test_rows": len(test), **metadata,
            }
            try:
                model, smear2 = fit_p2(train, params)
                forecast, failures2 = positive_forecast(model.predict(test[P2_FEATURES]), smear2)
                metrics = metric_record(test, forecast, failures2)
                trial.update(status="ok", error="", smear=smear2, **metrics)
                fold_rows.append({
                    "fold_id": fold_id, "candidate": candidate, **params,
                    **{f"p2_{key}": value for key, value in metrics.items()},
                    "p0_qlike": p0_metrics["qlike"], "p0_log_mse": p0_metrics["log_mse"],
                    "p0_alpha": alpha,
                })
            except Exception as exc:
                trial.update(status="failed", error=repr(exc), smear=np.nan, qlike=np.nan,
                             log_mse=np.nan, calibration_failures=np.nan,
                             mean_actual_forecast_ratio=np.nan)
            trials.append(trial)
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trials).to_csv(OUT / "trials.csv", index=False)
        append_root_trials(trials, run_id, exported_local_ids)

    folds = pd.DataFrame(fold_rows)
    folds["qlike_rank"] = folds.groupby("fold_id")["p2_qlike"].rank(method="average")
    folds["log_mse_rank"] = folds.groupby("fold_id")["p2_log_mse"].rank(method="average")
    summary = folds.groupby(
        ["candidate", "max_depth", "learning_rate", "l2_regularization"], as_index=False
    ).agg(
        median_qlike=("p2_qlike", "median"), median_log_mse=("p2_log_mse", "median"),
        median_qlike_rank=("qlike_rank", "median"), median_log_mse_rank=("log_mse_rank", "median"),
        folds_beating_p0=("p2_qlike", lambda values: 0), successful_folds=("fold_id", "nunique"),
    )
    beat_counts = folds.assign(beat=folds["p2_qlike"] < folds["p0_qlike"]).groupby("candidate")["beat"].sum()
    summary["folds_beating_p0"] = summary["candidate"].map(beat_counts).astype(int)
    summary = summary.sort_values(
        ["median_qlike_rank", "median_log_mse_rank", "max_depth", "l2_regularization", "learning_rate"],
        ascending=[True, True, True, False, True], kind="stable",
    ).reset_index(drop=True)
    selected = summary.iloc[0]
    neighbor_mask = (
        ((summary["max_depth"] != selected["max_depth"]) &
         summary["learning_rate"].eq(selected["learning_rate"]) &
         summary["l2_regularization"].eq(selected["l2_regularization"]))
        | (summary["max_depth"].eq(selected["max_depth"]) &
           (summary["learning_rate"] != selected["learning_rate"]) &
           summary["l2_regularization"].eq(selected["l2_regularization"]))
        | (summary["max_depth"].eq(selected["max_depth"]) &
           summary["learning_rate"].eq(selected["learning_rate"]) &
           (summary["l2_regularization"] != selected["l2_regularization"]))
    )
    neighbors = summary.loc[neighbor_mask].copy()
    neighbors["relative_qlike_gap"] = (
        neighbors["median_qlike"] - float(selected["median_qlike"])
    ).abs() / abs(float(selected["median_qlike"]))
    plateau_pass = len(neighbors) == 3 and bool(neighbors["relative_qlike_gap"].le(0.02).all())
    summary["selected_by_rank"] = summary["candidate"].eq(selected["candidate"])
    summary["selected_plateau_pass"] = False
    summary.loc[summary["selected_by_rank"], "selected_plateau_pass"] = plateau_pass
    summary["development_gate_pass"] = (
        summary["selected_by_rank"] & summary["selected_plateau_pass"]
        & summary["folds_beating_p0"].ge(20) & summary["successful_folds"].eq(28)
    )
    return folds, summary, neighbors, trials


def hac_dm(loss_difference: np.ndarray, lag: int = HAC_LAG) -> dict[str, float]:
    values = np.asarray(loss_difference, dtype=float)
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 3:
        return {"mean_difference": np.nan, "statistic": np.nan, "one_sided_pvalue": np.nan}
    centered = values - values.mean()
    effective_lag = min(lag, n - 1)
    long_run = float(np.dot(centered, centered) / n)
    for step in range(1, effective_lag + 1):
        weight = 1.0 - step / (effective_lag + 1.0)
        covariance = float(np.dot(centered[step:], centered[:-step]) / n)
        long_run += 2.0 * weight * covariance
    standard_error = math.sqrt(max(long_run, EPS) / n)
    statistic = float(values.mean() / standard_error)
    pvalue = float(0.5 * math.erfc(-statistic / math.sqrt(2.0)))
    return {"mean_difference": float(values.mean()), "statistic": statistic, "one_sided_pvalue": pvalue}


def run_walk_forward(
    frame: pd.DataFrame,
    selected_params: dict[str, Any],
    trials: list[dict[str, Any]],
    run_id: str,
    exported_local_ids: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validation = frame[frame["split"].eq("walk_forward_validation")].copy()
    months = validation["decision_time"].dt.tz_localize(None).dt.to_period("M")
    outputs: list[pd.DataFrame] = []
    raw_p0_calibration_failures = 0
    raw_p2_calibration_failures = 0
    for month in months.drop_duplicates().sort_values():
        test = validation.loc[months.eq(month)].copy()
        first_test = test["decision_time"].min()
        train = frame.loc[frame["label_end_at"].lt(first_test)].copy()
        if train.empty or not train["label_end_at"].lt(first_test).all():
            raise AssertionError("walk-forward label-end purge failed")
        context = f"WF-{month}"
        alpha = inner_select_p0(train, context, trials)
        p0, smear0 = fit_p0(train, alpha)
        pred0, fail0 = positive_forecast(p0.predict(test[P0_FEATURES]), smear0)
        metrics0 = metric_record(test, pred0, fail0)
        raw_p0_calibration_failures += fail0
        trials.append({
            "trial_id": f"{context}-P0-selected", "stage": "walk_forward", "fold_id": context,
            "model": "P0_Ridge", "params": json.dumps({"alpha": alpha}),
            "train_rows": len(train), "test_rows": len(test), "purged_rows": 0,
            "embargo_dates": "not_applicable_future_of_test", "test_blocks": 1,
            "status": "ok", "error": "", "smear": smear0, **metrics0,
        })
        p2, smear2 = fit_p2(train, selected_params)
        pred2, fail2 = positive_forecast(p2.predict(test[P2_FEATURES]), smear2)
        metrics2 = metric_record(test, pred2, fail2)
        raw_p2_calibration_failures += fail2
        trials.append({
            "trial_id": f"{context}-P2-selected", "stage": "walk_forward", "fold_id": context,
            "model": "P2_HistGradientBoosting", "candidate": _candidate_id(selected_params),
            "params": json.dumps(selected_params, sort_keys=True), "train_rows": len(train),
            "test_rows": len(test), "purged_rows": 0,
            "embargo_dates": "not_applicable_future_of_test", "test_blocks": 1,
            "status": "ok", "error": "", "smear": smear2, **metrics2,
        })
        low, high = train["p0_log_rv_60"].quantile([1 / 3, 2 / 3]).to_numpy(float)
        regime = np.where(test["p0_log_rv_60"].lt(low), "low",
                          np.where(test["p0_log_rv_60"].gt(high), "high", "middle"))
        outputs.append(pd.DataFrame({
            "decision_time": test["decision_time"].to_numpy(),
            "label_end_at": test["label_end_at"].to_numpy(),
            "month": str(month), "subperiod": np.where(
                test["decision_time"].dt.year.eq(2023),
                np.where(test["decision_time"].dt.month.le(6), "2023_H1", "2023_H2"), "2024_H1"
            ),
            "regime": regime, "actual_variance": test["rv_fwd_120"].to_numpy(float),
            "p0_forecast": pred0, "p2_forecast": pred2,
        }))
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(trials).to_csv(OUT / "trials.csv", index=False)
        append_root_trials(trials, run_id, exported_local_ids)
    predictions = pd.concat(outputs, ignore_index=True)
    predictions["p0_qlike"] = qlike(predictions["actual_variance"].to_numpy(), predictions["p0_forecast"].to_numpy())
    predictions["p2_qlike"] = qlike(predictions["actual_variance"].to_numpy(), predictions["p2_forecast"].to_numpy())
    predictions["loss_improvement"] = predictions["p0_qlike"] - predictions["p2_qlike"]
    predictions["p0_log_error_sq"] = (
        np.log(predictions["actual_variance"].clip(lower=EPS)) - np.log(predictions["p0_forecast"].clip(lower=EPS))
    ) ** 2
    predictions["p2_log_error_sq"] = (
        np.log(predictions["actual_variance"].clip(lower=EPS)) - np.log(predictions["p2_forecast"].clip(lower=EPS))
    ) ** 2

    def grouped_metrics(group: pd.DataFrame) -> pd.Series:
        p0q, p2q = group["p0_qlike"].mean(), group["p2_qlike"].mean()
        return pd.Series({
            "rows": len(group), "p0_qlike": p0q, "p2_qlike": p2q,
            "qlike_improvement_pct": 100.0 * (p0q - p2q) / p0q,
            "p0_log_mse": group["p0_log_error_sq"].mean(),
            "p2_log_mse": group["p2_log_error_sq"].mean(),
            "p0_mean_actual_forecast_ratio": (group["actual_variance"] / group["p0_forecast"]).mean(),
            "p2_mean_actual_forecast_ratio": (group["actual_variance"] / group["p2_forecast"]).mean(),
        })
    subperiod = predictions.groupby("subperiod", sort=True).apply(grouped_metrics, include_groups=False).reset_index()
    regime = predictions.groupby("regime", sort=True).apply(grouped_metrics, include_groups=False).reset_index()
    overall = grouped_metrics(predictions).to_dict()
    dm = hac_dm((predictions["p2_qlike"] - predictions["p0_qlike"]).to_numpy(), HAC_LAG)
    monthly = predictions.groupby("month", as_index=False).agg(
        rows=("decision_time", "size"), p0_qlike=("p0_qlike", "mean"),
        p2_qlike=("p2_qlike", "mean"), loss_improvement=("loss_improvement", "sum"),
    )
    total_improvement = float(predictions["loss_improvement"].sum())
    max_month_share = (
        float(monthly["loss_improvement"].clip(lower=0).max() / total_improvement)
        if total_improvement > 0 else float("inf")
    )
    gate = {
        "lower_qlike_every_subperiod": bool((subperiod["p2_qlike"] < subperiod["p0_qlike"]).all()),
        "aggregate_qlike_improvement_at_least_2pct": bool(overall["qlike_improvement_pct"] >= 2.0),
        "hac_dm_one_sided_p_below_005": bool(dm["one_sided_pvalue"] < 0.05),
        "positive_forecasts_no_more_failures_than_p0": bool(
            raw_p2_calibration_failures <= raw_p0_calibration_failures
        ),
        "p0_raw_calibration_failures": int(raw_p0_calibration_failures),
        "p2_raw_calibration_failures": int(raw_p2_calibration_failures),
        "max_month_improvement_share_below_35pct": bool(max_month_share <= 0.35),
        "max_month_improvement_share": max_month_share,
        "hac_dm": dm,
        "overall": overall,
    }
    gate["p2_vs_p0_predictive_gate_pass"] = bool(all(
        value for key, value in gate.items() if key in {
            "lower_qlike_every_subperiod", "aggregate_qlike_improvement_at_least_2pct",
            "hac_dm_one_sided_p_below_005", "positive_forecasts_no_more_failures_than_p0",
            "max_month_improvement_share_below_35pct",
        }
    ))
    return predictions, subperiod, regime, {"gate": gate, "monthly": monthly}


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    connection = duckdb.connect()
    try:
        connection.register("frame", frame)
        target = str(path).replace("'", "''")
        connection.execute(f"COPY frame TO '{target}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    finally:
        connection.close()


def root_trial_rows(trials: list[dict[str, Any]], run_id: str) -> list[dict[str, Any]]:
    rows = []
    for trial in trials:
        model = str(trial.get("model", ""))
        notes = {
            "namespace": "outputs/research_v2/p2",
            "stage": trial.get("stage"),
            "fold_id": trial.get("fold_id"),
            "train_rows": trial.get("train_rows"),
            "test_rows": trial.get("test_rows"),
            "qlike": trial.get("qlike"),
            "log_mse": trial.get("log_mse"),
            "calibration_failures": trial.get("calibration_failures"),
            "error": trial.get("error", ""),
            "run_id": run_id,
            "local_trial_id": trial["trial_id"],
        }
        rows.append({
            "trial_id": f"P2::{run_id}::{trial['trial_id']}",
            "hypothesis": "P0" if model == "P0_Ridge" else "P2",
            "registered_at": "2026-10-13T18:15:00+05:30",
            "status": "fit_ok" if trial.get("status") == "ok" else "fit_failed",
            "structure": "predictive_120m",
            "horizon_min": 120,
            "decision_time": trial.get("fold_id", ""),
            "dte_band": "NA",
            "parameter_grid": trial.get("params", ""),
            "outcome_opened": True,
            "notes": json.dumps(notes, sort_keys=True, default=str),
        })
    return rows


@contextmanager
def locked_root_trial_registry():
    """Lock the first byte of the root registry while appending exact-schema rows."""
    handle = ROOT_TRIALS.open("r+", encoding="utf-8", newline="")
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield handle
    finally:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def append_root_trials(
    trials: list[dict[str, Any]], run_id: str, exported_local_ids: set[str]
) -> None:
    pending = [trial for trial in trials if trial["trial_id"] not in exported_local_ids]
    if not pending:
        return
    rows = root_trial_rows(pending, run_id)
    with locked_root_trial_registry() as handle:
        handle.seek(0)
        header = next(csv.reader([handle.readline().rstrip("\r\n")]))
        if header != ROOT_TRIAL_COLUMNS:
            raise ValueError(f"root TRIALS.csv schema mismatch: {header}")
        handle.seek(0, os.SEEK_END)
        writer = csv.DictWriter(handle, fieldnames=ROOT_TRIAL_COLUMNS, lineterminator="\n")
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    exported_local_ids.update(trial["trial_id"] for trial in pending)
    export_path = OUT / "trials_root_append.csv"
    new_export = pd.DataFrame(root_trial_rows(trials, run_id), columns=ROOT_TRIAL_COLUMNS)
    if export_path.exists():
        existing_export = pd.read_csv(export_path, dtype=str)
        new_export = pd.concat([existing_export, new_export], ignore_index=True)
        new_export = new_export.drop_duplicates("trial_id", keep="first")
    new_export.to_csv(export_path, index=False)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def claim_pseudo_holdout_once(marker_path: Path = PSEUDO_MARKER_PATH) -> dict[str, Any]:
    """Atomically consume the one-shot right before any pseudo-row scan."""
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    opening = {
        "opening_id": "pseudo-open-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
        "status": "STARTED_AND_CONSUMED", "consumed": True,
        "spec_sha256": file_sha256(FROZEN_SPEC_PATH),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    try:
        descriptor = os.open(marker_path, flags, 0o600)
    except FileExistsError as exc:
        raise PermissionError(f"pseudo-holdout one-shot already consumed: {marker_path}") from exc
    try:
        data = (json.dumps(opening, indent=2) + "\n").encode("utf-8")
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return opening


def pseudo_stability_verdict(predictions: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen predictive-only stability thresholds."""
    required = {"decision_time", "actual_variance", "p0_forecast", "p2_forecast"}
    missing = sorted(required.difference(predictions.columns))
    if missing or predictions.empty:
        raise ValueError(f"missing/empty pseudo scoring inputs: {missing}")
    ordered = predictions.sort_values("decision_time", kind="stable").reset_index(drop=True).copy()
    numeric = ordered[["actual_variance", "p0_forecast", "p2_forecast"]].to_numpy(float)
    if not np.isfinite(numeric).all() or (numeric <= 0).any():
        raise ValueError("nonfinite or nonpositive pseudo scoring values")
    midpoint = len(ordered) // 2
    if midpoint == 0:
        raise ValueError("pseudo scoring requires two nonempty chronological halves")
    ordered["chronological_half"] = np.where(np.arange(len(ordered)) < midpoint, "first", "second")
    actual = ordered["actual_variance"].to_numpy(float)
    ordered["p0_qlike"] = qlike(actual, ordered["p0_forecast"].to_numpy(float))
    ordered["p2_qlike"] = qlike(actual, ordered["p2_forecast"].to_numpy(float))
    halves = ordered.groupby("chronological_half", sort=False).agg(
        rows=("decision_time", "size"), p0_qlike=("p0_qlike", "mean"),
        p2_qlike=("p2_qlike", "mean"),
    ).reset_index()
    p0_qlike = float(ordered["p0_qlike"].mean())
    p2_qlike = float(ordered["p2_qlike"].mean())
    improvement = float(100.0 * (p0_qlike - p2_qlike) / p0_qlike) if p0_qlike > 0 else -math.inf
    dm = hac_dm((ordered["p2_qlike"] - ordered["p0_qlike"]).to_numpy(), HAC_LAG)
    p0_log_mse = log_mse(np.log(actual), ordered["p0_forecast"].to_numpy(float))
    p2_log_mse = log_mse(np.log(actual), ordered["p2_forecast"].to_numpy(float))
    p0_failures = int(predictions.attrs.get("p0_raw_calibration_failures", 0))
    p2_failures = int(predictions.attrs.get("p2_raw_calibration_failures", 0))
    checks = {
        "p2_lower_qlike_each_chronological_half": bool((halves["p2_qlike"] < halves["p0_qlike"]).all()),
        "aggregate_qlike_improvement_at_least_2pct": bool(improvement >= 2.0),
        "hac_dm_one_sided_p_below_005": bool(dm["one_sided_pvalue"] < 0.05),
        "p2_log_mse_no_higher_than_p0": bool(p2_log_mse <= p0_log_mse),
        "p2_raw_calibration_failures_no_more_than_p0": bool(p2_failures <= p0_failures),
    }
    verdict = "research_stable" if all(checks.values()) else "research_failed"
    return {
        "verdict": verdict, "checks": checks, "chronological_halves": halves.to_dict("records"),
        "aggregate": {"p0_qlike": p0_qlike, "p2_qlike": p2_qlike,
                      "qlike_improvement_pct": improvement, "p0_log_mse": p0_log_mse,
                      "p2_log_mse": p2_log_mse, "p0_raw_calibration_failures": p0_failures,
                      "p2_raw_calibration_failures": p2_failures},
        "hac_dm_lag120": dm,
    }


def _write_model_hash_inventory(paths: list[Path]) -> None:
    lines = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        relative = path.relative_to(ROOT).as_posix()
        lines.append(f"{file_sha256(path)} *{relative}\n")
    MODEL_HASHES_PATH.write_text("".join(sorted(lines)), encoding="utf-8")


def freeze_authorized_predictive_artifact() -> dict[str, Any]:
    """Fit P2 and its P0 comparator on authorized pre-pseudo rows and freeze."""
    frozen_targets = [MODEL_PATH, MODEL_METADATA_PATH, P0_MODEL_PATH, P0_METADATA_PATH,
                      FROZEN_SPEC_PATH, MODEL_HASHES_PATH]
    existing = [str(path) for path in frozen_targets if path.exists()]
    if existing:
        raise FileExistsError(f"frozen P2 artifact already exists: {existing}")
    provenance = verify_approved_inputs()
    frame = load_permitted_panel()
    if set(frame["split"].unique()) != {DEVELOPMENT, WALK_FORWARD}:
        raise PermissionError("frozen training set must contain exactly development and walk-forward rows")
    runner_hash = file_sha256(Path(__file__))
    model, smear = fit_frozen_p2(frame)
    p0_selection_trials: list[dict[str, Any]] = []
    p0_alpha = inner_select_p0(frame, "FROZEN-P0-INNER", p0_selection_trials)
    p0_model, p0_smear = fit_p0(frame, p0_alpha)
    run_id = "freeze-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    payload = {
        "model": model,
        "model_family": "P2_HistGradientBoostingRegressor",
        "horizon_minutes": 120,
        "features": tuple(P2_FEATURES),
        "params": frozen_p2_params(),
        "max_iter": 200,
        "early_stopping": False,
        "seed": SEED,
        "smearing_log_adjustment": smear,
        "trained_splits": (DEVELOPMENT, WALK_FORWARD),
        "training_rows": int(len(frame)),
        "training_max_decision_time": str(frame["decision_time"].max()),
        "training_max_label_end_at": str(frame["label_end_at"].max()),
        "dataset_sha256": APPROVED_DATASET_SHA256,
        "splits_sha256": APPROVED_SPLITS_SHA256,
        "runner_sha256": runner_hash,
        "run_id": run_id,
        "sklearn_version": sklearn.__version__,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary_model = MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(payload, temporary_model)
    os.replace(temporary_model, MODEL_PATH)
    model_hash = file_sha256(MODEL_PATH)
    metadata = {key: value for key, value in payload.items() if key != "model"}
    metadata.update({
        "artifact_file": MODEL_PATH.name,
        "artifact_sha256": model_hash,
        "common_manifest_sha256": APPROVED_MANIFEST_SHA256,
        "provenance_verified": True,
        "pseudo_holdout_accessed": False,
        "strategy_pnl_computed": False,
    })
    _atomic_json(MODEL_METADATA_PATH, metadata)

    p0_payload = {
        "model": p0_model, "model_family": "P0_StandardScaler_Ridge",
        "horizon_minutes": 120, "features": tuple(P0_FEATURES), "params": {"alpha": p0_alpha},
        "smearing_log_adjustment": p0_smear,
        "selection": "training-only purged final-20%-sessions inner validation",
        "selection_candidates": P0_ALPHAS, "trained_splits": (DEVELOPMENT, WALK_FORWARD),
        "training_rows": int(len(frame)), "training_max_decision_time": str(frame["decision_time"].max()),
        "training_max_label_end_at": str(frame["label_end_at"].max()),
        "dataset_sha256": APPROVED_DATASET_SHA256, "splits_sha256": APPROVED_SPLITS_SHA256,
        "runner_sha256": runner_hash, "run_id": run_id, "sklearn_version": sklearn.__version__,
    }
    temporary_p0 = P0_MODEL_PATH.with_suffix(".joblib.tmp")
    joblib.dump(p0_payload, temporary_p0)
    os.replace(temporary_p0, P0_MODEL_PATH)
    p0_hash = file_sha256(P0_MODEL_PATH)
    p0_metadata = {key: value for key, value in p0_payload.items() if key != "model"}
    p0_metadata.update({"artifact_file": P0_MODEL_PATH.name, "artifact_sha256": p0_hash,
                        "common_manifest_sha256": APPROVED_MANIFEST_SHA256,
                        "provenance_verified": True, "pseudo_holdout_accessed": False,
                        "strategy_pnl_computed": False})
    _atomic_json(P0_METADATA_PATH, p0_metadata)

    freeze_trial = {
        "trial_id": "FROZEN-P2-AUTHORIZED-FIT",
        "stage": "frozen_authorized_fit",
        "fold_id": "development_plus_walk_forward",
        "model": "P2_HistGradientBoosting",
        "params": json.dumps(frozen_p2_params(), sort_keys=True),
        "status": "ok", "error": "", "train_rows": len(frame), "test_rows": 0,
        "qlike": np.nan, "log_mse": np.nan, "calibration_failures": np.nan,
    }
    append_root_trials([freeze_trial], run_id, set())
    p0_freeze_trial = {
        "trial_id": "FROZEN-P0-AUTHORIZED-FIT", "stage": "frozen_authorized_fit",
        "fold_id": "development_plus_walk_forward", "model": "P0_Ridge",
        "params": json.dumps({"alpha": p0_alpha}, sort_keys=True), "status": "ok", "error": "",
        "train_rows": len(frame), "test_rows": 0, "qlike": np.nan, "log_mse": np.nan,
        "calibration_failures": np.nan,
    }
    append_root_trials(p0_selection_trials + [p0_freeze_trial], run_id, set())

    spec = {
        "spec_version": "research_v2_p2_frozen_2",
        "status": "frozen_pre_pseudo",
        "model_family": "P2_HistGradientBoostingRegressor",
        "model_artifact": MODEL_PATH.relative_to(ROOT).as_posix(),
        "model_artifact_sha256": model_hash,
        "artifact_metadata": MODEL_METADATA_PATH.relative_to(ROOT).as_posix(),
        "p0_model_artifact": P0_MODEL_PATH.relative_to(ROOT).as_posix(),
        "p0_model_artifact_sha256": p0_hash,
        "p0_artifact_metadata": P0_METADATA_PATH.relative_to(ROOT).as_posix(),
        "p0_feature_order": P0_FEATURES,
        "p0_params": {"alpha": p0_alpha},
        "feature_order": P2_FEATURES,
        "target": "log(max(rv_fwd_120, 1e-12))",
        "variance_forecast": "exp(model_log_forecast + training_only_smearing_log_adjustment)",
        "params": frozen_p2_params(),
        "max_iter": 200,
        "early_stopping": False,
        "seed": SEED,
        "training_splits": [DEVELOPMENT, WALK_FORWARD],
        "training_rows": int(len(frame)),
        "training_max_label_end_at": str(frame["label_end_at"].max()),
        "dataset": PANEL.relative_to(ROOT).as_posix(),
        "dataset_sha256": APPROVED_DATASET_SHA256,
        "common_manifest_sha256": APPROVED_MANIFEST_SHA256,
        "splits_sha256": APPROVED_SPLITS_SHA256,
        "runner_sha256": runner_hash,
        "pseudo_holdout": {
            "label": "pseudo-holdout, previously exposed",
            "split": "pseudo_holdout_previously_exposed",
            "status": "NOT_OPENED",
            "one_shot_command": "py -3.13 research/research_v2_p2.py --score-pseudo-once",
            "marker": PSEUDO_MARKER_PATH.relative_to(ROOT).as_posix(),
            "predictions_output": PSEUDO_PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
            "result_output": PSEUDO_RESULT_PATH.relative_to(ROOT).as_posix(),
            "atomic_claim": "O_CREAT|O_EXCL marker before any pseudo-row scan",
            "rerun_policy": "marker is permanent; concurrent, failed, or completed openings consume the one shot",
            "permitted_metrics": ["QLIKE", "log_MSE", "raw_calibration_failures"],
            "allowed_verdicts": ["research_stable", "research_failed"],
            "stability_gate": {
                "chronological_halves": "stable decision_time order; first floor(n/2) rows versus remaining rows",
                "p2_lower_qlike_in_each_half": True,
                "aggregate_qlike_improvement_min_pct": 2.0,
                "hac_dm": {"alternative": "P2 QLIKE lower than P0", "lag": 120, "one_sided_p_max": 0.05},
                "p2_log_mse_no_higher_than_p0": True,
                "p2_raw_calibration_failures_no_more_than_p0": True,
                "missing_or_data_failure_verdict": "research_failed"
            },
            "strategy_pnl_permitted": False,
        },
    }
    _atomic_json(FROZEN_SPEC_PATH, spec)
    hash_paths = [
        Path(__file__), MODEL_PATH, MODEL_METADATA_PATH, P0_MODEL_PATH, P0_METADATA_PATH,
        ROOT / "MODEL_CARD_P2.md", OUT / "gate.json", SPLITS_PATH,
        ROOT / "SPLITS.sha256", COMMON_MANIFEST, PANEL, FROZEN_SPEC_PATH,
        ROOT / "tests" / "test_research_v2_p2.py",
    ]
    _write_model_hash_inventory(hash_paths)
    return {
        "run_id": run_id,
        "model_sha256": model_hash,
        "p0_model_sha256": p0_hash,
        "p0_alpha": p0_alpha,
        "metadata_sha256": file_sha256(MODEL_METADATA_PATH),
        "spec_sha256": file_sha256(FROZEN_SPEC_PATH),
        "training_rows": len(frame),
        "training_max_label_end_at": str(frame["label_end_at"].max()),
        "pseudo_holdout_accessed": False,
    }


def verify_frozen_artifact() -> tuple[dict[str, Any], dict[str, Any]]:
    if not MODEL_HASHES_PATH.is_file():
        raise FileNotFoundError(MODEL_HASHES_PATH)
    for line in MODEL_HASHES_PATH.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(" *", maxsplit=1)
        path = ROOT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"frozen file hash mismatch for {relative}: {actual} != {expected}")
    spec = json.loads(FROZEN_SPEC_PATH.read_text(encoding="utf-8"))
    artifact = joblib.load(MODEL_PATH)
    p0_artifact = joblib.load(P0_MODEL_PATH)
    if tuple(spec["feature_order"]) != tuple(artifact["features"]):
        raise ValueError("frozen feature order mismatch")
    if file_sha256(MODEL_PATH) != spec["model_artifact_sha256"]:
        raise ValueError("frozen model hash mismatch")
    if tuple(spec["p0_feature_order"]) != tuple(p0_artifact["features"]):
        raise ValueError("frozen P0 feature order mismatch")
    if file_sha256(P0_MODEL_PATH) != spec["p0_model_artifact_sha256"]:
        raise ValueError("frozen P0 model hash mismatch")
    verify_approved_inputs()
    return spec, {"p2": artifact, "p0": p0_artifact}


def score_pseudo_holdout_once() -> dict[str, Any]:
    """One-shot predictive scoring. Never called by artifact preparation."""
    spec, artifacts = verify_frozen_artifact()
    opening = claim_pseudo_holdout_once()
    opening_id = opening["opening_id"]
    try:
        conflicts = [path for path in (PSEUDO_PREDICTIONS_PATH, PSEUDO_RESULT_PATH) if path.exists()]
        if conflicts:
            raise FileExistsError(f"pre-existing pseudo output after atomic claim: {conflicts}")
        columns = ["decision_time", "label_end_at", "rv_fwd_120", "target", *P2_FEATURES]
        quoted = ",".join(f'"{column}"' for column in columns)
        connection = duckdb.connect()
        try:
            frame = connection.execute(
                f"SELECT {quoted} FROM read_parquet(?) WHERE split = ? ORDER BY decision_time",
                [str(PANEL), "pseudo_holdout_previously_exposed"],
            ).fetchdf()
        finally:
            connection.close()
        if frame.empty:
            raise ValueError("pseudo-holdout, previously exposed is empty")
        required = ["rv_fwd_120", "target", *P2_FEATURES]
        if not np.isfinite(frame[required].to_numpy(float)).all():
            raise ValueError("pseudo-holdout contains nonfinite required values")
        p2 = artifacts["p2"]
        p0 = artifacts["p0"]
        p2_forecast, p2_failures = positive_forecast(
            p2["model"].predict(frame[P2_FEATURES]), float(p2["smearing_log_adjustment"])
        )
        p0_forecast, p0_failures = positive_forecast(
            p0["model"].predict(frame[P0_FEATURES]), float(p0["smearing_log_adjustment"])
        )
        prediction = frame[["decision_time", "label_end_at"]].copy()
        prediction["label"] = "pseudo-holdout, previously exposed"
        prediction["actual_variance"] = frame["rv_fwd_120"].to_numpy(float)
        prediction["p0_forecast"] = p0_forecast
        prediction["p2_forecast"] = p2_forecast
        prediction.attrs["p0_raw_calibration_failures"] = p0_failures
        prediction.attrs["p2_raw_calibration_failures"] = p2_failures
        stability = pseudo_stability_verdict(prediction)
        write_parquet(prediction, PSEUDO_PREDICTIONS_PATH)
        result = {"opening_id": opening_id, "status": "COMPLETED_AND_CONSUMED",
                  "label": "pseudo-holdout, previously exposed", "rows": len(frame),
                  **stability, "strategy_pnl_computed": False}
        _atomic_json(PSEUDO_RESULT_PATH, result)
        _atomic_json(PSEUDO_MARKER_PATH, {
            **opening, "status": "COMPLETED_AND_CONSUMED", "verdict": result["verdict"],
            "result_sha256": file_sha256(PSEUDO_RESULT_PATH),
            "predictions_sha256": file_sha256(PSEUDO_PREDICTIONS_PATH),
        })
        return result
    except Exception as exc:
        failure = {"opening_id": opening_id, "status": "FAILED_AND_CONSUMED",
                   "verdict": "research_failed", "error_type": type(exc).__name__,
                   "error": str(exc), "strategy_pnl_computed": False}
        _atomic_json(PSEUDO_RESULT_PATH, failure)
        _atomic_json(PSEUDO_MARKER_PATH, {**opening, **failure})
        return failure


def main() -> None:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--freeze-predictive-artifact", action="store_true")
    action.add_argument("--score-pseudo-once", action="store_true")
    args = parser.parse_args()
    if args.freeze_predictive_artifact:
        print(json.dumps(freeze_authorized_predictive_artifact(), indent=2, default=str))
        return
    if args.score_pseudo_once:
        print(json.dumps(score_pseudo_holdout_once(), indent=2, default=str))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = verify_approved_inputs()
    frame = load_permitted_panel()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    exported_local_ids: set[str] = set()
    folds, summary, neighbors, trials = run_cpcv(frame, run_id, exported_local_ids)
    folds.to_csv(OUT / "cpcv_fold_metrics.csv", index=False)
    summary.to_csv(OUT / "cpcv_candidate_summary.csv", index=False)
    neighbors.to_csv(OUT / "selected_candidate_neighbors.csv", index=False)
    pd.DataFrame(trials).to_csv(OUT / "trials.csv", index=False)
    append_root_trials(trials, run_id, exported_local_ids)
    selected = summary.loc[summary["selected_by_rank"]].iloc[0]
    selected_params = {
        "max_depth": int(selected["max_depth"]),
        "learning_rate": float(selected["learning_rate"]),
        "l2_regularization": float(selected["l2_regularization"]),
    }
    development_pass = bool(selected["development_gate_pass"])
    result: dict[str, Any] = {
        "run_id": run_id, "seed": SEED, "panel_sha256": APPROVED_DATASET_SHA256,
        "splits_sha256": APPROVED_SPLITS_SHA256,
        "manifest_sha256": APPROVED_MANIFEST_SHA256,
        "provenance_verified": True,
        "permitted_rows": len(frame), "development_rows": int(frame["split"].eq("development").sum()),
        "walk_forward_rows": int(frame["split"].eq("walk_forward_validation").sum()),
        "pseudo_holdout_accessed": False, "strategy_pnl_computed": False,
        "selected_candidate": _candidate_id(selected_params), "selected_params": selected_params,
        "cpcv_folds_beating_p0": int(selected["folds_beating_p0"]),
        "cpcv_plateau_pass": bool(selected["selected_plateau_pass"]),
        "development_gate_pass": development_pass,
    }
    if development_pass:
        predictions, subperiod, regime, validation = run_walk_forward(
            frame, selected_params, trials, run_id, exported_local_ids
        )
        write_parquet(predictions, OUT / "walk_forward_predictions.parquet")
        subperiod.to_csv(OUT / "walk_forward_subperiods.csv", index=False)
        regime.to_csv(OUT / "walk_forward_regimes.csv", index=False)
        validation["monthly"].to_csv(OUT / "walk_forward_monthly.csv", index=False)
        pd.DataFrame(trials).to_csv(OUT / "trials.csv", index=False)
        append_root_trials(trials, run_id, exported_local_ids)
        result.update(validation["gate"])
        result["predictive_gate"] = (
            "PASS_VS_P0_PENDING_P1_COMPARISON"
            if validation["gate"]["p2_vs_p0_predictive_gate_pass"] else "FAIL"
        )
    else:
        result["predictive_gate"] = "FAIL_DEVELOPMENT_CPCV"
        result["walk_forward_opened"] = False
    (OUT / "gate.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    hashes = {}
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes[path.name] = file_sha256(path)
    (OUT / "SHA256SUMS").write_text(
        "".join(f"{digest} *{name}\n" for name, digest in sorted(hashes.items())), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
