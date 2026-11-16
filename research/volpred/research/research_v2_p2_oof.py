"""Generate development-only selected-P2 CPCV out-of-fold predictions.

This runner reads only the development split.  It does not access walk-forward,
pseudo-holdout, option, execution, or P&L data.
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from research.research_v2_p2 import (
    APPROVED_DATASET_SHA256,
    APPROVED_MANIFEST_SHA256,
    APPROVED_SPLITS_SHA256,
    OUT,
    PANEL,
    P2_FEATURES,
    ROOT,
    append_root_trials,
    file_sha256,
    fit_p2,
    frozen_p2_params,
    positive_forecast,
    purged_train_test,
    qlike,
    verify_approved_inputs,
    verify_frozen_artifact,
    write_parquet,
)
from research.research_v2_splits import (
    DEVELOPMENT,
    assert_calendar_partition,
    cpcv_8_choose_2,
    read_restricted_parquet,
)


PREDICTIONS_PATH = OUT / "development_cpcv_oof_predictions.parquet"
MANIFEST_PATH = OUT / "development_cpcv_oof_manifest.json"
HASHES_PATH = OUT / "development_cpcv_oof_SHA256SUMS"


def load_development_panel() -> pd.DataFrame:
    verify_approved_inputs()
    columns = [
        "decision_time", "decision_date", "label_end_at", "split", "rv_fwd_120", "target",
        *P2_FEATURES,
    ]
    frame = read_restricted_parquet(
        PANEL, allowed_splits=[DEVELOPMENT], columns=columns,
        context="selected-P2 development CPCV OOF",
    )
    if frame.empty or set(frame["split"].astype(str).unique()) != {DEVELOPMENT}:
        raise ValueError("development-only panel is empty or contaminated")
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True).dt.tz_convert("Asia/Kolkata")
    frame["label_end_at"] = pd.to_datetime(frame["label_end_at"], utc=True).dt.tz_convert("Asia/Kolkata")
    frame["decision_date"] = frame["decision_time"].dt.normalize()
    assert_calendar_partition(frame)
    required = ["rv_fwd_120", "target", *P2_FEATURES]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise ValueError("development panel contains nonfinite model inputs")
    frame = frame.sort_values("decision_time", kind="stable").reset_index(drop=True)
    if frame["decision_time"].duplicated().any():
        raise ValueError("development decision_time is not a unique row key")
    frame["development_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def validate_and_aggregate_components(
    development: pd.DataFrame, components: pd.DataFrame
) -> pd.DataFrame:
    required = {"development_row_id", "fold_id", "component_forecast", "raw_calibration_failure"}
    missing = sorted(required.difference(components.columns))
    if missing:
        raise ValueError(f"missing component columns: {missing}")
    if components.duplicated(["development_row_id", "fold_id"]).any():
        raise ValueError("duplicate row/fold component forecast")
    coverage = components.groupby("development_row_id").agg(
        component_forecast_count=("component_forecast", "size"),
        p2_forecast=("component_forecast", "mean"),
        raw_calibration_failures=("raw_calibration_failure", "sum"),
    )
    expected_ids = pd.Index(development["development_row_id"], name="development_row_id")
    coverage = coverage.reindex(expected_ids)
    if coverage["component_forecast_count"].isna().any():
        raise AssertionError("at least one eligible development row has no component forecasts")
    counts = coverage["component_forecast_count"].astype(int)
    if not counts.eq(7).all():
        bad = counts[counts.ne(7)].head().to_dict()
        raise AssertionError(f"every development row must have exactly 7 component forecasts: {bad}")
    if not np.isfinite(coverage["p2_forecast"]).all() or not coverage["p2_forecast"].gt(0).all():
        raise AssertionError("aggregated OOF forecasts must be finite and positive")
    result = development[[
        "development_row_id", "decision_time", "decision_date", "label_end_at", "split", "rv_fwd_120"
    ]].copy()
    result = result.merge(coverage.reset_index(), on="development_row_id", how="left", validate="one_to_one")
    return result.sort_values("decision_time", kind="stable").reset_index(drop=True)


def generate_development_oof() -> dict[str, Any]:
    for path in (PREDICTIONS_PATH, MANIFEST_PATH, HASHES_PATH):
        if path.exists():
            raise FileExistsError(f"development OOF artifact already exists: {path}")
    spec, _ = verify_frozen_artifact()
    if spec["params"] != frozen_p2_params():
        raise ValueError("frozen selected P2 parameters do not match the OOF runner")
    development = load_development_panel()
    sessions = development["decision_date"].drop_duplicates().sort_values().to_numpy()
    groups, folds = cpcv_8_choose_2(sessions)
    params = frozen_p2_params()
    run_id = "development-oof-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid.uuid4().hex[:8]
    components: list[pd.DataFrame] = []
    exported: set[str] = set()
    fold_manifest: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        fold_id = f"OOF-CPCV-{fold_number:02d}-g{fold.test_groups[0]}g{fold.test_groups[1]}"
        train, test, purge = purged_train_test(development, sessions, fold.test_dates)
        trial = {
            "trial_id": fold_id, "stage": "development_cpcv_oof_refit", "fold_id": fold_id,
            "model": "P2_HistGradientBoosting", "params": json.dumps(params, sort_keys=True),
            "train_rows": len(train), "test_rows": len(test), **purge,
        }
        try:
            model, smear = fit_p2(train, params)
            forecast, failures = positive_forecast(model.predict(test[P2_FEATURES]), smear)
            trial.update(status="ok", error="", smear=smear, qlike=float(np.mean(qlike(
                test["rv_fwd_120"].to_numpy(float), forecast
            ))), log_mse=float(np.mean((test["target"].to_numpy(float) - np.log(forecast)) ** 2)),
                         calibration_failures=failures)
            component = test[["development_row_id"]].copy()
            component["fold_id"] = fold_id
            component["component_forecast"] = forecast
            component["raw_calibration_failure"] = 0
            if failures:
                raise AssertionError("selected P2 produced raw calibration failures in OOF scoring")
            components.append(component)
            fold_manifest.append({
                "fold_id": fold_id, "held_out_groups": list(fold.test_groups),
                "train_rows": len(train), "test_rows": len(test), "smearing_log_adjustment": smear,
                "purged_or_embargoed_rows": purge["purged_or_embargoed_rows"],
                "raw_calibration_failures": failures,
            })
        except Exception as exc:
            trial.update(status="failed", error=repr(exc), qlike=np.nan, log_mse=np.nan,
                         calibration_failures=np.nan)
            append_root_trials([trial], run_id, exported)
            raise
        append_root_trials([trial], run_id, exported)
    if len(fold_manifest) != 28:
        raise AssertionError(f"expected 28 successful refits, got {len(fold_manifest)}")
    component_frame = pd.concat(components, ignore_index=True)
    predictions = validate_and_aggregate_components(development, component_frame)
    counts = predictions["component_forecast_count"].astype(int)
    calibration_ratio = float(np.mean(
        predictions["rv_fwd_120"].to_numpy(float) / predictions["p2_forecast"].to_numpy(float)
    ))
    OUT.mkdir(parents=True, exist_ok=True)
    write_parquet(predictions, PREDICTIONS_PATH)
    manifest = {
        "artifact": PREDICTIONS_PATH.relative_to(ROOT).as_posix(),
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": run_id,
        "scope": "development-only selected-P2 CPCV out-of-fold prediction",
        "development_rows": len(development), "component_rows": len(component_frame),
        "successful_refits": len(fold_manifest), "expected_refits": 28,
        "component_forecast_count_min": int(counts.min()),
        "component_forecast_count_max": int(counts.max()),
        "rows_with_exactly_7_components": int(counts.eq(7).sum()),
        "coverage_pass": bool(counts.eq(7).all()),
        "aggregate_raw_calibration_failures": int(predictions["raw_calibration_failures"].sum()),
        "mean_actual_forecast_ratio": calibration_ratio,
        "oof_qlike": float(np.mean(qlike(predictions["rv_fwd_120"], predictions["p2_forecast"]))),
        "params": params, "seed": 7319, "feature_order": P2_FEATURES,
        "dataset_sha256": APPROVED_DATASET_SHA256,
        "common_manifest_sha256": APPROVED_MANIFEST_SHA256,
        "splits_sha256": APPROVED_SPLITS_SHA256,
        "frozen_spec_sha256": file_sha256(ROOT / "FROZEN_PREDICTIVE_SPEC_V2.json"),
        "validation_accessed": False, "pseudo_holdout_accessed": False,
        "option_data_accessed": False, "strategy_pnl_computed": False,
        "folds": fold_manifest,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hash_paths = [PREDICTIONS_PATH, MANIFEST_PATH, Path(__file__),
                  ROOT / "tests" / "test_research_v2_p2_oof.py"]
    HASHES_PATH.write_text("".join(
        f"{file_sha256(path)} *{path.relative_to(ROOT).as_posix()}\n" for path in hash_paths
    ), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    print(json.dumps(generate_development_oof(), indent=2))
