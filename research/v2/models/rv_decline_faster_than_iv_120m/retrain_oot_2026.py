from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, mean_absolute_error

from refine import (
    IV_CHANGE_COPY,
    RV_CHANGE,
    add_engineered_features,
    candidate_specs,
    evaluate,
    model_score,
)
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
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain on 2019-2025 and evaluate once on the 2026 out-of-time holdout")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--holdout-csv",
        default="../../research/combined_atm_iv_rv_convergence_v2/artifacts_2026_holdout/combined_atm_iv_convergence_v2_features_2026.csv",
    )
    parser.add_argument("--validation-leaderboard", default="artifacts_refined/validation_leaderboard.csv")
    parser.add_argument("--artifacts-dir", default="artifacts_oot_2026")
    parser.add_argument("--selected-name", help="Use a validation-tested specification by exact name")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    settings, target_settings = config["training"], config["target"]
    artifacts = (config_path.parent / args.artifacts_dir).resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    holdout_path = (config_path.parent / args.holdout_csv).resolve()
    leaderboard_path = (config_path.parent / args.validation_leaderboard).resolve()

    leaderboard = pd.read_csv(leaderboard_path).sort_values(
        ["validation_roc_auc", "validation_balanced_accuracy"], ascending=False
    )
    if args.selected_name:
        matching = leaderboard[leaderboard["name"].eq(args.selected_name)]
        if matching.empty:
            raise ValueError(f"Specification not found in validation leaderboard: {args.selected_name}")
        selected = matching.iloc[0]
    else:
        selected = leaderboard.iloc[0]
    selected_name = str(selected["name"])
    spec = next(
        item for item in candidate_specs(int(settings["random_seed"]), int(settings["n_jobs"]))
        if item["name"] == selected_name
    )
    threshold = float(selected["validation_threshold"])
    lookback = int(spec["lookback"])

    manifest = json.loads(Path(config["data"]["feature_manifest"]).read_text(encoding="utf-8"))
    base_features = approved_features(manifest, lookback)
    required = list(dict.fromkeys([
        "datetime", "exit_datetime", "training_row_eligible", RV_CURRENT, RV_FUTURE, IV_CHANGE,
        *base_features,
    ]))
    historical = pd.read_csv(
        config["data"]["feature_csv"], usecols=required,
        parse_dates=["datetime", "exit_datetime"], low_memory=False,
    )
    holdout = pd.read_csv(
        holdout_path, usecols=required, parse_dates=["datetime", "exit_datetime"], low_memory=False,
    )
    historical, engineered = add_engineered_features(historical)
    holdout, holdout_engineered = add_engineered_features(holdout)
    if engineered != holdout_engineered:
        raise AssertionError("Historical and holdout engineered-feature schemas differ")
    features = base_features + [name for name in engineered if not ("120m" in name and lookback < 120)]

    historical = add_target(
        historical, float(target_settings["minimum_rv_drop"]),
        float(target_settings["minimum_rv_drop_beyond_iv"]),
    )
    holdout = add_target(
        holdout, float(target_settings["minimum_rv_drop"]),
        float(target_settings["minimum_rv_drop_beyond_iv"]),
    )
    historical = historical[
        historical["training_row_eligible"].fillna(False) & historical[TARGET].notna()
    ].copy()
    holdout = holdout[holdout["training_row_eligible"].fillna(False) & holdout[TARGET].notna()].copy()
    historical[TARGET] = historical[TARGET].astype(int)
    holdout[TARGET] = holdout[TARGET].astype(int)

    if historical["datetime"].max() >= holdout["datetime"].min():
        raise AssertionError("The out-of-time holdout overlaps the historical training period")
    forbidden = [name for name in features if name.startswith(("target_", "exit_fixed_", "exit_gap_"))]
    if forbidden:
        raise AssertionError(f"Future-derived features found: {forbidden}")

    y_components = [RV_CHANGE, IV_CHANGE_COPY]
    train_target = historical[TARGET] if spec["kind"] == "direct_classifier" else historical[y_components]
    final_model = clone(spec["model"]).fit(numeric_matrix(historical, features), train_target)
    holdout_score, holdout_components = model_score(
        final_model, spec["kind"], numeric_matrix(holdout, features), target_settings
    )
    holdout_metrics = evaluate(holdout[TARGET], holdout_score, threshold)
    if holdout_components is not None:
        holdout_metrics["rv_change_mae"] = float(mean_absolute_error(holdout[RV_CHANGE], holdout_components[:, 0]))
        holdout_metrics["iv_change_mae"] = float(mean_absolute_error(holdout[IV_CHANGE_COPY], holdout_components[:, 1]))
    tn, fp, fn, tp = confusion_matrix(holdout[TARGET], holdout_score >= threshold).ravel()
    holdout_metrics["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    scored = holdout.copy()
    scored["selected_model_score"] = holdout_score
    non_overlapping = non_overlapping_rows(scored, int(target_settings["horizon_minutes"]))
    non_overlapping_metrics = evaluate(
        non_overlapping[TARGET], non_overlapping["selected_model_score"].to_numpy(), threshold
    )
    monthly_metrics = {}
    for period, part in scored.groupby(scored["datetime"].dt.to_period("M")):
        if part[TARGET].nunique() == 2:
            monthly_metrics[str(period)] = evaluate(
                part[TARGET], part["selected_model_score"].to_numpy(), threshold
            )

    predictions = holdout[["datetime", TARGET, RV_CHANGE, IV_CHANGE_COPY]].copy()
    predictions["model_score"] = holdout_score
    predictions["predicted_event"] = holdout_score >= threshold
    if holdout_components is not None:
        predictions["predicted_rv_change_120m"] = holdout_components[:, 0]
        predictions["predicted_iv_change_120m"] = holdout_components[:, 1]
    predictions.to_csv(artifacts / "oot_2026_predictions.csv", index=False)

    bundle = {
        "model": final_model, "model_type": spec["kind"], "features": features,
        "engineered_features": engineered, "lookback_minutes": lookback, "threshold": threshold,
        "target_definition": target_settings, "name": selected_name,
        "training_period": {"start": str(historical["datetime"].min()), "end": str(historical["datetime"].max())},
    }
    joblib.dump(bundle, artifacts / "winner.joblib")
    audit = {
        "feature_leakage_found": False,
        "evaluation_leakage_corrected": True,
        "correction": "Historical test reused during iteration was moved into training; 2026 was evaluated once out of time.",
        "selected_specification_source": "historical validation leaderboard only",
        "historical_rows": int(len(historical)),
        "holdout_rows": int(len(holdout)),
        "feature_count": len(features),
        "historical_period": bundle["training_period"],
        "holdout_period": {"start": str(holdout["datetime"].min()), "end": str(holdout["datetime"].max())},
        "holdout": holdout_metrics,
        "holdout_non_overlapping_120m": non_overlapping_metrics,
        "holdout_by_month": monthly_metrics,
        "versions": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (artifacts / "metrics.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Leakage-safe retraining and 2026 out-of-time test", "",
        "No future-derived model feature was found. Historical lag and rolling features are backward-looking, and 120-minute labels are purged at chronological boundaries.", "",
        "The earlier 2024-2025 test period had been reused during iterative development. This run treats all 2019-2025 observations as training data and evaluates the frozen specification once on separate 2026 data.", "",
        f"- Frozen specification: {selected_name}.",
        f"- Historical training rows: {len(historical):,}.",
        f"- 2026 holdout rows: {len(holdout):,}.",
        f"- 2026 accuracy: {holdout_metrics['accuracy']:.2%}.",
        f"- 2026 balanced accuracy: {holdout_metrics['balanced_accuracy']:.2%}.",
        f"- 2026 ROC AUC: {holdout_metrics['roc_auc']:.4f}.",
        f"- 2026 precision: {holdout_metrics['precision']:.2%}; recall: {holdout_metrics['recall']:.2%}.",
        f"- Non-overlapping 120-minute rows: {non_overlapping_metrics['rows']:,}; accuracy: {non_overlapping_metrics['accuracy']:.2%}; ROC AUC: {non_overlapping_metrics['roc_auc']:.4f}.",
    ]
    (artifacts / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"model": selected_name, "holdout": holdout_metrics, "artifacts": str(artifacts)}, indent=2))


if __name__ == "__main__":
    main()
