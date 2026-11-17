"""Zero-brokerage v3: optimize only the well-supported zero-week segment.

This follows the supplied architecture's explicit recommendation to focus
model development on week 0 while retaining conservative fixed rules for the
smaller/less reliable expiry buckets. All optimization uses calendar 2023;
January 2024 onward remains evaluation-only. Brokerage is exactly zero while
statutory charges and two straddle points of slippage remain included.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_expiry_week_shared_stack as base
import optimize_expiry_week_shared_stack as optimizer


OUTPUT = HERE / "output" / "zero_week_focus_stack_zero_brokerage"
MODEL_ROOT = OUTPUT / "models"
BASE_MODEL_ROOT = HERE / "output" / "expiry_week_shared_stack" / "models"
EXTENSION_BUFFER_POINTS = 5.0


def load_frozen_models() -> tuple[dict[str, object], dict[str, str]]:
    folders = {
        "decay": "straddle_decay_model",
        "holding": "holding_period_model",
        "win": "win_probability_model",
        "tail": "tail_risk_model",
    }
    bundles = {
        key: joblib.load(BASE_MODEL_ROOT / folder / "model_bundle.joblib")
        for key, folder in folders.items()
    }
    models = {
        "decay": bundles["decay"]["model"],
        "holding": bundles["holding"]["model"],
        "win": bundles["win"]["model"],
        "win_calibrator": bundles["win"]["calibrator"],
        "tail": bundles["tail"]["model"],
        "tail_calibrator": bundles["tail"]["calibrator"],
    }
    selected = {key: str(bundle["family"]) for key, bundle in bundles.items()}
    return models, selected


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table, features = base.build_learning_table()
    models, selected = load_frozen_models()

    calibration = base.add_predictions(
        table.loc[(table.index >= base.THRESHOLD_START) & (table.index < base.TEST_START)],
        features,
        models,
    )
    calibration = optimizer.materialize_with_buffer(calibration, EXTENSION_BUFFER_POINTS)
    thresholds = base.calibrate_thresholds(calibration)
    week_zero_threshold, week_zero_selection = optimizer.optimize_group(
        calibration.loc[calibration["calibration_group"].eq("0")], "0"
    )
    thresholds["0"] = week_zero_threshold

    test = base.add_predictions(table.loc[table.index >= base.TEST_START], features, models)
    test = optimizer.materialize_with_buffer(test, EXTENSION_BUFFER_POINTS)
    test = base.apply_gates(test, thresholds)
    gated = test.loc[test["all_gates_pass"]]
    trades, overlap_rejections = base.enforce_one_open(gated)

    observed_dates = pd.DatetimeIndex(sorted(test.index.normalize().unique()))
    daily = (
        trades.groupby(trades.index.normalize())["net_rupees"].sum()
        .reindex(observed_dates, fill_value=0.0).rename_axis("date").reset_index()
    )
    summary = base.aggregate_metrics(trades, daily)
    summary.update({
        "brokerage_rupees": 0.0,
        "overlap_rejections": overlap_rejections,
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "extension_buffer_points": EXTENSION_BUFFER_POINTS,
    })
    trades = trades.copy()
    trades["year"] = trades.index.year
    expiry = base.grouped_metrics(trades, "calibration_group")
    yearly = base.grouped_metrics(trades, "year")
    funnel = pd.DataFrame([
        {"stage": "test candidates", "rows": len(test)},
        {"stage": "decay pass", "rows": int(test["decay_pass"].sum())},
        {"stage": "win pass", "rows": int(test["win_pass"].sum())},
        {"stage": "tail pass", "rows": int(test["tail_pass"].sum())},
        {"stage": "all gates", "rows": len(gated)},
        {"stage": "executed after one-open rule", "rows": len(trades)},
    ])
    selection = pd.read_csv(
        HERE / "output" / "expiry_week_shared_stack" / "model_family_selection_2021.csv"
    )

    selection.to_csv(OUTPUT / "model_family_selection_2021.csv", index=False)
    pd.DataFrame([week_zero_selection]).to_csv(OUTPUT / "week_zero_optimization_2023.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT / "summary.csv", index=False)
    trades.reset_index().to_csv(OUTPUT / "trades.csv", index=False)
    daily.to_csv(OUTPUT / "daily_pnl.csv", index=False)
    expiry.to_csv(OUTPUT / "expiry_week_metrics.csv", index=False)
    yearly.to_csv(OUTPUT / "yearly_metrics.csv", index=False)
    funnel.to_csv(OUTPUT / "gate_funnel.csv", index=False)
    test.reset_index().to_csv(OUTPUT / "all_test_candidates.csv.gz", index=False, compression="gzip")
    (OUTPUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (OUTPUT / "run_metadata.json").write_text(json.dumps({
        "architecture": "shared models; 2023-optimized week 0; fixed conservative weeks 1-4",
        "brokerage_rupees": 0.0,
        "statutory_charges_included": True,
        "slippage_points_per_straddle": 2.0,
        "selected_models": selected,
        "features": features,
        "horizons": base.HORIZONS,
        "extension_buffer_points": EXTENSION_BUFFER_POINTS,
        "large_loss_points": base.LARGE_LOSS_POINTS,
        "week_zero_optimization_period": "2023",
        "test_start": "2024-01-01",
        "thresholds": thresholds,
    }, indent=2), encoding="utf-8")

    old_root, old_model_root = base.ROOT, base.MODEL_ROOT
    base.ROOT, base.MODEL_ROOT = OUTPUT, MODEL_ROOT
    try:
        base.write_report(summary, expiry, yearly, funnel, selected)
        base.package_models(models, selected, features, thresholds, selection, summary)
    finally:
        base.ROOT, base.MODEL_ROOT = old_root, old_model_root

    report = (OUTPUT / "report.md").read_text(encoding="utf-8")
    report = report.replace(
        "## Frozen architecture",
        "## Zero-brokerage assumption\n\nBrokerage is fixed at **Rs 0.00**. Statutory charges and two straddle points of slippage remain included.\n\n## Frozen architecture",
    ).replace(
        "Only the entry thresholds vary by expiry bucket; weeks 3 and 4 share a calibration group.",
        "The five-point extension buffer is fixed from the supplied method. Only the zero-week gates were optimized on 2023; weeks 1–4 retain the original conservative quantile rules, and weeks 3–4 share a calibration group.",
    )
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
