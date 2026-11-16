from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

import train_fixed_atm_horizons as base
import train_high_win_rate_models as high


HORIZON = 90
PROBABILITY_GRID = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
EDGE_GRID = (0.0, 2.0, 5.0, 10.0, 15.0)


def hybrid_strategy(meta, y, probability, edge, probability_threshold, edge_threshold):
    score = np.where(
        (probability >= probability_threshold) & (edge >= edge_threshold),
        probability,
        -np.inf,
    )
    return base.non_overlapping_strategy(meta, y, score, HORIZON, probability_threshold)


def main() -> None:
    source = Path("artifacts/fixed_atm_horizons_leakage_free/prepared_fixed_atm_labels.parquet")
    output = Path("artifacts/high_win_rate_hybrid")
    output.mkdir(parents=True, exist_ok=True)
    frame = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(source)]).fetchdf()
    frame, features = base.add_entry_features(frame)
    usable, target = high.horizon_data(frame, HORIZON)
    exit_dt = f"exit_datetime_{HORIZON}"
    prepolicy = usable[
        (usable["datetime"] < base.VALIDATION_START)
        & (pd.to_datetime(usable[exit_dt]) < base.VALIDATION_START)
    ]
    validation = usable[
        (usable["datetime"] >= base.VALIDATION_START)
        & (usable["datetime"] < base.TEST_START)
        & (pd.to_datetime(usable[exit_dt]) < base.TEST_START)
    ]
    pretest = usable[
        (usable["datetime"] < base.TEST_START)
        & (pd.to_datetime(usable[exit_dt]) < base.TEST_START)
    ]
    test = usable[usable["datetime"] >= base.TEST_START]

    classifier = high.classifiers()["extra_trees_classifier"]
    regressor = base.candidate_models()["lightgbm_huber"]
    classifier.fit(prepolicy[features], prepolicy[target].gt(0).astype(int))
    regressor.fit(prepolicy[features], prepolicy[target])
    validation_probability = classifier.predict_proba(validation[features])[:, 1]
    validation_edge = regressor.predict(validation[features])

    policies = []
    for probability_threshold in PROBABILITY_GRID:
        for edge_threshold in EDGE_GRID:
            metrics, trades = hybrid_strategy(
                validation,
                validation[target].to_numpy(),
                validation_probability,
                validation_edge,
                probability_threshold,
                edge_threshold,
            )
            wins = int((trades["actual_net_points_after_2pt_cost"] > 0).sum())
            wilson = high.wilson_lower_bound(wins, len(trades))
            eligible = (
                metrics["trades"] >= 10
                and metrics["signal_days"] >= 10
                and metrics["net_points_after_2pt_slippage"] > 0
                and metrics["profit_factor"] > 1.0
            )
            policies.append(
                {
                    "probability_threshold": probability_threshold,
                    "edge_threshold": edge_threshold,
                    "wilson_win_rate_lower_95": wilson,
                    "eligible": eligible,
                    **metrics,
                }
            )
    eligible = [row for row in policies if row["eligible"]]
    selected = max(
        eligible or policies,
        key=lambda row: (row["wilson_win_rate_lower_95"], row["net_points_after_2pt_slippage"]),
    )

    classifier.fit(pretest[features], pretest[target].gt(0).astype(int))
    regressor.fit(pretest[features], pretest[target])
    test_probability = classifier.predict_proba(test[features])[:, 1]
    test_edge = regressor.predict(test[features])
    test_metrics, trades = hybrid_strategy(
        test,
        test[target].to_numpy(),
        test_probability,
        test_edge,
        selected["probability_threshold"],
        selected["edge_threshold"],
    )
    wins = int((trades["actual_net_points_after_2pt_cost"] > 0).sum())
    test_metrics["wilson_win_rate_lower_95"] = high.wilson_lower_bound(wins, len(trades))
    baseline, _ = base.non_overlapping_strategy(
        test, test[target].to_numpy(), np.ones(len(test)), HORIZON, 0.0
    )
    test_metrics["net_points_vs_no_model_baseline"] = (
        test_metrics["net_points_after_2pt_slippage"] - baseline["net_points_after_2pt_slippage"]
    )
    artifact = {
        "classifier": classifier,
        "regressor": regressor,
        "features": features,
        "horizon_minutes": HORIZON,
        "probability_threshold": selected["probability_threshold"],
        "predicted_net_edge_threshold": selected["edge_threshold"],
        "training_end_exclusive": str(base.TEST_START.date()),
    }
    joblib.dump(artifact, output / "high_win_rate_hybrid_90m.joblib")
    trades.to_csv(output / "test_trades_90m.csv", index=False)
    metrics = {
        "selected_validation_policy": selected,
        "all_validation_policies": policies,
        "test": test_metrics,
        "test_no_model_baseline": baseline,
        "rows": {"prepolicy": len(prepolicy), "validation": len(validation), "pretest": len(pretest), "test": len(test)},
        "leakage_audit": {
            "features": "signal-time only",
            "entry": "next one-minute full-chain quote",
            "policy_selection": "July-December 2023 only",
            "retest": "January 2024 onward",
        },
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    report = [
        "# High Win Rate Hybrid 90 Minute Model",
        "",
        f"The frozen validation policy requires win probability of at least **{selected['probability_threshold']:.2f}** and predicted net edge of at least **{selected['edge_threshold']:.1f} points**.",
        "",
        f"Validation: {selected['trades']} trades, {selected['win_rate_after_slippage']:.2%} win rate, {selected['net_points_after_2pt_slippage']:.1f} net points, profit factor {selected['profit_factor']:.2f}.",
        "",
        f"January 2024 onward retest: {test_metrics['trades']} trades, **{test_metrics['win_rate_after_slippage']:.2%} win rate**, **{test_metrics['net_points_after_2pt_slippage']:.1f} net points**, profit factor **{test_metrics['profit_factor']:.2f}**, maximum drawdown {test_metrics['maximum_drawdown_points']:.1f} points.",
        "",
        "The retest period has been examined previously and is not a pristine new holdout.",
        "",
    ]
    (output / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"validation": selected, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
