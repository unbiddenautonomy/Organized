from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_fixed_atm_horizons as base


PROBABILITY_THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)


def classifiers() -> dict[str, object]:
    return {
        "logistic_c01": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=0.1, max_iter=2500, random_state=42)),
            ]
        ),
        "logistic_c1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2500, random_state=43)),
            ]
        ),
        "hist_gb_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(
                    learning_rate=0.04, max_iter=300, max_leaf_nodes=15,
                    min_samples_leaf=80, l2_regularization=20.0, random_state=42,
                )),
            ]
        ),
        "extra_trees_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesClassifier(
                    n_estimators=320, min_samples_leaf=35, max_features=0.7,
                    class_weight="balanced", n_jobs=-1, random_state=42,
                )),
            ]
        ),
        "lightgbm_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMClassifier(
                    objective="binary", n_estimators=350, learning_rate=0.03,
                    num_leaves=15, min_child_samples=100, reg_alpha=2.0,
                    reg_lambda=20.0, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=-1,
                )),
            ]
        ),
    }


def wilson_lower_bound(wins: int, total: int, z: float = 1.96) -> float:
    if total <= 0:
        return 0.0
    p = wins / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre - margin) / denominator


def horizon_data(frame: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, str]:
    target = f"target_{horizon}"
    exit_dt = f"exit_datetime_{horizon}"
    out = frame.copy()
    out[target] = out["entry_straddle"] - out[f"exit_straddle_{horizon}"] - base.SLIPPAGE_POINTS
    entry_exact = (
        out["entry_quote_datetime"].notna()
        & out["entry_strike"].eq(out["strike"])
        & pd.to_datetime(out["entry_quote_datetime"]).eq(out["datetime"])
        & pd.to_datetime(out["signal_quote_available_at"]).lt(pd.to_datetime(out["entry_quote_available_at"]))
    )
    scheduled_exit = out["datetime"] + pd.Timedelta(horizon, unit="m")
    schedule_ok = (
        scheduled_exit.dt.normalize().eq(out["datetime"].dt.normalize())
        & (scheduled_exit.dt.hour * 60 + scheduled_exit.dt.minute).le(15 * 60 + 29)
    )
    exact = (
        schedule_ok
        & entry_exact
        & out[exit_dt].notna()
        & out[f"exit_strike_{horizon}"].eq(out["strike"])
        & ((pd.to_datetime(out[exit_dt]) - out["datetime"]).dt.total_seconds() == horizon * 60)
        & (
            (pd.to_datetime(out[f"exit_quote_available_at_{horizon}"]) - pd.to_datetime(out["entry_quote_available_at"]))
            .dt.total_seconds().eq(horizon * 60)
        )
    )
    if horizon == base.LATE_SESSION_HORIZON:
        minute = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute
        exact &= minute.ge(base.LATE_SESSION_START_MINUTE)
    return out.loc[exact].copy(), target


def select_classifier(train: pd.DataFrame, validation: pd.DataFrame, features: list[str], target: str):
    y_train = train[target].gt(0).astype(int)
    y_valid = validation[target].gt(0).astype(int)
    records = []
    fitted = classifiers()
    for name, model in fitted.items():
        model.fit(train[features], y_train)
        probability = model.predict_proba(validation[features])[:, 1]
        records.append(
            {
                "model": name,
                "roc_auc": float(roc_auc_score(y_valid, probability)),
                "log_loss": float(log_loss(y_valid, probability)),
            }
        )
    records.sort(key=lambda row: (-row["roc_auc"], row["log_loss"]))
    return records[0]["model"], records


def calibrate_policy(
    model_name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    target: str,
    horizon: int,
) -> tuple[float, dict, list[dict]]:
    model = classifiers()[model_name]
    model.fit(train[features], train[target].gt(0).astype(int))
    probability = model.predict_proba(validation[features])[:, 1]
    minimum_trades = 10 if horizon == base.LATE_SESSION_HORIZON else 20
    policies = []
    for threshold in PROBABILITY_THRESHOLDS:
        metrics, trades = base.non_overlapping_strategy(
            validation, validation[target].to_numpy(), probability, horizon, threshold
        )
        wins = int((trades["actual_net_points_after_2pt_cost"] > 0).sum())
        lower_bound = wilson_lower_bound(wins, len(trades))
        eligible = (
            metrics["trades"] >= minimum_trades
            and metrics["signal_days"] >= min(10, minimum_trades)
            and metrics["net_points_after_2pt_slippage"] > 0
            and metrics["profit_factor"] > 1.0
        )
        policies.append(
            {
                "probability_threshold": threshold,
                "wilson_win_rate_lower_95": lower_bound,
                "eligible": eligible,
                **metrics,
            }
        )
    eligible_policies = [row for row in policies if row["eligible"]]
    if not eligible_policies:
        selected = max(
            policies,
            key=lambda row: (row["wilson_win_rate_lower_95"], row["net_points_after_2pt_slippage"]),
        )
        selected = {**selected, "deployable": False}
    else:
        selected = max(
            eligible_policies,
            key=lambda row: (row["wilson_win_rate_lower_95"], row["net_points_after_2pt_slippage"]),
        )
        selected = {**selected, "deployable": True}
    return float(selected["probability_threshold"]), selected, policies


def train_one(frame: pd.DataFrame, features: list[str], horizon: int, output: Path) -> dict:
    usable, target = horizon_data(frame, horizon)
    exit_dt = f"exit_datetime_{horizon}"
    train_core = usable[
        (usable["datetime"] < base.MODEL_SELECTION_START)
        & (pd.to_datetime(usable[exit_dt]) < base.MODEL_SELECTION_START)
    ]
    model_validation = usable[
        (usable["datetime"] >= base.MODEL_SELECTION_START)
        & (usable["datetime"] < base.VALIDATION_START)
        & (pd.to_datetime(usable[exit_dt]) < base.VALIDATION_START)
    ]
    prepolicy = usable[
        (usable["datetime"] < base.VALIDATION_START)
        & (pd.to_datetime(usable[exit_dt]) < base.VALIDATION_START)
    ]
    policy_validation = usable[
        (usable["datetime"] >= base.VALIDATION_START)
        & (usable["datetime"] < base.TEST_START)
        & (pd.to_datetime(usable[exit_dt]) < base.TEST_START)
    ]
    pretest = usable[
        (usable["datetime"] < base.TEST_START)
        & (pd.to_datetime(usable[exit_dt]) < base.TEST_START)
    ]
    test = usable[usable["datetime"] >= base.TEST_START]

    winner, family_results = select_classifier(train_core, model_validation, features, target)
    threshold, selected_policy, policies = calibrate_policy(
        winner, prepolicy, policy_validation, features, target, horizon
    )
    final_model = classifiers()[winner]
    final_model.fit(pretest[features], pretest[target].gt(0).astype(int))
    probability = final_model.predict_proba(test[features])[:, 1]
    test_metrics, trades = base.non_overlapping_strategy(
        test, test[target].to_numpy(), probability, horizon, threshold
    )
    wins = int((trades["actual_net_points_after_2pt_cost"] > 0).sum())
    test_metrics["wilson_win_rate_lower_95"] = wilson_lower_bound(wins, len(trades))
    baseline, _ = base.non_overlapping_strategy(
        test, test[target].to_numpy(), np.ones(len(test)), horizon, 0.0
    )
    test_metrics["net_points_vs_no_model_baseline"] = (
        test_metrics["net_points_after_2pt_slippage"] - baseline["net_points_after_2pt_slippage"]
    )
    artifact = {
        "model": final_model,
        "model_name": winner,
        "horizon_minutes": horizon,
        "features": features,
        "probability_threshold": threshold,
        "target": "probability that fixed-strike straddle decay exceeds two points of round-trip cost",
        "training_end_exclusive": str(base.TEST_START.date()),
    }
    artifact_name = f"high_win_rate_{horizon}m.joblib"
    joblib.dump(artifact, output / artifact_name)
    trades.to_csv(output / f"test_trades_{horizon}m.csv", index=False)
    return {
        "horizon_minutes": horizon,
        "artifact": artifact_name,
        "selected_model": winner,
        "probability_threshold": threshold,
        "rows": {
            "train_core": len(train_core),
            "model_validation": len(model_validation),
            "policy_validation": len(policy_validation),
            "pretest": len(pretest),
            "test": len(test),
        },
        "family_selection": family_results,
        "selected_policy": selected_policy,
        "all_probability_policies": policies,
        "test": test_metrics,
        "test_no_model_baseline": baseline,
    }


def write_report(output: Path, results: list[dict], feature_count: int) -> None:
    deployable = [row for row in results if row["selected_policy"]["deployable"]]
    selected = max(
        deployable or results,
        key=lambda row: (
            row["selected_policy"]["wilson_win_rate_lower_95"],
            row["selected_policy"]["net_points_after_2pt_slippage"],
        ),
    )
    lines = [
        "# High Win Rate Fixed Strike Model Results",
        "",
        "## Decision",
        "",
        f"The pre-2024 process selected the **{selected['horizon_minutes']}-minute {selected['selected_model']}** classifier at probability threshold **{selected['probability_threshold']:.2f}**.",
        "",
        f"On the January 2024 onward retest it produced **{selected['test']['win_rate_after_slippage']:.2%}** wins across **{selected['test']['trades']} trades**, with **{selected['test']['net_points_after_2pt_slippage']:.1f} net points** and profit factor **{selected['test']['profit_factor']:.2f}**.",
        "",
        "## Method",
        "",
        f"- Entry-time features: {feature_count}",
        "- Model family selected on January-June 2023 ROC AUC.",
        "- Probability threshold and horizon selected on July-December 2023.",
        "- Policy selection maximizes the 95% Wilson lower confidence bound for win rate.",
        "- Policies must retain minimum activity, positive net points, and profit factor above one.",
        "- January 2024 onward is used only for retesting the frozen policy.",
        "- Strike and expiry remain fixed, execution is delayed one minute, and the target includes two points of round-trip cost.",
        "",
        "## Comparison",
        "",
        "| Horizon | Classifier | Probability | Validation wins | Validation trades | Test wins | Test trades | Test net points | Profit factor | Wilson lower 95% |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        validation = row["selected_policy"]
        test = row["test"]
        lines.append(
            f"| {row['horizon_minutes']}m | {row['selected_model']} | {row['probability_threshold']:.2f} | "
            f"{validation['win_rate_after_slippage']:.2%} | {validation['trades']} | "
            f"{test['win_rate_after_slippage']:.2%} | {test['trades']} | "
            f"{test['net_points_after_2pt_slippage']:.1f} | {test['profit_factor']:.2f} | "
            f"{test['wilson_win_rate_lower_95']:.2%} |"
        )
    lines.extend(
        [
            "",
            "The January 2024 onward period has already been examined by earlier experiments, so this is a retest rather than a pristine new holdout. A later unseen period remains necessary before deployment.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("artifacts/fixed_atm_horizons_leakage_free/prepared_fixed_atm_labels.parquet"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/high_win_rate_models"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    frame = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(args.prepared)]).fetchdf()
    frame, features = base.add_entry_features(frame)
    forbidden = [name for name in features if name.startswith(("exit_", "target_", "label_"))]
    forbidden.extend(
        name for name in features
        if name in {"entry_straddle", "entry_call_price", "entry_put_price", "entry_quote_available_at"}
    )
    if forbidden:
        raise AssertionError(f"Future or execution-price fields entered features: {forbidden}")
    results = []
    for horizon in base.HORIZONS:
        result = train_one(frame, features, horizon, args.output)
        results.append(result)
        print(json.dumps({"horizon": horizon, "model": result["selected_model"], "test": result["test"]}, indent=2))
    manifest = {
        "prepared_source": str(args.prepared),
        "features": features,
        "results": results,
        "leakage_audit": {
            "future_or_execution_fields_in_features": forbidden,
            "model_family_selection": "January-June 2023",
            "policy_and_horizon_selection": "July-December 2023",
            "retest_start": str(base.TEST_START.date()),
            "fixed_signal_atm_strike": True,
            "one_minute_execution_delay": True,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.output, results, len(features))


if __name__ == "__main__":
    main()
