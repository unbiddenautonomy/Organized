from __future__ import annotations

import argparse
import hashlib
import json
import os
import math
from pathlib import Path

import joblib
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import train_fixed_atm_horizons as fixed_atm


HORIZONS = (30, 60, 120, 180)
TEST_START = pd.Timestamp("2024-01-01")
VALIDATION_START = pd.Timestamp("2023-07-01")
TAIL_QUANTILE = 0.90
VETO_FRACTION = 0.20
SLIPPAGE_POINTS = 2.0
CACHE_SCHEMA_VERSION = 3
ROLLING_SCORE_WINDOW = 1000
PATH_CACHE_SCHEMA_VERSION = 1


def candidate_models() -> dict[str, object]:
    logistic = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=0.25,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    extra_trees = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=12,
                    max_features=0.7,
                    class_weight="balanced",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    random_forest = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=350,
                    min_samples_leaf=12,
                    max_features=0.7,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    hist_gb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.04,
                    max_iter=250,
                    max_leaf_nodes=15,
                    min_samples_leaf=35,
                    l2_regularization=10.0,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    lightgbm_regularized = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=400,
                    learning_rate=0.025,
                    num_leaves=15,
                    min_child_samples=80,
                    reg_alpha=3.0,
                    reg_lambda=25.0,
                    subsample=0.8,
                    colsample_bytree=0.75,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    lightgbm_conservative = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.025,
                    num_leaves=7,
                    min_child_samples=150,
                    reg_alpha=5.0,
                    reg_lambda=40.0,
                    subsample=0.85,
                    colsample_bytree=0.8,
                    class_weight="balanced",
                    random_state=43,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )
    soft_vote_linear_tree = VotingClassifier(
        estimators=[
            ("logistic", logistic),
            ("extra_trees", extra_trees),
            ("hist_gb", hist_gb),
        ],
        voting="soft",
        weights=[2.0, 1.0, 1.0],
        n_jobs=-1,
    )
    return {
        "logistic_l2_balanced": logistic,
        "extra_trees_balanced": extra_trees,
        "random_forest_balanced": random_forest,
        "hist_gb_balanced": hist_gb,
        "lightgbm_regularized": lightgbm_regularized,
        "lightgbm_conservative": lightgbm_conservative,
        "soft_vote_linear_tree": soft_vote_linear_tree,
    }


def add_tail_features(
    frame: pd.DataFrame, base_features: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Add entry-known interactions designed for volatility/tail ranking."""
    out = frame.copy()
    engineered: list[str] = []
    for lag in (5, 15, 30, 60):
        out[f"iv_impulse_{lag}"] = (
            out[f"change_{lag}_iv_call"] + out[f"change_{lag}_iv_put"]
        ) / 2.0
        out[f"premium_velocity_{lag}"] = (
            out[f"straddle_return_{lag}"] / float(lag)
        )
        out[f"return_velocity_{lag}"] = (
            out[f"abs_future_return_{lag}"] / math.sqrt(float(lag))
        )
        engineered.extend(
            [f"iv_impulse_{lag}", f"premium_velocity_{lag}", f"return_velocity_{lag}"]
        )
    out["return_acceleration_5_60"] = (
        out["return_velocity_5"] - out["return_velocity_60"]
    )
    out["premium_acceleration_5_60"] = (
        out["premium_velocity_5"] - out["premium_velocity_60"]
    )
    out["iv_acceleration_5_60"] = out["iv_impulse_5"] - out["iv_impulse_60"]
    out["trend_5_minus_30"] = out["future_return_5"] - out["future_return_30"]
    out["iv_to_realized_proxy"] = out["iv_mean"] / (
        out["return_velocity_60"].abs() + 1e-5
    )
    out["gamma_iv_interaction"] = out["gamma_per_premium"] * out["iv_mean"]
    engineered.extend(
        [
            "return_acceleration_5_60", "premium_acceleration_5_60",
            "iv_acceleration_5_60", "trend_5_minus_30", "iv_to_realized_proxy",
            "gamma_iv_interaction",
        ]
    )
    return out, [*base_features, *engineered]


def attach_path_maxima(
    frame: pd.DataFrame,
    full_chain_source: Path,
    cache_path: Path,
    source_fingerprint: dict,
    loader_hash: str,
) -> pd.DataFrame:
    """Attach maximum fixed-contract straddle observed within each horizon."""
    cache_key = {
        "schema_version": PATH_CACHE_SCHEMA_VERSION,
        "source_fingerprint": source_fingerprint,
        "loader_sha256": loader_hash,
        "row_count": len(frame),
        "row_id_min": int(frame["row_id"].min()),
        "row_id_max": int(frame["row_id"].max()),
    }
    cached = joblib.load(cache_path) if cache_path.exists() else None
    if isinstance(cached, dict) and cached.get("cache_key") == cache_key:
        maxima = cached["maxima"]
    else:
        entries = frame[["row_id", "datetime", "expiry", "strike"]].copy()
        entries = entries.rename(columns={"datetime": "entry_datetime"})
        connection = duckdb.connect()
        connection.register("tail_entries", entries)
        query = """
        WITH chain AS (
            SELECT
                CAST(CAST(Date AS VARCHAR) || ' ' || CAST(tick_time AS VARCHAR) AS TIMESTAMP)
                    AS quote_available_at,
                CAST(expiry AS TIMESTAMP) AS expiry,
                CAST(strike AS DOUBLE) AS strike,
                CAST(call_price AS DOUBLE) + CAST(put_price AS DOUBLE) AS straddle
            FROM read_csv_auto(?, header=true)
            WHERE CAST(is_monthly AS BOOLEAN)
        )
        SELECT
            e.row_id,
            MAX(c.straddle) FILTER (
                WHERE c.quote_available_at <= e.entry_datetime + INTERVAL '30 minutes'
            ) AS max_straddle_30,
            MAX(c.straddle) FILTER (
                WHERE c.quote_available_at <= e.entry_datetime + INTERVAL '60 minutes'
            ) AS max_straddle_60,
            MAX(c.straddle) FILTER (
                WHERE c.quote_available_at <= e.entry_datetime + INTERVAL '120 minutes'
            ) AS max_straddle_120,
            MAX(c.straddle) AS max_straddle_180
        FROM tail_entries e
        LEFT JOIN chain c
          ON c.expiry = e.expiry
         AND c.strike = e.strike
         AND c.quote_available_at >= e.entry_datetime
         AND c.quote_available_at <= e.entry_datetime + INTERVAL '180 minutes'
        GROUP BY e.row_id
        ORDER BY e.row_id
        """
        maxima = connection.execute(query, [str(full_chain_source)]).fetchdf()
        connection.close()
        if len(maxima) != len(frame) or maxima["row_id"].duplicated().any():
            raise AssertionError("Path-maximum label join is not one-to-one")
        joblib.dump({"cache_key": cache_key, "maxima": maxima}, cache_path, compress=3)
    return frame.merge(maxima, on="row_id", how="left", validate="one_to_one")


def safe_auc(metric, y: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return math.nan
    return float(metric(y, score))


def classifier_metrics(
    y: np.ndarray,
    score: np.ndarray,
    veto_threshold: float | None = None,
    veto: np.ndarray | None = None,
) -> dict[str, float]:
    if veto is None:
        if veto_threshold is None:
            raise ValueError("Provide veto_threshold or veto")
        veto = score >= veto_threshold
    veto = np.asarray(veto, dtype=bool)
    return {
        "rows": int(len(y)),
        "events": int(y.sum()),
        "event_rate": float(y.mean()),
        "average_precision": safe_auc(average_precision_score, y, score),
        "roc_auc": safe_auc(roc_auc_score, y, score),
        "brier": float(brier_score_loss(y, score)),
        "log_loss": float(log_loss(y, np.c_[1.0 - score, score], labels=[0, 1])),
        "veto_rate": float(veto.mean()),
        "tail_recall_at_veto": float(recall_score(y, veto, zero_division=0)),
        "tail_precision_at_veto": float(precision_score(y, veto, zero_division=0)),
    }


def causal_rolling_veto(
    scores: np.ndarray,
    initial_history: np.ndarray,
    veto_fraction: float = VETO_FRACTION,
    window: int = ROLLING_SCORE_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Veto against only earlier scores, adapting to probability-score drift."""
    history = list(np.asarray(initial_history, dtype=float)[-window:])
    veto = np.zeros(len(scores), dtype=bool)
    thresholds = np.empty(len(scores), dtype=float)
    for idx, score in enumerate(np.asarray(scores, dtype=float)):
        if not history:
            raise RuntimeError("Rolling veto requires pre-2024 score history")
        threshold = float(np.quantile(history, 1.0 - veto_fraction))
        thresholds[idx] = threshold
        veto[idx] = score >= threshold
        history.append(float(score))
        if len(history) > window:
            history.pop(0)
    return veto, thresholds


def risk_summary(loss_points: np.ndarray, extreme: np.ndarray) -> dict[str, float]:
    loss_points = np.asarray(loss_points, dtype=float)
    if not len(loss_points):
        return {
            "rows": 0,
            "extreme_event_rate": math.nan,
            "mean_adverse_points": math.nan,
            "p95_adverse_points": math.nan,
            "expected_shortfall_95_points": math.nan,
            "worst_adverse_points": math.nan,
        }
    q95 = float(np.quantile(loss_points, 0.95))
    return {
        "rows": int(len(loss_points)),
        "extreme_event_rate": float(np.mean(extreme)),
        "mean_adverse_points": float(np.mean(loss_points)),
        "p95_adverse_points": q95,
        "expected_shortfall_95_points": float(np.mean(loss_points[loss_points >= q95])),
        "worst_adverse_points": float(np.max(loss_points)),
    }


def expanding_oof_scores(
    model: object, frame: pd.DataFrame, features: list[str], horizon: int
) -> np.ndarray:
    """Purged out-of-sample scores from expanding chronological windows."""
    ordered = frame.sort_values("datetime")
    n = len(ordered)
    scores: list[np.ndarray] = []
    # Each score is produced by a model trained strictly on earlier rows.
    boundaries = [0.40, 0.55, 0.70, 0.85, 1.00]
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        train_end = int(n * left)
        score_end = int(n * right)
        score_part = ordered.iloc[train_end:score_end]
        score_start = score_part["datetime"].min()
        # Purge labels whose t+h observation reaches the scoring fold. This is
        # essential because adjacent intraday rows have overlapping outcomes.
        fit_part = ordered.iloc[:train_end]
        fit_part = fit_part[
            pd.to_datetime(fit_part[f"exit_datetime_{horizon}"]) < score_start
        ]
        if len(fit_part) == 0 or len(score_part) == 0:
            raise RuntimeError(f"Horizon {horizon}: empty purged calibration fold")
        if not (pd.to_datetime(fit_part[f"exit_datetime_{horizon}"]) < score_start).all():
            raise AssertionError(f"Horizon {horizon}: calibration fold was not purged")
        fold_model = clone(model)
        fold_model.fit(fit_part[features], fit_part["is_extreme"])
        scores.append(fold_model.predict_proba(score_part[features])[:, 1])
    return np.concatenate(scores)


def chronological_strategy(
    frame: pd.DataFrame, horizon: int, accept: np.ndarray
) -> tuple[dict[str, float], pd.DataFrame]:
    work = frame[["datetime", "trade_date", "expiry", "strike", "entry_straddle"]].copy()
    work["exit_straddle"] = frame[f"exit_straddle_{horizon}"].to_numpy(float)
    work["tail_probability"] = frame["tail_probability"].to_numpy(float)
    work["accepted"] = np.asarray(accept, dtype=bool)
    work = work.loc[work["accepted"]].sort_values("datetime")
    selected: list[int] = []
    next_available: pd.Timestamp | None = None
    active_day = None
    for idx, row in work.iterrows():
        if active_day is None or row["trade_date"] != active_day:
            active_day = row["trade_date"]
            next_available = None
        if next_available is None or row["datetime"] >= next_available:
            selected.append(idx)
            next_available = row["datetime"] + pd.Timedelta(horizon, unit="m")
    chosen = work.loc[selected].copy()
    chosen["net_short_pnl_points"] = (
        chosen["entry_straddle"] - chosen["exit_straddle"] - SLIPPAGE_POINTS
    )
    pnl = chosen["net_short_pnl_points"].to_numpy(float)
    losses = np.maximum(-pnl, 0.0)
    gains = pnl[pnl > 0].sum()
    loss_sum = -pnl[pnl < 0].sum()
    if len(pnl):
        equity = np.cumsum(pnl)
        peaks = np.maximum.accumulate(np.r_[0.0, equity])
        drawdown = equity - peaks[1:]
    else:
        drawdown = np.array([])
    q95 = float(np.quantile(losses, 0.95)) if len(losses) else math.nan
    es95 = float(losses[losses >= q95].mean()) if len(losses) else math.nan
    metrics = {
        "trades": int(len(chosen)),
        "days": int(chosen["trade_date"].nunique()),
        "net_points_after_2pt_cost": float(pnl.sum()),
        "average_net_points": float(pnl.mean()) if len(pnl) else math.nan,
        "win_rate": float(np.mean(pnl > 0)) if len(pnl) else math.nan,
        "profit_factor": float(gains / loss_sum) if loss_sum > 0 else math.inf,
        "maximum_drawdown_points": float(drawdown.min()) if len(drawdown) else math.nan,
        "p95_trade_loss_points": q95,
        "expected_shortfall_95_trade_loss_points": es95,
        "worst_trade_loss_points": float(losses.max()) if len(losses) else math.nan,
    }
    return metrics, chosen


def train_horizon(
    data: pd.DataFrame, features: list[str], horizon: int, output: Path
) -> dict:
    work = data.copy()
    exit_dt = f"exit_datetime_{horizon}"
    exit_strike = f"exit_strike_{horizon}"
    exact = (
        work[exit_dt].notna()
        & work[exit_strike].eq(work["strike"])
        & ((pd.to_datetime(work[exit_dt]) - work["datetime"]).dt.total_seconds() == horizon * 60)
    )
    # Require one hour of strictly pre-entry signal history. Execution-price
    # changes are deliberately not eligible as predictors.
    usable = work.loc[
        exact
        & work["change_60_signal_straddle"].notna()
        & work[f"max_straddle_{horizon}"].notna()
    ].copy()
    usable["adverse_points"] = (
        usable[f"max_straddle_{horizon}"] - usable["entry_straddle"]
    )
    usable["adverse_fraction"] = (
        usable["adverse_points"] / usable["entry_straddle"].replace(0, np.nan)
    )
    train_core = usable[
        (usable["datetime"] < VALIDATION_START)
        & (usable[exit_dt] < VALIDATION_START)
    ].copy()
    validation = usable[
        (usable["datetime"] >= VALIDATION_START)
        & (usable["datetime"] < TEST_START)
        & (usable[exit_dt] < TEST_START)
    ].copy()
    pretest = usable[
        (usable["datetime"] < TEST_START) & (usable[exit_dt] < TEST_START)
    ].copy()
    test = usable[usable["datetime"] >= TEST_START].copy()
    if min(len(train_core), len(validation), len(test)) == 0:
        raise RuntimeError(f"Horizon {horizon}: empty train, validation, or test partition")
    if not (pd.to_datetime(train_core[exit_dt]) < VALIDATION_START).all():
        raise AssertionError(f"Horizon {horizon}: training labels cross into validation")
    if not (pd.to_datetime(validation[exit_dt]) < TEST_START).all():
        raise AssertionError(f"Horizon {horizon}: validation labels cross into test")
    if not (pd.to_datetime(pretest[exit_dt]) < TEST_START).all():
        raise AssertionError(f"Horizon {horizon}: fitted labels cross into test")
    if not (test["datetime"] >= TEST_START).all():
        raise AssertionError(f"Horizon {horizon}: pre-2024 row entered the test set")

    # Frozen definition of an extreme loss. Neither validation nor test labels
    # influence this cutoff.
    tail_cutoff = float(train_core["adverse_fraction"].quantile(TAIL_QUANTILE))
    for part in (train_core, validation, pretest, test):
        part["is_extreme"] = (part["adverse_fraction"] >= tail_cutoff).astype(int)

    validation_results: list[dict] = []
    models = candidate_models()
    for name, model in models.items():
        model.fit(train_core[features], train_core["is_extreme"])
        score = model.predict_proba(validation[features])[:, 1]
        threshold = float(np.quantile(score, 1.0 - VETO_FRACTION))
        validation_results.append(
            {
                "model": name,
                "veto_threshold": threshold,
                **classifier_metrics(validation["is_extreme"].to_numpy(), score, threshold),
            }
        )
    # A veto model is judged first by how many true tail events it captures at
    # the fixed veto budget, then by ranking quality and calibration.
    validation_results.sort(
        key=lambda r: (
            r["tail_recall_at_veto"],
            r["average_precision"],
            -r["brier"],
        ),
        reverse=True,
    )
    winner_name = validation_results[0]["model"]
    winner = models[winner_name]
    oof_score = expanding_oof_scores(winner, pretest, features, horizon)
    veto_threshold = float(np.quantile(oof_score, 1.0 - VETO_FRACTION))
    winner.fit(pretest[features], pretest["is_extreme"])
    # Operational cutoff comes from expanding-window out-of-sample scores, not
    # the final model's overconfident in-sample probabilities.
    test_score = winner.predict_proba(test[features])[:, 1]
    test["tail_probability"] = test_score
    rolling_veto, rolling_threshold = causal_rolling_veto(test_score, oof_score)
    test["veto_threshold"] = rolling_threshold
    test["veto"] = rolling_veto
    test_metrics = classifier_metrics(
        test["is_extreme"].to_numpy(), test_score, veto=rolling_veto
    )

    all_risk = risk_summary(
        test["adverse_points"].to_numpy(), test["is_extreme"].to_numpy()
    )
    accepted = ~test["veto"].to_numpy()
    accepted_risk = risk_summary(
        test.loc[accepted, "adverse_points"].to_numpy(),
        test.loc[accepted, "is_extreme"].to_numpy(),
    )
    vetoed_risk = risk_summary(
        test.loc[~accepted, "adverse_points"].to_numpy(),
        test.loc[~accepted, "is_extreme"].to_numpy(),
    )
    baseline_strategy, _ = chronological_strategy(test, horizon, np.ones(len(test), dtype=bool))
    veto_strategy, trades = chronological_strategy(test, horizon, accepted)
    trades.to_csv(output / f"test_accepted_trades_{horizon}m.csv", index=False)
    test[
        [
            "datetime", "trade_date", "expiry", "strike", "entry_straddle",
            f"exit_straddle_{horizon}", f"max_straddle_{horizon}",
            "adverse_points", "adverse_fraction",
            "is_extreme", "tail_probability", "veto",
            "veto_threshold",
        ]
    ].to_csv(output / f"test_scores_{horizon}m.csv", index=False)

    artifact = {
        "model": winner,
        "model_name": winner_name,
        "horizon_minutes": horizon,
        "features": features,
        "training_end_exclusive": str(TEST_START.date()),
        "tail_definition": "(maximum fixed-strike straddle observed from entry through t+h - entry straddle) / entry straddle",
        "tail_quantile_from_train_core": TAIL_QUANTILE,
        "tail_cutoff_fraction": tail_cutoff,
        "veto_threshold": veto_threshold,
        "veto_fraction_design_target": VETO_FRACTION,
        "rolling_score_window": ROLLING_SCORE_WINDOW,
        "rolling_score_history": oof_score[-ROLLING_SCORE_WINDOW:],
        "strike_rule": "K_t is exact ATM in the known signal snapshot; execution uses the next quote and K_t plus expiry remain fixed through exact t+h",
        "timing_rule": "features are known before entry_quote_available_at; entry and exit horizons use quote-availability timestamps",
        "calibration_rule": "expanding chronological folds purge every training label whose exit reaches the scoring fold; live veto threshold is the trailing score-distribution percentile",
        "cache_schema_version": CACHE_SCHEMA_VERSION,
    }
    artifact_path = output / f"tail_veto_{horizon}m.joblib"
    joblib.dump(artifact, artifact_path)

    return {
        "horizon_minutes": horizon,
        "selected_model": winner_name,
        "artifact": artifact_path.name,
        "tail_cutoff_fraction": tail_cutoff,
        "tail_cutoff_percent": tail_cutoff * 100.0,
        "veto_threshold": veto_threshold,
        "available_exact_same_strike_rows": int(exact.sum()),
        "usable_rows": int(len(usable)),
        "train_core_rows": int(len(train_core)),
        "validation_rows": int(len(validation)),
        "pre_2024_refit_rows": int(len(pretest)),
        "test_rows": int(len(test)),
        "same_strike_rate": float(work.loc[exact, exit_strike].eq(work.loc[exact, "strike"]).mean()),
        "exact_horizon_rate": float(
            ((work.loc[exact, exit_dt] - work.loc[exact, "datetime"]).dt.total_seconds() == horizon * 60).mean()
        ),
        "validation_candidates": validation_results,
        "test_classifier": test_metrics,
        "test_risk_all_entries": all_risk,
        "test_risk_accepted_entries": accepted_risk,
        "test_risk_vetoed_entries": vetoed_risk,
        "test_strategy_no_veto": baseline_strategy,
        "test_strategy_with_veto": veto_strategy,
    }


def write_report(
    output: Path, source: Path, frame: pd.DataFrame, features: list[str], results: list[dict]
) -> None:
    # Preselect the horizon using validation tail recall. Test metrics are not
    # involved in this decision.
    best = max(
        results,
        key=lambda r: (
            r["validation_candidates"][0]["tail_recall_at_veto"],
            r["validation_candidates"][0]["average_precision"],
        ),
    )
    lines = [
        "# Tail-Risk Veto Models — Fixed Entry ATM Strike",
        "",
        "## Recommendation",
        "",
        f"The pre-2024 validation winner is the **{best['horizon_minutes']}-minute {best['selected_model']}** model. At a design veto budget of {VETO_FRACTION:.0%}, it captured **{best['validation_candidates'][0]['tail_recall_at_veto']:.1%}** of validation tail events.",
        "",
        f"On the locked January 2024 onward benchmark, it captured **{best['test_classifier']['tail_recall_at_veto']:.1%}** of extreme events while vetoing **{best['test_classifier']['veto_rate']:.1%}** of candidate entries. Accepted-entry extreme-event frequency changed from **{best['test_risk_all_entries']['extreme_event_rate']:.1%}** to **{best['test_risk_accepted_entries']['extreme_event_rate']:.1%}**.",
        "",
        "The improved winner is a research candidate, not yet deployment-approved: benchmark recall remains below its pre-2024 validation estimate and should be confirmed on a new future period. The benchmark horizon rows are diagnostics and are not used to replace it with a post-hoc winner.",
        "",
        "## Contract and leakage controls",
        "",
        "- K_t is the exact ATM strike in the known signal snapshot; execution is delayed to the next available quote.",
        "- K_t and its expiry remain fixed through the exact t+h execution-quote observation; the label never reselects ATM at exit.",
        "- Entry and exit clocks use quote-availability timestamps, not nominal bar timestamps.",
        "- Horizons are 30, 60, 120, and 180 minutes.",
        "- The tail cutoff is the pre-July-2023 90th percentile of the maximum fixed-strike straddle expansion anywhere inside t through t+h, divided by entry premium.",
        "- Model family is selected only on July-December 2023 validation tail recall at a 20% veto budget.",
        "- Final models are refit on all eligible pre-2024 rows. January 2024 onward is test-only.",
        "- Expanding threshold-calibration folds purge all training labels whose t+h exit reaches the scoring fold.",
        "- The live veto threshold is the 80th percentile of the preceding 1,000 model scores; it never uses future scores or labels.",
        "- The feature cache is versioned and bound to the preprocessing loader SHA-256 and both source-file fingerprints.",
        "- Exit labels use the full option chain, retaining the entry strike if it moves outside the five-strike snapshot.",
        "",
        "## Data",
        "",
        f"- Source: `{source}`",
        f"- Candidate entry rows: {len(frame):,}",
        f"- Entry-time features: {len(features)}",
        f"- Test range: {frame.loc[frame['datetime'] >= TEST_START, 'datetime'].min()} through {frame['datetime'].max()}",
        "",
        "## Out-of-sample comparison",
        "",
        "| Horizon | Selected model | Test rows | Tail rate | AP | ROC AUC | Veto rate | Tail recall | Accepted tail rate | ES95 before | ES95 accepted | Strategy max DD before | Strategy max DD after |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        c = r["test_classifier"]
        a = r["test_risk_all_entries"]
        k = r["test_risk_accepted_entries"]
        b = r["test_strategy_no_veto"]
        v = r["test_strategy_with_veto"]
        lines.append(
            f"| {r['horizon_minutes']}m | {r['selected_model']} | {c['rows']:,} | {c['event_rate']:.1%} | {c['average_precision']:.3f} | {c['roc_auc']:.3f} | {c['veto_rate']:.1%} | {c['tail_recall_at_veto']:.1%} | {k['extreme_event_rate']:.1%} | {a['expected_shortfall_95_points']:.1f} | {k['expected_shortfall_95_points']:.1f} | {b['maximum_drawdown_points']:.1f} | {v['maximum_drawdown_points']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Pre-2024 validation model comparison",
            "",
            "| Horizon | Model | Tail recall at 20% veto | Average precision | ROC AUC | Brier | Selected |",
            "|---:|---|---:|---:|---:|---:|:---:|",
        ]
    )
    for r in results:
        for candidate in r["validation_candidates"]:
            lines.append(
                f"| {r['horizon_minutes']}m | {candidate['model']} | {candidate['tail_recall_at_veto']:.1%} | {candidate['average_precision']:.3f} | {candidate['roc_auc']:.3f} | {candidate['brier']:.3f} | {'yes' if candidate['model'] == r['selected_model'] else ''} |"
            )
    lines.extend(
        [
            "",
            "## How to use the veto",
            "",
            "Score a proposed entry with the artifact for its intended holding horizon. Initialize score history from `rolling_score_history`; before each decision, compute the 80th percentile of the most recent `rolling_score_window` scores and reject when `tail_probability` meets or exceeds it. Append the current score only after deciding. This is a risk gate, not an entry signal.",
            "",
            "Expected shortfall (ES95) is the average maximum adverse fixed-strike straddle expansion among the worst 5% of holding windows. Strategy drawdown still uses executable endpoint P&L from chronological, non-overlapping intraday positions and deducts two points per round trip; statutory charges are not included.",
            "",
            "The recommended horizon was selected before inspecting the final test metrics. The other horizon test results are diagnostics, not additional selection evidence.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    location_file = Path(__file__).resolve().parent / "DatasetLocation.txt"
    source_from_txt = Path(location_file.read_text(encoding="utf-8").strip().strip('"'))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=source_from_txt,
    )
    parser.add_argument(
        "--full-chain",
        type=Path,
        default=Path(os.environ.get("VOLTR_43W_FULL_CHAIN_SOURCE", Path(__file__).resolve().parents[3] / "pipelines" / "weeks43" / "data" / "icharts_option_banknifty.csv")),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/tail_risk_veto")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    fixed_atm.HORIZONS = HORIZONS
    cache = args.output / "fixed_atm_feature_cache.joblib"
    loader_path = Path(fixed_atm.__file__).resolve()
    loader_hash = hashlib.sha256(loader_path.read_bytes()).hexdigest()
    source_fingerprint = {
        "entry_path": str(args.source.resolve()),
        "entry_size": args.source.stat().st_size,
        "entry_mtime_ns": args.source.stat().st_mtime_ns,
        "full_chain_path": str(args.full_chain.resolve()),
        "full_chain_size": args.full_chain.stat().st_size,
        "full_chain_mtime_ns": args.full_chain.stat().st_mtime_ns,
    }
    cached = joblib.load(cache) if cache.exists() else None
    cache_is_current = bool(
        isinstance(cached, dict)
        and cached.get("schema_version") == CACHE_SCHEMA_VERSION
        and cached.get("loader_sha256") == loader_hash
        and cached.get("source_fingerprint") == source_fingerprint
    )
    if cache_is_current:
        frame, features = cached["frame"], cached["features"]
    else:
        frame = fixed_atm.load_fixed_strike_rows(args.source, args.full_chain)
        # Keep only entries whose delayed execution quote exists and is known
        # strictly after the signal snapshot. Use quote-availability times as
        # the actual entry/exit clock so every horizon is measured causally.
        valid_entry = (
            frame["entry_quote_available_at"].notna()
            & frame["entry_straddle"].notna()
            & frame["signal_quote_available_at"].lt(frame["entry_quote_available_at"])
        )
        frame = frame.loc[valid_entry].copy()
        frame["datetime"] = pd.to_datetime(frame["entry_quote_available_at"])
        frame["trade_date"] = frame["datetime"].dt.normalize()
        for horizon in HORIZONS:
            frame[f"exit_datetime_{horizon}"] = pd.to_datetime(
                frame[f"exit_quote_available_at_{horizon}"]
            )
        frame, features = fixed_atm.add_entry_features(frame)
        joblib.dump(
            {
                "schema_version": CACHE_SCHEMA_VERSION,
                "loader_sha256": loader_hash,
                "source_fingerprint": source_fingerprint,
                "frame": frame,
                "features": features,
            },
            cache,
            compress=3,
        )
    frame = attach_path_maxima(
        frame,
        args.full_chain,
        args.output / "fixed_contract_path_maxima.joblib",
        source_fingerprint,
        loader_hash,
    )
    frame, features = add_tail_features(frame, features)
    forbidden = [
        f for f in features
        if f == "entry_straddle"
        or f.startswith(("entry_", "exit_", "target_", "label_", "adverse_"))
    ]
    if forbidden:
        raise AssertionError(f"Future data entered features: {forbidden}")
    required_timing = {"signal_quote_available_at", "datetime", "entry_quote_available_at"}
    if not required_timing.issubset(frame.columns):
        raise AssertionError("Leak-free quote-availability columns are missing from the cache")
    if not frame["signal_quote_available_at"].lt(frame["entry_quote_available_at"]).all():
        raise AssertionError("A signal quote is not known before its execution quote")
    results = []
    for horizon in HORIZONS:
        result = train_horizon(frame, features, horizon, args.output)
        results.append(result)
        print(
            json.dumps(
                {
                    "horizon": horizon,
                    "selected_model": result["selected_model"],
                    "test": result["test_classifier"],
                },
                indent=2,
            )
        )
    manifest = {
        "source": str(args.source),
        "full_chain_source": str(args.full_chain),
        "test_start": str(TEST_START.date()),
        "validation_start": str(VALIDATION_START.date()),
        "horizons": list(HORIZONS),
        "tail_quantile": TAIL_QUANTILE,
        "veto_fraction": VETO_FRACTION,
        "features": features,
        "results": results,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_report(args.output, args.source, frame, features, results)


if __name__ == "__main__":
    main()
