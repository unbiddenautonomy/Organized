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
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


HORIZONS = (30, 60, 90, 120, 180)
LAGS = (5, 15, 30, 60)
TEST_START = pd.Timestamp("2024-01-01")
VALIDATION_START = pd.Timestamp("2023-07-01")
MODEL_SELECTION_START = pd.Timestamp("2023-01-01")
SLIPPAGE_POINTS = 2.0
LATE_SESSION_HORIZON = 90
LATE_SESSION_START_MINUTE = 13 * 60 + 30


def load_fixed_strike_rows(entry_source: Path, full_chain_source: Path) -> pd.DataFrame:
    # The row stamped at signal_datetime is actually available at tick_time
    # (normally one second later).  The contract is selected from that known
    # signal snapshot, but execution is deliberately delayed to the next
    # one-minute quote.  Consequently no same-quote price is both a feature
    # and the assumed fill price.
    entry_sql = """
    WITH raw AS (
        SELECT
            CAST(datetime AS TIMESTAMP) AS signal_datetime,
            CAST(Date AS DATE) AS trade_date,
            CAST(tick_time AS VARCHAR) AS signal_tick_time,
            CAST(expiry AS DATE) AS expiry,
            CAST(strike AS DOUBLE) AS strike,
            CAST(atm_strike AS DOUBLE) AS atm_strike,
            CAST(days_to_expiry AS DOUBLE) AS days_to_expiry,
            CAST(spot_price AS DOUBLE) AS spot_price,
            CAST(future_open AS DOUBLE) AS future_open,
            CAST(Future_oi AS DOUBLE) AS future_oi,
            CAST(call_price AS DOUBLE) AS call_price,
            CAST(put_price AS DOUBLE) AS put_price,
            CAST(call_oi AS DOUBLE) AS call_oi,
            CAST(put_oi AS DOUBLE) AS put_oi,
            CAST(iv_call AS DOUBLE) AS iv_call,
            CAST(iv_put AS DOUBLE) AS iv_put,
            CAST(delta_call AS DOUBLE) AS delta_call,
            CAST(delta_put AS DOUBLE) AS delta_put,
            CAST(vega_call AS DOUBLE) AS vega_call,
            CAST(vega_put AS DOUBLE) AS vega_put,
            CAST(gamma_call AS DOUBLE) AS gamma_call,
            CAST(gamma_put AS DOUBLE) AS gamma_put,
            CAST(theta_call AS DOUBLE) AS theta_call,
            CAST(theta_put AS DOUBLE) AS theta_put,
            dte_flag,
            is_monthly
        FROM read_csv_auto(?, header=true)
        WHERE dte_flag = 'M1' AND CAST(is_monthly AS BOOLEAN)
          AND days_to_expiry >= 21.0 AND days_to_expiry < 35.0
    ), chosen_expiry AS (
        SELECT signal_datetime, expiry
        FROM raw
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY signal_datetime ORDER BY expiry ASC
        ) = 1
    ), base AS (
        SELECT r.* FROM raw r
        INNER JOIN chosen_expiry c USING (signal_datetime, expiry)
        QUALIFY ROW_NUMBER() OVER (
            PARTITION BY signal_datetime, expiry, strike ORDER BY signal_datetime
        ) = 1
    ), entries AS (
        SELECT * FROM base
        WHERE strike = atm_strike
    )
    SELECT *, call_price + put_price AS signal_straddle
    FROM entries ORDER BY signal_datetime
    """
    connection = duckdb.connect()
    frame = connection.execute(entry_sql, [str(entry_source)]).fetchdf()
    frame["signal_datetime"] = pd.to_datetime(frame["signal_datetime"])
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    frame["signal_quote_available_at"] = pd.to_datetime(
        frame["trade_date"].dt.strftime("%Y-%m-%d")
        + " "
        + frame["signal_tick_time"].astype(str),
        errors="coerce",
    )
    # One full minute of execution latency is conservative for this source:
    # 99%+ of quotes arrive one second after their nominal datetime.
    frame["datetime"] = frame["signal_datetime"] + pd.offsets.Minute(1)
    if not frame["signal_quote_available_at"].lt(frame["datetime"]).all():
        raise AssertionError("A signal quote is not available before its execution timestamp")
    frame["trade_date"] = frame["datetime"].dt.normalize()
    frame = frame.reset_index(drop=True)
    frame.insert(0, "row_id", np.arange(len(frame), dtype=np.int64))

    wanted_parts = []
    for horizon in (0, *HORIZONS):
        wanted_parts.append(
            pd.DataFrame(
                {
                    "row_id": frame["row_id"],
                    "horizon": horizon,
                    "quote_datetime": frame["datetime"] + pd.Timedelta(horizon, unit="m"),
                    "expiry": frame["expiry"],
                    "strike": frame["strike"],
                }
            )
        )
    wanted = pd.concat(wanted_parts, ignore_index=True)
    connection.register("wanted_quotes", wanted)
    # The full chain contains the original entry strike even after it moves
    # outside the five-strike snapshot.  Joining it prevents future-dependent
    # label availability from deciding which rows enter the experiment.
    exit_sql = """
    WITH chain AS (
        SELECT
            CAST(datetime AS TIMESTAMP) AS quote_datetime,
            CAST(Date AS DATE) AS quote_date,
            CAST(tick_time AS VARCHAR) AS quote_tick_time,
            CAST(expiry AS TIMESTAMP) AS expiry,
            CAST(strike AS DOUBLE) AS strike,
            CAST(call_price AS DOUBLE) AS quote_call_price,
            CAST(put_price AS DOUBLE) AS quote_put_price
        FROM read_csv_auto(?, header=true)
        WHERE CAST(is_monthly AS BOOLEAN)
    ), matched AS (
        SELECT
            w.row_id, w.horizon, c.quote_datetime, c.quote_date,
            c.quote_tick_time, c.expiry, c.strike,
            c.quote_call_price, c.quote_put_price,
            ROW_NUMBER() OVER (
                PARTITION BY w.row_id, w.horizon ORDER BY c.quote_datetime
            ) AS rn
        FROM wanted_quotes w
        INNER JOIN chain c
          ON c.quote_datetime = w.quote_datetime
         AND c.expiry = w.expiry
         AND c.strike = w.strike
    )
    SELECT * EXCLUDE (rn) FROM matched WHERE rn = 1
    """
    quotes = connection.execute(exit_sql, [str(full_chain_source)]).fetchdf()
    connection.close()
    quotes["quote_datetime"] = pd.to_datetime(quotes["quote_datetime"])
    quotes["quote_available_at"] = pd.to_datetime(
        pd.to_datetime(quotes["quote_date"]).dt.strftime("%Y-%m-%d")
        + " "
        + quotes["quote_tick_time"].astype(str),
        errors="coerce",
    )

    entries = quotes.loc[quotes["horizon"] == 0].copy().rename(
        columns={
            "quote_datetime": "entry_quote_datetime",
            "quote_available_at": "entry_quote_available_at",
            "strike": "entry_strike",
            "quote_call_price": "entry_call_price",
            "quote_put_price": "entry_put_price",
        }
    )
    entries["entry_straddle"] = entries["entry_call_price"] + entries["entry_put_price"]
    frame = frame.merge(
        entries[[
            "row_id", "entry_quote_datetime", "entry_quote_available_at", "entry_strike",
            "entry_straddle",
        ]],
        on="row_id",
        how="left",
        validate="one_to_one",
    )

    for horizon in HORIZONS:
        matched = quotes.loc[quotes["horizon"] == horizon].copy()
        matched = matched.rename(
            columns={
                "quote_datetime": f"exit_datetime_{horizon}",
                "quote_available_at": f"exit_quote_available_at_{horizon}",
                "strike": f"exit_strike_{horizon}",
                "quote_call_price": f"exit_call_price_{horizon}",
                "quote_put_price": f"exit_put_price_{horizon}",
            }
        )
        matched[f"exit_straddle_{horizon}"] = (
            matched[f"exit_call_price_{horizon}"] + matched[f"exit_put_price_{horizon}"]
        )
        keep = [
            "row_id", f"exit_datetime_{horizon}", f"exit_quote_available_at_{horizon}",
            f"exit_strike_{horizon}",
            f"exit_straddle_{horizon}",
        ]
        frame = frame.merge(matched[keep], on="row_id", how="left", validate="one_to_one")
        frame[f"exit_datetime_{horizon}"] = pd.to_datetime(frame[f"exit_datetime_{horizon}"])
    return frame


def add_entry_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = frame.sort_values(["datetime", "expiry"]).copy()
    out["basis"] = out["future_open"] / out["spot_price"] - 1.0
    out["oi_ratio"] = out["put_oi"] / out["call_oi"].replace(0, np.nan)
    out["premium_balance"] = (out["call_price"] - out["put_price"]) / out["signal_straddle"].replace(0, np.nan)
    out["moneyness"] = (out["spot_price"] - out["strike"]) / 100.0
    minute = out["datetime"].dt.hour * 60 + out["datetime"].dt.minute - 555
    out["time_sin"] = np.sin(2 * np.pi * minute / 375.0)
    out["time_cos"] = np.cos(2 * np.pi * minute / 375.0)
    out["day_of_week"] = out["datetime"].dt.dayofweek
    out["four_week"] = (out["days_to_expiry"] >= 28.0).astype(float)
    out["iv_mean"] = (out["iv_call"] + out["iv_put"]) / 2.0
    out["iv_skew"] = out["iv_put"] - out["iv_call"]
    out["oi_imbalance"] = (out["put_oi"] - out["call_oi"]) / (
        out["put_oi"] + out["call_oi"]
    ).replace(0, np.nan)
    out["theta_total"] = out["theta_call"] + out["theta_put"]
    out["vega_total"] = out["vega_call"] + out["vega_put"]
    out["gamma_total"] = out["gamma_call"] + out["gamma_put"]
    out["theta_per_premium"] = out["theta_total"] / out["signal_straddle"].replace(0, np.nan)
    out["gamma_per_premium"] = out["gamma_total"] / out["signal_straddle"].replace(0, np.nan)
    out["premium_per_dte"] = out["signal_straddle"] / out["days_to_expiry"].replace(0, np.nan)
    month_angle = 2 * np.pi * (out["datetime"].dt.month - 1) / 12.0
    out["month_sin"] = np.sin(month_angle)
    out["month_cos"] = np.cos(month_angle)

    lag_sources = [
        "spot_price", "future_open", "signal_straddle", "oi_ratio", "iv_call", "iv_put",
    ]
    keys = ["datetime", "trade_date", "expiry"]
    for lag in LAGS:
        previous = out[keys + lag_sources].copy()
        previous["datetime"] = previous["datetime"] + pd.Timedelta(lag, unit="m")
        previous = previous.rename(columns={c: f"lag_{lag}_{c}" for c in lag_sources})
        out = out.merge(previous, on=keys, how="left", validate="one_to_one")
        ratio = out["future_open"] / out[f"lag_{lag}_future_open"].replace(0, np.nan)
        out[f"future_return_{lag}"] = np.log(ratio.where(ratio > 0))
        spot_ratio = out["spot_price"] / out[f"lag_{lag}_spot_price"].replace(0, np.nan)
        out[f"spot_return_{lag}"] = np.log(spot_ratio.where(spot_ratio > 0))
        out[f"abs_future_return_{lag}"] = out[f"future_return_{lag}"].abs()
        out[f"straddle_return_{lag}"] = (
            out["signal_straddle"] / out[f"lag_{lag}_signal_straddle"].replace(0, np.nan) - 1.0
        )
        for source_name in ("signal_straddle", "oi_ratio", "iv_call", "iv_put"):
            out[f"change_{lag}_{source_name}"] = out[source_name] - out[f"lag_{lag}_{source_name}"]

    features = [
        "days_to_expiry", "basis", "signal_straddle", "oi_ratio", "premium_balance", "moneyness",
        "time_sin", "time_cos", "day_of_week", "four_week", "future_oi",
        "call_oi", "put_oi", "iv_call", "iv_put", "delta_call", "delta_put", "vega_call",
        "vega_put", "gamma_call", "gamma_put", "theta_call", "theta_put",
        "iv_mean", "iv_skew", "oi_imbalance", "theta_total", "vega_total", "gamma_total",
        "theta_per_premium", "gamma_per_premium", "premium_per_dte", "month_sin", "month_cos",
    ]
    for lag in LAGS:
        features.extend(
            [
                f"future_return_{lag}", f"spot_return_{lag}", f"abs_future_return_{lag}",
                f"straddle_return_{lag}", f"change_{lag}_signal_straddle", f"change_{lag}_oi_ratio",
                f"change_{lag}_iv_call", f"change_{lag}_iv_put",
            ]
        )
    return out, features


def candidate_models() -> dict[str, object]:
    ridge = lambda alpha: TransformedTargetRegressor(
        regressor=Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        ),
        transformer=StandardScaler(),
    )
    return {
        "ridge_alpha_10": ridge(10.0),
        "ridge_alpha_30": ridge(30.0),
        "ridge_alpha_100": ridge(100.0),
        "hist_gb_l2": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(
                    loss="squared_error", learning_rate=0.05, max_iter=250,
                    max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=10.0,
                    random_state=42,
                )),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(
                    n_estimators=240, min_samples_leaf=20, max_features=0.7,
                    n_jobs=-1, random_state=42,
                )),
            ]
        ),
        "lightgbm_huber": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(
                    objective="huber", n_estimators=350, learning_rate=0.03,
                    num_leaves=15, min_child_samples=100, reg_alpha=2.0,
                    reg_lambda=20.0, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=-1, verbosity=-1,
                )),
            ]
        ),
        "lightgbm_l1": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", lgb.LGBMRegressor(
                    objective="regression_l1", n_estimators=300, learning_rate=0.03,
                    num_leaves=15, min_child_samples=100, reg_alpha=2.0,
                    reg_lambda=20.0, subsample=0.8, colsample_bytree=0.8,
                    random_state=43, n_jobs=-1, verbosity=-1,
                )),
            ]
        ),
    }


def prediction_metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    corr = (
        float(np.corrcoef(y, pred)[0, 1])
        if len(y) > 1 and np.std(y) > 0 and np.std(pred) > 0
        else math.nan
    )
    direction = float(np.mean((y > 0) == (pred > 0)))
    return {
        "rows": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(mean_squared_error(y, pred) ** 0.5),
        "r2": float(r2_score(y, pred)),
        "correlation": corr,
        "positive_direction_accuracy": direction,
    }


def non_overlapping_strategy(
    meta: pd.DataFrame, y: np.ndarray, pred: np.ndarray, horizon: int, threshold: float
) -> tuple[dict[str, float], pd.DataFrame]:
    trades = meta[["datetime", "trade_date", "expiry", "strike"]].copy()
    trades["actual_net_points_after_2pt_cost"] = y
    trades["prediction"] = pred
    trades = trades[trades["prediction"] >= threshold].sort_values("datetime")
    selected: list[int] = []
    next_available: pd.Timestamp | None = None
    active_day: pd.Timestamp | None = None
    for idx, row in trades.iterrows():
        day = row["trade_date"]
        if active_day is None or day != active_day:
            active_day = day
            next_available = None
        if next_available is None or row["datetime"] >= next_available:
            selected.append(idx)
            next_available = row["datetime"] + pd.Timedelta(horizon, unit="m")
    chosen = trades.loc[selected].copy()
    pnl = chosen["actual_net_points_after_2pt_cost"].to_numpy(float)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(np.r_[0.0, equity])[-len(equity):] if len(equity) else np.array([])
    # Recompute drawdown with a zero starting point for clarity.
    if len(equity):
        peaks = np.maximum.accumulate(np.r_[0.0, equity])
        drawdown = equity - peaks[1:]
    metrics = {
        "prediction_threshold_net_points": float(threshold),
        "trades": int(len(chosen)),
        "signal_days": int(chosen["trade_date"].nunique()),
        "win_rate_after_slippage": float(np.mean(pnl > 0)) if len(pnl) else math.nan,
        "gross_points_before_cost": float(chosen["actual_net_points_after_2pt_cost"].sum() + len(chosen) * SLIPPAGE_POINTS),
        "net_points_after_2pt_slippage": float(pnl.sum()),
        "average_net_points": float(pnl.mean()) if len(pnl) else math.nan,
        "profit_factor": float(gains / losses) if losses > 0 else math.inf,
        "maximum_drawdown_points": float(drawdown.min()) if len(drawdown) else math.nan,
    }
    return metrics, chosen


def train_horizon(
    data: pd.DataFrame, features: list[str], horizon: int, output: Path
) -> tuple[dict, pd.DataFrame]:
    target = f"target_{horizon}"
    exit_dt = f"exit_datetime_{horizon}"
    exit_strike = f"exit_strike_{horizon}"
    data = data.copy()
    # Directly model executable decay: fixed-strike premium decay less the
    # assumed round-trip cost.  A constant cost does not change model ranking,
    # but keeping it in the label prevents downstream users from treating a
    # gross prediction as a net edge.
    data[target] = (
        data["entry_straddle"] - data[f"exit_straddle_{horizon}"] - SLIPPAGE_POINTS
    )
    entry_exact = (
        data["entry_quote_datetime"].notna()
        & data["entry_strike"].eq(data["strike"])
        & pd.to_datetime(data["entry_quote_datetime"]).eq(data["datetime"])
        & pd.to_datetime(data["signal_quote_available_at"]).lt(
            pd.to_datetime(data["entry_quote_available_at"])
        )
    )
    scheduled_exit = data["datetime"] + pd.Timedelta(horizon, unit="m")
    scheduled_same_day = scheduled_exit.dt.normalize().eq(data["datetime"].dt.normalize())
    scheduled_before_close = (
        scheduled_exit.dt.hour * 60 + scheduled_exit.dt.minute
    ).le(15 * 60 + 29)
    schedule_eligible = scheduled_same_day & scheduled_before_close
    exact = (
        schedule_eligible
        & entry_exact
        & data[exit_dt].notna()
        & data[exit_strike].eq(data["strike"])
        & ((pd.to_datetime(data[exit_dt]) - data["datetime"]).dt.total_seconds() == horizon * 60)
        & (
            (
                pd.to_datetime(data[f"exit_quote_available_at_{horizon}"])
                - pd.to_datetime(data["entry_quote_available_at"])
            ).dt.total_seconds()
            == horizon * 60
        )
    )
    # Missing historical lags are known at entry and are handled by the
    # train-fitted imputers. They must not remove otherwise valid labels.
    eligibility = exact
    if horizon == LATE_SESSION_HORIZON:
        minute_of_day = data["datetime"].dt.hour * 60 + data["datetime"].dt.minute
        eligibility = exact & minute_of_day.ge(LATE_SESSION_START_MINUTE)
    usable = data.loc[eligibility].copy()
    train_core = usable[
        (usable["datetime"] < MODEL_SELECTION_START)
        & (pd.to_datetime(usable[exit_dt]) < MODEL_SELECTION_START)
    ]
    model_validation = usable[
        (usable["datetime"] >= MODEL_SELECTION_START)
        & (usable["datetime"] < VALIDATION_START)
        & (pd.to_datetime(usable[exit_dt]) < VALIDATION_START)
    ]
    prepolicy = usable[
        (usable["datetime"] < VALIDATION_START)
        & (pd.to_datetime(usable[exit_dt]) < VALIDATION_START)
    ]
    validation = usable[
        (usable["datetime"] >= VALIDATION_START)
        & (usable["datetime"] < TEST_START)
        & (pd.to_datetime(usable[exit_dt]) < TEST_START)
    ]
    pretest = usable[
        (usable["datetime"] < TEST_START)
        & (pd.to_datetime(usable[exit_dt]) < TEST_START)
    ]
    test = usable[usable["datetime"] >= TEST_START]
    if min(len(train_core), len(model_validation), len(validation), len(test)) == 0:
        raise RuntimeError(
            f"Horizon {horizon} has an empty train, model-selection, policy-validation, or test partition"
        )

    candidates = candidate_models()
    validation_results = []
    for name, model in candidates.items():
        model.fit(train_core[features], train_core[target])
        pred = model.predict(model_validation[features])
        validation_results.append(
            {"model": name, **prediction_metrics(model_validation[target].to_numpy(), pred)}
        )
    validation_results.sort(key=lambda row: row["mae"])
    winner_name = validation_results[0]["model"]
    ridge_name = next(
        row["model"] for row in validation_results if row["model"].startswith("ridge_")
    )

    minimum_trades = 10 if horizon == LATE_SESSION_HORIZON else 20
    threshold_grid = (0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0)

    def calibrate_policy(model_name: str) -> tuple[object, np.ndarray, dict, list[dict]]:
        calibration_model = candidate_models()[model_name]
        calibration_model.fit(prepolicy[features], prepolicy[target])
        pred = calibration_model.predict(validation[features])
        policies: list[dict] = []
        for candidate_threshold in threshold_grid:
            policy_metrics, policy_trades = non_overlapping_strategy(
                validation,
                validation[target].to_numpy(),
                pred,
                horizon,
                candidate_threshold,
            )
            if len(policy_trades):
                monthly = policy_trades.assign(
                    month=pd.to_datetime(policy_trades["datetime"]).dt.to_period("M").astype(str)
                ).groupby("month")["actual_net_points_after_2pt_cost"].sum()
                profitable_months = int((monthly > 0).sum())
                evaluated_months = int(len(monthly))
            else:
                profitable_months = 0
                evaluated_months = 0
            activity_ok = (
                policy_metrics["trades"] >= minimum_trades
                and policy_metrics["signal_days"] >= min(10, minimum_trades)
            )
            stability_ratio = profitable_months / evaluated_months if evaluated_months else 0.0
            economic_score = (
                policy_metrics["net_points_after_2pt_slippage"]
                + 0.25 * policy_metrics["maximum_drawdown_points"]
            ) if activity_ok else -math.inf
            policy = {
                "threshold": candidate_threshold,
                "activity_ok": activity_ok,
                "profitable_months": profitable_months,
                "evaluated_months": evaluated_months,
                "profitable_month_ratio": stability_ratio,
                "economic_score": economic_score,
                **policy_metrics,
            }
            policies.append(policy)
        selected = max(policies, key=lambda item: item["economic_score"])
        return calibration_model, pred, selected, policies

    winner_calibration_model, calibration_predictions, validation_strategy, threshold_policies = calibrate_policy(
        winner_name
    )
    if not math.isfinite(validation_strategy["economic_score"]):
        raise RuntimeError(f"No validation policy had enough activity for horizon {horizon}")
    threshold = float(validation_strategy["threshold"])

    ridge_calibration_model, ridge_calibration_predictions, ridge_policy, ridge_threshold_policies = calibrate_policy(
        ridge_name
    )
    ridge_threshold = float(ridge_policy["threshold"])

    winner = candidate_models()[winner_name]
    winner.fit(pretest[features], pretest[target])
    test_predictions = winner.predict(test[features])
    test_metrics = prediction_metrics(test[target].to_numpy(), test_predictions)
    strategy_metrics, trades = non_overlapping_strategy(
        test, test[target].to_numpy(), test_predictions, horizon, threshold
    )
    constant_prediction = np.full(len(test), float(pretest[target].mean()))
    baseline_prediction = prediction_metrics(test[target].to_numpy(), constant_prediction)
    baseline_strategy, _ = non_overlapping_strategy(
        test, test[target].to_numpy(), np.ones(len(test)), horizon, 0.0
    )
    strategy_metrics["net_points_vs_no_model_baseline"] = (
        strategy_metrics["net_points_after_2pt_slippage"]
        - baseline_strategy["net_points_after_2pt_slippage"]
    )

    artifact = {
        "model": winner,
        "model_name": winner_name,
        "horizon_minutes": horizon,
        "features": features,
        "training_end_exclusive": str(TEST_START.date()),
        "entry_score_threshold_net_points": threshold,
        "score_threshold_source": "economically selected on out-of-sample July-December 2023 validation predictions",
        "target": "next-quote entry premium minus same-expiry same-signal-ATM-strike premium after an exact holding horizon minus 2 points round-trip cost",
        "strike_rule": "K_t is ATM at signal time and remains fixed through delayed entry and exit; never re-ATM",
        "availability_rule": "all model features come from the signal quote; the assumed fill is the next one-minute quote and is excluded from features",
        "entry_rule": (
            "late session only: entry time at or after 13:30, when a 120-minute position cannot finish by the normal 15:30 close"
            if horizon == LATE_SESSION_HORIZON
            else "all sessions with an exact same-strike t+h observation"
        ),
        "round_trip_cost_points_in_target": SLIPPAGE_POINTS,
    }
    artifact_path = output / f"fixed_atm_{horizon}m.joblib"
    joblib.dump(artifact, artifact_path)
    ridge_report = None
    ridge_artifact_name = None
    if horizon == LATE_SESSION_HORIZON:
        if ridge_name == winner_name:
            fitted_ridge = winner
            ridge_test_predictions = test_predictions
        else:
            fitted_ridge = candidate_models()[ridge_name]
            fitted_ridge.fit(pretest[features], pretest[target])
            ridge_test_predictions = fitted_ridge.predict(test[features])
        ridge_strategy, _ = non_overlapping_strategy(
            test, test[target].to_numpy(), ridge_test_predictions, horizon, ridge_threshold
        )
        ridge_report = {
            "model": ridge_name,
            "test_prediction": prediction_metrics(
                test[target].to_numpy(), ridge_test_predictions
            ),
            "test_strategy": ridge_strategy,
        }
        ridge_artifact_name = "fixed_atm_90m_best_ridge.joblib"
        joblib.dump(
            {
                **artifact,
                "model": fitted_ridge,
                "model_name": ridge_name,
                "entry_score_threshold_net_points": ridge_threshold,
            },
            output / ridge_artifact_name,
        )
    trades.to_csv(output / f"test_trades_{horizon}m.csv", index=False)
    report = {
        "horizon_minutes": horizon,
        "selected_model": winner_name,
        "artifact": artifact_path.name,
        "best_ridge_artifact": ridge_artifact_name,
        "available_exact_same_strike_rows": int(exact.sum()),
        "exact_delayed_entry_rows": int(entry_exact.sum()),
        "schedule_eligible_rows": int(schedule_eligible.sum()),
        "exact_exit_coverage_of_schedule_eligible": float(
            exact.sum() / schedule_eligible.sum()
        ),
        "eligible_rows_after_session_rule": int(eligibility.sum()),
        "usable_rows": int(len(usable)),
        "train_core_rows": int(len(train_core)),
        "model_selection_rows": int(len(model_validation)),
        "policy_calibration_rows": int(len(validation)),
        "validation_rows": int(len(validation)),
        "pre_2024_refit_rows": int(len(pretest)),
        "test_rows": int(len(test)),
        "same_strike_rate_in_labeled_rows": float(
            data.loc[exact, exit_strike].eq(data.loc[exact, "strike"]).mean()
        ),
        "exact_horizon_rate_in_labeled_rows": float(
            ((pd.to_datetime(data.loc[exact, exit_dt]) - data.loc[exact, "datetime"]).dt.total_seconds() == horizon * 60).mean()
        ),
        "signal_precedes_entry_rate_in_labeled_rows": float(
            pd.to_datetime(data.loc[exact, "signal_quote_available_at"])
            .lt(pd.to_datetime(data.loc[exact, "entry_quote_available_at"]))
            .mean()
        ),
        "validation_candidates": validation_results,
        "threshold_policies": threshold_policies,
        "ridge_threshold_policies": ridge_threshold_policies,
        "validation_strategy_for_horizon_selection": validation_strategy,
        "best_ridge_test": ridge_report,
        "test_prediction": test_metrics,
        "test_constant_mean_baseline": baseline_prediction,
        "test_strategy": strategy_metrics,
        "test_no_model_strategy": baseline_strategy,
    }
    return report, trades


def write_report(output: Path, source: Path, results: list[dict], frame: pd.DataFrame, features: list[str]) -> None:
    best = max(
        results,
        key=lambda r: r["validation_strategy_for_horizon_selection"]["net_points_after_2pt_slippage"],
    )
    deployable = (
        best["validation_strategy_for_horizon_selection"]["net_points_after_2pt_slippage"] > 0
        and best["validation_strategy_for_horizon_selection"]["profit_factor"] > 1.0
    )
    final_confirmed = (
        deployable
        and best["test_strategy"]["net_points_after_2pt_slippage"] > 0
        and best["test_strategy"]["profit_factor"] > 1.0
    )
    late = next(r for r in results if r["horizon_minutes"] == LATE_SESSION_HORIZON)
    late_candidates = sorted(late["validation_candidates"], key=lambda row: row["mae"])
    best_late_ridge = next(
        row for row in late_candidates if row["model"].startswith("ridge_")
    )
    ridge_test = late["best_ridge_test"]
    lines = [
        "# Leakage-Safe Fixed-Signal-ATM Multi-Horizon Model Results",
        "",
        "## Decision",
        "",
        f"The horizon selected before opening the final test set is **{best['horizon_minutes']} minutes**, using **{best['selected_model']}**. Model family was selected on January-June 2023; threshold and horizon were selected on July-December 2023.",
        "",
        f"On the untouched January 2024 onward test, the preselected {best['horizon_minutes']}-minute model produced **{best['test_strategy']['net_points_after_2pt_slippage']:.1f} net points**. Test results for the other horizons are reported as diagnostics and were not used to choose the winner.",
        "",
        ("The preselected model passed the validation profitability guard." if deployable else "No model passed the pre-2024 validation profitability guard, so the correct deployment decision is **no model**."),
        "",
        ("The preselected model also passed final confirmation." if final_confirmed else "The preselected model failed final confirmation, so the deployment decision remains **no model**. Positive results from non-preselected horizons are diagnostic only."),
        "",
        f"For the requested late-session 90-minute case, the validation winner is **{late['selected_model']}**. The best Ridge specification is **{best_late_ridge['model']}**.",
        "",
        "## Leakage and contract rules",
        "",
        "- The strike is ATM at signal time t and remains fixed through the delayed fill and exit; it is never re-ATM'd.",
        "- Signal fields are timestamped by their actual tick availability. Execution uses the next one-minute quote, never the quote used to compute features.",
        "- Every accepted label has an exact quote-availability difference equal to its holding horizon.",
        "- The 90-minute model accepts entries at or after 13:30 only, when a new 120-minute position cannot finish by the normal 15:30 close.",
        "- The fitted target is cost-adjusted decay: fixed-strike straddle decay minus two points of modeled round-trip cost.",
        "- Model-family and Ridge-alpha selection use January-June 2023 only.",
        "- Threshold and horizon selection use July-December 2023 only.",
        "- Final models are refit on all usable rows before 1 January 2024.",
        "- Rows from 1 January 2024 onward are used only for final evaluation.",
        "- Exit labels come from the full option chain, so the original entry strike is retained even when it leaves the five-strike snapshot.",
        "- Statutory charges are not included.",
        "",
        "## Data summary",
        "",
        f"- Source: `{source}`",
        f"- Eligible entry regime: 21 to below 35 calendar days to the nearest qualifying monthly expiry selected at each timestamp",
        f"- Candidate entry rows: {len(frame):,}",
        f"- Entry-time features: {len(features)}",
        f"- Test range: {frame.loc[frame['datetime'] >= TEST_START, 'datetime'].min()} through {frame['datetime'].max()}",
        "",
        "## Horizon comparison",
        "",
        "| Horizon | Exact-exit coverage | Validation net | Selected family | Test rows | MAE | R2 | Corr | Trades | Win rate | Model net points | No-model net points | Net lift | Profit factor | Max drawdown |",
        "|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        p = result["test_prediction"]
        s = result["test_strategy"]
        b = result["test_no_model_strategy"]
        v = result["validation_strategy_for_horizon_selection"]
        lines.append(
            f"| {result['horizon_minutes']}m | {result['exact_exit_coverage_of_schedule_eligible']:.2%} | {v['net_points_after_2pt_slippage']:.1f} | {result['selected_model']} | {p['rows']:,} | {p['mae']:.3f} | {p['r2']:.3f} | {p['correlation']:.3f} | "
            f"{s['trades']:,} | {s['win_rate_after_slippage']:.2%} | {s['net_points_after_2pt_slippage']:.1f} | "
            f"{b['net_points_after_2pt_slippage']:.1f} | {s['net_points_vs_no_model_baseline']:.1f} | "
            f"{s['profit_factor']:.2f} | {s['maximum_drawdown_points']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 90-minute model comparison on January-June 2023 selection data",
            "",
            "| Rank | Model | Rows | MAE | RMSE | R2 | Correlation | Direction accuracy |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(late_candidates, start=1):
        lines.append(
            f"| {rank} | {row['model']} | {row['rows']:,} | {row['mae']:.3f} | {row['rmse']:.3f} | "
            f"{row['r2']:.3f} | {row['correlation']:.3f} | {row['positive_direction_accuracy']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Dedicated 90-minute Ridge result",
            "",
            f"The saved **{ridge_test['model']}** model recorded test MAE **{ridge_test['test_prediction']['mae']:.3f}**, "
            f"R2 **{ridge_test['test_prediction']['r2']:.3f}**, and **{ridge_test['test_strategy']['net_points_after_2pt_slippage']:.1f}** "
            f"net points across **{ridge_test['test_strategy']['trades']}** non-overlapping trades. It remains available as the requested Ridge implementation; **{late['selected_model']}** is the frozen model-selection winner.",
            "",
            "## Interpretation",
            "",
            "The model family was frozen using January-June 2023, and the threshold/horizon using July-December 2023. January 2024 onward remains the final test period.",
            "",
            "A negative test R2 means the fitted model does not beat a constant-mean prediction in squared-error terms. The no-model comparison enters every eligible opportunity subject only to the same one-position-at-a-time rule. Positive strategy P&L alone is therefore not evidence that the machine-learning filter adds value.",
            "",
            "Prediction MAE is expressed in cost-adjusted straddle points. Trading rows are chronological and permit only one open position at a time within each day. Each entry threshold is chosen from a fixed positive net-edge grid using only July-December 2023 validation results, subject to minimum activity.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(os.environ.get("VOLTR_43W_ENTRY_SOURCE", Path(__file__).resolve().parents[3] / "pipelines" / "weeks43" / "data" / "icharts_option_banknifty_monthly_atm_5strikes.csv")),
    )
    parser.add_argument(
        "--full-chain",
        type=Path,
        default=Path(os.environ.get("VOLTR_43W_FULL_CHAIN_SOURCE", Path(__file__).resolve().parents[3] / "pipelines" / "weeks43" / "data" / "icharts_option_banknifty.csv")),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/fixed_atm_horizons_leakage_free")
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    prepared_cache = args.output / "prepared_fixed_atm_labels.parquet"
    if prepared_cache.exists():
        frame = duckdb.connect().execute(
            "SELECT * FROM read_parquet(?)", [str(prepared_cache)]
        ).fetchdf()
    else:
        frame = load_fixed_strike_rows(args.source, args.full_chain)
        cache_connection = duckdb.connect()
        cache_connection.register("prepared_frame", frame)
        cache_connection.execute(
            "COPY prepared_frame TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",
            [str(prepared_cache)],
        )
        cache_connection.close()
    required_prepared_columns = {
        "signal_datetime", "signal_quote_available_at", "datetime",
        "entry_quote_datetime", "entry_quote_available_at", "entry_straddle",
        "entry_strike", "signal_straddle",
    }
    missing_prepared = sorted(required_prepared_columns.difference(frame.columns))
    if missing_prepared:
        raise AssertionError(
            f"Prepared cache predates the leakage-safe execution schema: {missing_prepared}"
        )
    matched_entries = frame["entry_quote_available_at"].notna()
    if not pd.to_datetime(frame.loc[matched_entries, "signal_quote_available_at"]).lt(
        pd.to_datetime(frame.loc[matched_entries, "entry_quote_available_at"])
    ).all():
        raise AssertionError("Prepared cache contains an entry quote available before its signal")
    frame, features = add_entry_features(frame)
    forbidden = [name for name in features if name.startswith(("exit_", "target_", "label_"))]
    forbidden.extend(
        name for name in features
        if name in {"entry_straddle", "entry_call_price", "entry_put_price", "entry_quote_available_at"}
    )
    if forbidden:
        raise AssertionError(f"Future-dependent columns entered the feature list: {forbidden}")
    results = []
    for horizon in HORIZONS:
        result, _ = train_horizon(frame, features, horizon, args.output)
        results.append(result)
        print(json.dumps({"horizon": horizon, "selected": result["selected_model"], "test": result["test_strategy"]}, indent=2))
    manifest = {
        "source": str(args.source),
        "full_chain_source": str(args.full_chain),
        "test_start": str(TEST_START.date()),
        "validation_start": str(VALIDATION_START.date()),
        "model_selection_start": str(MODEL_SELECTION_START.date()),
        "horizons": list(HORIZONS),
        "features": features,
        "results": results,
        "leakage_audit": {
            "future_columns_in_features": forbidden,
            "signal_availability": "Date+tick_time; always strictly before the delayed execution quote",
            "execution": "next one-minute full-chain quote after the signal snapshot",
            "entry_price_in_features": False,
            "expiry_selection": "nearest qualifying monthly expiry at each entry timestamp",
            "exit_source": "full option chain joined on exact timestamp, expiry, and fixed entry strike",
            "missing_lags": "retained and handled by train-fitted imputers",
            "model_selection_source": "January-June 2023 prediction MAE",
            "threshold_source": "fixed positive net-edge grid selected on July-December 2023 policy-validation predictions",
            "horizon_selection_source": "July-December 2023 validation strategy results",
            "final_test_start": str(TEST_START.date()),
            "final_test_used_for_selection": False,
            "split_boundary_labels_purged": True,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_report(args.output, args.source, results, frame, features)


if __name__ == "__main__":
    main()
