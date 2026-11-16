from __future__ import annotations

import json
from pathlib import Path

import duckdb
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from .atm_iv_change_prediction import HORIZONS, ROLLING_TRAIN_DAYS, score, session_diff, use_previous_session_at_open
    from .simple_iv_change_models import FastQuantileRegressor
except ImportError:  # Allow direct execution from the research directory.
    from atm_iv_change_prediction import HORIZONS, ROLLING_TRAIN_DAYS, score, session_diff, use_previous_session_at_open
    from simple_iv_change_models import FastQuantileRegressor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "atm_iv_change_models" / "atm_iv_change_feature_panel.parquet"
OUT = ROOT / "outputs" / "log_iv_change_models"
WINDOWS = (5, 15, 60)
QUANTILES = (0.1, 0.5, 0.9)
EPS = 1e-12
SEED = 20261008


class TrainingWinsorizer(BaseEstimator, TransformerMixin):
    """Clip every feature to limits learned only from the training window."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, x, y=None):
        values = np.asarray(x, dtype=float)
        self.lower_ = np.nanquantile(values, self.lower, axis=0)
        self.upper_ = np.nanquantile(values, self.upper, axis=0)
        return self

    def transform(self, x):
        return np.clip(np.asarray(x, dtype=float), self.lower_, self.upper_)


LEVEL_FEATURES = [
    "log_iv_atm", "iv_atm_percentile", "iv_atm_zscore", "log_iv_deviation",
    "log_skew_rr", "log_butterfly", "log_put_skew", "log_call_skew",
    "forward_log_basis", "log_total_call_oi", "log_total_put_oi", "oi_ratio",
    "oi_imbalance", "days_to_expiry", "sqrt_dte", "time_fraction", "time_sin",
    "time_cos", "is_0915", "overnight_return",
]
WINDOW_FEATURES = [
    f"{name}_{w}"
    for w in WINDOWS
    for name in (
        "dlog_iv", "dlog_skew", "dlog_butterfly", "oi_total_change",
        "spot_return", "spot_abs_return", "log_spot_rv", "rv_iv_gap_log", "leverage_log",
    )
]
STRIKE_FEATURES = [f"dlog_iv_{p}_5" for p in ("m2", "m1", "p1", "p2")]
FEATURES = LEVEL_FEATURES + WINDOW_FEATURES + STRIKE_FEATURES + [
    "dlog_rv_5", "vol_of_vol_log_30", "vol_of_vol_log_60", "is_open_30", "is_close_30",
]
BASELINE_FEATURES = ["log_iv_atm", "dlog_iv_5", "spot_return_5", "time_fraction", "days_to_expiry"]


def load_and_engineer() -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute("SELECT * FROM read_parquet(?) ORDER BY datetime", [str(SOURCE)]).fetchdf()
    con.close()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["date"] = pd.to_datetime(frame["date"])

    for p in ("m2", "m1", "atm", "p1", "p2"):
        frame[f"log_iv_{p}"] = np.log(frame[f"iv_mid_{p}"].where(frame[f"iv_mid_{p}"].gt(0)))
    frame["log_iv_atm"] = frame["log_iv_atm"].where(np.isfinite(frame["log_iv_atm"]))
    frame["log_skew_rr"] = frame["log_iv_p2"] - frame["log_iv_m2"]
    frame["log_butterfly"] = frame["log_iv_p2"] + frame["log_iv_m2"] - 2 * frame["log_iv_atm"]
    frame["log_put_skew"] = frame["log_iv_m2"] - frame["log_iv_atm"]
    frame["log_call_skew"] = frame["log_iv_p2"] - frame["log_iv_atm"]

    roll = frame["log_iv_atm"].rolling(60 * 375, min_periods=1000)
    frame["log_iv_deviation"] = frame["log_iv_atm"] - roll.mean()
    frame["iv_atm_percentile"] = roll.rank(pct=True)

    for w in WINDOWS:
        frame[f"dlog_iv_{w}"] = session_diff(frame, "log_iv_atm", w)
        frame[f"dlog_skew_{w}"] = session_diff(frame, "log_skew_rr", w)
        frame[f"dlog_butterfly_{w}"] = session_diff(frame, "log_butterfly", w)
        for column in (f"dlog_iv_{w}", f"dlog_skew_{w}", f"dlog_butterfly_{w}"):
            use_previous_session_at_open(frame, column)
        frame[f"leverage_log_{w}"] = (
            frame[f"dlog_iv_{w}"] / frame[f"spot_return_{w}"].replace(0, np.nan)
        ).clip(-100, 100)
        frame[f"rv_iv_gap_log_{w}"] = frame[f"log_spot_rv_{w}"] - frame["log_iv_atm"]

    for p in ("m2", "m1", "p1", "p2"):
        column = f"dlog_iv_{p}_5"
        frame[column] = session_diff(frame, f"log_iv_{p}", 5)
        use_previous_session_at_open(frame, column)

    frame["dlog_rv_5"] = session_diff(frame, "log_spot_rv_5", 5)
    use_previous_session_at_open(frame, "dlog_rv_5")
    dlog_1 = frame.groupby("date", sort=False)["log_iv_atm"].diff()
    for w in (30, 60):
        column = f"vol_of_vol_log_{w}"
        frame[column] = dlog_1.groupby(frame["date"], sort=False).rolling(w, min_periods=10).std().reset_index(level=0, drop=True)
        use_previous_session_at_open(frame, column)

    minute = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute
    frame["is_open_30"] = minute.lt(585).astype("int8")
    frame["is_close_30"] = minute.ge(900).astype("int8")
    for h in HORIZONS:
        future = frame[f"future_iv_atm_{h}"]
        frame[f"target_dlog_iv_{h}"] = np.log(future.where(future.gt(0))) - frame["log_iv_atm"]
        frame[f"realized_d_iv_{h}"] = future - frame["iv_atm"]
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame


def quantile_pipeline(quantile: float, alpha: float):
    return make_pipeline(
        SimpleImputer(strategy="median"), TrainingWinsorizer(), StandardScaler(),
        FastQuantileRegressor(quantile=quantile, alpha=alpha, max_iter=500),
    )


def baseline_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"), TrainingWinsorizer(), StandardScaler(),
        FastQuantileRegressor(quantile=0.5, alpha=1e-4, max_iter=500),
    )


def make_lgb(params: dict):
    return lgb.LGBMRegressor(
        objective="huber", random_state=SEED, n_jobs=8, verbosity=-1,
        learning_rate=0.035, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, **params,
    )


def pinball(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    residual = y - pred
    return float(np.mean(np.maximum(q * residual, (q - 1) * residual)))


def metrics(y: np.ndarray, pred: np.ndarray, q: float = 0.5) -> dict:
    return {**score(y, pred), "pinball_loss": pinball(y, pred, q)}


def valid_for(frame: pd.DataFrame, h: int) -> pd.DataFrame:
    target = f"target_dlog_iv_{h}"
    mask = frame["available_strike_count"].ge(5) & np.isfinite(frame[target]) & np.isfinite(frame["log_iv_atm"])
    return frame.loc[mask].copy()


def bootstrap_mae_difference(part: pd.DataFrame, target: str, model: str, draws: int = 1000) -> dict:
    daily = []
    for _, day in part.groupby("date", sort=False):
        y = day[target].to_numpy()
        daily.append((np.abs(y - day[model].to_numpy()).mean(), np.abs(y).mean()))
    values = np.asarray(daily)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    diffs = (values[samples, 0] - values[samples, 1]).mean(axis=1)
    return {
        "mean_mae_difference_vs_zero": float((values[:, 0] - values[:, 1]).mean()),
        "ci_2_5": float(np.quantile(diffs, 0.025)), "ci_97_5": float(np.quantile(diffs, 0.975)),
        "probability_beats_zero": float(np.mean(diffs < 0)),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = load_and_engineer()
    con = duckdb.connect(); con.register("panel", frame)
    con.execute("COPY panel TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(OUT / "log_iv_feature_panel.parquet")]); con.close()

    lgb_candidates = [
        {"n_estimators": 250, "num_leaves": 15, "min_child_samples": 150},
        {"n_estimators": 350, "num_leaves": 31, "min_child_samples": 150},
        {"n_estimators": 300, "num_leaves": 15, "min_child_samples": 350},
    ]
    alpha_candidates = (1e-5, 1e-4, 1e-3)
    tuning_rows, manifest_rows, prediction_frames = [], [], []

    for h in HORIZONS:
        target = f"target_dlog_iv_{h}"
        data = valid_for(frame, h)
        dev, val = data[data["split"].eq("development")], data[data["split"].eq("validation")]
        best_alpha = {}
        for q in QUANTILES:
            best_loss = np.inf
            for alpha in alpha_candidates:
                fitted = quantile_pipeline(q, alpha).fit(dev[FEATURES], dev[target])
                pred = fitted.predict(val[FEATURES])
                loss = pinball(val[target].to_numpy(), pred, q)
                tuning_rows.append({"horizon_min": h, "model": f"quantile_{q:g}", "alpha": alpha, "pinball_loss": loss})
                if loss < best_loss:
                    best_loss, best_alpha[q] = loss, alpha
        best_lgb, best_lgb_mae = None, np.inf
        for params in lgb_candidates:
            fitted = make_lgb(params).fit(dev[FEATURES], dev[target])
            pred = fitted.predict(val[FEATURES])
            mae = np.mean(np.abs(val[target].to_numpy() - pred))
            tuning_rows.append({"horizon_min": h, "model": "lightgbm", "alpha": np.nan, "pinball_loss": mae / 2, "parameters": json.dumps(params)})
            if mae < best_lgb_mae:
                best_lgb_mae, best_lgb = mae, params

        evaluation = data[data["split"].eq("evaluation")].copy()
        evaluation["refit_period"] = evaluation["datetime"].dt.to_period("Q")
        blocks = []
        for period in evaluation["refit_period"].drop_duplicates().sort_values():
            test = evaluation[evaluation["refit_period"].eq(period)].copy()
            prior_days = data.loc[data["date"].lt(test["date"].min()), "date"].drop_duplicates().sort_values()
            train_days = prior_days.iloc[-ROLLING_TRAIN_DAYS:]
            train = data[data["date"].isin(train_days)]
            if len(train) < 1000 or test.empty:
                continue
            pred = test[["datetime", "date", "iv_atm", "is_0915", target, f"realized_d_iv_{h}"]].copy()
            pred["horizon_min"] = h
            pred["zero"] = 0.0
            pred["random_walk_5"] = test["dlog_iv_5"].fillna(0.0).to_numpy()
            base = baseline_pipeline().fit(train[BASELINE_FEATURES], train[target])
            pred["mean_reversion"] = base.predict(test[BASELINE_FEATURES])
            for q in QUANTILES:
                fitted = quantile_pipeline(q, best_alpha[q]).fit(train[FEATURES], train[target])
                pred[f"q{int(q * 100):02d}"] = fitted.predict(test[FEATURES])
            boost = make_lgb(best_lgb).fit(train[FEATURES], train[target])
            pred["lightgbm"] = boost.predict(test[FEATURES])
            pred["uncertainty_width"] = pred["q90"] - pred["q10"]
            pred["predicted_d_iv_q50"] = pred["iv_atm"] * np.expm1(pred["q50"])
            pred["predicted_d_iv_lightgbm"] = pred["iv_atm"] * np.expm1(pred["lightgbm"])
            blocks.append(pred)
        prediction_frames.append(pd.concat(blocks, ignore_index=True))

        latest_days = data["date"].drop_duplicates().sort_values().iloc[-ROLLING_TRAIN_DAYS:]
        final_train = data[data["date"].isin(latest_days)]
        for q in QUANTILES:
            fitted = quantile_pipeline(q, best_alpha[q]).fit(final_train[FEATURES], final_train[target])
            path = OUT / f"log_iv_{h}m_q{int(q * 100):02d}.joblib"
            joblib.dump({"model": fitted, "features": FEATURES, "horizon_min": h, "quantile": q}, path)
            manifest_rows.append({"horizon_min": h, "model": f"quantile_{q:g}", "model_file": path.name, "alpha": best_alpha[q], "train_start": final_train["date"].min(), "train_end": final_train["date"].max(), "training_rows": len(final_train)})
        fitted_lgb = make_lgb(best_lgb).fit(final_train[FEATURES], final_train[target])
        path = OUT / f"log_iv_{h}m_lightgbm.joblib"
        joblib.dump({"model": fitted_lgb, "features": FEATURES, "horizon_min": h}, path)
        manifest_rows.append({"horizon_min": h, "model": "lightgbm", "model_file": path.name, "alpha": np.nan, "train_start": final_train["date"].min(), "train_end": final_train["date"].max(), "training_rows": len(final_train)})

    predictions = pd.concat(prediction_frames, ignore_index=True)
    metric_rows, bootstrap_rows = [], []
    for subset, selected in (("intraday", predictions), ("0915", predictions[predictions["is_0915"].eq(1)])):
        for h in HORIZONS:
            part = selected[selected["horizon_min"].eq(h)]
            target = f"target_dlog_iv_{h}"
            y = part[target].to_numpy()
            for model in ("zero", "random_walk_5", "mean_reversion", "q50", "lightgbm"):
                metric_rows.append({"subset": subset, "horizon_min": h, "model": model, "n": len(part), **metrics(y, part[model].to_numpy())})
            for model in ("q50", "lightgbm"):
                bootstrap_rows.append({"subset": subset, "horizon_min": h, "model": model, **bootstrap_mae_difference(part, target, model)})

    pd.DataFrame(metric_rows).to_csv(OUT / "model_metrics.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(OUT / "day_block_bootstrap.csv", index=False)
    pd.DataFrame(tuning_rows).to_csv(OUT / "validation_tuning.csv", index=False)
    pd.DataFrame(manifest_rows).to_csv(OUT / "model_manifest.csv", index=False)
    predictions[predictions["is_0915"].eq(1)].to_csv(OUT / "opening_0915_predictions.csv", index=False)
    con = duckdb.connect(); con.register("predictions", predictions)
    con.execute("COPY predictions TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(OUT / "walk_forward_predictions.parquet")]); con.close()
    print(pd.DataFrame(metric_rows).to_string(index=False))
    print(pd.DataFrame(bootstrap_rows).to_string(index=False))


if __name__ == "__main__":
    main()
