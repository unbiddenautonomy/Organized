"""Train calibrated BANKNIFTY short-straddle win classifiers.

The entry contract is the monthly ATM call and put at time t.  Every target
uses that exact expiry and strike at t+h; it never rolls to the new ATM strike.
All observations from 2024-01-01 onward are held out from fitting and
calibration and are used only for the final comparison.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_LOCATION_FILE = WORKSPACE / "DatasetLocation.txt"
OUTPUT = HERE / "output" / "win_classifier"
HORIZONS = (30, 60, 120, 180)
TEST_START = pd.Timestamp("2024-01-01")
CALIBRATION_START = pd.Timestamp("2023-01-01")
SELECTION_START = pd.Timestamp("2022-01-01")
SELECTION_CALIBRATION_START = pd.Timestamp("2021-01-01")
FEATURES = (
    "future_log_return_5m",
    "future_log_return_15m",
    "future_log_return_30m",
    "future_log_return_60m",
    "basis",
    "pcr_log_change_5m",
    "pcr_log_change_15m",
    "pcr_log_change_30m",
    "straddle_log_change_5m",
    "straddle_log_change_15m",
    "straddle_log_change_30m",
    "straddle_premium",
    "premium_balance",
    "moneyness",
    "tod_sin",
    "tod_cos",
    "days_to_expiry_clipped",
    "pcr_level",
    "log_total_oi",
    "total_oi_log_change_5m",
    "total_oi_log_change_15m",
    "total_oi_log_change_30m",
    "future_realized_vol_15m",
    "future_realized_vol_30m",
    "future_realized_vol_60m",
    "future_momentum_acceleration",
    "straddle_momentum_acceleration",
    "straddle_to_future",
    "iv_mean",
    "iv_skew",
    "iv_dispersion",
    "theta_yield",
    "gamma_scaled",
    "vega_scaled",
    "abs_premium_balance",
    "abs_moneyness",
    "sqrt_dte",
    "inverse_sqrt_dte",
    "minutes_to_1500",
    "weekday_sin",
    "weekday_cos",
)
RAW_COLUMNS = (
    "datetime",
    "dte_flag",
    "strike",
    "expiry",
    "days_to_expiry",
    "future_open",
    "spot_price",
    "call_price",
    "put_price",
    "call_oi",
    "put_oi",
    "atm_strike",
    "Future_oi",
    "iv_call",
    "iv_put",
    "delta_call",
    "delta_put",
    "vega_call",
    "vega_put",
    "gamma_call",
    "gamma_put",
    "theta_call",
    "theta_put",
)


def dataset_path(location_file: Path) -> Path:
    path = Path(location_file.read_text(encoding="utf-8-sig").strip().strip('"'))
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def build_raw_cache(source: Path, cache: Path, chunk_size: int) -> pd.DataFrame:
    """Read the M1 rows without making any full-day expiry decision."""
    parts: list[pd.DataFrame] = []
    for number, chunk in enumerate(
        pd.read_csv(source, usecols=RAW_COLUMNS, chunksize=chunk_size), start=1
    ):
        chunk = chunk.loc[chunk["dte_flag"].eq("M1")].copy()
        if chunk.empty:
            continue
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="coerce").dt.floor("min")
        chunk["expiry"] = pd.to_datetime(chunk["expiry"], errors="coerce")
        for column in set(RAW_COLUMNS) - {"datetime", "expiry", "dte_flag"}:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
        chunk = chunk.dropna(subset=["datetime", "expiry", "strike"])
        parts.append(chunk)
        if number % 5 == 0:
            print(f"  read {number * chunk_size:,} raw rows", flush=True)
    if not parts:
        raise RuntimeError("No M1 monthly option rows found")
    raw = pd.concat(parts, ignore_index=True)
    raw["trade_date"] = raw["datetime"].dt.normalize()
    raw = raw.loc[raw["expiry"] >= raw["trade_date"]].copy()
    raw = raw.sort_values("datetime").drop_duplicates(
        ["datetime", "expiry", "strike"], keep="last"
    )
    cache.parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(cache)
    print(f"Cached {len(raw):,} M1 rows at {cache}", flush=True)
    return raw


def load_raw(source: Path, cache: Path, rebuild: bool, chunk_size: int) -> pd.DataFrame:
    if rebuild or not cache.exists():
        return build_raw_cache(source, cache, chunk_size)
    print(f"Using cached M1 rows: {cache}", flush=True)
    return pd.read_pickle(cache)


def build_atm_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a causal front-month ATM panel with a one-minute maximum fill."""
    atm = raw.loc[np.isclose(raw["strike"], raw["atm_strike"], equal_nan=False)].copy()
    # The front monthly expiry is the earliest unexpired expiry visible in the
    # current snapshot.  Unlike the former dominant-expiry rule, this does not
    # inspect observations from later in the trading day.
    front = atm.groupby("datetime")["expiry"].transform("min")
    atm = atm.loc[atm["expiry"].eq(front)].copy()
    atm = atm.sort_values("datetime").drop_duplicates("datetime", keep="last").set_index("datetime")
    days: list[pd.DataFrame] = []
    for day, group in atm.groupby(atm.index.normalize()):
        day = pd.Timestamp(day)
        grid = pd.date_range(
            pd.Timestamp(f"{day.date()} 09:15:00"),
            pd.Timestamp(f"{day.date()} 15:29:00"),
            freq="min",
        )
        day_frame = group.reindex(grid).ffill(limit=1)
        day_frame["is_observed"] = day_frame.index.isin(group.index)
        day_frame["trade_date"] = day
        days.append(day_frame)
    panel = pd.concat(days).sort_index()
    panel.index.name = "datetime"
    numeric = [
        "future_open", "spot_price", "call_price", "put_price", "call_oi",
        "put_oi", "strike", "atm_strike", "days_to_expiry", "Future_oi",
        "iv_call", "iv_put", "delta_call", "delta_put", "vega_call",
        "vega_put", "gamma_call", "gamma_put", "theta_call", "theta_put",
    ]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["straddle_premium"] = panel["call_price"] + panel["put_price"]
    panel["pcr"] = panel["put_oi"] / panel["call_oi"]
    panel.loc[(panel["pcr"] <= 0) | ~np.isfinite(panel["pcr"]), "pcr"] = np.nan
    panel["log_pcr"] = np.log(panel["pcr"])
    panel["log_straddle"] = np.log(panel["straddle_premium"].where(panel["straddle_premium"] > 0))
    panel["log_future"] = np.log(panel["future_open"].where(panel["future_open"] > 0))
    for minutes in (5, 15, 30, 60):
        panel[f"future_log_return_{minutes}m"] = panel["log_future"] - panel["log_future"].reindex(
            panel.index - pd.Timedelta(minutes, unit="min")
        ).to_numpy()
    for minutes in (5, 15, 30):
        panel[f"pcr_log_change_{minutes}m"] = panel["log_pcr"] - panel["log_pcr"].reindex(
            panel.index - pd.Timedelta(minutes, unit="min")
        ).to_numpy()
        panel[f"straddle_log_change_{minutes}m"] = panel["log_straddle"] - panel["log_straddle"].reindex(
            panel.index - pd.Timedelta(minutes, unit="min")
        ).to_numpy()
    panel["basis"] = panel["future_open"] / panel["spot_price"] - 1.0
    panel["premium_balance"] = (panel["call_price"] - panel["put_price"]) / panel["straddle_premium"]
    panel["moneyness"] = (panel["spot_price"] - panel["strike"]) / 100.0
    minute = panel.index.hour * 60 + panel.index.minute
    phase = 2.0 * np.pi * (minute - 9 * 60 - 15) / 375.0
    panel["tod_sin"] = np.sin(phase)
    panel["tod_cos"] = np.cos(phase)
    panel["days_to_expiry_clipped"] = panel["days_to_expiry"].clip(0, 45)
    panel["pcr_level"] = panel["pcr"]
    panel["log_total_oi"] = np.log(
        (panel["call_oi"] + panel["put_oi"]).where(
            panel["call_oi"] + panel["put_oi"] > 0
        )
    )
    for minutes in (5, 15, 30):
        panel[f"total_oi_log_change_{minutes}m"] = panel["log_total_oi"] - panel[
            "log_total_oi"
        ].reindex(panel.index - pd.Timedelta(minutes, unit="min")).to_numpy()
    one_minute_return = panel["log_future"] - panel["log_future"].reindex(
        panel.index - pd.Timedelta(1, unit="min")
    ).to_numpy()
    squared_return = one_minute_return.pow(2)
    for minutes in (15, 30, 60):
        panel[f"future_realized_vol_{minutes}m"] = np.sqrt(
            squared_return.rolling(
                f"{minutes}min", min_periods=max(5, minutes // 3)
            ).sum()
        )
    panel["future_momentum_acceleration"] = (
        panel["future_log_return_5m"] - panel["future_log_return_30m"] / 6.0
    )
    panel["straddle_momentum_acceleration"] = (
        panel["straddle_log_change_5m"] - panel["straddle_log_change_30m"] / 6.0
    )
    panel["straddle_to_future"] = panel["straddle_premium"] / panel["future_open"]
    panel["iv_mean"] = (panel["iv_call"] + panel["iv_put"]) / 2.0
    panel["iv_skew"] = panel["iv_put"] - panel["iv_call"]
    panel["iv_dispersion"] = (panel["iv_put"] - panel["iv_call"]).abs()
    panel["theta_yield"] = -(
        panel["theta_call"] + panel["theta_put"]
    ) / panel["straddle_premium"]
    panel["gamma_scaled"] = (
        panel["gamma_call"] + panel["gamma_put"]
    ) * panel["future_open"].pow(2) / panel["straddle_premium"]
    panel["vega_scaled"] = (
        panel["vega_call"] + panel["vega_put"]
    ) / panel["straddle_premium"]
    panel["abs_premium_balance"] = panel["premium_balance"].abs()
    panel["abs_moneyness"] = panel["moneyness"].abs()
    dte_safe = panel["days_to_expiry_clipped"].clip(lower=1.0 / 1440.0)
    panel["sqrt_dte"] = np.sqrt(dte_safe)
    panel["inverse_sqrt_dte"] = 1.0 / np.sqrt(dte_safe)
    panel["minutes_to_1500"] = np.maximum(15 * 60 - minute, 0)
    weekday_phase = 2.0 * np.pi * panel.index.dayofweek / 5.0
    panel["weekday_sin"] = np.sin(weekday_phase)
    panel["weekday_cos"] = np.cos(weekday_phase)
    panel["minute_of_day"] = minute
    return panel


def lot_size(dates: pd.DatetimeIndex) -> np.ndarray:
    values = np.full(len(dates), 25, dtype=float)
    values[dates < pd.Timestamp("2020-07-01")] = 20
    values[dates >= pd.Timestamp("2023-07-01")] = 15
    values[dates >= pd.Timestamp("2025-02-01")] = 30
    return values


def option_cost_rupees(entry: np.ndarray, exit_: np.ndarray, dates: pd.DatetimeIndex, lots: np.ndarray) -> np.ndarray:
    """Round-trip charges for selling two option legs, excluding slippage."""
    entry_turnover = entry * lots
    exit_turnover = exit_ * lots
    total = entry_turnover + exit_turnover
    stt_rate = np.where(dates < pd.Timestamp("2023-04-01"), 0.0005, np.where(dates < pd.Timestamp("2024-10-01"), 0.000625, 0.001))
    exchange_rate = np.where(dates < pd.Timestamp("2024-10-01"), 0.0005, 0.0003503)
    stt = entry_turnover * stt_rate
    exchange = total * exchange_rate
    sebi = total * 0.000001
    stamp = exit_turnover * 0.00003
    gst = 0.18 * (exchange + sebi)
    return stt + exchange + sebi + stamp + gst


def _linear_straddle_value(
    strikes: np.ndarray, premiums: np.ndarray, target_strike: float, future_price: float
) -> tuple[float, bool]:
    """Observed value when present, otherwise linear strike-surface valuation."""
    valid = np.isfinite(strikes) & np.isfinite(premiums)
    x = strikes[valid]
    y = premiums[valid]
    if not len(x):
        raise RuntimeError("Exit surface has no valid option premiums")
    order = np.argsort(x)
    x, y = x[order], y[order]
    exact = np.isclose(x, target_strike)
    if exact.any():
        return float(y[np.flatnonzero(exact)[-1]]), False
    if len(x) == 1:
        estimate = float(y[0])
    elif target_strike < x[0]:
        estimate = float(y[0] + (target_strike - x[0]) * (y[1] - y[0]) / (x[1] - x[0]))
    elif target_strike > x[-1]:
        estimate = float(y[-1] + (target_strike - x[-1]) * (y[-1] - y[-2]) / (x[-1] - x[-2]))
    else:
        estimate = float(np.interp(target_strike, x, y))
    # A European call-plus-put cannot be worth less than |F-K| at zero rates.
    return max(estimate, abs(float(future_price) - target_strike), 0.0), True


def horizon_dataset(panel: pd.DataFrame, raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Create labels without conditioning entry eligibility on a future quote."""
    # Use a fixed, live-known 15:00 latest scheduled exit.  This leaves a
    # 29-minute buffer before the stored session boundary for sparse snapshots.
    latest_entry = min(14 * 60 + 30, 15 * 60 - horizon)
    candidates = panel.loc[
        panel["is_observed"].fillna(False)
        & panel["minute_of_day"].between(10 * 60 + 15, latest_entry)
    ].copy()
    # Features are frozen one full minute before the executable entry quote.
    causal_features = panel[list(FEATURES)].reindex(
        candidates.index - pd.Timedelta(1, unit="min")
    )
    causal_features.index = candidates.index
    candidates.loc[:, list(FEATURES)] = causal_features.to_numpy()
    # Engineered-feature gaps are handled by imputers fitted on the relevant
    # training window.  Dropping them here would make coverage depend on vendor
    # feature availability and unnecessarily shrink later periods.
    candidates = candidates.dropna(
        subset=["expiry", "strike", "call_price", "put_price"]
    )
    candidates["scheduled_exit"] = candidates.index + pd.Timedelta(horizon, unit="min")

    surface_times = (
        raw[["datetime", "expiry"]].drop_duplicates().sort_values(["expiry", "datetime"])
    )
    exit_times = pd.Series(pd.NaT, index=candidates.index, dtype="datetime64[ns]")
    for expiry, positions in candidates.groupby("expiry").groups.items():
        available = surface_times.loc[surface_times["expiry"].eq(expiry), "datetime"].to_numpy()
        targets = candidates.loc[positions, "scheduled_exit"].to_numpy()
        right = np.searchsorted(available, targets, side="left")
        chosen = np.empty(len(targets), dtype="datetime64[ns]")
        for i, (target, pos) in enumerate(zip(targets, right, strict=True)):
            target_day = pd.Timestamp(target).normalize()
            if pos < len(available) and pd.Timestamp(available[pos]).normalize() == target_day:
                chosen[i] = available[pos]
            else:
                prior = min(pos - 1, len(available) - 1)
                if prior < 0 or pd.Timestamp(available[prior]).normalize() != target_day:
                    raise RuntimeError(f"No same-day exit surface for {target}")
                chosen[i] = available[prior]
        exit_times.loc[positions] = chosen
    candidates["exit_datetime"] = exit_times

    target_rows = candidates.reset_index().rename(columns={"datetime": "entry_datetime"})
    target_rows["candidate_id"] = np.arange(len(target_rows))
    surface = target_rows[["candidate_id", "exit_datetime", "expiry"]].merge(
        raw[["datetime", "expiry", "strike", "call_price", "put_price", "future_open"]],
        left_on=["exit_datetime", "expiry"],
        right_on=["datetime", "expiry"],
        how="left",
        validate="many_to_many",
    )
    values = np.empty(len(target_rows), dtype=float)
    synthetic = np.empty(len(target_rows), dtype=bool)
    entry_strikes = target_rows.set_index("candidate_id")["strike"]
    for candidate_id, group in surface.groupby("candidate_id", sort=False):
        premium = group["call_price"].to_numpy(float) + group["put_price"].to_numpy(float)
        future = group["future_open"].dropna()
        if future.empty:
            raise RuntimeError(f"Missing exit future price for candidate {candidate_id}")
        values[candidate_id], synthetic[candidate_id] = _linear_straddle_value(
            group["strike"].to_numpy(float), premium,
            float(entry_strikes.loc[candidate_id]), float(future.iloc[-1]),
        )
    if len(surface.groupby("candidate_id")) != len(target_rows):
        raise RuntimeError("At least one candidate lacks an exit surface")
    candidates["exit_straddle_premium"] = values
    candidates["synthetic_exit"] = synthetic
    candidates["exit_delay_minutes"] = (
        candidates["exit_datetime"] - candidates["scheduled_exit"]
    ).dt.total_seconds() / 60.0
    entry_premium = candidates["call_price"].to_numpy(float) + candidates["put_price"].to_numpy(float)
    exit_premium = candidates["exit_straddle_premium"].to_numpy(float)
    dates = pd.DatetimeIndex(candidates.index.normalize())
    lots = lot_size(dates)
    charges = option_cost_rupees(entry_premium, exit_premium, dates, lots)
    gross_points = entry_premium - exit_premium
    candidates["gross_points"] = gross_points
    candidates["cost_rupees"] = charges
    candidates["slippage_points"] = 2.0
    candidates["net_rupees"] = gross_points * lots - charges - 2.0 * lots
    candidates["win"] = (candidates["net_rupees"] > 0).astype(int)
    candidates["horizon_minutes"] = horizon
    return candidates


@dataclass
class Winsorizer:
    lower: pd.Series | None = None
    upper: pd.Series | None = None

    def fit(self, frame: pd.DataFrame) -> "Winsorizer":
        self.lower = frame.quantile(0.002)
        self.upper = frame.quantile(0.998)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.lower is None or self.upper is None:
            raise RuntimeError("Winsorizer is not fitted")
        return frame.clip(self.lower, self.upper, axis=1)


def model_candidates(seed: int) -> dict[str, object]:
    return {
        "logistic_l2": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.25, max_iter=3000, random_state=seed),
        ),
        "logistic_l1": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(
                C=0.05, penalty="l1", solver="liblinear", max_iter=3000,
                random_state=seed,
            ),
        ),
        "random_forest": make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=180, min_samples_leaf=30, max_features="sqrt", n_jobs=-1, random_state=seed),
        ),
        "extra_trees": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(n_estimators=220, min_samples_leaf=30, max_features="sqrt", n_jobs=-1, random_state=seed),
        ),
        "hist_gradient_boosting": make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(max_iter=180, learning_rate=0.06, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0, random_state=seed),
        ),
        "lightgbm": make_pipeline(
            SimpleImputer(strategy="median"),
            LGBMClassifier(
                n_estimators=300, learning_rate=0.025, num_leaves=15,
                max_depth=5, min_child_samples=80, subsample=0.85,
                colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=3.0,
                verbosity=-1, n_jobs=-1, random_state=seed,
            ),
        ),
        "xgboost": make_pipeline(
            SimpleImputer(strategy="median"),
            XGBClassifier(
                n_estimators=300, learning_rate=0.025, max_depth=4,
                min_child_weight=30, subsample=0.85, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=3.0, eval_metric="logloss",
                n_jobs=-1, random_state=seed,
            ),
        ),
    }


def calibrate_probabilities(raw_cal: np.ndarray, y_cal: np.ndarray, raw_test: np.ndarray, seed: int) -> tuple[np.ndarray, LogisticRegression]:
    eps = 1e-6
    cal_logit = np.log(np.clip(raw_cal, eps, 1 - eps) / np.clip(1 - raw_cal, eps, 1 - eps)).reshape(-1, 1)
    test_logit = np.log(np.clip(raw_test, eps, 1 - eps) / np.clip(1 - raw_test, eps, 1 - eps)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=1000, random_state=seed)
    calibrator.fit(cal_logit, y_cal)
    return calibrator.predict_proba(test_logit)[:, 1], calibrator


def apply_probability_calibrator(
    raw_probability: np.ndarray, calibrator: LogisticRegression
) -> np.ndarray:
    eps = 1e-6
    logit = np.log(
        np.clip(raw_probability, eps, 1 - eps)
        / np.clip(1 - raw_probability, eps, 1 - eps)
    ).reshape(-1, 1)
    return calibrator.predict_proba(logit)[:, 1]


def calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    total = len(y)
    return float(sum((bucket == i).sum() / total * abs(y[bucket == i].mean() - probability[bucket == i].mean()) for i in range(bins) if (bucket == i).any()))


def metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y, probability),
        "brier_score": brier_score_loss(y, probability),
        "log_loss": log_loss(y, probability),
        "expected_calibration_error": calibration_error(y, probability),
        "accuracy_at_0_5": accuracy_score(y, prediction),
        "precision_at_0_5": precision_score(y, prediction, zero_division=0),
        "recall_at_0_5": recall_score(y, prediction, zero_division=0),
        "f1_at_0_5": f1_score(y, prediction, zero_division=0),
    }


def train_horizon(
    frame: pd.DataFrame, horizon: int, output: Path, seed: int
) -> tuple[list[dict], dict]:
    selection_development = frame.loc[frame.index < SELECTION_CALIBRATION_START]
    selection_calibration = frame.loc[
        (frame.index >= SELECTION_CALIBRATION_START)
        & (frame.index < SELECTION_START)
    ]
    selection = frame.loc[
        (frame.index >= SELECTION_START) & (frame.index < CALIBRATION_START)
    ]
    development = frame.loc[frame.index < CALIBRATION_START]
    calibration = frame.loc[(frame.index >= CALIBRATION_START) & (frame.index < TEST_START)]
    test = frame.loc[frame.index >= TEST_START]
    if min(
        len(selection_development), len(selection_calibration), len(selection), len(development),
        len(calibration), len(test),
    ) == 0:
        raise RuntimeError(f"Empty split for {horizon}-minute horizon")

    # Choose the family exclusively on 2022, before calibration and test.
    selection_clipper = Winsorizer().fit(selection_development[list(FEATURES)])
    x_selection_development = selection_clipper.transform(
        selection_development[list(FEATURES)]
    )
    x_selection_calibration = selection_clipper.transform(
        selection_calibration[list(FEATURES)]
    )
    x_selection = selection_clipper.transform(selection[list(FEATURES)])
    y_selection_development = selection_development["win"].to_numpy()
    y_selection_calibration = selection_calibration["win"].to_numpy()
    y_selection = selection["win"].to_numpy()
    rows: list[dict] = []
    for name, template in model_candidates(seed).items():
        print(f"  selecting {name} on 2022", flush=True)
        selection_model = clone(template).fit(
            x_selection_development, y_selection_development
        )
        raw_selection_calibration = selection_model.predict_proba(
            x_selection_calibration
        )[:, 1]
        raw_selection = selection_model.predict_proba(x_selection)[:, 1]
        probability, _ = calibrate_probabilities(
            raw_selection_calibration, y_selection_calibration, raw_selection, seed
        )
        row = {
            "horizon_minutes": horizon,
            "model": name,
            "selection_development_rows": len(selection_development),
            "selection_calibration_rows": len(selection_calibration),
            "selection_rows": len(selection),
            "selection_win_rate": float(y_selection.mean()),
            **metrics(y_selection, probability),
        }
        rows.append(row)

    selected_name = min(rows, key=lambda item: (item["brier_score"], item["log_loss"]))["model"]
    clipper = Winsorizer().fit(development[list(FEATURES)])
    x_development = clipper.transform(development[list(FEATURES)])
    x_calibration = clipper.transform(calibration[list(FEATURES)])
    x_test = clipper.transform(test[list(FEATURES)])
    y_development = development["win"].to_numpy()
    y_calibration = calibration["win"].to_numpy()
    y_test = test["win"].to_numpy()
    selected_probability: np.ndarray | None = None
    selected_calibration_probability: np.ndarray | None = None
    for name, template in model_candidates(seed).items():
        print(f"  final fitting {name} through 2022; calibrating on 2023", flush=True)
        model = clone(template).fit(x_development, y_development)
        raw_calibration = model.predict_proba(x_calibration)[:, 1]
        raw_test = model.predict_proba(x_test)[:, 1]
        probability, calibrator = calibrate_probabilities(
            raw_calibration, y_calibration, raw_test, seed
        )
        joblib.dump(
            {"model": model, "calibrator": calibrator, "clipper": clipper, "features": FEATURES, "horizon_minutes": horizon, "selected_on_2022": name == selected_name, "development_end": "2022-12-31", "calibration_period": "2023", "test_start": "2024-01-01"},
            output / f"{horizon}m_{name}.joblib",
        )
        if name == selected_name:
            selected_probability = probability
            selected_calibration_probability = apply_probability_calibrator(
                raw_calibration, calibrator
            )
    if selected_probability is None or selected_calibration_probability is None:
        raise RuntimeError("Selected model was not fitted")
    predictions = pd.DataFrame(index=test.index)
    predictions["actual_win"] = y_test
    predictions["net_rupees"] = test["net_rupees"].to_numpy()
    predictions["probability"] = selected_probability
    predictions.to_csv(output / f"test_predictions_{horizon}m.csv.gz", compression="gzip")
    constant_brier = brier_score_loss(
        y_test, np.full(len(y_test), y_calibration.mean())
    )
    test_row = {
        "horizon_minutes": horizon,
        "selected_model": selected_name,
        "development_rows": len(development),
        "calibration_rows": len(calibration),
        "test_rows": len(test),
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "calibration_win_rate": float(y_calibration.mean()),
        "test_base_win_rate": float(y_test.mean()),
        **metrics(y_test, selected_probability),
        "constant_brier_score": constant_brier,
    }
    test_row["brier_skill_score"] = 1.0 - test_row["brier_score"] / constant_brier
    probability_gate = float(np.quantile(selected_calibration_probability, 0.80))
    admitted = selected_probability >= probability_gate
    test_row["probability_gate_2023_p80"] = probability_gate
    test_row["admitted_trade_rate_pct"] = 100.0 * admitted.mean()
    test_row["admitted_trades"] = int(admitted.sum())
    test_row["admitted_win_rate"] = (
        float(y_test[admitted].mean()) if admitted.any() else float("nan")
    )
    test_row["admitted_win_rate_lift"] = (
        test_row["admitted_win_rate"] - test_row["test_base_win_rate"]
    )
    return rows, test_row


def make_report(selection: pd.DataFrame, test: pd.DataFrame, metadata: dict) -> str:
    lines = [
        "# Calibrated win-probability classifier results",
        "",
        "All models are independent. No model prediction is an input to another model. Model families are selected on 2022 before the test period is opened.",
        "",
        "The final base model uses data through 31 December 2022, calendar 2023 is used only for sigmoid probability calibration, and 1 January 2024 onward is excluded from every code-level selection and fitting decision. Because earlier research runs already reported this period, it is a held-out confirmation diagnostic rather than a pristine new holdout.",
        "",
        "Entries use a front-month expiry visible at that timestamp and features frozen one minute before the entry quote. Candidate eligibility never depends on the future exit quote. Missing same-strike exit premiums are valued from the complete observed strike surface. The first snapshot at or after the scheduled exit is used; if a day has no later stored snapshot, its final prior surface is retained as a stale valuation rather than deleting the row.",
        "",
        "| Horizon | Pre-test selected model | Test rows | Base win rate | Brier skill | ROC AUC | Top-20% admitted trades | Admitted win rate | Win-rate lift |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in test.itertuples(index=False):
        lines.append(f"| {row.horizon_minutes} | {row.selected_model} | {row.test_rows} | {row.test_base_win_rate:.3f} | {row.brier_skill_score:.4f} | {row.roc_auc:.4f} | {row.admitted_trades} ({row.admitted_trade_rate_pct:.1f}%) | {row.admitted_win_rate:.3f} | {row.admitted_win_rate_lift:+.3f} |")
    lines.extend([
        "",
        "No winner is selected from this table; doing so would reuse the test period for model selection. The selected families are recorded in `model_selection_2022.csv`.",
        "",
        f"Test coverage ends {metadata['data_end']}.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--location-file", type=Path, default=DEFAULT_LOCATION_FILE)
    parser.add_argument("--source", type=Path, help="historical five-strike source CSV")
    parser.add_argument("--output", type=Path, help="isolated training output directory")
    parser.add_argument("--cache", type=Path, help="existing or new raw M1 cache")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output:
        OUTPUT = args.output.expanduser().resolve()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source = args.source.expanduser().resolve() if args.source else dataset_path(args.location_file)
    if not source.is_file():
        raise FileNotFoundError(source)
    cache = args.cache.expanduser().resolve() if args.cache else OUTPUT / "all_m1_rows_features_v2.pkl"
    raw = load_raw(source, cache, args.rebuild_cache, args.chunk_size)
    panel = build_atm_panel(raw)
    selection_rows: list[dict] = []
    test_rows: list[dict] = []
    dataset_counts: list[dict] = []
    for horizon in HORIZONS:
        print(f"Preparing {horizon}-minute same-strike target", flush=True)
        frame = horizon_dataset(panel, raw, horizon)
        dataset_counts.append(
            {
                "horizon_minutes": horizon,
                "rows": len(frame),
                "wins": int(frame["win"].sum()),
                "synthetic_exit_rows": int(frame["synthetic_exit"].sum()),
                "stale_prior_exit_rows": int((frame["exit_delay_minutes"] < 0).sum()),
                "median_exit_delay_minutes": float(frame["exit_delay_minutes"].median()),
                "p95_exit_delay_minutes": float(frame["exit_delay_minutes"].quantile(0.95)),
                "start": str(frame.index.min()),
                "end": str(frame.index.max()),
            }
        )
        print(f"Training {horizon}-minute models on {len(frame):,} rows", flush=True)
        selected, tested = train_horizon(frame, horizon, OUTPUT, args.seed)
        selection_rows.extend(selected)
        test_rows.append(tested)
    selection_results = pd.DataFrame(selection_rows).sort_values(
        ["horizon_minutes", "brier_score", "log_loss"]
    )
    test_results = pd.DataFrame(test_rows).sort_values("horizon_minutes")
    selection_results.to_csv(OUTPUT / "model_selection_2022.csv", index=False)
    test_results.to_csv(OUTPUT / "selected_model_test_results.csv", index=False)
    metadata = {
        "source": str(source),
        "source_bytes": source.stat().st_size,
        "data_start": str(raw["datetime"].min()),
        "data_end": str(raw["datetime"].max()),
        "development_period": "through 2022-12-31",
        "preselection_training_period": "through 2020-12-31",
        "preselection_calibration_period": "2021-01-01 through 2021-12-31",
        "model_selection_period": "2022-01-01 through 2022-12-31",
        "calibration_period": "2023-01-01 through 2023-12-31",
        "test_period": "2024-01-01 onward",
        "horizons": HORIZONS,
        "features": FEATURES,
        "same_strike_and_expiry": True,
        "future_quote_used_for_entry_eligibility": False,
        "expiry_selection": "earliest unexpired M1 expiry visible in the current snapshot",
        "feature_cutoff": "one minute before executable entry quote",
        "missing_exit_strike_method": "linear interpolation/extrapolation of observed exit straddle surface, floored at absolute futures moneyness",
        "model_outputs_used_as_features": False,
        "label": "1 if short-straddle net rupees > 0 after statutory option charges and 2 points slippage",
        "dataset_counts": dataset_counts,
    }
    (OUTPUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (OUTPUT / "report.md").write_text(
        make_report(selection_results, test_results, metadata), encoding="utf-8"
    )
    print("\nPre-test model selection (2022)\n", flush=True)
    print(selection_results.to_string(index=False), flush=True)
    print("\nFrozen selected-model test results (2024 onward)\n", flush=True)
    print(test_results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
