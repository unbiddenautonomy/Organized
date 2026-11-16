from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


OPTION_COLUMNS = [
    "datetime", "Date", "dte_flag", "expiry", "strike", "atm_strike",
    "days_to_expiry", "spot_price", "future_open", "Future_oi", "synthetic_future",
    "call_price", "put_price", "call_oi", "put_oi", "iv_call", "iv_put",
    "iv_call_converged", "iv_put_converged", "delta_call", "delta_put",
    "vega_call", "vega_put", "gamma_call", "gamma_put", "theta_call", "theta_put",
]
SURFACE_KEYS = ["datetime", "dte_flag", "expiry"]

ENTRY_FEATURES = [
    "entry_strike", "days_to_expiry", "future_open", "Future_oi", "spot_price",
    "call_price", "put_price", "call_oi", "put_oi", "synthetic_future", "atm_strike",
    "entry_iv_call", "entry_iv_put", "entry_iv_weighted", "entry_straddle_price",
    "entry_call_put_iv_difference", "delta_call", "delta_put", "vega_call", "vega_put",
    "gamma_call", "gamma_put", "theta_call", "theta_put", "surface_atm_iv_call",
    "surface_atm_iv_put", "surface_skew_call", "surface_skew_put", "surface_smile_call",
    "surface_smile_put", "surface_rmse_call", "surface_rmse_put", "surface_points_call",
    "surface_points_put", "atm_strike_exact", "iv_call_converged", "iv_put_converged",
    "day_of_week", "month", "time_sin", "time_cos",
    "quality_entry_iv_converged", "quality_price", "quality_open_interest",
    "quality_surface", "quality_call_put_agreement", "quality_atm_strike", "quality_entry_ok",
]

FORBIDDEN_FEATURE_PREFIXES = ("target_", "exit_fixed_", "exit_gap_")
FORBIDDEN_FEATURE_NAMES = {
    "exit_datetime", "matched_exit_timestamp", "spot_window_120m_complete",
    "quality_exact_exit", "quality_exact_spot_window", "quality_same_strike",
    "quality_exact_120m_timestamp", "label_available", "training_row_eligible",
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    base = path.parent
    for key in ("options_csv", "spot_csv", "output_csv"):
        config["data"][key] = str((base / config["data"][key]).resolve())
    return config


def _annualized_rv(squared_returns: pd.Series, minutes: int, annual_days: int, minutes_day: int) -> pd.Series:
    return np.sqrt(squared_returns * annual_days * minutes_day / minutes)


def assert_leakage_safe_features(features: list[str]) -> None:
    forbidden = [name for name in features if name in FORBIDDEN_FEATURE_NAMES or name.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if forbidden:
        raise ValueError(f"Future-derived columns found in feature list: {forbidden}")


def safe_feature_columns(frame: pd.DataFrame) -> list[str]:
    """Return an explicit, prediction-time-safe feature whitelist."""
    features = [name for name in ENTRY_FEATURES if name in frame.columns]
    features.extend(
        name for name in frame.columns
        if ("_lag_" in name or "_change_" in name)
        and not name.startswith(FORBIDDEN_FEATURE_PREFIXES)
        and name not in FORBIDDEN_FEATURE_NAMES
    )
    features.extend(
        name for name in frame.columns
        if name.startswith("spot_return_") or name.startswith("rv_trailing_") or name.startswith("rq_trailing_")
    )
    for name in ("rv_pred_120m", "entry_gap_to_rv_pred_call", "entry_gap_to_rv_pred_put"):
        if name in frame.columns:
            features.append(name)
    features = list(dict.fromkeys(features))
    assert_leakage_safe_features(features)
    return features


def column_manifest(frame: pd.DataFrame) -> dict[str, list[str]]:
    features = safe_feature_columns(frame)
    targets = [name for name in frame.columns if name.startswith("target_")]
    future_diagnostics = [name for name in frame.columns if name.startswith("exit_fixed_") or name in FORBIDDEN_FEATURE_NAMES]
    metadata = [name for name in ("datetime", "Date", "dte_flag", "expiry", "exit_datetime", "matched_exit_timestamp") if name in frame]
    assigned = set(features + targets + future_diagnostics + metadata)
    return {
        "features": features,
        "targets": targets,
        "future_diagnostics_never_use_as_features": list(dict.fromkeys(future_diagnostics)),
        "metadata": metadata,
        "unassigned_columns": [name for name in frame.columns if name not in assigned],
    }


def construct_spot_features(spot: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Create exact-clock spot features and the exact forward RV label."""
    c = config["construction"]
    horizon = int(c["horizon_minutes"])
    lags = sorted(set(int(x) for x in c["lag_minutes"]))
    annual_days = int(c["annual_trading_days"])
    minutes_day = int(c["minutes_per_trading_day"])
    frame = spot[["datetime", "spot_price"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    # A zero/negative spot cannot produce a log return. Removing it also makes
    # every window spanning that timestamp fail the exact-minute continuity check.
    frame = frame.dropna()
    frame = frame[frame["spot_price"] > 0]
    frame = frame.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    frame["Date"] = frame["datetime"].dt.date.astype(str)
    outputs = []
    one_minute = pd.Timedelta(minutes=1)
    for _, day in frame.groupby("Date", sort=False):
        day = day.sort_values("datetime").reset_index(drop=True)
        log_price = np.log(day["spot_price"])
        log_return = log_price.diff()
        squared_return = log_return.pow(2)
        fourth_return = log_return.pow(4)
        for lag in lags:
            exact = day["datetime"].shift(lag).eq(day["datetime"] - lag * one_minute)
            day[f"spot_return_{lag}m"] = (day["spot_price"] / day["spot_price"].shift(lag) - 1.0).where(exact)
            rv_sum = squared_return.rolling(lag, min_periods=lag).sum()
            day[f"rv_trailing_{lag}m"] = _annualized_rv(rv_sum, lag, annual_days, minutes_day).where(exact)
            rq_sum = fourth_return.rolling(lag, min_periods=lag).sum()
            annualization = annual_days * minutes_day / lag
            day[f"rq_trailing_{lag}m"] = ((lag / 3.0) * rq_sum * annualization**2).where(exact)
        exact_forward = day["datetime"].shift(-horizon).eq(day["datetime"] + horizon * one_minute)
        forward_sum = squared_return.rolling(horizon, min_periods=horizon).sum().shift(-horizon)
        day["target_rv_forward_120m"] = _annualized_rv(forward_sum, horizon, annual_days, minutes_day).where(exact_forward)
        day["target_spot_return_120m"] = (day["spot_price"].shift(-horizon) / day["spot_price"] - 1.0).where(exact_forward)
        day["spot_window_120m_complete"] = exact_forward
        outputs.append(day)
    return pd.concat(outputs, ignore_index=True).drop(columns="Date")


def _fit_quadratics(x: np.ndarray, y: np.ndarray, valid: np.ndarray, minimum_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Stable batched y=a+b*x+c*x^2 fits, returning x-scale coefficients."""
    scale = np.max(np.where(valid, np.abs(x), 0.0), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z = x / scale[:, None]
    design = np.stack((np.ones_like(z), z, z * z), axis=2)
    weights = valid.astype(float)
    xtx = np.einsum("gpi,gp,gpj->gij", design, weights, design)
    xty = np.einsum("gpi,gp,gp->gi", design, weights, np.where(valid, y, 0.0))
    counts = valid.sum(axis=1)
    eligible = counts >= minimum_points
    scaled = np.full((len(x), 3), np.nan)
    for row in np.flatnonzero(eligible):
        scaled[row] = np.linalg.lstsq(design[row, valid[row]], y[row, valid[row]], rcond=None)[0]
    fitted = np.einsum("gpi,gi->gp", design, np.nan_to_num(scaled))
    squared_error = np.where(valid, (y - fitted) ** 2, 0.0)
    rmse = np.full(len(x), np.nan)
    rmse[eligible] = np.sqrt(squared_error[eligible].sum(axis=1) / counts[eligible])
    coefficients = scaled.copy()
    coefficients[:, 1] /= scale
    coefficients[:, 2] /= scale * scale
    return coefficients, rmse


def construct_option_day(day: pd.DataFrame, spot_features: pd.DataFrame, config: dict) -> pd.DataFrame:
    c = config["construction"]
    horizon = int(c["horizon_minutes"])
    frame = day.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame["expiry"] = pd.to_datetime(frame["expiry"])
    for side in ("call", "put"):
        frame[f"iv_{side}_converged"] = frame[f"iv_{side}_converged"].astype(str).str.lower().eq("true")
    frame = frame.sort_values(SURFACE_KEYS + ["strike"])
    grouped = frame.groupby(SURFACE_KEYS, sort=False, observed=True)
    frame["surface_row_count"] = grouped["strike"].transform("size")
    frame = frame[frame["surface_row_count"] == 5].copy()
    if frame.empty:
        return frame
    frame["surface_group"] = frame.groupby(SURFACE_KEYS, sort=False, observed=True).ngroup()
    frame = frame.sort_values(["surface_group", "strike"])
    groups = int(frame["surface_group"].max()) + 1
    strike = frame["strike"].to_numpy(float).reshape(groups, 5)
    forward = frame["synthetic_future"].to_numpy(float).reshape(groups, 5)
    x = np.log(strike / forward)
    surface_values: dict[str, np.ndarray] = {}
    for side in ("call", "put"):
        y = frame[f"iv_{side}"].to_numpy(float).reshape(groups, 5)
        converged = frame[f"iv_{side}_converged"].to_numpy().reshape(groups, 5)
        valid = np.isfinite(x) & np.isfinite(y) & (y > 0) & converged
        coef, rmse = _fit_quadratics(x, y, valid, int(c["minimum_surface_points"]))
        surface_values[f"surface_atm_iv_{side}"] = coef[:, 0]
        surface_values[f"surface_skew_{side}"] = coef[:, 1]
        surface_values[f"surface_smile_{side}"] = 2.0 * coef[:, 2]
        surface_values[f"surface_rmse_{side}"] = rmse
        surface_values[f"surface_points_{side}"] = valid.sum(axis=1)

    # The source explicitly defines ATM. Fall back to the closest listed strike only if absent.
    distance = np.abs(strike - frame["atm_strike"].to_numpy(float).reshape(groups, 5))
    choice = distance.argmin(axis=1)
    flat_positions = np.arange(groups) * 5 + choice
    entry = frame.iloc[flat_positions].copy().reset_index(drop=True)
    entry["atm_strike_exact"] = np.isclose(entry["strike"], entry["atm_strike"])
    entry = entry.rename(columns={"strike": "entry_strike", "iv_call": "entry_iv_call", "iv_put": "entry_iv_put"})
    for name, values in surface_values.items():
        entry[name] = values
    total_vega = entry["vega_call"].abs() + entry["vega_put"].abs()
    entry["entry_iv_weighted"] = (
        entry["entry_iv_call"] * entry["vega_call"].abs() + entry["entry_iv_put"] * entry["vega_put"].abs()
    ) / total_vega.replace(0, np.nan)
    entry["entry_straddle_price"] = entry["call_price"] + entry["put_price"]
    entry["entry_call_put_iv_difference"] = (entry["entry_iv_call"] - entry["entry_iv_put"]).abs()
    entry["exit_datetime"] = entry["datetime"] + pd.Timedelta(minutes=horizon)

    minute = entry["datetime"].dt.hour * 60 + entry["datetime"].dt.minute
    session_minute = minute - (9 * 60 + 15)
    entry["day_of_week"] = entry["datetime"].dt.dayofweek
    entry["month"] = entry["datetime"].dt.month
    entry["time_sin"] = np.sin(2 * np.pi * session_minute / int(c["minutes_per_trading_day"]))
    entry["time_cos"] = np.cos(2 * np.pi * session_minute / int(c["minutes_per_trading_day"]))

    exit_rows = frame[["datetime", "dte_flag", "expiry", "strike", "iv_call", "iv_put", "call_price", "put_price", "vega_call", "vega_put"]].copy()
    exit_rows = exit_rows.rename(columns={
        "datetime": "exit_datetime", "strike": "exit_fixed_strike", "iv_call": "exit_fixed_iv_call",
        "iv_put": "exit_fixed_iv_put", "call_price": "exit_fixed_call_price", "put_price": "exit_fixed_put_price",
        "vega_call": "exit_fixed_vega_call", "vega_put": "exit_fixed_vega_put",
    })
    exit_rows["matched_exit_timestamp"] = exit_rows["exit_datetime"]
    entry = entry.merge(
        exit_rows,
        left_on=["exit_datetime", "dte_flag", "expiry", "entry_strike"],
        right_on=["exit_datetime", "dte_flag", "expiry", "exit_fixed_strike"],
        how="left", validate="many_to_one",
    )
    exit_vega = entry["exit_fixed_vega_call"].abs() + entry["exit_fixed_vega_put"].abs()
    entry["exit_fixed_iv_weighted"] = (
        entry["exit_fixed_iv_call"] * entry["exit_fixed_vega_call"].abs()
        + entry["exit_fixed_iv_put"] * entry["exit_fixed_vega_put"].abs()
    ) / exit_vega.replace(0, np.nan)

    lag_fields = [
        "surface_atm_iv_call", "surface_atm_iv_put", "surface_skew_call", "surface_skew_put",
        "surface_smile_call", "surface_smile_put", "entry_iv_weighted", "entry_straddle_price",
        "call_oi", "put_oi", "Future_oi", "future_open",
    ]
    keys = ["datetime", "dte_flag", "expiry"]
    for lag in sorted(set(int(x) for x in c["lag_minutes"])):
        prior = entry[keys + lag_fields].copy()
        prior["datetime"] += pd.Timedelta(minutes=lag)
        prior = prior.rename(columns={name: f"{name}_lag_{lag}m" for name in lag_fields})
        entry = entry.merge(prior, on=keys, how="left", validate="many_to_one")
        for name in lag_fields:
            entry[f"{name}_change_{lag}m"] = entry[name] - entry[f"{name}_lag_{lag}m"]

    # Consolidate the many lag blocks before adding labels and flags.
    entry = entry.copy()
    spot_day = spot_features[spot_features["datetime"].dt.date == entry["datetime"].dt.date.iloc[0]]
    entry = entry.merge(spot_day, on="datetime", how="left", suffixes=("", "_spotfile"), validate="many_to_one")
    for side in ("call", "put"):
        entry[f"target_fixed_iv_change_{side}_120m"] = entry[f"exit_fixed_iv_{side}"] - entry[f"entry_iv_{side}"]
        entry[f"target_converged_to_actual_rv_{side}"] = (
            (entry[f"exit_fixed_iv_{side}"] - entry["target_rv_forward_120m"]).abs()
            < (entry[f"entry_iv_{side}"] - entry["target_rv_forward_120m"]).abs()
        ).where(entry[f"exit_fixed_iv_{side}"].notna() & entry["target_rv_forward_120m"].notna())
    entry["target_fixed_iv_change_weighted_120m"] = entry["exit_fixed_iv_weighted"] - entry["entry_iv_weighted"]

    entry["quality_entry_iv_converged"] = entry["iv_call_converged"] & entry["iv_put_converged"]
    entry["quality_price"] = (entry["call_price"] >= float(c["minimum_option_price"])) & (entry["put_price"] >= float(c["minimum_option_price"]))
    entry["quality_open_interest"] = (entry["call_oi"] >= float(c["minimum_open_interest"])) & (entry["put_oi"] >= float(c["minimum_open_interest"]))
    entry["quality_surface"] = (entry["surface_rmse_call"] <= float(c["maximum_surface_rmse"])) & (entry["surface_rmse_put"] <= float(c["maximum_surface_rmse"]))
    entry["quality_call_put_agreement"] = entry["entry_call_put_iv_difference"] <= float(c["maximum_call_put_iv_difference"])
    entry["quality_atm_strike"] = entry["atm_strike_exact"]
    entry_quality_columns = [
        "quality_entry_iv_converged", "quality_price", "quality_open_interest",
        "quality_surface", "quality_call_put_agreement", "quality_atm_strike",
    ]
    entry["quality_entry_ok"] = entry[entry_quality_columns].all(axis=1)
    entry["quality_exact_exit"] = entry["exit_fixed_iv_call"].notna() & entry["exit_fixed_iv_put"].notna()
    entry["quality_exact_spot_window"] = entry["spot_window_120m_complete"].fillna(False)
    entry["quality_same_strike"] = np.isclose(entry["entry_strike"], entry["exit_fixed_strike"], equal_nan=False)
    entry["quality_exact_120m_timestamp"] = (
        entry["matched_exit_timestamp"] - entry["datetime"] == pd.Timedelta(minutes=horizon)
    ).fillna(False)
    entry["label_available"] = (
        entry["quality_exact_exit"] & entry["quality_exact_spot_window"]
        & entry["quality_same_strike"] & entry["quality_exact_120m_timestamp"]
    )
    # Historical row-selection flag only; deliberately excluded from the feature whitelist.
    entry["training_row_eligible"] = entry["quality_entry_ok"] & entry["label_available"]
    return entry.drop(columns=["surface_row_count", "surface_group"])


def add_rv_prediction_convergence(
    frame: pd.DataFrame,
    prediction_column: str = "rv_pred_120m",
    provenance_column: str = "rv_prediction_provenance",
    material_gap: float = 0.01,
) -> pd.DataFrame:
    """Attach convergence features using strictly out-of-sample RV predictions."""
    if prediction_column not in frame:
        raise KeyError(f"Missing RV prediction column: {prediction_column}")
    if provenance_column not in frame:
        raise KeyError(f"Missing RV prediction provenance column: {provenance_column}")
    allowed = {"oof", "walk_forward", "held_out"}
    invalid = set(frame[provenance_column].dropna().unique()) - allowed
    if invalid or frame[provenance_column].isna().any():
        raise ValueError(f"RV predictions must be out of sample; invalid provenance values: {sorted(invalid)}")
    result = frame.copy()
    for side in ("call", "put"):
        gap = result[f"entry_iv_{side}"] - result[prediction_column]
        exit_gap = result[f"exit_fixed_iv_{side}"] - result[prediction_column]
        result[f"entry_gap_to_rv_pred_{side}"] = gap
        result[f"exit_gap_to_rv_pred_{side}"] = exit_gap
        valid = gap.notna() & exit_gap.notna()
        result[f"target_converged_to_rv_pred_{side}"] = (exit_gap.abs() < gap.abs()).where(valid)
        result[f"target_fraction_gap_closed_{side}"] = ((result[f"entry_iv_{side}"] - result[f"exit_fixed_iv_{side}"]) / gap).where(valid & (gap.abs() >= material_gap))
    return result


def purged_chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    horizon_minutes: int = 120,
) -> dict[str, pd.DataFrame]:
    """Split whole timestamp groups and purge rows whose labels cross a boundary."""
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave non-empty train, validation, and test periods")
    ordered = frame.sort_values("datetime").copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"])
    if "exit_datetime" not in ordered:
        ordered["exit_datetime"] = ordered["datetime"] + pd.Timedelta(minutes=horizon_minutes)
    else:
        ordered["exit_datetime"] = pd.to_datetime(ordered["exit_datetime"])
    timestamps = np.sort(ordered["datetime"].dropna().unique())
    validation_start = pd.Timestamp(timestamps[int(len(timestamps) * train_fraction)])
    test_start = pd.Timestamp(timestamps[int(len(timestamps) * (train_fraction + validation_fraction))])
    train = ordered[(ordered["datetime"] < validation_start) & (ordered["exit_datetime"] < validation_start)].copy()
    validation = ordered[
        (ordered["datetime"] >= validation_start) & (ordered["datetime"] < test_start)
        & (ordered["exit_datetime"] < test_start)
    ].copy()
    test = ordered[ordered["datetime"] >= test_start].copy()
    return {"train": train, "validation": validation, "test": test}


def validate_temporal_integrity(frame: pd.DataFrame, horizon_minutes: int = 120) -> dict[str, int | float]:
    """Fail if an available exit changes the entry strike or timestamp horizon."""
    labeled = frame[frame["quality_exact_exit"]].copy()
    same_strike = np.isclose(labeled["entry_strike"], labeled["exit_fixed_strike"], equal_nan=False)
    exact_time = pd.to_datetime(labeled["matched_exit_timestamp"]) - pd.to_datetime(labeled["datetime"]) == pd.Timedelta(minutes=horizon_minutes)
    if not same_strike.all():
        raise AssertionError("At least one exit uses a different strike from the entry ATM strike")
    if not exact_time.all():
        raise AssertionError("At least one exit timestamp is not exactly horizon_minutes after entry")
    if not frame["atm_strike_exact"].all():
        raise AssertionError("At least one entry strike differs from the source ATM strike")
    return {
        "rows": int(len(frame)), "exact_exit_rows": int(len(labeled)),
        "entry_atm_match_rate": float(frame["atm_strike_exact"].mean()),
        "same_strike_exit_match_rate": float(same_strike.mean()) if len(labeled) else float("nan"),
        "exact_timestamp_match_rate": float(exact_time.mean()) if len(labeled) else float("nan"),
    }


def build_feature_table(config: dict, output: str | Path | None = None, max_days: int | None = None) -> dict:
    output_path = Path(output or config["data"]["output_csv"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    spot_raw = pd.read_csv(config["data"]["spot_csv"], usecols=["datetime", "spot_price"])
    spot_features = construct_spot_features(spot_raw, config)
    first = True
    carry = pd.DataFrame()
    days = rows = entry_quality_rows = label_rows = eligible_rows = 0
    last_built = None
    chunks = pd.read_csv(config["data"]["options_csv"], usecols=OPTION_COLUMNS, chunksize=int(config["data"]["chunksize"]), low_memory=False)
    for chunk in chunks:
        chunk["Date"] = chunk["Date"].astype(str)
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_date = chunk["Date"].iloc[-1]
        complete = chunk[chunk["Date"] != last_date]
        carry = chunk[chunk["Date"] == last_date].copy()
        for _, day in complete.groupby("Date", sort=False):
            built = construct_option_day(day, spot_features, config)
            if not built.empty:
                validate_temporal_integrity(built, int(config["construction"]["horizon_minutes"]))
                built.to_csv(output_path, mode="w" if first else "a", header=first, index=False, quoting=csv.QUOTE_MINIMAL)
                first = False
                rows += len(built)
                entry_quality_rows += int(built["quality_entry_ok"].sum())
                label_rows += int(built["label_available"].sum())
                eligible_rows += int(built["training_row_eligible"].sum())
                last_built = built
            days += 1
            if max_days is not None and days >= max_days:
                break
        if max_days is not None and days >= max_days:
            break
    if not carry.empty and max_days is None:
        built = construct_option_day(carry, spot_features, config)
        if not built.empty:
            validate_temporal_integrity(built, int(config["construction"]["horizon_minutes"]))
            built.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
            rows += len(built)
            entry_quality_rows += int(built["quality_entry_ok"].sum())
            label_rows += int(built["label_available"].sum())
            eligible_rows += int(built["training_row_eligible"].sum())
            last_built = built
        days += 1
    if last_built is not None:
        manifest_path = output_path.with_name(output_path.stem + "_column_manifest.json")
        manifest_path.write_text(json.dumps(column_manifest(last_built), indent=2), encoding="utf-8")
    else:
        manifest_path = None
    return {
        "days": days, "rows": rows, "entry_quality_rows": entry_quality_rows,
        "label_available_rows": label_rows, "training_eligible_rows": eligible_rows,
        "output": str(output_path), "column_manifest": str(manifest_path) if manifest_path else None,
    }

