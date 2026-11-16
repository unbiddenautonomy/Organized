from __future__ import annotations

import csv
import json
from datetime import timedelta
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
    "entry_iv_call", "entry_iv_put", "entry_atm_iv_combined", "entry_atm_iv_vega_weighted",
    "entry_call_put_iv_spread_signed", "entry_call_put_iv_difference", "entry_straddle_price",
    "entry_total_vega", "entry_net_delta", "entry_total_gamma", "entry_total_theta",
    "surface_atm_iv_combined", "surface_skew_combined", "surface_smile_combined",
    "surface_rmse_combined", "surface_points_combined", "atm_strike_exact",
    "iv_call_converged", "iv_put_converged", "day_of_week", "month", "time_sin", "time_cos",
    "quality_entry_iv_converged", "quality_price", "quality_open_interest", "quality_surface",
    "quality_call_put_agreement", "quality_atm_strike", "quality_entry_ok",
]

FORBIDDEN_FEATURE_PREFIXES = ("target_", "exit_fixed_", "exit_gap_")
FORBIDDEN_FEATURE_NAMES = {
    "exit_datetime", "matched_exit_timestamp", "spot_window_120m_complete",
    "quality_exact_exit", "quality_exit_iv_valid", "quality_exact_spot_window",
    "quality_same_strike", "quality_exact_120m_timestamp", "label_available",
    "training_row_eligible", "rv_prediction_provenance",
}


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["construction"].get("combination_method") != "arithmetic_mean":
        raise ValueError("V2 requires combination_method: arithmetic_mean")
    for key in ("options_csv", "spot_csv", "output_csv"):
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    return config


def combine_call_put_iv(call_iv: pd.Series, put_iv: pd.Series) -> pd.Series:
    """Return the call-put mean only where both IV observations are finite and positive."""
    call = pd.to_numeric(call_iv, errors="coerce")
    put = pd.to_numeric(put_iv, errors="coerce")
    valid = np.isfinite(call) & np.isfinite(put) & (call > 0) & (put > 0)
    return ((call + put) / 2.0).where(valid)


def _annualized_rv(squared_returns: pd.Series, minutes: int, annual_days: int, minutes_day: int) -> pd.Series:
    return np.sqrt(squared_returns * annual_days * minutes_day / minutes)


def assert_leakage_safe_features(features: list[str]) -> None:
    forbidden = [
        name for name in features
        if name in FORBIDDEN_FEATURE_NAMES or name.startswith(FORBIDDEN_FEATURE_PREFIXES)
    ]
    if forbidden:
        raise ValueError(f"Future-derived columns found in feature list: {forbidden}")


def safe_feature_columns(frame: pd.DataFrame) -> list[str]:
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
    for name in ("rv_pred_120m", "entry_gap_to_rv_pred_combined"):
        if name in frame.columns:
            features.append(name)
    features = list(dict.fromkeys(features))
    assert_leakage_safe_features(features)
    return features


def column_manifest(frame: pd.DataFrame) -> dict[str, list[str]]:
    features = safe_feature_columns(frame)
    targets = [name for name in frame.columns if name.startswith("target_")]
    future = [
        name for name in frame.columns
        if name.startswith("exit_fixed_") or name in FORBIDDEN_FEATURE_NAMES or name.startswith("exit_gap_")
    ]
    metadata = [
        name for name in ("datetime", "Date", "dte_flag", "expiry", "exit_datetime", "matched_exit_timestamp")
        if name in frame.columns
    ]
    assigned = set(features + targets + future + metadata)
    return {
        "goal": "V2 combined call-put fixed-strike ATM IV to RV convergence",
        "primary_target": "target_fixed_combined_iv_change_120m",
        "features": features,
        "targets": targets,
        "future_diagnostics_never_use_as_features": list(dict.fromkeys(future)),
        "metadata": metadata,
        "unassigned_columns": [name for name in frame.columns if name not in assigned],
    }


def construct_spot_features(spot: pd.DataFrame, config: dict) -> pd.DataFrame:
    c = config["construction"]
    horizon = int(c["horizon_minutes"])
    lags = sorted(set(int(value) for value in c["lag_minutes"]))
    annual_days = int(c["annual_trading_days"])
    minutes_day = int(c["minutes_per_trading_day"])
    frame = spot[["datetime", "spot_price"]].copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    frame = frame.dropna()
    frame = frame[frame["spot_price"] > 0]
    frame = frame.drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    frame["Date"] = frame["datetime"].dt.date.astype(str)
    outputs = []
    one_minute = timedelta(minutes=1)
    for _, day in frame.groupby("Date", sort=False):
        day = day.sort_values("datetime").reset_index(drop=True)
        log_return = np.log(day["spot_price"]).diff()
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
    scale = np.max(np.where(valid, np.abs(x), 0.0), axis=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    z = x / scale[:, None]
    design = np.stack((np.ones_like(z), z, z * z), axis=2)
    counts = valid.sum(axis=1)
    eligible = counts >= minimum_points
    coefficients_scaled = np.full((len(x), 3), np.nan)
    for row in np.flatnonzero(eligible):
        coefficients_scaled[row] = np.linalg.lstsq(
            design[row, valid[row]], y[row, valid[row]], rcond=None
        )[0]
    fitted = np.einsum("gpi,gi->gp", design, np.nan_to_num(coefficients_scaled))
    squared_error = np.where(valid, (y - fitted) ** 2, 0.0)
    rmse = np.full(len(x), np.nan)
    rmse[eligible] = np.sqrt(squared_error[eligible].sum(axis=1) / counts[eligible])
    coefficients = coefficients_scaled.copy()
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
    frame["iv_combined"] = combine_call_put_iv(frame["iv_call"], frame["iv_put"])
    frame["iv_both_converged"] = frame["iv_call_converged"] & frame["iv_put_converged"]
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
    y = frame["iv_combined"].to_numpy(float).reshape(groups, 5)
    converged = frame["iv_both_converged"].to_numpy().reshape(groups, 5)
    valid_surface = np.isfinite(x) & np.isfinite(y) & (y > 0) & converged
    coefficient, rmse = _fit_quadratics(x, y, valid_surface, int(c["minimum_surface_points"]))

    distance = np.abs(strike - frame["atm_strike"].to_numpy(float).reshape(groups, 5))
    choice = distance.argmin(axis=1)
    entry = frame.iloc[np.arange(groups) * 5 + choice].copy().reset_index(drop=True)
    entry["atm_strike_exact"] = np.isclose(entry["strike"], entry["atm_strike"])
    entry = entry.rename(columns={"strike": "entry_strike", "iv_call": "entry_iv_call", "iv_put": "entry_iv_put"})
    entry["entry_atm_iv_combined"] = combine_call_put_iv(entry["entry_iv_call"], entry["entry_iv_put"])
    total_vega = entry["vega_call"].abs() + entry["vega_put"].abs()
    entry["entry_atm_iv_vega_weighted"] = (
        entry["entry_iv_call"] * entry["vega_call"].abs()
        + entry["entry_iv_put"] * entry["vega_put"].abs()
    ) / total_vega.replace(0, np.nan)
    entry["entry_call_put_iv_spread_signed"] = entry["entry_iv_call"] - entry["entry_iv_put"]
    entry["entry_call_put_iv_difference"] = entry["entry_call_put_iv_spread_signed"].abs()
    entry["entry_straddle_price"] = entry["call_price"] + entry["put_price"]
    entry["entry_total_vega"] = total_vega
    entry["entry_net_delta"] = entry["delta_call"] + entry["delta_put"]
    entry["entry_total_gamma"] = entry["gamma_call"] + entry["gamma_put"]
    entry["entry_total_theta"] = entry["theta_call"] + entry["theta_put"]
    entry["surface_atm_iv_combined"] = coefficient[:, 0]
    entry["surface_skew_combined"] = coefficient[:, 1]
    entry["surface_smile_combined"] = 2.0 * coefficient[:, 2]
    entry["surface_rmse_combined"] = rmse
    entry["surface_points_combined"] = valid_surface.sum(axis=1)
    entry["exit_datetime"] = entry["datetime"] + timedelta(minutes=horizon)

    minute = entry["datetime"].dt.hour * 60 + entry["datetime"].dt.minute
    session_minute = minute - (9 * 60 + 15)
    entry["day_of_week"] = entry["datetime"].dt.dayofweek
    entry["month"] = entry["datetime"].dt.month
    entry["time_sin"] = np.sin(2 * np.pi * session_minute / int(c["minutes_per_trading_day"]))
    entry["time_cos"] = np.cos(2 * np.pi * session_minute / int(c["minutes_per_trading_day"]))

    exit_rows = frame[[
        "datetime", "dte_flag", "expiry", "strike", "iv_call", "iv_put",
        "iv_call_converged", "iv_put_converged", "call_price", "put_price",
    ]].copy()
    exit_rows["exit_fixed_atm_iv_combined"] = combine_call_put_iv(exit_rows["iv_call"], exit_rows["iv_put"])
    exit_rows = exit_rows.rename(columns={
        "datetime": "exit_datetime", "strike": "exit_fixed_strike",
        "iv_call": "exit_fixed_iv_call", "iv_put": "exit_fixed_iv_put",
        "iv_call_converged": "exit_fixed_iv_call_converged",
        "iv_put_converged": "exit_fixed_iv_put_converged",
        "call_price": "exit_fixed_call_price", "put_price": "exit_fixed_put_price",
    })
    exit_rows["matched_exit_timestamp"] = exit_rows["exit_datetime"]
    entry = entry.merge(
        exit_rows,
        left_on=["exit_datetime", "dte_flag", "expiry", "entry_strike"],
        right_on=["exit_datetime", "dte_flag", "expiry", "exit_fixed_strike"],
        how="left", validate="many_to_one",
    )

    lag_fields = [
        "entry_atm_iv_combined", "entry_atm_iv_vega_weighted",
        "entry_call_put_iv_spread_signed", "entry_call_put_iv_difference",
        "surface_atm_iv_combined", "surface_skew_combined", "surface_smile_combined",
        "entry_straddle_price", "call_oi", "put_oi", "Future_oi", "future_open",
    ]
    keys = ["datetime", "dte_flag", "expiry"]
    for lag in sorted(set(int(value) for value in c["lag_minutes"])):
        prior = entry[keys + lag_fields].copy()
        prior["datetime"] += timedelta(minutes=lag)
        prior = prior.rename(columns={name: f"{name}_lag_{lag}m" for name in lag_fields})
        entry = entry.merge(prior, on=keys, how="left", validate="many_to_one")
        for name in lag_fields:
            entry[f"{name}_change_{lag}m"] = entry[name] - entry[f"{name}_lag_{lag}m"]

    entry = entry.copy()
    spot_day = spot_features[spot_features["datetime"].dt.date == entry["datetime"].dt.date.iloc[0]]
    entry = entry.merge(spot_day, on="datetime", how="left", suffixes=("", "_spotfile"), validate="many_to_one")
    entry["target_fixed_combined_iv_change_120m"] = entry["exit_fixed_atm_iv_combined"] - entry["entry_atm_iv_combined"]
    entry["target_fixed_combined_iv_return_120m"] = (
        entry["exit_fixed_atm_iv_combined"] / entry["entry_atm_iv_combined"] - 1.0
    )
    actual_entry_gap = entry["entry_atm_iv_combined"] - entry["target_rv_forward_120m"]
    actual_exit_gap = entry["exit_fixed_atm_iv_combined"] - entry["target_rv_forward_120m"]
    valid_actual = actual_entry_gap.notna() & actual_exit_gap.notna()
    entry["target_combined_converged_to_actual_rv"] = (actual_exit_gap.abs() < actual_entry_gap.abs()).where(valid_actual)
    entry["target_combined_actual_rv_gap_change"] = (actual_exit_gap.abs() - actual_entry_gap.abs()).where(valid_actual)

    entry["quality_entry_iv_converged"] = entry["iv_call_converged"] & entry["iv_put_converged"]
    entry["quality_price"] = (
        (entry["call_price"] >= float(c["minimum_option_price"]))
        & (entry["put_price"] >= float(c["minimum_option_price"]))
    )
    entry["quality_open_interest"] = (
        (entry["call_oi"] >= float(c["minimum_open_interest"]))
        & (entry["put_oi"] >= float(c["minimum_open_interest"]))
    )
    entry["quality_surface"] = entry["surface_rmse_combined"] <= float(c["maximum_surface_rmse"])
    entry["quality_call_put_agreement"] = (
        entry["entry_call_put_iv_difference"] <= float(c["maximum_call_put_iv_difference"])
    )
    entry["quality_atm_strike"] = entry["atm_strike_exact"]
    entry_quality = [
        "quality_entry_iv_converged", "quality_price", "quality_open_interest",
        "quality_surface", "quality_call_put_agreement", "quality_atm_strike",
    ]
    entry["quality_entry_ok"] = entry[entry_quality].all(axis=1)
    entry["quality_exact_exit"] = entry["exit_fixed_atm_iv_combined"].notna()
    entry["quality_exit_iv_valid"] = (
        entry["exit_fixed_iv_call_converged"].eq(True)
        & entry["exit_fixed_iv_put_converged"].eq(True)
        & entry["exit_fixed_atm_iv_combined"].notna()
    )
    entry["quality_exact_spot_window"] = entry["spot_window_120m_complete"].fillna(False)
    entry["quality_same_strike"] = np.isclose(entry["entry_strike"], entry["exit_fixed_strike"], equal_nan=False)
    entry["quality_exact_120m_timestamp"] = (
        entry["matched_exit_timestamp"] - entry["datetime"] == timedelta(minutes=horizon)
    ).fillna(False)
    entry["label_available"] = (
        entry["quality_exact_exit"] & entry["quality_exit_iv_valid"]
        & entry["quality_exact_spot_window"] & entry["quality_same_strike"]
        & entry["quality_exact_120m_timestamp"]
    )
    entry["training_row_eligible"] = entry["quality_entry_ok"] & entry["label_available"]
    return entry.drop(columns=["surface_row_count", "surface_group", "iv_combined", "iv_both_converged"])


def add_rv_prediction_convergence(
    frame: pd.DataFrame,
    prediction_column: str = "rv_pred_120m",
    provenance_column: str = "rv_prediction_provenance",
    material_gap: float = 0.01,
) -> pd.DataFrame:
    if prediction_column not in frame:
        raise KeyError(f"Missing RV prediction column: {prediction_column}")
    if provenance_column not in frame:
        raise KeyError(f"Missing RV prediction provenance column: {provenance_column}")
    allowed = {"oof", "walk_forward", "held_out"}
    invalid = set(frame[provenance_column].dropna().unique()) - allowed
    if invalid or frame[provenance_column].isna().any():
        raise ValueError(f"RV predictions must be out of sample; invalid provenance values: {sorted(invalid)}")
    result = frame.copy()
    entry_gap = result["entry_atm_iv_combined"] - result[prediction_column]
    exit_gap = result["exit_fixed_atm_iv_combined"] - result[prediction_column]
    result["entry_gap_to_rv_pred_combined"] = entry_gap
    result["exit_gap_to_rv_pred_combined"] = exit_gap
    valid = entry_gap.notna() & exit_gap.notna()
    result["target_converged_to_rv_pred_combined"] = (exit_gap.abs() < entry_gap.abs()).where(valid)
    result["target_fraction_gap_closed_combined"] = (
        (result["entry_atm_iv_combined"] - result["exit_fixed_atm_iv_combined"]) / entry_gap
    ).where(valid & (entry_gap.abs() >= material_gap))
    return result


def purged_chronological_split(
    frame: pd.DataFrame,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    horizon_minutes: int = 120,
) -> dict[str, pd.DataFrame]:
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Fractions must leave non-empty train, validation, and test periods")
    ordered = frame.sort_values("datetime").copy()
    ordered["datetime"] = pd.to_datetime(ordered["datetime"])
    if "exit_datetime" not in ordered:
        ordered["exit_datetime"] = ordered["datetime"] + timedelta(minutes=horizon_minutes)
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
    labeled = frame[frame["quality_exact_exit"]].copy()
    same_strike = np.isclose(labeled["entry_strike"], labeled["exit_fixed_strike"], equal_nan=False)
    exact_time = (
        pd.to_datetime(labeled["matched_exit_timestamp"]) - pd.to_datetime(labeled["datetime"])
        == timedelta(minutes=horizon_minutes)
    )
    if not frame["atm_strike_exact"].all():
        raise AssertionError("At least one entry strike differs from the source ATM strike")
    if not same_strike.all():
        raise AssertionError("At least one exit uses a different strike from the entry ATM strike")
    if not exact_time.all():
        raise AssertionError("At least one exit timestamp is not exactly the requested horizon after entry")
    return {
        "rows": int(len(frame)),
        "exact_exit_rows": int(len(labeled)),
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
    chunks = pd.read_csv(
        config["data"]["options_csv"], usecols=OPTION_COLUMNS,
        chunksize=int(config["data"]["chunksize"]), low_memory=False,
    )
    stop = False
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
                stop = True
                break
        if stop:
            break
    if not carry.empty and max_days is None:
        built = construct_option_day(carry, spot_features, config)
        if not built.empty:
            validate_temporal_integrity(built, int(config["construction"]["horizon_minutes"]))
            built.to_csv(output_path, mode="w" if first else "a", header=first, index=False, quoting=csv.QUOTE_MINIMAL)
            rows += len(built)
            entry_quality_rows += int(built["quality_entry_ok"].sum())
            label_rows += int(built["label_available"].sum())
            eligible_rows += int(built["training_row_eligible"].sum())
            last_built = built
        days += 1
    if last_built is None:
        raise RuntimeError("No feature rows were built")
    manifest_path = output_path.with_name(output_path.stem + "_column_manifest.json")
    manifest_path.write_text(json.dumps(column_manifest(last_built), indent=2), encoding="utf-8")
    audit = {
        "goal": config["construction"]["goal_name"],
        "combination_method": config["construction"]["combination_method"],
        "horizon_minutes": int(config["construction"]["horizon_minutes"]),
        "days": days,
        "rows": rows,
        "entry_quality_rows": entry_quality_rows,
        "label_available_rows": label_rows,
        "training_eligible_rows": eligible_rows,
        "entry_quality_rate": entry_quality_rows / rows if rows else None,
        "label_availability_rate": label_rows / rows if rows else None,
        "training_eligibility_rate": eligible_rows / rows if rows else None,
        "output": str(output_path),
        "column_manifest": str(manifest_path),
    }
    audit_path = output_path.with_name(output_path.stem + "_data_audit.json")
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit["data_audit"] = str(audit_path)
    return audit
