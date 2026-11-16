from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .surface import make_entry_rows


ENTRY_COLUMNS = [
    "source_index", "datetime", "dte_flag", "strike", "expiry", "Date", "days_to_expiry",
    "future_open", "Future_oi", "spot_price", "call_price", "put_price", "call_oi", "put_oi",
    "atm_strike", "iv_call", "iv_call_converged", "delta_call", "vega_call", "gamma_call", "theta_call",
    "iv_put", "iv_put_converged", "delta_put", "vega_put", "gamma_put", "theta_put",
]

EXIT_COLUMNS = [
    "datetime", "dte_flag", "strike", "expiry", "Date", "days_to_expiry", "future_open", "spot_price",
    "call_price", "put_price", "call_oi", "put_oi", "iv_call", "iv_call_converged", "iv_call_reason",
    "delta_call", "vega_call", "gamma_call", "theta_call", "iv_put", "iv_put_converged", "iv_put_reason",
    "delta_put", "vega_put", "gamma_put", "theta_put",
]

FORBIDDEN_FEATURE_PREFIXES = ("exit_", "label_", "quality_")
NON_FEATURE_COLUMNS = {
    "row_id", "source_index", "datetime", "Date", "expiry", "dte_flag", "split",
}


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.fillna(False).astype(str).str.lower().isin({"true", "1", "yes"})


def _add_lag_features(entries: pd.DataFrame, lag_minutes: list[int]) -> pd.DataFrame:
    output = entries.sort_values("datetime").copy()
    source_columns = [
        "entry_spot_price", "entry_future_price", "entry_iv_call", "entry_iv_put",
        "surface_atm_iv_call", "surface_atm_iv_put", "surface_skew_call", "surface_skew_put",
        "surface_smile_call", "surface_smile_put", "entry_straddle_price", "entry_call_oi", "entry_put_oi",
    ]
    base = output[["datetime", "Date", "dte_flag", "expiry"] + source_columns]
    for minutes in lag_minutes:
        lag = base.copy()
        lag["datetime"] = lag["datetime"] + pd.Timedelta(minutes=minutes)
        lag = lag.rename(columns={column: f"lag_{minutes}m_{column}" for column in source_columns})
        output = output.merge(lag, on=["datetime", "Date", "dte_flag", "expiry"], how="left", validate="many_to_one")
        for column in source_columns:
            output[f"change_{minutes}m_{column}"] = output[column] - output[f"lag_{minutes}m_{column}"]
    minute = output["datetime"].dt.hour * 60 + output["datetime"].dt.minute - (9 * 60 + 15)
    output["entry_time_sin"] = np.sin(2 * np.pi * minute / 375)
    output["entry_time_cos"] = np.cos(2 * np.pi * minute / 375)
    output["entry_day_of_week"] = output["datetime"].dt.dayofweek
    return output


def build_entries(config: dict, start_date: str | None = None, max_days: int | None = None) -> dict:
    source = Path(config["data"]["entry_surface_csv"])
    destination = Path(config["data"]["entries_csv"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    carry = pd.DataFrame()
    collected = []
    days = 0
    stop = False
    for chunk in pd.read_csv(source, usecols=ENTRY_COLUMNS, chunksize=int(config["data"]["chunksize"]), low_memory=False):
        chunk["Date"] = chunk["Date"].astype(str)
        if start_date:
            chunk = chunk[chunk["Date"] >= start_date]
        if chunk.empty:
            continue
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_date = chunk["Date"].iloc[-1]
        complete = chunk[chunk["Date"] != last_date]
        carry = chunk[chunk["Date"] == last_date].copy()
        for _, day in complete.groupby("Date", sort=False):
            rows = make_entry_rows(day, int(config["data"]["minimum_surface_points"]), config["features"]["surface_forward_column"])
            if not rows.empty:
                collected.append(rows)
            days += 1
            if max_days is not None and days >= max_days:
                stop = True
                break
        if stop:
            break
    if not stop and not carry.empty:
        rows = make_entry_rows(carry, int(config["data"]["minimum_surface_points"]), config["features"]["surface_forward_column"])
        if not rows.empty:
            collected.append(rows)
        days += 1
    if not collected:
        raise ValueError("No entry rows were produced for the requested period")
    entries = pd.concat(collected, ignore_index=True)
    entries["datetime"] = pd.to_datetime(entries["datetime"])
    entries["expiry"] = pd.to_datetime(entries["expiry"])
    entries["exit_datetime"] = entries["datetime"] + pd.Timedelta(minutes=int(config["data"]["horizon_minutes"]))
    entries["exit_date"] = entries["exit_datetime"].dt.strftime("%Y-%m-%d")
    entries = _add_lag_features(entries, [int(value) for value in config["features"]["lag_minutes"]])
    entries = entries.sort_values(["datetime", "dte_flag", "expiry"]).reset_index(drop=True)
    entries.insert(0, "row_id", np.arange(len(entries), dtype=np.int64))
    entries.to_csv(destination, index=False)
    return {"days": days, "rows": len(entries), "path": str(destination), "first_entry": str(entries["datetime"].min()), "last_entry": str(entries["datetime"].max())}


def attach_exact_exits(config: dict) -> dict:
    entries_path = Path(config["data"]["entries_csv"])
    destination = Path(config["data"]["dataset_csv"])
    entries = pd.read_csv(entries_path, parse_dates=["datetime", "exit_datetime", "expiry"], low_memory=False)
    keys = entries[["exit_datetime", "exit_date", "dte_flag", "expiry", "entry_strike"]].drop_duplicates()
    # Join using source-format strings to avoid repeatedly parsing the 16 GB file.
    key_strings = keys.copy()
    key_strings["exit_datetime"] = key_strings["exit_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    key_strings["expiry"] = key_strings["expiry"].dt.strftime("%Y-%m-%d")
    earliest = key_strings["exit_date"].min()
    latest = key_strings["exit_date"].max()
    matches = []
    rows_scanned = 0
    for chunk in pd.read_csv(config["data"]["full_chain_csv"], usecols=EXIT_COLUMNS, chunksize=int(config["data"]["chunksize"]), low_memory=False):
        rows_scanned += len(chunk)
        chunk["Date"] = chunk["Date"].astype(str)
        chunk_min, chunk_max = chunk["Date"].min(), chunk["Date"].max()
        if chunk_max < earliest:
            continue
        if bool(config["data"].get("assume_full_chain_sorted", False)) and chunk_min > latest:
            break
        dates = chunk.loc[(chunk["Date"] >= earliest) & (chunk["Date"] <= latest), "Date"].unique()
        if len(dates) == 0:
            continue
        wanted = key_strings[key_strings["exit_date"].isin(dates)]
        if wanted.empty:
            continue
        candidate = chunk.rename(columns={"datetime": "exit_datetime", "strike": "entry_strike", "Date": "exit_date"})
        found = candidate.merge(wanted, on=["exit_datetime", "exit_date", "dte_flag", "expiry", "entry_strike"], how="inner")
        # Preserve the matched source keys as auditable fields. These are
        # labels/metadata and are excluded from model features.
        found["matched_strike"] = found["entry_strike"]
        found["source_datetime"] = found["exit_datetime"]
        if not found.empty:
            matches.append(found)
    if matches:
        exits = pd.concat(matches, ignore_index=True)
        exits = exits.sort_values(["exit_datetime", "expiry", "entry_strike"]).drop_duplicates(
            ["exit_datetime", "dte_flag", "expiry", "entry_strike"], keep="last"
        )
    else:
        exits = pd.DataFrame(columns=["exit_datetime", "dte_flag", "expiry", "entry_strike"])

    rename = {column: f"exit_{column}" for column in EXIT_COLUMNS if column not in {"datetime", "dte_flag", "strike", "expiry", "Date"}}
    rename.update({"matched_strike": "exit_matched_strike", "source_datetime": "exit_source_datetime"})
    exits = exits.rename(columns=rename)
    exits["exit_datetime"] = pd.to_datetime(exits["exit_datetime"])
    exits["expiry"] = pd.to_datetime(exits["expiry"])
    exits = exits.drop(columns=["exit_date"], errors="ignore")
    result = entries.merge(exits, on=["exit_datetime", "dte_flag", "expiry", "entry_strike"], how="left", validate="many_to_one")
    # Lag construction adds many columns incrementally; compact before adding
    # labels to avoid pandas fragmentation on full-history runs.
    result = result.copy()
    result["label_exit_iv_call"] = result.get("exit_iv_call")
    result["label_exit_iv_put"] = result.get("exit_iv_put")
    result["label_iv_change_call"] = result["label_exit_iv_call"] - result["entry_iv_call"]
    result["label_iv_change_put"] = result["label_exit_iv_put"] - result["entry_iv_put"]
    result["label_exit_straddle_price"] = result.get("exit_call_price") + result.get("exit_put_price")
    result["label_short_straddle_pnl"] = result["entry_straddle_price"] - result["label_exit_straddle_price"]
    result["label_spot_move"] = result.get("exit_spot_price") - result["entry_spot_price"]
    result["label_call_iv_direction"] = np.sign(result["label_iv_change_call"])
    result["label_put_iv_direction"] = np.sign(result["label_iv_change_put"])

    quality = config["quality"]
    result["quality_entry"] = (
        (~bool(quality["require_exact_atm_strike"]) | _as_bool(result["entry_atm_exact"]))
        & (~bool(quality["require_entry_iv_converged"]) | (_as_bool(result["entry_iv_call_converged"]) & _as_bool(result["entry_iv_put_converged"])))
        & (result["surface_rmse_call"] <= float(quality["maximum_surface_rmse"]))
        & (result["surface_rmse_put"] <= float(quality["maximum_surface_rmse"]))
        & (result["entry_call_price"] >= float(quality["minimum_option_price"]))
        & (result["entry_put_price"] >= float(quality["minimum_option_price"]))
    )
    exit_call_ok = _as_bool(result.get("exit_iv_call_converged", pd.Series(False, index=result.index)))
    exit_put_ok = _as_bool(result.get("exit_iv_put_converged", pd.Series(False, index=result.index)))
    result["quality_exit"] = result["label_exit_iv_call"].notna() & result["label_exit_iv_put"].notna()
    if bool(quality["require_exit_iv_converged"]):
        result["quality_exit"] &= exit_call_ok & exit_put_ok
    result["quality_trainable"] = result["quality_entry"] & result["quality_exit"]
    result, split_report = _add_purged_splits(result, config)
    result.to_csv(destination, index=False)
    artifact_report = _write_model_artifacts(result, config)
    return {
        "entries": len(result), "matched_exact_exits": int(result["label_exit_iv_call"].notna().sum()),
        "trainable_rows": int(result["quality_trainable"].sum()), "rows_scanned_from_full_chain": rows_scanned,
        "path": str(destination), "splits": split_report, "model_artifacts": artifact_report,
    }


def _add_purged_splits(frame: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, dict]:
    """Chronological train/validation/test split with label-horizon purging."""
    output = frame.sort_values(["datetime", "dte_flag", "expiry"]).reset_index(drop=True).copy()
    n = len(output)
    validation_fraction = float(config["splits"]["validation_fraction"])
    test_fraction = float(config["splits"]["test_fraction"])
    validation_position = min(max(int(n * (1.0 - validation_fraction - test_fraction)), 1), n - 2)
    test_position = min(max(int(n * (1.0 - test_fraction)), validation_position + 1), n - 1)
    validation_start = output.loc[validation_position, "datetime"]
    test_start = output.loc[test_position, "datetime"]
    purge = pd.Timedelta(minutes=int(config["splits"]["purge_minutes"]))
    if purge < pd.Timedelta(minutes=int(config["data"]["horizon_minutes"])):
        raise ValueError("purge_minutes must be at least the prediction horizon")
    split = np.full(n, "purged", dtype=object)
    train_mask = (
        (output["datetime"] < validation_start - purge)
        & (output["exit_datetime"] < validation_start)
    )
    validation_mask = (
        (output["datetime"] >= validation_start) & (output["datetime"] < test_start - purge)
        & (output["exit_datetime"] < test_start)
    )
    test_mask = output["datetime"] >= test_start
    split[train_mask] = "train"
    split[validation_mask] = "validation"
    split[test_mask] = "test"
    output["split"] = split
    report = {
        "validation_start": str(validation_start), "test_start": str(test_start),
        "purge_minutes": int(purge.total_seconds() / 60),
        "counts": output["split"].value_counts().to_dict(),
    }
    return output, report


def _model_feature_columns(frame: pd.DataFrame) -> list[str]:
    numeric_or_bool = set(frame.select_dtypes(include=[np.number, "bool"]).columns)
    columns = [
        column for column in frame.columns
        if column in numeric_or_bool
        and column not in NON_FEATURE_COLUMNS
        and not column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    ]
    contaminated = [column for column in columns if column.startswith(FORBIDDEN_FEATURE_PREFIXES)]
    if contaminated:
        raise AssertionError(f"Future-dependent columns entered feature list: {contaminated}")
    return columns


def _write_model_artifacts(frame: pd.DataFrame, config: dict) -> dict:
    """Physically separate entry-time features from future labels."""
    feature_columns = _model_feature_columns(frame)
    eligible = frame["quality_trainable"].fillna(False).astype(bool)
    selected = frame.loc[eligible].copy()
    features = selected[["row_id"] + feature_columns]
    target_columns = [column for column in frame.columns if column.startswith(("label_", "quality_"))]
    metadata = [
        "row_id", "datetime", "exit_datetime", "dte_flag", "expiry", "atm_strike", "entry_strike",
        "exit_matched_strike", "exit_source_datetime", "split",
    ]
    targets = selected[metadata + target_columns]
    features.to_csv(config["data"]["model_features_csv"], index=False)
    targets.to_csv(config["data"]["model_targets_csv"], index=False)
    manifest = {
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "row_id_is_feature": False,
        "forbidden_prefixes": list(FORBIDDEN_FEATURE_PREFIXES),
        "non_feature_columns": sorted(NON_FEATURE_COLUMNS),
        "target_columns": target_columns,
        "instruction": "Train only with feature_columns. Join targets by row_id. Never infer features automatically from the combined dataset.",
    }
    Path(config["data"]["feature_manifest_json"]).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "rows": len(selected), "feature_count": len(feature_columns),
        "features_path": config["data"]["model_features_csv"],
        "targets_path": config["data"]["model_targets_csv"],
        "manifest_path": config["data"]["feature_manifest_json"],
    }


def validate_dataset(config: dict) -> dict:
    frame = pd.read_csv(config["data"]["dataset_csv"], parse_dates=["datetime", "exit_datetime", "expiry"], low_memory=False)
    horizon = int(config["data"]["horizon_minutes"])
    delta = (frame["exit_datetime"] - frame["datetime"]).dt.total_seconds() / 60
    duplicates = int(frame.duplicated(["datetime", "dte_flag", "expiry"]).sum())
    matched = frame["label_exit_iv_call"].notna() & frame["label_exit_iv_put"].notna()
    atm_same = np.isclose(frame["entry_strike"], frame["atm_strike"], rtol=0, atol=1e-9)
    matched_rows = frame[matched]
    strike_same = np.isclose(matched_rows["entry_strike"], matched_rows["exit_matched_strike"], rtol=0, atol=1e-9)
    source_time_same = pd.to_datetime(matched_rows["exit_source_datetime"]) == matched_rows["exit_datetime"]
    features = pd.read_csv(config["data"]["model_features_csv"], nrows=1)
    manifest = json.loads(Path(config["data"]["feature_manifest_json"]).read_text(encoding="utf-8"))
    leaked_feature_columns = [
        column for column in features.columns
        if column != "row_id" and column.startswith(FORBIDDEN_FEATURE_PREFIXES)
    ]
    report = {
        "rows": len(frame), "first_entry": str(frame["datetime"].min()), "last_entry": str(frame["datetime"].max()),
        "exact_horizon_rows": int((delta == horizon).sum()), "duplicate_entry_keys": duplicates,
        "exact_atm_entries": int(atm_same.sum()),
        "same_strike_at_exit_rows": int(strike_same.sum()),
        "same_source_exit_timestamp_rows": int(source_time_same.sum()),
        "matched_call_and_put_exits": int(matched.sum()), "match_rate": float(matched.mean()),
        "trainable_rows": int(frame["quality_trainable"].fillna(False).sum()),
        "call_target_mean": float(frame.loc[matched, "label_iv_change_call"].mean()) if matched.any() else None,
        "put_target_mean": float(frame.loc[matched, "label_iv_change_put"].mean()) if matched.any() else None,
        "model_feature_count": len(manifest["feature_columns"]),
        "leaked_feature_columns": leaked_feature_columns,
        "split_counts": frame["split"].value_counts().to_dict(),
        "feature_leakage_rule": "Features are physically separated; train only with feature_columns from feature_manifest.json.",
    }
    Path(config["data"]["quality_report_json"]).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["exact_horizon_rows"] != report["rows"]:
        raise AssertionError("Some rows do not have an exact 120-minute horizon")
    if duplicates:
        raise AssertionError(f"Found {duplicates} duplicate entry keys")
    if not atm_same.all():
        raise AssertionError("Some entry strikes are not the declared ATM strike at entry")
    if len(matched_rows) != int(strike_same.sum()):
        raise AssertionError("Some matched exit strikes differ from the entry ATM strike")
    if len(matched_rows) != int(source_time_same.sum()):
        raise AssertionError("Some source exit timestamps differ from entry time + 120 minutes")
    if leaked_feature_columns:
        raise AssertionError(f"Future-dependent columns found in model features: {leaked_feature_columns}")
    return report

