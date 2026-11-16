"""Build the frozen, causal 120-minute BANKNIFTY predictive dataset.

This module constructs features and labels only.  It does not fit, score, or
compare models and it never summarizes target outcomes by split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.integrity import (  # noqa: E402
    localize_asia_kolkata,
    spot_integrity_flags,
    valid_spot_windows,
)


SPOT_PATH = ROOT / "data" / "raw" / "icharts_banknifty_spot_1min.csv"
SPLITS_PATH = ROOT / "SPLITS.md"
SPLITS_HASH_PATH = ROOT / "SPLITS.sha256"
OUTPUT_DIR = ROOT / "outputs" / "research_v2" / "common"
DATASET_PATH = OUTPUT_DIR / "predictive_dataset_120m.parquet"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
HASHES_PATH = OUTPUT_DIR / "SHA256SUMS"

EPS = 1e-12
HORIZON = 120
MAX_TRAILING_WINDOW = 60

P0_FEATURES = ["p0_log_rv_5", "p0_log_rv_15", "p0_log_rv_60"]
P1_EXTENSION_FEATURES = [
    "p1_log_rq_15",
    "p1_log_rq_60",
    "p1_downside_rv_15",
    "p1_upside_rv_15",
    "p1_downside_rv_60",
    "p1_upside_rv_60",
    "p1_jump_rv_60",
    "p1_abs_return",
    "p1_tod_sin",
    "p1_tod_cos",
]
P1_FEATURES = P0_FEATURES + P1_EXTENSION_FEATURES

SPLIT_BOUNDS = {
    "development": (pd.Timestamp("2019-10-11"), pd.Timestamp("2022-12-30")),
    "walk_forward_validation": (pd.Timestamp("2023-01-02"), pd.Timestamp("2024-06-28")),
    "pseudo_holdout_previously_exposed": (pd.Timestamp("2024-07-01"), pd.Timestamp("2025-05-22")),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_splits_checksum(
    splits_path: Path = SPLITS_PATH,
    checksum_path: Path = SPLITS_HASH_PATH,
) -> dict[str, Any]:
    expected = checksum_path.read_text(encoding="utf-8").strip().split()[0].lower()
    actual = sha256_file(splits_path)
    if actual != expected:
        raise ValueError(f"SPLITS.md checksum mismatch: expected {expected}, got {actual}")
    return {"expected": expected, "actual": actual, "matched": True}


def assign_split_labels(decision_times: pd.Series) -> pd.Series:
    stamps = localize_asia_kolkata(decision_times)
    dates = stamps.dt.tz_localize(None).dt.normalize()
    result = pd.Series(pd.NA, index=decision_times.index, dtype="string", name="split")
    for label, (start, end) in SPLIT_BOUNDS.items():
        result.loc[dates.between(start, end)] = label
    return result


def _scheduled_decision(timestamp: pd.Timestamp) -> bool:
    minute_of_day = timestamp.hour * 60 + timestamp.minute
    return 9 * 60 + 45 <= minute_of_day <= 13 * 60 + 30 and timestamp.minute % 5 == 0


def _window_features(returns: np.ndarray, position: int, timestamp: pd.Timestamp) -> dict[str, float]:
    trailing: dict[int, np.ndarray] = {
        window: returns[position - window + 1 : position + 1] for window in (5, 15, 60)
    }
    rv = {window: float(np.sum(values**2)) for window, values in trailing.items()}
    result = {
        "p0_log_rv_5": float(np.log(max(rv[5], EPS))),
        "p0_log_rv_15": float(np.log(max(rv[15], EPS))),
        "p0_log_rv_60": float(np.log(max(rv[60], EPS))),
    }
    for window in (15, 60):
        values = trailing[window]
        rq = (window / 3.0) * float(np.sum(values**4))
        result[f"p1_log_rq_{window}"] = float(np.log(max(rq, EPS)))
        result[f"p1_downside_rv_{window}"] = float(np.sum(np.square(values[values < 0])))
        result[f"p1_upside_rv_{window}"] = float(np.sum(np.square(values[values > 0])))
    values_60 = trailing[60]
    bipower_60 = (np.pi / 2.0) * float(np.sum(np.abs(values_60[1:]) * np.abs(values_60[:-1])))
    result["p1_jump_rv_60"] = max(rv[60] - bipower_60, 0.0)
    result["p1_abs_return"] = float(abs(returns[position]))
    session_fraction = ((timestamp.hour * 60 + timestamp.minute) - (9 * 60 + 15)) / 375.0
    result["p1_tod_sin"] = float(np.sin(2.0 * np.pi * session_fraction))
    result["p1_tod_cos"] = float(np.cos(2.0 * np.pi * session_fraction))
    return result


def build_predictive_dataset(spot: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Construct causal features and exact t+1..t+120 labels."""
    required = {"datetime", "spot_price"}
    missing = required.difference(spot.columns)
    if missing:
        raise KeyError(f"spot data missing columns: {sorted(missing)}")

    raw_rows = len(spot)
    work = spot[["datetime", "spot_price"]].copy()
    work["datetime"] = localize_asia_kolkata(work["datetime"])
    work["spot_price"] = pd.to_numeric(work["spot_price"], errors="coerce")
    null_timestamp_rows = int(work["datetime"].isna().sum())
    duplicate_timestamp_rows = int(work["datetime"].duplicated(keep=False).sum())
    work = work.dropna(subset=["datetime"])
    work = work.loc[~work["datetime"].duplicated(keep=False)].sort_values("datetime", kind="stable")
    work = work.reset_index(drop=True)

    flags = spot_integrity_flags(work)
    work = pd.concat([work, flags], axis=1)
    work["decision_date"] = work["datetime"].dt.normalize()

    candidate_positions: list[int] = []
    rows: list[dict[str, Any]] = []
    insufficient_position = 0
    for _, day in work.groupby("decision_date", sort=True):
        positions = day.index.to_numpy()
        timestamps = day["datetime"].tolist()
        prices = day["spot_price"].to_numpy(dtype=float)
        returns = np.full(len(day), np.nan, dtype=float)
        if len(day) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                returns[1:] = np.log(prices[1:] / prices[:-1])
        for local_position, global_position in enumerate(positions):
            timestamp = timestamps[local_position]
            if not _scheduled_decision(timestamp):
                continue
            candidate_positions.append(int(global_position))
            if local_position < MAX_TRAILING_WINDOW or local_position + HORIZON >= len(day):
                insufficient_position += 1
                continue
            feature_slice = returns[local_position - MAX_TRAILING_WINDOW + 1 : local_position + 1]
            target_slice = returns[local_position + 1 : local_position + HORIZON + 1]
            label_end_at = timestamps[local_position + HORIZON]
            feature_values_finite = bool(np.isfinite(feature_slice).all())
            target_values_finite = bool(np.isfinite(target_slice).all())
            record: dict[str, Any] = {
                "decision_time": timestamp,
                "decision_date": timestamp.normalize(),
                "label_end_at": label_end_at,
                "spot_price": float(prices[local_position]),
                "rv_fwd_120": float(np.sum(target_slice**2)) if target_values_finite else np.nan,
            }
            record["target"] = float(np.log(max(record["rv_fwd_120"], EPS)))
            if feature_values_finite:
                record.update(_window_features(returns, local_position, timestamp))
            else:
                record.update({feature: np.nan for feature in P1_FEATURES})
            record["_global_position"] = int(global_position)
            record["_feature_values_finite"] = feature_values_finite
            record["_target_values_finite"] = target_values_finite
            rows.append(record)

    candidates_with_positions = pd.DataFrame(rows)
    if candidates_with_positions.empty:
        raise ValueError("no candidate rows have sufficient positional history and horizon")

    trailing_starts = work.loc[
        candidates_with_positions["_global_position"].to_numpy() - MAX_TRAILING_WINDOW, "datetime"
    ].reset_index(drop=True)
    future_ends = candidates_with_positions["label_end_at"].reset_index(drop=True)
    decisions = candidates_with_positions["decision_time"].reset_index(drop=True)
    trailing_valid = valid_spot_windows(trailing_starts, decisions, work).to_numpy()
    target_valid = valid_spot_windows(decisions, future_ends, work).to_numpy()
    finite = (
        candidates_with_positions["_feature_values_finite"].to_numpy()
        & candidates_with_positions["_target_values_finite"].to_numpy()
    )
    candidates_with_positions["split"] = assign_split_labels(candidates_with_positions["decision_time"])
    in_frozen_split = candidates_with_positions["split"].notna().to_numpy()
    keep = trailing_valid & target_valid & finite & in_frozen_split

    dataset = candidates_with_positions.loc[keep].drop(
        columns=["_global_position", "_feature_values_finite", "_target_values_finite"]
    )
    ordered = [
        "decision_time", "decision_date", "label_end_at", "split", "spot_price",
        "rv_fwd_120", "target", *P0_FEATURES, *P1_EXTENSION_FEATURES,
    ]
    dataset = dataset[ordered].sort_values("decision_time", kind="stable").reset_index(drop=True)

    split_coverage: dict[str, Any] = {}
    for label in SPLIT_BOUNDS:
        selected = dataset["split"].eq(label)
        split_coverage[label] = {
            "rows": int(selected.sum()),
            "sessions": int(dataset.loc[selected, "decision_date"].nunique()),
            "first_decision": str(dataset.loc[selected, "decision_time"].min()),
            "last_decision": str(dataset.loc[selected, "decision_time"].max()),
        }
    audit = {
        "raw_rows": int(raw_rows),
        "null_timestamp_rows": null_timestamp_rows,
        "duplicate_timestamp_rows": duplicate_timestamp_rows,
        "ordinary_primary_rows": int(work["is_primary_session"].sum()),
        "nonpositive_or_nonfinite_spot_rows": int((~work["spot_is_positive"]).sum()),
        "rows_after_gap_flag": int(work["gap_from_previous"].sum()),
        "scheduled_decision_rows": int(len(candidate_positions)),
        "excluded_insufficient_position": int(insufficient_position),
        "candidate_rows_with_positional_windows": int(len(candidates_with_positions)),
        "excluded_invalid_trailing_window": int((~trailing_valid).sum()),
        "excluded_invalid_target_window": int((~target_valid).sum()),
        "excluded_nonfinite_feature_or_target_input": int((~finite).sum()),
        "excluded_outside_frozen_splits": int((~in_frozen_split).sum()),
        "retained_rows": int(len(dataset)),
        "split_coverage": split_coverage,
    }
    return dataset, audit


def write_artifacts(dataset: pd.DataFrame, audit: dict[str, Any]) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.register("predictive_dataset", dataset)
        destination = str(DATASET_PATH).replace("'", "''")
        connection.execute(
            f"COPY predictive_dataset TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)"
        )
    finally:
        connection.close()

    split_checksum = verify_splits_checksum()
    manifest = {
        "dataset_version": "research_v2_spot_120m_v1",
        "source": str(SPOT_PATH.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256_file(SPOT_PATH),
        "splits_sha256": split_checksum["actual"],
        "builder": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "builder_sha256": sha256_file(Path(__file__)),
        "dataset_file": DATASET_PATH.name,
        "dataset_sha256": sha256_file(DATASET_PATH),
        "horizon_trading_minutes": HORIZON,
        "target_definition": "sum(log(S[t+i]/S[t+i-1])^2, i=1..120)",
        "model_target": "log(max(rv_fwd_120, 1e-12))",
        "p0_features": P0_FEATURES,
        "p1_features": P1_FEATURES,
        "audit": audit,
        "target_outcomes_summarized": False,
        "models_fitted": False,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    hashes = {
        DATASET_PATH.name: sha256_file(DATASET_PATH),
        MANIFEST_PATH.name: sha256_file(MANIFEST_PATH),
    }
    HASHES_PATH.write_text(
        "".join(f"{digest} *{name}\n" for name, digest in sorted(hashes.items())),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SPOT_PATH)
    args = parser.parse_args()
    verify_splits_checksum()
    spot = pd.read_csv(args.source, usecols=["datetime", "spot_price"])
    dataset, audit = build_predictive_dataset(spot)
    manifest = write_artifacts(dataset, audit)
    print(json.dumps({
        "dataset": str(DATASET_PATH),
        "rows": len(dataset),
        "dataset_sha256": manifest["dataset_sha256"],
        "split_rows": {key: value["rows"] for key, value in audit["split_coverage"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
