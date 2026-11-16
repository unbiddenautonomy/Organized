"""Reproducible, outcome-blind audit of the BANKNIFTY source files.

This intentionally inspects market-data integrity only.  It does not calculate
strategy signals, labels, trades, or P&L.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs" / "data_audit"
OUT.mkdir(parents=True, exist_ok=True)


def file_identity(path: Path) -> dict:
    with path.open("rb") as fh:
        header = fh.readline()
    return {
        "path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "bytes": path.stat().st_size,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def audit_spot(path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path)
    ts = pd.to_datetime(df["datetime"], errors="coerce")
    dates = ts.dt.normalize()
    counts = ts.groupby(dates).size().rename("rows")
    first = ts.groupby(dates).min().rename("first_timestamp")
    last = ts.groupby(dates).max().rename("last_timestamp")
    daily = pd.concat([counts, first, last], axis=1).reset_index(names="date")
    daily["weekday"] = daily["date"].dt.day_name()
    daily["internal_gaps_over_60s"] = 0
    daily["max_internal_gap_seconds"] = 0.0
    for day, grp in pd.DataFrame({"ts": ts}).dropna().groupby(ts.dt.date):
        delta = grp["ts"].sort_values().diff().dt.total_seconds().dropna()
        sel = daily["date"].dt.date == day
        # Sources mix :59 and :00 stamps, so adjacent minute bars can differ by
        # 59, 60, or 61 seconds without a missing minute.  Count only >61 s.
        daily.loc[sel, "internal_gaps_over_60s"] = int((delta > 61).sum())
        daily.loc[sel, "max_internal_gap_seconds"] = float(delta.max()) if len(delta) else 0.0
    metrics = {
        "rows": int(len(df)),
        "timestamp_min": str(ts.min()),
        "timestamp_max": str(ts.max()),
        "trading_dates": int(dates.nunique()),
        "duplicate_timestamps": int(ts.duplicated().sum()),
        "null_timestamps": int(ts.isna().sum()),
        "non_59_second_timestamps": int((ts.dt.second != 59).sum()),
        "outside_0915_1530": int((((ts.dt.hour * 60 + ts.dt.minute) < 555) | ((ts.dt.hour * 60 + ts.dt.minute) > 930)).sum()),
        "weekend_rows": int((ts.dt.dayofweek >= 5).sum()),
        "nonpositive_spot": int((pd.to_numeric(df["spot_price"], errors="coerce") <= 0).sum()),
        "daily_rows_min": int(counts.min()),
        "daily_rows_median": float(counts.median()),
        "daily_rows_max": int(counts.max()),
        "days_with_internal_gaps": int((daily["internal_gaps_over_60s"] > 0).sum()),
        "max_internal_gap_seconds": float(daily["max_internal_gap_seconds"].max()),
    }
    return metrics, daily


def audit_monthly(path: Path, spot_timestamps: set[pd.Timestamp]) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    usecols = [
        "source_index", "strike_position", "available_strike_count", "datetime", "dte_flag",
        "strike", "expiry", "Date", "Time", "days_to_expiry", "expiry_type", "is_monthly",
        "spot_price", "is_weekly", "tick_time", "call_price", "put_price", "call_oi", "put_oi",
        "iv_call", "iv_call_converged", "iv_put", "iv_put_converged",
    ]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=200_000, low_memory=False):
        chunk["datetime"] = pd.to_datetime(chunk["datetime"], errors="coerce")
        chunk["expiry"] = pd.to_datetime(chunk["expiry"], errors="coerce")
        parts.append(chunk)
    df = pd.concat(parts, ignore_index=True)
    ts = df["datetime"]
    key = ["datetime", "expiry", "strike"]
    snapshot = df.groupby(["datetime", "expiry"], observed=True).agg(
        strikes=("strike", "nunique"), stated_available=("available_strike_count", "first")
    ).reset_index()
    expiries = df.groupby("expiry", observed=True).agg(
        first_observation=("datetime", "min"), last_observation=("datetime", "max"),
        rows=("datetime", "size"), timestamps=("datetime", "nunique"),
        min_strike=("strike", "min"), max_strike=("strike", "max"), distinct_strikes=("strike", "nunique"),
        min_dte=("days_to_expiry", "min"), max_dte=("days_to_expiry", "max"),
    ).reset_index()
    expiries["expiry_weekday"] = expiries["expiry"].dt.day_name()
    date_from_field = pd.to_datetime(df["Date"], errors="coerce")
    recomputed_dte = (df["expiry"] + pd.Timedelta(hours=15, minutes=30) - ts).dt.total_seconds() / 86400
    tick = pd.to_datetime(df["Date"].astype(str) + " " + df["tick_time"].astype(str), errors="coerce")
    tick_lag = (tick - ts).dt.total_seconds()
    exact_ts = pd.Index(ts.dropna().unique())
    exact_matches = sum(pd.Timestamp(x) in spot_timestamps for x in exact_ts)
    metrics = {
        "rows": int(len(df)),
        "timestamp_min": str(ts.min()),
        "timestamp_max": str(ts.max()),
        "trading_dates": int(ts.dt.date.nunique()),
        "unique_timestamps": int(ts.nunique()),
        "unique_expiries": int(df["expiry"].nunique()),
        "unique_strikes": int(df["strike"].nunique()),
        "duplicate_datetime_expiry_strike": int(df.duplicated(key).sum()),
        "null_timestamps": int(ts.isna().sum()),
        "non_59_second_timestamps": int((ts.dt.second != 59).sum()),
        "outside_0915_1530": int((((ts.dt.hour * 60 + ts.dt.minute) < 555) | ((ts.dt.hour * 60 + ts.dt.minute) > 930)).sum()),
        "weekend_rows": int((ts.dt.dayofweek >= 5).sum()),
        "date_field_mismatches": int((date_from_field.dt.date != ts.dt.date).sum()),
        "tick_lag_seconds_counts": {str(k): int(v) for k, v in tick_lag.value_counts(dropna=False).head(10).items()},
        "dte_abs_error_over_1_second": int(((recomputed_dte - pd.to_numeric(df["days_to_expiry"], errors="coerce")).abs() > 1 / 86400).sum()),
        "nonpositive_call_price": int((pd.to_numeric(df["call_price"], errors="coerce") <= 0).sum()),
        "nonpositive_put_price": int((pd.to_numeric(df["put_price"], errors="coerce") <= 0).sum()),
        "negative_call_oi": int((pd.to_numeric(df["call_oi"], errors="coerce") < 0).sum()),
        "negative_put_oi": int((pd.to_numeric(df["put_oi"], errors="coerce") < 0).sum()),
        "invalid_call_iv": int(((pd.to_numeric(df["iv_call"], errors="coerce") <= 0) | ~pd.to_numeric(df["iv_call"], errors="coerce").map(pd.notna)).sum()),
        "invalid_put_iv": int(((pd.to_numeric(df["iv_put"], errors="coerce") <= 0) | ~pd.to_numeric(df["iv_put"], errors="coerce").map(pd.notna)).sum()),
        "call_iv_not_converged": int((df["iv_call_converged"].astype(str).str.lower() != "true").sum()),
        "put_iv_not_converged": int((df["iv_put_converged"].astype(str).str.lower() != "true").sum()),
        "is_monthly_not_true": int((df["is_monthly"].astype(str).str.lower() != "true").sum()),
        "is_weekly_true": int((df["is_weekly"].astype(str).str.lower() == "true").sum()),
        "expiry_type_counts": {str(k): int(v) for k, v in df["expiry_type"].value_counts(dropna=False).items()},
        "dte_flag_counts": {str(k): int(v) for k, v in df["dte_flag"].value_counts(dropna=False).items()},
        "snapshot_strike_count_counts": {str(k): int(v) for k, v in snapshot["strikes"].value_counts().sort_index().items()},
        "snapshot_stated_count_mismatches": int((snapshot["strikes"] != snapshot["stated_available"]).sum()),
        "timestamps_with_exact_spot_match": int(exact_matches),
        "timestamps_without_exact_spot_match": int(len(exact_ts) - exact_matches),
    }
    daily = ts.groupby(ts.dt.normalize()).nunique().rename("option_timestamps").reset_index()
    daily.columns = ["date", "option_timestamps"]
    return metrics, expiries, daily


def audit_contract_master(path: Path) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(path, compression="gzip", low_memory=False)
    bank = df[df["TckrSymb"].astype(str).str.upper().eq("BANKNIFTY")].copy()
    cols = ["FinInstrmNm", "TckrSymb", "XpryDt", "StrkPric", "OptnTp", "MinLot", "NewBrdLotQty", "StartDt", "EndDt", "MtrtyDt", "LastUpd"]
    available = [c for c in cols if c in bank.columns]
    sample = bank[available].drop_duplicates().sort_values(available[:2]).head(5000)
    metrics = {
        "all_rows": int(len(df)),
        "banknifty_rows": int(len(bank)),
        "banknifty_instrument_names": {str(k): int(v) for k, v in bank["FinInstrmNm"].value_counts(dropna=False).items()},
        "banknifty_min_lot_counts": {str(k): int(v) for k, v in bank["MinLot"].value_counts(dropna=False).items()},
        "banknifty_board_lot_counts": {str(k): int(v) for k, v in bank["NewBrdLotQty"].value_counts(dropna=False).items()},
    }
    return metrics, sample


def main() -> None:
    spot_path = RAW / "icharts_banknifty_spot_1min.csv"
    monthly_path = RAW / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
    master_path = RAW / "NSE_FO_contract_06102026.csv.gz"
    spot, spot_daily = audit_spot(spot_path)
    spot_ts = set(pd.to_datetime(pd.read_csv(spot_path, usecols=["datetime"])["datetime"], errors="coerce").dropna())
    monthly, expiries, option_daily = audit_monthly(monthly_path, spot_ts)
    contracts, contract_sample = audit_contract_master(master_path)
    files = [
        RAW / "icharts_banknifty_spot_1min.csv",
        RAW / "icharts_option_banknifty.csv",
        RAW / "icharts_option_banknifty_monthly_atm.csv",
        RAW / "icharts_option_banknifty_monthly_atm_5strikes.csv",
        monthly_path,
        master_path,
    ]
    result = {"files": [file_identity(p) for p in files], "spot": spot, "monthly_up_to_5strikes": monthly, "contract_master": contracts}
    (OUT / "audit_summary.json").write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    spot_daily.to_csv(OUT / "spot_daily_coverage.csv", index=False)
    option_daily.to_csv(OUT / "option_daily_coverage.csv", index=False)
    expiries.to_csv(OUT / "monthly_expiry_inventory.csv", index=False)
    contract_sample.to_csv(OUT / "contract_master_banknifty_sample.csv", index=False)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
