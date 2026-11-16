from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REPO = Path(__file__).resolve().parents[2]
RAW_DIR = Path(os.environ.get("VOLTR_RAW_DIR", REPO / "research" / "v2" / "data"))
USER_CSV = RAW_DIR / "icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
SPOT_CSV = RAW_DIR / "icharts_banknifty_spot_1min.csv"
FULL_CHAIN_CSV = RAW_DIR / "icharts_option_banknifty.csv"
PROJECT = Path(os.environ.get("VOLTR_V2_FEATURE_ROOT", REPO / "research" / "v2" / "research"))
INFERENCE = Path(__file__).resolve().parent.parent / "families" / "ivrv" / 'engine' / 'Inference'
OUT_DIR = Path(__file__).resolve().parent / 'analysis_outputs' / 'paper_rules_3000_1500'

START_DATE = "2024-09-19"
END_DATE = "2025-05-09"
TARGET_INR = 3000.0
STOP_INR = -1500.0
LOT_SIZE = 30
MAX_OPEN_POSITIONS = 3
HORIZON_MINUTES = 120
CONFIDENCE_THRESHOLD = 0.90
SQUARE_OFF = time(15, 14, 30)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def resolved_yaml(path: Path, keys: tuple[str, ...]) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in keys:
        config["data"][key] = str((path.parent / config["data"][key]).resolve())
    return config


def read_option_days(usecols: list[str]):
    carry = pd.DataFrame()
    for chunk in pd.read_csv(USER_CSV, usecols=usecols, chunksize=250_000, low_memory=False):
        chunk["Date"] = chunk["Date"].astype(str)
        chunk = chunk.loc[chunk["Date"].between(START_DATE, END_DATE)]
        if chunk.empty:
            continue
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_date = chunk["Date"].iloc[-1]
        complete = chunk.loc[chunk["Date"] != last_date]
        carry = chunk.loc[chunk["Date"] == last_date].copy()
        for _, day in complete.groupby("Date", sort=False):
            yield day
    if not carry.empty:
        yield carry


def build_inference_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    combined_module = load_module(
        "combined_feature_pipeline_oos",
        PROJECT / "combined_atm_iv_rv_convergence_v2" / "feature_pipeline.py",
    )
    rv_module = load_module(
        "rv_feature_pipeline_oos",
        PROJECT / "fixed_strike_iv_rv_convergence" / "feature_pipeline.py",
    )
    fixed_root = PROJECT / "fixed_strike_iv_120m"
    sys.path.insert(0, str(fixed_root))
    from fixed_strike_iv.pipeline import ENTRY_COLUMNS, _add_lag_features
    from fixed_strike_iv.surface import make_entry_rows

    combined_config = resolved_yaml(
        PROJECT / "combined_atm_iv_rv_convergence_v2" / "config.yaml",
        ("options_csv", "spot_csv", "output_csv"),
    )
    rv_config = resolved_yaml(
        PROJECT / "fixed_strike_iv_rv_convergence" / "config.yaml",
        ("options_csv", "spot_csv", "output_csv"),
    )
    spot_raw = pd.read_csv(SPOT_CSV, usecols=["datetime", "spot_price"])
    combined_spot = combined_module.construct_spot_features(spot_raw.copy(), combined_config)
    rv_spot = rv_module.construct_spot_features(spot_raw.copy(), rv_config)

    option_columns = sorted(
        set(combined_module.OPTION_COLUMNS)
        | set(rv_module.OPTION_COLUMNS)
        | set(ENTRY_COLUMNS)
    )
    combined_days: list[pd.DataFrame] = []
    rv_days: list[pd.DataFrame] = []
    leg_days: list[pd.DataFrame] = []
    for day in read_option_days(option_columns):
        combined = combined_module.construct_option_day(
            day[list(combined_module.OPTION_COLUMNS)].copy(), combined_spot, combined_config
        )
        rv = rv_module.construct_option_day(
            day[list(rv_module.OPTION_COLUMNS)].copy(), rv_spot, rv_config
        )
        legs = make_entry_rows(
            day[ENTRY_COLUMNS].copy(),
            int(combined_config["construction"]["minimum_surface_points"]),
            "future_open",
        )
        if not combined.empty:
            combined_days.append(combined)
        if not rv.empty:
            rv_days.append(rv)
        if not legs.empty:
            leg_days.append(legs)

    combined = pd.concat(combined_days, ignore_index=True)
    rv = pd.concat(rv_days, ignore_index=True)
    legs = pd.concat(leg_days, ignore_index=True)
    legs["datetime"] = pd.to_datetime(legs["datetime"])
    legs["expiry"] = pd.to_datetime(legs["expiry"])
    legs = _add_lag_features(legs, [5, 15, 30, 60, 120])
    legs = legs.sort_values(["datetime", "dte_flag", "expiry"], kind="stable").reset_index(drop=True)
    return combined, legs, rv


def make_predictions() -> pd.DataFrame:
    combined, legs, rv = build_inference_frames()
    sys.path.insert(0, str(INFERENCE))
    from inference_pipeline import FixedStrikeInferencePipeline

    keys = ["datetime", "dte_flag", "expiry", "entry_strike"]
    for frame in (combined, legs, rv):
        frame["datetime"] = pd.to_datetime(frame["datetime"])
        frame["expiry"] = pd.to_datetime(frame["expiry"])
    common = combined[keys].merge(legs[keys], on=keys, how="inner").merge(rv[keys], on=keys, how="inner")
    common = common.drop_duplicates(keys).sort_values(keys, kind="stable").reset_index(drop=True)
    combined = common.merge(combined, on=keys, how="left", validate="one_to_one")
    legs = common.merge(legs, on=keys, how="left", validate="one_to_one")
    rv = common.merge(rv, on=keys, how="left", validate="one_to_one")

    predictions = FixedStrikeInferencePipeline().predict(combined, legs, rv)
    keep = [
        "quality_entry_ok", "entry_call_price", "entry_put_price", "entry_straddle_price",
        "entry_spot_price", "entry_future_price",
    ]
    available = [name for name in keep if name in combined.columns]
    predictions = pd.concat(
        [predictions.reset_index(drop=True), combined[available].reset_index(drop=True)], axis=1
    )
    minute = predictions["datetime"].dt.hour * 60 + predictions["datetime"].dt.minute
    predictions["live_feature_ready"] = minute >= (9 * 60 + 15 + 120)
    predictions["entry_before_cutoff"] = predictions["datetime"].dt.time < SQUARE_OFF
    predictions["base_eligible"] = (
        predictions.get("quality_entry_ok", pd.Series(True, index=predictions.index)).fillna(False).astype(bool)
        & predictions["live_feature_ready"]
        & predictions["entry_before_cutoff"]
    )
    return predictions


def load_established_test_predictions() -> pd.DataFrame:
    path = INFERENCE / "artifacts" / "current_pipeline_test_predictions_with_pnl.csv"
    predictions = pd.read_csv(path, parse_dates=["datetime", "expiry"], low_memory=False)
    # The live runtime trades one selected (nearest) expiry. The research table can
    # contain two monthly expiries at the same timestamp, so retain the nearest one.
    nearest = predictions.groupby("datetime")["expiry"].transform("min")
    predictions = predictions.loc[predictions["expiry"].eq(nearest)].copy()
    minute = predictions["datetime"].dt.hour * 60 + predictions["datetime"].dt.minute
    predictions["live_feature_ready"] = minute >= (9 * 60 + 15 + 120)
    predictions["entry_before_cutoff"] = predictions["datetime"].dt.time < SQUARE_OFF
    predictions["base_eligible"] = (
        predictions["quality_trainable"].fillna(False).astype(bool)
        & predictions["live_feature_ready"]
        & predictions["entry_before_cutoff"]
    )
    return predictions.reset_index(drop=True)


def locate_date_offset(path: Path, target: str) -> int:
    size = path.stat().st_size
    with path.open("rb") as handle:
        low, high = 0, size
        while high - low > 1_000_000:
            mid = (low + high) // 2
            handle.seek(mid)
            handle.readline()
            position = handle.tell()
            line = handle.readline()
            if not line:
                high = mid
                continue
            parts = line.split(b",", 3)
            stamp = parts[1][:10].decode("ascii", errors="ignore") if len(parts) > 1 else ""
            if stamp < target:
                low = position
            else:
                high = mid
        handle.seek(max(low, 0))
        if low:
            handle.readline()
        while True:
            position = handle.tell()
            line = handle.readline()
            if not line:
                return position
            parts = line.split(b",", 3)
            if len(parts) > 1 and parts[1][:10].decode("ascii", errors="ignore") >= target:
                return position


def load_relevant_prices(predictions: pd.DataFrame) -> pd.DataFrame:
    wanted = {
        (str(row.Date), row.dte_flag, str(row.expiry.date()), float(row.entry_strike))
        for row in predictions.loc[predictions["base_eligible"]].itertuples()
    }
    base_rows: list[pd.DataFrame] = []
    usecols = ["datetime", "dte_flag", "strike", "expiry", "Date", "call_price", "put_price"]
    for chunk in pd.read_csv(USER_CSV, usecols=usecols, chunksize=500_000, low_memory=False):
        chunk["Date"] = chunk["Date"].astype(str)
        chunk = chunk.loc[chunk["Date"].between(START_DATE, END_DATE)]
        if chunk.empty:
            continue
        mask = [
            (str(date_value), dte, str(expiry), float(strike)) in wanted
            for date_value, dte, expiry, strike in chunk[["Date", "dte_flag", "expiry", "strike"]].itertuples(index=False, name=None)
        ]
        selected = chunk.loc[mask, usecols].copy()
        if not selected.empty:
            base_rows.append(selected)

    header = FULL_CHAIN_CSV.open("r", encoding="utf-8", newline="").readline().rstrip("\r\n").split(",")
    needed = ["datetime", "dte_flag", "strike", "expiry", "Date", "call_price", "put_price"]
    indices = {name: header.index(name) for name in needed}
    rows: list[dict] = []
    wanted_dates = sorted({key[0] for key in wanted})
    for wanted_date in wanted_dates:
        offset = locate_date_offset(FULL_CHAIN_CSV, wanted_date)
        with FULL_CHAIN_CSV.open("r", encoding="utf-8", newline="") as handle:
            handle.seek(offset)
            reader = csv.reader(handle)
            for fields in reader:
                if len(fields) <= max(indices.values()):
                    continue
                date_value = fields[indices["Date"]]
                if date_value > wanted_date:
                    break
                if date_value < wanted_date:
                    continue
                try:
                    strike = float(fields[indices["strike"]])
                except ValueError:
                    continue
                key = (
                    date_value,
                    fields[indices["dte_flag"]],
                    fields[indices["expiry"]],
                    strike,
                )
                if key not in wanted:
                    continue
                try:
                    call_price = float(fields[indices["call_price"]])
                    put_price = float(fields[indices["put_price"]])
                except ValueError:
                    continue
                if math.isfinite(call_price) and math.isfinite(put_price):
                    rows.append({
                        "datetime": fields[indices["datetime"]],
                        "dte_flag": key[1],
                        "strike": strike,
                        "expiry": key[2],
                        "call_price": call_price,
                        "put_price": put_price,
                    })
    exact = pd.DataFrame(rows)
    prices = pd.concat([*base_rows, exact], ignore_index=True)
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    prices["expiry"] = pd.to_datetime(prices["expiry"])
    prices["straddle_price"] = prices["call_price"] + prices["put_price"]
    prices = prices.drop_duplicates(["datetime", "dte_flag", "expiry", "strike"], keep="last")
    return prices.sort_values(["datetime", "dte_flag", "expiry", "strike"], kind="stable")


def simulate(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    use_confidence: bool,
    maximum_open_positions: int | None = MAX_OPEN_POSITIONS,
) -> tuple[pd.DataFrame, dict]:
    pred = predictions.loc[predictions["base_eligible"]].copy()
    pred["signal"] = True
    if use_confidence:
        pred["signal"] = pred["pred_convergence_probability_classifier"].astype(float) > CONFIDENCE_THRESHOLD
    pred_keyed = {
        (row.datetime, row.dte_flag, row.expiry, float(row.entry_strike)): row
        for row in pred.loc[pred["signal"]].itertuples()
    }
    timeline: dict[pd.Timestamp, list] = {}
    for key, row in pred_keyed.items():
        timeline.setdefault(key[0], []).append(row)
    prices_by_time = {stamp: group for stamp, group in prices.groupby("datetime", sort=True)}
    cutoff_timestamps = {
        pd.Timestamp(date_value) + pd.Timedelta(54899, unit="s")
        for date_value in pred["datetime"].dt.strftime("%Y-%m-%d").unique()
    }
    timestamps = sorted(set(timeline) | set(prices_by_time) | cutoff_timestamps)
    open_positions: list[dict] = []
    trades: list[dict] = []
    skipped_cap = 0
    skipped_price = 0

    for stamp in timestamps:
        marks = prices_by_time.get(stamp)
        if open_positions:
            mark_map = {
                (row.dte_flag, row.expiry, float(row.strike)): row
                for row in marks.itertuples()
            } if marks is not None else {}
            survivors = []
            for position in open_positions:
                mark = mark_map.get((position["dte_flag"], position["expiry"], position["strike"]))
                if mark is not None:
                    position["last_straddle_price"] = float(mark.straddle_price)
                if mark is None and stamp.time() < SQUARE_OFF:
                    survivors.append(position)
                    continue
                exit_price = float(position["last_straddle_price"])
                pnl = (position["entry_straddle_price"] - exit_price) * LOT_SIZE
                elapsed = (stamp - position["entry_datetime"]).total_seconds() / 60.0
                reason = None
                if stamp.time() >= SQUARE_OFF:
                    reason = "session_square_off_151430"
                elif pnl >= TARGET_INR:
                    reason = "profit_target"
                elif pnl <= STOP_INR:
                    reason = "stop_loss"
                elif elapsed >= HORIZON_MINUTES:
                    reason = "time_exit_120m"
                if reason is None:
                    survivors.append(position)
                else:
                    trades.append({
                        **position,
                        "exit_datetime": stamp,
                        "exit_straddle_price": exit_price,
                        "realized_pnl_inr": pnl,
                        "holding_minutes": elapsed,
                        "exit_reason": reason,
                    })
            open_positions = survivors

        for signal in timeline.get(stamp, []):
            if maximum_open_positions is not None and len(open_positions) >= maximum_open_positions:
                skipped_cap += 1
                continue
            marks = prices_by_time.get(stamp)
            if marks is None:
                skipped_price += 1
                continue
            match = marks.loc[
                marks["dte_flag"].eq(signal.dte_flag)
                & marks["expiry"].eq(signal.expiry)
                & marks["strike"].eq(float(signal.entry_strike))
            ]
            if match.empty:
                skipped_price += 1
                continue
            mark = match.iloc[-1]
            open_positions.append({
                "signal_datetime": stamp,
                "entry_datetime": stamp,
                "dte_flag": signal.dte_flag,
                "expiry": signal.expiry,
                "strike": float(signal.entry_strike),
                "entry_straddle_price": float(mark["straddle_price"]),
                "last_straddle_price": float(mark["straddle_price"]),
                "convergence_probability": float(signal.pred_convergence_probability_classifier),
            })

    if open_positions:
        raise RuntimeError(
            f"{len(open_positions)} positions remained open after the final price timestamp: {open_positions}"
        )
    ledger = pd.DataFrame(trades)
    pnl = ledger["realized_pnl_inr"].astype(float) if not ledger.empty else pd.Series(dtype=float)
    equity = pnl.cumsum()
    drawdown = equity - equity.cummax().clip(lower=0)
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    summary = {
        "scenario": (
            f"with_probability_gt_{CONFIDENCE_THRESHOLD:.2f}"
            if use_confidence else "without_probability_filter"
        ),
        "target_inr": TARGET_INR,
        "stop_loss_inr": STOP_INR,
        "maximum_open_positions": maximum_open_positions,
        "trades": int(len(ledger)),
        "gross_pnl_inr": float(pnl.sum()),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "average_pnl_inr": float(pnl.mean()) if len(pnl) else None,
        "profit_factor": gains / losses if losses else None,
        "maximum_drawdown_inr": float(drawdown.min()) if len(drawdown) else 0.0,
        "exit_reasons": dict(Counter(ledger["exit_reason"])) if not ledger.empty else {},
        "qualified_signals": int(pred["signal"].sum()),
        "signals_skipped_at_position_cap": skipped_cap,
        "signals_skipped_missing_entry_price": skipped_price,
    }
    return ledger, summary


def main() -> None:
    global TARGET_INR, STOP_INR, CONFIDENCE_THRESHOLD, MAX_OPEN_POSITIONS, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--target", type=float, default=TARGET_INR)
    parser.add_argument("--stop", type=float, default=STOP_INR)
    parser.add_argument("--confidence-thresholds", type=float, nargs="+", default=[CONFIDENCE_THRESHOLD])
    parser.add_argument("--max-open-positions", type=int, default=MAX_OPEN_POSITIONS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--uncapped-confidence", action="store_true")
    args = parser.parse_args()
    TARGET_INR = float(args.target)
    STOP_INR = float(args.stop)
    MAX_OPEN_POSITIONS = int(args.max_open_positions)
    OUT_DIR = args.output_dir.resolve()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = OUT_DIR / "established_test_predictions_nearest_expiry.csv.gz"
    predictions = load_established_test_predictions()
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    predictions["Date"] = predictions["datetime"].dt.strftime("%Y-%m-%d")
    prices = load_relevant_prices(predictions)
    summaries = []
    ledger, summary = simulate(predictions, prices, False, MAX_OPEN_POSITIONS)
    ledger.to_csv(OUT_DIR / "trades_without_confidence_filter.csv", index=False)
    summaries.append(summary)
    for threshold in args.confidence_thresholds:
        CONFIDENCE_THRESHOLD = float(threshold)
        confidence_cap = None if args.uncapped_confidence else MAX_OPEN_POSITIONS
        ledger, summary = simulate(predictions, prices, True, confidence_cap)
        stem = f"{int(round(CONFIDENCE_THRESHOLD * 100))}pct"
        ledger.to_csv(OUT_DIR / f"trades_with_{stem}_filter.csv", index=False)
        summaries.append(summary)
    payload = {
        "source_csv": str(USER_CSV),
        "evaluation_window": {"start": START_DATE, "end": END_DATE},
        "evaluation_status": "reused held-out diagnostic; not a fresh post-training holdout",
        "accounting": "gross before costs and slippage",
        "mark_price_source": str(FULL_CHAIN_CSV),
        "historical_entry_assumption": "entry at the signal minute's recorded fixed-strike prices",
        "constraints": {
            "fixed_entry_strike_and_expiry": True,
            "one_lot_quantity": LOT_SIZE,
            "maximum_open_positions": MAX_OPEN_POSITIONS,
            "confidence_filtered_maximum_open_positions": (
                None if args.uncapped_confidence else MAX_OPEN_POSITIONS
            ),
            "profit_target_inr": TARGET_INR,
            "stop_loss_inr": STOP_INR,
            "maximum_holding_minutes": HORIZON_MINUTES,
            "square_off_time": SQUARE_OFF.isoformat(),
            "confidence_rules": [
                f"strictly greater than {float(value):.2f}"
                for value in args.confidence_thresholds
            ],
        },
        "prediction_rows": int(len(predictions)),
        "base_eligible_rows": int(predictions["base_eligible"].sum()),
        "scenarios": summaries,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
