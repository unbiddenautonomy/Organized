"""Apply intraday net-P&L target/stop pairs to the final model signals.

The exact entry expiry and strike are held fixed. Open P&L is checked at every
available source-data minute. Brokerage is zero; statutory charges and two
straddle points of round-trip slippage are included before a target or stop is
considered hit. A position exits at the first target/stop crossing, its model-
selected 60/120-minute time exit, or the 15:14:30 hard cutoff. Only one position
may be open; an early exit can therefore admit a later valid signal.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_expiry_week_shared_stack as metrics_lib
import train_win_classifier as data_lib


INPUT = HERE / "output" / "zero_week_focus_stack_zero_brokerage"
OUTPUT = HERE / "output" / "zero_week_focus_target_stop"
RAW_CACHE = HERE / "output" / "win_classifier" / "all_m1_rows_features_v2.pkl"
PAIRS = ((1500.0, -750.0), (3000.0, -1500.0))
HARD_CUTOFF = pd.Timedelta(hours=15, minutes=14, seconds=30)


@dataclass
class ExpirySurface:
    times: np.ndarray
    surfaces: list[tuple[np.ndarray, np.ndarray, float]]


def build_surfaces(raw: pd.DataFrame, expiries: set[pd.Timestamp]) -> dict[pd.Timestamp, ExpirySurface]:
    work = raw.loc[
        raw["expiry"].isin(expiries)
        & raw["datetime"].ge(pd.Timestamp("2024-01-01")),
        ["datetime", "expiry", "strike", "call_price", "put_price", "future_open"],
    ].copy()
    work["straddle"] = pd.to_numeric(work["call_price"], errors="coerce") + pd.to_numeric(
        work["put_price"], errors="coerce"
    )
    result: dict[pd.Timestamp, ExpirySurface] = {}
    for expiry, expiry_rows in work.groupby("expiry", sort=True):
        times: list[np.datetime64] = []
        values: list[tuple[np.ndarray, np.ndarray, float]] = []
        for timestamp, group in expiry_rows.groupby("datetime", sort=True):
            future = pd.to_numeric(group["future_open"], errors="coerce").dropna()
            if future.empty:
                continue
            times.append(np.datetime64(pd.Timestamp(timestamp), "ns"))
            values.append((
                pd.to_numeric(group["strike"], errors="coerce").to_numpy(float),
                pd.to_numeric(group["straddle"], errors="coerce").to_numpy(float),
                float(future.iloc[-1]),
            ))
        result[pd.Timestamp(expiry)] = ExpirySurface(np.asarray(times, dtype="datetime64[ns]"), values)
    return result


def premium_path(
    surface: ExpirySurface,
    entry: pd.Timestamp,
    scheduled_exit: pd.Timestamp,
    cutoff: pd.Timestamp,
    strike: float,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    start = int(np.searchsorted(surface.times, np.datetime64(entry, "ns"), side="right"))
    stop = int(np.searchsorted(surface.times, np.datetime64(cutoff, "ns"), side="right"))
    times = surface.times[start:stop]
    premiums: list[float] = []
    synthetic: list[bool] = []
    for strikes, values, future in surface.surfaces[start:stop]:
        premium, was_synthetic = data_lib._linear_straddle_value(strikes, values, strike, future)
        premiums.append(premium)
        synthetic.append(was_synthetic)
        if times[len(premiums) - 1] >= np.datetime64(scheduled_exit, "ns"):
            break
    return (
        pd.DatetimeIndex(times[: len(premiums)]),
        np.asarray(premiums, dtype=float),
        np.asarray(synthetic, dtype=bool),
    )


def simulate_trade(
    row: pd.Series,
    entry: pd.Timestamp,
    surfaces: dict[pd.Timestamp, ExpirySurface],
    target: float,
    stop: float,
) -> dict | None:
    expiry = pd.Timestamp(row["expiry"])
    if expiry not in surfaces:
        return None
    horizon = int(row["horizon_minutes"])
    scheduled = entry + pd.Timedelta(minutes=horizon)
    cutoff = entry.normalize() + HARD_CUTOFF
    final_limit = min(scheduled, cutoff)
    times, premiums, synthetic = premium_path(
        surfaces[expiry], entry, scheduled, cutoff, float(row["strike"])
    )
    if not len(times):
        return None
    entry_premium = float(row["call_price"] + row["put_price"])
    lot = float(data_lib.lot_size(pd.DatetimeIndex([entry.normalize()]))[0])
    dates = pd.DatetimeIndex(np.repeat(entry.normalize().to_datetime64(), len(times)))
    entry_values = np.full(len(times), entry_premium)
    lots = np.full(len(times), lot)
    statutory = data_lib.option_cost_rupees(entry_values, premiums, dates, lots)
    slippage = 2.0 * lot
    gross_points = entry_premium - premiums
    gross_rupees = gross_points * lot
    net_rupees = gross_rupees - statutory - slippage

    before_or_at_exit = times <= final_limit
    eligible = np.flatnonzero(before_or_at_exit)
    if not len(eligible):
        eligible = np.array([0])
    target_hits = np.flatnonzero((net_rupees >= target) & before_or_at_exit)
    stop_hits = np.flatnonzero((net_rupees <= stop) & before_or_at_exit)
    first_target = int(target_hits[0]) if len(target_hits) else None
    first_stop = int(stop_hits[0]) if len(stop_hits) else None
    if first_target is not None and (first_stop is None or first_target < first_stop):
        position, reason = first_target, "TARGET"
    elif first_stop is not None:
        position, reason = first_stop, "STOP"
    else:
        at_or_after = np.flatnonzero(times >= final_limit)
        if len(at_or_after):
            position = int(at_or_after[0])
        else:
            position = int(eligible[-1])
        reason = "CUTOFF" if cutoff <= scheduled and times[position] >= cutoff else "TIME"

    return {
        "entry_datetime": entry,
        "exit_datetime": times[position],
        "scheduled_exit_datetime": scheduled,
        "hard_cutoff_datetime": cutoff,
        "expiry": expiry,
        "strike": float(row["strike"]),
        "expiry_group": row["calibration_group"],
        "horizon_minutes": horizon,
        "exit_reason": reason,
        "entry_premium": entry_premium,
        "exit_premium": float(premiums[position]),
        "synthetic_exit": bool(synthetic[position]),
        "lot_size": int(lot),
        "gross_points": float(gross_points[position]),
        "gross_rupees": float(gross_rupees[position]),
        "statutory_cost_rupees": float(statutory[position]),
        "slippage_rupees": float(slippage),
        "brokerage_rupees": 0.0,
        "total_cost_rupees": float(statutory[position] + slippage),
        "net_points": float(net_rupees[position] / lot),
        "net_rupees": float(net_rupees[position]),
        "target_rupees": target,
        "stop_rupees": stop,
        "win_probability": float(row["win_probability"]),
        "large_loss_probability": float(row["large_loss_probability"]),
        "predicted_decay_60_points": float(row["predicted_decay_60_points"]),
    }


def simulate_pair(
    signals: pd.DataFrame,
    surfaces: dict[pd.Timestamp, ExpirySurface],
    target: float,
    stop: float,
) -> tuple[pd.DataFrame, int, int]:
    records: list[dict] = []
    active_until = pd.Timestamp.min
    rejected_overlap = 0
    missing_path = 0
    for entry, row in signals.sort_index().iterrows():
        entry = pd.Timestamp(entry)
        if entry < active_until:
            rejected_overlap += 1
            continue
        record = simulate_trade(row, entry, surfaces, target, stop)
        if record is None:
            missing_path += 1
            continue
        records.append(record)
        active_until = pd.Timestamp(record["exit_datetime"])
    return pd.DataFrame.from_records(records), rejected_overlap, missing_path


def scenario_summary(
    trades: pd.DataFrame,
    market_dates: pd.DatetimeIndex,
    target: float,
    stop: float,
    overlap: int,
    missing: int,
) -> tuple[dict, pd.DataFrame]:
    daily = (
        trades.assign(trade_date=pd.to_datetime(trades["entry_datetime"]).dt.normalize())
        .groupby("trade_date")["net_rupees"].sum()
        .reindex(market_dates, fill_value=0.0)
        .rename_axis("date").reset_index()
    )
    normalized = trades.rename(columns={"statutory_cost_rupees": "cost_rupees"}).copy()
    summary = metrics_lib.aggregate_metrics(normalized, daily)
    summary.update({
        "target_rupees": target,
        "stop_rupees": stop,
        "brokerage_rupees": 0.0,
        "target_exits": int(trades["exit_reason"].eq("TARGET").sum()),
        "stop_exits": int(trades["exit_reason"].eq("STOP").sum()),
        "time_exits": int(trades["exit_reason"].eq("TIME").sum()),
        "cutoff_exits": int(trades["exit_reason"].eq("CUTOFF").sum()),
        "overlap_rejections": overlap,
        "missing_monitoring_paths": missing,
    })
    return summary, daily


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(
        INPUT / "all_test_candidates.csv.gz",
        parse_dates=["datetime", "expiry", "exit_datetime"],
    ).set_index("datetime").sort_index()
    signals = candidates.loc[candidates["all_gates_pass"].astype(bool)].copy()
    raw = pd.read_pickle(RAW_CACHE)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["expiry"] = pd.to_datetime(raw["expiry"])
    print(f"Building minute surfaces for {signals['expiry'].nunique()} expiries", flush=True)
    surfaces = build_surfaces(raw, set(pd.to_datetime(signals["expiry"]).unique()))
    market_dates = pd.DatetimeIndex(sorted(candidates.index.normalize().unique()))

    summaries: list[dict] = []
    comparison_rows: list[dict] = []
    for target, stop in PAIRS:
        print(f"Simulating target {target:.0f}, stop {stop:.0f}", flush=True)
        trades, overlap, missing = simulate_pair(signals, surfaces, target, stop)
        summary, daily = scenario_summary(trades, market_dates, target, stop, overlap, missing)
        summaries.append(summary)
        label = f"target_{int(target)}_stop_{abs(int(stop))}"
        folder = OUTPUT / label
        folder.mkdir(parents=True, exist_ok=True)
        trades.to_csv(folder / "trades.csv", index=False)
        daily.to_csv(folder / "daily_pnl.csv", index=False)
        pd.DataFrame([summary]).to_csv(folder / "summary.csv", index=False)
        comparison_rows.append({
            "scenario": label,
            "trades": summary["trades"],
            "win_rate_pct": summary["win_rate_pct"],
            "target_exits": summary["target_exits"],
            "stop_exits": summary["stop_exits"],
            "time_exits": summary["time_exits"],
            "net_rupees": summary["net_rupees"],
            "profit_factor": summary["profit_factor"],
            "max_drawdown_rupees": summary["max_drawdown_rupees"],
            "daily_sharpe": summary["daily_sharpe"],
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT / "comparison.csv", index=False)
    pd.DataFrame(summaries).to_csv(OUTPUT / "all_summaries.csv", index=False)
    report = [
        "# BANKNIFTY target/stop comparison", "",
        "Targets and stops are net rupee P&L per position. Open P&L is checked at every available source-data minute using the exact entry expiry and strike. Brokerage is zero; statutory charges and two straddle points of slippage are included before trigger evaluation.", "",
        "Entries use the final zero-week-focused model gates. Only one position may be open, and early exits can admit later qualifying signals. A trade exits at the first target/stop crossing, its 60/120-minute model time exit, or the 15:14:30 hard cutoff.", "",
        markdown_table(comparison), "",
        "Targets/stops may be exceeded between minute snapshots because exits occur at the first observed stored surface, not at an assumed exact fill at the threshold.", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    (OUTPUT / "run_metadata.json").write_text(json.dumps({
        "signal_source": str(INPUT / "all_test_candidates.csv.gz"),
        "target_stop_pairs_rupees": PAIRS,
        "trigger_basis": "net PnL after statutory costs and two straddle points slippage",
        "brokerage_rupees": 0.0,
        "monitoring": "every available source-data minute",
        "hard_cutoff": "15:14:30",
        "one_open_position": True,
        "same_entry_expiry_and_strike": True,
    }, indent=2), encoding="utf-8")
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
