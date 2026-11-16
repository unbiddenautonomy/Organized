"""Backtest the final signals with TP 3000, SL -1500, and max three open."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_straddle_target_stop_pairs as engine


INPUT = HERE / "output" / "zero_week_focus_stack_zero_brokerage"
OUTPUT = HERE / "output" / "target_3000_stop_1500_max3"
TARGET = 3000.0
STOP = -1500.0
MAX_OPEN = 3


def simulate_max_open(
    signals: pd.DataFrame,
    surfaces: dict,
) -> tuple[pd.DataFrame, int, int]:
    records: list[dict] = []
    active_exits: list[pd.Timestamp] = []
    rejected_capacity = 0
    missing_path = 0
    for entry, row in signals.sort_index().iterrows():
        entry = pd.Timestamp(entry)
        active_exits = [exit_time for exit_time in active_exits if exit_time > entry]
        if len(active_exits) >= MAX_OPEN:
            rejected_capacity += 1
            continue
        record = engine.simulate_trade(row, entry, surfaces, TARGET, STOP)
        if record is None:
            missing_path += 1
            continue
        records.append(record)
        active_exits.append(pd.Timestamp(record["exit_datetime"]))
    return pd.DataFrame.from_records(records), rejected_capacity, missing_path


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(
        INPUT / "all_test_candidates.csv.gz",
        parse_dates=["datetime", "expiry", "exit_datetime"],
    ).set_index("datetime").sort_index()
    signals = candidates.loc[candidates["all_gates_pass"].astype(bool)].copy()
    raw = pd.read_pickle(engine.RAW_CACHE)
    raw["datetime"] = pd.to_datetime(raw["datetime"])
    raw["expiry"] = pd.to_datetime(raw["expiry"])
    print(f"Building minute surfaces for {signals['expiry'].nunique()} expiries", flush=True)
    surfaces = engine.build_surfaces(raw, set(pd.to_datetime(signals["expiry"]).unique()))
    trades, rejected, missing = simulate_max_open(signals, surfaces)
    market_dates = pd.DatetimeIndex(sorted(candidates.index.normalize().unique()))
    summary, daily = engine.scenario_summary(
        trades, market_dates, TARGET, STOP, rejected, missing
    )
    summary["maximum_open_positions"] = MAX_OPEN
    summary["capacity_rejections"] = summary.pop("overlap_rejections")

    trades.to_csv(OUTPUT / "trades.csv", index=False)
    daily.to_csv(OUTPUT / "daily_pnl.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT / "summary.csv", index=False)
    (OUTPUT / "run_metadata.json").write_text(json.dumps({
        "target_rupees_net": TARGET,
        "stop_rupees_net": STOP,
        "maximum_open_positions": MAX_OPEN,
        "capacity_rule": "accept only when fewer than three earlier positions remain open",
        "monitoring": "every available source-data minute",
        "hard_cutoff": "15:14:30",
        "brokerage_rupees": 0.0,
        "statutory_charges_included": True,
        "slippage_points_per_straddle": 2.0,
        "same_entry_expiry_and_strike": True,
    }, indent=2), encoding="utf-8")
    report = [
        "# Target Rs 3,000 / stop -Rs 1,500 / maximum three open", "",
        "A signal is accepted only when fewer than three positions are open. A slot is released immediately when a position exits by target, stop, its model-selected 60/120-minute duration, or the 15:14:30 hard cutoff.", "",
        "Targets and stops use net rupee P&L after statutory charges and two straddle points of slippage. Brokerage is zero. The exact entry expiry and strike are retained.", "",
        f"- Trades: {summary['trades']}",
        f"- Net P&L: Rs {summary['net_rupees']:,.2f}",
        f"- Win rate: {summary['win_rate_pct']:.2f}%",
        f"- Profit factor: {summary['profit_factor']:.3f}",
        f"- Maximum drawdown: Rs {summary['max_drawdown_rupees']:,.2f}",
        f"- Target / stop / time exits: {summary['target_exits']} / {summary['stop_exits']} / {summary['time_exits']}",
        f"- Capacity-rejected signals: {summary['capacity_rejections']}", "",
    ]
    (OUTPUT / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
