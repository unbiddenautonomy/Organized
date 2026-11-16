from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path

import pandas as pd


LOGS = Path(__file__).resolve().parent.parent / "families" / "ivrv" / 'engine' / 'Inference' / 'logs'
OUTPUT_ROOT = Path(__file__).resolve().parent / 'analysis_outputs'
TARGET = 1500.0
STOP = -750.0
LOT_SIZE = 30
MAX_OPEN = 3


def explicit_costs(entry: float, exit_: float) -> dict[str, float]:
    entry_turnover = entry * LOT_SIZE
    exit_turnover = exit_ * LOT_SIZE
    total_turnover = entry_turnover + exit_turnover
    stt = entry_turnover * 0.0015
    nse = total_turnover * 0.0003553
    sebi = total_turnover * 0.000001
    stamp = exit_turnover * 0.00003
    gst = 0.18 * (nse + sebi)
    total = stt + nse + sebi + stamp + gst
    return {"stt": stt, "nse": nse, "sebi": sebi, "stamp": stamp, "gst": gst, "total": total}


def load_signals(threshold: float) -> pd.DataFrame:
    rows = []
    with (LOGS / "live_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            strategy = event.get("strategy_signal", {})
            probability = strategy.get("convergence_probability")
            if probability is None or float(probability) <= threshold:
                continue
            rules = strategy.get("rules", {})
            rows.append({
                "signal_datetime": event["exchange_minute"],
                "strike": float(event["fixed_entry_strike"]),
                "expiry": event["expiry"],
                "probability": float(probability),
                "quality_entry_ok": bool(event.get("market", {}).get("quality_entry_ok")),
                "legacy_three_rule_signal": bool(strategy.get("legacy_three_rule_signal")),
                "rule_above": bool(rules.get("iv_above_rv_and_predicted_iv_declines_to_predicted_rv")),
                "rule_near": bool(rules.get("iv_rv_within_half_volatility_point")),
                "rule_below": bool(rules.get("iv_below_rv_and_fast_rv_decline")),
            })
    frame = pd.DataFrame(rows)
    frame["signal_datetime"] = pd.to_datetime(frame["signal_datetime"])
    return frame.sort_values("signal_datetime", kind="stable").reset_index(drop=True)


def attach_outcomes(signals: pd.DataFrame) -> pd.DataFrame:
    wanted = {(str(row.signal_datetime), row.strike) for row in signals.itertuples()}
    grouped: dict[tuple[str, float], list[dict]] = {key: [] for key in wanted}
    with (LOGS / "fixed_strike_120m_paths.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row.get("entry_datetime")), float(row.get("entry_strike", 0)))
            if key in grouped:
                grouped[key].append(row)
    outcomes = []
    for signal in signals.itertuples():
        key = (str(signal.signal_datetime), signal.strike)
        path = sorted(grouped[key], key=lambda row: int(row["holding_minutes"]))
        if not path:
            raise RuntimeError(f"Missing fixed-strike path for {key}")
        chosen = None
        for row in path:
            stamp = pd.Timestamp(row["observation_datetime"])
            pnl = float(row["short_pnl_per_lot"])
            if stamp.time() >= pd.Timestamp("15:14:30").time():
                reason = "session_square_off_151430"
            elif pnl >= TARGET:
                reason = "profit_target"
            elif pnl <= STOP:
                reason = "stop_loss"
            elif int(row["holding_minutes"]) >= 120:
                reason = "time_exit_120m"
            else:
                continue
            chosen = dict(row)
            chosen["exit_reason"] = reason
            break
        if chosen is None:
            raise RuntimeError(f"No exit found for {key}")
        outcomes.append({
            **signal._asdict(),
            "entry_straddle_price": float(chosen["entry_straddle_price"]),
            "exit_datetime": pd.Timestamp(chosen["observation_datetime"]),
            "exit_straddle_price": float(chosen["observation_straddle_price"]),
            "gross_pnl_inr": float(chosen["short_pnl_per_lot"]),
            "exit_reason": chosen["exit_reason"],
        })
    return pd.DataFrame(outcomes)


def simulate(frame: pd.DataFrame, mask: pd.Series, label: str) -> tuple[pd.DataFrame, dict]:
    candidates = frame.loc[mask].sort_values("signal_datetime", kind="stable")
    open_exits: list[int] = []
    accepted = []
    for row in candidates.itertuples():
        signal_ns = int(pd.Timestamp(row.signal_datetime).value)
        while open_exits and open_exits[0] <= signal_ns:
            heapq.heappop(open_exits)
        if len(open_exits) >= MAX_OPEN:
            continue
        heapq.heappush(open_exits, int(pd.Timestamp(row.exit_datetime).value))
        accepted.append(row._asdict())
    ledger = pd.DataFrame(accepted)
    cost_rows = []
    for row in ledger.itertuples():
        costs = explicit_costs(float(row.entry_straddle_price), float(row.exit_straddle_price))
        cost_rows.append(costs)
    if len(ledger):
        ledger["explicit_costs_inr"] = [row["total"] for row in cost_rows]
        ledger["net_pnl_inr"] = ledger["gross_pnl_inr"] - ledger["explicit_costs_inr"]
    gross = float(ledger["gross_pnl_inr"].sum()) if len(ledger) else 0.0
    costs = float(ledger["explicit_costs_inr"].sum()) if len(ledger) else 0.0
    summary = {
        "scenario": label,
        "qualified_signals": int(mask.sum()),
        "executed_trades": int(len(ledger)),
        "skipped_at_three_position_cap": int(mask.sum() - len(ledger)),
        "gross_pnl_inr": gross,
        "explicit_costs_inr": costs,
        "net_pnl_inr": gross - costs,
        "net_with_0.5_point_slippage_per_leg_per_fill_inr": gross - costs - len(ledger) * 60.0,
        "net_with_1_point_slippage_per_leg_per_fill_inr": gross - costs - len(ledger) * 120.0,
    }
    return ledger, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    args = parser.parse_args()
    signals = load_signals(args.threshold)
    outcomes = attach_outcomes(signals)
    scenarios = []
    output = OUTPUT_ROOT / f"today_2026-10-27_threshold_{args.threshold * 100:.4f}"
    output.mkdir(parents=True, exist_ok=True)
    masks = {
        "literal_current_paper_engine_gate": pd.Series(True, index=outcomes.index),
        "also_require_legacy_three_rule_signal": outcomes["legacy_three_rule_signal"].astype(bool),
        "also_require_legacy_rule_and_quality_entry_ok": (
            outcomes["legacy_three_rule_signal"].astype(bool)
            & outcomes["quality_entry_ok"].astype(bool)
        ),
    }
    for label, mask in masks.items():
        ledger, summary = simulate(outcomes, mask, label)
        ledger.to_csv(output / f"trades_{label}.csv", index=False)
        scenarios.append(summary)
    payload = {
        "date": "2026-10-27",
        "status": "counterfactual reconstruction from today's logs",
        "threshold_rule": f"convergence probability strictly greater than {args.threshold:.8f}",
        "risk": {"target_inr": TARGET, "stop_loss_inr": STOP, "maximum_open_positions": MAX_OPEN},
        "costs": "configured statutory/exchange costs, zero brokerage; slippage shown separately",
        "scenarios": scenarios,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
