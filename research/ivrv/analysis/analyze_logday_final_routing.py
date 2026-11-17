from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path

import pandas as pd


TARGET = 1500.0
STOP = -750.0
LOT_SIZE = 30
MAX_OPEN = 3
EVENT_SCORE_CUTOFF = 0.0


def costs(entry: float, exit_: float) -> float:
    entry_turnover = entry * LOT_SIZE
    exit_turnover = exit_ * LOT_SIZE
    turnover = entry_turnover + exit_turnover
    stt = entry_turnover * 0.0015
    nse = turnover * 0.0003553
    sebi = turnover * 0.000001
    stamp = exit_turnover * 0.00003
    gst = 0.18 * (nse + sebi)
    return stt + nse + sebi + stamp + gst


def load_signals(logdir: Path, above_threshold: float) -> pd.DataFrame:
    rows = []
    with (logdir / "live_predictions.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            stamp = pd.Timestamp(event["exchange_minute"])
            if stamp.time() >= pd.Timestamp("15:14:30").time():
                continue
            strategy = event.get("strategy_signal", {})
            predictions = event.get("predictions", {})
            rules = strategy.get("rules", {})
            gap = strategy.get("iv_minus_rv")
            probability = predictions.get("pred_convergence_probability_classifier")
            score = predictions.get("pred_no120_event_score")
            if gap is None or not math.isfinite(float(gap)):
                continue
            gap = float(gap)
            branch = None
            if (
                gap > 0.005
                and bool(rules.get("iv_above_rv_and_predicted_iv_declines_to_predicted_rv"))
                and probability is not None
                and float(probability) > above_threshold
            ):
                branch = "iv_above_rv"
            elif abs(gap) <= 0.005 and score is not None and float(score) >= EVENT_SCORE_CUTOFF:
                branch = "iv_near_rv"
            elif gap < -0.005 and score is not None and float(score) >= EVENT_SCORE_CUTOFF:
                branch = "iv_below_rv"
            if branch is None:
                continue
            rows.append({
                "signal_datetime": stamp,
                "strike": float(event["fixed_entry_strike"]),
                "expiry": event["expiry"],
                "branch": branch,
                "iv_minus_rv": gap,
                "convergence_probability": probability,
                "rv_decline_event_score": score,
                "quality_entry_ok": bool(event.get("market", {}).get("quality_entry_ok")),
            })
    return pd.DataFrame(rows).sort_values("signal_datetime", kind="stable").reset_index(drop=True)


def attach_outcomes(logdir: Path, signals: pd.DataFrame) -> pd.DataFrame:
    wanted = {(str(row.signal_datetime), row.strike) for row in signals.itertuples()}
    paths = {key: [] for key in wanted}
    with (logdir / "fixed_strike_120m_paths.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row.get("entry_datetime")), float(row.get("entry_strike", 0)))
            if key in paths:
                paths[key].append(row)
    outcomes = []
    for signal in signals.itertuples():
        key = (str(signal.signal_datetime), signal.strike)
        observations = sorted(paths[key], key=lambda row: int(row["holding_minutes"]))
        if not observations:
            raise RuntimeError(f"Missing path for {key}")
        chosen = None
        for row in observations:
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
            chosen = row
            break
        if chosen is None:
            raise RuntimeError(f"No exit for {key}")
        entry = float(chosen["entry_straddle_price"])
        exit_ = float(chosen["observation_straddle_price"])
        gross = float(chosen["short_pnl_per_lot"])
        fee = costs(entry, exit_)
        outcomes.append({
            **signal._asdict(),
            "entry_straddle_price": entry,
            "exit_datetime": pd.Timestamp(chosen["observation_datetime"]),
            "exit_straddle_price": exit_,
            "exit_reason": reason,
            "gross_pnl_inr": gross,
            "explicit_costs_inr": fee,
            "net_pnl_inr": gross - fee,
        })
    return pd.DataFrame(outcomes)


def apply_cap(frame: pd.DataFrame) -> pd.DataFrame:
    heap: list[int] = []
    accepted = []
    for row in frame.sort_values("signal_datetime", kind="stable").itertuples():
        signal_ns = int(pd.Timestamp(row.signal_datetime).value)
        while heap and heap[0] <= signal_ns:
            heapq.heappop(heap)
        if len(heap) >= MAX_OPEN:
            continue
        heapq.heappush(heap, int(pd.Timestamp(row.exit_datetime).value))
        accepted.append(row._asdict())
    return pd.DataFrame(accepted)


def summarize(candidates: pd.DataFrame, ledger: pd.DataFrame, label: str) -> dict:
    gross = float(ledger["gross_pnl_inr"].sum()) if len(ledger) else 0.0
    fee = float(ledger["explicit_costs_inr"].sum()) if len(ledger) else 0.0
    return {
        "scenario": label,
        "qualified_signals": int(len(candidates)),
        "executed_trades": int(len(ledger)),
        "skipped_at_three_position_cap": int(len(candidates) - len(ledger)),
        "executed_by_branch": ledger["branch"].value_counts().to_dict() if len(ledger) else {},
        "gross_pnl_inr": gross,
        "explicit_costs_inr": fee,
        "net_pnl_inr": gross - fee,
        "net_with_0.5_point_slippage_per_leg_per_fill_inr": gross - fee - len(ledger) * 60.0,
        "net_with_1_point_slippage_per_leg_per_fill_inr": gross - fee - len(ledger) * 120.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--above-threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    signals = load_signals(args.logdir, args.above_threshold)
    outcomes = attach_outcomes(args.logdir, signals)
    ledger = apply_cap(outcomes)
    quality_candidates = outcomes.loc[outcomes["quality_entry_ok"].astype(bool)].copy()
    quality_ledger = apply_cap(quality_candidates)
    args.output.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(args.output / "trades_final_routing.csv", index=False)
    quality_ledger.to_csv(args.output / "trades_final_routing_with_quality_gate.csv", index=False)
    payload = {
        "date": args.date,
        "status": "counterfactual reconstruction from saved logs",
        "routing": {
            "iv_above_rv": f"gap > 0.005, convergence rule, probability > {args.above_threshold}",
            "iv_near_rv": "abs(gap) <= 0.005 and RV-decline event score >= 0",
            "iv_below_rv": "gap < -0.005 and RV-decline event score >= 0",
        },
        "risk": {"target_inr": TARGET, "stop_loss_inr": STOP, "maximum_open_positions": MAX_OPEN},
        "scenarios": [
            summarize(outcomes, ledger, "literal_current_paper_engine_quality_behavior"),
            summarize(quality_candidates, quality_ledger, "also_require_quality_entry_ok"),
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
