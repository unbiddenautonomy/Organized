"""Reprice the 11 already-entered IVRV 50% trades using their raw option ticks."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets" / "banknifty" / "2026-11-02" / "inference_replay" / "runs" / "final_20261102"
TICKS = ROOT / "tick_reprice_50" / "selected_token_ticks.csv"
TRADES = ROOT / "ivrv" / "threshold_50" / "trades.jsonl"
OUT = ROOT / "tick_reprice_50"
QTY = 30
TARGET = 1500.0
STOP = -750.0
MAX_AGE = timedelta(seconds=60)
RATE = dict(stt=0.0015, nse=0.0003553, sebi=0.000001, stamp=0.00003, gst=0.18)


def charges(entry_premium: float, exit_premium: float):
    sell = entry_premium * QTY
    buy = exit_premium * QTY
    turnover = sell + buy
    result = dict(brokerage=0.0, stt=sell * RATE["stt"], nse=turnover * RATE["nse"],
                  sebi=turnover * RATE["sebi"], stamp=buy * RATE["stamp"])
    result["gst"] = (result["nse"] + result["sebi"]) * RATE["gst"]
    result["total"] = sum(result.values())
    return result


def raw_ticks():
    by_token = defaultdict(list)
    with TICKS.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            by_token[int(row["token"])].append((datetime.fromisoformat(row["timestamp"]),
                                                float(row["last_traded_price"]),
                                                int(row["source_line_order"])))
    return by_token


def original_trades():
    rows = [json.loads(line) for line in TRADES.read_text(encoding="utf-8").splitlines()]
    entries = [row for row in rows if row["event"] == "ENTRY"]
    exits = {row["position_id"]: row for row in rows if row["event"] == "EXIT"}
    return entries, exits


def reprice_one(entry, ticks, original):
    start = datetime.fromisoformat(entry["entry_time"])
    cutoff = datetime.combine(start.date(), time(15, 14, 30))
    horizon = start + timedelta(minutes=120)
    deadline = min(horizon, cutoff)
    final = datetime.combine(start.date(), time(15, 15, 10))
    call_token, put_token = entry["call_token"], entry["put_token"]
    events = sorted(((ts, order, token, price)
                     for token in (call_token, put_token)
                     for ts, price, order in ticks[token]
                     if start - MAX_AGE <= ts <= final), key=lambda x: (x[0], x[1]))
    latest = {}
    cursor = 0
    while cursor < len(events) and events[cursor][0] <= start:
        ts, _, token, price = events[cursor]
        latest[token] = (price, ts)
        cursor += 1

    def pair(now):
        if call_token not in latest or put_token not in latest:
            return None
        a, b = latest[call_token], latest[put_token]
        if any(ts > now or now - ts > MAX_AGE for _, ts in (a, b)):
            return None
        return a, b

    entry_pair = pair(start)
    base = dict(position_id=entry["position_id"], strike=entry["strike"],
                call_token=call_token, put_token=put_token, scheduled_entry_time=start.isoformat(),
                original_minute_net_pnl=original["net_pnl"],
                scanned_call_ticks=sum(token == call_token for _, _, token, _ in events),
                scanned_put_ticks=sum(token == put_token for _, _, token, _ in events))
    if entry_pair is None:
        return dict(base, status="NO_FRESH_RAW_ENTRY_PAIR")

    entry_call, entry_put = entry_pair
    entry_premium = entry_call[0] + entry_put[0]
    def valuation(now):
        q = pair(now)
        if q is None:
            return None
        exit_call, exit_put = q
        exit_premium = exit_call[0] + exit_put[0]
        cost = charges(entry_premium, exit_premium)
        gross = (entry_premium - exit_premium) * QTY
        return dict(exit_call=exit_call[0], exit_put=exit_put[0],
                    exit_call_tick=exit_call[1].isoformat(), exit_put_tick=exit_put[1].isoformat(),
                    gross_pnl=gross, charges=cost, net_pnl=gross - cost["total"])

    result = dict(base, status="OPEN", entry_time=start.isoformat(), entry_call=entry_call[0],
                  entry_put=entry_put[0], entry_call_tick=entry_call[1].isoformat(),
                  entry_put_tick=entry_put[1].isoformat(), entry_premium=entry_premium,
                  quantity_call=QTY, quantity_put=QTY)
    last_update = start
    while cursor < len(events):
        ts, _, token, price = events[cursor]
        cursor += 1
        # Scheduled horizon/cutoff takes priority once it is reached. A quote
        # must be fresh at the scheduled time; no future tick is used early.
        if last_update < deadline < ts:
            value = valuation(deadline)
            if value is not None:
                return dict(result, status="CLOSED", exit_time=deadline.isoformat(),
                            exit_reason="square_off" if deadline == cutoff else "time_exit",
                            **value, pnl_change_from_minute=value["net_pnl"] - original["net_pnl"])
        latest[token] = (price, ts)
        last_update = ts
        if ts < start:
            continue
        value = valuation(ts)
        if value is None:
            continue
        reason = ("square_off" if ts >= cutoff else
                  "profit_target" if value["net_pnl"] >= TARGET else
                  "stop_loss" if value["net_pnl"] <= STOP else
                  "time_exit" if ts >= horizon else None)
        if reason:
            return dict(result, status="CLOSED", exit_time=ts.isoformat(), exit_reason=reason,
                        **value, pnl_change_from_minute=value["net_pnl"] - original["net_pnl"])
    value = valuation(deadline)
    if value is not None:
        return dict(result, status="CLOSED", exit_time=deadline.isoformat(),
                    exit_reason="square_off" if deadline == cutoff else "time_exit",
                    **value, pnl_change_from_minute=value["net_pnl"] - original["net_pnl"])
    return dict(result, status="UNRESOLVED_NO_FRESH_EXIT_PAIR")


def main():
    ticks = raw_ticks()
    entries, exits = original_trades()
    results = [reprice_one(entry, ticks, exits[entry["position_id"]]) for entry in entries]
    fields = ["position_id", "strike", "call_token", "put_token", "status", "scheduled_entry_time",
              "entry_time", "entry_call_tick", "entry_put_tick", "entry_call", "entry_put",
              "exit_time", "exit_reason", "exit_call_tick", "exit_put_tick", "exit_call", "exit_put",
              "quantity_call", "quantity_put", "gross_pnl", "stt", "nse", "sebi", "stamp", "gst",
              "brokerage", "total_cost", "net_pnl", "original_minute_net_pnl", "pnl_change_from_minute",
              "scanned_call_ticks", "scanned_put_ticks"]
    with (OUT / "tick_repriced_trades.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        for value in results:
            flat = dict(value)
            cost = flat.pop("charges", {})
            flat.update(cost)
            flat["total_cost"] = cost.get("total", "")
            flat.pop("total", None)
            writer.writerow({name: flat.get(name, "") for name in fields})
    closed = [x for x in results if x["status"] == "CLOSED"]
    summary = dict(selected_positions=len(entries), closed=len(closed), unresolved=len(entries) - len(closed),
                   gross_pnl=sum(x["gross_pnl"] for x in closed),
                   total_cost=sum(x["charges"]["total"] for x in closed),
                   net_pnl=sum(x["net_pnl"] for x in closed),
                   original_minute_net_pnl=sum(x["net_pnl"] for x in exits.values()),
                   selected_tokens=sorted(ticks))
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ticks", type=Path, default=TICKS)
    parser.add_argument("--trades", type=Path, default=TRADES)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    TICKS = args.ticks.resolve()
    TRADES = args.trades.resolve()
    OUT = args.output.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    main()
