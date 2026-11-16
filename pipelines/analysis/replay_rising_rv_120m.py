"""Replay the rising-RV classifier with the IV/RV paper-book quote rules."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_outputs" / os.environ.get("IVRV_OUTPUT_DIR", "rising_rv_120m_2026-10-29")
PRED = OUT / "predictions.csv"
QUOTES = ROOT / "Data" / "banknifty_atm5_minute.csv"
LOT = 30


def charges(entry_sell: float, exit_sell: float, entry_buy: float, exit_buy: float) -> float:
    sell = (entry_sell + exit_sell) * LOT
    buy = (entry_buy + exit_buy) * LOT
    turnover = sell + buy
    transaction = turnover * 0.0003553
    sebi = turnover * 0.000001
    return sell * 0.0015 + transaction + sebi + buy * 0.00003 + 0.18 * (transaction + sebi)


def main() -> None:
    p = pd.read_csv(PRED, parse_dates=["datetime"])
    q = pd.read_csv(QUOTES, usecols=["minute", "strike", "option_type", "BestBuyPrice_1", "BestSellPrice_1",
                                     "t120_BestBuyPrice_1", "t120_BestSellPrice_1"])
    q["minute"] = pd.to_datetime(q.minute)
    quote = {(r.minute, float(r.strike), r.option_type): (float(r.BestBuyPrice_1), float(r.BestSellPrice_1))
             for r in q.itertuples(index=False)}
    signals = {pd.Timestamp(r.datetime) + pd.Timedelta(minutes=1): r for r in p.itertuples(index=False)
               if r.pred_rising_rv_convergence_signal and r.quality_entry_ok}
    books = {"short": [], "long": []}
    trades, skipped = [], []
    for minute in sorted(q.minute.unique()):
        minute = pd.Timestamp(minute)
        for side in books:
            active = []
            for position in books[side]:
                call = quote.get((minute, position["strike"], "CE"))
                put = quote.get((minute, position["strike"], "PE"))
                if call is None or put is None:
                    active.append(position)
                    continue
                exit_premium = call[1] + put[1] if side == "short" else call[0] + put[0]
                gross = (position["entry_premium"] - exit_premium) * LOT * (1 if side == "short" else -1)
                held = int((minute - position["entry_time"]).total_seconds() / 60)
                reason = ("target" if gross >= 1500 else "stop" if gross <= -750 else
                          "horizon" if held >= 120 else "square_off" if minute.time() >= pd.Timestamp("15:14:30").time() else None)
                if reason is None:
                    active.append(position)
                    continue
                cost = (charges(position["entry_premium"], 0, 0, exit_premium) if side == "short"
                        else charges(0, exit_premium, position["entry_premium"], 0))
                trades.append({**position, "side": side, "exit_time": minute, "exit_premium": exit_premium,
                               "exit_reason": reason, "holding_minutes": held, "gross_pnl": gross,
                               "charges": cost, "net_pnl": gross - cost})
            books[side] = active
            signal = signals.get(minute)
            if signal is None:
                continue
            if minute.time() >= pd.Timestamp("15:14:30").time():
                skipped.append({"side": side, "signal_time": signal.datetime, "reason": "entry_cutoff"})
                continue
            if len(active) >= 3:
                skipped.append({"side": side, "signal_time": signal.datetime, "reason": "max_3"})
                continue
            strike = float(signal.entry_strike)
            call, put = quote.get((minute, strike, "CE")), quote.get((minute, strike, "PE"))
            if call is None or put is None:
                skipped.append({"side": side, "signal_time": signal.datetime, "reason": "missing_next_minute_quote"})
                continue
            premium = call[0] + put[0] if side == "short" else call[1] + put[1]
            active.append({"signal_time": signal.datetime, "entry_time": minute, "strike": strike,
                           "probability": float(signal.pred_rising_rv_convergence_probability),
                           "entry_premium": premium})
    trade_frame = pd.DataFrame(trades)
    trade_frame.to_csv(OUT / "rising_rv_paper_trades.csv", index=False)
    pd.DataFrame(skipped).to_csv(OUT / "rising_rv_skipped.csv", index=False)
    summary = {"prediction_rows": len(p), "eligible_rows": int(p.pred_rising_rv_eligible.sum()),
               "signals": int(p.pred_rising_rv_convergence_signal.sum()), "threshold": 0.5,
               "quality_signals": len(signals), "lot_size": LOT, "entry": "next minute bid for short, ask for long",
               "exit": "ask for short, bid for long; 1500 target, -750 stop, 120-minute horizon, 15:14:30 square-off",
               "book_limit": 3, "brokerage": 0,
               "sides": {}}
    for side in books:
        t = trade_frame[trade_frame.side.eq(side)] if not trade_frame.empty else trade_frame
        summary["sides"][side] = {"closed_trades": len(t), "skipped": sum(s["side"] == side for s in skipped),
                                   "still_open": len(books[side]),
                                   "gross_pnl": float(t.gross_pnl.sum()) if len(t) else 0.0,
                                   "charges": float(t.charges.sum()) if len(t) else 0.0,
                                   "net_pnl": float(t.net_pnl.sum()) if len(t) else 0.0,
                                   "wins_net": int((t.net_pnl > 0).sum()) if len(t) else 0}
    # The source preserves a fixed-strike quote at entry+120 even when that
    # strike later leaves the moving five-strike minute window. This supports
    # a horizon-only check, but not path-dependent target/stop execution.
    indexed = q.set_index(["minute", "strike", "option_type"])
    horizon_trades = []
    for side in ("short", "long"):
        next_available = []
        for minute, signal in sorted(signals.items()):
            if minute.time() > pd.Timestamp("13:29:00").time():
                continue
            next_available = [exit_time for exit_time in next_available if exit_time > minute]
            if len(next_available) >= 3:
                continue
            strike = float(signal.entry_strike)
            try:
                call, put = (indexed.loc[(minute, strike, option)] for option in ("CE", "PE"))
            except KeyError:
                continue
            entry = (call.BestBuyPrice_1 + put.BestBuyPrice_1 if side == "short"
                     else call.BestSellPrice_1 + put.BestSellPrice_1)
            exit_price = (call.t120_BestSellPrice_1 + put.t120_BestSellPrice_1 if side == "short"
                          else call.t120_BestBuyPrice_1 + put.t120_BestBuyPrice_1)
            if not pd.notna(entry) or not pd.notna(exit_price) or entry <= 0 or exit_price <= 0:
                continue
            gross = (entry - exit_price) * LOT * (1 if side == "short" else -1)
            cost = (charges(entry, 0, 0, exit_price) if side == "short" else charges(0, exit_price, entry, 0))
            exit_time = pd.Timestamp(minute) + pd.Timedelta(minutes=120)
            next_available.append(exit_time)
            horizon_trades.append({"side": side, "signal_time": signal.datetime, "entry_time": minute,
                                   "exit_time": exit_time, "strike": strike, "probability": float(signal.pred_rising_rv_convergence_probability),
                                   "entry_premium": entry, "exit_premium": exit_price,
                                   "gross_pnl": gross, "charges": cost, "net_pnl": gross - cost})
    horizon = pd.DataFrame(horizon_trades)
    horizon.to_csv(OUT / "rising_rv_horizon_only_trades.csv", index=False)
    summary["horizon_only"] = {"method": "next-minute entry; fixed entry strike; 120-minute saved exit quote; max 3 concurrent positions; no stop or target",
                               "sides": {side: {"trades": int((horizon.side == side).sum()),
                                                "gross_pnl": float(horizon.loc[horizon.side == side, "gross_pnl"].sum()),
                                                "charges": float(horizon.loc[horizon.side == side, "charges"].sum()),
                                                "net_pnl": float(horizon.loc[horizon.side == side, "net_pnl"].sum())}
                                         for side in ("short", "long")}}
    (OUT / "rising_rv_pnl_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
