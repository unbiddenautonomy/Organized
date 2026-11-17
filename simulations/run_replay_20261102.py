"""Offline, causal replay of the three copied BANKNIFTY inference workers.

Model features and entry signals read only current-minute fields. The
option-chain target_* fields are used only when valuing a fixed-strike exit at
its 120-minute horizon and the current-minute chain no longer contains it.
"""
from __future__ import annotations

import csv
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "banknifty" / "2026-11-02"
PACK = ROOT / "pipelines"
OUT = DATA / "inference_replay"
BASE_OUT = OUT
FAMILIES = PACK / "families"
IVROOT = FAMILIES / "ivrv" / "engine" / "Inference"
sys.path[:0] = [str(ROOT), str(FAMILIES / "ivrv" / "standalone"), str(IVROOT), str(FAMILIES), str(FAMILIES / "weeks43")]

import pandas as pd
from voltr.market.options import greeks as _greeks, implied_volatility
from live.live_inference import (LiveFeatureInference, MinuteBar, NSE_EPOCH,
                                 paper_entry_route, required_live_input_features)
from live.position_alert import notify_position_action, notify_signal
from inference_pipeline import FixedStrikeInferencePipeline
from PCR.runtime import PCRRuntime, Models as PCRModels, load_config as pcr_load_config
from weeks43_engine import Ledger, ModelRunner, PaperEngine, second as nse_second, stamp as nse_stamp
from weeks43_settings import read_settings

EXPIRY = date(2026, 11, 23)
FUTURE_TOKEN = 61466
LOT_SIZE = 30
MAX_POSITIONS = 3
TARGET_INR = 1500.0
STOP_INR = -750.0
SQUARE_OFF = time(15, 14, 30)
HORIZON = 120
FEES = dict(stt_sell_premium_rate=0.0015, nse_options_transaction_rate=0.0003553,
            sebi_turnover_rate=0.000001, stamp_buy_premium_rate=0.00003,
            gst_rate=0.18, brokerage=0.0)


def dump(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(value, allow_nan=False, default=str, separators=(",", ":")) + "\n")


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return clean(value.item())
    return value


def minute_key(text: str) -> datetime:
    return datetime.fromisoformat(text)


def read_data():
    options = defaultdict(dict)
    pairs = defaultdict(dict)
    with (DATA / "banknifty_option_chain_1min.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = minute_key(r["minute"])
            options[t]
            if not r["last_traded_price"] or not r["open_interest"] or not r["source_tick_time"]:
                dump(OUT / "input_rejections.jsonl", dict(minute=t, token=r["option_token"], reason="missing_option_quote"))
                continue
            tick = datetime.fromisoformat(r["source_tick_time"])
            age = (t + timedelta(minutes=1) - tick).total_seconds()
            if not 0 <= age <= 60 or float(r["last_traded_price"]) <= 0:
                dump(OUT / "input_rejections.jsonl", dict(minute=t, token=r["option_token"], reason="stale_or_nonpositive_option_quote", age_seconds=age))
                continue
            strike = float(r["strike"])
            side = r["option_type"]
            # Target values are held for exit valuation only. Feature builders
            # below read only price, OI, and tick from the current minute.
            q = dict(strike=strike, side=side, token=int(r["option_token"]),
                     price=float(r["last_traded_price"]), oi=int(r["open_interest"]),
                     tick=tick,
                     target_minute=datetime.fromisoformat(r["target_minute"]),
                     target_price=float(r["target_last_traded_price"]) if r["target_last_traded_price"] else None,
                     target_tick=datetime.fromisoformat(r["target_source_tick_time"]) if r["target_source_tick_time"] else None)
            options[t][(strike, side)] = q
            pairs[strike][side] = q["token"]
    futures = {}
    with (DATA / "banknifty_futures_1min.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if int(r["token"]) == FUTURE_TOKEN:
                t = minute_key(r["minute"])
                if not r["last_traded_price"] or not r["open_interest"] or not r["source_tick_time"]:
                    dump(OUT / "input_rejections.jsonl", dict(minute=t, token=FUTURE_TOKEN, reason="missing_future_quote"))
                    continue
                tick = datetime.fromisoformat(r["source_tick_time"])
                age = (t + timedelta(minutes=1) - tick).total_seconds()
                if not 0 <= age <= 60 or float(r["last_traded_price"]) <= 0:
                    dump(OUT / "input_rejections.jsonl", dict(minute=t, token=FUTURE_TOKEN, reason="stale_or_nonpositive_future_quote", age_seconds=age))
                    continue
                futures[t] = dict(price=float(r["last_traded_price"]),
                                  oi=int(r["open_interest"]), tick=tick)
    pairs = {strike: sides for strike, sides in pairs.items() if "CE" in sides and "PE" in sides}
    if not options or not futures:
        raise ValueError("Missing near-expiry options or future")
    if any(float(r["price"]) <= 0 for quotes in options.values() for r in quotes.values()):
        raise ValueError("A selected option has nonpositive LTP")
    return options, futures, pairs


def fees(entry: float, exit_: float, quantity: int = LOT_SIZE, slippage_points: float = 0):
    sell = entry * quantity
    buy = exit_ * quantity
    turnover = sell + buy
    values = dict(brokerage=0.0, stt=sell * FEES["stt_sell_premium_rate"],
                  nse_transaction_charges=turnover * FEES["nse_options_transaction_rate"],
                  sebi_charges=turnover * FEES["sebi_turnover_rate"],
                  stamp_duty=buy * FEES["stamp_buy_premium_rate"],
                  slippage=slippage_points * quantity)
    values["gst"] = FEES["gst_rate"] * (values["nse_transaction_charges"] + values["sebi_charges"])
    values["total_cost"] = sum(values.values())
    return values


def quote_at(options, t, strike, side):
    return options.get(t, {}).get((strike, side))


def fresh_pair(options, t, call_strike, put_strike, max_age=60):
    a, b = quote_at(options, t, call_strike, "CE"), quote_at(options, t, put_strike, "PE")
    if not a or not b:
        return None
    now = t + timedelta(seconds=10)
    if any((now - q["tick"]).total_seconds() < 0 or (now - q["tick"]).total_seconds() > max_age for q in (a, b)):
        return None
    return a, b


def ivrv_predictions(options, futures, pairs):
    out = OUT / "ivrv"
    pipeline = FixedStrikeInferencePipeline(IVROOT / "config.yaml")
    engine = object.__new__(LiveFeatureInference)
    engine.config = dict(option_oi_max_age_minutes=15, future_oi_max_age_minutes=15,
                         maximum_surface_rmse=0.01, minimum_open_interest=100,
                         minimum_option_price=1.0, maximum_call_put_iv_difference=0.05)
    engine.expiry = EXPIRY
    engine.future_token = FUTURE_TOKEN
    engine.rate = 0.065
    engine.minimum_surface_points = 4
    engine.implied_volatility = implied_volatility
    engine.greeks = _greeks
    engine.pairs = dict(pairs)
    engine.oi_by_token = {}
    engine.rows_by_minute = {}
    engine.spot_by_minute = {}
    engine.required_feature_names = required_live_input_features(pipeline.artifacts)
    predictions = {}
    for t in sorted(options):
        market = futures.get(t)
        if not market:
            dump(out / "inference.jsonl", dict(minute=t, state="NO_INFERENCE", reason="missing_future"))
            continue
        m = int((t - NSE_EPOCH).total_seconds() // 60)
        quotes = options[t]
        strikes = sorted({strike for strike, _ in quotes})
        prices = {strike: {side: quotes[(strike, side)]["price"] for side in ("CE", "PE")}
                  for strike in strikes if (strike, "CE") in quotes and (strike, "PE") in quotes}
        for q in quotes.values():
            engine.oi_by_token[q["token"]] = (q["oi"], int((q["tick"] - NSE_EPOCH).total_seconds()), 17208)
        engine.oi_by_token[FUTURE_TOKEN] = (market["oi"], int((market["tick"] - NSE_EPOCH).total_seconds()), 17208)
        if len(prices) != 5:
            dump(out / "inference.jsonl", dict(minute=t, state="NO_INFERENCE", reason="incomplete_five_strike_window"))
            continue
        spot = market["price"]
        try:
            row = engine._build_current_row(m, spot, spot, sorted(prices), prices)
            if row is None:
                raise ValueError("invalid_iv_or_surface")
            engine.rows_by_minute[m] = row
            engine.spot_by_minute[m] = spot
            wide = engine._add_lags_and_spot_features(m, row)
            missing = [name for name in engine.required_feature_names if name not in wide]
            if missing:
                raise ValueError("missing_features:" + ",".join(missing))
            if any(m - lag not in engine.rows_by_minute for lag in (5, 15, 30, 60)):
                raise ValueError("feature_warmup")
            prediction = pipeline.predict(pd.DataFrame([wide])).iloc[0].to_dict()
            predictions[t] = dict(row=row, outputs=clean(prediction), spot=spot)
            dump(out / "inference.jsonl", clean(dict(minute=t, decision_time=t + timedelta(minutes=2),
                     state="READY", spot_proxy_source="near_future_ltp", spot=spot,
                     strike=row["entry_strike"], features=wide, outputs=prediction)))
        except (ValueError, KeyError, TypeError, ArithmeticError) as exc:
            dump(out / "inference.jsonl", dict(minute=t, state="NO_INFERENCE", reason=str(exc)))
    return predictions


def replay_ivrv(options, predictions, threshold):
    label = f"threshold_{int(threshold * 100)}"
    out = OUT / "ivrv" / label
    events = out / "decisions.jsonl"
    trades_path = out / "trades.jsonl"
    out.mkdir(parents=True, exist_ok=True)
    for path in (events, trades_path, out / "marks.jsonl"):
        path.write_text("", encoding="utf-8")
    positions = {}
    horizon_quotes = {}
    trades = []
    times = sorted(options)
    reverse = {t + timedelta(minutes=2): (t, p) for t, p in predictions.items()}
    max_open = 0
    for t in times:
        # Mark existing fixed-strike positions before admitting a new one.
        for pid, p in list(positions.items()):
            q = fresh_pair(options, t, p["strike"], p["strike"])
            quote_source = "current_minute"
            due = p["entry_time"].replace(second=0) + timedelta(minutes=HORIZON)
            if q is None and t == due:
                saved = horizon_quotes.get(pid)
                if saved and saved[0]["target_minute"] == t and saved[1]["target_minute"] == t:
                    q = (dict(price=saved[0]["target_price"]), dict(price=saved[1]["target_price"]))
                    quote_source = "saved_same_strike_horizon_quote"
            if q is None:
                dump(events, dict(minute=t, event="MARK_SKIPPED", position_id=pid, reason="missing_or_stale_fixed_strike_quote"))
                continue
            call, put = q
            exit_premium = call["price"] + put["price"]
            charges = fees(p["entry_premium"], exit_premium)
            gross = (p["entry_premium"] - exit_premium) * LOT_SIZE
            net = gross - charges["total_cost"]
            reason = ("square_off" if t.time() >= SQUARE_OFF else
                      "profit_target" if net >= TARGET_INR else
                      "stop_loss" if net <= STOP_INR else
                      "time_exit" if (t + timedelta(seconds=10) - p["entry_time"]).total_seconds() >= HORIZON * 60 else None)
            mark = clean(dict(minute=t, event="MARK", position_id=pid, gross_pnl=gross,
                              net_pnl=net, charges=charges, call_price=call["price"],
                              put_price=put["price"], quote_source=quote_source))
            dump(out / "marks.jsonl", mark)
            if reason:
                trade = dict(p, exit_time=t + timedelta(seconds=10), exit_reason=reason,
                             exit_call=call["price"], exit_put=put["price"],
                             gross_pnl=gross, net_pnl=net, charges=charges)
                trades.append(trade)
                dump(trades_path, clean(dict(event="EXIT", **trade)))
                notify_position_action()
                del positions[pid]
                horizon_quotes.pop(pid, None)
        source = reverse.get(t)
        if source is None:
            dump(events, dict(minute=t, event="NO_SIGNAL", reason="warmup_or_no_model_output"))
            continue
        feature_minute, pred = source
        output = pred["outputs"]
        routing = paper_entry_route(output, threshold, 0.005, 0.0)
        reason = None
        if not routing["eligible"]:
            reason = routing["reason"]
        elif len(positions) >= MAX_POSITIONS:
            reason = "three_position_limit"
        elif t.time() >= SQUARE_OFF:
            reason = "after_square_off"
        strike = float(pred["row"]["entry_strike"])
        pair = fresh_pair(options, t, strike, strike)
        if reason is None and pair is None:
            reason = "missing_or_stale_fixed_strike_quote"
        event = clean(dict(minute=t, feature_minute=feature_minute, event="REJECTED" if reason else "ACCEPTED",
                           reason=reason, route=routing["route"], threshold=threshold,
                           probability=routing["convergence_probability"], event_score=routing["rv_decline_event_score"],
                           strike=strike, open_before=len(positions)))
        if routing["eligible"]:
            notify_signal()
        if reason is None:
            call, put = pair
            pid = f"{t:%Y%m%d%H%M}-{strike:g}"
            p = dict(position_id=pid, entry_time=t + timedelta(seconds=10),
                     strike=strike, call_token=call["token"], put_token=put["token"],
                     entry_call=call["price"], entry_put=put["price"],
                     entry_premium=call["price"] + put["price"], quantity_call=LOT_SIZE,
                     quantity_put=LOT_SIZE, lots=1, threshold=threshold, route=routing["route"])
            positions[pid] = p
            horizon_quotes[pid] = (call, put)
            max_open = max(max_open, len(positions))
            event["position_id"] = pid
            dump(trades_path, clean(dict(event="ENTRY", **p)))
            notify_position_action()
        dump(events, event)
        if t.time() == time(15, 14):
            cutoff_time = t.replace(second=30)
            for pid, p in list(positions.items()):
                q = fresh_pair(options, t, p["strike"], p["strike"])
                if q is None:
                    dump(events, dict(minute=cutoff_time, event="SQUARE_OFF_PENDING",
                                      position_id=pid, reason="missing_or_stale_fixed_strike_quote"))
                    continue
                call, put = q
                exit_premium = call["price"] + put["price"]
                charges = fees(p["entry_premium"], exit_premium)
                gross = (p["entry_premium"] - exit_premium) * LOT_SIZE
                net = gross - charges["total_cost"]
                trade = dict(p, exit_time=cutoff_time, exit_reason="square_off",
                             exit_call=call["price"], exit_put=put["price"],
                             gross_pnl=gross, net_pnl=net, charges=charges)
                trades.append(trade)
                dump(trades_path, clean(dict(event="EXIT", **trade)))
                notify_position_action()
                del positions[pid]
                horizon_quotes.pop(pid, None)
    # The prepared chain intentionally ends at 15:15. A fixed entry strike may
    # have moved outside the current ATM +/-2 window. Its saved 15:15 same-token
    # quote is the requested session-end fallback for an otherwise open book.
    final_minute = datetime.combine(times[0].date(), time(15, 15))
    for pid, p in list(positions.items()):
        saved = horizon_quotes.get(pid)
        if not saved or any(q["target_minute"] != final_minute or q["target_price"] is None for q in saved):
            dump(events, dict(minute=final_minute, event="UNRESOLVED_POSITION", position_id=pid,
                              reason="no_same_strike_1515_quote"))
            continue
        call_price, put_price = (q["target_price"] for q in saved)
        exit_premium = call_price + put_price
        charges = fees(p["entry_premium"], exit_premium)
        gross = (p["entry_premium"] - exit_premium) * LOT_SIZE
        net = gross - charges["total_cost"]
        trade = dict(p, exit_time=final_minute + timedelta(seconds=10),
                     exit_reason="fixed_strike_1515_fallback", exit_quote_source="saved_same_strike_target",
                     exit_call=call_price, exit_put=put_price,
                     gross_pnl=gross, net_pnl=net, charges=charges)
        trades.append(trade)
        dump(trades_path, clean(dict(event="EXIT", **trade)))
        notify_position_action()
        del positions[pid]
        horizon_quotes.pop(pid, None)
    result = dict(threshold=threshold, accepted=sum(1 for _ in trades) + len(positions),
                  closed=len(trades), open_positions=len(positions), maximum_open=max_open,
                  gross_pnl=sum(p["gross_pnl"] for p in trades),
                  total_cost=sum(p["charges"]["total_cost"] for p in trades),
                  net_pnl=sum(p["net_pnl"] for p in trades),
                  rejected=Counter(json.loads(s)["reason"] for s in events.read_text().splitlines()
                                   if json.loads(s).get("event") == "REJECTED"))
    (out / "summary.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


def replay_pcr(options, futures, pairs):
    out = OUT / "pcr"
    config = pcr_load_config()
    config.update(maximum_open_positions=MAX_POSITIONS, lots_per_position=1,
                  profit_target_net_rupees=TARGET_INR, stop_loss_net_rupees=STOP_INR,
                  round_trip_slippage_points=0.0, brokerage_rupees=0.0)
    context = dict(mode="recorded_replay", source_mode="recorded_replay", expiry=EXPIRY.isoformat(),
                   future_token=FUTURE_TOKEN, lot_size=LOT_SIZE, pairs={str(k): v for k, v in pairs.items()},
                   risk_free_rate=0.065, cost_rates=FEES, initial_positions=[])
    runtime = PCRRuntime(context, out, models=PCRModels(FAMILIES / "PCR"), config=config)
    for t in sorted(options):
        market = futures.get(t)
        if market is None:
            dump(out / "input_rejections.jsonl", dict(minute=t, reason="missing_or_stale_future"))
            continue
        now = t + timedelta(seconds=10)
        if market["tick"] > now:
            dump(out / "input_rejections.jsonl", dict(minute=t, reason="future_tick_after_snapshot_time"))
            continue
        available = [q for q in options[t].values() if q["tick"] <= now]
        quote_dict = {str(q["token"]): [q["price"], q["tick"].isoformat()] for q in available}
        oi_dict = {str(q["token"]): [q["oi"], q["tick"].isoformat()] for q in available}
        oi_dict[str(FUTURE_TOKEN)] = [market["oi"], market["tick"].isoformat()]
        snapshot = dict(timestamp=now.isoformat(), quotes=quote_dict, oi=oi_dict,
                        spot=[market["price"], market["tick"].isoformat()],
                        future=[market["price"], market["tick"].isoformat()],
                        future_open={t.isoformat(): market["price"]}, market_update=True)
        runtime.on_snapshot(snapshot, processing_delay=0.0)
        if t.time() == time(15, 14):
            runtime.on_snapshot(dict(timestamp=t.replace(second=30).isoformat(), market_update=False), 0.0)
    status = runtime.book.status()
    result = dict(closed=status["closed_count"], open_positions=len(status["open_positions"]),
                  position_limit=MAX_POSITIONS, gross_pnl=status["realized_gross_rupees"],
                  total_cost=status["realized_cost_rupees"], net_pnl=status["realized_net_rupees"])
    (out / "summary.json").write_text(json.dumps(clean(result), indent=2), encoding="utf-8")
    return result


def otm_row(t, quotes, market):
    strikes = sorted({k for k, _ in quotes})
    if len(strikes) != 5:
        return None
    atm = strikes[2]
    cs, ps = strikes[3], strikes[1]
    if cs - atm != 100 or atm - ps != 100:
        return None
    required = [(cs, "CE"), (ps, "PE"), (atm, "CE"), (atm, "PE")]
    if any(key not in quotes for key in required):
        return None
    signal = t + timedelta(seconds=59)
    expiry = datetime.combine(EXPIRY, time(15, 30))
    return dict(signal_time=signal.isoformat(sep=" "), available_at=(t + timedelta(minutes=1)).isoformat(sep=" "),
                expiry=EXPIRY.isoformat(), days_to_expiry=(expiry - signal).total_seconds() / 86400,
                spot=market["price"], future=market["price"], atm=atm, step=100.0,
                call_strike=cs, put_strike=ps, call=quotes[(cs, "CE")]["price"],
                put=quotes[(ps, "PE")]["price"], atm_call=quotes[(atm, "CE")]["price"],
                atm_put=quotes[(atm, "PE")]["price"], call_oi=quotes[(cs, "CE")]["oi"],
                put_oi=quotes[(ps, "PE")]["oi"], atm_call_oi=quotes[(atm, "CE")]["oi"],
                atm_put_oi=quotes[(atm, "PE")]["oi"], call_key=str(quotes[(cs, "CE")]["token"]),
                put_key=str(quotes[(ps, "PE")]["token"]), quantity=LOT_SIZE)


def replay_43weeks(options, futures):
    out = OUT / "43weeks"
    config = json.loads((FAMILIES / "weeks43/config.json").read_text())
    config.update(max_open_positions=MAX_POSITIONS, lots_per_position=1,
                  profit_target_inr=TARGET_INR, stop_loss_inr=STOP_INR,
                  round_trip_slippage_points=0.0, brokerage_per_order_inr=0.0,
                  historical_cost_mode="current_config_rates",
                  **{k: v for k, v in FEES.items() if k in config})
    ledger = Ledger(out, durable=False)
    engine = PaperEngine(config, ledger, ModelRunner(FAMILIES / "weeks43"),
                         live=False, entry_settings=read_settings(FAMILIES / "weeks43"))
    pending_row = None
    for t in sorted(options):
        market = futures.get(t)
        if market is None:
            dump(out / "input_rejections.jsonl", dict(minute=t, reason="missing_or_stale_future"))
            continue
        now = t + timedelta(seconds=10)
        quote_events = [(str(q["token"]), q["price"], int((q["tick"] - NSE_EPOCH).total_seconds()))
                        for q in options[t].values() if q["tick"] <= now]
        engine.on_quotes(quote_events, nse_second(now.isoformat(sep=" ")))
        if pending_row is not None:
            engine.on_snapshot(pending_row, nse_second(now.isoformat(sep=" ")))
        pending_row = otm_row(t, options[t], market)
        if pending_row is None:
            dump(out / "input_rejections.jsonl", dict(minute=t, reason="missing_otm_pair_or_strike_step"))
        if t.time() == time(15, 14):
            engine.on_quotes([], nse_second(t.replace(second=30).isoformat(sep=" ")))
    trades = ledger.all("trades")
    positions = [p for p in ledger.all("positions") if p.get("status") == "OPEN"]
    signals = ledger.all("signals")
    result = dict(closed=len(trades), open_positions=len(positions), position_limit=MAX_POSITIONS,
                  gross_pnl=sum(p["gross_pnl"] for p in trades),
                  total_cost=sum(p["costs"]["total"] for p in trades),
                  net_pnl=sum(p["net_pnl"] for p in trades),
                  accepted=sum(s["decision"] == "PENDING" for s in signals),
                  rejected=sum(s["decision"] == "REJECTED" for s in signals))
    (out / "summary.json").write_text(json.dumps(clean(result), indent=2), encoding="utf-8")
    engine.close()
    return result


def main():
    global OUT, DATA, EXPIRY, FUTURE_TOKEN
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="Directory with the prepared one-minute BANKNIFTY files")
    parser.add_argument("--output", type=Path,
                        help="Empty directory for this replay's logs and results")
    args = parser.parse_args()
    DATA = args.data.resolve()
    with (DATA / "banknifty_futures_1min.csv").open(newline="", encoding="utf-8") as source:
        near = next((row for row in csv.DictReader(source) if row["is_near_expiry"] == "True"), None)
    if near is None:
        raise ValueError("Prepared futures file has no near-expiry contract")
    EXPIRY = date.fromisoformat(near["expiry_master"])
    FUTURE_TOKEN = int(near["token"])
    session_date = date.fromisoformat(near["minute"][:10])
    OUT = (args.output or ROOT / "runtime" / "replays" / session_date.isoformat() /
           datetime.now().strftime("run_%Y%m%d_%H%M%S")).resolve()
    base_out = OUT.parent
    if OUT.exists() and any(OUT.iterdir()):
        raise ValueError(f"Output directory is not empty: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    options, futures, pairs = read_data()
    (OUT / "assumptions.json").write_text(json.dumps(dict(
        date=session_date.isoformat(), expiry=EXPIRY.isoformat(), spot_proxy="near_future_ltp",
        future_token=FUTURE_TOKEN, quantity_call=LOT_SIZE, quantity_put=LOT_SIZE,
        maximum_open_positions_per_worker=MAX_POSITIONS, horizon_minutes=HORIZON,
        target_net_inr=TARGET_INR, stop_net_inr=STOP_INR, fees=FEES,
        option_source="first_tick_in_each_minute", target_columns_used_for_features=False,
        target_columns_used_for_horizon_exit=True,
        fill_rule="observed LTP within one minute, fixed entry tokens",
        slippage_points=0.0), indent=2), encoding="utf-8")
    predictions = ivrv_predictions(options, futures, pairs)
    summaries = {"ivrv": {str(int(x * 100)): replay_ivrv(options, predictions, x) for x in (0.75, 0.70, 0.50)},
                 "pcr": replay_pcr(options, futures, pairs),
                 "43weeks": replay_43weeks(options, futures)}
    (OUT / "summary.json").write_text(json.dumps(clean(summaries), indent=2, default=str), encoding="utf-8")
    base_out.mkdir(parents=True, exist_ok=True)
    (base_out / "latest_run.json").write_text(json.dumps({"output": str(OUT)}, indent=2), encoding="utf-8")
    print(json.dumps(clean(summaries), indent=2, default=str))


if __name__ == "__main__":
    main()
