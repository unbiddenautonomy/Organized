"""Independent PCR models, position ledger and causal live feature consumer."""
from __future__ import annotations

from datetime import datetime, timedelta, time
from pathlib import Path
import hashlib
import json
import math

import joblib
import numpy as np
import pandas as pd

from .features import feature_at, option_stats
from .storage import append, atomic_json, clean, read_lines, read_text

ROOT = Path(__file__).resolve().parent
MODEL_FOLDERS = {"decay": "straddle_decay_model", "win": "win_probability_model",
                 "holding": "holding_period_model", "tail": "tail_risk_model"}


def load_config():
    c = json.loads(read_text(ROOT / "config.json"))
    if c["maximum_open_positions"] != 3 or c["brokerage_rupees"] != 0:
        raise ValueError("PCR requires max 3 positions and zero brokerage")
    if c["horizons_minutes"] != [60, 120] or c["profit_target_net_rupees"] <= 0 or c["stop_loss_net_rupees"] >= 0:
        raise ValueError("invalid PCR horizon or exit configuration")
    for key in ("0", "1", "2", "3_4"):
        v = c["win_probability_thresholds"][key]
        if not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("Win cutoffs must be between 0 and 1 for all expiry groups")
    return c


class Models:
    def __init__(self, root=ROOT):
        self.bundles = {}
        for key, folder in MODEL_FOLDERS.items():
            directory = Path(root) / "models" / folder
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            expected = manifest["files"]["model_bundle.joblib"]
            if hashlib.sha256((directory / "model_bundle.joblib").read_bytes()).hexdigest() != expected:
                raise ValueError(f"Model hash mismatch: {folder}")
            b = joblib.load(directory / "model_bundle.joblib")
            if key == "holding":
                b["model"].n_jobs = 1  # Same predictions, avoid worker process spawning per minute.
            self.bundles[key] = b
        self.names = list(self.bundles["decay"]["features"])
        if any(list(b["features"]) != self.names for b in self.bundles.values()):
            raise ValueError("Model feature contracts disagree")
        self.thresholds = json.loads((Path(root) / "thresholds.json").read_text(encoding="utf-8"))
        if any(b["thresholds"] != self.thresholds for b in self.bundles.values()):
            raise ValueError("Packaged thresholds disagree")

    def predict(self, features):
        x = pd.DataFrame([features], columns=self.names).replace([np.inf, -np.inf], np.nan)
        def prob(key):
            b = self.bundles[key]
            p = b["model"].predict_proba(x)[:, 1]
            z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            return float(b["calibrator"].predict_proba(z.reshape(-1, 1))[0, 1])
        holds = self.bundles["holding"]["model"].predict(x)[0]
        result = {"decay_60_points": float(self.bundles["decay"]["model"].predict(x)[0]),
                  "win_probability_60m": prob("win"), "holding_60_points": float(holds[0]),
                  "holding_120_points": float(holds[1]), "tail_loss_probability": prob("tail")}
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError("nonfinite model prediction")
        return result


def costs(entry, exit_, quantity, rates, slippage_points=2.0):
    sell, buy = entry * quantity, exit_ * quantity
    turnover = sell + buy
    stt = sell * float(rates["stt_sell_premium_rate"])
    exchange = turnover * float(rates["nse_options_transaction_rate"])
    sebi = turnover * float(rates["sebi_turnover_rate"])
    stamp = buy * float(rates["stamp_buy_premium_rate"])
    gst = (exchange + sebi) * float(rates["gst_rate"])
    statutory = stt + exchange + sebi + stamp + gst
    slippage = slippage_points * quantity
    return {"brokerage_rupees": 0., "stt_rupees": stt, "exchange_rupees": exchange,
            "sebi_rupees": sebi, "stamp_rupees": stamp, "gst_rupees": gst,
            "statutory_rupees": statutory, "slippage_rupees": slippage,
            "total_cost_rupees": statutory + slippage}


class Book:
    def __init__(self, directory, config, context):
        self.directory = Path(directory)
        self.config, self.context = config, context
        self.positions, self.closed = {}, []
        # The append-only entry/exit journal is the recovery source, including
        # any unresolved earlier-day position. Entries are never silently lost.
        for file in sorted(self.directory.glob("????-??-??/trades.jsonl")):
            for event in read_lines(file):
                p = event.get("position", {})
                if event.get("event") == "ENTRY":
                    self.positions[p["id"]] = p
                elif event.get("event") == "EXIT" and p.get("id") in self.positions:
                    self.positions.pop(p["id"])
                    self.closed.append(p)

    def log(self, now, name, value):
        append(self.directory / now.date().isoformat() / f"{name}.jsonl", value)

    def enter(self, now, strike, pair, prices, horizon, signal):
        if len(self.positions) >= self.config["maximum_open_positions"]:
            return False
        # Prior-day unclosed positions require reconciliation before new risk.
        if any(p["entry_datetime"][:10] != now.date().isoformat() for p in self.positions.values()):
            return False
        if not time.fromisoformat(self.config["entry_start"]) <= now.time() <= time.fromisoformat(self.config["entry_end"]):
            return False
        key = signal["id"]
        if key in self.positions or any(p["id"] == key for p in self.closed):
            return False
        q = int(self.context["lot_size"]) * int(self.config["lots_per_position"])
        p = {"id": key, "entry_datetime": now.isoformat(), "entry_strike": strike,
             "expiry": self.context["expiry"], "call_token": pair["CE"], "put_token": pair["PE"],
             "quantity": q, "lot_size": int(self.context["lot_size"]), "entry_call": prices[0], "entry_put": prices[1],
             "entry_premium": sum(prices), "scheduled_exit": (now + timedelta(minutes=horizon)).isoformat(),
             "cutoff": datetime.combine(now.date(), time.fromisoformat(self.config["square_off"])).isoformat(),
             "horizon_minutes": horizon, "entry_outputs": signal["outputs"], "thresholds": signal["thresholds"],
             "feature_timestamp": signal["feature_timestamp"], "group": signal["expiry_group"],
             "cost_rates": dict(self.context["cost_rates"]), "pending_exit": None,
             "profit_target_net_rupees": self.config["profit_target_net_rupees"],
             "stop_loss_net_rupees": self.config["stop_loss_net_rupees"],
             "round_trip_slippage_points": self.config["round_trip_slippage_points"]}
        # Record before exposing the open position; replay recovers a crash here.
        self.log(now, "trades", {"event": "ENTRY", "timestamp": now.isoformat(), "position": p})
        self.positions[key] = p
        from voltr.web.alerts import notify_position_action
        notify_position_action()
        return True

    def mark(self, now, quote_lookup):
        for key, p in list(self.positions.items()):
            cutoff, scheduled = datetime.fromisoformat(p["cutoff"]), datetime.fromisoformat(p["scheduled_exit"])
            due = "CUTOFF" if now >= cutoff else "TIME" if now >= scheduled else None
            quotes = quote_lookup([p["call_token"], p["put_token"]], now)
            if quotes is None:
                p["quote_status"] = "STALE_OR_MISSING"
                if due:
                    p["pending_exit"] = due
                continue  # Never claim execution using a stale or synthetic quote.
            premium = sum(quotes)
            fee = costs(p["entry_premium"], premium, p["quantity"], p["cost_rates"], p["round_trip_slippage_points"])
            gross = (p["entry_premium"] - premium) * p["quantity"]
            net = gross - fee["total_cost_rupees"]
            mark = {"timestamp": now.isoformat(), "id": key, "entry_strike": p["entry_strike"], "expiry": p["expiry"],
                    "call_token": p["call_token"], "put_token": p["put_token"], "call_price": quotes[0], "put_price": quotes[1],
                    "gross_rupees": gross, "net_rupees": net, **fee}
            p.update({"last_mark": mark, "quote_status": "FRESH"})
            self.log(now, "marks", mark)
            reason = p.get("pending_exit") or due
            if reason is None:
                reason = "TARGET" if net >= p["profit_target_net_rupees"] else "STOP" if net <= p["stop_loss_net_rupees"] else None
            if reason:
                p.update({"exit_datetime": now.isoformat(), "exit_strike": p["entry_strike"], "exit_expiry": p["expiry"],
                          "exit_call": quotes[0], "exit_put": quotes[1], "exit_reason": reason, "gross_rupees": gross,
                          "net_rupees": net, **fee})
                self.log(now, "trades", {"event": "EXIT", "timestamp": now.isoformat(), "position": p})
                self.closed.append(p)
                del self.positions[key]
                from voltr.web.alerts import notify_position_action
                notify_position_action()

    def status(self):
        closed = self.closed
        total = sum(p["net_rupees"] for p in closed)
        return {"open_positions": list(self.positions.values()), "closed_count": len(closed),
                "green_trades": sum(p["net_rupees"] > 0 for p in closed), "red_trades": sum(p["net_rupees"] < 0 for p in closed),
                "realized_net_rupees": total, "realized_gross_rupees": sum(p["gross_rupees"] for p in closed),
                "realized_cost_rupees": sum(p["total_cost_rupees"] for p in closed),
                "unrealized_net_rupees": sum(p.get("last_mark", {}).get("net_rupees", 0) for p in self.positions.values()),
                "unpriced_open_positions": sum("last_mark" not in p or p.get("quote_status") != "FRESH" for p in self.positions.values()),
                "maximum_open_positions": self.config["maximum_open_positions"], "recent_closed": closed[-100:]}


class PCRRuntime:
    def __init__(self, context, directory, models=None, config=None):
        self.config = config or load_config()
        self.external_config = config is None
        self.context = context
        self.directory = Path(directory)
        self.models = models or Models()
        if context.get("mode") == "recorded_replay" and not any(self.directory.glob("????-??-??/trades.jsonl")):
            # A daily replay must retain unresolved carry-in risk rather than
            # silently start flat and admit trades the live ledger rejected.
            for position in context.get("initial_positions", []):
                append(self.directory / position["entry_datetime"][:10] / "trades.jsonl",
                       {"event": "ENTRY", "timestamp": position["entry_datetime"], "position": position,
                        "replay_carry_in": True})
        self.book = Book(self.directory, self.config, context)
        self.pairs = {float(k): v for k, v in context["pairs"].items()}
        self.quotes, self.oi = {}, {}
        self.spot = self.future = None
        self.future_open = {}
        self.rows = []
        self.pending_row = None
        self.last_time = None
        self.last_signal_minute = None
        self.config_minute = None
        self.context_days = set()
        self.fault = None
        self.latest_signal = None
        self.highwater = None
        self.mode = context.get("mode", "live")
        self.seen = set()
        for file in sorted(self.directory.glob("????-??-??/signals.jsonl")):
            events = read_lines(file)
            self.seen.update(e["id"] for e in events)
            if events:
                self.latest_signal = events[-1]
        self.seen.update(self.book.positions)
        self.seen.update(p["id"] for p in self.book.closed)
        for file in sorted(self.directory.glob("????-??-??/minute_rows.jsonl"))[-1:]:
            self.rows = read_lines(file)[-180:]
        self.checkpoint("WAITING_FOR_FEED")

    def prices(self, tokens, now):
        q = [self.quotes.get(str(token)) for token in tokens]
        if any(x is None for x in q):
            return None
        times = [datetime.fromisoformat(x[1]) for x in q]
        if any(not 0 <= (now - stamp).total_seconds() <= self.config["price_max_age_seconds"] for stamp in times):
            return None
        if (max(times) - min(times)).total_seconds() > self.config["max_leg_timestamp_difference_seconds"]:
            return None
        return [float(x[0]) for x in q]

    def _oi(self, token, now):
        value = self.oi.get(str(token))
        if value and 0 <= (now - datetime.fromisoformat(value[1])).total_seconds() <= self.config["oi_max_age_seconds"]:
            return float(value[0])
        return math.nan

    def current_row(self, now):
        if self.spot is None or self.future is None:
            return None
        if any(not 0 <= (now - datetime.fromisoformat(x[1])).total_seconds() <= self.config["price_max_age_seconds"] for x in (self.spot, self.future)):
            return None
        strike = min(self.pairs, key=lambda k: (abs(k - self.spot[0]), k))
        pair = self.pairs[strike]
        prices = self.prices([pair["CE"], pair["PE"]], now)
        if prices is None:
            return None
        expiry = datetime.combine(datetime.fromisoformat(self.context["expiry"]).date(), time(15, 30))
        minute = now.replace(second=0, microsecond=0)
        dte = (expiry - now).total_seconds() / 86400
        if dte <= 0:
            return None
        result = {"datetime": minute.isoformat(), "observed_at": now.isoformat(), "strike": strike, "atm_strike": strike,
                  "expiry": self.context["expiry"], "days_to_expiry": dte, "spot_price": self.spot[0],
                  "future_open": self.future_open.get(minute.isoformat(), self.future[0]),
                  "call_price": prices[0], "put_price": prices[1], "call_oi": self._oi(pair["CE"], now),
                  "put_oi": self._oi(pair["PE"], now), "Future_oi": self._oi(self.context["future_token"], now)}
        for side, price in zip(("call", "put"), prices):
            stats = option_stats(price, self.spot[0], strike, dte / 365, self.context["risk_free_rate"], side == "call")
            result.update({f"{name}_{side}": value for name, value in stats.items()})
        return result

    def on_snapshot(self, snapshot, processing_delay=0.0):
        snapshot = dict(snapshot)
        if self.mode == "recorded_replay":
            processing_delay = float(snapshot.get("processing_delay_seconds", processing_delay))
        snapshot["processing_delay_seconds"] = processing_delay
        now = datetime.fromisoformat(snapshot["timestamp"])
        if self.last_time and now < self.last_time:
            self.fault = "OUT_OF_ORDER_FEED: restart replay in a fresh session directory"
            self.checkpoint("HALTED")
            return
        minute = now.replace(second=0, microsecond=0)
        if self.external_config and self.mode != "recorded_replay" and self.config_minute != minute:
            updated = load_config()
            snapshot = dict(snapshot, config=updated)
            self.config_minute = minute
        if "config" in snapshot:
            updated = snapshot["config"]
            if updated != self.config:
                self.book.log(now, "config_changes", {"timestamp": now.isoformat(), "before": self.config, "after": updated})
                self.config = updated
                self.book.config = updated
        day = now.date().isoformat()
        if day not in self.context_days:
            context_path = self.directory / day / "session_context.json"
            if not context_path.exists():
                atomic_json(context_path, dict(self.context, initial_config=self.config,
                                               initial_positions=list(self.book.positions.values())))
            self.context_days.add(day)
        append(self.directory / day / "inputs.jsonl", snapshot)
        if snapshot.get("transport_fault"):
            self.fault = snapshot["transport_fault"]
        self.last_time = now
        self.highwater = now.isoformat()
        # Complete the old minute before ingesting the new minute's prices/OI.
        if self.pending_row and self.pending_row["datetime"] < minute.isoformat():
            row = self.pending_row
            self.rows.append(row)
            self.rows = [r for r in self.rows[-181:] if r["datetime"][:10] == now.date().isoformat()]
            self.book.log(datetime.fromisoformat(row["datetime"]), "minute_rows", row)
            self.pending_row = None
        for token, value in snapshot.get("quotes", {}).items():
            if datetime.fromisoformat(value[1]) <= now and (str(token) not in self.quotes or value[1] >= self.quotes[str(token)][1]):
                self.quotes[str(token)] = value
        for token, value in snapshot.get("oi", {}).items():
            if datetime.fromisoformat(value[1]) <= now and (str(token) not in self.oi or value[1] >= self.oi[str(token)][1]):
                self.oi[str(token)] = value
        for key in ("spot", "future"):
            value = snapshot.get(key)
            previous = getattr(self, key)
            if value and value[1] <= now.isoformat() and (previous is None or value[1] >= previous[1]):
                setattr(self, key, value)
        self.future_open.update(snapshot.get("future_open", {}))
        self.future_open = {k: v for k, v in self.future_open.items() if k >= (now - timedelta(hours=3)).isoformat()}
        self.book.mark(now, self.prices)
        row = self.current_row(now) if snapshot.get("market_update", True) else None
        if row:
            self.pending_row = row
        signal_id = f"PCR-{minute:%Y%m%d-%H%M}"
        # Allow the first 10 seconds for asynchronous spot/leg updates. Entries
        # always use the actual observation time, never a retroactive minute fill.
        if signal_id not in self.seen and (row is not None or now.second >= 10):
            self.signal(now, row, signal_id, processing_delay)
        self.checkpoint("HALTED" if self.fault else "RUNNING")

    def signal(self, now, row, key, processing_delay):
        self.seen.add(key)
        signal = {"id": key, "timestamp": now.isoformat(), "outputs": None, "gates": None,
                  "reason": None, "entered": False, "entry_strike": row["strike"] if row else None,
                  "expiry": self.context["expiry"], "feature_timestamp": None,
                  "positions_before": len(self.book.positions), "processing_delay_seconds": processing_delay}
        try:
            minute = now.replace(second=0, microsecond=0)
            today_rows = [r for r in self.rows if r["datetime"][:10] == now.date().isoformat()]
            if row is None:
                raise ValueError("missing_or_stale_entry_quotes")
            if len(today_rows) < self.config["minimum_history_minutes"]:
                raise ValueError(f"warmup_{len(today_rows)}_of_{self.config['minimum_history_minutes']}_minutes")
            features, group, stamp = feature_at(today_rows, minute, row["days_to_expiry"], self.models.names)
            missing = [k for k, v in features.items() if not math.isfinite(v)]
            if len(missing) / len(features) > self.config["max_missing_feature_fraction"]:
                raise ValueError("too_many_missing_features")
            outputs = self.models.predict(features)
            thresholds = dict(self.models.thresholds[group])
            thresholds["minimum_win_probability"] = self.config["win_probability_thresholds"][group]
            horizon = 120 if outputs["holding_120_points"] > outputs["holding_60_points"] + self.config["extension_buffer_points"] else 60
            gates = {"decay": outputs["decay_60_points"] > thresholds["minimum_expected_points"],
                     "win": outputs["win_probability_60m"] >= thresholds["minimum_win_probability"],
                     "tail": outputs["tail_loss_probability"] <= thresholds["maximum_large_loss_probability"]}
            signal.update({"features": features, "missing_features": missing, "feature_timestamp": stamp,
                           "expiry_group": group, "outputs": outputs, "thresholds": thresholds,
                           "gates": gates, "horizon_minutes": horizon, "qualified": all(gates.values()),
                           "entry_call": row["call_price"], "entry_put": row["put_price"]})
            if not self.config["enabled"]:
                reason = "configuration_disabled"
            elif self.fault:
                reason = "transport_or_feed_fault"
            elif (self.mode == "live" or self.context.get("source_mode") == "live") and (processing_delay < -1 or processing_delay > self.config["max_processing_delay_seconds"]):
                reason = "processing_delay_or_clock_skew"
            elif not time.fromisoformat(self.config["entry_start"]) <= now.time() <= time.fromisoformat(self.config["entry_end"]):
                reason = "outside_entry_window"
            elif not all(gates.values()):
                reason = "model_gate_rejected"
            elif any(p["entry_datetime"][:10] != now.date().isoformat() for p in self.book.positions.values()):
                reason = "unresolved_prior_day_position"
            elif len(self.book.positions) >= self.config["maximum_open_positions"]:
                reason = "capacity_full"
            else:
                reason = "entered"
            signal["reason"] = reason
            # Journal intent first; restart will never duplicate this minute.
            self.book.log(now, "signal_intents", signal)
            if reason == "entered":
                signal["entered"] = self.book.enter(now, row["strike"], self.pairs[row["strike"]],
                                                     [row["call_price"], row["put_price"]], horizon, signal)
                self.book.log(now, "signal_actions", {"id": key, "entered": signal["entered"]})
                if not signal["entered"]:
                    signal["reason"] = "entry_not_accepted"
            self.book.log(now, "signals", signal)
        except ValueError as error:
            signal["reason"] = str(error)
            self.book.log(now, "signals", signal)
        self.latest_signal = clean(signal)
        if signal.get('qualified'):
            from voltr.web.alerts import notify_signal
            notify_signal()

    def checkpoint(self, state):
        snapshot = {"state": state, "fault": self.fault, "updated_at": self.highwater,
                    "paper_only": True, "mode": self.mode, "config": self.config,
                    "context": self.context, "latest_signal": self.latest_signal,
                    "history_minutes": len(self.rows), "thresholds": self.models.thresholds,
                    "models": {k: b.get("family") for k, b in getattr(self.models, "bundles", {}).items()},
                    "book": self.book.status()}
        atomic_json(self.directory / "status.json", snapshot)
        return snapshot
