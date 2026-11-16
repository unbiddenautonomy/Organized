"""Replay a prepared IVRV date into an isolated paper dashboard.

The source files contain one quote observation per minute, so paper marks in
this mock run are minute-resolution observations rather than every market tick.
Nothing here opens market sockets or writes to the scheduled service's logs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from datetime import date, datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "banknifty"
MOCK_OUTPUT_ROOT = ROOT / "runtime" / "mock_sessions"
IVROOT = ROOT / "pipelines" / "families" / "ivrv" / "engine" / "Inference"
sys.path.insert(0, str(IVROOT))
from live import live_inference as live

MOCK_HTML = Path(__file__).with_name("mock_session_dashboard.html")


def exchange_second(stamp: datetime) -> int:
    return int((stamp - live.NSE_EPOCH).total_seconds())


def load_records(data_dir: Path, inference_path: Path):
    futures = {}
    with (data_dir / "banknifty_futures_1min.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["is_near_expiry"] != "True":
                continue
            minute = datetime.fromisoformat(row["minute"])
            futures[minute] = dict(spot=float(row["last_traded_price"]), expiry=date.fromisoformat(row["expiry_master"]),
                                   future_token=int(row["token"]), tick=datetime.fromisoformat(row["source_tick_time"]))
    options = {}
    pairs = {}
    with (data_dir / "banknifty_option_chain_1min.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row["last_traded_price"] or not row["source_tick_time"]:
                continue
            minute = datetime.fromisoformat(row["minute"])
            strike = float(row["strike"])
            side = row["option_type"]
            token = int(row["option_token"])
            options.setdefault(minute, []).append(dict(token=token, strike=strike, side=side,
                price=float(row["last_traded_price"]), tick=datetime.fromisoformat(row["source_tick_time"])))
            pairs.setdefault(strike, {})[side] = token
    predictions = {}
    waits = {}
    with inference_path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            minute = datetime.fromisoformat(record["minute"])
            if record["state"] == "READY":
                when = datetime.fromisoformat(record["decision_time"])
                predictions[when] = record
            else:
                waits[minute] = record.get("reason", "no_model_output")
    if not futures or not options:
        raise ValueError("Prepared data has no future or option minutes")
    if set(options) - set(futures):
        raise ValueError("Option minutes lack the near future")
    return futures, options, pairs, predictions, waits


class EntryContext:
    _fresh_paper_prices = live.LiveFeatureInference._fresh_paper_prices
    _try_open_paper_position = live.LiveFeatureInference._try_open_paper_position

    def __init__(self, paper_engine, pairs, expiry, config):
        self.paper_engine = paper_engine
        self.pairs = pairs
        self.expiry = expiry
        self.config = config
        self.latest_option_ticks = {}
        self.latest_exchange_second = None


class MockSession:
    def __init__(self, data_dir: Path, inference_path: Path, output: Path, speed: float):
        if speed < 0:
            raise ValueError("speed must be nonnegative")
        output = output.resolve()
        if not (output.is_relative_to(MOCK_OUTPUT_ROOT.resolve()) or output.is_relative_to(DATA_ROOT.resolve())):
            raise ValueError("mock output must stay under runtime/mock_sessions or datasets/banknifty")
        if output.exists() and any(output.iterdir()):
            raise ValueError(f"mock output must be empty: {output}")
        self.data_dir = data_dir
        self.inference_path = inference_path
        self.futures, self.options, pairs, self.predictions, self.waits = load_records(data_dir, inference_path)
        self.minutes = sorted(self.options)
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.speed = speed
        self.lock = threading.RLock()
        self.recent = deque(maxlen=100)
        self._trade_read_offset = 0
        self._realized_gross_pnl = 0.0
        self._highest_probability = None
        self.status = dict(mode="MOCK_REPLAY", state="READY", source_date=self.minutes[0].date().isoformat(),
                           virtual_time=None, processed_minutes=0, total_minutes=len(self.minutes),
                           spot_proxy=None, latest_probability=None, latest_route=None, latest_decision=None,
                           last_wait_reason=None, paper_trading={}, realized_gross_pnl=0.0,
                           highest_probability=None, entries_taken=0,
                           completed=False, source_resolution="one quote observation per minute")
        self.config = json.loads((IVROOT / "live" / "live_config.json").read_text(encoding="utf-8"))
        self.config.update(paper_trade_events_log="paper_trade_events.jsonl", paper_position_state="paper_position_state.json")
        os.environ["POSITION_ALERT_DISABLED"] = "1"
        live.BASE_DIR = self.output
        expiry = self.futures[self.minutes[0]]["expiry"]
        lot_size = 30
        self.paper = live.PaperTradingEngine(self.config, lot_size)
        self.entry = EntryContext(self.paper, pairs, expiry, self.config)
        self.events = (self.output / "mock_events.jsonl").open("w", encoding="utf-8", buffering=1)
        (self.output / "run_metadata.json").write_text(json.dumps(dict(
            mode="MOCK_REPLAY", source_data=str(data_dir.resolve()), source_inference=str(inference_path.resolve()),
            source_resolution="one quote observation per minute", spot_source="near_future_price_proxy",
            output=str(self.output), live_service_modified=False, speed_seconds_per_minute=speed,
            entry_threshold=self.paper.convergence_probability_threshold,
            replacement_improvement=self.paper.replacement_min_improvement,
            replacement_cooldown_seconds=self.paper.replacement_cooldown_seconds,
            maximum_replacements_per_day=self.paper.max_replacements_per_day), indent=2), encoding="utf-8")

    def _fresh(self, position, second):
        return self.entry._fresh_paper_prices(int(position["call_token"]), int(position["put_token"]), second)

    def _mark_open(self, second):
        for position in list(self.paper.positions.values()):
            prices = self._fresh(position, second)
            if prices is not None:
                self.paper.mark(position["position_id"], second, *prices)

    def _sync_realized(self):
        path = self.output / "paper_trade_events.jsonl"
        with path.open(encoding="utf-8") as handle:
            handle.seek(self._trade_read_offset)
            for line in handle:
                record = json.loads(line)
                if record.get("event_type") == "paper_trade_exit":
                    self._realized_gross_pnl += float(record["realized_pnl_inr"])
            self._trade_read_offset = handle.tell()

    def step(self, minute: datetime):
        # The prepared chain stores first observed tick per minute and carried quotes.
        # The source tick timestamp, not the CSV row time, controls freshness.
        now = minute + timedelta(seconds=10)
        second = exchange_second(now)
        self.entry.latest_exchange_second = second
        for quote in self.options[minute]:
            self.entry.latest_option_ticks[quote["token"]] = (quote["price"], exchange_second(quote["tick"]))
        self._mark_open(second)
        record = self.predictions.get(minute)
        decision = None
        probability = None
        route = None
        if record is not None:
            outputs = record["outputs"]
            probability = outputs.get("pred_convergence_probability_classifier")
            if probability is not None:
                self._highest_probability = max(float(probability), self._highest_probability or 0.0)
            decision = self.entry._try_open_paper_position(exchange_second(datetime.fromisoformat(record["minute"])) // 60,
                dict(entry_strike=record["strike"]), outputs)
            route = decision.get("route")
            if route is None:
                gap = outputs.get("pred_no120_gate_iv_minus_rv")
                if gap is not None:
                    near = self.paper.iv_rv_near_gap_decimal
                    route = "iv_above_rv_convergence" if gap > near else (
                        "iv_below_rv_sharp_rv_decline" if gap < -near else "iv_near_rv_sharp_rv_decline")
        self._sync_realized()
        event = dict(virtual_time=now.isoformat(sep=" "), spot_proxy=self.futures[minute]["spot"],
                     prediction_probability=probability, route=route, decision=decision,
                     model_state="READY" if record else "WARMUP_OR_NO_INFERENCE",
                     wait_reason=None if record else self.waits.get(minute - timedelta(minutes=2), "no_model_output"),
                     paper_trading=self.paper.snapshot())
        self.events.write(json.dumps(event, default=str, allow_nan=False) + "\n")
        with self.lock:
            self.recent.appendleft(event)
            self.status.update(state="PLAYING", virtual_time=now.isoformat(sep=" "),
                processed_minutes=self.status["processed_minutes"] + 1,
                spot_proxy=self.futures[minute]["spot"], latest_probability=probability,
                latest_route=route, latest_decision=decision, last_wait_reason=event["wait_reason"],
                paper_trading=event["paper_trading"], highest_probability=self._highest_probability,
                entries_taken=self.status["entries_taken"] + int(bool(decision and decision.get("entered"))),
                realized_gross_pnl=self._realized_gross_pnl)
        return event

    def run(self):
        try:
            for minute in self.minutes:
                self.step(minute)
                if self.speed:
                    time.sleep(self.speed)
            with self.lock:
                self.status.update(state="COMPLETE", completed=True)
        except Exception as exc:
            with self.lock:
                self.status.update(state="ERROR", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.paper.close()
            self.events.close()


def serve(session: MockSession, host: str, port: int):
    current = {"session": session, "worker": None}
    current_lock = threading.Lock()

    def start_worker(item):
        worker = threading.Thread(target=item.run, daemon=True)
        worker.start()
        current["worker"] = worker

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlparse(self.path).path
            with current_lock:
                item = current["session"]
            if path == "/api/status":
                with item.lock:
                    body = json.dumps(item.status, allow_nan=False).encode()
                content_type = "application/json"
            elif path == "/api/events":
                with item.lock:
                    body = json.dumps(list(item.recent), allow_nan=False).encode()
                content_type = "application/json"
            elif path == "/":
                body = MOCK_HTML.read_bytes()
                content_type = "text/html; charset=utf-8"
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urlparse(self.path).path != "/api/restart":
                self.send_error(404)
                return
            with current_lock:
                old = current["session"]
                with old.lock:
                    finished = old.status["completed"] or old.status["state"] == "ERROR"
                if not finished:
                    self.send_error(409, "Replay is still playing")
                    return
                output = old.output.parent / f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
                new = MockSession(old.data_dir, old.inference_path, output, old.speed)
                current["session"] = new
                start_worker(new)
            body = json.dumps({"ok": True, "output": str(output)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    start_worker(session)
    print(f"Mock dashboard http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        current["worker"].join(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-11-03", help="saved market date, YYYY-MM-DD")
    parser.add_argument("--data", type=Path, help="prepared one-minute data directory")
    parser.add_argument("--inference", type=Path, help="saved IVRV inference.jsonl for the selected date")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--seconds-per-minute", type=float, default=0.3)
    args = parser.parse_args()
    selected_date = date.fromisoformat(args.date)
    data_dir = args.data or DATA_ROOT / selected_date.isoformat()
    if args.inference:
        inference_path = args.inference
    else:
        candidates = sorted((data_dir / "inference_replay" / "runs").glob("*/ivrv/inference.jsonl"))
        if len(candidates) != 1:
            parser.error(f"found {len(candidates)} inference files under {data_dir}; pass --inference")
        inference_path = candidates[0]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output or MOCK_OUTPUT_ROOT / selected_date.isoformat() / f"run_{stamp}"
    session = MockSession(data_dir, inference_path, output, args.seconds_per_minute)
    serve(session, args.host, args.port)


if __name__ == "__main__":
    main()
