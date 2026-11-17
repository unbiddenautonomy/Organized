"""Small, nonblocking feed tee. This module intentionally imports no ML library.

The existing application owns UDP reception. PCR receives one-second batches
on a bounded queue and runs inference in a separate Python process. Failure or
overflow disables PCR entries and never changes the parent's strategy state.
"""
from __future__ import annotations
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

from voltr.core.clock import NSE_EPOCH

from .storage import atomic_json, read_text

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("PCR.bridge")


class Bridge:
    def __init__(self, parent_config, contracts, lot_size, expiry, future_token, source):
        self.settings = json.loads(read_text(ROOT / "config.json"))
        self.enabled = bool(self.settings["enabled"])
        self.source = source
        self.future_token = future_token
        self.price_divisor = float(parent_config.get("price_divisor", 100))
        self.contracts = contracts
        self.replay = parent_config.get("runtime_mode") == "historical_udp_replay"
        mode = "historical_udp_replay" if self.replay else "live"
        rates = {k: float(parent_config[k]) for k in (
            "stt_sell_premium_rate", "nse_options_transaction_rate", "sebi_turnover_rate",
            "stamp_buy_premium_rate", "gst_rate")}
        pairs = {}
        for c in contracts.values():
            pairs.setdefault(str(float(c.strike_price)), {})[c.option_type] = int(c.token)
        pairs = {k: v for k, v in pairs.items() if {"CE", "PE"} <= v.keys()}
        self.context = {"mode": mode, "lot_size": int(lot_size), "expiry": expiry.isoformat(),
                        "future_token": int(future_token), "pairs": pairs, "cost_rates": rates,
                        "risk_free_rate": float(parent_config.get("risk_free_rate", .065)),
                        "contract_source": str(parent_config.get("contract_file")),
                        "session_date": str(parent_config.get("contract_asof_date", datetime.now().date().isoformat()))}
        self.directory = ROOT / "logs" if not self.replay else ROOT / "replays" / datetime.now().strftime("udp_%Y%m%d_%H%M%S")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.context_path = self.directory / "session_context.json"
        atomic_json(self.context_path, self.context)
        self.queue = queue.Queue(maxsize=int(self.settings["transport_queue_capacity"]))
        self.process = None
        self.writer = None
        self.log_handle = None
        self.batch = None
        self.second = None
        self.closed = False
        self.transport_fault = None
        self.future_minute = None
        self.oi_versions = {}
        self.last_pulse = 0.0

    def start(self):
        if not self.enabled or self.process:
            return
        self.log_handle = (self.directory / "worker.log").open("a", encoding="utf-8")
        environment = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        self.process = subprocess.Popen(
            [sys.executable, "-u", "-m", "PCR.worker", "--context", str(self.context_path), "--output", str(self.directory)],
            cwd=str(ROOT.parent), stdin=subprocess.PIPE, stdout=self.log_handle, stderr=self.log_handle,
            text=True, encoding="utf-8", bufsize=1, env=environment,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.writer = threading.Thread(target=self._write, name="PCR-feed-writer", daemon=True)
        self.writer.start()
        atomic_json(ROOT / "bridge_status.json", {"state": "STARTED", "pid": self.process.pid, "mode": self.context["mode"], "output": str(self.directory)})

    def _write(self):
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                if self.transport_fault:
                    item["transport_fault"] = self.transport_fault
                self.process.stdin.write(json.dumps(item, allow_nan=False) + "\n")
                self.process.stdin.flush()
            self.process.stdin.close()
        except (OSError, ValueError, BrokenPipeError):
            LOGGER.exception("PCR worker transport stopped; parent strategy continues")
            self.transport_fault = "WORKER_TRANSPORT_FAILED"
        finally:
            if self.process.stdin and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except OSError:
                    pass

    def _flush(self):
        if not self.batch or not self.process:
            return
        try:
            self.queue.put_nowait(self.batch)
        except queue.Full:
            self.transport_fault = "FEED_QUEUE_OVERFLOW: restart PCR after checking worker"
            atomic_json(ROOT / "bridge_status.json", {"state": "FAULT", "error": self.transport_fault})
        self.batch = None

    def _at(self, second):
        if not self.enabled or not self.process or self.closed:
            return None
        second = int(second)
        if self.second is not None and second < self.second:
            # CM, FO and OI streams can arrive out of order. Accept a late
            # observation at the current clock, retaining its original field
            # timestamp below. The worker never revises a completed minute.
            second = self.second
        if self.second != second:
            self._flush()
            self.second = second
        if self.batch is None:
            self.batch = {"timestamp": (NSE_EPOCH + timedelta(seconds=second)).isoformat(),
                          "quotes": {}, "oi": {}, "future_open": {}, "market_update": False}
        return self.batch

    def index_message(self, message):
        decoded = self.source["decode_banknifty_7207"](message, self.price_divisor)
        if decoded:
            second, spot = decoded
            b = self._at(second)
            if b is not None:
                value = [float(spot), (NSE_EPOCH + timedelta(seconds=int(second))).isoformat()]
                if "spot" not in b or value[1] >= b["spot"][1]:
                    b["spot"] = value
                b["market_update"] = True

    def market(self, update):
        if update.last_traded_price <= 0:
            return
        b = self._at(update.log_time)
        if b is None:
            return
        value = [float(update.last_traded_price), (NSE_EPOCH + timedelta(seconds=int(update.log_time))).isoformat()]
        if update.token == self.future_token:
            if "future" not in b or value[1] >= b["future"][1]:
                b["future"] = value
            minute = int(update.log_time) // 60
            if self.future_minute is None or minute > self.future_minute:
                b["future_open"][(NSE_EPOCH + timedelta(minutes=minute)).isoformat()] = value[0]
                self.future_minute = minute
        else:
            token = str(update.token)
            if token not in b["quotes"] or value[1] >= b["quotes"][token][1]:
                b["quotes"][token] = value
        b["market_update"] = True

    def oi_message(self, message):
        try:
            for update in self.source["decode_oi_message"](message):
                if update.token not in self.contracts and update.token != self.future_token:
                    continue
                version = (int(update.log_time), {7201:1,7202:2,17201:3,17202:4}.get(getattr(update,"transaction_code",0),0))
                if self.oi_versions.get(update.token, (-1,-1)) > version:
                    continue
                b = self._at(update.log_time)
                if b is not None:
                    self.oi_versions[update.token] = version
                    b["oi"][str(update.token)] = [int(update.open_interest), (NSE_EPOCH + timedelta(seconds=int(update.log_time))).isoformat()]
        except self.source["OiDecodeError"]:
            return

    def pulse(self):
        if time.monotonic() - self.last_pulse < 1.0:
            return
        self.last_pulse = time.monotonic()
        if self.process and self.process.poll() is not None:
            self.enabled = False
            self.transport_fault = "WORKER_EXITED"
            atomic_json(ROOT / "bridge_status.json", {"state": "FAILED", "exit_code": self.process.returncode, "output": str(self.directory)})
            return
        if self.replay:
            # Feed dates, not today's wall clock, drive historical replay.
            self._flush()
            return
        now = datetime.now()  # Host uses India time; worker also verifies timestamps.
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("Asia/Kolkata")).replace(tzinfo=None)
        sec = int((now - NSE_EPOCH).total_seconds())
        # Flush without advancing the feed clock unless feed has been silent.
        self._flush()
        if self.second is not None and sec > self.second + 1:
            b = self._at(sec)
            if b is not None:
                self._flush()

    def close(self):
        if self.closed:
            return
        self._flush()
        self.closed = True
        if self.writer:
            try:
                self.queue.put(None, timeout=2)
            except queue.Full:
                pass
            self.writer.join(timeout=3)
        if self.process:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=3)
        if self.log_handle:
            self.log_handle.close()
        atomic_json(ROOT / "bridge_status.json", {"state": "STOPPED", "output": str(self.directory)})


def create_bridge(*args):
    try:
        return Bridge(*args)
    except Exception:
        LOGGER.exception("PCR initialization failed; parent strategy continues")
        return None


def safe_call(bridge, method, *args):
    if bridge is None:
        return
    try:
        getattr(bridge, method)(*args)
    except Exception:
        LOGGER.exception("PCR %s failed; parent strategy continues", method)
        bridge.transport_fault = "BRIDGE_ERROR"
