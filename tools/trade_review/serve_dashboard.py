"""Serve the live review on loopback and read the existing demo ledger."""
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import math
import replay_stop
from functools import lru_cache
from threading import Lock

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
# IVRV writes its logs under pipelines/ivrv/logs (live_config.json prediction_log).
PREDICTIONS = REPO / "pipelines/ivrv/logs/live_predictions.jsonl"
DEMO_LEDGER_NAME = "live_predictions_demo_positions.json"


@lru_cache(maxsize=2)
def market_history(modified, size, path=PREDICTIONS, session=None):
    minutes = {}
    with path.open("rb") as source:
        for line in source:
            try:
                event = json.loads(line)
            except ValueError:
                if not line.endswith(b"\n"):
                    break
                raise
            minute = event.get("exchange_minute")
            if not minute:
                continue
            market = event.get("market") or {}
            signal = event.get("strategy_signal") or {}
            pred = event.get("predictions") or {}
            values = [market.get("spot"), market.get("combined_iv"), signal.get("current_rv"),
                      market.get("straddle_price"), pred.get("pred_combined_iv_120m_ensemble"),
                      pred.get("pred_rv_120m_xgboost"), event.get("fixed_entry_strike")]
            minutes[minute] = [minute, *[v if isinstance(v,(float,int)) and math.isfinite(v) else None for v in values]]
    latest_day = session or max(minutes, default="")[:10]
    return json.dumps([minutes[m] for m in sorted(minutes) if m.startswith(latest_day)]).encode()


def read_events(path):
    with path.open("rb") as source:
        for line in source:
            try:
                yield json.loads(line)
            except ValueError:
                if not line.endswith(b"\n"):
                    break
                raise


@lru_cache(maxsize=2)
def execution_history(signatures, session=None):
    trades = {}
    events = {}
    latest_open = {}
    observations = []
    for e in read_events(signatures[0][0]):
        at = e.get("exchange_minute")
        if not at:
            continue
        observations.append([at, (e.get("market") or {}).get("combined_iv"),
                             (e.get("strategy_signal") or {}).get("current_rv"), e.get("fixed_entry_strike")])
        for book in ("paper_trading", "paper_long_trading"):
            for pos in (e.get(book) or {}).get("open_positions", []):
                key = (pos["side"], pos["position_id"])
                latest_open[key] = pos
                entry = trades.setdefault(key, {"marks": {}})
                mark = pos.get("last_mark_datetime")
                if mark and pos.get("unrealized_pnl_inr") is not None:
                    entry["marks"][mark] = pos["unrealized_pnl_inr"]
    for file, *_ in signatures[1:]:
        for pos in read_events(file):
            if pos.get("event_type") not in ("paper_trade_entry","paper_trade_exit"):
                continue
            key = (pos["side"], pos["position_id"])
            events[key] = pos if pos.get("exit_datetime") else latest_open.get(key,pos)
            entry = trades.setdefault(key, {"marks": {}})
            entry.update(side=pos["side"], start=pos["entry_datetime"], strike=pos["fixed_strike"])
            entry["marks"][pos["entry_datetime"]] = 0
            if pos.get("exit_datetime"):
                entry.update(end=pos["exit_datetime"], reason=pos.get("exit_reason"), pnl=pos.get("realized_pnl_inr"))
                entry["marks"][pos["exit_datetime"]] = entry["pnl"]
    day = session or max([row[0][:10] for row in observations]+[e["start"][:10] for e in trades.values() if "start" in e],default="")
    records = []
    for entry in trades.values():
        if entry.get("start","")[:10] != day or "start" not in entry:
            continue
        entry["marks"] = sorted([t,v] for t,v in entry["marks"].items() if entry["start"] <= t <= entry.get("end","9999"))
        records.append(entry)
    market = {row[0]:row for row in observations if row[0].startswith(day)}
    replay_stop.apply(records, {(p["side"],p["entry_datetime"],p["fixed_strike"]):p for p in events.values()
                               if p["entry_datetime"][:10]==day},signatures[0][0].parent,PREDICTIONS.parent,day)
    return json.dumps({"date":day,"market":[market[t] for t in sorted(market)],
                       "trades":sorted(records,key=lambda e:(e["start"],e["side"]))},allow_nan=False).encode()


def executions():
    files = [PREDICTIONS, *sorted(PREDICTIONS.parent.glob("paper_trade_events*.jsonl"))]
    return execution_history(tuple((f,f.stat().st_mtime_ns,f.stat().st_size) for f in files))


class CutoffMarks:
    """Index completed 15:15 observations while the fixed-strike path log grows."""
    def __init__(self):
        self.lock = Lock()
        self.offset = 0
        self.signature = None
        self.quotes = {}

    def read(self, path=None):
        path = path or PREDICTIONS.with_name("fixed_strike_120m_paths.jsonl")
        with self.lock, path.open("rb") as f:
            stat = path.stat()
            signature = (stat.st_ino, stat.st_size, stat.st_mtime_ns)
            if self.signature and (signature[0] != self.signature[0] or
                    stat.st_size < self.signature[1] or
                    (stat.st_size == self.signature[1] and signature != self.signature)):
                self.quotes.clear()
                self.offset = 0
            f.seek(self.offset)
            for line in f:
                if not line.endswith(b"\n"):
                    break
                if b"15:15:00" in line:
                    r = json.loads(line)
                    t = r.get("observation_datetime", "")
                    if t[11:] == "15:15:00" and r.get("status") == "complete":
                        key = (t[:10],r["expiry"],r["entry_strike"])
                        price = r.get("observation_straddle_price")
                        if isinstance(price,(int,float)) and math.isfinite(price) and price>=0:
                            self.quotes.setdefault(key,set()).add(round(price,8))
                self.offset += len(line)
            self.signature = signature
            return json.dumps([[*key,next(iter(prices)) if len(prices)==1 else None]
                               for key,prices in sorted(self.quotes.items())]).encode()


cutoff_marks = CutoffMarks()


class Dashboard(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # The windowless dashboard process has no stderr stream.
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        try:
            if path == "/":
                # Today's review from the saved logs; its panels then poll the routes below.
                import generate_daily
                today = datetime.now(timezone(timedelta(hours=5, minutes=30))).date().isoformat()
                data = generate_daily.render(today).encode("utf-8")
                mime = "text/html; charset=utf-8"
            elif path == "/positions":
                # The IVRV engine keeps this ledger; read it directly so the review
                # works in both the single-process and the isolated service mode.
                data = PREDICTIONS.with_name(DEMO_LEDGER_NAME).read_bytes()
                mime = "application/json"
            elif path == "/market":
                stat = PREDICTIONS.stat()
                data = market_history(stat.st_mtime_ns, stat.st_size)
                mime = "application/json"
            elif path == "/executions":
                data = executions()
                mime = "application/json"
            elif path == "/cutoff-marks":
                data = cutoff_marks.read()
                mime = "application/json"
            else:
                self.send_error(404)
                return
        except (OSError, ValueError):
            self.send_error(503, "Live data unavailable")
            return
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8795), Dashboard).serve_forever()
