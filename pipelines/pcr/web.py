"""PCR dashboard routes; stdlib only, no change to the original dashboard API."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

from .storage import atomic_json, read_lines, read_text

ROOT = Path(__file__).resolve().parent
TOKEN = secrets.token_urlsafe(32)
JOB_LOCK = threading.Lock()
JOB = None


def read_json(path, default=None):
    try:
        return json.loads(read_text(path))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def date_arg(query):
    day = query.get("date", [datetime.now(ZoneInfo("Asia/Kolkata")).date().isoformat()])[0]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", day):
        raise ValueError("Invalid date")
    datetime.fromisoformat(day)
    return day


def rows(kind, query):
    directory = ROOT / "logs" / date_arg(query)
    if kind == "signals":
        data = read_lines(directory / "signals.jsonl")
        entries = {e["position"]["id"] for e in read_lines(directory / "trades.jsonl") if e["event"] == "ENTRY"}
        for row in data:
            row["entered"] = row["id"] in entries
        return data
    if kind == "trades":
        return read_lines(directory / "trades.jsonl")
    if kind == "marks":
        return read_lines(directory / "marks.jsonl")
    raise ValueError("Unknown log kind")


def job_status():
    if JOB is None:
        return {"state": "IDLE"}
    code = JOB["process"].poll()
    result = {k: v for k, v in JOB.items() if k not in {"process", "handle"}}
    result["state"] = "RUNNING" if code is None else "COMPLETE" if code == 0 else "FAILED"
    result["exit_code"] = code
    folder = Path(result["output"])
    if (folder / "summary.csv").exists():
        with (folder / "summary.csv").open(encoding="utf-8") as f:
            result["summary"] = next(csv.DictReader(f), {})
    if (folder / "status.json").exists():
        result["replay_status"] = read_json(folder / "status.json")
    log = folder / "job.log"
    if code is not None and log.exists():
        result["log_tail"] = log.read_text(encoding="utf-8", errors="replace")[-4000:]
    return result


def start_job(body):
    global JOB
    with JOB_LOCK:
        if JOB and JOB["process"].poll() is None:
            raise ValueError("A PCR backtest is already running")
        kind = body.get("kind", "reference")
        if kind not in {"reference", "recorded"}:
            raise ValueError("Unknown backtest kind")
        identifier = datetime.now().strftime("%Y%m%d_%H%M%S_") + secrets.token_hex(3)
        output = ROOT / "backtests" / identifier
        command = [sys.executable, "-u", "-m", __package__ + ".backtest", "--output", str(output)]
        if kind == "recorded":
            day = date_arg({"date": [str(body.get("date", ""))]})
            inputs = ROOT / "logs" / day / "inputs.jsonl"
            context = ROOT / "logs" / day / "session_context.json"
            if not inputs.is_file() or not context.is_file():
                raise ValueError("No recorded PCR inputs/context for this date")
            command = [sys.executable, "-u", "-m", __package__ + ".worker", "--context", str(context),
                       "--inputs", str(inputs), "--output", str(output)]
        output.mkdir(parents=True)
        handle = (output / "job.log").open("w", encoding="utf-8")
        try:
            process = subprocess.Popen(command, cwd=str(ROOT.parents[1]), stdout=handle, stderr=handle,
                                       env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"),
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except Exception:
            handle.close()
            raise
        handle.close()
        JOB = {"id": identifier, "kind": kind, "output": str(output), "process": process}
        atomic_json(output / "request.json", body)
        return job_status()


def send(handler, value, status=200):
    handler._send(json.dumps(value, allow_nan=False).encode("utf-8"), "application/json; charset=utf-8", status)


def handle_get(handler, parsed):
    if parsed.path not in {"/PCR", "/PCR/"} and not parsed.path.startswith("/PCR/"):
        return False
    try:
        query = parse_qs(parsed.query)
        route = parsed.path.rstrip("/")
        if route == "/PCR":
            handler._send((ROOT / "dashboard.html").read_bytes(), "text/html; charset=utf-8")
        elif route == "/PCR/api/status":
            state = read_json(ROOT / "logs" / "status.json", {"state": "NOT_STARTED", "book": {}})
            state["bridge"] = read_json(ROOT / "bridge_status.json", {"state": "NOT_STARTED"})
            state["desired_config"] = read_json(ROOT / "config.json")
            state["csrf_token"] = TOKEN
            state["dates"] = sorted(p.name for p in (ROOT / "logs").glob("????-??-??") if p.is_dir())
            state["reference_summary"] = {}
            path = ROOT / "reference" / "max3_original" / "summary.csv"
            if path.exists():
                with path.open(encoding="utf-8") as f:
                    state["reference_summary"] = next(csv.DictReader(f), {})
            send(handler, state)
        elif route in {"/PCR/api/signals", "/PCR/api/trades", "/PCR/api/marks"}:
            kind = route.rsplit("/", 1)[1]
            data = rows(kind, query)
            offset = max(0, int(query.get("offset", [0])[0]))
            limit = min(500, max(1, int(query.get("limit", [100])[0])))
            data.reverse()
            send(handler, {"total": len(data), "offset": offset, "limit": limit, "rows": data[offset:offset + limit]})
        elif route == "/PCR/api/download":
            kind = query.get("kind", ["signals"])[0]
            data = rows(kind, query)
            body = "".join(json.dumps(row, allow_nan=False) + "\n" for row in data)
            handler._send(body.encode("utf-8"), "application/x-ndjson")
        elif route == "/PCR/api/backtest":
            send(handler, job_status())
        else:
            send(handler, {"error": "PCR route not found"}, 404)
    except (OSError, ValueError, KeyError) as error:
        send(handler, {"error": str(error)}, 400)
    return True


def handle_post(handler, parsed):
    if not parsed.path.startswith("/PCR/"):
        return False
    try:
        origin = handler.headers.get("Origin")
        if origin and urlparse(origin).netloc != handler.headers.get("Host"):
            raise ValueError("Cross-origin request refused")
        if handler.headers.get("X-PCR-Token") != TOKEN:
            send(handler, {"error": "PCR page token required"}, 403)
            return True
        length = int(handler.headers.get("Content-Length", 0))
        if not 0 < length <= 4096:
            raise ValueError("Invalid request size")
        body = json.loads(handler.rfile.read(length))
        if parsed.path == "/PCR/api/backtest":
            send(handler, start_job(body), 202)
        elif parsed.path == "/PCR/api/thresholds":
            values = body["win_probability_thresholds"]
            if set(values) != {"0", "1", "2", "3_4"} or any(type(v) not in (int, float) or not 0 <= v <= 1 for v in values.values()):
                raise ValueError("Supply four numeric win probabilities between 0 and 1")
            config = read_json(ROOT / "config.json")
            config["win_probability_thresholds"] = values
            atomic_json(ROOT / "config.json", config)
            send(handler, {"ok": True, "applies": "next PCR signal; existing positions keep their exit rules", "win_probability_thresholds": values})
        else:
            send(handler, {"error": "PCR route not found"}, 404)
    except (ValueError, KeyError, TypeError, OSError) as error:
        send(handler, {"error": str(error)}, 400)
    return True


def main():
    # Offline inspection only. The normal startup serves PCR on the host's 8780.
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8781)
    args = p.parse_args()
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        def do_GET(self):
            if not handle_get(self, urlparse(self.path)):
                self.send_error(404)
        def do_POST(self):
            if not handle_post(self, urlparse(self.path)):
                self.send_error(404)
        def log_message(self, *_):
            pass
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
