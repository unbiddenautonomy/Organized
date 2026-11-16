"""Local HTTP API for the live inference output."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "runtime" / "predictions.jsonl"
DEFAULT_WORKER_PID = ROOT / "runtime" / "inference.pid"
DEFAULT_API_PID = ROOT / "runtime" / "inference_api.pid"
DEFAULT_TRADE_INTENTS = ROOT / "runtime" / "trade_intents.jsonl"


def records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    result = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.append(json.loads(line))
    return result


def pid_running(pid_path: Path) -> bool:
    try:
        pid = int(pid_path.read_text(encoding="ascii").strip())
        if os.name == "nt":
            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            ctypes.windll.kernel32.CloseHandle(handle)
            return True
        os.kill(pid, 0)
        return True
    except (FileNotFoundError, ValueError, OSError):
        return False


def handler_factory(predictions: Path, worker_pid: Path, trade_intents: Path = DEFAULT_TRADE_INTENTS):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
            request = urlparse(self.path)
            data = records(predictions)
            if request.path == "/health":
                self.send_json(200, {"status": "ok", "worker_running": pid_running(worker_pid), "records": len(data)})
            elif request.path == "/latest":
                self.send_json(200 if data else 404, data[-1] if data else {"error": "no predictions yet"})
            elif request.path == "/predictions":
                try:
                    limit = max(1, min(int(parse_qs(request.query).get("limit", [100])[0]), 1000))
                except ValueError:
                    self.send_json(400, {"error": "limit must be an integer"})
                    return
                self.send_json(200, data[-limit:])
            elif request.path == "/trade-intent":
                intents = records(trade_intents)
                self.send_json(200 if intents else 404, intents[-1] if intents else {"error":"no trade intent yet"})
            elif request.path == "/":
                self.send_json(200, {"service": "volpred-live", "endpoints": ["/health", "/latest", "/predictions?limit=100", "/trade-intent"]})
            else:
                self.send_json(404, {"error": "not found"})

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} {format % args}", flush=True)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--worker-pid", type=Path, default=DEFAULT_WORKER_PID)
    parser.add_argument("--pid-file", type=Path, default=DEFAULT_API_PID)
    parser.add_argument("--trade-intents", type=Path, default=DEFAULT_TRADE_INTENTS)
    args = parser.parse_args()

    args.pid_file.parent.mkdir(parents=True, exist_ok=True)
    args.pid_file.write_text(str(os.getpid()), encoding="ascii")
    server = ThreadingHTTPServer((args.host, args.port), handler_factory(args.predictions, args.worker_pid,args.trade_intents))
    print(f"Inference API listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
