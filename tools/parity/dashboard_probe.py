"""Serve the IVRV dashboard on a free local port from saved logs and record its routes.

Only GET requests are made. JSON bodies are reduced to their shape (keys and
value types) plus a hash of the stable values; HTML bodies are hashed. The
location-specific imports live in load_runtime().
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import urllib.request
from pathlib import Path

os.environ.setdefault("POSITION_ALERT_DISABLED", "1")
REPO = Path(__file__).resolve().parents[2]
ROUTES = ["/", "/pipelines", "/api/status", "/api/pipelines", "/api/predictions?limit=5", "/api/health",
          "/PCR", "/PCR/api/status", "/firstTwoWeeks", "/firstTwoWeeks/api/status",
          "/firstTwoWeeks/api/settings", "/firstTwoWeeks/api/models", "/firstTwoWeeks/api/backtests"]
VOLATILE = {"server_time", "csrf_token", "time", "now", "updated_at", "mtime", "age_seconds", "generated_at"}


def load_runtime():
    sys.path.insert(0, str(REPO))
    from pipelines.ivrv import live as li
    config = li.load_live_config(REPO / "config" / "live_config.json")
    return li, config


def shape(value):
    if isinstance(value, dict):
        return {k: shape(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def stable(value):
    if isinstance(value, dict):
        return {k: stable(v) for k, v in sorted(value.items()) if k not in VOLATILE}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    li, config = load_runtime()
    config = dict(config)
    for key, name in (("prediction_log", "live_predictions.jsonl"), ("latest_prediction", "latest_prediction.json"),
                      ("paper_position_state", "paper_position_state.json")):
        config[key] = str(args.logs / name)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = li.DashboardServer("127.0.0.1", port, li.DASHBOARD_HTML, li.LogDashboardView(config))
    server.start()
    result = {}
    try:
        for route in ROUTES:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}{route}", timeout=20) as response:
                    body, status, kind = response.read(), response.status, response.headers.get("Content-Type", "")
            except urllib.error.HTTPError as error:
                body, status, kind = error.read(), error.code, error.headers.get("Content-Type", "")
            entry = {"status": status, "type": kind.split(";")[0]}
            if "json" in kind:
                data = json.loads(body)
                entry["shape"] = shape(data)
                entry["values_sha256"] = hashlib.sha256(json.dumps(stable(data), sort_keys=True).encode()).hexdigest()
            else:
                entry["body_sha256"] = hashlib.sha256(body).hexdigest()
            result[route] = entry
    finally:
        server.close()
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
