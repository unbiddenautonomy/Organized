"""Embedded read-only HTTP dashboard for the paper account."""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable


def _json_safe(value):
    """Recursively replace non-finite model outputs before API encoding."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class DashboardServer:
    def __init__(self, host: str, port: int, state_provider: Callable[[], dict]) -> None:
        self.host, self.port, self.state_provider = host, port, state_provider
        self.html = (Path(__file__).resolve().parent / "dashboard.html").read_bytes()
        self.retraining_html = (Path(__file__).resolve().parent / "retraining.html").read_bytes()
        provider, html, retraining_html = self.state_provider, self.html, self.retraining_html

        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner_self) -> None:
                path = inner_self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    body, kind = html, "text/html; charset=utf-8"
                elif path == "/retraining.html":
                    body, kind = retraining_html, "text/html; charset=utf-8"
                elif path == "/api/state":
                    body = json.dumps(_json_safe(provider()), separators=(",", ":"), allow_nan=False).encode()
                    kind = "application/json"
                else:
                    inner_self.send_error(404)
                    return
                inner_self.send_response(200)
                inner_self.send_header("Content-Type", kind)
                inner_self.send_header("Cache-Control", "no-store")
                inner_self.send_header("Content-Length", str(len(body)))
                inner_self.end_headers()
                inner_self.wfile.write(body)

            def log_message(inner_self, *_: object) -> None:
                return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="paper-dashboard", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
