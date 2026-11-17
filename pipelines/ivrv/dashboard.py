"""IVRV dashboard: the HTTP server (with PCR and 43Weeks routes) and the read-only standby view from saved logs."""
from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import logging
from pathlib import Path
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from voltr.core.clock import MARKET_TIMEZONE
from voltr.market.atm_features import REQUIRED_FEATURE_MINUTES
from . import paths
from . import extensions
from .config import _load_json

LOGGER = logging.getLogger("fixed_strike_live")  # one historic logger name for all IVRV modules


class LogDashboardView:
    """Read-only dashboard data source used while the scheduled live session is idle."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.prediction_path = (
            paths.BASE_DIR / str(config.get("prediction_log", "logs/live_predictions.jsonl"))
        ).resolve()
        self.latest_path = (
            paths.BASE_DIR / str(config.get("latest_prediction", "logs/latest_prediction.json"))
        ).resolve()
        self.paper_state_path = (
            paths.BASE_DIR / str(config.get("paper_position_state", "logs/paper_position_state.json"))
        ).resolve()
        self.events: deque[dict] = deque(maxlen=500)
        if self.prediction_path.is_file():
            try:
                with self.prediction_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            event = json.loads(line)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            continue
                        if event.get("event_type") == "fixed_strike_inference":
                            self.events.append(event)
            except OSError:
                LOGGER.warning("Could not read standby dashboard history from %s", self.prediction_path)

    def _paper_snapshot(self) -> dict[str, Any]:
        snapshot: dict[str, Any] = {}
        if self.paper_state_path.is_file():
            try:
                snapshot = _load_json(self.paper_state_path)
            except (OSError, ValueError, json.JSONDecodeError):
                snapshot = {}
        positions = snapshot.get("open_positions")
        if not isinstance(positions, list):
            positions = []
        maximum = int(self.config.get("paper_max_open_positions", 3))
        return {
            **snapshot,
            "mode": "PAPER",
            "status": "OPEN" if positions else "FLAT",
            "open_positions": positions,
            "open_position_count": len(positions),
            "maximum_open_positions": maximum,
            "available_position_slots": max(maximum - len(positions), 0),
            "replacement_min_confidence_improvement": float(self.config.get("paper_replacement_min_confidence_improvement", 0.10)),
            "replacement_cooldown_seconds": int(self.config.get("paper_replacement_cooldown_seconds", 600)),
            "maximum_replacements_per_day": int(self.config.get("paper_max_replacements_per_day", 3)),
            "entry_convergence_probability_threshold": float(
                self.config.get("paper_entry_convergence_probability_threshold", 0.75)
            ),
            "entry_probability_operator": ">",
            "entry_iv_rv_near_gap_decimal": float(
                self.config.get("paper_entry_iv_rv_near_gap_decimal", 0.005)
            ),
            "entry_rv_decline_event_score_threshold": float(
                self.config.get("paper_entry_rv_decline_event_score_threshold", 0.0)
            ),
            "entry_rv_decline_event_score_operator": ">=",
        }

    def status(self) -> dict[str, Any]:
        latest: dict[str, Any] = dict(self.events[-1]) if self.events else {
            "state": "WAITING",
            "reason": "no_saved_prediction_available",
        }
        if self.latest_path.is_file():
            try:
                latest = _load_json(self.latest_path)
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        latest["paper_trading"] = self._paper_snapshot()
        latest["runtime"] = {
            **latest.get("runtime", {}),
            "paper_only": True,
            "mode": "standby_dashboard",
            "required_feature_minutes": REQUIRED_FEATURE_MINUTES,
            "server_time": datetime.now(MARKET_TIMEZONE).isoformat(sep=" "),
        }
        return latest

    def recent(self, limit: int) -> list[dict]:
        return list(self.events)[-limit:]


class DashboardServer:
    def __init__(self, host: str, port: int, html_path: Path, engine: Any) -> None:
        owner = self
        try:
            weeks43_api = extensions.weeks43_module()
        except Exception:
            LOGGER.exception("Optional 43Weeks dashboard could not load")
            weeks43_api = None

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                try:
                    if extensions.PCR_WEB is not None and extensions.PCR_WEB.handle_get(self, parsed):
                        return
                    if parsed.path in {"/", "/index.html"}:
                        self._send(html_path.read_bytes(), "text/html; charset=utf-8")
                    elif parsed.path == "/firstTwoWeeks" or parsed.path.startswith("/firstTwoWeeks/"):
                        if weeks43_api is None:
                            self._json({"error": "43Weeks extension unavailable"}, 503)
                        else:
                            weeks43_api.handle_get(self, parsed)
                    elif parsed.path == "/api/status":
                        self._json(engine.status())
                    elif parsed.path == "/api/pipelines":
                        self._json(engine.pipeline_status() if hasattr(engine,'pipeline_status') else {'state':'LEGACY_OR_PREVIEW'})
                    elif parsed.path == "/pipelines":
                        self._send(paths.SERVICE_DASHBOARD_HTML.read_bytes(),'text/html; charset=utf-8')
                    elif parsed.path == "/api/predictions":
                        try:
                            limit = min(max(int(parse_qs(parsed.query).get("limit", ["100"])[0]), 1), 500)
                        except ValueError:
                            limit = 100
                        self._json(engine.recent(limit))
                    elif parsed.path == "/api/health":
                        self._json({"ok": True, "paper_only": True, "server_time": datetime.now().isoformat(sep=" ")})
                    else:
                        self.send_error(404)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    self._json({"error": str(error)}, 500)

            def do_POST(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path.startswith("/firstTwoWeeks/"):
                    if weeks43_api is None:
                        self._json({"error": "43Weeks extension unavailable"}, 503)
                    else:
                        try:
                            weeks43_api.handle_post(self, parsed)
                        except Exception as error:
                            self._json({"error": str(error)}, 500)
                    return
                if extensions.PCR_WEB is not None and extensions.PCR_WEB.handle_post(self, parsed):
                    return
                self.send_error(405)

            def _json(self, value: object, status: int = 200) -> None:
                self._send(json.dumps(value, allow_nan=False).encode("utf-8"), "application/json", status)

            def _send(self, body: bytes, content_type: str, status: int = 200) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format: str, *_args: object) -> None:
                return

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, name="fixed-strike-dashboard", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
