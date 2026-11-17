"""Template worker adapter. The service calls these methods; see docs/ARCHITECTURE.md ("Adapter contract")."""
from __future__ import annotations

from typing import Any

from voltr.core.clock import wall_nse_second


class TemplateAdapter:
    """Counts what it receives. Replace the bodies with the strategy's features, model and paper book."""

    def __init__(self, config: dict, contracts: dict[int, Any], lot_size: int, expiry, future_token: int,
                 source: dict[str, Any]) -> None:
        self.config = config
        self.contracts = contracts
        self.lot_size = lot_size
        self.expiry = expiry
        self.future_token = future_token
        self.decode_spot = source["decode_banknifty_7207"]
        self.counts = {"index": 0, "market": 0, "oi": 0}
        self.fault_reason: str | None = None
        self.last_pulse: int | None = None

    def index(self, message: bytes) -> None:
        """A 7207 index broadcast (spot)."""
        if self.decode_spot(message, float(self.config.get("price_divisor", 100))) is not None:
            self.counts["index"] += 1

    def market(self, update: Any) -> None:
        """One decoded 7208/17208 market-by-price update for a tracked token."""
        self.counts["market"] += 1

    def oi(self, message: bytes) -> None:
        """A raw open-interest broadcast; decode with source['decode_oi_message'] if needed."""
        self.counts["oi"] += 1

    def pulse(self) -> None:
        """Called about once a second, even without market data (time-based exits go here)."""
        self.last_pulse = wall_nse_second()

    def fault(self, reason: str | None) -> None:
        """Feed problems: block new entries while a reason is set; manage open positions as usual."""
        self.fault_reason = reason

    def status(self) -> dict[str, Any]:
        """Shown on the service dashboard; include 'fault' or 'error' to mark the worker DEGRADED."""
        return {"state": "RUNNING", "fault": self.fault_reason, "received": dict(self.counts)}

    def close(self) -> None:
        """Flush ledgers and release files at the end of the session."""
