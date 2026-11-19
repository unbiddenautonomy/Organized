"""Independent observation ledger for skipped, high-probability short straddles."""

from __future__ import annotations

import csv
from datetime import datetime, timedelta, time
from io import StringIO
import json
import math
import os
from pathlib import Path
import threading


CSV_FIELDS = (
    "position_id", "side", "signal_time", "entry_time", "expiry", "strike", "probability", "threshold",
    "paper_skip_reason", "quantity", "entry_premium", "last_premium",
    "pnl_inr", "peak_pnl_inr", "max_drawdown_inr", "target_profit_inr",
    "stop_loss_inr", "target_hit", "stop_hit", "horizon_time", "evaluation_time",
    "horizon_pnl_inr", "converged", "status",
)


class DemoPositionLedger:
    """Tracks marks and the two-hour outcome without touching the paper book."""

    def __init__(self, state_path: Path, lot_size: int) -> None:
        self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.lot_size = int(lot_size)
        self.lock = threading.RLock()
        self.positions: dict[str, dict] = {}
        if self.state_path.is_file():
            try:
                saved = json.loads(self.state_path.read_text(encoding="utf-8"))
                self.positions = {row["position_id"]: row for row in saved.get("positions", [])}
            except (OSError, ValueError, KeyError, TypeError):
                self.positions = {}

    def _save(self) -> None:
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps({"positions": list(self.positions.values())},
                                        allow_nan=False), encoding="utf-8")
        os.replace(temporary, self.state_path)

    def consider(self, *, signal_time: datetime, entry_time: datetime, expiry: str,
                 strike: float, probability: float, threshold: float, paper_entered: bool,
                 paper_skip_reason: str, call_token: int, put_token: int,
                 call_price: float, put_price: float, target: float, stop: float,
                 side: str = "SHORT_STRADDLE") -> bool:
        if (paper_entered or not math.isfinite(probability) or probability <= threshold
                or entry_time.time() >= time(15, 15)):
            return False
        if side not in {"SHORT_STRADDLE", "LONG_STRADDLE"}:
            raise ValueError("invalid demo straddle side")
        position_id = f"{side}|{signal_time.isoformat(sep=' ')}|{expiry}|{strike:g}"
        with self.lock:
            if position_id in self.positions:
                return False
            premium = float(call_price) + float(put_price)
            self.positions[position_id] = {
                "position_id": position_id, "side": side,
                "signal_time": signal_time.isoformat(sep=" "),
                "entry_time": entry_time.isoformat(sep=" "), "expiry": expiry,
                "strike": float(strike), "probability": float(probability),
                "threshold": float(threshold),
                "paper_skip_reason": paper_skip_reason, "quantity": self.lot_size,
                "call_token": int(call_token), "put_token": int(put_token),
                "entry_premium": premium, "last_premium": premium,
                "pnl_inr": 0.0, "peak_pnl_inr": 0.0, "max_drawdown_inr": 0.0,
                "target_profit_inr": float(target), "stop_loss_inr": float(stop),
                "target_hit": False, "stop_hit": False,
                "horizon_time": (entry_time + timedelta(hours=2)).isoformat(sep=" "),
                "evaluation_time": None, "horizon_pnl_inr": None,
                "converged": None, "status": "TRACKING",
            }
            self._save()
        return True

    def mark(self, now: datetime, quotes: dict[int, tuple[float, int]], exchange_second: int,
             updated_token: int | None = None, max_age_seconds: int = 60) -> None:
        changed = False
        with self.lock:
            for row in self.positions.values():
                if row["status"] != "TRACKING":
                    continue
                if updated_token is not None and updated_token not in (row["call_token"], row["put_token"]):
                    continue
                if row.get("last_mark_exchange_second") == exchange_second:
                    continue
                call = quotes.get(row["call_token"])
                put = quotes.get(row["put_token"])
                if call is None or put is None or any(
                    exchange_second - quote[1] > max_age_seconds or quote[1] > exchange_second
                    for quote in (call, put)
                ):
                    continue
                premium = float(call[0]) + float(put[0])
                direction = -1 if row.get("side") == "LONG_STRADDLE" else 1
                pnl = (row["entry_premium"] - premium) * self.lot_size * direction
                row["last_premium"] = premium
                row["last_mark_exchange_second"] = exchange_second
                row["pnl_inr"] = pnl
                row["peak_pnl_inr"] = max(row["peak_pnl_inr"], pnl)
                row["max_drawdown_inr"] = min(row["max_drawdown_inr"], pnl - row["peak_pnl_inr"])
                row["target_hit"] = row["target_hit"] or pnl >= row["target_profit_inr"]
                row["stop_hit"] = row["stop_hit"] or pnl <= row["stop_loss_inr"]
                if now >= datetime.fromisoformat(row["horizon_time"]):
                    row["evaluation_time"] = now.isoformat(sep=" ")
                    row["horizon_pnl_inr"] = pnl
                    row["converged"] = pnl > 0
                    row["status"] = "CONVERGED" if pnl > 0 else "NOT_CONVERGED"
                changed = True
            if changed:
                self._save()

    def snapshot(self) -> dict:
        with self.lock:
            rows = [dict(row) for row in self.positions.values()]
        return {"positions": rows, "count": len(rows),
                "tracking": sum(row["status"] == "TRACKING" for row in rows),
                "converged": sum(row["status"] == "CONVERGED" for row in rows)}

    def close_session(self, now: datetime) -> None:
        with self.lock:
            changed = False
            for row in self.positions.values():
                if row["status"] == "TRACKING" and row["entry_time"][:10] == now.date().isoformat():
                    row["status"] = "HORIZON_UNAVAILABLE"
                    changed = True
            if changed:
                self._save()

    def csv_bytes(self) -> bytes:
        output = StringIO()
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(self.snapshot()["positions"])
        return output.getvalue().encode("utf-8-sig")

    def save_session_csv(self, session_date: str) -> Path:
        path = self.state_path.with_name(f"{self.state_path.stem}_{session_date}.csv")
        path.write_bytes(self.csv_bytes())
        return path
