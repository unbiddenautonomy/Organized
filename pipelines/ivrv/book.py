"""IVRV paper book: fixed-strike straddle positions, tick-time marks, target/stop/horizon exits and replacements."""
from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import logging
import math
from pathlib import Path
from typing import Any, TextIO
from voltr.core.clock import NSE_EPOCH
from . import paths
from .config import _load_json

LOGGER = logging.getLogger("fixed_strike_live")  # one historic logger name for all IVRV modules


class PaperTradingEngine:
    """Overlapping one-lot, fixed-strike paper positions driven by live ticks."""

    def __init__(self, config: dict, lot_size: int, side: str = "SHORT_STRADDLE") -> None:
        if side not in {"SHORT_STRADDLE", "LONG_STRADDLE"}:
            raise ValueError(f"Unsupported paper side: {side}")
        self.side = side
        self.lot_size = int(lot_size)
        self.target = float(config.get("paper_profit_target_inr", 1500.0))
        self.stop = float(config.get("paper_stop_loss_inr", -750.0))
        self.horizon_minutes = int(config.get("fixed_strike_path_horizon_minutes", 120))
        self.max_open_positions = int(config.get("paper_max_open_positions", 3))
        if self.max_open_positions <= 0:
            raise ValueError("paper_max_open_positions must be positive")
        self.convergence_probability_threshold = float(
            config.get("paper_entry_convergence_probability_threshold", 0.75)
        )
        self.replacement_min_improvement = float(config.get("paper_replacement_min_confidence_improvement", 0.10))
        self.replacement_cooldown_seconds = int(config.get("paper_replacement_cooldown_seconds", 600))
        self.max_replacements_per_day = int(config.get("paper_max_replacements_per_day", 3))
        if not 0 < self.replacement_min_improvement <= 1 or self.replacement_cooldown_seconds < 0 or self.max_replacements_per_day < 0:
            raise ValueError("invalid paper replacement controls")
        self.iv_rv_near_gap_decimal = float(
            config.get("paper_entry_iv_rv_near_gap_decimal", 0.005)
        )
        self.rv_decline_event_score_threshold = float(
            config.get("paper_entry_rv_decline_event_score_threshold", 0.0)
        )
        self.square_off_time = datetime.strptime(
            str(config.get("paper_square_off_time", "15:14:30")), "%H:%M:%S"
        ).time()
        short_events = Path(str(config.get("paper_trade_events_log", "logs/paper_trade_events.jsonl")))
        short_state = Path(str(config.get("paper_position_state", "logs/paper_position_state.json")))
        events_name = (short_events if side == "SHORT_STRADDLE" else
                       short_events.with_name(f"{short_events.stem}_long{short_events.suffix}"))
        state_name = (short_state if side == "SHORT_STRADDLE" else
                      short_state.with_name(f"{short_state.stem}_long{short_state.suffix}"))
        events_path = paths.BASE_DIR / events_name
        state_path = paths.BASE_DIR / state_name
        events_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path = events_path.resolve()
        self.state_path = state_path.resolve()
        self.output: TextIO = self.events_path.open("a", encoding="utf-8", buffering=1)
        self.positions: dict[str, dict[str, Any]] = {}
        self.last_event: dict[str, Any] | None = None
        self.replacement_history_seconds: list[int] = []
        if self.state_path.is_file():
            try:
                saved = _load_json(self.state_path)
                open_positions = saved.get("open_positions")
                if isinstance(open_positions, list):
                    self.positions = {
                        str(position["position_id"]): position
                        for position in open_positions
                        if isinstance(position, dict) and position.get("position_id")
                    }
                elif saved.get("status") == "OPEN" and isinstance(saved.get("active_position"), dict):
                    position = saved["active_position"]
                    self.positions[str(position["position_id"])] = position
                if isinstance(saved.get("last_event"), dict):
                    self.last_event = saved["last_event"]
                self.replacement_history_seconds = [int(second) for second in saved.get("replacement_history_seconds", [])]
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.warning("Could not restore paper position state from %s", self.state_path)

    def _persist(self) -> None:
        state = {
            "mode": "paper",
            "side": self.side,
            "status": "OPEN" if self.positions else "FLAT",
            "overlapping_positions_allowed": self.max_open_positions > 1,
            "maximum_open_positions": self.max_open_positions,
            "entry_convergence_probability_threshold": self.convergence_probability_threshold,
            "entry_probability_operator": ">",
            "entry_iv_rv_near_gap_decimal": self.iv_rv_near_gap_decimal,
            "entry_rv_decline_event_score_threshold": self.rv_decline_event_score_threshold,
            "entry_rv_decline_event_score_operator": ">=",
            "open_position_count": len(self.positions),
            "open_positions": list(self.positions.values()),
            "last_event": self.last_event,
            "replacement_history_seconds": self.replacement_history_seconds,
            "replacement_min_confidence_improvement": self.replacement_min_improvement,
            "replacement_cooldown_seconds": self.replacement_cooldown_seconds,
            "maximum_replacements_per_day": self.max_replacements_per_day,
        }
        from voltr.core.io import atomic_write_json as atomic_json
        atomic_json(self.state_path,state)

    def _record(self, event: dict[str, Any]) -> None:
        self.last_event = event
        if event.get("event_type") == "paper_trade_exit" and event.get("exit_reason") == "higher_confidence_signal":
            self.replacement_history_seconds.append(int(event["exit_exchange_second"]))
        self.output.write(
            json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        )
        self._persist()

    def try_enter(
        self,
        *,
        exchange_second: int,
        signal_minute: int,
        expiry: date,
        strike: float,
        call_token: int,
        put_token: int,
        call_price: float,
        put_price: float,
        triggered_rules: list[str],
        convergence_probability: float | None = None,
        entry_route: str | None = None,
        entry_model_score: float | None = None,
    ) -> bool:
        stamp = NSE_EPOCH + timedelta(seconds=int(exchange_second))
        if stamp.time() >= self.square_off_time:
            return False
        if entry_route is None:
            if (
                convergence_probability is None
                or not math.isfinite(float(convergence_probability))
                or float(convergence_probability) <= self.convergence_probability_threshold
            ):
                return False
        elif entry_route not in {
            "iv_above_rv_convergence",
            "iv_near_rv_sharp_rv_decline",
            "iv_below_rv_sharp_rv_decline",
            "iv_above_rv_rising_rv_convergence",
        }:
            return False
        entry_premium = float(call_price) + float(put_price)
        signal_stamp = NSE_EPOCH + timedelta(minutes=signal_minute)
        position_id = f"{signal_stamp.isoformat(sep=' ')}|{expiry.isoformat()}|{float(strike):g}"
        if position_id in self.positions:
            return False
        if len(self.positions) >= self.max_open_positions:
            return False
        position = {
            "position_id": position_id,
            "status": "OPEN",
            "side": self.side,
            "lots": 1,
            "quantity": self.lot_size,
            "expiry": expiry.isoformat(),
            "fixed_strike": float(strike),
            "call_token": int(call_token),
            "put_token": int(put_token),
            "signal_datetime": signal_stamp.isoformat(sep=" "),
            "entry_datetime": stamp.isoformat(sep=" "),
            "entry_exchange_second": int(exchange_second),
            "entry_call_price": float(call_price),
            "entry_put_price": float(put_price),
            "entry_straddle_price": entry_premium,
            "last_call_price": float(call_price),
            "last_put_price": float(put_price),
            "last_straddle_price": entry_premium,
            "unrealized_pnl_inr": 0.0,
            "target_inr": self.target,
            "stop_loss_inr": self.stop,
            "maximum_holding_minutes": self.horizon_minutes,
            "square_off_time": self.square_off_time.isoformat(),
            "triggered_rules": list(triggered_rules),
            "entry_convergence_probability": convergence_probability,
            "entry_convergence_probability_threshold": self.convergence_probability_threshold,
            "entry_route": entry_route or "legacy_convergence_probability",
            "entry_model_score": entry_model_score,
        }
        self.positions[position_id] = position
        self._record({
            "event_type": "paper_trade_entry",
            "paper_only": True,
            **position,
        })
        from voltr.web.alerts import notify_position_action
        notify_position_action()
        return True

    def mark(
        self, position_id: str, exchange_second: int, call_price: float, put_price: float,
        *, force_reason: str | None = None,
        replacement_probability: float | None = None,
    ) -> dict[str, Any] | None:
        position = self.positions.get(position_id)
        if position is None:
            return None
        stamp = NSE_EPOCH + timedelta(seconds=int(exchange_second))
        if int(exchange_second) < int(position["entry_exchange_second"]):
            return None
        exit_premium = float(call_price) + float(put_price)
        direction = -1 if position.get("side", self.side) == "LONG_STRADDLE" else 1
        pnl = (float(position["entry_straddle_price"]) - exit_premium) * self.lot_size * direction
        elapsed_seconds = int(exchange_second) - int(position["entry_exchange_second"])
        position.update({
            "last_mark_datetime": stamp.isoformat(sep=" "),
            "last_call_price": float(call_price),
            "last_put_price": float(put_price),
            "last_straddle_price": exit_premium,
            "unrealized_pnl_inr": pnl,
            "holding_seconds": elapsed_seconds,
        })

        reason: str | None = force_reason
        if reason is None and stamp.time() >= self.square_off_time:
            reason = "session_square_off_151430"
        elif reason is None and pnl >= self.target:
            reason = "profit_target"
        elif reason is None and pnl <= self.stop:
            reason = "stop_loss"
        elif reason is None and elapsed_seconds >= self.horizon_minutes * 60:
            reason = "time_exit_120m"
        if reason is None:
            return None

        closed_position = dict(position)
        event = {
            "event_type": "paper_trade_exit",
            "paper_only": True,
            **closed_position,
            "status": "CLOSED",
            "exit_reason": reason,
            "exit_datetime": stamp.isoformat(sep=" "),
            "exit_exchange_second": int(exchange_second),
            "exit_call_price": float(call_price),
            "exit_put_price": float(put_price),
            "exit_straddle_price": exit_premium,
            "realized_pnl_inr": pnl,
            "holding_minutes": elapsed_seconds / 60.0,
        }
        if force_reason is not None:
            event["replacement_signal_confidence"] = replacement_probability
        self.positions.pop(position_id, None)
        self._record(event)
        from voltr.web.alerts import notify_position_action
        notify_position_action()
        return event

    def replacement_candidate(self, incoming_probability: float, incoming_route: str, exchange_second: int) -> tuple[dict[str, Any] | None, str]:
        """Select a weaker same-route holding, subject to daily churn limits."""
        if len(self.positions) < self.max_open_positions or not math.isfinite(incoming_probability):
            return None, "maximum_open_positions_reached"
        session_date = (NSE_EPOCH + timedelta(seconds=int(exchange_second))).date()
        today = [second for second in self.replacement_history_seconds
                 if (NSE_EPOCH + timedelta(seconds=second)).date() == session_date]
        if len(today) >= self.max_replacements_per_day:
            return None, "daily_replacement_limit_reached"
        if today and int(exchange_second) - max(today) < self.replacement_cooldown_seconds:
            return None, "replacement_cooldown_active"
        candidates = []
        for position in self.positions.values():
            if position.get("entry_route") != incoming_route:
                continue
            try:
                confidence = float(position.get("entry_convergence_probability"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(confidence) and incoming_probability - confidence >= self.replacement_min_improvement - 1e-12:
                candidates.append(position)
        if not candidates:
            return None, "no_same_route_position_with_sufficient_improvement"
        return min(candidates, key=lambda p: (float(p["entry_convergence_probability"]),
                                               int(p["entry_exchange_second"]))), "replacement_allowed"

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "PAPER",
            "side": self.side,
            "lots_per_position": 1,
            "overlapping_positions_allowed": self.max_open_positions > 1,
            "maximum_open_positions": self.max_open_positions,
            "available_position_slots": max(self.max_open_positions - len(self.positions), 0),
            "replacement_min_confidence_improvement": self.replacement_min_improvement,
            "replacement_cooldown_seconds": self.replacement_cooldown_seconds,
            "maximum_replacements_per_day": self.max_replacements_per_day,
            "entry_convergence_probability_threshold": self.convergence_probability_threshold,
            "entry_probability_operator": ">",
            "entry_iv_rv_near_gap_decimal": self.iv_rv_near_gap_decimal,
            "entry_rv_decline_event_score_threshold": self.rv_decline_event_score_threshold,
            "entry_rv_decline_event_score_operator": ">=",
            "status": "OPEN" if self.positions else "FLAT",
            "open_position_count": len(self.positions),
            "open_positions": [dict(position) for position in self.positions.values()],
            "last_event": dict(self.last_event) if self.last_event else None,
        }

    def close(self) -> None:
        self._persist()
        self.output.close()
