"""Deterministic BANKNIFTY short-straddle paper execution and risk engine."""

from __future__ import annotations

import json
import math
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO

from voltr.instruments.master import OptionContract
from voltr.feed.messages.indices import decode_indices_message
from voltr.feed.messages.market_by_price import MarketUpdate7208
from voltr.feed.order_book import BankNiftyOrderBook, OrderBookEntry
from voltr.core.clock import NSE_EPOCH
from ..pnl_live_inference import LivePnlInferenceAdapter



REASON_DESCRIPTIONS = {
    "outside_entry_window": "Decision time is outside the configured paper-entry window.",
    "spot_iv_timestamp_mismatch": "Latest spot and IV predictions are too far apart in time.",
    "iv_warmup_incomplete": "IV features have not accumulated the required full warm-up history.",
    "daily_loss_limit": "Daily realized loss has reached the configured limit.",
    "daily_trade_limit": "The configured maximum number of entries for the session has been reached.",
    "entry_cooldown": "The minimum delay since the previous entry has not elapsed.",
    "vix_unavailable_or_warming": "India VIX or its five-minute change is not available yet.",
    "vix_above_limit": "India VIX is above the configured maximum.",
    "vix_rising": "India VIX rose more than the permitted five-minute amount.",
    "no_supportive_horizon_available": "No forecast horizon has acceptable predicted IV and enough time before forced exit.",
    "no_profitable_pnl_forecast": "The stacked P&L/MAE model found no horizon with sufficient reward and controlled downside.",
    "missing_executable_quotes": "A valid best bid or ask is missing for at least one straddle leg.",
    "wide_bid_ask_spread": "At least one option leg has a bid/ask spread above the configured limit.",
    "missing_or_stale_contract_oi": "Fresh positive open interest is unavailable for at least one option leg.",
    "spot_already_outside_selected_straddle_bep": "Spot is already outside the selected straddle break-even range.",
    "risk_or_margin_budget_allows_zero_lots": "Risk or margin sizing does not permit one complete BANKNIFTY lot.",
    "confirmation_direction_mismatch": "The remaining ten-minute direction at t+5 does not match the original fifteen-minute direction.",
    "confirmation_iv_not_supportive": "The estimated remaining ten-minute IV change is above the configured short-volatility limit.",
    "confirmation_no_reserved_lots": "Initial sizing did not leave a second tranche available.",
    "confirmation_position_already_reduced": "The position reached an exit stage before confirmation.",
    "confirmation_missing_quotes": "Executable quotes for the original straddle are unavailable at confirmation.",
    "confirmation_wide_spread": "The original straddle has an excessive bid/ask spread at confirmation.",
    "confirmation_risk_or_margin_failed": "Adding the second tranche would exceed the trade-risk or margin budget.",
}


def combined_short_mtm(entry_credit: float, current_debit: float, lot_size: int, lots: int) -> float:
    """Mark-to-market P&L for closing both short legs at their current asks."""
    return (entry_credit - current_debit) * lot_size * lots


def bep_stress_margin_per_lot(
    spot: float, strike: float, entry_credit: float, lot_size: int,
    stress_pct: float, configured_margin: float, buffer: float,
) -> tuple[float, float, float, float]:
    """Return lower/upper BEP, stressed expiry loss and conservative paper margin.

    This is a paper risk reserve, not an NSE SPAN or broker margin quote.
    """
    lower_bep, upper_bep = strike - entry_credit, strike + entry_credit
    stress_down, stress_up = spot * (1 - stress_pct), spot * (1 + stress_pct)
    stress_loss = max(lower_bep - stress_down, stress_up - upper_bep, 0.0) * lot_size
    return lower_bep, upper_bep, stress_loss, max(configured_margin, stress_loss * buffer)


@dataclass(slots=True)
class PaperPosition:
    trade_id: int
    entry_time: str
    entry_minute: int
    horizon: int
    direction: str
    strike: float
    call_token: int
    put_token: int
    original_lots: int
    open_lots: int
    lot_size: int
    call_entry: float
    put_entry: float
    entry_credit: float
    lower_bep: float
    upper_bep: float
    margin_reserved: float
    stop_debit: float
    planned_lots: int
    confirmation_due_minute: int | None
    initial_predicted_spot: float
    planned_exit_minute: int
    maximum_exit_minute: int
    extensions: int = 0
    confirmation_checked: bool = False
    target_stage: int = 0
    realized_pnl: float = 0.0
    last_call_ask: float = 0.0
    last_put_ask: float = 0.0
    unrealized_pnl: float = 0.0


class PaperTradingEngine:
    def __init__(
        self,
        config: dict,
        contracts: dict[int, OptionContract],
        order_book: BankNiftyOrderBook,
        oi_state: dict[int, tuple[int, int, int]],
        pnl_adapter: LivePnlInferenceAdapter | None = None,
    ) -> None:
        self.config = config
        self.enabled = bool(config.get("paper_trading_enabled", True))
        self.contracts = contracts
        self.order_book = order_book
        self.oi_state = oi_state
        self.pnl_adapter = pnl_adapter
        self.latest_pnl_forecasts: dict[str, dict] = {}
        self.latest_stacked_minute: int | None = None
        self.horizons = tuple(int(value) for value in config.get("paper_prediction_horizons", [5, 15, 30, 45, 60]))
        self.starting_capital = float(config.get("paper_starting_capital", 1_000_000))
        self.cash = self.starting_capital
        self.lot_size = int(config.get("paper_lot_size", 30))
        self.realized_pnl = 0.0
        self.daily_realized_pnl = 0.0
        self.trade_id = 0
        self.position: PaperPosition | None = None
        self.closed_trades: list[dict] = []
        self.rejections: list[dict] = []
        self.spot_prediction_audit: list[dict] = []
        self.iv_prediction_audit: list[dict] = []
        self._spot_forecast_keys: set[tuple[int, int]] = set()
        self._iv_forecast_keys: set[tuple[int, int]] = set()
        self.latest_spot_prediction: dict | None = None
        self.latest_iv_prediction: dict | None = None
        self.latest_decision: dict | None = None
        self.current_spot: float | None = None
        self.current_vix: float | None = None
        self.vix_history: list[tuple[int, float]] = []
        self.last_entry_minute: int | None = None
        self.daily_trade_count = 0
        self.latest_index_minute: int | None = None
        self.lock = threading.RLock()

        self.by_strike: dict[float, dict[str, int]] = {}
        for item in contracts.values():
            self.by_strike.setdefault(float(item.strike_price), {})[item.option_type] = item.token
        self.by_strike = {strike: pair for strike, pair in self.by_strike.items() if {"CE", "PE"} <= pair.keys()}

        path = Path(config.get("paper_trade_file", "logs/paper_trades.jsonl"))
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = path
        self.output: TextIO = path.open("a", encoding="utf-8", buffering=1)
        decision_path = Path(config.get("paper_decision_file", "logs/paper_decisions.jsonl"))
        if not decision_path.is_absolute():
            decision_path = Path(__file__).resolve().parents[1] / decision_path
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        self.decision_output_path = decision_path
        self.decision_output: TextIO = decision_path.open("a", encoding="utf-8", buffering=1)

    @staticmethod
    def _event_minute(event: dict) -> int:
        stamp = datetime.fromisoformat(event["exchange_minute"])
        return int((stamp - NSE_EPOCH).total_seconds() // 60)

    @staticmethod
    def _best_bid(entry: OrderBookEntry | None) -> float | None:
        if entry is None:
            return None
        return next((level.price for level in entry.bids if level.price > 0 and level.quantity > 0), None)

    @staticmethod
    def _best_ask(entry: OrderBookEntry | None) -> float | None:
        if entry is None:
            return None
        return next((level.price for level in entry.asks if level.price > 0 and level.quantity > 0), None)

    def on_spot_prediction(self, event: dict) -> None:
        with self.lock:
            minute = self._event_minute(event)
            current = float(event["current_spot"])
            self._settle_forecasts(self.spot_prediction_audit, minute, current)
            self._record_forecasts(
                self.spot_prediction_audit, self._spot_forecast_keys, minute,
                current, event.get("predictions", {}), "predicted_spot",
            )
            self.latest_spot_prediction = event
            self.current_spot = current
            self._update_stacked_forecasts()
            if self.position is None:
                self._consider_entry()
            else:
                self._consider_confirmation()

    def on_iv_prediction(self, event: dict) -> None:
        with self.lock:
            minute = self._event_minute(event)
            current = float(event["current_iv"])
            self._settle_forecasts(self.iv_prediction_audit, minute, current)
            self._record_forecasts(
                self.iv_prediction_audit, self._iv_forecast_keys, minute,
                current, event.get("predictions", {}), "predicted_iv",
            )
            self.latest_iv_prediction = event
            self._update_stacked_forecasts()
            if self.position is None:
                self._consider_entry()
            else:
                self._consider_confirmation()

    @staticmethod
    def _settle_forecasts(rows: list[dict], actual_minute: int, actual_value: float) -> None:
        for row in rows:
            if row["status"] == "pending" and actual_minute >= row["target_minute"]:
                row["actual"] = actual_value
                row["actual_minute"] = actual_minute
                row["error"] = row["predicted"] - actual_value
                row["absolute_error"] = abs(row["error"])
                row["status"] = "matured"

    def _update_stacked_forecasts(self) -> None:
        """Refresh HMM/P&L for every new synchronized model pair."""
        if self.pnl_adapter is None or not self.latest_spot_prediction or not self.latest_iv_prediction:
            return
        spot_minute = self._event_minute(self.latest_spot_prediction)
        iv_minute = self._event_minute(self.latest_iv_prediction)
        if spot_minute != iv_minute or self.latest_stacked_minute == spot_minute:
            return
        self.latest_pnl_forecasts = self.pnl_adapter.score(
            self.latest_spot_prediction, self.latest_iv_prediction
        )
        self.latest_stacked_minute = spot_minute

    @staticmethod
    def _record_forecasts(
        rows: list[dict], keys: set[tuple[int, int]], source_minute: int,
        current: float, predictions: dict, prediction_field: str,
    ) -> None:
        for horizon_text, forecast in predictions.items():
            horizon = int(horizon_text)
            key = (source_minute, horizon)
            predicted = forecast.get(prediction_field)
            if key in keys or predicted is None or not math.isfinite(float(predicted)):
                continue
            keys.add(key)
            rows.append({
                "prediction_time": (NSE_EPOCH + timedelta(minutes=source_minute)).isoformat(sep=" "),
                "source_minute": source_minute, "horizon": horizon,
                "target_time": (NSE_EPOCH + timedelta(minutes=source_minute + horizon)).isoformat(sep=" "),
                "target_minute": source_minute + horizon, "current": current,
                "predicted": float(predicted), "actual": None, "actual_minute": None,
                "error": None, "absolute_error": None, "status": "pending",
            })
        if len(rows) > 2500:
            removed = rows[:-2000]
            del rows[:-2000]
            for row in removed:
                keys.discard((row["source_minute"], row["horizon"]))

    def on_index_message(self, message: bytes) -> None:
        log_time, values = decode_indices_message(message, float(self.config.get("price_divisor", 100.0)))
        vix = next((value for name, value in values.items() if "VIX" in name), None)
        bank_spot = next((value for name, value in values.items() if name in {"BANK NIFTY", "BANKNIFTY", "NIFTY BANK"}), None)
        if vix is None and bank_spot is None:
            return
        minute = log_time // 60
        with self.lock:
            tolerance = int(self.config.get("replay_rewind_tolerance_minutes", 5))
            if self.latest_index_minute is not None and minute < self.latest_index_minute - tolerance:
                self._reset_for_replay(self.latest_index_minute, minute)
            self.latest_index_minute = minute if self.latest_index_minute is None else max(self.latest_index_minute, minute)
            if bank_spot is not None:
                self.current_spot = bank_spot
                self._settle_forecasts(self.spot_prediction_audit, minute, bank_spot)
            if vix is None:
                return
            self.current_vix = vix
            if not self.vix_history or self.vix_history[-1][0] != minute:
                self.vix_history.append((minute, vix))
            else:
                self.vix_history[-1] = (minute, vix)
            self.vix_history = self.vix_history[-120:]

    def _reset_for_replay(self, old_minute: int, new_minute: int) -> None:
        event = {
            "event_type": "replay_reset", "decision": "RESET",
            "old_exchange_minute": (NSE_EPOCH + timedelta(minutes=old_minute)).isoformat(sep=" "),
            "new_exchange_minute": (NSE_EPOCH + timedelta(minutes=new_minute)).isoformat(sep=" "),
            "reason": "exchange_clock_rewound",
        }
        self._write_decision(event)
        self.cash = self.starting_capital
        self.realized_pnl = 0.0
        self.daily_realized_pnl = 0.0
        self.trade_id = 0
        self.position = None
        self.closed_trades.clear()
        self.rejections.clear()
        self.spot_prediction_audit.clear()
        self.iv_prediction_audit.clear()
        self._spot_forecast_keys.clear()
        self._iv_forecast_keys.clear()
        self.latest_spot_prediction = None
        self.latest_iv_prediction = None
        self.latest_pnl_forecasts = {}
        self.latest_stacked_minute = None
        self.latest_decision = None
        self.current_spot = None
        self.current_vix = None
        self.vix_history.clear()
        self.last_entry_minute = None
        self.daily_trade_count = 0
        self.order_book.reset()

    def on_market_update(self, update: MarketUpdate7208) -> None:
        with self.lock:
            if self.position is None or update.token not in (self.position.call_token, self.position.put_token):
                return
            self._mark_and_manage(update.log_time // 60)

    def _vix_change(self, minutes: int = 5) -> float | None:
        if self.current_vix is None or len(self.vix_history) < 2:
            return None
        target = self.vix_history[-1][0] - minutes
        old = next((value for minute, value in reversed(self.vix_history) if minute <= target), None)
        return None if old is None else self.current_vix - old

    def _oi_fresh(self, token: int, minute: int) -> bool:
        state = self.oi_state.get(token)
        if state is None or state[0] <= 0:
            return False
        return state[1] <= 0 or 0 <= minute - state[1] // 60 <= int(self.config.get("paper_oi_max_age_minutes", 15))

    def _reject(self, minute: int, reasons: list[str], **details: object) -> None:
        event = {
            "event_type": "entry_rejected", "decision": "REJECT",
            "exchange_minute": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
            "reasons": reasons,
            "reason_descriptions": [REASON_DESCRIPTIONS.get(reason, reason) for reason in reasons],
            **details,
        }
        if not self.rejections or self.rejections[-1] != event:
            self.rejections.append(event)
            self.rejections = self.rejections[-30:]
            self._write(event)
            self._write_decision(event)

    def _select_horizon(self, spot_event: dict, iv_event: dict, minute_of_day: int) -> tuple[int, dict, dict, dict] | None:
        """Rank all tradable horizons by leakage-free stacked P&L forecasts."""
        max_iv_change = float(self.config.get("paper_max_predicted_iv_change", 0.00025))
        force_exit = int(self.config.get("paper_force_exit_minute", 915))
        if self.pnl_adapter is None:
            return None
        candidates: list[tuple[float, int, dict, dict, dict]] = []
        for horizon in self.horizons:
            spot_forecast = spot_event.get("predictions", {}).get(str(horizon))
            iv_forecast = iv_event.get("predictions", {}).get(str(horizon))
            pnl = self.latest_pnl_forecasts.get(str(horizon))
            if not spot_forecast or not iv_forecast or not pnl or minute_of_day + horizon > force_exit:
                continue
            iv_change = float(iv_forecast["predicted_iv_change"])
            if not math.isfinite(iv_change) or iv_change > max_iv_change:
                continue
            if (pnl["expected_pnl"] < float(self.config.get("paper_min_expected_pnl", 0.0)) or
                    pnl["profit_probability"] < float(self.config.get("paper_min_profit_probability", 0.50)) or
                    pnl["pnl_q10"] < float(self.config.get("paper_min_pnl_q10", -500.0)) or
                    pnl["mae_q90"] > float(self.config.get("paper_max_entry_mae_q90", 3000.0)) or
                    not pnl["hmm_in_distribution"]):
                continue
            score = pnl["expected_pnl"] * pnl["profit_probability"] / max(abs(min(pnl["pnl_q10"], 0.0)), 100.0)
            pnl["selection_score"] = float(score)
            candidates.append((score, horizon, spot_forecast, iv_forecast, pnl))
        if not candidates:
            return None
        _, horizon, spot_forecast, iv_forecast, pnl = max(candidates, key=lambda item: (item[0], -item[1]))
        return horizon, spot_forecast, iv_forecast, pnl

    def _consider_entry(self) -> None:
        if not self.enabled or self.position is not None or not self.latest_spot_prediction or not self.latest_iv_prediction:
            return
        spot_event, iv_event = self.latest_spot_prediction, self.latest_iv_prediction
        spot_minute, iv_minute = self._event_minute(spot_event), self._event_minute(iv_event)
        # Spot is normally emitted first and IV follows milliseconds later. Do
        # not evaluate the new spot event against the previous minute's IV.
        # The second callback will re-enter here with an exact synchronized pair.
        if spot_minute != iv_minute:
            return
        minute = max(spot_minute, iv_minute)
        reasons: list[str] = []
        timestamp = NSE_EPOCH + timedelta(minutes=minute)
        minute_of_day = timestamp.hour * 60 + timestamp.minute
        entry_start = int(self.config.get("paper_entry_start_minute", 565))
        entry_end = int(self.config.get("paper_entry_end_minute", 900))
        if not entry_start <= minute_of_day <= entry_end:
            reasons.append("outside_entry_window")
        if not bool(iv_event.get("warmup_complete", False)):
            reasons.append("iv_warmup_incomplete")
        if self.daily_realized_pnl <= -self.starting_capital * float(self.config.get("paper_daily_loss_pct", 0.01)):
            reasons.append("daily_loss_limit")
        if self.daily_trade_count >= int(self.config.get("paper_max_trades_per_day", 6)):
            reasons.append("daily_trade_limit")
        if self.last_entry_minute is not None and minute - self.last_entry_minute < int(self.config.get("paper_cooldown_minutes", 5)):
            reasons.append("entry_cooldown")
        vix_change = self._vix_change(5)
        if bool(self.config.get("paper_require_vix", True)) and (self.current_vix is None or vix_change is None):
            reasons.append("vix_unavailable_or_warming")
        if self.current_vix is not None and self.current_vix > float(self.config.get("paper_vix_max", 35.0)):
            reasons.append("vix_above_limit")
        if vix_change is not None and vix_change > float(self.config.get("paper_vix_max_5m_rise", 1.0)):
            reasons.append("vix_rising")
        if reasons:
            self._reject(minute, reasons)
            return

        selected = self._select_horizon(spot_event, iv_event, minute_of_day)
        if selected is None:
            self._reject(
                minute, ["no_profitable_pnl_forecast"],
                configured_horizons=list(self.horizons),
                predicted_iv_changes={key: value.get("predicted_iv_change") for key, value in iv_event.get("predictions", {}).items()},
                maximum_allowed_iv_change=float(self.config.get("paper_max_predicted_iv_change", 0.00025)),
            )
            return
        horizon, spot_forecast, iv_forecast, pnl_forecast = selected

        current_spot = float(spot_event["current_spot"])
        predicted_spot = float(spot_forecast["predicted_spot"])
        predicted_iv_change = float(iv_forecast["predicted_iv_change"])
        move = predicted_spot - current_spot
        threshold = float(self.config.get("paper_neutral_move_points", 50.0))
        direction = "NEUTRAL" if abs(move) < threshold else ("UP" if move > 0 else "DOWN")
        target = current_spot if direction == "NEUTRAL" else predicted_spot
        strike = min(self.by_strike, key=lambda value: abs(value - target))
        self.latest_decision = {
            "exchange_minute": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
            "horizon": horizon, "direction": direction, "current_spot": current_spot,
            "predicted_spot": predicted_spot, "current_iv": float(iv_event["current_iv"]),
            "predicted_iv_change": predicted_iv_change, "strike": strike,
            "pnl_forecast": pnl_forecast,
        }
        pair = self.by_strike[strike]
        call_entry, put_entry = self.order_book.get(pair["CE"]), self.order_book.get(pair["PE"])
        call_bid, put_bid = self._best_bid(call_entry), self._best_bid(put_entry)
        call_ask, put_ask = self._best_ask(call_entry), self._best_ask(put_entry)
        if None in (call_bid, put_bid, call_ask, put_ask):
            self._reject(minute, ["missing_executable_quotes"], horizon=horizon, strike=strike,
                         call_token=pair["CE"], put_token=pair["PE"], call_bid=call_bid,
                         call_ask=call_ask, put_bid=put_bid, put_ask=put_ask)
            return
        max_spread = float(self.config.get("paper_max_leg_spread_pct", 0.05))
        if (call_ask - call_bid) / call_ask > max_spread or (put_ask - put_bid) / put_ask > max_spread:
            self._reject(minute, ["wide_bid_ask_spread"], horizon=horizon, strike=strike,
                         call_spread_pct=(call_ask - call_bid) / call_ask,
                         put_spread_pct=(put_ask - put_bid) / put_ask, maximum_spread_pct=max_spread)
            return
        oi_fresh = self._oi_fresh(pair["CE"], minute) and self._oi_fresh(pair["PE"], minute)
        if bool(self.config.get("paper_require_fresh_oi", False)) and not oi_fresh:
            self._reject(minute, ["missing_or_stale_contract_oi"], horizon=horizon, strike=strike,
                         call_token=pair["CE"], put_token=pair["PE"],
                         call_oi_state=self.oi_state.get(pair["CE"]), put_oi_state=self.oi_state.get(pair["PE"]))
            return

        credit = float(call_bid + put_bid)
        lower_bep, upper_bep, bep_stress_per_lot, margin_per_lot = bep_stress_margin_per_lot(
            current_spot, strike, credit, self.lot_size,
            float(self.config.get("paper_bep_stress_pct", 0.02)),
            float(self.config.get("paper_margin_per_lot", 150_000.0)),
            float(self.config.get("paper_bep_margin_buffer", 1.25)),
        )
        if not lower_bep < current_spot < upper_bep:
            self._reject(minute, ["spot_already_outside_selected_straddle_bep"])
            return
        stop_loss_per_lot = float(self.config.get("paper_stop_loss_per_lot", 1500.0))
        stop_pct = stop_loss_per_lot / max(credit * self.lot_size, 1.0)
        risk_budget = float(self.config.get("paper_risk_budget_rupees", 6000.0))
        margin_budget = self.starting_capital * float(self.config.get("paper_max_margin_fraction", 0.30))
        lots_by_risk = int(risk_budget // max(stop_loss_per_lot, 1.0))
        lots_by_margin = int(margin_budget // max(margin_per_lot, 1.0))
        vix_multiplier = float(self.config.get("paper_normal_vix_size_multiplier", 1.0))
        if self.current_vix is not None and self.current_vix >= float(self.config.get("paper_elevated_vix", 22.0)):
            vix_multiplier = float(self.config.get("paper_elevated_vix_size_multiplier", 0.5))
        high_confidence = (pnl_forecast["selection_score"] >= float(self.config.get("paper_four_lot_score", 1.9229049019)) and
                           pnl_forecast["pnl_q10"] >= -stop_loss_per_lot and
                           pnl_forecast["mae_q90"] <= stop_loss_per_lot)
        model_lots = int(self.config.get("paper_max_lots", 4)) if high_confidence else 1
        lots = min(model_lots, lots_by_risk, lots_by_margin)
        lots = int(math.floor(lots * vix_multiplier))
        if lots < 1:
            self._reject(
                minute, ["risk_or_margin_budget_allows_zero_lots"], horizon=horizon,
                entry_credit=credit, stop_pct=stop_pct, stop_loss_per_lot=stop_loss_per_lot,
                risk_budget=risk_budget, margin_per_lot=margin_per_lot,
                margin_budget=margin_budget, lots_by_risk=lots_by_risk,
                lots_by_margin=lots_by_margin,
            )
            return

        planned_lots = lots
        scale_enabled = bool(self.config.get("paper_15m_two_stage_entry", True)) and horizon == 15
        initial_lots = max(1, math.ceil(planned_lots / 2)) if scale_enabled else planned_lots
        self.trade_id += 1
        position = PaperPosition(
            trade_id=self.trade_id, entry_time=(NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
            entry_minute=minute, horizon=horizon, direction=direction, strike=strike,
            call_token=pair["CE"], put_token=pair["PE"], original_lots=initial_lots, open_lots=initial_lots,
            lot_size=self.lot_size, call_entry=call_bid, put_entry=put_bid, entry_credit=credit,
            lower_bep=lower_bep, upper_bep=upper_bep, margin_reserved=margin_per_lot * initial_lots,
            stop_debit=credit * (1 + stop_pct), last_call_ask=call_ask, last_put_ask=put_ask,
            planned_lots=planned_lots,
            confirmation_due_minute=minute + 5 if scale_enabled and planned_lots > initial_lots else None,
            initial_predicted_spot=predicted_spot,
            planned_exit_minute=minute + horizon,
            maximum_exit_minute=minute + int(self.config.get("paper_max_hold_minutes", 120)),
        )
        self.position = position
        self.last_entry_minute = minute
        self.daily_trade_count += 1
        self._write_decision({
            "event_type": "entry_decision", "decision": "ACCEPT",
            "exchange_minute": position.entry_time,
            "reasons": ["all_mandatory_guardrails_passed"],
            "reason_descriptions": ["All mandatory paper-entry checks passed and sizing allowed at least one lot."],
            "horizon": horizon, "direction": direction, "strike": strike,
            "current_spot": current_spot, "predicted_spot": predicted_spot,
            "current_iv": float(iv_event["current_iv"]), "predicted_iv_change": predicted_iv_change,
            "india_vix": self.current_vix, "vix_change_5m": vix_change,
            "entry_credit": credit, "stop_pct": stop_pct, "stop_loss_per_lot": stop_loss_per_lot,
            "risk_budget": risk_budget, "margin_per_lot": margin_per_lot,
            "margin_budget": margin_budget, "lots_by_risk": lots_by_risk,
            "lots_by_margin": lots_by_margin, "initial_lots": initial_lots,
            "reserved_confirmation_lots": planned_lots - initial_lots, "planned_lots": planned_lots,
            "confirmation_due_minute": position.confirmation_due_minute, "final_lots": initial_lots, "oi_fresh": oi_fresh,
            "pnl_forecast": pnl_forecast, "high_confidence_four_lot": high_confidence,
            "passed_guardrails": ["entry_window", "signal_sync", "iv_warmup", "daily_loss",
                                  "cooldown", "vix", "iv_support", "quotes", "spread",
                                  "break_even", "risk_budget", "margin_budget"],
        })
        self._write({"event_type": "paper_entry", **asdict(position), "current_spot": current_spot,
                     "predicted_spot": predicted_spot, "current_iv": iv_event["current_iv"],
                     "predicted_iv_change": predicted_iv_change, "india_vix": self.current_vix,
                     "bep_stress_loss_per_lot": bep_stress_per_lot, "oi_fresh": oi_fresh,
                     "pnl_forecast": pnl_forecast,
                     "horizon_selection": "best eligible stacked expected-PnL/downside score across all horizons"})

    @staticmethod
    def _direction(current: float, predicted: float, threshold: float) -> str:
        move = predicted - current
        return "NEUTRAL" if abs(move) < threshold else ("UP" if move > 0 else "DOWN")

    @staticmethod
    def _interpolated_ten_minute(event: dict, field: str) -> float | None:
        """Estimate t+10 midway between the current 5m and 15m forecasts."""
        five = event.get("predictions", {}).get("5", {}).get(field)
        fifteen = event.get("predictions", {}).get("15", {}).get(field)
        if five is None or fifteen is None:
            return None
        return (float(five) + float(fifteen)) / 2.0

    def _confirmation_decision(self, minute: int, decision: str, reasons: list[str], **details: object) -> None:
        event = {
            "event_type": "confirmation_entry_decision", "decision": decision,
            "exchange_minute": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
            "reasons": reasons,
            "reason_descriptions": [REASON_DESCRIPTIONS.get(reason, reason) for reason in reasons],
            **details,
        }
        self._write_decision(event)

    def _consider_confirmation(self) -> None:
        position = self.position
        if position is None or position.confirmation_due_minute is None or position.confirmation_checked:
            return
        if not self.latest_spot_prediction or not self.latest_iv_prediction:
            return
        spot_minute = self._event_minute(self.latest_spot_prediction)
        iv_minute = self._event_minute(self.latest_iv_prediction)
        if spot_minute != iv_minute:
            return
        minute = min(spot_minute, iv_minute)
        if minute < position.confirmation_due_minute:
            return
        position.confirmation_checked = True
        common = {
            "trade_id": position.trade_id, "original_t": position.entry_time,
            "original_t_plus_15": (NSE_EPOCH + timedelta(minutes=position.entry_minute + 15)).isoformat(sep=" "),
            "confirmation_t_plus_5": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
            "original_direction": position.direction, "original_predicted_spot_t_plus_15": position.initial_predicted_spot,
        }
        remaining = position.planned_lots - position.original_lots
        if remaining <= 0:
            self._confirmation_decision(minute, "REJECT", ["confirmation_no_reserved_lots"], **common)
            return
        if position.target_stage > 0:
            self._confirmation_decision(minute, "REJECT", ["confirmation_position_already_reduced"], **common)
            return
        predicted_spot_10 = self._interpolated_ten_minute(self.latest_spot_prediction, "predicted_spot")
        predicted_iv_10 = self._interpolated_ten_minute(self.latest_iv_prediction, "predicted_iv")
        if predicted_spot_10 is None or predicted_iv_10 is None:
            self._confirmation_decision(minute, "REJECT", ["no_supportive_horizon_available"], **common)
            return
        current_spot = float(self.latest_spot_prediction["current_spot"])
        current_iv = float(self.latest_iv_prediction["current_iv"])
        threshold = float(self.config.get("paper_neutral_move_points", 50.0))
        confirmation_direction = self._direction(current_spot, predicted_spot_10, threshold)
        iv_change_10 = predicted_iv_10 - current_iv
        common.update({
            "estimated_predicted_spot_t_plus_10": predicted_spot_10,
            "estimated_predicted_iv_t_plus_10": predicted_iv_10,
            "estimated_iv_change_t_plus_10": iv_change_10,
            "confirmation_direction": confirmation_direction,
            "t_plus_10_method": "midpoint interpolation of 5m and 15m forecasts at t+5",
        })
        reasons: list[str] = []
        if confirmation_direction != position.direction:
            reasons.append("confirmation_direction_mismatch")
        if iv_change_10 > float(self.config.get("paper_max_predicted_iv_change", 0.00025)):
            reasons.append("confirmation_iv_not_supportive")
        if reasons:
            self._confirmation_decision(minute, "REJECT", reasons, **common)
            return
        call_book, put_book = self.order_book.get(position.call_token), self.order_book.get(position.put_token)
        call_bid, put_bid = self._best_bid(call_book), self._best_bid(put_book)
        call_ask, put_ask = self._best_ask(call_book), self._best_ask(put_book)
        if None in (call_bid, put_bid, call_ask, put_ask):
            self._confirmation_decision(minute, "REJECT", ["confirmation_missing_quotes"], **common)
            return
        max_spread = float(self.config.get("paper_max_leg_spread_pct", 0.05))
        if (call_ask - call_bid) / call_ask > max_spread or (put_ask - put_bid) / put_ask > max_spread:
            self._confirmation_decision(minute, "REJECT", ["confirmation_wide_spread"], **common)
            return
        new_total = position.original_lots + remaining
        new_call = (position.call_entry * position.original_lots + call_bid * remaining) / new_total
        new_put = (position.put_entry * position.original_lots + put_bid * remaining) / new_total
        new_credit = new_call + new_put
        stop_pct = float(self.config.get("paper_stop_pct_by_horizon", {}).get("15", 0.12))
        total_stop_risk = new_credit * stop_pct * self.lot_size * new_total
        risk_budget = self.starting_capital * float(self.config.get("paper_risk_per_trade_pct", 0.02))
        _, _, _, margin_per_lot = bep_stress_margin_per_lot(
            current_spot, position.strike, new_credit, self.lot_size,
            float(self.config.get("paper_bep_stress_pct", 0.02)),
            float(self.config.get("paper_margin_per_lot", 150_000.0)),
            float(self.config.get("paper_bep_margin_buffer", 1.25)),
        )
        margin_budget = self.starting_capital * float(self.config.get("paper_max_margin_fraction", 0.30))
        if total_stop_risk > risk_budget or margin_per_lot * new_total > margin_budget:
            self._confirmation_decision(minute, "REJECT", ["confirmation_risk_or_margin_failed"],
                                        total_stop_risk=total_stop_risk, risk_budget=risk_budget,
                                        total_margin=margin_per_lot * new_total, margin_budget=margin_budget, **common)
            return
        position.call_entry, position.put_entry, position.entry_credit = new_call, new_put, new_credit
        position.original_lots, position.open_lots = new_total, position.open_lots + remaining
        position.lower_bep, position.upper_bep = position.strike - new_credit, position.strike + new_credit
        position.stop_debit = new_credit * (1 + stop_pct)
        position.margin_reserved = margin_per_lot * new_total
        position.last_call_ask, position.last_put_ask = call_ask, put_ask
        accepted = {**common, "added_lots": remaining, "total_lots": new_total,
                    "second_entry_call_bid": call_bid, "second_entry_put_bid": put_bid,
                    "weighted_entry_credit": new_credit, "total_stop_risk": total_stop_risk,
                    "total_margin": position.margin_reserved}
        self._confirmation_decision(minute, "ACCEPT", ["confirmation_direction_matched"], **accepted)
        self._write({"event_type": "paper_scale_in", "exchange_minute": accepted["confirmation_t_plus_5"], **accepted})

    def _mark_and_manage(self, minute: int) -> None:
        position = self.position
        if position is None:
            return
        call_ask = self._best_ask(self.order_book.get(position.call_token))
        put_ask = self._best_ask(self.order_book.get(position.put_token))
        if call_ask is None or put_ask is None:
            return
        position.last_call_ask, position.last_put_ask = call_ask, put_ask
        debit = call_ask + put_ask
        position.unrealized_pnl = combined_short_mtm(position.entry_credit, debit, position.lot_size, position.open_lots)
        timestamp = NSE_EPOCH + timedelta(minutes=minute)
        if timestamp.hour * 60 + timestamp.minute >= int(self.config.get("paper_force_exit_minute", 915)):
            self._close_lots(position.open_lots, debit, minute, "end_of_day_exit")
            return
        if position.unrealized_pnl <= -float(self.config.get("paper_stop_loss_per_lot", 1500.0)) * position.open_lots:
            self._close_lots(position.open_lots, debit, minute, "fixed_rupee_stop")
            return
        if self.current_spot is not None and bool(self.config.get("paper_exit_at_bep", True)) and not position.lower_bep < self.current_spot < position.upper_bep:
            self._close_lots(position.open_lots, debit, minute, "break_even_breach")
            return
        if position.unrealized_pnl >= float(self.config.get("paper_profit_target_per_lot", 3000.0)) * position.open_lots:
            self._close_lots(position.open_lots, debit, minute, "fixed_rupee_profit_target")
            return
        if minute >= position.planned_exit_minute:
            if self._extend_if_supported(position, minute):
                return
            self._close_lots(position.open_lots, debit, minute, "model_horizon_exit")

    def _extend_if_supported(self, position: PaperPosition, minute: int) -> bool:
        max_extensions = int(self.config.get("paper_max_extensions", 4))
        extension = int(self.config.get("paper_extension_minutes", 15))
        reasons: list[str] = []
        forecasts: dict[str, dict] = {}
        if position.extensions >= max_extensions:
            reasons.append("maximum_extensions_reached")
        if minute + extension > position.maximum_exit_minute:
            reasons.append("maximum_holding_period_reached")
        if not self.latest_spot_prediction or not self.latest_iv_prediction or self.pnl_adapter is None:
            reasons.append("fresh_stacked_forecast_unavailable")
        if not reasons:
            forecasts = self.pnl_adapter.score(self.latest_spot_prediction, self.latest_iv_prediction)
            eligible = []
            for horizon_text, forecast in forecasts.items():
                horizon = int(horizon_text)
                if (forecast["profit_probability"] >= float(self.config.get("paper_extension_min_probability", 0.55)) and
                        forecast["expected_pnl"] >= 0.0 and forecast["pnl_q10"] >= -500.0 and
                        forecast["mae_q90"] <= float(self.config.get("paper_extension_max_mae_q90", 3000.0)) and
                        forecast["hmm_in_distribution"]):
                    score = forecast["expected_pnl"] * forecast["profit_probability"] / max(abs(min(forecast["pnl_q10"], 0.0)), 100.0)
                    eligible.append((score, horizon, forecast))
            if not eligible:
                reasons.append("next_horizon_not_favorable")
            else:
                score, horizon, chosen = max(eligible)
                position.extensions += 1
                position.planned_exit_minute = minute + extension
                event = {"event_type": "holding_extension_decision", "decision": "EXTEND", "trade_id": position.trade_id,
                         "exchange_minute": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
                         "new_planned_exit_minute": position.planned_exit_minute, "extensions": position.extensions,
                         "supporting_horizon": horizon, "selection_score": score, "pnl_forecast": chosen}
                self._write_decision(event); self._write(event)
                return True
        event = {"event_type": "holding_extension_decision", "decision": "EXIT", "trade_id": position.trade_id,
                 "exchange_minute": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
                 "reasons": reasons, "extensions": position.extensions, "forecasts": forecasts}
        self._write_decision(event); self._write(event)
        return False

    def _close_lots(self, lots: int, debit: float, minute: int, reason: str) -> None:
        position = self.position
        if position is None or lots <= 0:
            return
        lots = min(lots, position.open_lots)
        pnl = combined_short_mtm(position.entry_credit, debit, position.lot_size, lots)
        costs = float(self.config.get("paper_round_trip_cost_per_lot", 100.0)) * lots
        net = pnl - costs
        position.realized_pnl += net
        position.open_lots -= lots
        position.margin_reserved *= position.open_lots / (position.open_lots + lots)
        self.realized_pnl += net
        self.daily_realized_pnl += net
        self.cash += net
        event = {"event_type": "paper_exit", "trade_id": position.trade_id,
                 "exchange_minute": (NSE_EPOCH + timedelta(minutes=minute)).isoformat(sep=" "),
                 "reason": reason, "lots": lots, "exit_debit": debit, "gross_pnl": pnl,
                 "costs": costs, "net_pnl": net, "remaining_lots": position.open_lots}
        self._write(event)
        if position.open_lots == 0:
            completed = {**asdict(position), "exit_time": event["exchange_minute"], "exit_reason": reason}
            self.closed_trades.append(completed)
            self.closed_trades = self.closed_trades[-100:]
            self.position = None

    def _write(self, event: dict) -> None:
        self.output.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")

    def _write_decision(self, event: dict) -> None:
        self.decision_output.write(json.dumps(event, separators=(",", ":"), allow_nan=False) + "\n")

    def snapshot(self) -> dict:
        with self.lock:
            position = asdict(self.position) if self.position else None
            equity = self.cash + (self.position.unrealized_pnl if self.position else 0.0)
            return {
                "enabled": self.enabled, "mode": "PAPER ONLY", "starting_capital": self.starting_capital,
                "cash": self.cash, "equity": equity, "realized_pnl": self.realized_pnl,
                "daily_realized_pnl": self.daily_realized_pnl, "unrealized_pnl": 0.0 if not self.position else self.position.unrealized_pnl,
                "margin_used": 0.0 if not self.position else self.position.margin_reserved,
                "margin_available": max(0.0, equity - (0.0 if not self.position else self.position.margin_reserved)),
                "spot": self.current_spot, "india_vix": self.current_vix, "vix_change_5m": self._vix_change(5),
                "spot_prediction": self.latest_spot_prediction, "iv_prediction": self.latest_iv_prediction,
                "pnl_forecasts": self.latest_pnl_forecasts,
                "stacked_prediction_minute": None if self.latest_stacked_minute is None else (NSE_EPOCH + timedelta(minutes=self.latest_stacked_minute)).isoformat(sep=" "),
                "latest_decision": self.latest_decision,
                "position": position, "closed_trades": list(reversed(self.closed_trades[-20:])),
                "spot_actual_vs_predicted": list(reversed(self.spot_prediction_audit[-100:])),
                "iv_actual_vs_predicted": list(reversed(self.iv_prediction_audit[-100:])),
                "rejections": list(reversed(self.rejections[-15:])), "updated_at": datetime.now().isoformat(timespec="seconds"),
            }

    def close(self) -> None:
        self.output.close()
        self.decision_output.close()
