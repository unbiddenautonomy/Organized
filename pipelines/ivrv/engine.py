"""LiveFeatureInference: the shared ATM feature stream plus IVRV scoring, paper books and published predictions."""
from __future__ import annotations

from collections import deque
from datetime import date, datetime, time, timedelta
import json
import logging
import math
import threading
from typing import Any, TextIO
import numpy as np
import pandas as pd
from .model import FixedStrikeInferencePipeline
from voltr.core.clock import NSE_EPOCH
from voltr.market.atm_features import LAGS, REQUIRED_FEATURE_MINUTES, AtmFeatureStream
from . import paths
from .strategy import _json_value, current_costs, paper_entry_route
from .book import PaperTradingEngine
from .demo_positions import DemoPositionLedger
from .paper_history import closed_paper_positions

LOGGER = logging.getLogger("fixed_strike_live")  # one historic logger name for all IVRV modules


PIPELINE_GENERATED_FEATURES = {
    "rv_pred_120m",
    "entry_gap_to_rv_pred_combined",
    "engineered_log_entry_iv",
    "engineered_straddle_to_spot",
    "engineered_put_call_oi_ratio",
    "engineered_put_call_oi_imbalance",
    *(f"engineered_iv_minus_rv_{horizon}m" for horizon in (5, 15, 30, 60)),
    *(f"engineered_iv_to_rv_ratio_{horizon}m" for horizon in (5, 15, 30, 60)),
    *(f"engineered_log_rv_{horizon}m" for horizon in (5, 15, 30, 60)),
    *(
        f"engineered_rv_term_slope_{short}m_{long}m"
        for short, long in ((5, 15), (15, 30), (30, 60))
    ),
}


def required_live_input_features(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    """Return model features that must exist before the stacked pipeline runs."""
    required: set[str] = set()
    for name, payload in artifacts.items():
        if name == "iv_gt_rv_rising_120m":
            continue  # Its training schema is mapped from live fields in the pipeline.
        required.update(payload.get("features", ()))
    return sorted(required - PIPELINE_GENERATED_FEATURES)


def _weeks43_notify(engine, method, *args):
    bridge = getattr(engine, "weeks43", None)
    if bridge is not None:
        try:
            getattr(bridge, method)(*args)
        except Exception:
            LOGGER.exception("Optional 43Weeks extension failed during %s", method)


class LiveFeatureInference(AtmFeatureStream):
    """IVRV: scores each completed ATM row, runs the paper books and publishes to the dashboard."""

    def __init__(
        self,
        config: dict,
        contracts: dict[int, Any],
        lot_size: int,
        expiry: date,
        future_token: int,
        pipeline: FixedStrikeInferencePipeline,
        source: dict[str, Any],
    ) -> None:
        super().__init__(config, contracts, lot_size, expiry, future_token, source)
        if len(self.strikes) < 5:
            raise RuntimeError("monthly option chain contains fewer than five paired strikes")
        self.pipeline = pipeline
        self.events: deque[dict] = deque(maxlen=500)
        self.latest_event: dict = {
            "state": "WAITING",
            "reason": "waiting_for_spot_future_and_option_udp_packets",
        }
        self.lock = threading.RLock()

        prediction_path = paths.BASE_DIR / str(config.get("prediction_log", "logs/live_predictions.jsonl"))
        fixed_strike_path = paths.BASE_DIR / str(
            config.get("fixed_strike_path_log", "logs/fixed_strike_120m_paths.jsonl")
        )
        latest_path = paths.BASE_DIR / str(config.get("latest_prediction", "logs/latest_prediction.json"))
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        fixed_strike_path.parent.mkdir(parents=True, exist_ok=True)
        latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.prediction_path = prediction_path.resolve()
        self.fixed_strike_path = fixed_strike_path.resolve()
        self.latest_path = latest_path.resolve()
        self.output: TextIO = self.prediction_path.open("a", encoding="utf-8", buffering=1)
        self.fixed_strike_output: TextIO = self.fixed_strike_path.open(
            "a", encoding="utf-8", buffering=1
        )
        self._restore_prediction_history()
        self.paper_engine = PaperTradingEngine(config, lot_size)
        self.long_paper_engine = PaperTradingEngine(config, lot_size, "LONG_STRADDLE")
        self.long_paper_engine.convergence_probability_threshold = pipeline.config.rising_rv_probability_threshold
        self.demo_ledger = DemoPositionLedger(
            self.prediction_path.with_name(self.prediction_path.stem + "_demo_positions.json"), lot_size
        )
        self.last_demo_csv_date: str | None = None

        self.required_feature_names = required_live_input_features(self.pipeline.artifacts)

    def _restore_prediction_history(self) -> None:
        """Restore the current active log into the dashboard's bounded history."""
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
            LOGGER.warning("Could not restore prediction history from %s", self.prediction_path)
        if self.events:
            self.latest_event = dict(self.events[-1])

    def _on_option_tick(self, token: int, price: float, log_time: int) -> None:
        _weeks43_notify(self, "tick", token, price, log_time)

    def _on_open_interest(self, token: int, open_interest: int, log_time: int) -> None:
        _weeks43_notify(self, "oi", token, open_interest, log_time)

    def _set_wait(self, minute: int, reason: str, **details: Any) -> None:
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        event = {
            "state": "WARMUP",
            "exchange_minute": stamp.isoformat(sep=" "),
            "reason": reason,
            "available_feature_minutes": len(self.rows_by_minute),
            "required_feature_minutes": REQUIRED_FEATURE_MINUTES,
            **details,
        }
        with self.lock:
            self.latest_event = event

    def _finalize_minute(self, minute: int) -> None:
        _weeks43_notify(self, "snapshot", minute)
        spot_bar = self.spot_bars.get(minute)
        future_bar = self.future_bars.get(minute)
        if spot_bar is None or future_bar is None:
            self._set_wait(minute, "missing_completed_spot_or_front_future_bar")
            return
        spot = spot_bar.close
        self._save_fixed_strike_path_observations(minute, spot, future_bar.open)
        nearest = min(range(len(self.strikes)), key=lambda index: abs(self.strikes[index] - spot))
        if nearest < 2 or nearest + 2 >= len(self.strikes):
            self._set_wait(minute, "five_strike_window_outside_contract_chain", spot=spot)
            return
        selected = self.strikes[nearest - 2 : nearest + 3]
        prices: dict[float, dict[str, float]] = {}
        for strike in selected:
            pair = self.pairs[strike]
            call = self.option_bars.get(pair["CE"], {}).get(minute)
            put = self.option_bars.get(pair["PE"], {}).get(minute)
            if call is None or put is None:
                self._set_wait(minute, "missing_completed_option_bar", spot=spot, strike=strike)
                return
            prices[strike] = {"CE": call.close, "PE": put.close}

        row = self._build_current_row(minute, spot, future_bar.open, selected, prices)
        if row is None:
            self._set_wait(minute, "invalid_iv_or_surface", spot=spot)
            return
        self.rows_by_minute[minute] = row
        _weeks43_notify(self, "atm_row", dict(row), self.latest_exchange_second)
        self.spot_by_minute[minute] = spot
        wide = self._add_lags_and_spot_features(minute, row)
        missing_columns = [name for name in self.required_feature_names if name not in wide]
        if missing_columns:
            self._set_wait(minute, "feature_contract_incomplete", missing_features=missing_columns)
            return
        if any((minute - lag) not in self.rows_by_minute for lag in LAGS):
            self._set_wait(
                minute,
                f"need_{REQUIRED_FEATURE_MINUTES}_consecutive_feature_minutes",
                spot=spot,
                strike=row["entry_strike"],
            )
            return
        try:
            prediction = self.pipeline.predict(pd.DataFrame([wide])).iloc[0].to_dict()
        except Exception as error:
            LOGGER.exception("live prediction failed")
            self._set_wait(minute, "model_prediction_error", error=str(error))
            return
        threshold = self.paper_engine.convergence_probability_threshold
        convergence_probability = prediction.get("pred_convergence_probability_classifier")
        routing = paper_entry_route(
            prediction,
            threshold,
            self.paper_engine.iv_rv_near_gap_decimal,
            self.paper_engine.rv_decline_event_score_threshold,
        )
        row.update({
            "entry_no120_event_score": prediction.get("pred_no120_event_score"),
            "entry_no120_sharp_rv_decline": prediction.get("pred_no120_sharp_rv_decline"),
            "entry_no120_iv_rv_gate_open": prediction.get(
                "pred_no120_gate_iv_near_or_below_rv"
            ),
            "entry_convergence_probability_classifier": convergence_probability,
            "entry_convergence_probability_threshold": threshold,
            "entry_short_straddle_signal": bool(routing["eligible"]),
            "entry_paper_route": routing["route"],
            "entry_legacy_three_rule_signal": prediction.get("pred_short_straddle_120m_signal"),
            "entry_rule_iv_above_rv_iv_declines_to_rv": prediction.get(
                "pred_strategy_rule_iv_above_rv_iv_declines_to_rv"
            ),
            "entry_rule_iv_rv_within_half_point": prediction.get(
                "pred_strategy_rule_iv_rv_within_half_point"
            ),
            "entry_rule_iv_below_rv_fast_rv_decline": prediction.get(
                "pred_strategy_rule_iv_below_rv_fast_rv_decline"
            ),
        })
        paper_entry_decision = self._try_open_paper_position(minute, row, prediction)
        long_paper_entry_decision = self._try_open_long_paper_position(minute, row, prediction)
        self._consider_demo_position(minute, row, prediction, paper_entry_decision,
                                     self.paper_engine, "SHORT_STRADDLE",
                                     "pred_convergence_probability_classifier")
        self._consider_demo_position(minute, row, prediction, long_paper_entry_decision,
                                     self.long_paper_engine, "LONG_STRADDLE",
                                     "pred_rising_rv_convergence_probability")
        self._publish(minute, wide, prediction, paper_entry_decision, long_paper_entry_decision)

    def _consider_demo_position(self, minute: int, row: dict, prediction: dict,
                                paper_decision: dict, paper_engine: PaperTradingEngine,
                                side: str, probability_field: str) -> None:
        try:
            probability = float(prediction.get(probability_field, math.nan))
            if paper_decision.get("entered") or not math.isfinite(probability) or probability <= paper_engine.convergence_probability_threshold:
                return
            strike = float(row["entry_strike"])
            pair = self.pairs.get(strike)
            second = self.latest_exchange_second
            if pair is None or second is None:
                return
            prices = self._fresh_paper_prices(pair["CE"], pair["PE"], second)
            if prices is None:
                return
            self.demo_ledger.consider(
                signal_time=NSE_EPOCH + timedelta(minutes=minute),
                entry_time=NSE_EPOCH + timedelta(seconds=second),
                expiry=self.expiry.isoformat(), strike=strike, probability=probability,
                threshold=paper_engine.convergence_probability_threshold,
                paper_entered=False, paper_skip_reason=str(paper_decision.get("reason", "")),
                call_token=pair["CE"], put_token=pair["PE"],
                call_price=prices[0], put_price=prices[1],
                target=paper_engine.target, stop=paper_engine.stop, side=side,
            )
        except (OSError, ValueError, KeyError, TypeError):
            LOGGER.exception("Demo position observer failed to record a signal")

    def _mark_demo_positions(self, exchange_second: int, updated_token: int | None = None) -> None:
        try:
            stamp = NSE_EPOCH + timedelta(seconds=int(exchange_second))
            self.demo_ledger.mark(
                stamp, self.latest_option_ticks, int(exchange_second), updated_token,
                max_age_seconds=int(self.config.get("paper_price_max_age_seconds", 60)),
            )
            if stamp.time() >= time(15, 15) and self.last_demo_csv_date != stamp.date().isoformat():
                self.demo_ledger.close_session(stamp)
                self.demo_ledger.save_session_csv(stamp.date().isoformat())
                self.last_demo_csv_date = stamp.date().isoformat()
        except (OSError, ValueError, TypeError):
            LOGGER.exception("Demo position observer failed to mark or export")

    def demo_positions(self) -> dict:
        return self.demo_ledger.snapshot()

    def demo_csv(self) -> bytes:
        return self.demo_ledger.csv_bytes()

    def closed_paper_positions(self) -> dict:
        return closed_paper_positions(paths.BASE_DIR, self.config)

    def _fresh_paper_prices(
        self, call_token: int, put_token: int, exchange_second: int, *, enforce_age: bool = True
    ) -> tuple[float, float] | None:
        call_tick = self.latest_option_ticks.get(call_token)
        put_tick = self.latest_option_ticks.get(put_token)
        if call_tick is None or put_tick is None:
            return None
        if enforce_age:
            maximum_age = int(self.config.get("paper_price_max_age_seconds", 60))
            if any(
                exchange_second - tick_time > maximum_age or tick_time > exchange_second
                for _, tick_time in (call_tick, put_tick)
            ):
                return None
        return float(call_tick[0]), float(put_tick[0])

    def _try_open_paper_position(
        self, minute: int, row: dict[str, Any], prediction: dict[str, Any]
    ) -> dict[str, Any]:
        threshold = self.paper_engine.convergence_probability_threshold
        raw_probability = prediction.get("pred_convergence_probability_classifier")
        try:
            probability = float(raw_probability)
        except (TypeError, ValueError):
            probability = math.nan
        routing = paper_entry_route(
            prediction,
            threshold,
            self.paper_engine.iv_rv_near_gap_decimal,
            self.paper_engine.rv_decline_event_score_threshold,
        )
        decision: dict[str, Any] = {
            "rule": "regime_routed_iv_convergence_or_sharp_rv_decline",
            "probability": _json_value(probability),
            "threshold": threshold,
            "strict_operator": ">",
            "route": routing["route"],
            "iv_minus_rv": routing["iv_minus_rv"],
            "near_gap_decimal": routing["near_gap_decimal"],
            "rv_decline_event_score": routing["rv_decline_event_score"],
            "rv_decline_event_score_threshold": routing[
                "rv_decline_event_score_threshold"
            ],
            "rv_decline_event_score_operator": ">=",
            "maximum_open_positions": self.paper_engine.max_open_positions,
            "open_positions_before": len(self.paper_engine.positions),
            "eligible": bool(routing["eligible"]),
            "entered": False,
        }
        if not decision["eligible"]:
            decision.update({"state": "NO_TRADE", "reason": routing["reason"]})
            return decision
        strike = float(row["entry_strike"])
        pair = self.pairs.get(strike)
        exchange_second = self.latest_exchange_second
        if pair is None or exchange_second is None:
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "missing_contract_or_exchange_time"})
            return decision
        if (NSE_EPOCH + timedelta(seconds=int(exchange_second))).time() >= self.paper_engine.square_off_time:
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "after_square_off"})
            return decision
        prices = self._fresh_paper_prices(pair["CE"], pair["PE"], exchange_second)
        if prices is None:
            LOGGER.warning("PAPER_ENTRY_SKIPPED stale_or_missing_prices strike=%.0f", strike)
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "stale_or_missing_prices"})
            return decision
        if len(self.paper_engine.positions) >= self.paper_engine.max_open_positions:
            signal_stamp = NSE_EPOCH + timedelta(minutes=minute)
            new_position_id = f"{signal_stamp.isoformat(sep=' ')}|{self.expiry.isoformat()}|{strike:g}"
            if new_position_id in self.paper_engine.positions:
                decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "entry_already_open"})
                return decision
            candidate, replacement_reason = self.paper_engine.replacement_candidate(
                probability, str(routing["route"]), int(exchange_second)
            )
            if candidate is None:
                decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": replacement_reason})
                return decision
            old_prices = self._fresh_paper_prices(
                int(candidate["call_token"]), int(candidate["put_token"]), exchange_second
            )
            if old_prices is None:
                decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "replacement_exit_prices_stale_or_missing"})
                return decision
            exit_event = self.paper_engine.mark(
                str(candidate["position_id"]), exchange_second, old_prices[0], old_prices[1],
                force_reason="higher_confidence_signal", replacement_probability=probability,
            )
            if exit_event is None:
                decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "replacement_exit_not_accepted"})
                return decision
            decision["replaced_position_id"] = exit_event["position_id"]
            decision["replaced_position_confidence"] = candidate["entry_convergence_probability"]
            decision["replacement_exit_pnl_inr"] = exit_event["realized_pnl_inr"]
            LOGGER.warning(
                "PAPER_POSITION_REPLACED id=%s old_confidence=%.4f new_confidence=%.4f exit_pnl=%.2f",
                exit_event["position_id"], float(candidate["entry_convergence_probability"]),
                probability, exit_event["realized_pnl_inr"],
            )
        triggered = [str(routing["route"])]
        entry_model_score = (
            probability
            if routing["route"] == "iv_above_rv_convergence"
            else routing["rv_decline_event_score"]
        )
        entered = self.paper_engine.try_enter(
            exchange_second=exchange_second,
            signal_minute=minute,
            expiry=self.expiry,
            strike=strike,
            call_token=pair["CE"],
            put_token=pair["PE"],
            call_price=prices[0],
            put_price=prices[1],
            triggered_rules=triggered,
            convergence_probability=probability,
            entry_route=str(routing["route"]),
            entry_model_score=entry_model_score,
        )
        if entered:
            LOGGER.warning(
                "PAPER_POSITION_OPEN strike=%.0f call=%.2f put=%.2f route=%s model_score=%s",
                strike, prices[0], prices[1], routing["route"], entry_model_score,
            )
            decision.update({"state": "PAPER_ENTRY_TAKEN",
                             "reason": "position_opened_after_replacement" if "replaced_position_id" in decision else "position_opened",
                             "entered": True})
        else:
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "entry_rejected"})
        decision["open_positions_after"] = len(self.paper_engine.positions)
        return decision

    def _try_open_long_paper_position(
        self, minute: int, row: dict[str, Any], prediction: dict[str, Any]
    ) -> dict[str, Any]:
        engine = self.long_paper_engine
        threshold = self.pipeline.config.rising_rv_probability_threshold
        probability = prediction.get("pred_rising_rv_convergence_probability")
        eligible = bool(prediction.get("pred_rising_rv_convergence_signal", False)) and bool(row.get("quality_entry_ok", False))
        decision: dict[str, Any] = {
            "side": "LONG_STRADDLE", "rule": "iv_above_rv_rising_rv_convergence",
            "probability": _json_value(probability), "threshold": threshold,
            "eligible": eligible, "entered": False, "state": "NO_TRADE",
            "reason": "model_or_quality_gate_failed", "open_positions_before": len(engine.positions),
        }
        if not eligible:
            return decision
        # The rising-RV long book is kept as a deliberate feature but opens no
        # new paper positions unless live_config.json enables it explicitly.
        if not bool(getattr(self, "config", {}).get("paper_long_entries_enabled", False)):
            decision.update(state="QUALIFIED_BUT_SKIPPED", reason="long_entries_disabled")
            return decision
        strike = float(row["entry_strike"])
        pair = self.pairs.get(strike)
        exchange_second = self.latest_exchange_second
        if pair is None or exchange_second is None:
            decision.update(state="QUALIFIED_BUT_SKIPPED", reason="missing_contract_or_exchange_time")
            return decision
        if (NSE_EPOCH + timedelta(seconds=int(exchange_second))).time() >= engine.square_off_time:
            decision.update(state="QUALIFIED_BUT_SKIPPED", reason="after_square_off")
            return decision
        prices = self._fresh_paper_prices(pair["CE"], pair["PE"], exchange_second)
        if prices is None:
            decision.update(state="QUALIFIED_BUT_SKIPPED", reason="stale_or_missing_prices")
            return decision
        if len(engine.positions) >= engine.max_open_positions:
            candidate, replacement_reason = engine.replacement_candidate(
                float(probability), "iv_above_rv_rising_rv_convergence", int(exchange_second)
            )
            if candidate is None:
                decision.update(state="QUALIFIED_BUT_SKIPPED", reason=replacement_reason)
                return decision
            old_prices = self._fresh_paper_prices(int(candidate["call_token"]), int(candidate["put_token"]), exchange_second)
            if old_prices is None:
                decision.update(state="QUALIFIED_BUT_SKIPPED", reason="replacement_exit_prices_stale_or_missing")
                return decision
            exit_event = engine.mark(str(candidate["position_id"]), exchange_second, old_prices[0], old_prices[1],
                                     force_reason="higher_confidence_signal", replacement_probability=float(probability))
            if exit_event is None:
                decision.update(state="QUALIFIED_BUT_SKIPPED", reason="replacement_exit_not_accepted")
                return decision
            decision["replaced_position_id"] = exit_event["position_id"]
            decision["replacement_exit_pnl_inr"] = exit_event["realized_pnl_inr"]
        entered = engine.try_enter(
            exchange_second=exchange_second, signal_minute=minute, expiry=self.expiry,
            strike=strike, call_token=pair["CE"], put_token=pair["PE"],
            call_price=prices[0], put_price=prices[1],
            triggered_rules=["iv_above_rv_rising_rv_convergence"],
            convergence_probability=float(probability), entry_route="iv_above_rv_rising_rv_convergence",
            entry_model_score=float(probability),
        )
        decision.update(state="PAPER_ENTRY_TAKEN" if entered else "QUALIFIED_BUT_SKIPPED",
                        reason="position_opened" if entered else "entry_rejected", entered=entered,
                        open_positions_after=len(engine.positions))
        return decision

    def _mark_paper_position(
        self,
        exchange_second: int,
        *,
        updated_token: int | None = None,
        time_only: bool = False,
    ) -> None:
        engines = [self.paper_engine]
        long_engine = getattr(self, "long_paper_engine", None)
        if long_engine is not None:
            engines.append(long_engine)
        positions = [(engine, position) for engine in engines for position in engine.positions.values()]
        if not positions:
            return
        stamp = NSE_EPOCH + timedelta(seconds=int(exchange_second))
        at_cutoff = stamp.time() >= self.paper_engine.square_off_time
        if time_only:
            if self.last_paper_time_check_second == int(exchange_second):
                return
            self.last_paper_time_check_second = int(exchange_second)
            if not at_cutoff:
                positions = [
                    (engine, position) for engine, position in positions
                    if int(exchange_second) - int(position["entry_exchange_second"])
                    >= self.paper_engine.horizon_minutes * 60
                ]
        elif updated_token is not None and not at_cutoff:
            positions = [
                (engine, position) for engine, position in positions
                if updated_token in (int(position["call_token"]), int(position["put_token"]))
            ]
        if not positions:
            return
        exits: list[dict[str, Any]] = []
        for engine, position in positions:
            prices = self._fresh_paper_prices(
                int(position["call_token"]), int(position["put_token"]),
                int(exchange_second),
                enforce_age=stamp.time() < self.paper_engine.square_off_time,
            )
            if prices is None:
                continue
            exit_event = engine.mark(
                str(position["position_id"]), exchange_second, prices[0], prices[1]
            )
            if exit_event is not None:
                exits.append(exit_event)
                LOGGER.warning(
                    "PAPER_POSITION_CLOSED id=%s strike=%.0f reason=%s pnl=%.2f holding=%.2f",
                    exit_event["position_id"], exit_event["fixed_strike"],
                    exit_event["exit_reason"], exit_event["realized_pnl_inr"],
                    exit_event["holding_minutes"],
                )
        if not exits:
            return
        with self.lock:
            if self.latest_event:
                self.latest_event["paper_trading"] = self.paper_engine.snapshot()
                if long_engine is not None:
                    self.latest_event["paper_long_trading"] = long_engine.snapshot()
                latest = dict(self.latest_event)
            else:
                latest = None
        if latest is not None:
            temporary = self.latest_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps(latest, ensure_ascii=False, allow_nan=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.latest_path)

    def _save_fixed_strike_path_observations(
        self, minute: int, spot: float, future_open: float
    ) -> None:
        """Save same-strike price paths for ATM entries from the prior horizon."""
        horizon = int(self.config.get("fixed_strike_path_horizon_minutes", 120))
        if horizon <= 0:
            return
        observation_stamp = NSE_EPOCH + timedelta(minutes=minute)
        expiry_stamp = datetime.combine(self.expiry, time(15, 30))
        years = max((expiry_stamp - observation_stamp).total_seconds(), 60.0) / (365.0 * 86400.0)
        target = float(self.config.get("paper_profit_target_inr", 1500.0))
        stop = float(self.config.get("paper_stop_loss_inr", -750.0))
        snapshots: dict[float, dict[str, Any] | None] = {}

        def snapshot(strike: float) -> dict[str, Any] | None:
            if strike in snapshots:
                return snapshots[strike]
            pair = self.pairs.get(strike)
            call_bar = self.option_bars.get(pair["CE"], {}).get(minute) if pair else None
            put_bar = self.option_bars.get(pair["PE"], {}).get(minute) if pair else None
            if pair is None or call_bar is None or put_bar is None:
                snapshots[strike] = None
                return None
            call_price = float(call_bar.close)
            put_price = float(put_bar.close)
            try:
                call_iv_value = self.implied_volatility(
                    call_price, spot, strike, years, self.rate, True
                )
                put_iv_value = self.implied_volatility(
                    put_price, spot, strike, years, self.rate, False
                )
                call_iv = float(call_iv_value)
                put_iv = float(put_iv_value)
            except (ArithmeticError, TypeError, ValueError, OverflowError):
                call_iv = put_iv = math.nan
            if not math.isfinite(call_iv) or call_iv <= 0:
                call_iv = math.nan
            if not math.isfinite(put_iv) or put_iv <= 0:
                put_iv = math.nan
            result = {
                "pair": pair,
                "call_price": call_price,
                "put_price": put_price,
                "straddle_price": call_price + put_price,
                "call_iv": call_iv,
                "put_iv": put_iv,
                "combined_iv": (
                    (call_iv + put_iv) / 2.0
                    if math.isfinite(call_iv) and math.isfinite(put_iv)
                    else math.nan
                ),
            }
            snapshots[strike] = result
            return result

        first_entry_minute = minute - horizon
        for entry_minute in range(first_entry_minute, minute):
            entry = self.rows_by_minute.get(entry_minute)
            if entry is None:
                continue
            holding_minutes = minute - entry_minute
            strike = float(entry["entry_strike"])
            observation = snapshot(strike)
            entry_stamp = NSE_EPOCH + timedelta(minutes=entry_minute)
            record: dict[str, Any] = {
                "record_type": "fixed_strike_path_observation",
                "entry_datetime": entry_stamp.isoformat(sep=" "),
                "entry_id": (
                    f"{entry_stamp.isoformat(sep=' ')}|{self.expiry.isoformat()}|{strike:g}"
                ),
                "observation_datetime": observation_stamp.isoformat(sep=" "),
                "holding_minutes": holding_minutes,
                "horizon_minutes": horizon,
                "is_horizon_exit": holding_minutes == horizon,
                "expiry": self.expiry.isoformat(),
                "entry_strike": strike,
                "lot_size": self.lot_size,
                "entry_spot": _json_value(entry.get("spot_price")),
                "entry_future_open": _json_value(entry.get("future_open")),
                "entry_call_price": _json_value(entry.get("call_price")),
                "entry_put_price": _json_value(entry.get("put_price")),
                "entry_straddle_price": _json_value(entry.get("entry_straddle_price")),
                "entry_call_iv": _json_value(entry.get("entry_iv_call")),
                "entry_put_iv": _json_value(entry.get("entry_iv_put")),
                "entry_combined_iv": _json_value(entry.get("entry_atm_iv_combined")),
                "entry_quality_ok": _json_value(entry.get("quality_entry_ok")),
                "entry_no120_event_score": _json_value(entry.get("entry_no120_event_score")),
                "entry_no120_sharp_rv_decline": _json_value(
                    entry.get("entry_no120_sharp_rv_decline")
                ),
                "entry_no120_iv_rv_gate_open": _json_value(
                    entry.get("entry_no120_iv_rv_gate_open")
                ),
                "entry_convergence_probability_classifier": _json_value(
                    entry.get("entry_convergence_probability_classifier")
                ),
                "entry_convergence_probability_threshold": _json_value(
                    entry.get("entry_convergence_probability_threshold")
                ),
                "entry_short_straddle_signal": _json_value(
                    entry.get("entry_short_straddle_signal")
                ),
                "entry_paper_route": _json_value(entry.get("entry_paper_route")),
                "entry_rv_decline_event_score_threshold": float(
                    self.config.get("paper_entry_rv_decline_event_score_threshold", 0.0)
                ),
                "entry_legacy_three_rule_signal": _json_value(
                    entry.get("entry_legacy_three_rule_signal")
                ),
                "entry_rule_iv_above_rv_iv_declines_to_rv": _json_value(
                    entry.get("entry_rule_iv_above_rv_iv_declines_to_rv")
                ),
                "entry_rule_iv_rv_within_half_point": _json_value(
                    entry.get("entry_rule_iv_rv_within_half_point")
                ),
                "entry_rule_iv_below_rv_fast_rv_decline": _json_value(
                    entry.get("entry_rule_iv_below_rv_fast_rv_decline")
                ),
                "profit_target_inr": target,
                "stop_loss_inr": stop,
                "observation_spot": _json_value(spot),
                "observation_future_open": _json_value(future_open),
                "observation_call_price": None,
                "observation_put_price": None,
                "observation_straddle_price": None,
                "observation_call_iv": None,
                "observation_put_iv": None,
                "observation_combined_iv": None,
                "observation_call_oi": None,
                "observation_put_oi": None,
                "short_pnl_per_unit": None,
                "short_pnl_per_lot": None,
                "profit_target_hit_at_observation": None,
                "stop_loss_hit_at_observation": None,
                "status": "missing_exit_option_bar",
            }
            if observation is not None:
                pair = observation["pair"]
                exit_straddle = float(observation["straddle_price"])
                entry_straddle = float(entry["entry_straddle_price"])
                pnl_per_unit = entry_straddle - exit_straddle
                pnl_per_lot = pnl_per_unit * self.lot_size
                record.update({
                    "observation_call_price": observation["call_price"],
                    "observation_put_price": observation["put_price"],
                    "observation_straddle_price": exit_straddle,
                    "observation_call_iv": _json_value(observation["call_iv"]),
                    "observation_put_iv": _json_value(observation["put_iv"]),
                    "observation_combined_iv": _json_value(observation["combined_iv"]),
                    "observation_call_oi": _json_value(
                        self._oi_value(
                            pair["CE"], minute, int(self.config["option_oi_max_age_minutes"])
                        )
                    ),
                    "observation_put_oi": _json_value(
                        self._oi_value(
                            pair["PE"], minute, int(self.config["option_oi_max_age_minutes"])
                        )
                    ),
                    "short_pnl_per_unit": pnl_per_unit,
                    "short_pnl_per_lot": pnl_per_lot,
                    "profit_target_hit_at_observation": pnl_per_lot >= target,
                    "stop_loss_hit_at_observation": pnl_per_lot <= stop,
                    "status": "complete",
                })
            self.fixed_strike_output.write(
                json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                + "\n"
            )

    def _probability_analysis(self, inputs: dict[str, Any], prediction: dict[str, Any]) -> dict:
        """Record exact classifier inputs and per-feature log-odds contributions."""
        short_payload = self.pipeline.artifacts["combined_convergence_classifier"]
        short_inputs = dict(inputs)
        rv_prediction = float(prediction["pred_rv_120m_xgboost"])
        short_inputs["rv_pred_120m"] = rv_prediction
        short_inputs["entry_gap_to_rv_pred_combined"] = float(inputs["entry_atm_iv_combined"]) - rv_prediction
        short_matrix = self.pipeline._matrix(pd.DataFrame([short_inputs]), short_payload)

        spot = float(inputs["spot_price"])
        call_iv = float(inputs["entry_iv_call"])
        put_iv = float(inputs["entry_iv_put"])
        timestamp = pd.to_datetime(inputs["datetime"])
        long_inputs = {
            "current_iv": 50.0 * (call_iv + put_iv),
            "rv_30": inputs.get("rv_sample_30m"),
            "rv_60": inputs.get("rv_sample_60m"),
            "iv_skew": 100.0 * (put_iv - call_iv),
            "spot_price": spot,
            "strike": inputs["entry_strike"],
            "moneyness": (spot - float(inputs["entry_strike"])) / spot if spot else math.nan,
            "days_to_expiry": inputs.get("days_to_expiry"),
            "minute_of_day": timestamp.hour * 60 + timestamp.minute,
            "weekday": timestamp.weekday(),
        }
        long_payload = self.pipeline.artifacts["iv_gt_rv_rising_120m"]
        for name in long_payload["features"]:
            if name not in long_inputs:
                long_inputs[name] = inputs.get(name)
        long_matrix = self.pipeline._matrix(pd.DataFrame([long_inputs]), long_payload)

        def explain(payload: dict, matrix: pd.DataFrame, probability: Any,
                    threshold: float, eligible: bool) -> dict:
            feature_names = list(payload["features"])
            values = matrix.iloc[0]
            result = {
                "model": "LightGBM classifier",
                "probability": _json_value(probability),
                "threshold": threshold,
                "eligible": eligible,
                "feature_count": len(feature_names),
                "features": [{"name": name, "value": _json_value(values[name]),
                              "contribution_log_odds": None} for name in feature_names],
                "bias_log_odds": None,
            }
            if not eligible:
                return result
            contributions = np.asarray(payload["model"].booster_.predict(matrix, pred_contrib=True)).reshape(-1)
            if len(contributions) != len(feature_names) + 1:
                raise ValueError("Classifier contribution count differs from feature contract")
            for feature, contribution in zip(result["features"], contributions[:-1]):
                feature["contribution_log_odds"] = _json_value(contribution)
            result["bias_log_odds"] = _json_value(contributions[-1])
            raw_score = float(contributions.sum())
            exp_score = math.exp(-abs(raw_score))
            result["reconstructed_probability"] = (
                1.0 / (1.0 + exp_score) if raw_score >= 0 else exp_score / (1.0 + exp_score)
            )
            return result

        return {
            "short": explain(short_payload, short_matrix,
                             prediction.get("pred_convergence_probability_classifier"),
                             self.paper_engine.convergence_probability_threshold, True),
            "long": explain(long_payload, long_matrix,
                            prediction.get("pred_rising_rv_convergence_probability"),
                            self.pipeline.config.rising_rv_probability_threshold,
                            bool(prediction.get("pred_rising_rv_eligible", False))),
            "decision_context": {
                name: _json_value(prediction.get(name)) for name in (
                    "pred_convergence_vote_count", "pred_convergence_vote_fraction",
                    "pred_converged_direct_lightgbm", "pred_converged_leg_hgb_average",
                    "pred_converged_classifier", "pred_no120_gate_iv_near_or_below_rv",
                    "pred_no120_event_score", "pred_rising_rv_current_iv_points",
                    "pred_rising_rv_current_rv_points",
                )
            },
        }

    def _publish(
        self,
        minute: int,
        inputs: dict[str, Any],
        prediction: dict[str, Any],
        paper_entry_decision: dict[str, Any],
        long_paper_entry_decision: dict[str, Any] | None = None,
    ) -> None:
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        clean_predictions = {name: _json_value(value) for name, value in prediction.items()}
        prediction_fields = {name: value for name, value in clean_predictions.items() if name.startswith("pred_")}
        try:
            probability_analysis = self._probability_analysis(inputs, prediction)
        except Exception:
            LOGGER.exception("Could not build probability input explanation")
            probability_analysis = {"error": "probability_inputs_unavailable"}
        spot = float(inputs["spot_price"])
        entry_premium = float(inputs["entry_straddle_price"])
        costs = current_costs(entry_premium, entry_premium, self.lot_size, self.config)
        threshold = self.paper_engine.convergence_probability_threshold
        routing = paper_entry_route(
            prediction_fields,
            threshold,
            self.paper_engine.iv_rv_near_gap_decimal,
            self.paper_engine.rv_decline_event_score_threshold,
        )
        short_signal = bool(routing["eligible"])
        legacy_three_rule_signal = bool(
            prediction_fields.get("pred_short_straddle_120m_signal", False)
        )
        event = {
            "event_type": "fixed_strike_inference",
            "paper_only": True,
            "exchange_minute": stamp.isoformat(sep=" "),
            "state": "READY",
            "fixed_entry_strike": float(inputs["entry_strike"]),
            "expiry": self.expiry.isoformat(),
            "lot_size": self.lot_size,
            "market": {
                "spot": spot,
                "front_future_open": _json_value(inputs["future_open"]),
                "front_future_oi": _json_value(inputs["Future_oi"]),
                "call_price": _json_value(inputs["call_price"]),
                "put_price": _json_value(inputs["put_price"]),
                "straddle_price": entry_premium,
                "call_iv": _json_value(inputs["entry_iv_call"]),
                "put_iv": _json_value(inputs["entry_iv_put"]),
                "combined_iv": _json_value(inputs["entry_atm_iv_combined"]),
                "quality_entry_ok": bool(inputs["quality_entry_ok"]),
            },
            "spot_prediction": {
                "interpretation": "nondirectional range from predicted 120-minute RV",
                "reference_spot": spot,
                "one_sigma_low": prediction_fields.get("pred_spot_range_1sigma_low"),
                "one_sigma_high": prediction_fields.get("pred_spot_range_1sigma_high"),
                "approx_90pct_low": prediction_fields.get("pred_spot_range_90pct_low"),
                "approx_90pct_high": prediction_fields.get("pred_spot_range_90pct_high"),
            },
            "cost_estimate": {
                "basis": "current rates, zero brokerage, flat-premium round-trip estimate",
                **{name: _json_value(value) for name, value in costs.items()},
            },
            "strategy_signal": {
                "state": paper_entry_decision["state"],
                "execution_basis": "regime_routed_iv_convergence_or_sharp_rv_decline",
                "entry_eligible": short_signal,
                "entry_taken": bool(paper_entry_decision.get("entered", False)),
                "entry_reason": paper_entry_decision.get("reason"),
                "convergence_probability": prediction_fields.get(
                    "pred_convergence_probability_classifier"
                ),
                "convergence_probability_threshold": threshold,
                "probability_operator": ">",
                "entry_route": routing["route"],
                "iv_rv_near_gap_decimal": routing["near_gap_decimal"],
                "rv_decline_event_score": routing["rv_decline_event_score"],
                "rv_decline_event_score_threshold": routing[
                    "rv_decline_event_score_threshold"
                ],
                "rv_decline_event_score_operator": ">=",
                "maximum_open_positions": self.paper_engine.max_open_positions,
                "legacy_three_rule_signal": legacy_three_rule_signal,
                "model_evaluated": True,
                "gate_open": bool(
                    prediction_fields.get("pred_no120_gate_iv_near_or_below_rv", False)
                ),
                "model_predicts_sharp_rv_decline": bool(
                    prediction_fields.get("pred_no120_sharp_rv_decline", False)
                ),
                "rules": {
                    "iv_above_rv_and_predicted_iv_declines_to_predicted_rv": bool(
                        prediction_fields.get(
                            "pred_strategy_rule_iv_above_rv_iv_declines_to_rv", False
                        )
                    ),
                    "iv_rv_within_half_volatility_point": bool(
                        prediction_fields.get(
                            "pred_strategy_rule_iv_rv_within_half_point", False
                        )
                    ),
                    "iv_below_rv_and_fast_rv_decline": bool(
                        prediction_fields.get(
                            "pred_strategy_rule_iv_below_rv_fast_rv_decline", False
                        )
                    ),
                },
                "current_combined_iv": prediction_fields.get("entry_atm_iv_combined", clean_predictions.get("entry_atm_iv_combined")),
                "current_rv": prediction_fields.get("pred_no120_gate_current_rv"),
                "iv_minus_rv": prediction_fields.get("pred_no120_gate_iv_minus_rv"),
                "maximum_gate_gap": prediction_fields.get("pred_no120_gate_max_gap"),
                "trade_plan": {
                    "paper_only": True,
                    "position": "short_atm_straddle",
                    "fixed_strike": float(inputs["entry_strike"]),
                    "horizon_minutes": prediction_fields.get("pred_short_straddle_horizon_minutes"),
                    "profit_target_inr": prediction_fields.get("pred_short_straddle_profit_target_inr"),
                    "stop_loss_inr": prediction_fields.get("pred_short_straddle_stop_loss_inr"),
                },
            },
            "paper_entry_decision": paper_entry_decision,
            "paper_trading": self.paper_engine.snapshot(),
            "long_strategy_signal": {
                "side": "LONG_STRADDLE", "paper_only": True,
                "entry_eligible": bool(prediction_fields.get("pred_rising_rv_convergence_signal", False)) and bool(inputs["quality_entry_ok"]),
                "model_eligible": bool(prediction_fields.get("pred_rising_rv_eligible", False)),
                "probability": prediction_fields.get("pred_rising_rv_convergence_probability"),
                "threshold": self.pipeline.config.rising_rv_probability_threshold,
                "current_iv_points": prediction_fields.get("pred_rising_rv_current_iv_points"),
                "current_rv_points": prediction_fields.get("pred_rising_rv_current_rv_points"),
                "entry_taken": bool((long_paper_entry_decision or {}).get("entered", False)),
            },
            "long_paper_entry_decision": long_paper_entry_decision,
            "paper_long_trading": self.long_paper_engine.snapshot(),
            "prediction_count": len(prediction_fields),
            "predictions": prediction_fields,
            "probability_analysis": probability_analysis,
        }
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.output.write(encoded + "\n")
        from voltr.core.io import atomic_write_json as atomic_json
        atomic_json(self.latest_path,event)
        with self.lock:
            self.events.append(event)
            self.latest_event = event
        LOGGER.info("FIXED_STRIKE_PREDICTION minute=%s strike=%.0f fields=%d", stamp, inputs["entry_strike"], len(prediction_fields))
        if short_signal:
            from voltr.web.alerts import notify_signal
            notify_signal()
            LOGGER.warning(
                "PAPER_SHORT_STRADDLE_SIGNAL minute=%s strike=%.0f probability=%s decision=%s reason=%s",
                stamp,
                inputs["entry_strike"],
                prediction_fields.get("pred_convergence_probability_classifier"),
                paper_entry_decision.get("state"),
                paper_entry_decision.get("reason"),
            )

    def status(self) -> dict:
        with self.lock:
            result = dict(self.latest_event)
        result["paper_trading"] = self.paper_engine.snapshot()
        result["paper_long_trading"] = self.long_paper_engine.snapshot()
        result["runtime"] = {
            "paper_only": True,
            "mode": str(self.config.get("runtime_mode", "scheduled_live")),
            "source_root": str(paths.REPO_DIR / "voltr"),  # where the shared feed decoders live
            "expiry": self.expiry.isoformat(),
            "front_future_token": self.future_token,
            "lot_size": self.lot_size,
            "latest_spot_minute": self.latest_spot_minute,
            "latest_future_minute": self.latest_future_minute,
            "available_feature_minutes": len(self.rows_by_minute),
            "required_feature_minutes": REQUIRED_FEATURE_MINUTES,
            "server_time": datetime.now().isoformat(sep=" "),
        }
        return result

    def recent(self, limit: int) -> list[dict]:
        with self.lock:
            return list(self.events)[-limit:]

    def close(self) -> None:
        self.output.close()
        self.fixed_strike_output.close()
        self.paper_engine.close()
        self.long_paper_engine.close()
        _weeks43_notify(self, "close")
