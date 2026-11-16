"""Live BANKNIFTY fixed-strike IV/RV inference from the NSE UDP broadcast.

UDP multicast reception, sequential packet framing, and LZO decompression are
vendored in this package.  Market-message decoders and contract helpers remain
shared with ``E:\\Models\\Live Trades``.  This runtime is inference-only and
never places orders.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
import json
import logging
from logging.handlers import RotatingFileHandler
import math
from pathlib import Path
import selectors
import shutil
import sys
import threading
import time as clock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, TextIO
from urllib.parse import parse_qs, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yaml

LIVE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = LIVE_DIR.parent if LIVE_DIR.name.lower() == "live" else LIVE_DIR
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from inference_pipeline import FixedStrikeInferencePipeline

try:
    from .contract_bootstrap import ensure_latest_contract_file
    from .lzo_codec import Lzo1zDecoder, LzoError
    from .udp_receiver import NseFoUdpReceiver, PacketError
except ImportError:  # Direct script execution.
    from contract_bootstrap import ensure_latest_contract_file
    from lzo_codec import Lzo1zDecoder, LzoError
    from udp_receiver import NseFoUdpReceiver, PacketError


BASE_DIR = PROJECT_DIR
NSE_EPOCH = datetime(1980, 1, 1)
LAGS = (5, 15, 30, 60)
REQUIRED_FEATURE_MINUTES = max(LAGS) + 1
ANNUAL_MINUTES = 252.0 * 375.0
EPSILON = 1e-12
LOGGER = logging.getLogger("fixed_strike_live")


def _weeks43_module():
    """Optional isolated OTM extension. Its models and logs live outside this app."""
    import importlib
    folder = PROJECT_DIR.parents[1] / "43Weeksmodels"
    if not (folder / "weeks43_bridge.py").is_file():
        return None
    if str(folder) not in sys.path:
        sys.path.append(str(folder))
    return importlib.import_module("weeks43_bridge")


def _weeks43_notify(engine, method, *args):
    bridge = getattr(engine, "weeks43", None)
    if bridge is not None:
        try:
            getattr(bridge, method)(*args)
        except Exception:
            LOGGER.exception("Optional 43Weeks extension failed during %s", method)


MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
# Optional PCR extension: independent worker and namespaced dashboard routes.
# Existing models, routing, paper positions, configuration and logs are untouched.
PCR_BRIDGE = PCR_WEB = None
try:
    _pcr_parent = PROJECT_DIR.parents[1]
    if (_pcr_parent / "PCR" / "bridge.py").is_file():
        if str(_pcr_parent) not in sys.path:
            sys.path.append(str(_pcr_parent))
        from PCR import bridge as PCR_BRIDGE, web as PCR_WEB
except Exception:
    LOGGER.exception("Optional PCR extension unavailable; existing strategy continues")
DEFAULT_SESSION_START = time(9, 0)
DEFAULT_SESSION_END = time(15, 40)
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


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_live_config(path: Path) -> dict:
    local = _load_json(path.resolve())
    source_config = Path(local["source_config"]).resolve()
    source = _load_json(source_config)
    inference_config = Path(local["inference_config"]).resolve()
    inference_yaml = yaml.safe_load(inference_config.read_text(encoding="utf-8")) or {}
    udp = inference_yaml.get("live_udp", {})
    if udp:
        required = {
            "fo_multicast_ip", "fo_port", "cm_multicast_ip", "cm_port",
            "interface_ip", "cm_interface_ip",
        }
        missing = required.difference(udp)
        if missing:
            raise ValueError(f"config.yaml live_udp missing keys: {', '.join(sorted(missing))}")
        source.update({
            "multicast_ip": str(udp["fo_multicast_ip"]),
            "port": int(udp["fo_port"]),
            "cm_multicast_ip": str(udp["cm_multicast_ip"]),
            "cm_port": int(udp["cm_port"]),
            "interface_ip": str(udp["interface_ip"]),
            "cm_interface_ip": str(udp["cm_interface_ip"]),
        })
    signal = inference_yaml.get("rv_iv_sharp_decline_signal", {})
    if signal:
        source.update({
            "fixed_strike_path_horizon_minutes": int(signal["horizon_minutes"]),
            "paper_profit_target_inr": float(signal["profit_target_inr"]),
            "paper_stop_loss_inr": float(signal["stop_loss_inr"]),
        })
    source.update(local)
    source["source_config"] = str(source_config)
    return source


def parse_schedule_time(value: str) -> time:
    """Parse a 24-hour HH:MM schedule value."""
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as error:
        raise ValueError(f"invalid schedule time {value!r}; expected HH:MM") from error


def next_session_window(
    now: datetime,
    start: time = DEFAULT_SESSION_START,
    end: time = DEFAULT_SESSION_END,
) -> tuple[datetime, datetime]:
    """Return the current or next weekday session in the supplied timezone."""
    if now.tzinfo is None:
        raise ValueError("scheduler requires a timezone-aware datetime")
    if start >= end:
        raise ValueError("session start must be earlier than session end")

    candidate = now.date()
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    session_start = datetime.combine(candidate, start, tzinfo=now.tzinfo)
    session_end = datetime.combine(candidate, end, tzinfo=now.tzinfo)
    if now >= session_end:
        candidate += timedelta(days=1)
        while candidate.weekday() >= 5:
            candidate += timedelta(days=1)
        session_start = datetime.combine(candidate, start, tzinfo=now.tzinfo)
        session_end = datetime.combine(candidate, end, tzinfo=now.tzinfo)
    return session_start, session_end


def archive_previous_day_logs(config: dict, session_date: date) -> list[Path]:
    """Move stale active log files into prevDayLogs/<file-date>/.

    Files already modified on the session date are retained so restarting the
    runtime intraday appends to that day's logs instead of archiving them.
    """
    configured = {
        (BASE_DIR / str(config.get("runtime_log", "logs/live_runtime.log"))).resolve(),
        (BASE_DIR / str(config.get("prediction_log", "logs/live_predictions.jsonl"))).resolve(),
        (BASE_DIR / str(config.get("fixed_strike_path_log", "logs/fixed_strike_120m_paths.jsonl"))).resolve(),
        (BASE_DIR / str(config.get("paper_trade_events_log", "logs/paper_trade_events.jsonl"))).resolve(),
        (BASE_DIR / str(config.get("paper_position_state", "logs/paper_position_state.json"))).resolve(),
        (BASE_DIR / str(config.get("latest_prediction", "logs/latest_prediction.json"))).resolve(),
    }
    archived: list[Path] = []
    archive_root = BASE_DIR / "logs" / "prevDayLogs"
    log_root = BASE_DIR / "logs"
    if log_root.is_dir():
        configured.update(path.resolve() for path in log_root.iterdir() if path.is_file())
    for source in sorted(configured):
        if not source.is_file():
            continue
        modified = datetime.fromtimestamp(source.stat().st_mtime, MARKET_TIMEZONE).date()
        if modified >= session_date:
            continue
        destination_dir = archive_root / modified.isoformat()
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / source.name
        if destination.exists():
            destination = destination_dir / f"{source.stem}_{int(source.stat().st_mtime)}{source.suffix}"
        shutil.move(str(source), str(destination))
        archived.append(destination)
    return archived


def filter_historical_date(frame: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    """Select one trading date from a saved feature frame."""
    if "datetime" in frame.columns:
        values = pd.to_datetime(frame["datetime"], errors="raise").dt.date
    elif "Date" in frame.columns:
        values = pd.to_datetime(frame["Date"], errors="raise").dt.date
    else:
        raise ValueError("--historical-date requires a 'datetime' or 'Date' column")
    selected = frame.loc[values == trading_date].reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"no historical feature rows found for {trading_date.isoformat()}")
    return selected


def run_historical_inference(
    config: dict,
    combined_path: Path,
    *,
    leg_path: Path | None = None,
    rv_path: Path | None = None,
    output_path: Path | None = None,
    trading_date: date | None = None,
) -> Path:
    """Run the saved inference stack against feature CSVs without live I/O."""
    combined = pd.read_csv(combined_path.resolve(), low_memory=False)
    legs = pd.read_csv(leg_path.resolve(), low_memory=False) if leg_path else None
    rv = pd.read_csv(rv_path.resolve(), low_memory=False) if rv_path else None
    if trading_date is not None:
        combined = filter_historical_date(combined, trading_date)
        if legs is not None:
            legs = filter_historical_date(legs, trading_date)
        if rv is not None:
            rv = filter_historical_date(rv, trading_date)

    pipeline = FixedStrikeInferencePipeline(Path(config["inference_config"]).resolve())
    predictions = pipeline.predict(combined, legs, rv)
    if output_path is None:
        now = datetime.now(MARKET_TIMEZONE)
        result_date = trading_date or now.date()
        output_path = (
            BASE_DIR / "historical_predictions" / result_date.isoformat()
            / f"predictions_{now:%H%M%S}.csv"
        )
    destination = output_path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(destination, index=False)
    print(f"Wrote {len(predictions):,} historical inference rows to {destination}", flush=True)
    return destination


def historical_replay_config(
    config: dict,
    fo_port: int,
    *,
    cm_port: int | None = None,
    started_at: datetime | None = None,
) -> dict:
    """Build an isolated live configuration for an after-hours UDP replay."""
    ports = [fo_port] + ([cm_port] if cm_port is not None else [])
    if any(port is None or not 1 <= port <= 65535 for port in ports):
        raise ValueError("historical replay ports must be between 1 and 65535")
    stamp = started_at or datetime.now(MARKET_TIMEZONE)
    replay = dict(config)
    replay["port"] = int(fo_port)
    if cm_port is not None:
        replay["cm_port"] = int(cm_port)
    directory = f"logs/historicalReplay/{stamp:%Y-%m-%d}"
    suffix = f"{stamp:%H%M%S}"
    replay["runtime_log"] = f"{directory}/live_runtime_{suffix}.log"
    replay["prediction_log"] = f"{directory}/live_predictions_{suffix}.jsonl"
    replay["fixed_strike_path_log"] = f"{directory}/fixed_strike_120m_paths_{suffix}.jsonl"
    replay["paper_trade_events_log"] = f"{directory}/paper_trade_events_{suffix}.jsonl"
    replay["paper_position_state"] = f"{directory}/paper_position_state_{suffix}.json"
    replay["latest_prediction"] = f"{directory}/latest_prediction_{suffix}.json"
    replay["runtime_mode"] = "historical_udp_replay"
    return replay


def required_live_input_features(artifacts: dict[str, dict[str, Any]]) -> list[str]:
    """Return model features that must exist before the stacked pipeline runs."""
    required: set[str] = set()
    for payload in artifacts.values():
        required.update(payload.get("features", ()))
    return sorted(required - PIPELINE_GENERATED_FEATURES)


def install_source_imports(config: dict) -> dict[str, Any]:
    """Load shared decoders while keeping UDP/LZO reception package-local."""
    source_root = Path(config["source_root"]).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(source_root)
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    from banknifty_120m_live_inference import load_monthly_contracts  # type: ignore
    from contract_repository import load_front_future  # type: ignore
    from decoder_7208 import DecodeError, decode_7208  # type: ignore
    from iv_inference import _greeks, implied_volatility  # type: ignore
    from iv_prediction.oi_decoder import (  # type: ignore
        OiDecodeError,
        SUPPORTED_OI_CODES,
        decode_oi_message,
    )
    from spot_inference import decode_banknifty_7207  # type: ignore

    return {
        "load_monthly_contracts": load_monthly_contracts,
        "load_front_future": load_front_future,
        "DecodeError": DecodeError,
        "decode_7208": decode_7208,
        "_greeks": _greeks,
        "implied_volatility": implied_volatility,
        "OiDecodeError": OiDecodeError,
        "SUPPORTED_OI_CODES": SUPPORTED_OI_CODES,
        "decode_oi_message": decode_oi_message,
        "decode_banknifty_7207": decode_banknifty_7207,
        "Lzo1zDecoder": Lzo1zDecoder,
        "LzoError": LzoError,
        "NseFoUdpReceiver": NseFoUdpReceiver,
        "PacketError": PacketError,
    }


@dataclass(slots=True)
class MinuteBar:
    open: float
    high: float
    low: float
    close: float

    def update(self, value: float) -> None:
        self.high = max(self.high, value)
        self.low = min(self.low, value)
        self.close = value


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat(sep=" ") if isinstance(value, (pd.Timestamp, datetime)) else value.isoformat()
    return value


def _quadratic_surface(x: np.ndarray, y: np.ndarray, minimum_points: int) -> tuple[float, float, float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y) & (y > 0)
    count = int(valid.sum())
    if count < minimum_points:
        return (math.nan, math.nan, math.nan, math.nan, count)
    scale = max(float(np.max(np.abs(x[valid]))), 1e-8)
    z = x[valid] / scale
    design = np.column_stack((np.ones(count), z, z * z))
    coefficients = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    fitted = design @ coefficients
    rmse = float(np.sqrt(np.mean((y[valid] - fitted) ** 2)))
    return (
        float(coefficients[0]),
        float(coefficients[1] / scale),
        float(2.0 * coefficients[2] / (scale * scale)),
        rmse,
        count,
    )


def current_costs(entry_premium: float, exit_premium: float, quantity: int, config: dict) -> dict[str, float]:
    """Current exchange/statutory costs with brokerage forced to zero."""
    sell_turnover = max(entry_premium, 0.0) * quantity
    buy_turnover = max(exit_premium, 0.0) * quantity
    total_turnover = sell_turnover + buy_turnover
    brokerage = 0.0
    stt = sell_turnover * float(config["stt_sell_premium_rate"])
    exchange = total_turnover * float(config["nse_options_transaction_rate"])
    sebi = total_turnover * float(config["sebi_turnover_rate"])
    stamp = buy_turnover * float(config["stamp_buy_premium_rate"])
    gst = float(config["gst_rate"]) * (exchange + sebi)
    total = brokerage + stt + exchange + sebi + stamp + gst
    return {
        "brokerage": brokerage,
        "stt": stt,
        "nse_transaction_charges": exchange,
        "sebi_charges": sebi,
        "stamp_duty": stamp,
        "gst": gst,
        "estimated_total": total,
        "assumed_exit_premium": exit_premium,
        "quantity": quantity,
    }


def convergence_entry_eligible(prediction: dict[str, Any], threshold: float) -> bool:
    """Return whether the classifier probability strictly clears the paper-entry gate."""
    try:
        probability = float(prediction["pred_convergence_probability_classifier"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(probability) and probability > float(threshold)


def paper_entry_route(
    prediction: dict[str, Any],
    convergence_threshold: float,
    near_gap_decimal: float,
    rv_decline_score_threshold: float,
) -> dict[str, Any]:
    """Route a paper entry to the model assigned to the current IV/RV regime."""
    try:
        gap = float(prediction["pred_no120_gate_iv_minus_rv"])
    except (KeyError, TypeError, ValueError):
        gap = math.nan
    try:
        event_score = float(prediction["pred_no120_event_score"])
    except (KeyError, TypeError, ValueError):
        event_score = math.nan
    try:
        probability = float(prediction["pred_convergence_probability_classifier"])
    except (KeyError, TypeError, ValueError):
        probability = math.nan

    result = {
        "eligible": False,
        "route": None,
        "reason": "missing_or_non_finite_iv_rv_gap",
        "iv_minus_rv": _json_value(gap),
        "near_gap_decimal": float(near_gap_decimal),
        "convergence_probability": _json_value(probability),
        "convergence_probability_threshold": float(convergence_threshold),
        "rv_decline_event_score": _json_value(event_score),
        "rv_decline_event_score_threshold": float(rv_decline_score_threshold),
    }
    if not math.isfinite(gap):
        return result

    if gap > float(near_gap_decimal):
        result["route"] = "iv_above_rv_convergence"
        convergence_rule = bool(
            prediction.get("pred_strategy_rule_iv_above_rv_iv_declines_to_rv", False)
        )
        probability_ok = math.isfinite(probability) and probability > float(convergence_threshold)
        result["eligible"] = convergence_rule and probability_ok
        result["reason"] = (
            "iv_above_rv_convergence_passed"
            if result["eligible"]
            else "iv_above_rv_convergence_rule_or_probability_failed"
        )
        return result

    score_ok = math.isfinite(event_score) and event_score >= float(rv_decline_score_threshold)
    if gap < -float(near_gap_decimal):
        result["route"] = "iv_below_rv_sharp_rv_decline"
        result["reason"] = (
            "iv_below_rv_sharp_rv_decline_passed"
            if score_ok else "iv_below_rv_event_score_failed"
        )
    else:
        result["route"] = "iv_near_rv_sharp_rv_decline"
        result["reason"] = (
            "iv_near_rv_sharp_rv_decline_passed"
            if score_ok else "iv_near_rv_event_score_failed"
        )
    result["eligible"] = score_ok
    return result


class PaperTradingEngine:
    """Overlapping one-lot, fixed-strike paper positions driven by live ticks."""

    def __init__(self, config: dict, lot_size: int) -> None:
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
        self.iv_rv_near_gap_decimal = float(
            config.get("paper_entry_iv_rv_near_gap_decimal", 0.005)
        )
        self.rv_decline_event_score_threshold = float(
            config.get("paper_entry_rv_decline_event_score_threshold", 0.0)
        )
        self.square_off_time = datetime.strptime(
            str(config.get("paper_square_off_time", "15:14:30")), "%H:%M:%S"
        ).time()
        events_path = BASE_DIR / str(
            config.get("paper_trade_events_log", "logs/paper_trade_events.jsonl")
        )
        state_path = BASE_DIR / str(
            config.get("paper_position_state", "logs/paper_position_state.json")
        )
        events_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path = events_path.resolve()
        self.state_path = state_path.resolve()
        self.output: TextIO = self.events_path.open("a", encoding="utf-8", buffering=1)
        self.positions: dict[str, dict[str, Any]] = {}
        self.last_event: dict[str, Any] | None = None
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
            except (OSError, ValueError, json.JSONDecodeError):
                LOGGER.warning("Could not restore paper position state from %s", self.state_path)

    def _persist(self) -> None:
        state = {
            "mode": "paper",
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
        }
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8"
        )
        temporary.replace(self.state_path)

    def _record(self, event: dict[str, Any]) -> None:
        self.last_event = event
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
            "side": "SHORT_STRADDLE",
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
        return True

    def mark(
        self, position_id: str, exchange_second: int, call_price: float, put_price: float
    ) -> dict[str, Any] | None:
        position = self.positions.get(position_id)
        if position is None:
            return None
        stamp = NSE_EPOCH + timedelta(seconds=int(exchange_second))
        if int(exchange_second) < int(position["entry_exchange_second"]):
            return None
        exit_premium = float(call_price) + float(put_price)
        pnl = (float(position["entry_straddle_price"]) - exit_premium) * self.lot_size
        elapsed_seconds = int(exchange_second) - int(position["entry_exchange_second"])
        position.update({
            "last_mark_datetime": stamp.isoformat(sep=" "),
            "last_call_price": float(call_price),
            "last_put_price": float(put_price),
            "last_straddle_price": exit_premium,
            "unrealized_pnl_inr": pnl,
            "holding_seconds": elapsed_seconds,
        })

        reason: str | None = None
        if stamp.time() >= self.square_off_time:
            reason = "session_square_off_151430"
        elif pnl >= self.target:
            reason = "profit_target"
        elif pnl <= self.stop:
            reason = "stop_loss"
        elif elapsed_seconds >= self.horizon_minutes * 60:
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
        self.positions.pop(position_id, None)
        self._record(event)
        return event

    def snapshot(self) -> dict[str, Any]:
        return {
            "mode": "PAPER",
            "lots_per_position": 1,
            "overlapping_positions_allowed": self.max_open_positions > 1,
            "maximum_open_positions": self.max_open_positions,
            "available_position_slots": max(self.max_open_positions - len(self.positions), 0),
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


class LiveFeatureInference:
    """Minute aggregator, training-compatible feature builder, and scorer."""

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
        self.config = config
        self.contracts = contracts
        self.lot_size = lot_size
        self.expiry = expiry
        self.future_token = future_token
        self.pipeline = pipeline
        self.decode_spot = source["decode_banknifty_7207"]
        self.implied_volatility = source["implied_volatility"]
        self.greeks = source["_greeks"]
        self.decode_oi = source["decode_oi_message"]
        self.oi_error = source["OiDecodeError"]
        self.price_divisor = float(config.get("price_divisor", 100.0))
        self.rate = float(config.get("risk_free_rate", 0.065))
        self.minimum_surface_points = int(config.get("minimum_surface_points", 4))

        self.pairs: dict[float, dict[str, int]] = {}
        for contract in contracts.values():
            self.pairs.setdefault(float(contract.strike_price), {})[contract.option_type] = contract.token
        self.pairs = {strike: pair for strike, pair in self.pairs.items() if {"CE", "PE"} <= pair.keys()}
        self.strikes = sorted(self.pairs)
        if len(self.strikes) < 5:
            raise RuntimeError("monthly option chain contains fewer than five paired strikes")

        self.spot_bars: dict[int, MinuteBar] = {}
        self.future_bars: dict[int, MinuteBar] = {}
        self.option_bars: dict[int, dict[int, MinuteBar]] = {}
        self.latest_option_ticks: dict[int, tuple[float, int]] = {}
        self.oi_by_token: dict[int, tuple[int, int, int]] = {}
        self.latest_exchange_second: int | None = None
        self.last_paper_time_check_second: int | None = None
        self.latest_spot_minute: int | None = None
        self.latest_future_minute: int | None = None
        self.last_finalized_minute: int | None = None
        self.rows_by_minute: dict[int, dict[str, Any]] = {}
        self.spot_by_minute: dict[int, float] = {}
        self.events: deque[dict] = deque(maxlen=500)
        self.latest_event: dict = {
            "state": "WAITING",
            "reason": "waiting_for_spot_future_and_option_udp_packets",
        }
        self.lock = threading.RLock()

        prediction_path = BASE_DIR / str(config.get("prediction_log", "logs/live_predictions.jsonl"))
        fixed_strike_path = BASE_DIR / str(
            config.get("fixed_strike_path_log", "logs/fixed_strike_120m_paths.jsonl")
        )
        latest_path = BASE_DIR / str(config.get("latest_prediction", "logs/latest_prediction.json"))
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

    @staticmethod
    def _add_tick(bars: dict[int, MinuteBar], minute: int, value: float) -> None:
        bar = bars.get(minute)
        if bar is None:
            bars[minute] = MinuteBar(value, value, value, value)
        else:
            bar.update(value)

    def on_index_message(self, message: bytes) -> None:
        decoded = self.decode_spot(message, self.price_divisor)
        if decoded is None:
            return
        log_time, spot = decoded
        self.latest_exchange_second = (
            log_time
            if self.latest_exchange_second is None
            else max(self.latest_exchange_second, log_time)
        )
        minute = log_time // 60
        self._reset_if_rewound(minute)
        self._add_tick(self.spot_bars, minute, spot)
        self.latest_spot_minute = minute if self.latest_spot_minute is None else max(self.latest_spot_minute, minute)
        self._mark_paper_position(log_time, time_only=True)
        self._finalize_ready_minutes()

    def on_market_update(self, update: Any) -> None:
        if update.last_traded_price <= 0:
            return
        minute = update.log_time // 60
        self.latest_exchange_second = (
            update.log_time
            if self.latest_exchange_second is None
            else max(self.latest_exchange_second, update.log_time)
        )
        self._reset_if_rewound(minute)
        if update.token == self.future_token:
            self._add_tick(self.future_bars, minute, update.last_traded_price)
            self.latest_future_minute = minute if self.latest_future_minute is None else max(self.latest_future_minute, minute)
        elif update.token in self.contracts:
            self._add_tick(self.option_bars.setdefault(update.token, {}), minute, update.last_traded_price)
            self.latest_option_ticks[update.token] = (
                float(update.last_traded_price), int(update.log_time)
            )
            _weeks43_notify(self, "tick", update.token, update.last_traded_price, update.log_time)
        if update.token in self.contracts:
            self._mark_paper_position(update.log_time, updated_token=update.token)
        else:
            self._mark_paper_position(update.log_time, time_only=True)
        self._finalize_ready_minutes()

    def on_oi_message(self, message: bytes) -> None:
        try:
            updates = self.decode_oi(message)
        except self.oi_error:
            return
        tracked = set(self.contracts) | {self.future_token}
        priority = {7201: 1, 7202: 2, 17201: 3, 17202: 4}
        for update in updates:
            if update.token not in tracked:
                continue
            old = self.oi_by_token.get(update.token)
            if old and (update.log_time < old[1] or (
                update.log_time == old[1] and priority.get(update.transaction_code, 0) < priority.get(old[2], 0)
            )):
                continue
            self.oi_by_token[update.token] = (update.open_interest, update.log_time, update.transaction_code)
            _weeks43_notify(self, "oi", update.token, update.open_interest, update.log_time)

    def _reset_if_rewound(self, minute: int) -> None:
        latest = max(
            (value for value in (self.latest_spot_minute, self.latest_future_minute) if value is not None),
            default=None,
        )
        tolerance = int(self.config.get("replay_rewind_tolerance_minutes", 5))
        if latest is None or minute >= latest - tolerance:
            return
        LOGGER.warning("feed replay rewound from %d to %d; clearing live state", latest, minute)
        self.spot_bars.clear()
        self.future_bars.clear()
        self.option_bars.clear()
        self.latest_option_ticks.clear()
        self.oi_by_token.clear()
        self.rows_by_minute.clear()
        self.spot_by_minute.clear()
        self.latest_spot_minute = self.latest_future_minute = self.last_finalized_minute = None
        self.latest_exchange_second = None
        self.last_paper_time_check_second = None

    def _oi_value(self, token: int, minute: int, maximum_age: int) -> float:
        state = self.oi_by_token.get(token)
        if state is None:
            return math.nan
        value, log_time, _ = state
        if log_time > 0 and abs(minute - log_time // 60) > maximum_age:
            return math.nan
        return float(value)

    def _finalize_ready_minutes(self) -> None:
        if self.latest_spot_minute is None or self.latest_future_minute is None:
            return
        completed_through = min(self.latest_spot_minute, self.latest_future_minute) - 2
        start = completed_through if self.last_finalized_minute is None else self.last_finalized_minute + 1
        for minute in range(start, completed_through + 1):
            self._finalize_minute(minute)
        self.last_finalized_minute = completed_through
        path_horizon = int(self.config.get("fixed_strike_path_horizon_minutes", 120))
        cutoff = completed_through - (max(max(LAGS), path_horizon) + 5)
        self.spot_bars = {m: b for m, b in self.spot_bars.items() if m >= cutoff}
        self.future_bars = {m: b for m, b in self.future_bars.items() if m >= cutoff}
        self.option_bars = {
            token: {m: b for m, b in bars.items() if m >= cutoff}
            for token, bars in self.option_bars.items()
        }
        self.rows_by_minute = {m: row for m, row in self.rows_by_minute.items() if m >= cutoff}
        self.spot_by_minute = {m: value for m, value in self.spot_by_minute.items() if m >= cutoff}

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
        self._publish(minute, wide, prediction, paper_entry_decision)

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
        if len(self.paper_engine.positions) >= self.paper_engine.max_open_positions:
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "maximum_open_positions_reached"})
            return decision
        strike = float(row["entry_strike"])
        pair = self.pairs.get(strike)
        exchange_second = self.latest_exchange_second
        if pair is None or exchange_second is None:
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "missing_contract_or_exchange_time"})
            return decision
        prices = self._fresh_paper_prices(pair["CE"], pair["PE"], exchange_second)
        if prices is None:
            LOGGER.warning("PAPER_ENTRY_SKIPPED stale_or_missing_prices strike=%.0f", strike)
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "stale_or_missing_prices"})
            return decision
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
            decision.update({"state": "PAPER_ENTRY_TAKEN", "reason": "position_opened", "entered": True})
        else:
            decision.update({"state": "QUALIFIED_BUT_SKIPPED", "reason": "entry_rejected"})
        decision["open_positions_after"] = len(self.paper_engine.positions)
        return decision

    def _mark_paper_position(
        self,
        exchange_second: int,
        *,
        updated_token: int | None = None,
        time_only: bool = False,
    ) -> None:
        positions = list(self.paper_engine.positions.values())
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
                    position for position in positions
                    if int(exchange_second) - int(position["entry_exchange_second"])
                    >= self.paper_engine.horizon_minutes * 60
                ]
        elif updated_token is not None and not at_cutoff:
            positions = [
                position for position in positions
                if updated_token in (int(position["call_token"]), int(position["put_token"]))
            ]
        if not positions:
            return
        exits: list[dict[str, Any]] = []
        for position in positions:
            prices = self._fresh_paper_prices(
                int(position["call_token"]), int(position["put_token"]),
                int(exchange_second),
                enforce_age=stamp.time() < self.paper_engine.square_off_time,
            )
            if prices is None:
                continue
            exit_event = self.paper_engine.mark(
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

    def _build_current_row(
        self,
        minute: int,
        spot: float,
        future_open: float,
        selected: list[float],
        prices: dict[float, dict[str, float]],
    ) -> dict[str, Any] | None:
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        expiry_stamp = datetime.combine(self.expiry, time(15, 30))
        years = max((expiry_stamp - stamp).total_seconds(), 60.0) / (365.0 * 86400.0)
        dte = years * 365.0
        ivs: dict[float, dict[str, float]] = {}
        greeks: dict[float, dict[str, dict[str, float]]] = {}
        synthetic: dict[float, float] = {}
        for strike in selected:
            call = prices[strike]["CE"]
            put = prices[strike]["PE"]
            call_iv = self.implied_volatility(call, spot, strike, years, self.rate, True)
            put_iv = self.implied_volatility(put, spot, strike, years, self.rate, False)
            if not (math.isfinite(call_iv) and math.isfinite(put_iv) and call_iv > 0 and put_iv > 0):
                return None
            ivs[strike] = {"CE": call_iv, "PE": put_iv}
            greeks[strike] = {
                "CE": self.greeks(spot, strike, years, self.rate, call_iv, True),
                "PE": self.greeks(spot, strike, years, self.rate, put_iv, False),
            }
            synthetic[strike] = strike + call - put

        strike_array = np.asarray(selected, dtype=float)
        call_iv_array = np.asarray([ivs[s]["CE"] for s in selected])
        put_iv_array = np.asarray([ivs[s]["PE"] for s in selected])
        combined_array = (call_iv_array + put_iv_array) / 2.0
        forward_array = np.full(5, future_open, dtype=float)
        call_surface = _quadratic_surface(np.log(strike_array / forward_array), call_iv_array, self.minimum_surface_points)
        put_surface = _quadratic_surface(np.log(strike_array / forward_array), put_iv_array, self.minimum_surface_points)
        combined_surface = _quadratic_surface(
            np.log(strike_array / np.asarray([synthetic[s] for s in selected])),
            combined_array,
            self.minimum_surface_points,
        )

        atm = selected[2]
        pair = self.pairs[atm]
        call, put = prices[atm]["CE"], prices[atm]["PE"]
        call_iv, put_iv = ivs[atm]["CE"], ivs[atm]["PE"]
        call_g, put_g = greeks[atm]["CE"], greeks[atm]["PE"]
        call_oi = self._oi_value(pair["CE"], minute, int(self.config["option_oi_max_age_minutes"]))
        put_oi = self._oi_value(pair["PE"], minute, int(self.config["option_oi_max_age_minutes"]))
        future_oi = self._oi_value(self.future_token, minute, int(self.config["future_oi_max_age_minutes"]))
        total_vega = abs(call_g["vega"]) + abs(put_g["vega"])
        weighted_iv = (
            (call_iv * abs(call_g["vega"]) + put_iv * abs(put_g["vega"])) / total_vega
            if total_vega > 0 else math.nan
        )
        combined_iv = (call_iv + put_iv) / 2.0
        session_minute = stamp.hour * 60 + stamp.minute - (9 * 60 + 15)
        iv_difference = abs(call_iv - put_iv)
        quality_surface = combined_surface[3] <= float(self.config["maximum_surface_rmse"])
        quality_oi = call_oi >= float(self.config["minimum_open_interest"]) and put_oi >= float(self.config["minimum_open_interest"])
        quality_price = call >= float(self.config["minimum_option_price"]) and put >= float(self.config["minimum_option_price"])

        row: dict[str, Any] = {
            "datetime": stamp.isoformat(sep=" "), "Date": stamp.date().isoformat(),
            "dte_flag": "M1", "expiry": self.expiry.isoformat(),
            "entry_strike": atm, "days_to_expiry": dte,
            "future_open": future_open, "Future_oi": future_oi, "spot_price": spot,
            "call_price": call, "put_price": put, "call_oi": call_oi, "put_oi": put_oi,
            "synthetic_future": synthetic[atm], "atm_strike": atm,
            "entry_iv_call": call_iv, "entry_iv_put": put_iv,
            "entry_iv_weighted": weighted_iv, "entry_atm_iv_combined": combined_iv,
            "entry_atm_iv_vega_weighted": weighted_iv,
            "entry_call_put_iv_spread_signed": call_iv - put_iv,
            "entry_call_put_iv_difference": iv_difference,
            "entry_straddle_price": call + put,
            "entry_total_vega": total_vega,
            "entry_net_delta": call_g["delta"] + put_g["delta"],
            "entry_total_gamma": call_g["gamma"] + put_g["gamma"],
            "entry_total_theta": call_g["theta"] + put_g["theta"],
            "delta_call": call_g["delta"], "delta_put": put_g["delta"],
            "vega_call": call_g["vega"], "vega_put": put_g["vega"],
            "gamma_call": call_g["gamma"], "gamma_put": put_g["gamma"],
            "theta_call": call_g["theta"], "theta_put": put_g["theta"],
            "surface_atm_iv_call": call_surface[0], "surface_skew_call": call_surface[1],
            "surface_smile_call": call_surface[2], "surface_rmse_call": call_surface[3],
            "surface_points_call": call_surface[4],
            "surface_atm_iv_put": put_surface[0], "surface_skew_put": put_surface[1],
            "surface_smile_put": put_surface[2], "surface_rmse_put": put_surface[3],
            "surface_points_put": put_surface[4],
            "surface_atm_iv_combined": combined_surface[0], "surface_skew_combined": combined_surface[1],
            "surface_smile_combined": combined_surface[2], "surface_rmse_combined": combined_surface[3],
            "surface_points_combined": combined_surface[4],
            "atm_strike_exact": True, "iv_call_converged": True, "iv_put_converged": True,
            "day_of_week": stamp.weekday(), "month": stamp.month,
            "time_sin": math.sin(2.0 * math.pi * session_minute / 375.0),
            "time_cos": math.cos(2.0 * math.pi * session_minute / 375.0),
            "quality_entry_iv_converged": True, "quality_price": quality_price,
            "quality_open_interest": quality_oi, "quality_surface": quality_surface,
            "quality_call_put_agreement": iv_difference <= float(self.config["maximum_call_put_iv_difference"]),
            "quality_atm_strike": True,
        }
        row["quality_entry_ok"] = all(row[name] for name in (
            "quality_entry_iv_converged", "quality_price", "quality_open_interest",
            "quality_surface", "quality_call_put_agreement", "quality_atm_strike",
        ))

        row.update({
            "entry_future_price": future_open, "entry_future_oi": future_oi,
            "entry_spot_price": spot, "entry_call_price": call, "entry_put_price": put,
            "entry_call_oi": call_oi, "entry_put_oi": put_oi,
            "entry_iv_call_converged": True, "entry_iv_put_converged": True,
            "entry_delta_call": call_g["delta"], "entry_vega_call": call_g["vega"],
            "entry_gamma_call": call_g["gamma"], "entry_theta_call": call_g["theta"],
            "entry_delta_put": put_g["delta"], "entry_vega_put": put_g["vega"],
            "entry_gamma_put": put_g["gamma"], "entry_theta_put": put_g["theta"],
            "entry_atm_exact": True, "entry_strike_distance_from_atm": 0.0,
            "entry_log_moneyness": math.log(atm / future_open) if future_open > 0 else math.nan,
            "entry_time_sin": math.sin(2.0 * math.pi * session_minute / 375.0),
            "entry_time_cos": math.cos(2.0 * math.pi * session_minute / 375.0),
            "entry_day_of_week": stamp.weekday(),
        })
        return row

    def _add_lags_and_spot_features(self, minute: int, current: dict[str, Any]) -> dict[str, Any]:
        output = dict(current)
        combined_fields = (
            "entry_atm_iv_combined", "entry_atm_iv_vega_weighted",
            "entry_call_put_iv_spread_signed", "entry_call_put_iv_difference",
            "surface_atm_iv_combined", "surface_skew_combined", "surface_smile_combined",
            "surface_atm_iv_call", "surface_atm_iv_put",
            "surface_skew_call", "surface_skew_put",
            "surface_smile_call", "surface_smile_put",
            "entry_iv_weighted", "entry_straddle_price",
            "call_oi", "put_oi", "Future_oi", "future_open",
        )
        leg_fields = (
            "entry_spot_price", "entry_future_price", "entry_iv_call", "entry_iv_put",
            "surface_atm_iv_call", "surface_atm_iv_put", "surface_skew_call", "surface_skew_put",
            "surface_smile_call", "surface_smile_put", "entry_straddle_price",
            "entry_call_oi", "entry_put_oi",
        )
        for lag in LAGS:
            prior = self.rows_by_minute.get(minute - lag, {})
            for name in combined_fields:
                old = prior.get(name, math.nan)
                output[f"{name}_lag_{lag}m"] = old
                output[f"{name}_change_{lag}m"] = current[name] - old
            for name in leg_fields:
                old = prior.get(name, math.nan)
                output[f"lag_{lag}m_{name}"] = old
                output[f"change_{lag}m_{name}"] = current[name] - old

            prices = [self.spot_by_minute.get(m, math.nan) for m in range(minute - lag, minute + 1)]
            array = np.asarray(prices, dtype=float)
            exact = len(array) == lag + 1 and np.isfinite(array).all() and np.all(array > 0)
            if exact:
                returns = np.diff(np.log(array))
                output[f"spot_return_{lag}m"] = array[-1] / array[0] - 1.0
                output[f"rv_trailing_{lag}m"] = math.sqrt(float(np.sum(returns**2)) * ANNUAL_MINUTES / lag)
                output[f"rq_trailing_{lag}m"] = (
                    (lag / 3.0) * float(np.sum(returns**4)) * (ANNUAL_MINUTES / lag) ** 2
                )
            else:
                output[f"spot_return_{lag}m"] = math.nan
                output[f"rv_trailing_{lag}m"] = math.nan
                output[f"rq_trailing_{lag}m"] = math.nan
            output[f"abs_spot_return_{lag}m"] = abs(output[f"spot_return_{lag}m"])
            output[f"log_rv_trailing_{lag}m"] = math.log(max(output[f"rv_trailing_{lag}m"], 1e-8))
            output[f"log_rq_trailing_{lag}m"] = math.log(max(output[f"rq_trailing_{lag}m"], 1e-8))
        return output

    def _publish(
        self,
        minute: int,
        inputs: dict[str, Any],
        prediction: dict[str, Any],
        paper_entry_decision: dict[str, Any],
    ) -> None:
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        clean_predictions = {name: _json_value(value) for name, value in prediction.items()}
        prediction_fields = {name: value for name, value in clean_predictions.items() if name.startswith("pred_")}
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
            "prediction_count": len(prediction_fields),
            "predictions": prediction_fields,
        }
        encoded = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        self.output.write(encoded + "\n")
        temporary = self.latest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(event, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
        temporary.replace(self.latest_path)
        with self.lock:
            self.events.append(event)
            self.latest_event = event
        LOGGER.info("FIXED_STRIKE_PREDICTION minute=%s strike=%.0f fields=%d", stamp, inputs["entry_strike"], len(prediction_fields))
        if short_signal:
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
        result["runtime"] = {
            "paper_only": True,
            "mode": str(self.config.get("runtime_mode", "scheduled_live")),
            "source_root": str(Path(self.config["source_root"]).resolve()),
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
        _weeks43_notify(self, "close")


class LogDashboardView:
    """Read-only dashboard data source used while the scheduled live session is idle."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.prediction_path = (
            BASE_DIR / str(config.get("prediction_log", "logs/live_predictions.jsonl"))
        ).resolve()
        self.latest_path = (
            BASE_DIR / str(config.get("latest_prediction", "logs/latest_prediction.json"))
        ).resolve()
        self.paper_state_path = (
            BASE_DIR / str(config.get("paper_position_state", "logs/paper_position_state.json"))
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
            weeks43_api = _weeks43_module()
        except Exception:
            LOGGER.exception("Optional 43Weeks dashboard could not load")
            weeks43_api = None

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                try:
                    if PCR_WEB is not None and PCR_WEB.handle_get(self, parsed):
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
                if PCR_WEB is not None and PCR_WEB.handle_post(self, parsed):
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


def configure_logging(path: Path, level: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[file_handler, console],
        force=True,
    )


def wait_until(moment: datetime) -> None:
    """Sleep until a timezone-aware wall-clock moment."""
    while True:
        remaining = (moment - datetime.now(moment.tzinfo)).total_seconds()
        if remaining <= 0:
            return
        clock.sleep(min(remaining, 30.0))


def run_live_session(
    config: dict,
    *,
    start_at: datetime | None = None,
    stop_at: datetime | None = None,
    session_date: date | None = None,
    preflight: bool = False,
) -> None:
    """Prepare and run one live session, closing every resource on exit."""
    bootstrap_asof = (
        session_date
        or (
            date.fromisoformat(config["contract_asof_date"])
            if config.get("contract_asof_date")
            else datetime.now(MARKET_TIMEZONE).date()
        )
    )
    contract_state = ensure_latest_contract_file(config, asof=bootstrap_asof)
    LOGGER.info(
        "CONTRACT_READY file=%s file_date=%s downloaded=%s nearest_expiry=%s nearest_tokens=%d "
        "pipeline_monthly_expiry=%s pipeline_tokens=%d token_manifest=%s",
        contract_state.contract_file, contract_state.file_date, contract_state.downloaded,
        contract_state.nearest_expiry, contract_state.nearest_option_tokens,
        contract_state.pipeline_monthly_expiry, contract_state.pipeline_option_tokens,
        contract_state.manifest_file,
    )
    source = install_source_imports(config)

    contract_file = Path(config["contract_file"]).resolve()
    asof = (
        date.fromisoformat(config["contract_asof_date"])
        if config.get("contract_asof_date")
        else datetime.now(MARKET_TIMEZONE).date()
    )
    contracts, lot_size, expiry = source["load_monthly_contracts"](
        contract_file, asof, float(config.get("price_divisor", 100.0))
    )
    future_token, future_expiry = source["load_front_future"](config)
    pipeline = FixedStrikeInferencePipeline(Path(config["inference_config"]).resolve())
    engine = LiveFeatureInference(config, contracts, lot_size, expiry, future_token, pipeline, source)
    lzo_path = Path(config["lzo_dll"]).resolve()
    source["Lzo1zDecoder"](lzo_path)
    receiver_module = Path(sys.modules[source["NseFoUdpReceiver"].__module__].__file__).resolve()

    LOGGER.info(
        "PREFLIGHT_OK paper_only=true expiry=%s option_tokens=%d future_token=%d future_expiry=%s "
        "lot_size=%d model_artifacts=%d feature_contract=%d udp_receiver=%s lzo_dll=%s",
        expiry, len(contracts), future_token, future_expiry, lot_size,
        len(pipeline.artifacts), len(engine.required_feature_names),
        receiver_module, lzo_path,
    )
    if preflight:
        engine.close()
        return

    pcr = (
        PCR_BRIDGE.create_bridge(config, contracts, lot_size, expiry, future_token, source)
        if PCR_BRIDGE is not None else None
    )

    dashboard = DashboardServer(
        str(config["dashboard_host"]), int(config["dashboard_port"]),
        LIVE_DIR / "live_dashboard.html", engine,
    )
    fo_config = dict(config, receiver_name="NSE FO")
    cm_config = dict(
        config,
        receiver_name="NSE CM indices",
        multicast_ip=config["cm_multicast_ip"],
        port=int(config["cm_port"]),
        interface_ip=config.get("cm_interface_ip", config["interface_ip"]),
    )
    fo_receiver = source["NseFoUdpReceiver"](fo_config, source["Lzo1zDecoder"](lzo_path))
    cm_receiver = source["NseFoUdpReceiver"](cm_config, source["Lzo1zDecoder"](lzo_path))
    tracked = set(contracts) | {future_token}

    def handle(message: bytes, receiver: Any) -> None:
        if len(message) < 12:
            return
        code = int.from_bytes(message[10:12], "big", signed=True)
        if code == 7207:
            engine.on_index_message(message)
            if PCR_BRIDGE is not None:
                PCR_BRIDGE.safe_call(pcr, "index_message", message)
        elif receiver is not cm_receiver and code in source["SUPPORTED_OI_CODES"]:
            engine.on_oi_message(message)
            if PCR_BRIDGE is not None:
                PCR_BRIDGE.safe_call(pcr, "oi_message", message)
        elif receiver is not cm_receiver and code in (7208, 17208):
            try:
                for update in source["decode_7208"](message, float(config["price_divisor"]), tracked):
                    engine.on_market_update(update)
                    if PCR_BRIDGE is not None:
                        PCR_BRIDGE.safe_call(pcr, "market", update)
            except source["DecodeError"]:
                return

    selector: selectors.BaseSelector | None = None
    dashboard_started = False
    try:
        if start_at is not None:
            wait_until(start_at)
        if stop_at is not None and datetime.now(stop_at.tzinfo) >= stop_at:
            LOGGER.warning("SESSION_SKIPPED stop time already passed: %s", stop_at.isoformat())
            return

        try:
            extension = _weeks43_module()
            engine.weeks43 = extension.attach(engine) if extension is not None else None
        except Exception:
            LOGGER.exception("43Weeks startup failed; existing paper trader continues")
            engine.weeks43 = None

        if PCR_BRIDGE is not None:
            PCR_BRIDGE.safe_call(pcr, "start")
        dashboard.start()
        dashboard_started = True
        LOGGER.info("DASHBOARD_READY http://%s:%s", config["dashboard_host"], config["dashboard_port"])
        fo_receiver.open()
        cm_receiver.open()
        selector = selectors.DefaultSelector()
        selector.register(fo_receiver.socket, selectors.EVENT_READ, fo_receiver)
        selector.register(cm_receiver.socket, selectors.EVENT_READ, cm_receiver)
        LOGGER.info(
            "SESSION_STARTED start=%s scheduled_stop=%s timezone=%s",
            datetime.now(MARKET_TIMEZONE).isoformat(),
            stop_at.isoformat() if stop_at else "manual",
            MARKET_TIMEZONE.key,
        )
        while stop_at is None or datetime.now(stop_at.tzinfo) < stop_at:
            timeout = 1.0
            if stop_at is not None:
                timeout = max(0.0, min(timeout, (stop_at - datetime.now(stop_at.tzinfo)).total_seconds()))
            for key, _ in selector.select(timeout=timeout):
                receiver = key.data
                datagram, _sender = receiver.socket.recvfrom(int(receiver.config["max_datagram_bytes"]))
                try:
                    for message in receiver.unpack_datagram(datagram):
                        handle(message, receiver)
                except (source["PacketError"], source["LzoError"]) as error:
                    LOGGER.debug("discarded malformed packet: %s", error)
            if PCR_BRIDGE is not None:
                PCR_BRIDGE.safe_call(pcr, "pulse")
    finally:
        LOGGER.info("SESSION_STOPPED time=%s", datetime.now(MARKET_TIMEZONE).isoformat())
        if selector is not None:
            selector.close()
        fo_receiver.close()
        cm_receiver.close()
        if PCR_BRIDGE is not None:
            PCR_BRIDGE.safe_call(pcr, "close")
        if dashboard_started:
            dashboard.close()
        else:
            dashboard.server.server_close()
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Live UDP inference for the fixed-strike IV/RV stack")
    parser.add_argument("--config", type=Path, default=LIVE_DIR / "live_config.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--preflight", action="store_true", help="validate imports, contracts, and models without opening sockets")
    mode.add_argument("--run-now", action="store_true", help="bypass the daily scheduler and run until interrupted")
    mode.add_argument(
        "--historical-data", type=Path, metavar="CSV",
        help="run after-hours inference on a combined/wide historical feature CSV",
    )
    mode.add_argument(
        "--historical-replay-port", type=int, metavar="PORT",
        help="run after-hours from the historical UDP feed on this FO port",
    )
    parser.add_argument(
        "--historical-replay-cm-port", type=int, metavar="PORT",
        help="optional historical UDP port for the CM/index feed",
    )
    parser.add_argument("--historical-leg-data", type=Path, metavar="CSV", help="optional call/put feature CSV")
    parser.add_argument("--historical-rv-data", type=Path, metavar="CSV", help="optional RV feature CSV")
    parser.add_argument(
        "--historical-date", metavar="YYYY-MM-DD",
        help="CSV trading-date filter or UDP replay contract as-of date",
    )
    parser.add_argument("--historical-output", type=Path, metavar="CSV", help="historical prediction destination")
    args = parser.parse_args()

    if args.historical_data:
        try:
            trading_date = date.fromisoformat(args.historical_date) if args.historical_date else None
        except ValueError:
            parser.error("--historical-date must use YYYY-MM-DD")
        historical_local = _load_json(args.config.resolve())
        inference_config = Path(historical_local["inference_config"])
        if not inference_config.is_absolute():
            inference_config = args.config.resolve().parent / inference_config
        run_historical_inference(
            {"inference_config": str(inference_config.resolve())},
            args.historical_data,
            leg_path=args.historical_leg_data,
            rv_path=args.historical_rv_data,
            output_path=args.historical_output,
            trading_date=trading_date,
        )
        return 0
    if args.historical_replay_port is not None:
        if args.historical_leg_data or args.historical_rv_data or args.historical_output:
            parser.error("CSV historical options cannot be combined with UDP replay mode")
        config = load_live_config(args.config)
        try:
            replay_config = historical_replay_config(
                config,
                args.historical_replay_port,
                cm_port=args.historical_replay_cm_port,
            )
            if args.historical_date:
                replay_config["contract_asof_date"] = date.fromisoformat(args.historical_date).isoformat()
        except ValueError as error:
            parser.error(str(error))
        configure_logging(
            BASE_DIR / replay_config["runtime_log"],
            str(replay_config.get("log_level", "INFO")),
        )
        LOGGER.info(
            "HISTORICAL_REPLAY_MODE fo=%s:%d cm=%s:%d replay_date=%s",
            replay_config["multicast_ip"], replay_config["port"],
            replay_config["cm_multicast_ip"], replay_config["cm_port"],
            replay_config.get("contract_asof_date", "today"),
        )
        try:
            run_live_session(replay_config)
        except KeyboardInterrupt:
            LOGGER.info("historical UDP replay stopped by user")
        return 0
    if args.historical_replay_cm_port is not None:
        parser.error("--historical-replay-cm-port requires --historical-replay-port")
    if args.historical_leg_data or args.historical_rv_data or args.historical_date or args.historical_output:
        parser.error("historical options require --historical-data")

    config = load_live_config(args.config)
    runtime_log = BASE_DIR / config["runtime_log"]
    if args.preflight or args.run_now:
        configure_logging(runtime_log, str(config.get("log_level", "INFO")))
        try:
            run_live_session(config, preflight=args.preflight)
        except KeyboardInterrupt:
            LOGGER.info("live inference stopped by user")
        return 0

    session_start = parse_schedule_time(str(config.get("session_start", "09:00")))
    session_end = parse_schedule_time(str(config.get("session_end", "15:40")))
    prepare_seconds = max(0, int(config.get("schedule_prepare_seconds", 300)))

    try:
        while True:
            now = datetime.now(MARKET_TIMEZONE)
            start_at, stop_at = next_session_window(now, session_start, session_end)
            preparation_at = start_at - timedelta(seconds=prepare_seconds)
            if now < preparation_at:
                print(
                    f"Scheduler waiting: next session {start_at:%Y-%m-%d %H:%M}–{stop_at:%H:%M} "
                    f"{MARKET_TIMEZONE.key}",
                    flush=True,
                )
                if bool(config.get("standby_dashboard_enabled", False)):
                    standby = DashboardServer(
                        str(config["dashboard_host"]),
                        int(config["dashboard_port"]),
                        LIVE_DIR / "live_dashboard.html",
                        LogDashboardView(config),
                    )
                    try:
                        standby.start()
                        print(
                            f"Standby dashboard ready: http://{config['dashboard_host']}:"
                            f"{config['dashboard_port']}",
                            flush=True,
                        )
                        wait_until(preparation_at)
                    finally:
                        standby.close()
                else:
                    wait_until(preparation_at)

            logging.shutdown()
            archived = archive_previous_day_logs(config, start_at.date())
            configure_logging(runtime_log, str(config.get("log_level", "INFO")))
            for path in archived:
                LOGGER.info("LOG_ARCHIVED destination=%s", path)
            LOGGER.info(
                "SESSION_SCHEDULED start=%s stop=%s preparation_seconds=%d",
                start_at.isoformat(), stop_at.isoformat(), prepare_seconds,
            )
            # ``ensure_latest_contract_file`` records its as-of date in config.
            # Override that cached value for every scheduled session so a
            # long-running scheduler cannot reuse yesterday's expired tokens.
            run_live_session(
                config,
                start_at=start_at,
                stop_at=stop_at,
                session_date=start_at.date(),
            )
            logging.shutdown()
    except KeyboardInterrupt:
        LOGGER.info("scheduler stopped by user")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
