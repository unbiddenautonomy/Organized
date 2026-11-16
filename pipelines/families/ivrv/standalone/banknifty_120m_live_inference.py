"""Live 120-minute BANKNIFTY expanded-hybrid inference and paper strategy.

This is intentionally PAPER ONLY.  It consumes the existing NSE multicast feed,
builds the RV features plus the 29 features used by the expanded hybrid, selects
the nearest monthly BANKNIFTY option expiry, and logs fixed-strike one-lot paper
trades.  It has no broker/order API.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
from logging.handlers import RotatingFileHandler
import math
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
import selectors
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TextIO
from urllib.parse import parse_qs, urlparse

import joblib
import numpy as np
import pandas as pd

from contract_repository import OptionContract
from decoder_7208 import DecodeError, MarketUpdate7208, decode_7208
from iv_inference import _greeks, implied_volatility
from iv_prediction import SUPPORTED_OI_CODES
from iv_prediction.oi_decoder import OiDecodeError, decode_oi_message
from lzo_codec import Lzo1zDecoder, LzoError
from order_book import BankNiftyOrderBook, OrderBookEntry
from portable_config import load_source_config
from spot_inference import NSE_EPOCH, decode_banknifty_7207
from udp_receiver import NseFoUdpReceiver, PacketError


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "rv120" / "rv_intraday_120m_model.joblib"
DEFAULT_HYBRID_MODEL = BASE_DIR / "models" / "rv120" / "expanded_hybrid_models.joblib"
SESSION_OPEN_MINUTE = 9 * 60 + 15
LATEST_ENTRY_MINUTE = 13 * 60 + 30
FORCE_EXIT_MINUTE = 15 * 60 + 29
HORIZON_MINUTES = 120
N_LOTS = 1
MAX_TRADES_PER_DAY = 10
EPSILON = 1e-12
LOGGER = logging.getLogger("banknifty_120m")


@dataclass(frozen=True, slots=True)
class Quote:
    bid: float
    ask: float
    mid: float
    spread_pct: float
    log_time: int


@dataclass(slots=True)
class PaperPosition:
    trade_id: int
    entry_time: str
    entry_minute: int
    expiry: str
    structure: str
    component: str
    put_strike: float
    call_strike: float
    call_token: int
    put_token: int
    lots: int
    lot_size: int
    call_entry_bid: float
    put_entry_bid: float
    entry_credit: float
    entry_spot: float
    entry_iv: float
    predicted_total_variance: float
    implied_total_variance: float
    expected_long_pnl_points: float
    estimated_one_lot_cost: float
    model_probability: float | None


def parse_nse_date(value: str) -> date:
    text = str(value).strip()
    try:
        return (NSE_EPOCH + timedelta(seconds=int(float(text)))).date()
    except ValueError:
        return date.fromisoformat(text)


def load_monthly_contracts(path: Path, asof: date, price_divisor: float) -> tuple[dict[int, OptionContract], int, date]:
    """Load the nearest non-expired monthly BANKNIFTY option chain."""
    if not path.exists():
        raise FileNotFoundError(path)
    opener = gzip.open if path.suffix.lower() == ".gz" else open
    rows: list[dict[str, str]] = []
    with opener(path, "rt", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"FinInstrmId", "UndrlygFinInstrmId", "FinInstrmNm", "TckrSymb", "XpryDt", "StrkPric", "OptnTp"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"contract file missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("DelFlg", "N").strip().upper() == "Y":
                continue
            if row.get("TckrSymb", "").strip().upper() != "BANKNIFTY":
                continue
            if row.get("FinInstrmNm", "").strip().upper() != "OPTIDX":
                continue
            if row.get("OptnTp", "").strip().upper() not in {"CE", "PE"}:
                continue
            expiry = parse_nse_date(row["XpryDt"])
            if expiry >= asof:
                row["_expiry"] = expiry  # type: ignore[assignment]
                rows.append(row)
    if not rows:
        raise RuntimeError(f"no non-expired BANKNIFTY options in {path}")
    monthly = {max(r["_expiry"] for r in rows if r["_expiry"].year == year and r["_expiry"].month == month)
               for year, month in {(r["_expiry"].year, r["_expiry"].month) for r in rows}}
    expiry = min(value for value in monthly if value >= asof)
    selected = [row for row in rows if row["_expiry"] == expiry]
    divisor = Decimal(str(price_divisor))
    contracts: dict[int, OptionContract] = {}
    lot_sizes: list[int] = []
    for row in selected:
        token = int(float(row["FinInstrmId"]))
        contracts[token] = OptionContract(
            token=token,
            underlying_token=int(float(row.get("UndrlygFinInstrmId", "0") or 0)),
            instrument_name="OPTIDX",
            symbol="BANKNIFTY",
            expiry=expiry,
            strike_price=Decimal(row.get("StrkPric", "0") or "0") / divisor,
            option_type=row["OptnTp"].strip().upper(),
        )
        raw_lot = row.get("NewBrdLotQty") or row.get("MinLot") or "0"
        try:
            lot = int(float(raw_lot))
            if lot > 0:
                lot_sizes.append(lot)
        except ValueError:
            pass
    sides = {contract.option_type for contract in contracts.values()}
    if sides != {"CE", "PE"}:
        raise RuntimeError(f"monthly expiry {expiry} is missing a CE or PE side")
    if not lot_sizes:
        raise RuntimeError("could not determine BANKNIFTY lot size from contract master")
    lot_size = max(set(lot_sizes), key=lot_sizes.count)
    return contracts, lot_size, expiry


def best_quote(entry: OrderBookEntry | None) -> Quote | None:
    if entry is None:
        return None
    bid = next((level.price for level in entry.bids if level.price > 0 and level.quantity > 0), None)
    ask = next((level.price for level in entry.asks if level.price > 0 and level.quantity > 0), None)
    if bid is None or ask is None or ask < bid:
        return None
    mid = (bid + ask) / 2.0
    return Quote(bid, ask, mid, (ask - bid) / max(mid, EPSILON), entry.log_time)


def charge_components(buy_points: float, sell_points: float, quantity: int, slippage_bps: float) -> dict[str, float]:
    """Current Indian option charges used in the prior Zerodha evaluation."""
    buy_turnover = buy_points * quantity
    sell_turnover = sell_points * quantity
    total_turnover = buy_turnover + sell_turnover
    brokerage = 80.0  # CE/PE entry and exit, combined lot quantity per leg
    stt = float(round(sell_turnover * 0.0015))
    exchange = total_turnover * 0.0003553
    sebi = total_turnover * 0.000001
    ipft = total_turnover * 0.000000001
    stamp = buy_turnover * 0.00003
    gst = 0.18 * (brokerage + exchange + sebi + ipft)
    slippage = total_turnover * slippage_bps / 10000.0
    total = brokerage + stt + exchange + sebi + ipft + stamp + gst + slippage
    return {
        "brokerage": brokerage, "stt": stt, "exchange": exchange, "sebi": sebi,
        "ipft": ipft, "stamp": stamp, "gst": gst, "slippage": slippage,
        "buy_turnover": buy_turnover, "sell_turnover": sell_turnover, "total_cost": total,
    }


def training_greeks(forward: float, strike: float, years: float, volatility: float,
                    spot: float, is_call: bool) -> dict[str, float]:
    """Match the Greek units used to train the expanded-hybrid classifiers."""
    if min(forward, strike, years, volatility, spot) <= 0 or not all(
        math.isfinite(value) for value in (forward, strike, years, volatility, spot)
    ):
        return {name: math.nan for name in ("delta", "gamma", "vega", "theta_day")}
    root_t = math.sqrt(years)
    d1 = (math.log(forward / strike) + 0.5 * volatility * volatility * years) / (volatility * root_t)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    normal = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    scale = forward / spot
    delta_forward = normal if is_call else normal - 1.0
    return {
        "delta": delta_forward * scale,
        "gamma": pdf / (forward * volatility * root_t) * scale * scale,
        # Training uses price points per 1.00 absolute volatility, not per 1 vol point.
        "vega": forward * pdf * root_t,
        "theta_day": (-forward * pdf * volatility / (2.0 * root_t)) / 365.0,
    }


class LiveRvStrategy:
    def __init__(self, config: dict, contracts: dict[int, OptionContract], lot_size: int,
                 expiry: date, model_path: Path, hybrid_model_path: Path, logs_dir: Path) -> None:
        self.config = config
        self.contracts = contracts
        self.lot_size = lot_size
        self.expiry = expiry
        self.rate = float(config.get("iv_risk_free_rate", 0.065))
        self.max_quote_age = int(config.get("rv120_max_quote_age_seconds", 120))
        self.max_spread_pct = float(config.get("rv120_max_leg_spread_pct", 0.05))
        self.hurdle_multiple = float(config.get("rv120_cost_hurdle_multiple", 2.0))
        self.selection_slippage_bps = float(config.get("rv120_selection_slippage_bps", 10.0))
        self.paper_only = True
        self.hybrid_enabled = bool(config.get("expanded_hybrid_enabled", True))

        self.pairs: dict[float, dict[str, int]] = {}
        for contract in contracts.values():
            self.pairs.setdefault(float(contract.strike_price), {})[contract.option_type] = contract.token
        self.pairs = {strike: pair for strike, pair in self.pairs.items() if {"CE", "PE"} <= pair.keys()}
        self.strikes = sorted(self.pairs)
        if len(self.strikes) < 5:
            raise RuntimeError("monthly chain has fewer than five paired strikes")

        artifact = joblib.load(model_path)
        if int(artifact.get("horizon_min", 0)) != HORIZON_MINUTES:
            raise ValueError(f"expected a 120-minute model, got {artifact.get('horizon_min')}")
        self.model = artifact["model"]
        self.features = list(artifact["features"])
        self.bias_correction = float(artifact["log_bias_correction"])
        self.clip_lower = pd.Series(artifact["clip_lower"], dtype=float)
        self.clip_upper = pd.Series(artifact["clip_upper"], dtype=float)
        self.model_name = str(artifact.get("model_name", type(self.model).__name__))

        hybrid_artifact = joblib.load(hybrid_model_path)
        self.primary_hybrid_model = hybrid_artifact["primary_model"]
        self.primary_threshold = float(hybrid_artifact["primary_threshold"])
        self.secondary_hybrid_model = hybrid_artifact["secondary_model"]
        self.secondary_threshold = float(hybrid_artifact["secondary_threshold"])
        self.hybrid_features = list(hybrid_artifact["features"])
        self.secondary_structure = str(hybrid_artifact["secondary_structure"])
        if int(hybrid_artifact["secondary_horizon_min"]) != HORIZON_MINUTES:
            raise ValueError("expanded hybrid secondary horizon is not 120 minutes")

        logs_dir.mkdir(parents=True, exist_ok=True)
        self.signal_path = logs_dir / "rv120_signals.jsonl"
        self.trade_path = logs_dir / "rv120_paper_trades.jsonl"
        self.status_path = logs_dir / "rv120_latest_status.json"
        self.signal_output: TextIO = self.signal_path.open("a", encoding="utf-8", buffering=1)
        self.trade_output: TextIO = self.trade_path.open("a", encoding="utf-8", buffering=1)
        self.order_book = BankNiftyOrderBook(contracts, logs_dir / "unused_orderbook.jsonl", False)
        self.oi_by_token: dict[int, tuple[int, int]] = {}

        self.current_spot_minute: int | None = None
        self.current_spot_close: float | None = None
        self.spot_history: deque[tuple[int, float]] = deque(maxlen=400)
        self.iv_history: deque[tuple[int, float]] = deque(maxlen=400)
        self.oi_history: deque[tuple[int, float]] = deque(maxlen=400)
        self.previous_session_close: float | None = None
        self.session_open: float | None = None
        self.last_signal_minute: int | None = None
        self.trading_date: date | None = None
        self.daily_trade_count = 0
        self.fallback_used_today = False
        self.trade_id = 0
        self.positions: list[PaperPosition] = []
        self.ewma_r2: float | None = None

    def on_oi_message(self, message: bytes) -> None:
        try:
            updates = decode_oi_message(message)
        except OiDecodeError:
            return
        for update in updates:
            if update.token in self.contracts:
                old = self.oi_by_token.get(update.token)
                if old is None or update.log_time >= old[1]:
                    self.oi_by_token[update.token] = (update.open_interest, update.log_time)

    def on_option_update(self, update: MarketUpdate7208) -> None:
        self.order_book.update(update)

    def on_index_message(self, message: bytes) -> None:
        decoded = decode_banknifty_7207(message, float(self.config.get("price_divisor", 100.0)))
        if decoded is None:
            return
        log_time, spot = decoded
        minute = log_time // 60
        if self.current_spot_minute is None:
            self.current_spot_minute, self.current_spot_close = minute, spot
            return
        if minute < self.current_spot_minute - 5:
            LOGGER.warning("feed replay rewound; clearing rolling state")
            self._reset_history()
            self.current_spot_minute, self.current_spot_close = minute, spot
            return
        if minute == self.current_spot_minute:
            self.current_spot_close = spot
            return
        completed_minute, completed_spot = self.current_spot_minute, self.current_spot_close
        self.current_spot_minute, self.current_spot_close = minute, spot
        assert completed_spot is not None
        self._on_completed_minute(completed_minute, completed_spot, log_time)

    def _reset_history(self) -> None:
        self.spot_history.clear()
        self.iv_history.clear()
        self.oi_history.clear()
        self.ewma_r2 = None
        self.last_signal_minute = None

    def _write_status(self, event: dict) -> None:
        temporary = self.status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(event, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(self.status_path)

    def _emit_signal(self, event: dict) -> None:
        event = {"event_type": "rv120_signal", "paper_only": True, **event}
        encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
        self.signal_output.write(encoded + "\n")
        self._write_status(event)
        LOGGER.info("RV120_SIGNAL %s", encoded)

    def _emit_trade(self, event: dict) -> None:
        encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
        self.trade_output.write(encoded + "\n")
        LOGGER.info("RV120_TRADE %s", encoded)

    def _option_snapshot(self, spot: float, minute: int, decision_log_time: int) -> dict | None:
        nearest_index = min(range(len(self.strikes)), key=lambda index: abs(self.strikes[index] - spot))
        if nearest_index < 2 or nearest_index + 2 >= len(self.strikes):
            return None
        selected = self.strikes[nearest_index - 2: nearest_index + 3]
        quotes: dict[float, dict[str, Quote]] = {}
        for strike in selected:
            pair = self.pairs[strike]
            call = best_quote(self.order_book.get(pair["CE"]))
            put = best_quote(self.order_book.get(pair["PE"]))
            if call is None or put is None:
                return None
            if decision_log_time - call.log_time > self.max_quote_age or decision_log_time - put.log_time > self.max_quote_age:
                return None
            if call.spread_pct > self.max_spread_pct or put.spread_pct > self.max_spread_pct:
                return None
            quotes[strike] = {"CE": call, "PE": put}

        atm = selected[2]
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        expiry_stamp = datetime.combine(self.expiry, time(15, 30))
        years = max((expiry_stamp - stamp).total_seconds(), 60.0) / (365.0 * 86400.0)
        discount = math.exp(-self.rate * years)
        atm_call, atm_put = quotes[atm]["CE"].mid, quotes[atm]["PE"].mid
        forward = atm + (atm_call - atm_put) / discount
        forward_spot = forward * discount

        ivs: dict[float, dict[str, float]] = {}
        for strike in selected:
            call_iv = implied_volatility(quotes[strike]["CE"].mid, forward_spot, strike, years, self.rate, True)
            put_iv = implied_volatility(quotes[strike]["PE"].mid, forward_spot, strike, years, self.rate, False)
            if not (math.isfinite(call_iv) and math.isfinite(put_iv)):
                return None
            ivs[strike] = {"CE": call_iv, "PE": put_iv}
        iv_atm = (ivs[atm]["CE"] + ivs[atm]["PE"]) / 2.0
        lower, upper = selected[0], selected[4]
        smile_skew = ivs[upper]["CE"] - ivs[lower]["PE"]
        smile_butterfly = ivs[upper]["CE"] + ivs[lower]["PE"] - 2.0 * iv_atm
        greeks: dict[float, dict[str, dict[str, float]]] = {}
        for strike in selected:
            greeks[strike] = {
                "CE": training_greeks(forward, strike, years, ivs[strike]["CE"], spot, True),
                "PE": training_greeks(forward, strike, years, ivs[strike]["PE"], spot, False),
            }
        call_greeks = greeks[atm]["CE"]
        put_greeks = greeks[atm]["PE"]
        pair = self.pairs[atm]
        oi_call = self._oi(pair["CE"], decision_log_time)
        oi_put = self._oi(pair["PE"], decision_log_time)
        oi_available = oi_call is not None and oi_put is not None and (oi_call + oi_put) > 0
        oi_ratio = oi_call / max(oi_put, 1) if oi_available else 1.0
        oi_imbalance = (oi_call - oi_put) / max(oi_call + oi_put, 1) if oi_available else 0.0
        return {
            "expiry": self.expiry.isoformat(), "strikes": selected, "strike": atm,
            "call_token": pair["CE"], "put_token": pair["PE"], "quotes": quotes,
            "forward": forward, "forward_spot": forward_spot, "years": years,
            "days_to_expiry": years * 365.0, "iv_atm": iv_atm,
            "iv_call": ivs[atm]["CE"], "iv_put": ivs[atm]["PE"],
            "iv_skew": ivs[atm]["CE"] - ivs[atm]["PE"],
            "ivs": ivs, "greeks": greeks,
            "smile_skew": smile_skew, "smile_butterfly": smile_butterfly,
            "theta_day": call_greeks["theta_day"] + put_greeks["theta_day"],
            "oi_ratio": float(oi_ratio), "oi_imbalance": float(oi_imbalance),
            "oi_total": float(oi_call + oi_put) if oi_available else None,
            "oi_available": oi_available,
        }

    def _oi(self, token: int, decision_log_time: int) -> float | None:
        state = self.oi_by_token.get(token)
        if state is None or decision_log_time - state[1] > 15 * 60:
            return None
        return float(state[0])

    def _build_features(self, minute: int, snapshot: dict) -> dict[str, float] | None:
        if len(self.spot_history) < 61 or len(self.iv_history) < 6:
            return None
        spot_minutes = [row[0] for row in self.spot_history]
        iv_minutes = [row[0] for row in self.iv_history]
        if spot_minutes[-1] != minute or iv_minutes[-1] != minute:
            return None
        prices = np.asarray([row[1] for row in self.spot_history], dtype=float)
        returns = np.diff(np.log(prices))
        if len(returns) < 60:
            return None
        current_r2 = float(returns[-1] ** 2)
        if self.ewma_r2 is None:
            ewma = float(pd.Series(returns ** 2).ewm(alpha=0.06, adjust=False).mean().iloc[-1])
        else:
            ewma = 0.06 * current_r2 + 0.94 * self.ewma_r2
        self.ewma_r2 = ewma
        rv5 = float(np.sum(returns[-5:] ** 2))
        rv15 = float(np.sum(returns[-15:] ** 2))
        rv60 = float(np.sum(returns[-60:] ** 2))
        rq15 = float((15.0 / 3.0) * np.sum(returns[-15:] ** 4))
        iv = float(snapshot["iv_atm"])
        iv_change_5 = iv - float(self.iv_history[-6][1])
        iv_var15 = iv * iv * 15.0 / (252.0 * 375.0)
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        time_fraction = ((stamp.hour * 60 + stamp.minute) - SESSION_OPEN_MINUTE) / 375.0
        return {
            "log_rv_past_5": math.log(max(rv5, EPSILON)),
            "log_rv_past_15": math.log(max(rv15, EPSILON)),
            "log_rv_past_60": math.log(max(rv60, EPSILON)),
            "log_current_r2": math.log(max(current_r2, EPSILON)),
            "log_ewma_r2": math.log(max(ewma, EPSILON)),
            "log_iv_trade": math.log(max(iv, EPSILON)),
            "log_rq_15": math.log(max(rq15, EPSILON)),
            "iv_change_5": iv_change_5,
            "smile_skew": float(snapshot["smile_skew"]),
            "smile_butterfly": float(snapshot["smile_butterfly"]),
            "rv_iv_gap_15": math.log(max(rv15, EPSILON)) - math.log(max(iv_var15, EPSILON)),
            "time_sin": math.sin(2.0 * math.pi * time_fraction),
            "time_cos": math.cos(2.0 * math.pi * time_fraction),
            "days_to_expiry": float(snapshot["days_to_expiry"]),
            "oi_ratio": float(snapshot["oi_ratio"]),
            "oi_imbalance": float(snapshot["oi_imbalance"]),
        }

    @staticmethod
    def _relative_change(history: deque[tuple[int, float]], periods: int) -> float | None:
        if len(history) <= periods:
            return None
        current = float(history[-1][1])
        previous = float(history[-1 - periods][1])
        return (current - previous) / max(previous, 1.0)

    def _structure_features(self, minute: int, spot: float, snapshot: dict,
                            structure: str) -> dict[str, float] | None:
        """Build the exact 28-column input expected by the frozen classifiers."""
        if len(self.spot_history) < 61 or len(self.iv_history) < 31 or len(self.oi_history) < 61:
            return None
        prices = np.asarray([value for _, value in self.spot_history], dtype=float)
        returns = np.diff(np.log(prices))
        if len(returns) < 60:
            return None
        iv_values = np.asarray([value for _, value in self.iv_history], dtype=float)
        iv_changes = np.diff(iv_values[-31:])
        oi_changes = {h: self._relative_change(self.oi_history, h) for h in (5, 15, 60)}
        if any(value is None for value in oi_changes.values()):
            return None

        selected = snapshot["strikes"]
        if structure == "short_2_strangle":
            legs = ((selected[0], "PE", -1.0), (selected[4], "CE", -1.0))
        elif structure == "short_atm_straddle":
            legs = ((selected[2], "PE", -1.0), (selected[2], "CE", -1.0))
        else:
            raise ValueError(f"unsupported expanded-hybrid structure: {structure}")

        entry_value = 0.0
        totals = {name: 0.0 for name in ("delta", "gamma", "vega", "theta_day")}
        for strike, side, weight in legs:
            entry_value += weight * float(snapshot["quotes"][strike][side].mid)
            for name in totals:
                totals[name] += weight * float(snapshot["greeks"][strike][side][name])

        stamp = NSE_EPOCH + timedelta(minutes=minute)
        minute_of_day = stamp.hour * 60 + stamp.minute
        time_fraction = (minute_of_day - SESSION_OPEN_MINUTE) / 375.0
        overnight = 0.0
        if self.previous_session_close and self.session_open:
            overnight = math.log(self.session_open / self.previous_session_close)
        return_past = {h: float(np.sum(returns[-h:])) for h in (5, 15, 60)}
        abs_return_past = {h: float(np.sum(np.abs(returns[-h:]))) for h in (5, 15, 60)}
        log_rv = {h: math.log(max(float(np.sum(returns[-h:] ** 2)), EPSILON)) for h in (5, 15, 60)}
        latest_iv_change = float(self.iv_history[-1][1] - self.iv_history[-2][1])
        leverage = float(np.clip(latest_iv_change / return_past[5], -100.0, 100.0)) \
            if abs(return_past[5]) > EPSILON else 0.0
        values = {
            "return_past_5": return_past[5], "return_past_15": return_past[15],
            "return_past_60": return_past[60],
            "abs_return_past_5": abs_return_past[5], "abs_return_past_15": abs_return_past[15],
            "abs_return_past_60": abs_return_past[60],
            "log_rv_past_5": log_rv[5], "log_rv_past_15": log_rv[15],
            "log_rv_past_60": log_rv[60], "overnight_return": overnight,
            "log_iv_trade": math.log(max(float(snapshot["iv_atm"]), EPSILON)),
            "iv_skew": float(snapshot["iv_skew"]),
            "forward_log_basis": math.log(float(snapshot["forward"]) / spot),
            "oi_imbalance": float(snapshot["oi_imbalance"]),
            "oi_total_change_5": float(oi_changes[5]), "oi_total_change_15": float(oi_changes[15]),
            "oi_total_change_60": float(oi_changes[60]),
            "vol_of_vol_30": float(np.std(iv_changes, ddof=1)), "leverage_effect": leverage,
            "time_fraction": time_fraction, "time_sin": math.sin(2.0 * math.pi * time_fraction),
            "time_cos": math.cos(2.0 * math.pi * time_fraction),
            "sqrt_dte": math.sqrt(max(float(snapshot["days_to_expiry"]), 0.0)),
            "entry_value_pct": abs(entry_value) / spot, "delta": totals["delta"],
            "gamma_s2": totals["gamma"] * spot * spot, "vega": totals["vega"],
            "theta_day": totals["theta_day"], "days_to_expiry": float(snapshot["days_to_expiry"]),
        }
        if list(values) != self.hybrid_features:
            missing = [name for name in self.hybrid_features if name not in values]
            if missing:
                raise RuntimeError(f"live expanded-hybrid features missing: {missing}")
        ordered = {name: float(values[name]) for name in self.hybrid_features}
        return ordered if all(math.isfinite(value) for value in ordered.values()) else None

    def _hybrid_probability(self, features: dict[str, float], primary: bool) -> float:
        frame = pd.DataFrame([[features[name] for name in self.hybrid_features]], columns=self.hybrid_features)
        model = self.primary_hybrid_model if primary else self.secondary_hybrid_model
        return float(model.predict_proba(frame)[0, 1])

    def _on_completed_minute(self, minute: int, spot: float, decision_log_time: int) -> None:
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        minute_of_day = stamp.hour * 60 + stamp.minute
        if self.trading_date != stamp.date():
            for position in list(self.positions):
                self._try_exit(position, minute, spot, "date_change")
            if self.spot_history:
                self.previous_session_close = float(self.spot_history[-1][1])
            self.trading_date = stamp.date()
            self.daily_trade_count = 0
            self.fallback_used_today = False
            self._reset_history()
            self.session_open = spot
        if self.spot_history and minute != self.spot_history[-1][0] + 1:
            LOGGER.warning("minute gap at %s; clearing feature history", stamp)
            self._reset_history()
        self.spot_history.append((minute, spot))

        for position in list(self.positions):
            if minute - position.entry_minute >= HORIZON_MINUTES:
                self._try_exit(position, minute, spot, "120m_horizon")
            elif minute_of_day >= FORCE_EXIT_MINUTE:
                self._try_exit(position, minute, spot, "session_end")

        if not SESSION_OPEN_MINUTE <= minute_of_day <= FORCE_EXIT_MINUTE:
            return
        snapshot = self._option_snapshot(spot, minute, decision_log_time)
        if snapshot is None:
            self.iv_history.clear()
            self._emit_signal({"exchange_minute": stamp.isoformat(sep=" "), "decision": "WAIT",
                               "reason": "five_strike_quotes_missing_stale_or_wide", "spot": spot})
            return
        if self.iv_history and minute != self.iv_history[-1][0] + 1:
            self.iv_history.clear()
        self.iv_history.append((minute, float(snapshot["iv_atm"])))
        if snapshot["oi_total"] is None:
            self.oi_history.clear()
        else:
            if self.oi_history and minute != self.oi_history[-1][0] + 1:
                self.oi_history.clear()
            self.oi_history.append((minute, float(snapshot["oi_total"])))
        features = self._build_features(minute, snapshot)
        if features is None:
            self._emit_signal({"exchange_minute": stamp.isoformat(sep=" "), "decision": "WARMUP",
                               "reason": "need_61_consecutive_spot_and_6_iv_minutes", "spot": spot,
                               "spot_bars": len(self.spot_history), "iv_bars": len(self.iv_history),
                               "strike": snapshot["strike"], "iv_atm": snapshot["iv_atm"]})
            return
        frame = pd.DataFrame([[features[name] for name in self.features]], columns=self.features)
        clipped = frame.clip(lower=self.clip_lower, upper=self.clip_upper, axis=1)
        predicted_total_variance = math.exp(float(self.model.predict(clipped)[0]) + self.bias_correction)
        iv = float(snapshot["iv_atm"])
        implied_total_variance = iv * iv * HORIZON_MINUTES / (252.0 * 375.0)
        variance_ratio = predicted_total_variance / max(implied_total_variance, EPSILON)
        theta_decay_points = abs(float(snapshot["theta_day"])) * HORIZON_MINUTES / 1440.0
        expected_long_points = theta_decay_points * (variance_ratio - 1.0)
        atm = float(snapshot["strike"])
        quotes = snapshot["quotes"][atm]
        mid_premium = quotes["CE"].mid + quotes["PE"].mid
        one_lot_cost = charge_components(mid_premium, mid_premium, self.lot_size, self.selection_slippage_bps)["total_cost"]
        expected_long_rupees_one_lot = expected_long_points * self.lot_size

        reason = "eligible"
        decision = "SHORT" if expected_long_rupees_one_lot < -self.hurdle_multiple * one_lot_cost else "NO_TRADE"
        structure = "short_atm_straddle"
        component = "legacy_rv120_base"
        model_probability: float | None = None

        primary_features = self._structure_features(minute, spot, snapshot, "short_2_strangle")
        secondary_features = self._structure_features(minute, spot, snapshot, "short_atm_straddle")
        primary_probability = (
            self._hybrid_probability(primary_features, True)
            if self.hybrid_enabled and primary_features is not None else None
        )
        secondary_probability = (
            self._hybrid_probability(secondary_features, False)
            if self.hybrid_enabled and secondary_features is not None else None
        )
        if decision == "NO_TRADE" and not self.fallback_used_today and not self.positions:
            if primary_probability is not None and primary_probability >= self.primary_threshold:
                decision, reason = "SHORT", "expanded_primary_probability_gate"
                structure, component, model_probability = (
                    "short_2_strangle", "120m_short_2_strangle_q75", primary_probability,
                )
            elif secondary_probability is not None and secondary_probability >= self.secondary_threshold:
                decision, reason = "SHORT", "expanded_secondary_probability_gate"
                structure, component, model_probability = (
                    "short_atm_straddle", "120m_short_atm_straddle_secondary", secondary_probability,
                )
        if structure == "short_2_strangle":
            selected = snapshot["strikes"]
            selected_mid_premium = (
                snapshot["quotes"][selected[0]]["PE"].mid
                + snapshot["quotes"][selected[4]]["CE"].mid
            )
            one_lot_cost = charge_components(
                selected_mid_premium, selected_mid_premium, self.lot_size,
                self.selection_slippage_bps,
            )["total_cost"]
        if self.daily_trade_count >= MAX_TRADES_PER_DAY:
            decision, reason = "NO_TRADE", "daily_trade_limit"
        elif self.positions:
            decision, reason = "NO_TRADE", "position_already_open"
        elif minute_of_day > LATEST_ENTRY_MINUTE:
            decision, reason = "NO_TRADE", "too_late_for_120m_exit"
        elif decision == "NO_TRADE":
            reason = "all_priority_gates_below_threshold"

        event = {
            "exchange_minute": stamp.isoformat(sep=" "), "decision": decision, "reason": reason,
            "model": self.model_name, "horizon_minutes": HORIZON_MINUTES, "spot": spot,
            "expiry": self.expiry.isoformat(), "strike": atm,
            "call_token": snapshot["call_token"], "put_token": snapshot["put_token"],
            "iv_atm": iv, "predicted_total_variance": predicted_total_variance,
            "implied_total_variance": implied_total_variance, "predicted_rv_to_iv_variance_ratio": variance_ratio,
            "theta_decay_points": theta_decay_points, "expected_long_pnl_points": expected_long_points,
            "expected_long_pnl_rupees_one_lot": expected_long_rupees_one_lot,
            "estimated_roundtrip_cost_one_lot": one_lot_cost, "cost_hurdle_multiple": self.hurdle_multiple,
            "structure": structure, "component": component,
            "primary_model_probability": primary_probability,
            "primary_probability_threshold": self.primary_threshold,
            "secondary_model_probability": secondary_probability,
            "secondary_probability_threshold": self.secondary_threshold,
            "selected_model_probability": model_probability,
            "live_policy_note": "causal first threshold crossing; offline daily-best/no-base-date filter is not live-observable",
            "oi_available": snapshot["oi_available"], "oi_fallback": not snapshot["oi_available"],
            "call_bid": quotes["CE"].bid, "call_ask": quotes["CE"].ask,
            "put_bid": quotes["PE"].bid, "put_ask": quotes["PE"].ask,
            "features": features,
        }
        self._emit_signal(event)
        self.last_signal_minute = minute
        if decision == "SHORT":
            self._enter(minute, spot, snapshot, predicted_total_variance, implied_total_variance,
                        expected_long_points, one_lot_cost, structure, component, model_probability)

    def _enter(self, minute: int, spot: float, snapshot: dict, predicted_var: float,
               implied_var: float, expected_long_points: float, one_lot_cost: float,
               structure: str, component: str, model_probability: float | None) -> None:
        selected = snapshot["strikes"]
        if structure == "short_2_strangle":
            put_strike, call_strike = float(selected[0]), float(selected[4])
        else:
            put_strike = call_strike = float(selected[2])
        call = snapshot["quotes"][call_strike]["CE"]
        put = snapshot["quotes"][put_strike]["PE"]
        self.trade_id += 1
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        position = PaperPosition(
            trade_id=self.trade_id, entry_time=stamp.isoformat(sep=" "), entry_minute=minute,
            expiry=self.expiry.isoformat(), structure=structure, component=component,
            put_strike=put_strike, call_strike=call_strike,
            call_token=int(self.pairs[call_strike]["CE"]), put_token=int(self.pairs[put_strike]["PE"]),
            lots=N_LOTS, lot_size=self.lot_size,
            call_entry_bid=call.bid, put_entry_bid=put.bid, entry_credit=call.bid + put.bid,
            entry_spot=spot, entry_iv=float(snapshot["iv_atm"]),
            predicted_total_variance=predicted_var, implied_total_variance=implied_var,
            expected_long_pnl_points=expected_long_points, estimated_one_lot_cost=one_lot_cost,
            model_probability=model_probability,
        )
        self.positions.append(position)
        self.daily_trade_count += 1
        if component != "legacy_rv120_base":
            self.fallback_used_today = True
        self._emit_trade({"event_type": "paper_entry", "paper_only": True, **asdict(position)})

    def _try_exit(self, position: PaperPosition, minute: int, spot: float, reason: str) -> bool:
        call = best_quote(self.order_book.get(position.call_token))
        put = best_quote(self.order_book.get(position.put_token))
        if call is None or put is None:
            LOGGER.warning("cannot paper-exit trade %d: missing fixed-strike ask", position.trade_id)
            return False
        exit_debit = call.ask + put.ask
        quantity = position.lot_size * position.lots
        gross = (position.entry_credit - exit_debit) * quantity
        # Bid/ask execution already embeds spread/slippage; add only statutory charges here.
        costs = charge_components(exit_debit, position.entry_credit, quantity, 0.0)
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        self._emit_trade({
            "event_type": "paper_exit", "paper_only": True, "trade_id": position.trade_id,
            "entry_time": position.entry_time, "exit_time": stamp.isoformat(sep=" "), "exit_reason": reason,
            "expiry": position.expiry, "structure": position.structure, "component": position.component,
            "put_strike": position.put_strike, "call_strike": position.call_strike,
            "model_probability": position.model_probability, "lots": position.lots,
            "lot_size": position.lot_size, "entry_credit": position.entry_credit,
            "call_exit_ask": call.ask, "put_exit_ask": put.ask, "exit_debit": exit_debit,
            "entry_spot": position.entry_spot, "exit_spot": spot,
            "gross_pnl": gross, "costs": costs, "net_pnl": gross - costs["total_cost"],
            "holding_minutes": minute - position.entry_minute,
        })
        if position in self.positions:
            self.positions.remove(position)
        return True

    def close(self) -> None:
        self.signal_output.close()
        self.trade_output.close()
        self.order_book.close()


class DashboardServer:
    """Small local HTTP server for the live status and append-only JSONL logs."""

    def __init__(self, host: str, port: int, html_path: Path, strategy: LiveRvStrategy) -> None:
        self.host = host
        self.port = port
        self.html_path = html_path
        self.strategy = strategy
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
                parsed = urlparse(self.path)
                try:
                    if parsed.path in {"/", "/index.html"}:
                        self._send(owner.html_path.read_bytes(), "text/html; charset=utf-8")
                    elif parsed.path == "/api/status":
                        self._json(owner._status())
                    elif parsed.path == "/api/signals":
                        limit = owner._limit(parse_qs(parsed.query))
                        self._json(owner._read_jsonl(owner.strategy.signal_path, limit))
                    elif parsed.path == "/api/trades":
                        limit = owner._limit(parse_qs(parsed.query))
                        self._json(owner._read_jsonl(owner.strategy.trade_path, limit))
                    elif parsed.path == "/api/health":
                        self._json({"ok": True, "paper_only": True, "server_time": datetime.now().isoformat(sep=" ")})
                    else:
                        self.send_error(404)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    self._json({"error": str(error)}, status=500)

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
        self.thread = threading.Thread(target=self.server.serve_forever, name="rv120-dashboard", daemon=True)

    @staticmethod
    def _limit(query: dict[str, list[str]]) -> int:
        try:
            return min(max(int(query.get("limit", ["100"])[0]), 1), 1000)
        except ValueError:
            return 100

    @staticmethod
    def _read_jsonl(path: Path, limit: int) -> list[dict]:
        if not path.exists():
            return []
        rows: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines()[-limit:]:
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def _status(self) -> dict:
        if self.strategy.status_path.exists():
            status = json.loads(self.strategy.status_path.read_text(encoding="utf-8"))
        else:
            status = {"decision": "WAITING", "reason": "waiting_for_first_completed_market_minute"}
        status["runtime"] = {
            "paper_only": True,
            "expiry": self.strategy.expiry.isoformat(),
            "lot_size": self.strategy.lot_size,
            "lots": N_LOTS,
            "horizon_minutes": HORIZON_MINUTES,
            "daily_trade_count": self.strategy.daily_trade_count,
            "max_trades_per_day": MAX_TRADES_PER_DAY,
            "position": asdict(self.strategy.positions[0]) if self.strategy.positions else None,
            "positions": [asdict(position) for position in self.strategy.positions],
            "open_position_count": len(self.strategy.positions),
            "server_time": datetime.now().isoformat(sep=" "),
        }
        return status

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)


def load_config(path: Path) -> dict:
    config = load_source_config(path)
    for key in ("multicast_ip", "port", "cm_multicast_ip", "cm_port", "interface_ip", "price_divisor", "max_datagram_bytes"):
        if key not in config:
            raise ValueError(f"config missing {key}")
    return config


def configure_logging(log_path: Path, level: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(log_path, maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[file_handler, console])


def main() -> int:
    parser = argparse.ArgumentParser(description="120-minute BANKNIFTY expanded-hybrid live inference (paper only)")
    parser.add_argument("--config", type=Path, default=BASE_DIR / "config.json")
    parser.add_argument("--contract-file", type=Path, help="current FO contract master; otherwise use config contract_file")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--hybrid-model", type=Path, default=DEFAULT_HYBRID_MODEL)
    parser.add_argument("--lzo-dll", type=Path, default=BASE_DIR / "native" / "lzo2.dll")
    parser.add_argument("--logs-dir", type=Path, default=BASE_DIR / "logs" / "rv120")
    parser.add_argument("--dashboard-host", default="127.0.0.1")
    parser.add_argument("--dashboard-port", type=int, default=8770)
    parser.add_argument("--preflight", action="store_true", help="validate contracts/model/config without opening multicast sockets")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    contract_file = args.contract_file or (Path(config["contract_file"]) if config.get("contract_file") else None)
    if contract_file is None:
        parser.error("set contract_file in local config or pass --contract-file with the current FO master")
    contract_file = contract_file.resolve()
    configure_logging(args.logs_dir.resolve() / "runtime.log", str(config.get("log_level", "INFO")))
    contracts, lot_size, expiry = load_monthly_contracts(
        contract_file, date.today(), float(config.get("price_divisor", 100.0)))
    strategy = LiveRvStrategy(
        config, contracts, lot_size, expiry, args.model.resolve(),
        args.hybrid_model.resolve(), args.logs_dir.resolve(),
    )
    ce_count = sum(c.option_type == "CE" for c in contracts.values())
    pe_count = sum(c.option_type == "PE" for c in contracts.values())
    LOGGER.info(
        "PREFLIGHT_OK paper_only=true contract=%s expiry=%s CE=%d PE=%d lot_size=%d "
        "rv_model=%s rv_features=%d hybrid_model=%s hybrid_features=%d primary_threshold=%.6f secondary_threshold=%.6f",
        contract_file, expiry, ce_count, pe_count, lot_size,
        args.model.resolve(), len(strategy.features), args.hybrid_model.resolve(),
        len(strategy.hybrid_features), strategy.primary_threshold, strategy.secondary_threshold,
    )
    if args.preflight:
        strategy.close()
        return 0

    dashboard_path = BASE_DIR / "rv120_dashboard.html"
    if not dashboard_path.exists():
        raise FileNotFoundError(dashboard_path)
    dashboard = DashboardServer(args.dashboard_host, args.dashboard_port, dashboard_path, strategy)
    dashboard.start()
    LOGGER.info("DASHBOARD_READY url=http://%s:%d", args.dashboard_host, args.dashboard_port)

    tracked_tokens = set(contracts)
    fo_config = dict(config, receiver_name="NSE FO")
    cm_config = dict(config, receiver_name="NSE CM indices",
                     multicast_ip=config["cm_multicast_ip"], port=int(config["cm_port"]),
                     interface_ip=config.get("cm_interface_ip", config["interface_ip"]))
    fo_receiver = NseFoUdpReceiver(fo_config, Lzo1zDecoder(args.lzo_dll.resolve()))
    cm_receiver = NseFoUdpReceiver(cm_config, Lzo1zDecoder(args.lzo_dll.resolve()))
    fo_receiver.open()
    cm_receiver.open()
    selector = selectors.DefaultSelector()
    selector.register(fo_receiver.socket, selectors.EVENT_READ, fo_receiver)
    selector.register(cm_receiver.socket, selectors.EVENT_READ, cm_receiver)
    LOGGER.info("INFERENCE_STARTED paper_only=true fo=%s:%s cm=%s:%s interface=%s",
                config["multicast_ip"], config["port"], config["cm_multicast_ip"], config["cm_port"], config["interface_ip"])

    def handle_message(message: bytes, source: NseFoUdpReceiver) -> None:
        if len(message) < 12:
            return
        code = int.from_bytes(message[10:12], "big", signed=True)
        if code == 7207:
            strategy.on_index_message(message)
            return
        if source is cm_receiver:
            return
        if code in SUPPORTED_OI_CODES:
            strategy.on_oi_message(message)
            return
        if code not in (7208, 17208):
            return
        try:
            for update in decode_7208(message, float(config["price_divisor"]), tracked_tokens):
                strategy.on_option_update(update)
        except DecodeError:
            return

    try:
        while True:
            for key, _ in selector.select(timeout=1.0):
                receiver: NseFoUdpReceiver = key.data
                assert receiver.socket is not None
                datagram, _sender = receiver.socket.recvfrom(int(receiver.config["max_datagram_bytes"]))
                try:
                    for message in receiver.unpack_datagram(datagram):
                        handle_message(message, receiver)
                except (PacketError, LzoError) as error:
                    LOGGER.debug("discarded malformed packet: %s", error)
    except KeyboardInterrupt:
        LOGGER.info("stopping inference")
    finally:
        selector.close()
        fo_receiver.close()
        cm_receiver.close()
        dashboard.close()
        strategy.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
