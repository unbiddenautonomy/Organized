"""Live BANKNIFTY ATM implied-volatility inference for the saved IV v3 models."""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import TextIO
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd

from contract_repository import OptionContract
from decoder_7208 import MarketUpdate7208
from spot_inference import MinuteBar, NSE_EPOCH, decode_banknifty_7207


LOGGER = logging.getLogger(__name__)
HORIZONS = (5, 15)
LAGS = (1, 5, 15, 30, 60)


def _nan_window_stats(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Column-wise NaN-aware mean/std without warnings for padded history."""
    valid = np.isfinite(block)
    count = valid.sum(axis=0)
    total = np.where(valid, block, 0.0).sum(axis=0)
    mean = np.full(block.shape[1], np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    squared = np.where(valid, (block - mean) ** 2, 0.0).sum(axis=0)
    variance = np.full(block.shape[1], np.nan, dtype=np.float64)
    np.divide(squared, count, out=variance, where=count > 0)
    return mean, np.sqrt(variance)


@dataclass(frozen=True, slots=True)
class IvRow:
    minute: int
    values: dict[str, float]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def _bs_price(spot: float, strike: float, years: float, rate: float, sigma: float, is_call: bool) -> float:
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discounted = strike * math.exp(-rate * years)
    if is_call:
        return spot * _normal_cdf(d1) - discounted * _normal_cdf(d2)
    return discounted * _normal_cdf(-d2) - spot * _normal_cdf(-d1)


def implied_volatility(price: float, spot: float, strike: float, years: float, rate: float, is_call: bool) -> float:
    """Stable bisection IV; returns NaN when the quote violates arbitrage bounds."""
    if min(price, spot, strike, years) <= 0:
        return math.nan
    discounted = strike * math.exp(-rate * years)
    lower = max(0.0, spot - discounted) if is_call else max(0.0, discounted - spot)
    upper = spot if is_call else discounted
    if price < lower - 1e-6 or price > upper + 1e-6:
        return math.nan
    lo, hi = 0.0001, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if _bs_price(spot, strike, years, rate, mid, is_call) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _greeks(spot: float, strike: float, years: float, rate: float, sigma: float, is_call: bool) -> dict[str, float]:
    if not math.isfinite(sigma) or sigma <= 0 or years <= 0:
        return {name: math.nan for name in ("delta", "gamma", "vega", "theta")}
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    pdf = _normal_pdf(d1)
    delta = _normal_cdf(d1) if is_call else _normal_cdf(d1) - 1.0
    gamma = pdf / (spot * sigma * root_t)
    vega = spot * pdf * root_t / 100.0
    decay = -(spot * pdf * sigma) / (2.0 * root_t)
    if is_call:
        theta = (decay - rate * strike * math.exp(-rate * years) * _normal_cdf(d2)) / 365.0
    else:
        theta = (decay + rate * strike * math.exp(-rate * years) * _normal_cdf(-d2)) / 365.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


class IvPredictionPipeline:
    """Build training-compatible one-minute ATM rows and forecast same-contract IV."""

    def __init__(self, config: dict, contracts: dict[int, OptionContract], future_token: int) -> None:
        self.config = config
        self.price_divisor = float(config.get("price_divisor", 100.0))
        self.rate = float(config.get("iv_risk_free_rate", 0.065))
        self.future_token = future_token
        self.expiry = next(iter(contracts.values())).expiry
        self.pairs: dict[float, dict[str, int]] = {}
        self.contract_by_token = contracts
        for contract in contracts.values():
            self.pairs.setdefault(float(contract.strike_price), {})[contract.option_type] = contract.token
        self.pairs = {strike: pair for strike, pair in self.pairs.items() if {"CE", "PE"} <= pair.keys()}

        self.spot_bars: dict[int, MinuteBar] = {}
        self.future_bars: dict[int, MinuteBar] = {}
        self.option_bars: dict[int, dict[int, MinuteBar]] = {}
        self.latest_spot_minute: int | None = None
        self.latest_future_minute: int | None = None
        self.last_finalized_minute: int | None = None
        self.raw_history: deque[IvRow] = deque(maxlen=181)
        self.feature_history: deque[IvRow] = deque(maxlen=60)
        self.last_prediction_minute: int | None = None
        self.prediction_listeners: list[Callable[[dict], None]] = []
        self.oi_by_token: dict[int, tuple[int, int, int]] = {}
        self.oi_max_age_minutes = int(config.get("iv_oi_max_age_minutes", 15))
        self.min_warmup_minutes = max(5, int(config.get("iv_min_warmup_minutes", 5)))

        model_dir = Path(config.get("iv_model_dir") or
                         Path(__file__).resolve().parent / "models" / "legacy_spot_iv" / "iv")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*loading a serialized model.*", category=UserWarning)
            self.artifacts = {h: joblib.load(model_dir / f"horizon_{h}.joblib") for h in HORIZONS}
        output = Path(config.get("iv_prediction_file", "logs/iv_predictions.jsonl"))
        if not output.is_absolute():
            output = Path(__file__).resolve().parent / output
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output
        self.output: TextIO = output.open("a", encoding="utf-8", buffering=1)
        LOGGER.info("IV_PIPELINE_READY expiry=%s strikes=%d models=%d output=%s", self.expiry, len(self.pairs), len(self.artifacts), output)

    @staticmethod
    def _add_tick(bars: dict[int, MinuteBar], minute: int, price: float) -> None:
        bar = bars.get(minute)
        if bar is None:
            bars[minute] = MinuteBar(price, price, price, price)
        else:
            bar.update(price)
        if len(bars) > 240:
            cutoff = minute - 185
            for old_minute in tuple(bars):
                if old_minute < cutoff:
                    del bars[old_minute]

    def on_broadcast_message(self, message: bytes) -> None:
        decoded = decode_banknifty_7207(message, self.price_divisor)
        if decoded is None:
            return
        log_time, spot = decoded
        minute = log_time // 60
        self._reset_if_replay_rewound(minute)
        self._add_tick(self.spot_bars, minute, spot)
        self.latest_spot_minute = minute if self.latest_spot_minute is None else max(self.latest_spot_minute, minute)
        self._finalize_ready_minutes()

    def on_7208(self, update: MarketUpdate7208) -> None:
        if update.last_traded_price <= 0:
            return
        minute = update.log_time // 60
        self._reset_if_replay_rewound(minute)
        if update.token == self.future_token:
            self._add_tick(self.future_bars, minute, update.last_traded_price)
            self.latest_future_minute = minute if self.latest_future_minute is None else max(self.latest_future_minute, minute)
        elif update.token in self.contract_by_token:
            token_bars = self.option_bars.setdefault(update.token, {})
            self._add_tick(token_bars, minute, update.last_traded_price)
            # Only the last few completed minutes are needed before sealing an
            # ATM row. Bound this even if spot/index packets are unavailable.
            if len(token_bars) > 4:
                cutoff = minute - 3
                for old_minute in tuple(token_bars):
                    if old_minute < cutoff:
                        del token_bars[old_minute]
        self._finalize_ready_minutes()

    def _reset_if_replay_rewound(self, minute: int) -> None:
        latest = max((value for value in (self.latest_spot_minute, self.latest_future_minute) if value is not None), default=None)
        tolerance = int(self.config.get("replay_rewind_tolerance_minutes", 5))
        if latest is None or minute >= latest - tolerance:
            return
        LOGGER.warning("IV_REPLAY_REWIND old_minute=%d new_minute=%d; resetting aggregation and OI state", latest, minute)
        self.spot_bars.clear()
        self.future_bars.clear()
        self.option_bars.clear()
        self.latest_spot_minute = None
        self.latest_future_minute = None
        self.last_finalized_minute = None
        self.raw_history.clear()
        self.feature_history.clear()
        self.last_prediction_minute = None
        self.oi_by_token.clear()

    def on_oi_update(self, token: int, open_interest: int, log_time: int, source_code: int) -> None:
        if token not in self.contract_by_token or open_interest < 0:
            return
        existing = self.oi_by_token.get(token)
        priority = {7201: 1, 7202: 2, 17201: 3, 17202: 4}
        if existing is not None:
            _, old_time, old_code = existing
            if log_time < old_time:
                return
            if log_time == old_time and priority.get(source_code, 0) < priority.get(old_code, 0):
                return
        self.oi_by_token[token] = (open_interest, log_time, source_code)

    def _oi_value(self, token: int, minute: int) -> float:
        state = self.oi_by_token.get(token)
        if state is None:
            return math.nan
        value, log_time, _ = state
        if log_time > 0 and abs(minute - log_time // 60) > self.oi_max_age_minutes:
            return math.nan
        return float(value)

    def _finalize_ready_minutes(self) -> None:
        if self.latest_spot_minute is None or self.latest_future_minute is None:
            return
        # Hold one extra completed minute so late/out-of-order option packets
        # have time to populate the ATM CE/PE bars before the row is sealed.
        through = min(self.latest_spot_minute, self.latest_future_minute) - 2
        start = through if self.last_finalized_minute is None else self.last_finalized_minute + 1
        for minute in range(start, through + 1):
            self._finalize_minute(minute)
        self.last_finalized_minute = through
        cutoff = through - 185
        self.spot_bars = {m: b for m, b in self.spot_bars.items() if m >= cutoff}
        self.future_bars = {m: b for m, b in self.future_bars.items() if m >= cutoff}
        self.option_bars = {t: {m: b for m, b in bars.items() if m >= cutoff} for t, bars in self.option_bars.items()}

    def _reset(self) -> None:
        self.raw_history.clear()
        self.feature_history.clear()

    def _finalize_minute(self, minute: int) -> None:
        spot_bar, future_bar = self.spot_bars.get(minute), self.future_bars.get(minute)
        if spot_bar is None or future_bar is None:
            self._reset()
            return
        strike = min(self.pairs, key=lambda value: abs(value - spot_bar.close))
        pair = self.pairs[strike]
        call_bar = self.option_bars.get(pair["CE"], {}).get(minute)
        put_bar = self.option_bars.get(pair["PE"], {}).get(minute)
        if call_bar is None or put_bar is None:
            self._reset()
            return
        if self.raw_history and minute != self.raw_history[-1].minute + 1:
            self._reset()
        values = self._make_raw_values(
            minute, spot_bar.close, future_bar.open, strike, call_bar.close, put_bar.close,
            self._oi_value(pair["CE"], minute), self._oi_value(pair["PE"], minute),
        )
        if not math.isfinite(values["iv"]):
            self._reset()
            return
        self.raw_history.append(IvRow(minute, values))
        self.feature_history.append(IvRow(minute, values))
        if any(len(self.raw_history) >= int(a["required_history_minutes"]) for a in self.artifacts.values()):
            self._predict(minute, pair, values["iv"])

    def _make_raw_values(
        self, minute: int, spot: float, future_open: float, strike: float,
        call: float, put: float, call_oi: float, put_oi: float,
    ) -> dict[str, float]:
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        expiry_stamp = datetime.combine(self.expiry, time(15, 30))
        years = max((expiry_stamp - stamp).total_seconds(), 60.0) / (365.0 * 86400.0)
        call_iv = implied_volatility(call, spot, strike, years, self.rate, True)
        put_iv = implied_volatility(put, spot, strike, years, self.rate, False)
        valid = [value for value in (call_iv, put_iv) if math.isfinite(value)]
        iv = float(np.mean(valid)) if valid else math.nan
        cg = _greeks(spot, strike, years, self.rate, call_iv, True)
        pg = _greeks(spot, strike, years, self.rate, put_iv, False)
        straddle_theta = cg["theta"] + pg["theta"]
        straddle_gamma = cg["gamma"] + pg["gamma"]
        snapshot = {
            "strike": strike, "days_to_expiry": years * 365.0, "future_open": future_open,
            "spot_price": spot, "call_price": call, "put_price": put,
            "call_oi": call_oi, "put_oi": put_oi,
            "synthetic_future": strike + call - put, "atm_distance": abs(strike - spot),
            "iv_call": call_iv, "iv_put": put_iv, "iv": iv,
            "gamma_put": pg["gamma"], "vega_put": pg["vega"],
            "straddle_theta": straddle_theta,
            "gamma_theta_ratio": straddle_gamma / abs(straddle_theta) if straddle_theta else math.nan,
            "straddle_value": call + put, "future_basis": future_open - spot,
            "minutes_since_open": float(stamp.hour * 60 + stamp.minute - 555),
            "is_expiry_day": float(stamp.date() == self.expiry),
        }
        return snapshot

    def _direct_features(self, horizon: int) -> dict[str, float]:
        rows=list(self.raw_history); values=dict(rows[-1].values); current=values
        values["future_basis_pct"]=current["future_basis"]/current["spot_price"]
        values["iv_call_put_spread"]=current["iv_call"]-current["iv_put"]
        values["put_call_oi_ratio"]=current["put_oi"]/current["call_oi"] if current["call_oi"] else math.nan
        for window in (1,5,15,30,45,60):
            if window>horizon: continue
            old=rows[-1-window].values
            for name in ("spot_price","future_open","iv","iv_call","iv_put","straddle_value"):
                values[f"{name}_return_{window}m"]=current[name]/old[name]-1.0 if old[name] else math.nan
                if name.startswith("iv"): values[f"{name}_change_{window}m"]=current[name]-old[name]
        changes=np.diff([row.values["iv"] for row in rows])
        for window in (5,15,30,45,60):
            if window>horizon: continue
            block=changes[-window:]
            values[f"iv_change_mean_{window}m"]=float(np.mean(block))
            values[f"iv_change_std_{window}m"]=float(np.std(block,ddof=0))
        stamp=NSE_EPOCH+timedelta(minutes=rows[-1].minute); minute=stamp.hour*60+stamp.minute
        values["time_sin"]=math.sin(2*math.pi*minute/1440); values["time_cos"]=math.cos(2*math.pi*minute/1440)
        return values

    def _engineer_lags(self) -> dict[str, float]:
        rows = list(self.raw_history)
        values = dict(rows[-1].values)
        for lag in LAGS:
            old = rows[-1 - lag].values if len(rows) > lag else None
            for name in ("spot_price", "future_open", "iv", "iv_call", "iv_put", "straddle_value"):
                current_value = values[name]
                old_value = old[name] if old is not None else math.nan
                values[f"{name}_return_{lag}m"] = current_value / old_value - 1.0 if old_value and math.isfinite(old_value) else math.nan
                if name.startswith("iv"):
                    values[f"{name}_change_{lag}m"] = current_value - old_value if math.isfinite(old_value) else math.nan
        return values

    @staticmethod
    def _generic_vector(sequence: np.ndarray, stamp: datetime) -> np.ndarray:
        pieces = [sequence[-1]]
        for window in (5, 15, 30, 60):
            block = sequence[-window:]
            mean, std = _nan_window_stats(block)
            pieces.extend((mean, std, block[-1] - block[0]))
        minute_of_day = stamp.hour * 60 + stamp.minute
        pieces.extend((np.asarray([math.sin(2 * math.pi * minute_of_day / 1440)]), np.asarray([math.cos(2 * math.pi * minute_of_day / 1440)])))
        return np.concatenate(pieces).reshape(1, -1)

    def _predict(self, minute: int, pair: dict[str, int], current_iv: float) -> None:
        if minute == self.last_prediction_minute:
            return
        stamp = NSE_EPOCH + timedelta(minutes=minute)
        history = list(self.raw_history)
        forecasts: dict[str, dict] = {}
        for horizon, artifact in self.artifacts.items():
            required=int(artifact["required_history_minutes"])
            if len(history)<required: continue
            method = artifact["selected_model"]
            if artifact["kind"] == "none" or artifact["model"] is None:
                change = 0.0
            elif artifact["kind"] == "direct":
                names=artifact["input_features"]; values=self._direct_features(horizon)
                raw=pd.DataFrame([[values.get(name,math.nan) for name in names]],columns=names)
                change=float(artifact["model"].predict(raw)[0])
            else:
                raise RuntimeError(f"unsupported live IV artifact kind: {artifact['kind']}")
            forecasts[str(horizon)] = {"model": method, "predicted_iv_change": change, "predicted_iv": current_iv + change}
        event = {
            "event_type": "iv_prediction", "exchange_minute": stamp.isoformat(sep=" "),
            "expiry": self.expiry.isoformat(), "strike": self.raw_history[-1].values["strike"],
            "call_token": pair["CE"], "put_token": pair["PE"], "current_iv": current_iv,
            "available_history_minutes": len(history), "warmup_complete": bool(forecasts),
            "horizon_readiness": {str(h): len(history)>=int(a["required_history_minutes"]) for h,a in self.artifacts.items()},
            "missing_live_features": [name for name in ("call_oi", "put_oi") if not math.isfinite(self.raw_history[-1].values[name])],
            "oi_sources": {
                "call": self.oi_by_token.get(pair["CE"], (None, None, None))[2],
                "put": self.oi_by_token.get(pair["PE"], (None, None, None))[2],
            },
            "oi_policy": "live token OI; saved median imputer only when missing/stale", "predictions": forecasts,
            "live_features": self._stacking_features(self._engineer_lags()),
            # JSON forecast keys are strings; the history lookup requires an
            # integer horizon.
            "hmm_features": {str(h): self._hmm_features(int(h)) for h in forecasts},
        }
        encoded = json.dumps(event, separators=(",", ":"), allow_nan=False)
        self.output.write(encoded + "\n")
        LOGGER.info("IV_PREDICTION %s", encoded)
        for listener in tuple(self.prediction_listeners):
            listener(event)
        self.last_prediction_minute = minute

    def _hmm_features(self,horizon:int)->dict[str,float]:
        rows=list(self.raw_history); current=rows[-1].values; old=rows[-1-horizon].values
        returns=np.diff(np.log([row.values["spot_price"] for row in rows[-(horizon+1):]]))
        rv=float(np.std(returns,ddof=1)*math.sqrt(252*375)) if len(returns)>=2 else math.nan
        return {f"spot_return_{horizon}m":current["spot_price"]/old["spot_price"]-1,
                f"iv_change_{horizon}m":current["iv"]-old["iv"],
                f"straddle_return_{horizon}m":current["straddle_value"]/old["straddle_value"]-1,
                "future_basis_pct":current["future_basis"]/current["spot_price"],
                f"rv_{horizon}m":rv,f"vrp_{horizon}m":current["iv"]-rv}

    def _stacking_features(self, values: dict[str, float]) -> dict[str, float | None]:
        """Small, bounded snapshot used by the downstream P&L stack."""
        rows = list(self.raw_history)
        returns = np.diff(np.log([row.values["spot_price"] for row in rows if row.values["spot_price"] > 0]))
        rv = float(np.std(returns, ddof=1) * math.sqrt(252.0 * 375.0)) if len(returns) >= 30 else math.nan
        current_iv = float(values["iv"])
        snapshot = {
            "straddle_value": float(values["straddle_value"]),
            "future_basis_pct": float(values["future_basis"] / values["spot_price"]),
            "days_to_expiry": float(values["days_to_expiry"]), "is_expiry_day": float(values["is_expiry_day"]),
            "minutes_since_open": float(values["minutes_since_open"]),
            "spot_price_return_15m": float(values.get("spot_price_return_15m", math.nan)),
            "iv_change_15m": float(values.get("iv_change_15m", math.nan)),
            "straddle_value_return_15m": float(values.get("straddle_value_return_15m", math.nan)),
            "1d_RV": rv, "VRP_1d_RV": current_iv - rv if math.isfinite(rv) else math.nan,
        }
        return {name: (value if math.isfinite(value) else None) for name, value in snapshot.items()}

    def add_prediction_listener(self, listener: Callable[[dict], None]) -> None:
        self.prediction_listeners.append(listener)

    def readiness_snapshot(self) -> dict:
        available=len(self.raw_history)
        return {str(h): {"required_bars":int(a["required_history_minutes"]),"available_bars":available,
                         "bars_remaining":max(0,int(a["required_history_minutes"])-available),
                         "ready":available>=int(a["required_history_minutes"]),
                         "latest_spot_minute":self.latest_spot_minute,"latest_future_minute":self.latest_future_minute,
                         "spot_pending_bars":len(self.spot_bars),"future_pending_bars":len(self.future_bars),
                         "option_tokens_with_bars":len(self.option_bars)} for h,a in self.artifacts.items()}

    def close(self) -> None:
        self.output.close()
