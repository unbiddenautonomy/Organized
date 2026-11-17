"""Dedicated live BANKNIFTY spot-prediction pipeline.

This module intentionally contains only the data path required by the saved
Spot Boosting (XGBoost) models: 7207 spot decoding, front-future tick capture,
one-minute aggregation, training-compatible feature engineering, preprocessing,
and 5/15/30/45/60-minute spot predictions.
"""

from __future__ import annotations

import json
import logging
import math
import warnings
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TextIO
from collections.abc import Callable

import joblib
import numpy as np
import pandas as pd
import psycopg

import _repo_path  # noqa: F401  (temporary, until step 10)
from voltr.core.clock import NSE_EPOCH
from voltr.instruments.master import OptionContract, is_monthly_expiry, load_front_future
from voltr.market.bars import MinuteBar
from voltr.market.stats import nan_window_stats as _nan_window_stats
from voltr.feed.messages.indices import decode_banknifty_7207  # noqa: F401  (re-exported for callers)
from voltr.feed.messages.market_by_price import MarketUpdate7208


LOGGER = logging.getLogger(__name__)
BASE_FEATURES = (
    "atm_distance",
    "future_basis",
    "future_basis_pct",
    "is_weekly",
    "spot_price_return_30m",
    "is_monthly",
    "spot_price_return_5m",
    "future_open_return_60m",
    "future_open_return_30m",
    "spot_price_return_60m",
    "future_open_return_5m",
)
HORIZON_MODELS = {
    5: "xgb_absolute",
    15: "xgb_absolute",
    30: "xgb_pseudohuber",
    45: "xgb_pseudohuber",
    60: "xgb_absolute",
}


@dataclass(frozen=True, slots=True)
class PriceRow:
    minute: int
    spot_price: float
    future_open: float


@dataclass(frozen=True, slots=True)
class LoadedHorizon:
    horizon: int
    model_name: str
    preprocessor: object
    model: object
    target_scale: float
    input_features: tuple[str, ...]
    required_history_minutes: int


class SpotPredictionPipeline:
    """Stateful, single-threaded live feature and inference pipeline."""

    def __init__(self, config: dict, option_contracts: dict[int, OptionContract]) -> None:
        self.config = config
        self.price_divisor = float(config.get("price_divisor", 100.0))
        self.option_expiry = next(iter(option_contracts.values())).expiry
        self.option_strikes = tuple(sorted({float(item.strike_price) for item in option_contracts.values()}))
        self.future_token, self.future_expiry = self._load_front_future()
        self.is_monthly = float(self._is_monthly_expiry())
        self.is_weekly = 1.0 - self.is_monthly

        self.spot_bars: dict[int, MinuteBar] = {}
        self.future_bars: dict[int, MinuteBar] = {}
        self.latest_spot_minute: int | None = None
        self.latest_future_minute: int | None = None
        self.last_finalized_minute: int | None = None
        self.price_history: deque[PriceRow] = deque(maxlen=181)
        self.last_prediction_minute: int | None = None
        self.prediction_listeners: list[Callable[[dict], None]] = []

        model_dir = Path(config.get("spot_model_dir") or
                         Path(__file__).resolve().parent / "models" / "legacy_spot_iv" / "spot")
        self.models = self._load_models(model_dir)

        output = Path(config.get("spot_prediction_file", "logs/spot_predictions.jsonl"))
        if not output.is_absolute():
            output = Path(__file__).resolve().parent / output
        output.parent.mkdir(parents=True, exist_ok=True)
        self.output_path = output
        self.output: TextIO = output.open("a", encoding="utf-8", buffering=1)

        LOGGER.info(
            "SPOT_PIPELINE_READY index=BANK_NIFTY future_token=%d future_expiry=%s "
            "option_expiry=%s strikes=%d models=%d output=%s",
            self.future_token, self.future_expiry, self.option_expiry,
            len(self.option_strikes), len(self.models), output,
        )

    def _db_options(self) -> dict:
        return {
            "host": self.config["db_host"],
            "port": int(self.config["db_port"]),
            "user": self.config["db_user"],
            "password": self.config["db_password"],
            "dbname": self.config["db_name"],
            "sslmode": self.config.get("db_sslmode", "disable"),
            "connect_timeout": 5,
        }

    def _load_front_future(self) -> tuple[int, object]:
        if self.config.get("contract_file"):
            return load_front_future(self.config)
        sql = """
            SELECT fininstrmid, xprydt::date
            FROM public.nse_fo_contract
            WHERE UPPER(tckrsymb) = %(symbol)s
              AND UPPER(fininstrmnm) = 'FUTIDX'
              AND xprydt::date >= CURRENT_DATE
            ORDER BY xprydt::date, fininstrmid::bigint
            LIMIT 1
        """
        with psycopg.connect(**self._db_options()) as connection:
            row = connection.execute(
                sql, {"symbol": str(self.config.get("contract_symbol", "BANKNIFTY")).upper()}
            ).fetchone()
        if row is None:
            raise RuntimeError("no non-expired BANKNIFTY FUTIDX contract found")
        return int(row[0]), row[1]

    def _is_monthly_expiry(self) -> bool:
        if self.config.get("contract_file"):
            return is_monthly_expiry(self.config, self.option_expiry)
        sql = """
            SELECT %(expiry)s::date = MAX(xprydt::date)
            FROM public.nse_fo_contract
            WHERE UPPER(tckrsymb) = %(symbol)s
              AND UPPER(optntp) IN ('CE', 'PE')
              AND EXTRACT(YEAR FROM xprydt::date) = EXTRACT(YEAR FROM %(expiry)s::date)
              AND EXTRACT(MONTH FROM xprydt::date) = EXTRACT(MONTH FROM %(expiry)s::date)
        """
        with psycopg.connect(**self._db_options()) as connection:
            value = connection.execute(sql, {
                "symbol": str(self.config.get("contract_symbol", "BANKNIFTY")).upper(),
                "expiry": self.option_expiry,
            }).fetchone()[0]
        return bool(value)

    @staticmethod
    def _load_models(model_dir: Path) -> dict[int, LoadedHorizon]:
        loaded: dict[int, LoadedHorizon] = {}
        configured = (5, 15)
        for horizon in configured:
            model_name = HORIZON_MODELS[horizon]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=r".*loading a serialized model.*", category=UserWarning)
                prep_bundle = joblib.load(model_dir / f"horizon_{horizon}_preprocessor.joblib")
                model_bundle = joblib.load(model_dir / f"horizon_{horizon}_{model_name}.joblib")
            if prep_bundle.get("architecture") != "direct_horizon_warmup":
                raise RuntimeError(f"horizon {horizon} is not a direct horizon-warmup artifact")
            if int(model_bundle["horizon"]) != horizon:
                raise RuntimeError(f"horizon mismatch in {model_name} artifact")
            loaded[horizon] = LoadedHorizon(
                horizon=horizon,
                model_name=model_name,
                preprocessor=prep_bundle["pipeline"],
                model=model_bundle["model"],
                target_scale=float(model_bundle.get("target_scale", prep_bundle["target_scale"])),
                input_features=tuple(prep_bundle["input_features"]),
                required_history_minutes=int(prep_bundle["required_history_minutes"]),
            )
        return loaded

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
        if update.token != self.future_token or update.last_traded_price <= 0:
            return
        minute = update.log_time // 60
        self._reset_if_replay_rewound(minute)
        self._add_tick(self.future_bars, minute, update.last_traded_price)
        self.latest_future_minute = minute if self.latest_future_minute is None else max(self.latest_future_minute, minute)
        self._finalize_ready_minutes()

    def _reset_if_replay_rewound(self, minute: int) -> None:
        latest = max((value for value in (self.latest_spot_minute, self.latest_future_minute) if value is not None), default=None)
        tolerance = int(self.config.get("replay_rewind_tolerance_minutes", 5))
        if latest is None or minute >= latest - tolerance:
            return
        LOGGER.warning("SPOT_REPLAY_REWIND old_minute=%d new_minute=%d; resetting aggregation state", latest, minute)
        self.spot_bars.clear()
        self.future_bars.clear()
        self.latest_spot_minute = None
        self.latest_future_minute = None
        self.last_finalized_minute = None
        self.price_history.clear()
        self.last_prediction_minute = None

    def _finalize_ready_minutes(self) -> None:
        if self.latest_spot_minute is None or self.latest_future_minute is None:
            return
        # Match the IV pipeline's sealing watermark. IV deliberately allows one
        # extra minute for late ATM option packets, so sealing spot at the same
        # watermark makes both model events carry the identical exchange minute.
        completed_through = min(self.latest_spot_minute, self.latest_future_minute) - 2
        start = completed_through if self.last_finalized_minute is None else self.last_finalized_minute + 1
        for minute in range(start, completed_through + 1):
            self._finalize_minute(minute)
        self.last_finalized_minute = completed_through
        cutoff = completed_through - 185
        self.spot_bars = {key: value for key, value in self.spot_bars.items() if key >= cutoff}
        self.future_bars = {key: value for key, value in self.future_bars.items() if key >= cutoff}

    def _finalize_minute(self, minute: int) -> None:
        spot = self.spot_bars.get(minute)
        future = self.future_bars.get(minute)
        if spot is None or future is None:
            # Never bridge missing minutes: lag positions must remain true minutes.
            if self.price_history:
                LOGGER.warning("SPOT_PIPELINE_GAP minute=%d; rolling history reset", minute)
            self.price_history.clear()
            return
        if self.price_history and minute != self.price_history[-1].minute + 1:
            self.price_history.clear()
        self.price_history.append(PriceRow(minute, spot.close, future.open))
        if any(len(self.price_history) >= artifact.required_history_minutes for artifact in self.models.values()):
            self._predict(minute, spot.close)

    def _direct_features(self, horizon: int) -> dict[str, float]:
        rows = list(self.price_history)
        current = rows[-1]
        atm_reference = round(current.spot_price / 50.0) * 50.0
        selected_strike = min(self.option_strikes, key=lambda strike: abs(strike - atm_reference))
        future_basis = current.future_open - current.spot_price
        values = {
            "atm_distance": abs(selected_strike - atm_reference), "future_basis": future_basis,
            "future_basis_pct": future_basis / current.spot_price,
            "is_weekly": self.is_weekly, "is_monthly": self.is_monthly,
        }
        for window in (1, 5, 15, 30, 45, 60):
            if window > horizon: continue
            old = rows[-1-window]
            old_basis_pct = (old.future_open-old.spot_price)/old.spot_price
            values[f"spot_return_{window}m"] = current.spot_price/old.spot_price-1.0
            values[f"future_return_{window}m"] = current.future_open/old.future_open-1.0
            values[f"basis_pct_change_{window}m"] = values["future_basis_pct"]-old_basis_pct
        spot_values=np.asarray([row.spot_price for row in rows],dtype=float)
        future_values=np.asarray([row.future_open for row in rows],dtype=float)
        spot_1m=spot_values[1:]/spot_values[:-1]-1.0
        future_1m=future_values[1:]/future_values[:-1]-1.0
        for window in (5,15,30,45,60):
            if window > horizon: continue
            values[f"spot_1m_mean_{window}m"]=float(np.mean(spot_1m[-window:]))
            values[f"spot_1m_std_{window}m"]=float(np.std(spot_1m[-window:],ddof=0))
            values[f"future_1m_mean_{window}m"]=float(np.mean(future_1m[-window:]))
            values[f"future_1m_std_{window}m"]=float(np.std(future_1m[-window:],ddof=0))
        minute=current.minute
        timestamp = NSE_EPOCH + timedelta(minutes=minute)
        minute_of_day = timestamp.hour * 60 + timestamp.minute
        values["time_sin"]=math.sin(2*math.pi*minute_of_day/1440)
        values["time_cos"]=math.cos(2*math.pi*minute_of_day/1440)
        return values

    def _predict(self, minute: int, current_spot: float) -> None:
        if minute == self.last_prediction_minute:
            return
        timestamp = NSE_EPOCH + timedelta(minutes=minute)
        predictions: dict[str, dict] = {}
        for horizon, artifact in self.models.items():
            if len(self.price_history) < artifact.required_history_minutes: continue
            values=self._direct_features(horizon)
            raw=pd.DataFrame([[values.get(name,math.nan) for name in artifact.input_features]],columns=artifact.input_features)
            transformed = artifact.preprocessor.transform(raw)
            raw_prediction = float(artifact.model.predict(transformed)[0])
            predicted_return = raw_prediction / artifact.target_scale
            predictions[str(horizon)] = {
                "model": artifact.model_name,
                "predicted_return": predicted_return,
                "predicted_spot": current_spot * math.exp(predicted_return),
            }
        event = {
            "event_type": "spot_prediction",
            "exchange_minute": timestamp.isoformat(sep=" "),
            "current_spot": current_spot,
            "front_future_token": self.future_token,
            "front_future_expiry": self.future_expiry.isoformat(),
            "available_history_minutes": len(self.price_history),
            "horizon_readiness": {str(h): len(self.price_history)>=a.required_history_minutes for h,a in self.models.items()},
            "predictions": predictions,
        }
        encoded = json.dumps(event, separators=(",", ":"))
        self.output.write(encoded + "\n")
        LOGGER.info("SPOT_PREDICTION %s", encoded)
        for listener in tuple(self.prediction_listeners):
            listener(event)
        self.last_prediction_minute = minute

    def add_prediction_listener(self, listener: Callable[[dict], None]) -> None:
        self.prediction_listeners.append(listener)

    def readiness_snapshot(self) -> dict:
        available=len(self.price_history)
        return {str(h): {"required_bars":a.required_history_minutes,"available_bars":available,
                         "bars_remaining":max(0,a.required_history_minutes-available),
                         "ready":available>=a.required_history_minutes,
                         "latest_spot_minute":self.latest_spot_minute,"latest_future_minute":self.latest_future_minute,
                         "spot_pending_bars":len(self.spot_bars),"future_pending_bars":len(self.future_bars)} for h,a in self.models.items()}

    def close(self) -> None:
        self.output.close()
