"""Minute bars, open interest and the five-strike ATM feature row shared by pipelines.

AtmFeatureStream turns decoded feed updates into completed one-minute bars and,
for each finished minute, a training-compatible ATM row: IV per leg, greeks,
quadratic IV surfaces, OI, quality flags, then lagged and realised-volatility
features. IVRV scores these rows with its models; 43Weeks reads them for its
ATM diagnostics. Subclasses implement `_finalize_minute` and may override the
hooks `_mark_paper_position`, `_mark_demo_positions`, `_on_option_tick`, `_on_open_interest`, `_set_wait`.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
import logging
import math
from typing import Any

import numpy as np

from voltr.core.clock import NSE_EPOCH
from voltr.market.bars import MinuteBar

# Historic logger name, kept so existing log readers still match.
LOGGER = logging.getLogger("fixed_strike_live")
LAGS = (5, 15, 30, 60)
REQUIRED_FEATURE_MINUTES = max(LAGS) + 1
ANNUAL_MINUTES = 252.0 * 375.0


def quadratic_surface(x: np.ndarray, y: np.ndarray, minimum_points: int) -> tuple[float, float, float, float, int]:
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


class AtmFeatureStream:
    """Feed updates in, completed minutes out. See the module docstring."""

    def __init__(
        self,
        config: dict,
        contracts: dict[int, Any],
        lot_size: int,
        expiry: date,
        future_token: int,
        source: dict[str, Any],
    ) -> None:
        self.config = config
        self.contracts = contracts
        self.lot_size = lot_size
        self.expiry = expiry
        self.future_token = future_token
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

    # Hooks for subclasses.
    def _finalize_minute(self, minute: int) -> None:
        raise NotImplementedError

    def _mark_paper_position(self, exchange_second: int, **_: Any) -> None:
        """Called on every tick so a paper book can mark positions; no-op here."""

    def _mark_demo_positions(self, exchange_second: int, updated_token: int | None = None) -> None:
        """Called after `_mark_paper_position` so an observation ledger can mark; no-op here."""

    def _on_option_tick(self, token: int, price: float, log_time: int) -> None:
        """Called for each tracked option tick; no-op here."""

    def _on_open_interest(self, token: int, open_interest: int, log_time: int) -> None:
        """Called for each accepted OI update; no-op here."""

    def _set_wait(self, minute: int, reason: str, **details: Any) -> None:
        """Called when a minute cannot produce a row; no-op here."""

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
        self._mark_demo_positions(log_time)
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
            self._on_option_tick(update.token, update.last_traded_price, update.log_time)
        if update.token in self.contracts:
            self._mark_paper_position(update.log_time, updated_token=update.token)
        else:
            self._mark_paper_position(update.log_time, time_only=True)
        self._mark_demo_positions(update.log_time, updated_token=update.token)
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
            self._on_open_interest(update.token, update.open_interest, update.log_time)

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
        call_surface = quadratic_surface(np.log(strike_array / forward_array), call_iv_array, self.minimum_surface_points)
        put_surface = quadratic_surface(np.log(strike_array / forward_array), put_iv_array, self.minimum_surface_points)
        combined_surface = quadratic_surface(
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
        # Match train.py: sample standard deviation of observed one-minute log
        # returns, annualized as volatility percentage points. Require a spot
        # observation at the left edge so a partial session is never treated as
        # a complete trailing window.
        for horizon in (30, 60, 120):
            observed = [(m, self.spot_by_minute[m]) for m in range(minute - horizon, minute + 1)
                        if m in self.spot_by_minute and self.spot_by_minute[m] > 0]
            value = math.nan
            if observed and observed[0][0] <= minute - horizon + 2 and observed[-1][0] == minute:
                prices = np.asarray([price for _, price in observed], dtype=float)
                returns = np.diff(np.log(prices))
                if len(returns) >= (60 if horizon == 120 else 2):
                    value = float(np.std(returns, ddof=1) * math.sqrt(ANNUAL_MINUTES) * 100.0)
            output[f"rv_sample_{horizon}m"] = value
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
