"""Option pricing, variance-clock and execution primitives for research only."""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt
from typing import Iterable

import numpy as np
from scipy.special import ndtr


EPS = 1e-12


def _cdf(x):
    return ndtr(np.asarray(x, float))


def _pdf(x):
    return np.exp(-0.5 * np.asarray(x, float) ** 2) / sqrt(2.0 * pi)


def parity_forward(strike, call_price, put_price, rate: float, tau):
    """Forward implied by put-call parity with configurable discounting."""
    return np.asarray(strike, float) + np.exp(rate * np.asarray(tau, float)) * (
        np.asarray(call_price, float) - np.asarray(put_price, float)
    )


def black_price(forward, strike, tau, sigma, is_call: bool, rate: float = 0.0):
    f, k, t, v = np.broadcast_arrays(
        np.asarray(forward, float), np.asarray(strike, float),
        np.asarray(tau, float), np.asarray(sigma, float),
    )
    t = np.maximum(t, EPS); v = np.maximum(v, EPS)
    root = np.sqrt(t)
    d1 = (np.log(f / k) + 0.5 * v * v * t) / (v * root)
    d2 = d1 - v * root
    undiscounted = f * _cdf(d1) - k * _cdf(d2) if is_call else k * _cdf(-d2) - f * _cdf(-d1)
    return np.exp(-rate * t) * undiscounted


def bs_price(spot, strike, tau, sigma, is_call: bool, rate: float = 0.0, carry: float = 0.0):
    t = np.asarray(tau, float)
    forward = np.asarray(spot, float) * np.exp((rate - carry) * t)
    return black_price(forward, strike, t, sigma, is_call, rate)


def bs_greeks(spot, strike, tau, sigma, is_call: bool, rate: float = 0.0, carry: float = 0.0):
    s, k, t, v = np.broadcast_arrays(
        np.asarray(spot, float), np.asarray(strike, float),
        np.asarray(tau, float), np.asarray(sigma, float),
    )
    t = np.maximum(t, EPS); v = np.maximum(v, EPS); root = np.sqrt(t)
    d1 = (np.log(s / k) + (rate - carry + 0.5 * v * v) * t) / (v * root)
    d2 = d1 - v * root
    disc_q = np.exp(-carry * t)
    delta = disc_q * (_cdf(d1) if is_call else _cdf(d1) - 1.0)
    gamma = disc_q * _pdf(d1) / (s * v * root)
    vega = s * disc_q * _pdf(d1) * root
    theta_year = (
        -s * disc_q * _pdf(d1) * v / (2.0 * root)
        - rate * k * np.exp(-rate * t) * (_cdf(d2) if is_call else _cdf(-d2))
        + carry * s * disc_q * (_cdf(d1) if is_call else _cdf(-d1))
    )
    vanna = -disc_q * _pdf(d1) * d2 / v
    volga = vega * d1 * d2 / v
    return {"delta": delta, "gamma": gamma, "vega": vega,
            "theta_day": theta_year / 365.0, "vanna": vanna, "volga": volga}


def adjusted_iv_variance(sigma: float, omega: float, minute_profile: np.ndarray,
                         start_minute: int, horizon: int) -> float:
    """Horizon implied variance on the trading-day clock requested by the spec."""
    profile = np.asarray(minute_profile, float)
    if len(profile) != 375 or not np.isclose(profile.sum(), 1.0, atol=1e-8):
        raise ValueError("minute_profile must contain 375 nonnegative weights summing to one")
    lo = max(0, int(start_minute) + 1)
    hi = min(375, lo + int(horizon))
    mass = float(profile[lo:hi].sum())
    return float(sigma * sigma / 252.0 * omega * mass)


def bs_consistent_theta_window(gamma: float, spot: float, iv_variance: float) -> float:
    """Long-option theta contribution over a window under the unified clock."""
    return float(-0.5 * gamma * spot * spot * iv_variance)


def modeled_fill(mid: float, signed_qty: float, entering: bool, half_spread_bps: float) -> float:
    """Long buys/short covers pay ask; long sells/short entries receive bid."""
    buy = (signed_qty > 0) == entering
    side = 1.0 if buy else -1.0
    return float(mid * (1.0 + side * half_spread_bps / 10000.0))


@dataclass(frozen=True)
class PriceState:
    spot: float
    tau: float
    rate: float = 0.0
    carry: float = 0.0


def full_repricing_pnl(legs: Iterable[dict], entry: PriceState, exit: PriceState,
                       exit_vols: dict[tuple, float], lot_size: int,
                       half_spread_bps: float) -> float:
    """Full nonlinear P&L; Greeks are deliberately absent from this calculation."""
    total = 0.0
    for leg in legs:
        key = (leg["expiry"], leg["strike"], leg["option_type"])
        call = leg["option_type"] == "call"
        qty = float(leg["qty"])
        entry_mid = float(leg.get("entry_mid", bs_price(entry.spot, leg["strike"], entry.tau,
                                                        leg["iv"], call, entry.rate, entry.carry)))
        exit_mid = float(bs_price(exit.spot, leg["strike"], exit.tau, exit_vols[key],
                                  call, exit.rate, exit.carry))
        entry_fill = modeled_fill(entry_mid, qty, True, half_spread_bps)
        exit_fill = modeled_fill(exit_mid, qty, False, half_spread_bps)
        total += qty * (exit_fill - entry_fill) * lot_size
    return float(total)

