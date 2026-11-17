"""European option maths on spot: price, implied volatility and greeks.

Units: volatility as a decimal, vega per volatility point, theta per calendar day.
"""
from __future__ import annotations

import math


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def bs_price(spot: float, strike: float, years: float, rate: float, sigma: float, is_call: bool) -> float:
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    discounted = strike * math.exp(-rate * years)
    if is_call:
        return spot * normal_cdf(d1) - discounted * normal_cdf(d2)
    return discounted * normal_cdf(-d2) - spot * normal_cdf(-d1)


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
        if bs_price(spot, strike, years, rate, mid, is_call) < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def greeks(spot: float, strike: float, years: float, rate: float, sigma: float, is_call: bool) -> dict[str, float]:
    if not math.isfinite(sigma) or sigma <= 0 or years <= 0:
        return {name: math.nan for name in ("delta", "gamma", "vega", "theta")}
    root_t = math.sqrt(years)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * years) / (sigma * root_t)
    d2 = d1 - sigma * root_t
    pdf = normal_pdf(d1)
    delta = normal_cdf(d1) if is_call else normal_cdf(d1) - 1.0
    gamma = pdf / (spot * sigma * root_t)
    vega = spot * pdf * root_t / 100.0
    decay = -(spot * pdf * sigma) / (2.0 * root_t)
    if is_call:
        theta = (decay - rate * strike * math.exp(-rate * years) * normal_cdf(d2)) / 365.0
    else:
        theta = (decay + rate * strike * math.exp(-rate * years) * normal_cdf(-d2)) / 365.0
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
