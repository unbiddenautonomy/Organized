"""The 41 causal numeric features used in training plus five expiry indicators.

Feature values use the preceding completed minute. Expiry group is known at
the actual entry timestamp. Missing minutes use at most one forward fill,
matching the historical builder. There is no backward fill or future label.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def feature_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    p = pd.DataFrame(rows).set_index("datetime").sort_index()
    p.index = pd.to_datetime(p.index)
    p = p[~p.index.duplicated(keep="last")]
    parts = []
    for day, group in p.groupby(p.index.normalize()):
        parts.append(group.reindex(pd.date_range(day + pd.Timedelta(555, unit="min"), group.index.max(), freq="min")).ffill(limit=1))
    p = pd.concat(parts).sort_index()
    def lag(s, m):
        return s.reindex(p.index - pd.Timedelta(m, unit="min")).to_numpy()
    p["straddle_premium"] = p.call_price + p.put_price
    p["pcr"] = p.put_oi / p.call_oi
    p.loc[(p.pcr <= 0) | ~np.isfinite(p.pcr), "pcr"] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        lpcr = np.log(p.pcr)
        lstraddle = np.log(p.straddle_premium.where(p.straddle_premium > 0))
        lfuture = np.log(p.future_open.where(p.future_open > 0))
        for m in (5, 15, 30, 60):
            p[f"future_log_return_{m}m"] = lfuture - lag(lfuture, m)
        for m in (5, 15, 30):
            p[f"pcr_log_change_{m}m"] = lpcr - lag(lpcr, m)
            p[f"straddle_log_change_{m}m"] = lstraddle - lag(lstraddle, m)
        p["basis"] = p.future_open / p.spot_price - 1
        p["premium_balance"] = (p.call_price - p.put_price) / p.straddle_premium
        p["moneyness"] = (p.spot_price - p.strike) / 100
        minute = p.index.hour * 60 + p.index.minute
        phase = 2 * np.pi * (minute - 555) / 375
        p["tod_sin"], p["tod_cos"] = np.sin(phase), np.cos(phase)
        p["days_to_expiry_clipped"] = p.days_to_expiry.clip(0, 45)
        p["pcr_level"] = p.pcr
        total_oi = p.call_oi + p.put_oi
        p["log_total_oi"] = np.log(total_oi.where(total_oi > 0))
        for m in (5, 15, 30):
            p[f"total_oi_log_change_{m}m"] = p.log_total_oi - lag(p.log_total_oi, m)
        squared = (lfuture - lag(lfuture, 1)).pow(2)
        for m in (15, 30, 60):
            p[f"future_realized_vol_{m}m"] = np.sqrt(squared.rolling(f"{m}min", min_periods=max(5, m // 3)).sum())
        p["future_momentum_acceleration"] = p.future_log_return_5m - p.future_log_return_30m / 6
        p["straddle_momentum_acceleration"] = p.straddle_log_change_5m - p.straddle_log_change_30m / 6
        p["straddle_to_future"] = p.straddle_premium / p.future_open
        p["iv_mean"] = (p.iv_call + p.iv_put) / 2
        p["iv_skew"] = p.iv_put - p.iv_call
        p["iv_dispersion"] = p.iv_skew.abs()
        p["theta_yield"] = -(p.theta_call + p.theta_put) / p.straddle_premium
        p["gamma_scaled"] = (p.gamma_call + p.gamma_put) * p.future_open.pow(2) / p.straddle_premium
        p["vega_scaled"] = (p.vega_call + p.vega_put) / p.straddle_premium
        p["abs_premium_balance"] = p.premium_balance.abs()
        p["abs_moneyness"] = p.moneyness.abs()
        dte = p.days_to_expiry_clipped.clip(lower=1 / 1440)
        p["sqrt_dte"], p["inverse_sqrt_dte"] = np.sqrt(dte), 1 / np.sqrt(dte)
        p["minutes_to_1500"] = np.maximum(900 - minute, 0)
        phase = 2 * np.pi * p.index.dayofweek / 5
        p["weekday_sin"], p["weekday_cos"] = np.sin(phase), np.cos(phase)
    return p.replace([np.inf, -np.inf], np.nan)


def expiry_group(dte: float) -> tuple[int, str]:
    if not math.isfinite(dte) or dte < 0:
        raise ValueError("expired or invalid contract")
    bucket = min(4, int(dte // 7))
    return bucket, "3_4" if bucket >= 3 else str(bucket)


def feature_at(rows, signal_minute, dte, names):
    stamp = pd.Timestamp(signal_minute) - pd.Timedelta(1, unit="min")
    # Even if a caller supplies later rows they cannot enter the feature table.
    past = [r for r in rows if pd.Timestamp(r["datetime"]) <= stamp]
    frame = feature_frame(past)
    if stamp not in frame.index:
        raise ValueError("preceding completed minute unavailable")
    result = frame.loc[stamp].to_dict()
    bucket, group = expiry_group(dte)
    for week in range(5):
        result[f"expiry_week_{week}"] = float(bucket == week)
    return {name: float(result[name]) for name in names}, group, stamp.isoformat()


def option_stats(price, spot, strike, years, rate, call):
    """Spot Black-Scholes, decimal IV, daily theta, vega per volatility point.

    Same units and formula as the existing feed's iv_inference helper.
    """
    missing = {"iv": math.nan, "theta": math.nan, "vega": math.nan, "gamma": math.nan}
    if min(price, spot, strike, years) <= 0:
        return missing
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))
    discounted = strike * math.exp(-rate * years)
    lower = max(0, spot - discounted) if call else max(0, discounted - spot)
    if price < lower - 1e-6 or price > (spot if call else discounted) + 1e-6:
        return missing
    lo, hi = .0001, 5.
    for _ in range(80):
        sigma = (lo + hi) / 2
        d1 = (math.log(spot / strike) + (rate + .5 * sigma * sigma) * years) / (sigma * math.sqrt(years))
        d2 = d1 - sigma * math.sqrt(years)
        value = spot * cdf(d1) - discounted * cdf(d2) if call else discounted * cdf(-d2) - spot * cdf(-d1)
        if value < price:
            lo = sigma
        else:
            hi = sigma
    sigma = (lo + hi) / 2
    d1 = (math.log(spot / strike) + (rate + .5 * sigma * sigma) * years) / (sigma * math.sqrt(years))
    d2 = d1 - sigma * math.sqrt(years)
    pdf = math.exp(-.5 * d1 * d1) / math.sqrt(2 * math.pi)
    decay = -spot * pdf * sigma / (2 * math.sqrt(years))
    theta = (decay - rate * discounted * cdf(d2)) / 365 if call else (decay + rate * discounted * cdf(-d2)) / 365
    return {"iv": sigma, "theta": theta, "gamma": pdf / (spot * sigma * math.sqrt(years)), "vega": spot * pdf * math.sqrt(years) / 100}
