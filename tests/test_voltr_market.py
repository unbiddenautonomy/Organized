"""Tests for voltr.market: option maths, minute bars and window statistics."""
import math
import random

import numpy as np

from voltr.market.bars import MinuteBar
from voltr.market.options import bs_price, greeks, implied_volatility
from voltr.market.stats import nan_window_stats


def _pcr_original_iv(price, spot, strike, years, rate, call):
    """PCR's pre-refactor inline bisection, kept here as the reference."""
    if min(price, spot, strike, years) <= 0:
        return math.nan
    cdf = lambda x: .5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    discounted = strike * math.exp(-rate * years)
    lower = max(0, spot - discounted) if call else max(0, discounted - spot)
    if price < lower - 1e-6 or price > (spot if call else discounted) + 1e-6:
        return math.nan
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
    return (lo + hi) / 2


def test_shared_iv_equals_pcr_original_bit_for_bit():
    rng = random.Random(7)
    for _ in range(400):
        spot = rng.uniform(45000, 60000)
        strike = round(spot / 100) * 100 + rng.choice((-500, -100, 0, 100, 500))
        years = rng.uniform(1, 40) / 365
        call = rng.random() < 0.5
        price = bs_price(spot, strike, years, 0.065, rng.uniform(0.08, 0.4), call) + rng.uniform(-5, 5)
        a, b = implied_volatility(price, spot, strike, years, 0.065, call), _pcr_original_iv(price, spot, strike, years, 0.065, call)
        assert (math.isnan(a) and math.isnan(b)) or a == b


def test_iv_round_trip_and_arbitrage_bounds():
    price = bs_price(55000, 55000, 10 / 365, 0.065, 0.18, True)
    assert abs(implied_volatility(price, 55000, 55000, 10 / 365, 0.065, True) - 0.18) < 1e-9
    assert math.isnan(implied_volatility(0.0, 55000, 55000, 10 / 365, 0.065, True))
    assert math.isnan(implied_volatility(60000.0, 55000, 55000, 10 / 365, 0.065, True))


def test_greeks_signs_and_missing_volatility():
    call, put = greeks(55000, 55000, 10 / 365, 0.065, 0.18, True), greeks(55000, 55000, 10 / 365, 0.065, 0.18, False)
    assert 0 < call["delta"] < 1 and -1 < put["delta"] < 0
    assert call["gamma"] == put["gamma"] > 0 and call["vega"] == put["vega"] > 0
    assert call["theta"] < 0
    assert all(math.isnan(v) for v in greeks(55000, 55000, 10 / 365, 0.065, math.nan, True).values())


def test_minute_bar_tracks_high_low_close():
    bar = MinuteBar(100.0, 100.0, 100.0, 100.0)
    for price in (101.0, 99.5, 100.25):
        bar.update(price)
    assert (bar.open, bar.high, bar.low, bar.close) == (100.0, 101.0, 99.5, 100.25)


def test_nan_window_stats_ignores_padding():
    block = np.array([[1.0, np.nan], [3.0, np.nan]])
    mean, std = nan_window_stats(block)
    assert mean[0] == 2.0 and std[0] == 1.0
    assert np.isnan(mean[1]) and np.isnan(std[1])
