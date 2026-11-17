"""AtmFeatureStream: ticks become completed minutes, OI is kept per token, rewinds reset state."""
from datetime import date
from decimal import Decimal
from types import SimpleNamespace

from voltr.feed.messages.open_interest import OiDecodeError
from voltr.instruments.master import OptionContract
from voltr.market.atm_features import AtmFeatureStream, quadratic_surface
from voltr.market.options import greeks, implied_volatility

import numpy as np


class Recorder(AtmFeatureStream):
    def __init__(self, *args):
        super().__init__(*args)
        self.finished, self.ticks, self.oi = [], [], []

    def _finalize_minute(self, minute):
        self.finished.append(minute)

    def _on_option_tick(self, token, price, log_time):
        self.ticks.append((token, price, log_time))

    def _on_open_interest(self, token, open_interest, log_time):
        self.oi.append((token, open_interest, log_time))


def stream():
    contracts = {
        token: OptionContract(token, 0, "OPTIDX", "BANKNIFTY", date(2026, 11, 24), Decimal(strike), side)
        for token, strike, side in ((1, 55000, "CE"), (2, 55000, "PE"), (3, 55100, "CE"))
    }
    source = dict(decode_banknifty_7207=lambda message, divisor: message, implied_volatility=implied_volatility,
                  _greeks=greeks, decode_oi_message=lambda message: message, OiDecodeError=OiDecodeError)
    return Recorder({}, contracts, 30, date(2026, 11, 24), 99, source)


def tick(token, price, second):
    return SimpleNamespace(token=token, last_traded_price=price, log_time=second)


def test_pairs_only_keep_strikes_with_both_sides():
    s = stream()
    assert s.pairs == {55000.0: {"CE": 1, "PE": 2}} and s.strikes == [55000.0]


def test_minutes_finalize_two_minutes_behind_spot_and_future():
    s = stream()
    for second in (600, 660, 720, 780):
        s.on_index_message((second, 55000.0 + second))
        s.on_market_update(tick(99, 55050.0, second))
        s.on_market_update(tick(1, 300.0, second + 1))
    assert s.finished == [8, 9, 10, 11]  # each minute once, two behind the latest spot/future minute
    assert s.spot_bars[10].close == 55600.0 and s.ticks[-1] == (1, 300.0, 781)


def test_open_interest_keeps_newest_and_ignores_untracked():
    s = stream()
    s.on_oi_message([SimpleNamespace(token=1, open_interest=500, log_time=100, transaction_code=7202),
                     SimpleNamespace(token=1, open_interest=400, log_time=90, transaction_code=7202),
                     SimpleNamespace(token=77, open_interest=1, log_time=100, transaction_code=7202)])
    assert s.oi_by_token == {1: (500, 100, 7202)} and len(s.oi) == 1
    assert s._oi_value(1, 1, 15) == 500.0 and np.isnan(s._oi_value(2, 1, 15))


def test_rewind_clears_state():
    s = stream()
    s.on_index_message((6000, 55000.0))
    s.on_market_update(tick(99, 55050.0, 6000))
    s.on_index_message((60, 54000.0))
    assert list(s.spot_bars) == [1] and s.future_bars == {}


def test_quadratic_surface_recovers_a_parabola():
    x = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    level, slope, curvature, rmse, count = quadratic_surface(x, 0.15 + 0.5 * x + 3.0 * x * x, 4)
    assert abs(level - 0.15) < 1e-12 and abs(slope - 0.5) < 1e-9 and abs(curvature - 6.0) < 1e-6
    assert rmse < 1e-12 and count == 5
    assert np.isnan(quadratic_surface(x[:2], x[:2] + 1, 4)[0])
