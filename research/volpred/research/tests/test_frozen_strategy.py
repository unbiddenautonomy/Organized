import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frozen_near_expiry_strategy import lot_size, transaction_cost


def _trade():
    return pd.Series({"date":pd.Timestamp("2024-06-01"), "call_mid":300., "put_mid":300.,
                      "exit_call_mid":250., "exit_put_mid":250.,
                      "expiry":pd.Timestamp("2024-06-27")})


def test_historical_lot_regimes():
    assert lot_size(pd.Timestamp("2023-06-30")) == 25
    assert lot_size(pd.Timestamp("2023-07-01")) == 15
    assert lot_size(pd.Timestamp("2025-01-30")) == 15
    assert lot_size(pd.Timestamp("2025-02-27")) == 30


def test_wider_spread_strictly_reduces_short_straddle_net_pnl():
    narrow = transaction_cost(_trade(), 10)[2]
    base = transaction_cost(_trade(), 25)[2]
    wide = transaction_cost(_trade(), 50)[2]
    assert narrow > base > wide
