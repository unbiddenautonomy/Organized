import numpy as np
import pandas as pd

from hypothesis_backtest import first_barrier, vega_weighted_iv


def test_vega_weighted_iv():
    result = vega_weighted_iv(
        pd.Series([0.2]), pd.Series([0.4]), pd.Series([3.0]), pd.Series([1.0])
    )
    assert np.isclose(result.iloc[0], 0.25)


def test_stop_is_first_actual_breach():
    path = pd.DataFrame({
        "datetime": pd.to_datetime(["2025-01-01 10:01", "2025-01-01 10:02"]),
        "call_price": [55.0, 80.0], "put_price": [55.0, 80.0],
    })
    timestamp, reason, pnl = first_barrier(path, 100.0, 25, 750.0, 1500.0, 0.0)
    assert timestamp == pd.Timestamp("2025-01-01 10:02")
    assert reason == "stop_loss"
    assert pnl == -1500.0
