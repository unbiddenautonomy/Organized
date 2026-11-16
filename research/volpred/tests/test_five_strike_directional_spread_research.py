import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import five_strike_directional_spread_research as directional


def test_directional_structures_are_same_side_defined_risk_spreads():
    assert directional.STRUCTURES["bull_call_spread_2"] == {
        (0, "call"): 1, (2, "call"): -1
    }
    assert directional.STRUCTURES["bear_put_spread_2"] == {
        (0, "put"): 1, (-2, "put"): -1
    }
    assert directional.STRUCTURES["bull_call_plus_put_2"] == {
        (0, "call"): 1, (-2, "put"): 1
    }


def test_choose_direction_uses_only_matching_side():
    panel = pd.DataFrame({
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 10:15", "2025-01-01 10:15"]),
        "structure": ["bull_call_spread_1", "bear_put_spread_1"],
        "signed_return": [0.01, 0.01],
        "predicted_return": [0.002, 0.002],
        "expected_edge_gate": [True, True],
    })
    chosen = directional.choose_direction(panel, "debit_spread_1", "forecast_sign", True)
    assert chosen["structure"].tolist() == ["bull_call_spread_1"]


def test_session_minute_convention_accepts_bar_seconds_only():
    times = pd.to_datetime(["2025-01-01 15:30:59", "2025-01-01 18:30:59"])
    minute = times.hour * 60 + times.minute
    assert minute[0] == 15 * 60 + 30
    assert not (9 * 60 + 15 <= minute[1] <= 15 * 60 + 30)
