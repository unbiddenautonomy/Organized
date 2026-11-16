import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import hybrid_coverage_optimization as coverage
import hybrid_profit_optimization as profit


def sample_frame():
    return pd.DataFrame({
        "entry_datetime": pd.to_datetime([
            "2025-01-01 10:00", "2025-01-01 12:30", "2025-01-02 10:00"
        ]),
        "exit_datetime": pd.to_datetime([
            "2025-01-01 12:00", "2025-01-01 14:30", "2025-01-02 12:00"
        ]),
        "expected_net_rupees": [100.0, 200.0, 300.0],
        "signal_name": [profit.CALL, profit.PUT, profit.SHORT],
        "days_to_expiry": [3.0, 3.0, 2.0],
    })


def test_frozen_rule_keeps_near_call_and_short_but_rejects_near_put():
    chosen = coverage.apply_coverage_rule(sample_frame(), "near_expiry_only", 4, 0)
    assert chosen["signal_name"].tolist() == [profit.CALL, profit.SHORT]


def test_unknown_call_mode_is_rejected():
    try:
        coverage.apply_coverage_rule(sample_frame(), "bad", 4, 0)
    except ValueError:
        return
    raise AssertionError("unknown call mode should raise ValueError")
