import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import hybrid_profit_optimization as optimization


def test_apply_rule_enforces_sources_and_dte_floor():
    frame = pd.DataFrame({
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 12:00"]),
        "expected_net_rupees": [100.0, 200.0],
        "signal_name": [optimization.CALL, optimization.PUT],
        "days_to_expiry": [10.0, 5.0],
    })
    chosen = optimization.apply_rule(frame, (optimization.PUT,), 4)
    assert chosen["signal_name"].tolist() == [optimization.PUT]


def test_apply_rule_rejects_short_dte():
    frame = pd.DataFrame({
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 12:00"]),
        "expected_net_rupees": [200.0],
        "signal_name": [optimization.PUT],
        "days_to_expiry": [3.9],
    })
    assert optimization.apply_rule(frame, (optimization.PUT,), 4).empty
