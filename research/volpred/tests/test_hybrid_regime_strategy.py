import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import hybrid_regime_strategy as hybrid


def test_portfolio_chooses_larger_edge_at_same_timestamp():
    frame = pd.DataFrame({
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 12:00"]),
        "expected_net_rupees": [100.0, 200.0],
        "signal_name": ["lower", "higher"],
    })
    chosen = hybrid.portfolio_nonoverlap(frame)
    assert chosen["signal_name"].tolist() == ["higher"]


def test_portfolio_rejects_overlapping_trade():
    frame = pd.DataFrame({
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 11:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 12:00", "2025-01-01 13:00"]),
        "expected_net_rupees": [100.0, 300.0],
        "signal_name": ["first", "overlap"],
    })
    chosen = hybrid.portfolio_nonoverlap(frame)
    assert chosen["signal_name"].tolist() == ["first"]
