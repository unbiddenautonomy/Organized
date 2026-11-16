import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import retrained_short_vol_hybrid as retrained


def test_first_threshold_crossing_uses_first_live_observation():
    frame = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 09:30", "2025-01-01 10:30", "2025-01-02 09:30"]),
        "model_probability": [0.6, 0.8, 0.7],
        "label": ["low", "high", "only"],
    })
    assert retrained.first_threshold_crossing(frame, 0.55)["label"].tolist() == ["low", "only"]


def test_combine_uses_fallback_only_on_empty_current_dates():
    current = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00"]),
    })
    fallback = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 11:00", "2025-01-02 11:00"]),
    })
    combined = retrained.combine(current, fallback)
    assert combined["date"].dt.strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-01-02"]
