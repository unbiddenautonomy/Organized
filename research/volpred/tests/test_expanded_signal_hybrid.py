import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import expanded_signal_hybrid as expanded


def test_add_on_empty_dates_preserves_base_and_adds_only_unused_dates():
    base = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 09:30", "2025-01-01 12:00"]),
        "source": ["base", "base"],
    })
    candidate = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-02 10:00"]),
        "source": ["candidate", "candidate"],
    })
    result = expanded.add_on_empty_dates(base, candidate)
    assert result["source"].tolist() == ["base", "base", "candidate"]
    assert result["date"].nunique() == 2


def test_integrity_audit_detects_no_violations_for_valid_windows():
    trades = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 09:15", "2025-01-01 12:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 11:15", "2025-01-01 14:00"]),
        "horizon_min": [120, 120],
        "expiry": pd.to_datetime(["2025-01-30", "2025-01-30"]),
    })
    audit = expanded.integrity_audit(trades)
    assert audit["passed"].all()
    assert audit["violations"].sum() == 0


def test_integrity_audit_flags_overlap_and_session_breach():
    trades = pd.DataFrame({
        "date": pd.to_datetime(["2025-01-01", "2025-01-01"]),
        "entry_datetime": pd.to_datetime(["2025-01-01 14:00", "2025-01-01 15:00"]),
        "exit_datetime": pd.to_datetime(["2025-01-01 16:00", "2025-01-01 15:30"]),
        "horizon_min": [120, 30],
        "expiry": pd.to_datetime(["2025-01-30", "2025-01-30"]),
    })
    audit = expanded.integrity_audit(trades).set_index("check")
    assert audit.loc["outside_0915_1530", "violations"] == 1
    assert audit.loc["overlapping_positions", "violations"] == 1
