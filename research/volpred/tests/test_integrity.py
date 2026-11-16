import pandas as pd
import pytest

from research.integrity import (
    current_row_iv_quality_flags,
    filter_primary_session,
    localize_asia_kolkata,
    match_next_snapshot,
    option_available_at,
    session_flags,
    spot_integrity_flags,
    valid_spot_windows,
)


def test_localization_and_available_at_use_later_tick_time():
    frame = pd.DataFrame(
        {
            "datetime": ["2025-01-02 09:15:59", "2025-01-02 09:16:00", "bad"],
            "Date": ["2025-01-02", "2025-01-02", "2025-01-02"],
            "tick_time": ["09:16:00", "09:15:59", None],
        }
    )
    available = option_available_at(frame)
    assert str(available.dt.tz) == "Asia/Kolkata"
    assert available.iloc[0] == pd.Timestamp("2025-01-02 09:16:00", tz="Asia/Kolkata")
    assert available.iloc[1] == pd.Timestamp("2025-01-02 09:16:00", tz="Asia/Kolkata")
    assert pd.isna(available.iloc[2])
    utc = pd.Series([pd.Timestamp("2025-01-02 03:45:00", tz="UTC")])
    assert localize_asia_kolkata(utc).iloc[0].hour == 9


def test_primary_session_excludes_out_of_hours_and_special_session():
    timestamps = pd.Series(
        [
            "2025-01-03 09:15:00",  # Friday
            "2025-01-03 15:30:59",
            "2025-01-03 15:31:00",
            "2025-01-04 10:00:00",  # recorded Saturday diagnostic
        ]
    )
    flags = session_flags(timestamps)
    assert flags["is_primary_session"].tolist() == [True, True, False, False]
    assert flags["is_special_session"].tolist() == [False, False, False, True]
    frame = pd.DataFrame({"datetime": timestamps, "value": range(4)})
    assert filter_primary_session(frame)["value"].tolist() == [0, 1]


def test_spot_flags_do_not_forward_fill_nonpositive_values_or_gaps():
    spot = pd.DataFrame(
        {
            "datetime": [
                "2025-01-03 09:15:00",
                "2025-01-03 09:16:00",
                "2025-01-03 09:17:00",
                "2025-01-03 09:20:00",
            ],
            "spot_price": [45000.0, 0.0, 45010.0, 45020.0],
        }
    )
    flags = spot_integrity_flags(spot)
    assert flags["spot_row_usable"].tolist() == [True, False, True, False]

    starts = pd.Series(
        ["2025-01-03 09:15:00", "2025-01-03 09:17:00", "2025-01-03 09:18:00", "2025-01-03 09:20:00"]
    )
    ends = pd.Series(
        ["2025-01-03 09:17:00", "2025-01-03 09:20:00", "2025-01-03 09:20:00", "2025-01-03 09:20:00"]
    )
    assert valid_spot_windows(starts, ends, spot).tolist() == [False, False, False, True]


def test_iv_quality_is_current_row_only_and_side_specific():
    frame = pd.DataFrame(
        {
            "iv_call": [0.2, 0.0, 0.3],
            "iv_call_converged": [True, True, "false"],
            "iv_put": [0.21, 0.22, float("nan")],
            "iv_put_converged": ["TRUE", False, True],
        }
    )
    flags = current_row_iv_quality_flags(frame)
    assert flags["call_iv_usable"].tolist() == [True, False, False]
    assert flags["put_iv_usable"].tolist() == [True, False, False]
    assert flags["all_iv_usable"].tolist() == [True, False, False]


def test_execution_uses_strictly_next_same_session_snapshot_by_contract():
    decisions = pd.DataFrame(
        {
            "decision_time": ["2025-01-03 10:00:00", "2025-01-03 15:30:00"],
            "strike": [45000, 45000],
        }
    )
    quotes = pd.DataFrame(
        {
            "available_at": [
                "2025-01-03 10:00:00",  # signal-time quote must not execute
                "2025-01-03 10:00:01",
                "2025-01-06 09:15:00",  # must not cross the session
            ],
            "strike": [45000, 45000, 45000],
            "call_price": [100.0, 101.0, 102.0],
        }
    )
    matched = match_next_snapshot(decisions, quotes, by=["strike"])
    assert matched.loc[0, "execution_call_price"] == 101.0
    assert matched.loc[0, "execution_available_at"] > matched.loc[0, "decision_time"]
    assert not matched.loc[1, "execution_matched"]


def test_execution_rejects_ambiguous_duplicate_snapshots():
    decisions = pd.DataFrame({"decision_time": ["2025-01-03 10:00:00"]})
    quotes = pd.DataFrame(
        {"available_at": ["2025-01-03 10:01:00"] * 2, "price": [1.0, 2.0]}
    )
    with pytest.raises(ValueError, match="not unique"):
        match_next_snapshot(decisions, quotes)


def test_execution_does_not_use_out_of_hours_quote():
    decisions = pd.DataFrame({"decision_time": ["2025-01-03 15:30:59"]})
    quotes = pd.DataFrame(
        {"available_at": ["2025-01-03 15:31:00"], "price": [1.0]}
    )
    matched = match_next_snapshot(decisions, quotes)
    assert not matched.loc[0, "execution_matched"]
