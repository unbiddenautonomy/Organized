from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from feature_pipeline import (
    add_rv_prediction_convergence,
    assert_leakage_safe_features,
    construct_spot_features,
    purged_chronological_split,
    validate_temporal_integrity,
)


def _config():
    return {
        "construction": {
            "horizon_minutes": 120, "lag_minutes": [5, 15, 30, 60, 120],
            "annual_trading_days": 252, "minutes_per_trading_day": 375,
        }
    }


def test_forward_rv_requires_exact_120_clock_minutes():
    times = pd.date_range("2026-01-02 09:15:59", periods=121, freq="min")
    spot = pd.DataFrame({"datetime": times, "spot_price": 100 * np.exp(np.arange(121) * 0.001)})
    result = construct_spot_features(spot, _config())
    assert bool(result.loc[0, "spot_window_120m_complete"])
    assert np.isfinite(result.loc[0, "target_rv_forward_120m"])
    broken = construct_spot_features(spot.drop(index=60), _config())
    assert not bool(broken.loc[0, "spot_window_120m_complete"])
    assert np.isnan(broken.loc[0, "target_rv_forward_120m"])


def test_prediction_convergence_keeps_the_entry_strike_definition():
    frame = pd.DataFrame({
        "entry_iv_call": [0.20], "exit_fixed_iv_call": [0.17],
        "entry_iv_put": [0.21], "exit_fixed_iv_put": [0.18],
        "rv_pred_120m": [0.16],
        "rv_prediction_provenance": ["oof"],
    })
    result = add_rv_prediction_convergence(frame)
    assert bool(result.loc[0, "target_converged_to_rv_pred_call"])
    assert np.isclose(result.loc[0, "target_fraction_gap_closed_call"], 0.75)


def test_future_columns_are_rejected_as_features():
    for column in ["target_rv_forward_120m", "exit_fixed_iv_call", "quality_exact_exit"]:
        try:
            assert_leakage_safe_features([column])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {column} to be rejected")


def test_split_keeps_timestamp_groups_and_purges_label_overlap():
    times = pd.date_range("2026-01-01 09:15:59", periods=1000, freq="min")
    frame = pd.DataFrame({"datetime": np.repeat(times, 2)})
    frame["exit_datetime"] = frame["datetime"] + pd.Timedelta(minutes=120)
    parts = purged_chronological_split(frame)
    assert set(parts["train"].datetime).isdisjoint(set(parts["validation"].datetime))
    assert set(parts["validation"].datetime).isdisjoint(set(parts["test"].datetime))
    assert parts["train"].exit_datetime.max() < parts["validation"].datetime.min()
    assert parts["validation"].exit_datetime.max() < parts["test"].datetime.min()


def test_temporal_integrity_requires_same_atm_strike_and_exact_exit_time():
    frame = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-01-02 10:00:59")],
        "matched_exit_timestamp": [pd.Timestamp("2026-01-02 12:00:59")],
        "entry_strike": [50000.0], "exit_fixed_strike": [50000.0],
        "quality_exact_exit": [True], "atm_strike_exact": [True],
    })
    report = validate_temporal_integrity(frame)
    assert report["entry_atm_match_rate"] == 1.0
    assert report["same_strike_exit_match_rate"] == 1.0
    assert report["exact_timestamp_match_rate"] == 1.0

