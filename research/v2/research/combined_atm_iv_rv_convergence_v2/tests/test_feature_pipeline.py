import numpy as np
import pandas as pd
from datetime import timedelta

from feature_pipeline import (
    add_rv_prediction_convergence,
    assert_leakage_safe_features,
    combine_call_put_iv,
    construct_spot_features,
    purged_chronological_split,
    validate_temporal_integrity,
)


def _config():
    return {
        "construction": {
            "horizon_minutes": 120,
            "lag_minutes": [5, 15, 30, 60, 120],
            "annual_trading_days": 252,
            "minutes_per_trading_day": 375,
        }
    }


def test_combined_iv_is_simple_call_put_mean_and_requires_two_valid_ivs():
    call = pd.Series([0.20, 0.20, np.nan, -0.10])
    put = pd.Series([0.24, np.nan, 0.24, 0.24])
    result = combine_call_put_iv(call, put)
    assert np.isclose(result.iloc[0], 0.22)
    assert result.iloc[1:].isna().all()


def test_forward_rv_requires_exact_120_clock_minutes():
    times = pd.date_range("2026-01-02 09:15:59", periods=121, freq="min")
    spot = pd.DataFrame({"datetime": times, "spot_price": 100 * np.exp(np.arange(121) * 0.001)})
    result = construct_spot_features(spot, _config())
    assert bool(result.loc[0, "spot_window_120m_complete"])
    assert np.isfinite(result.loc[0, "target_rv_forward_120m"])
    broken = construct_spot_features(spot.drop(index=60), _config())
    assert not bool(broken.loc[0, "spot_window_120m_complete"])
    assert np.isnan(broken.loc[0, "target_rv_forward_120m"])


def test_combined_convergence_uses_only_out_of_sample_rv_prediction():
    frame = pd.DataFrame({
        "entry_atm_iv_combined": [0.22],
        "exit_fixed_atm_iv_combined": [0.18],
        "rv_pred_120m": [0.16],
        "rv_prediction_provenance": ["oof"],
    })
    result = add_rv_prediction_convergence(frame)
    assert bool(result.loc[0, "target_converged_to_rv_pred_combined"])
    assert np.isclose(result.loc[0, "target_fraction_gap_closed_combined"], 2 / 3)
    bad = frame.assign(rv_prediction_provenance="in_sample")
    try:
        add_rv_prediction_convergence(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("In-sample RV predictions must be rejected")


def test_future_columns_are_rejected_as_features():
    for column in [
        "target_fixed_combined_iv_change_120m",
        "exit_fixed_atm_iv_combined",
        "quality_exact_exit",
        "training_row_eligible",
    ]:
        try:
            assert_leakage_safe_features([column])
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {column} to be rejected")


def test_split_keeps_timestamp_groups_and_purges_label_overlap():
    times = pd.date_range("2026-01-01 09:15:59", periods=1000, freq="min")
    frame = pd.DataFrame({"datetime": np.repeat(times, 2)})
    frame["exit_datetime"] = frame["datetime"] + timedelta(minutes=120)
    parts = purged_chronological_split(frame)
    assert set(parts["train"].datetime).isdisjoint(set(parts["validation"].datetime))
    assert set(parts["validation"].datetime).isdisjoint(set(parts["test"].datetime))
    assert parts["train"].exit_datetime.max() < parts["validation"].datetime.min()
    assert parts["validation"].exit_datetime.max() < parts["test"].datetime.min()


def test_temporal_integrity_requires_same_atm_strike_and_exact_exit_time():
    frame = pd.DataFrame({
        "datetime": [pd.Timestamp("2026-01-02 10:00:59")],
        "matched_exit_timestamp": [pd.Timestamp("2026-01-02 12:00:59")],
        "entry_strike": [50000.0],
        "exit_fixed_strike": [50000.0],
        "quality_exact_exit": [True],
        "atm_strike_exact": [True],
    })
    report = validate_temporal_integrity(frame)
    assert report["entry_atm_match_rate"] == 1.0
    assert report["same_strike_exit_match_rate"] == 1.0
    assert report["exact_timestamp_match_rate"] == 1.0
