import numpy as np
import pandas as pd

from fixed_strike_iv.pipeline import _add_lag_features, _add_purged_splits, _model_feature_columns
from fixed_strike_iv.surface import make_entry_rows


def test_entry_uses_declared_atm_strike_and_separate_surfaces():
    strikes = np.array([90, 95, 100, 105, 110], dtype=float)
    x = np.log(strikes / 101.0)
    frame = pd.DataFrame({
        "datetime": pd.Timestamp("2026-01-02 10:00"), "dte_flag": "M1", "expiry": pd.Timestamp("2026-01-29"),
        "strike": strikes, "atm_strike": 100.0, "future_open": 101.0, "spot_price": 100.5,
        "iv_call": .20 - .1*x + .3*x*x, "iv_put": .22 - .2*x + .5*x*x,
        "iv_call_converged": True, "iv_put_converged": True, "call_price": 10.0, "put_price": 11.0,
    })
    entry = make_entry_rows(frame, 4, "future_open")
    assert entry.loc[0, "entry_strike"] == 100.0
    assert bool(entry.loc[0, "entry_atm_exact"])
    assert np.isclose(entry.loc[0, "surface_atm_iv_call"], .20)
    assert np.isclose(entry.loc[0, "surface_skew_put"], -.20)


def test_lags_only_point_backward():
    times = pd.date_range("2026-01-02 10:00", periods=4, freq="5min")
    frame = pd.DataFrame({
        "datetime": times, "Date": "2026-01-02", "dte_flag": "M1",
        "expiry": pd.Timestamp("2026-01-29"),
    })
    source_columns = [
        "entry_spot_price", "entry_future_price", "entry_iv_call", "entry_iv_put",
        "surface_atm_iv_call", "surface_atm_iv_put", "surface_skew_call", "surface_skew_put",
        "surface_smile_call", "surface_smile_put", "entry_straddle_price", "entry_call_oi", "entry_put_oi",
    ]
    for column in source_columns:
        frame[column] = np.arange(4, dtype=float)
    result = _add_lag_features(frame, [5])
    assert np.isnan(result.loc[0, "lag_5m_entry_iv_call"])
    assert result.loc[2, "lag_5m_entry_iv_call"] == result.loc[1, "entry_iv_call"]
    assert result.loc[2, "lag_5m_entry_iv_call"] != result.loc[3, "entry_iv_call"]


def test_split_purges_labels_that_cross_boundaries():
    times = pd.date_range("2026-01-02 09:15", periods=40, freq="15min")
    frame = pd.DataFrame({
        "datetime": times,
        "exit_datetime": times + pd.Timedelta(minutes=120),
        "dte_flag": "M1",
        "expiry": pd.Timestamp("2026-01-29"),
    })
    config = {
        "data": {"horizon_minutes": 120},
        "splits": {"validation_fraction": .15, "test_fraction": .15, "purge_minutes": 120},
    }
    result, report = _add_purged_splits(frame, config)
    validation_start = pd.Timestamp(report["validation_start"])
    test_start = pd.Timestamp(report["test_start"])
    assert (result.loc[result["split"] == "train", "exit_datetime"] < validation_start).all()
    assert (result.loc[result["split"] == "validation", "exit_datetime"] < test_start).all()


def test_feature_selector_rejects_future_columns():
    frame = pd.DataFrame({
        "row_id": [1], "entry_iv_call": [.2], "lag_5m_entry_iv_call": [.19],
        "exit_iv_call": [.21], "label_iv_change_call": [.01], "quality_trainable": [True],
    })
    assert _model_feature_columns(frame) == ["entry_iv_call", "lag_5m_entry_iv_call"]

