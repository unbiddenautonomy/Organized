import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from train import TARGET, add_target, feature_lookback
from refine import add_engineered_features


def test_target_requires_both_sharp_and_relative_decline():
    frame = pd.DataFrame({
        "rv_trailing_120m": [0.20, 0.20, 0.20],
        "target_rv_forward_120m": [0.16, 0.19, 0.16],
        "target_fixed_combined_iv_change_120m": [-0.01, 0.01, -0.035],
    })
    result = add_target(frame, minimum_rv_drop=0.02, relative_margin=0.01)
    assert result[TARGET].tolist() == [True, False, False]


def test_feature_lookback_parsing():
    assert feature_lookback("rv_trailing_60m") == 60
    assert feature_lookback("entry_atm_iv_combined_change_120m") == 120
    assert feature_lookback("time_sin") == 0


def test_engineered_iv_rv_features_are_entry_time_calculations():
    frame = pd.DataFrame({
        "entry_atm_iv_combined": [0.20],
        "rv_trailing_60m": [0.15],
        "entry_straddle_price": [100.0],
        "spot_price": [50000.0],
        "call_oi": [200.0],
        "put_oi": [300.0],
    })
    result, created = add_engineered_features(frame)
    assert "engineered_iv_minus_rv_60m" in created
    assert np.isclose(result.loc[0, "engineered_iv_minus_rv_60m"], 0.05)
    assert np.isclose(result.loc[0, "engineered_put_call_oi_ratio"], 1.5)


def test_no120_winner_has_no_120_minute_input_features():
    model_path = Path(__file__).parent / "artifacts_no120_accuracy" / "winner.joblib"
    if not model_path.exists():
        return
    bundle = joblib.load(model_path)
    assert bundle["lookback_minutes"] == 60
    assert not [name for name in bundle["features"] if "120m" in name.lower()]
    assert bundle["threshold"] == 0.0
