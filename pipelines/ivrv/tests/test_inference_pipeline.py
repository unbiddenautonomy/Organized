import numpy as np
import pandas as pd

from pipelines.ivrv.model import (
    FixedStrikeInferencePipeline,
    _future_from_log_ratio,
    _inverse_log_variance,
)


def test_transforms():
    assert np.isclose(_future_from_log_ratio(np.array([0.2]), np.array([0.0]), 2.0)[0], 0.2)
    assert np.isclose(_inverse_log_variance(np.array([np.log(0.2**2)]))[0], 0.2)


def test_saved_stack_smoke_prediction():
    pipeline = FixedStrikeInferencePipeline()

    combined_features = set()
    leg_features = set()
    rv_features = set()
    for name, payload in pipeline.artifacts.items():
        features = set(payload.get("features", []))
        if name.startswith("combined_"):
            combined_features.update(features)
        elif name.startswith(("call_", "put_")):
            leg_features.update(features)
        elif name == "rv_xgboost":
            rv_features.update(features)
        elif name == "rv_iv_sharp_decline_event":
            combined_features.update(features)

    combined = pd.DataFrame([{name: 0.0 for name in combined_features}])
    legs = pd.DataFrame([{name: 0.0 for name in leg_features}])
    rv = pd.DataFrame([{name: 0.0 for name in rv_features}])
    combined["entry_atm_iv_combined"] = 0.20
    combined["rv_trailing_60m"] = 0.198
    legs["entry_iv_call"] = 0.20
    legs["entry_iv_put"] = 0.21
    rv["entry_iv_weighted"] = 0.205
    rv["rv_trailing_120m"] = 0.15
    rv["spot_price"] = 50000.0

    result = pipeline.predict(combined, legs, rv)
    required = {
        "pred_rv_120m_xgboost",
        "pred_combined_iv_120m_lightgbm",
        "pred_convergence_probability_classifier",
        "pred_call_iv_120m_hgb",
        "pred_put_iv_120m_hgb",
        "pred_combined_iv_120m_from_legs_hgb",
        "pred_convergence_vote_fraction",
        "pred_no120_event_score",
        "pred_no120_gate_iv_near_or_below_rv",
        "pred_short_straddle_120m_signal",
    }
    assert required.issubset(result.columns)
    assert np.isfinite(result[list(required)].select_dtypes(include=["number"]).to_numpy()).all()
    assert bool(result.loc[0, "pred_strategy_rule_iv_rv_within_half_point"])
    assert bool(result.loc[0, "pred_short_straddle_120m_signal"])


def test_no120_model_outputs_its_gate_and_positive_event():
    class PositiveEventModel:
        def predict(self, matrix):
            return np.tile(np.array([[-0.03, -0.005]]), (len(matrix), 1))

    pipeline = FixedStrikeInferencePipeline()
    payload = pipeline.artifacts["rv_iv_sharp_decline_event"]
    payload["model"] = PositiveEventModel()
    frame = pd.DataFrame([{name: 0.0 for name in payload["features"]}])
    frame["entry_atm_iv_combined"] = 0.20
    frame["rv_trailing_60m"] = 0.198

    output = pd.DataFrame(index=frame.index)
    pipeline._predict_sharp_decline_event(frame, output)
    assert bool(output.loc[0, "pred_no120_sharp_rv_decline"])
    assert bool(output.loc[0, "pred_no120_gate_iv_near_or_below_rv"])

    frame["rv_trailing_60m"] = 0.19
    output = pd.DataFrame(index=frame.index)
    pipeline._predict_sharp_decline_event(frame, output)
    assert not bool(output.loc[0, "pred_no120_gate_iv_near_or_below_rv"])


def test_rising_rv_model_live_feature_mapping_and_gate():
    pipeline = FixedStrikeInferencePipeline()
    payload = pipeline.artifacts["iv_gt_rv_rising_120m"]
    frame = pd.DataFrame([{name: 1.0 for name in payload["features"]}])
    frame["datetime"] = "2026-10-29 13:00:00"
    frame["spot_price"] = 55000.0
    frame["entry_strike"] = 55000.0
    frame["entry_iv_call"] = 0.18
    frame["entry_iv_put"] = 0.20
    frame["rv_sample_30m"] = 14.0
    frame["rv_sample_60m"] = 15.0
    frame["rv_sample_120m"] = 17.0
    frame["days_to_expiry"] = 25.0
    expected = pd.DataFrame([{**{name: 1.0 for name in payload["features"]},
                              "current_iv": 19.0, "rv_30": 14.0, "rv_60": 15.0,
                              "iv_skew": 2.0, "spot_price": 55000.0, "strike": 55000.0,
                              "moneyness": 0.0, "days_to_expiry": 25.0,
                              "minute_of_day": 780, "weekday": 3}])[payload["features"]]
    output = pd.DataFrame(index=frame.index)
    pipeline._predict_rising_rv_convergence(frame, output)
    assert bool(output.loc[0, "pred_rising_rv_eligible"])
    assert np.isclose(output.loc[0, "pred_rising_rv_convergence_probability"],
                      payload["model"].predict_proba(expected)[0, 1])
    frame["rv_sample_120m"] = 20.0
    pipeline._predict_rising_rv_convergence(frame, output)
    assert not bool(output.loc[0, "pred_rising_rv_eligible"])
    assert np.isnan(output.loc[0, "pred_rising_rv_convergence_probability"])

