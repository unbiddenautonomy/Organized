import math
from types import SimpleNamespace

import pandas as pd

from pipelines.ivrv.model import FixedStrikeInferencePipeline
from pipelines.ivrv.engine import LiveFeatureInference


def test_saved_classifier_inputs_reconstruct_both_probabilities():
    pipeline = FixedStrikeInferencePipeline()
    feature_names = {name for payload in pipeline.artifacts.values()
                     for name in payload.get("features", [])}
    row = {name: 1.0 for name in feature_names}
    row.update(datetime="2026-11-17 11:00:00", spot_price=55000.0,
               entry_strike=55000.0, entry_iv_call=0.18, entry_iv_put=0.20,
               entry_atm_iv_combined=0.19, entry_iv_weighted=0.19,
               rv_sample_30m=14.0, rv_sample_60m=15.0, rv_sample_120m=17.0,
               days_to_expiry=20.0, rv_trailing_60m=0.15,
               rv_trailing_120m=0.17)
    prediction = pipeline.predict(pd.DataFrame([row])).iloc[0].to_dict()
    engine = object.__new__(LiveFeatureInference)
    engine.pipeline = pipeline
    engine.paper_engine = SimpleNamespace(convergence_probability_threshold=0.7)

    analysis = engine._probability_analysis(row, prediction)
    for side, expected_field in (
        ("short", "pred_convergence_probability_classifier"),
        ("long", "pred_rising_rv_convergence_probability"),
    ):
        model = analysis[side]
        assert model["feature_count"] == len(pipeline.artifacts[
            "combined_convergence_classifier" if side == "short"
            else "iv_gt_rv_rising_120m"]["features"])
        assert math.isclose(model["probability"], prediction[expected_field], abs_tol=1e-12)
        assert math.isclose(model["reconstructed_probability"], model["probability"], abs_tol=1e-10)
        assert len(model["features"]) == model["feature_count"]
        assert all(feature["contribution_log_odds"] is not None for feature in model["features"])
