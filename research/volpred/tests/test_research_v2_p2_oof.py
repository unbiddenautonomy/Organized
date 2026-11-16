from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.research_v2_p2 import P2_FEATURES
from research.research_v2_p2_oof import validate_and_aggregate_components
from research.research_v2_splits import cpcv_8_choose_2


def development_frame(rows: int = 8) -> pd.DataFrame:
    decision = pd.date_range("2021-01-04 10:15", periods=rows, freq="B", tz="Asia/Kolkata")
    return pd.DataFrame({
        "development_row_id": np.arange(rows), "decision_time": decision,
        "decision_date": decision.normalize(), "label_end_at": decision + pd.Timedelta(minutes=120),
        "split": "development", "rv_fwd_120": np.linspace(1.0, 2.0, rows),
    })


def components_for(frame: pd.DataFrame, count: int = 7) -> pd.DataFrame:
    rows = []
    for row_id in frame["development_row_id"]:
        for number in range(count):
            rows.append({"development_row_id": row_id, "fold_id": f"fold-{number}",
                         "component_forecast": float(row_id + number + 1),
                         "raw_calibration_failure": 0})
    return pd.DataFrame(rows)


def test_each_cpcv_group_is_held_out_in_exactly_seven_folds():
    groups, folds = cpcv_8_choose_2(pd.bdate_range("2021-01-01", periods=80))
    assert len(groups) == 8 and len(folds) == 28
    for group in range(8):
        assert sum(group in fold.test_groups for fold in folds) == 7


def test_oof_aggregation_requires_exactly_seven_unique_components_per_row():
    frame = development_frame()
    aggregated = validate_and_aggregate_components(frame, components_for(frame))
    assert aggregated["component_forecast_count"].eq(7).all()
    expected = np.array([row_id + 4.0 for row_id in frame["development_row_id"]])
    np.testing.assert_allclose(aggregated["p2_forecast"], expected)


def test_oof_aggregation_rejects_six_components_or_missing_row():
    frame = development_frame()
    with pytest.raises(AssertionError, match="exactly 7"):
        validate_and_aggregate_components(frame, components_for(frame, count=6))
    missing = components_for(frame).query("development_row_id != 0")
    with pytest.raises(AssertionError, match="no component"):
        validate_and_aggregate_components(frame, missing)


def test_oof_aggregation_rejects_duplicate_fold_component():
    frame = development_frame()
    components = components_for(frame)
    duplicated = pd.concat([components, components.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_and_aggregate_components(frame, duplicated)
