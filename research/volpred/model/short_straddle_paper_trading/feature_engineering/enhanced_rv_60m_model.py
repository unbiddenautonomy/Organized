"""Feature engineering for `enhanced_rv_60m_model.joblib`."""

from __future__ import annotations

import pandas as pd

from ._enhanced_rv import build_har_features, prepare_with_frozen_bounds


HORIZON_MINUTES = 60
FEATURE_NAMES = ("log_rv_past_5", "log_rv_past_15", "log_rv_past_60")


def build_features(bars: pd.DataFrame) -> pd.DataFrame:
    return build_har_features(bars, FEATURE_NAMES)


def prepare_for_model(bars: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    return prepare_with_frozen_bounds(bars, artifact, FEATURE_NAMES)

