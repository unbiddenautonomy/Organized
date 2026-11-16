"""Feature engineering for `dte4_7_pnl_gate.joblib`."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ._short_straddle_gate import GATE_FEATURES, build_gate_features


DTE_BAND = "4-7"
FEATURE_NAMES = GATE_FEATURES


def build_features(atm_quote: Mapping, live_features: Mapping) -> pd.DataFrame:
    return build_gate_features(atm_quote, live_features, FEATURE_NAMES)

