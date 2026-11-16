"""Feature engineering for `dte8_20_huber_gate.joblib`.

The short-straddle paper route uses the artifact's 8-14 DTE submodel. The same
raw feature contract also describes its retained 15-20 research submodel.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from ._short_straddle_gate import GATE_FEATURES, build_gate_features


ACTIVE_DTE_BAND = "8-14"
FEATURE_NAMES = GATE_FEATURES


def build_features(atm_quote: Mapping, live_features: Mapping) -> pd.DataFrame:
    return build_gate_features(atm_quote, live_features, FEATURE_NAMES)

