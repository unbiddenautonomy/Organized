"""Shared causal feature logic for the ATM short-straddle decision gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


GATE_FEATURES = (
    "dte",
    "log_premium_ratio",
    "atm_iv",
    "iv_skew",
    "log_rv_past_5",
    "log_rv_past_15",
    "log_rv_past_60",
    "overnight_return",
    "vol_of_vol_30",
    "oi_imbalance",
)
LIVE_OR_QUOTE_FEATURES = (
    "log_rv_past_5",
    "log_rv_past_15",
    "log_rv_past_60",
    "overnight_return",
    "vol_of_vol_30",
)


def _number(source: Mapping, name: str, default: float = np.nan) -> float:
    value = source.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_gate_features(
    atm_quote: Mapping,
    live_features: Mapping,
    feature_names: Sequence[str],
) -> pd.DataFrame:
    """Build the exact ten raw inputs expected by a stored gate pipeline."""

    if tuple(feature_names) != GATE_FEATURES:
        raise ValueError(f"unexpected short-straddle feature contract: {list(feature_names)}")

    spot = _number(atm_quote, "spot_price")
    call_price = _number(atm_quote, "call_price")
    put_price = _number(atm_quote, "put_price")
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot_price must be finite and positive")
    if not np.isfinite(call_price + put_price) or call_price + put_price <= 0:
        raise ValueError("ATM call_price + put_price must be finite and positive")

    iv_call = _number(atm_quote, "iv_call")
    iv_put = _number(atm_quote, "iv_put")
    call_oi = _number(atm_quote, "call_oi", 0.0)
    put_oi = _number(atm_quote, "put_oi", 0.0)
    if not np.isfinite(call_oi):
        call_oi = 0.0
    if not np.isfinite(put_oi):
        put_oi = 0.0
    oi_total = call_oi + put_oi

    lagged = {}
    for name in LIVE_OR_QUOTE_FEATURES:
        value = _number(live_features, name)
        lagged[name] = value if np.isfinite(value) else _number(atm_quote, name)

    values = {
        "dte": _number(atm_quote, "days_to_expiry", _number(atm_quote, "dte")),
        "log_premium_ratio": float(np.log((call_price + put_price) / spot)),
        "atm_iv": float(np.nanmean([iv_call, iv_put])),
        "iv_skew": iv_put - iv_call,
        **lagged,
        "oi_imbalance": float((put_oi - call_oi) / max(oi_total, 1.0)),
    }
    row = pd.DataFrame([[values[name] for name in feature_names]], columns=list(feature_names))
    invalid = [name for name in feature_names if not np.isfinite(row.iloc[0][name])]
    if invalid:
        raise ValueError(f"missing or non-finite short-straddle features: {invalid}")
    return row

