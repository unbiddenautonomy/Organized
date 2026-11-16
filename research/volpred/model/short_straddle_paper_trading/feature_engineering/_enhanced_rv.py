"""Shared leakage-safe feature logic for enhanced-RV horizon artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


EPS = 1e-12
PRICE_COLUMNS = ("spot_price", "close", "ltp", "price")
TIME_COLUMNS = ("datetime", "timestamp", "time", "date")


def _normalise_bars(bars: pd.DataFrame) -> pd.DataFrame:
    lower = {str(column).lower(): column for column in bars.columns}
    time_column = next((lower[name] for name in TIME_COLUMNS if name in lower), None)
    price_column = next((lower[name] for name in PRICE_COLUMNS if name in lower), None)
    if time_column is None or price_column is None:
        raise ValueError("bars need a datetime/timestamp and spot_price/close column")

    frame = bars[[time_column, price_column]].rename(
        columns={time_column: "datetime", price_column: "spot_price"}
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["spot_price"] = pd.to_numeric(frame["spot_price"], errors="coerce")
    frame = frame.dropna().loc[lambda value: value["spot_price"].gt(0)].copy()
    frame["datetime"] = frame["datetime"].dt.floor("min")
    frame = frame.drop_duplicates("datetime", keep="last").sort_values("datetime")
    if frame.empty:
        raise ValueError("bars contain no valid timestamp/price rows")
    frame["date"] = frame["datetime"].dt.date
    return frame.reset_index(drop=True)


def build_har_features(bars: pd.DataFrame, feature_names: Sequence[str]) -> pd.DataFrame:
    """Build the exact raw HAR inputs used by the deployed horizon model."""

    frame = _normalise_bars(bars)
    current_date = frame.iloc[-1]["date"]
    session = (
        frame[frame["date"].eq(current_date)]
        .set_index("datetime")
        .between_time("09:15", "15:30")
        .reset_index()
    )
    if len(session) < 61:
        raise ValueError(f"need 61 same-session prices; received {len(session)}")

    gaps = session.tail(61)["datetime"].diff().dropna()
    if not gaps.eq(pd.Timedelta(minutes=1)).all():
        raise ValueError("the latest 61-bar feature window is not one-minute contiguous")

    returns = np.log(session["spot_price"] / session["spot_price"].shift(1)).dropna()
    nonzero_fraction = float(returns.tail(60).abs().gt(1e-10).mean())
    if nonzero_fraction < 0.25:
        raise ValueError(
            f"frozen or sparse feed: only {nonzero_fraction:.1%} nonzero returns"
        )

    values = {
        f"log_rv_past_{window}": float(
            np.log(max(float(returns.tail(window).pow(2).sum()), EPS))
        )
        for window in (5, 15, 60)
    }
    missing = [name for name in feature_names if name not in values]
    if missing:
        raise ValueError(f"unsupported artifact feature contract: {missing}")
    return pd.DataFrame([[values[name] for name in feature_names]], columns=list(feature_names))


def prepare_with_frozen_bounds(
    bars: pd.DataFrame,
    artifact: Mapping,
    expected_features: Sequence[str],
) -> pd.DataFrame:
    """Build then clip raw inputs using bounds fitted into the artifact."""

    artifact_features = list(artifact["features"])
    if artifact_features != list(expected_features):
        raise ValueError(
            f"feature contract mismatch: expected {list(expected_features)}, "
            f"artifact has {artifact_features}"
        )
    raw = build_har_features(bars, expected_features)
    lower = pd.Series(artifact["clip_lower"])
    upper = pd.Series(artifact["clip_upper"])
    return raw.clip(lower=lower, upper=upper, axis=1)

