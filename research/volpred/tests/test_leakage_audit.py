"""Focused causal-alignment tests for the leakage audit.

These tests use synthetic prices only.  They deliberately do not read any
research evaluation period or future holdout.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

import enhanced_rv_model as rv  # noqa: E402


def synthetic_spot(rows: int = 181) -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-02 09:15", periods=rows, freq="min")
    # Deterministic, non-constant returns make target-window mistakes visible.
    returns = 0.0002 + np.arange(rows, dtype=float) * 0.000001
    prices = 45_000.0 * np.exp(np.cumsum(returns))
    frame = pd.DataFrame({"datetime": timestamps, "spot_price": prices})
    frame["date"] = timestamps.normalize()
    frame["return"] = np.log(frame["spot_price"] / frame["spot_price"].shift(1))
    return frame


def build_from(frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch) -> pd.DataFrame:
    monkeypatch.setattr(rv, "prepare_spot", lambda: frame.copy(deep=True))
    return rv.build_panel()


def assert_prefix_invariant(
    builder,
    source: pd.DataFrame,
    cutoff: int,
    columns: list[str],
) -> None:
    """Reject a feature builder if changing the future changes the cutoff row."""
    baseline = builder(source.copy(deep=True))
    mutated = source.copy(deep=True)
    mutated.loc[mutated.index > cutoff, "spot_price"] *= np.linspace(
        1.2, 2.0, len(mutated) - cutoff - 1
    )
    candidate = builder(mutated)
    pd.testing.assert_series_equal(
        baseline.loc[cutoff, columns],
        candidate.loc[cutoff, columns],
        check_names=False,
        check_exact=True,
    )


def test_enhanced_rv_features_are_prefix_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    source = synthetic_spot()

    def builder(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.copy()
        frame["return"] = np.log(frame["spot_price"] / frame["spot_price"].shift(1))
        return build_from(frame, monkeypatch)

    assert_prefix_invariant(builder, source, cutoff=90, columns=rv.ENHANCED_FEATURES)


def test_future_variance_label_starts_after_decision_bar(monkeypatch: pytest.MonkeyPatch) -> None:
    source = synthetic_spot()
    panel = build_from(source, monkeypatch)
    t, horizon = 90, 15
    expected = source["return"].iloc[t + 1 : t + horizon + 1].pow(2).sum()
    assert panel.loc[t, f"future_var_{horizon}"] == pytest.approx(expected)
    assert panel.loc[t, f"future_var_{horizon}"] != pytest.approx(
        source["return"].iloc[t : t + horizon].pow(2).sum()
    )


def test_future_shift_negative_control_is_detected() -> None:
    source = synthetic_spot()

    def deliberately_leaky_builder(frame: pd.DataFrame) -> pd.DataFrame:
        # A canonical negative control: a feature shifted from t+1 onto row t.
        return pd.DataFrame({"future_price_feature": frame["spot_price"].shift(-1)})

    with pytest.raises(AssertionError):
        assert_prefix_invariant(
            deliberately_leaky_builder,
            source,
            cutoff=90,
            columns=["future_price_feature"],
        )


def test_next_fill_alignment_requires_strictly_later_available_snapshot() -> None:
    decision = pd.Timestamp("2026-01-02 09:45:00")
    snapshots = pd.DataFrame(
        {
            "datetime": pd.to_datetime(
                ["2026-01-02 09:44:59", "2026-01-02 09:45:00", "2026-01-02 09:45:59"]
            ),
            "tick_time": ["09:45:00", "09:45:01", "09:46:00"],
        }
    )
    tick = pd.to_datetime(
        snapshots["datetime"].dt.strftime("%Y-%m-%d") + " " + snapshots["tick_time"]
    )
    snapshots["available_at"] = snapshots[["datetime"]].max(axis=1)
    snapshots["available_at"] = pd.concat(
        [snapshots["available_at"], tick.rename("tick")], axis=1
    ).max(axis=1)
    eligible = snapshots[snapshots["available_at"] > decision]
    fill = eligible.sort_values("available_at").iloc[0]
    assert fill["available_at"] == pd.Timestamp("2026-01-02 09:45:01")
    assert fill["available_at"] > decision

