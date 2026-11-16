from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research.research_v2_data import (
    P0_FEATURES,
    P1_FEATURES,
    assign_split_labels,
    build_predictive_dataset,
    verify_splits_checksum,
)


def synthetic_session(day: str, *, shock_after: str | None = None) -> pd.DataFrame:
    timestamps = pd.date_range(f"{day} 09:15:59", f"{day} 15:30:59", freq="1min")
    returns = np.linspace(-0.0003, 0.0005, len(timestamps))
    prices = 40_000.0 * np.exp(np.cumsum(returns))
    if shock_after is not None:
        after = timestamps > pd.Timestamp(f"{day} {shock_after}")
        prices[after] *= np.exp(np.linspace(0.0, 0.04, int(after.sum())))
    return pd.DataFrame({"datetime": timestamps, "spot_price": prices})


def row_at(dataset: pd.DataFrame, clock: str) -> pd.Series:
    selected = dataset[dataset["decision_time"].dt.strftime("%H:%M").eq(clock)]
    assert len(selected) == 1
    return selected.iloc[0]


def test_splits_checksum_matches_frozen_file():
    check = verify_splits_checksum()
    assert check["matched"]
    assert check["actual"] == "4409030241bf656af200c811a50a39387c49a8a4e3a4b01700b06fde1c79b48c"


def test_checksum_mismatch_fails_closed(tmp_path: Path):
    splits = tmp_path / "SPLITS.md"
    checksum = tmp_path / "SPLITS.sha256"
    splits.write_text("changed", encoding="utf-8")
    checksum.write_text("0" * 64 + " *SPLITS.md\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_splits_checksum(splits, checksum)


def test_target_is_exactly_t_plus_1_through_t_plus_120():
    source = synthetic_session("2022-06-15")
    dataset, _ = build_predictive_dataset(source)
    observed = row_at(dataset, "11:00")
    ts = pd.to_datetime(source["datetime"])
    decision_position = int(np.flatnonzero(ts.dt.strftime("%H:%M").eq("11:00"))[0])
    prices = source["spot_price"].to_numpy(float)
    returns = np.r_[np.nan, np.diff(np.log(prices))]
    expected = float(np.sum(returns[decision_position + 1 : decision_position + 121] ** 2))
    assert observed["rv_fwd_120"] == pytest.approx(expected, rel=0, abs=1e-18)
    assert observed["label_end_at"] == pd.Timestamp("2022-06-15 13:00:59", tz="Asia/Kolkata")


def test_features_are_prefix_causal_when_future_prices_change():
    base, _ = build_predictive_dataset(synthetic_session("2022-06-15"))
    changed, _ = build_predictive_dataset(synthetic_session("2022-06-15", shock_after="11:00:59"))
    before = row_at(base, "11:00")
    after = row_at(changed, "11:00")
    assert before[P1_FEATURES].to_numpy(float) == pytest.approx(after[P1_FEATURES].to_numpy(float))
    assert before["rv_fwd_120"] != pytest.approx(after["rv_fwd_120"])


def test_split_labels_use_frozen_calendar_boundaries():
    stamps = pd.Series([
        "2022-12-30 10:15:59", "2022-12-31 10:15:59", "2023-01-02 10:15:59",
        "2024-06-28 10:15:59", "2024-07-01 10:15:59", "2025-05-22 10:15:59",
        "2025-05-23 10:15:59",
    ])
    assert assign_split_labels(stamps).tolist() == [
        "development", pd.NA, "walk_forward_validation", "walk_forward_validation",
        "pseudo_holdout_previously_exposed", "pseudo_holdout_previously_exposed", pd.NA,
    ]


def test_label_end_is_purge_ready_and_features_complete():
    frames = [
        synthetic_session("2022-12-30"),
        synthetic_session("2023-01-02"),
        synthetic_session("2024-07-01"),
    ]
    dataset, audit = build_predictive_dataset(pd.concat(frames, ignore_index=True))
    assert (dataset["label_end_at"] > dataset["decision_time"]).all()
    assert (dataset["label_end_at"].dt.normalize() == dataset["decision_time"].dt.normalize()).all()
    assert ((dataset["label_end_at"] - dataset["decision_time"]) <= pd.Timedelta(seconds=120 * 61)).all()
    assert dataset[P1_FEATURES + ["rv_fwd_120", "target"]].notna().all().all()
    assert np.isfinite(dataset[P1_FEATURES + ["rv_fwd_120", "target"]].to_numpy(float)).all()
    assert set(P0_FEATURES).issubset(P1_FEATURES)
    assert set(dataset["split"]) == {
        "development", "walk_forward_validation", "pseudo_holdout_previously_exposed"
    }
    assert audit["retained_rows"] == len(dataset)


def test_gap_or_bad_price_excludes_affected_windows():
    source = synthetic_session("2022-06-15")
    source.loc[source["datetime"].eq(pd.Timestamp("2022-06-15 10:30:59")), "spot_price"] = 0.0
    dataset, audit = build_predictive_dataset(source)
    assert not dataset["decision_time"].dt.strftime("%H:%M").eq("11:00").any()
    assert audit["excluded_invalid_trailing_window"] > 0
    assert audit["excluded_invalid_target_window"] > 0
