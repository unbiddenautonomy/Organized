from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))

from research_v2_splits import (  # noqa: E402
    DEVELOPMENT,
    PSEUDO_HOLDOUT,
    WALK_FORWARD,
    assert_calendar_partition,
    assert_purge_embargo,
    calendar_split,
    cpcv_8_choose_2,
    purged_embargoed_train,
    read_restricted_parquet,
    restricted_view,
    split_protocol_hash,
)


TZ = "Asia/Kolkata"


def test_frozen_calendar_boundaries_are_exact() -> None:
    dates = pd.to_datetime([
        "2019-10-10", "2019-10-11", "2022-12-30", "2022-12-31",
        "2023-01-02", "2024-06-28", "2024-06-29", "2024-07-01",
        "2025-05-22", "2025-05-23",
    ])
    assert calendar_split(dates).tolist() == [
        "excluded", DEVELOPMENT, DEVELOPMENT, "excluded", WALK_FORWARD,
        WALK_FORWARD, "excluded", PSEUDO_HOLDOUT, PSEUDO_HOLDOUT, "excluded",
    ]
    frame = pd.DataFrame({"decision_time": dates.tz_localize(TZ)})
    frame["split"] = calendar_split(frame["decision_time"]).to_numpy()
    assert_calendar_partition(frame)
    frame.loc[1, "split"] = WALK_FORWARD
    with pytest.raises(ValueError, match="calendar split mismatch"):
        assert_calendar_partition(frame)


def test_cpcv_is_chronological_balanced_8_choose_2() -> None:
    sessions = pd.bdate_range("2022-01-03", periods=83)
    groups, folds = cpcv_8_choose_2(sessions)
    assert len(groups) == 8
    assert len(folds) == 28
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    assert tuple(date for group in groups for date in group) == tuple(sessions)
    assert all(len(fold.test_groups) == 2 and len(fold.train_groups) == 6 for fold in folds)
    assert len({fold.test_groups for fold in folds}) == 28


def synthetic_rows() -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    sessions = pd.bdate_range("2022-01-03", periods=16)
    rows = []
    for date in sessions:
        decision = pd.Timestamp(date.date()).tz_localize(TZ) + pd.Timedelta(hours=10)
        rows.append({"decision_time": decision, "label_end_at": decision + pd.Timedelta(minutes=120)})
    return pd.DataFrame(rows), sessions


def test_purge_removes_overlap_and_one_full_following_session() -> None:
    frame, sessions = synthetic_rows()
    test_dates = [sessions[4], sessions[5], sessions[10], sessions[11]]
    # Make the preceding row's label overlap the first held-out block.
    frame.loc[3, "label_end_at"] = frame.loc[4, "decision_time"]
    train = purged_embargoed_train(frame, test_dates=test_dates, all_sessions=sessions)
    remaining = set(train["decision_time"].dt.tz_localize(None).dt.normalize())
    assert sessions[3] not in remaining  # label overlap purge
    assert sessions[6] not in remaining  # full session after first block
    assert sessions[12] not in remaining  # full session after second block
    assert not remaining.intersection(test_dates)


def test_guard_detects_overlap_and_embargo_violations() -> None:
    frame, sessions = synthetic_rows()
    test = frame.iloc[[4, 5]].copy()
    bad_overlap = frame.iloc[[3]].copy()
    bad_overlap.loc[:, "label_end_at"] = test["decision_time"].min()
    with pytest.raises(AssertionError, match="label overlaps"):
        assert_purge_embargo(bad_overlap, test, all_sessions=sessions)
    bad_embargo = frame.iloc[[6]].copy()
    with pytest.raises(AssertionError, match="embargo session"):
        assert_purge_embargo(bad_embargo, test, all_sessions=sessions)


def test_pseudo_holdout_is_denied_to_research_views() -> None:
    frame = pd.DataFrame({
        "split": [DEVELOPMENT, WALK_FORWARD, PSEUDO_HOLDOUT],
        "feature": [1.0, 2.0, 999.0],
        "target": [0.1, 0.2, 999.0],
    })
    safe = restricted_view(
        frame, allowed_splits=[DEVELOPMENT], columns=["feature", "target"], context="CPCV"
    )
    assert safe["feature"].tolist() == [1.0]
    with pytest.raises(PermissionError, match="unapproved split"):
        restricted_view(
            frame, allowed_splits=[PSEUDO_HOLDOUT], columns=["target"], context="model selection"
        )


def test_parquet_reader_pushes_down_the_split_restriction(tmp_path: Path) -> None:
    import duckdb

    path = tmp_path / "panel.parquet"
    source = pd.DataFrame({
        "split": [DEVELOPMENT, WALK_FORWARD, PSEUDO_HOLDOUT],
        "feature": [1.0, 2.0, 999.0],
        "target": [0.1, 0.2, 999.0],
    })
    connection = duckdb.connect()
    connection.register("source", source)
    connection.execute("COPY source TO ? (FORMAT PARQUET)", [str(path)])
    connection.close()
    safe = read_restricted_parquet(
        path, allowed_splits=[DEVELOPMENT, WALK_FORWARD],
        columns=["feature", "target"], context="predictive research",
    )
    assert safe["feature"].tolist() == [1.0, 2.0]
    with pytest.raises(PermissionError):
        read_restricted_parquet(
            path, allowed_splits=[PSEUDO_HOLDOUT], columns=["target"], context="premature open"
        )


def test_split_protocol_checksum_is_still_frozen() -> None:
    expected = (ROOT / "SPLITS.sha256").read_text(encoding="ascii").strip().split()[0]
    assert split_protocol_hash(ROOT / "SPLITS.md") == expected
