"""Fail-closed split and leakage guards for Research V2.

This module contains no model or outcome logic.  It operates on timestamps,
session dates, split labels, and column names so it can be tested without
opening the pseudo-holdout targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEVELOPMENT = "development"
WALK_FORWARD = "walk_forward_validation"
PSEUDO_HOLDOUT = "pseudo_holdout_previously_exposed"
ALLOWED_RESEARCH_SPLITS = frozenset({DEVELOPMENT, WALK_FORWARD})

DEVELOPMENT_START = pd.Timestamp("2019-10-11")
DEVELOPMENT_END = pd.Timestamp("2022-12-30")
WALK_FORWARD_START = pd.Timestamp("2023-01-02")
WALK_FORWARD_END = pd.Timestamp("2024-06-28")
PSEUDO_HOLDOUT_START = pd.Timestamp("2024-07-01")
PSEUDO_HOLDOUT_END = pd.Timestamp("2025-05-22")


def _dates(values: Iterable[object]) -> pd.Series:
    if isinstance(values, (set, frozenset)):
        values = sorted(values)
    series = pd.Series(values, copy=False)
    parsed = pd.to_datetime(series, errors="raise")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return parsed.dt.normalize()


def calendar_split(values: Iterable[object]) -> pd.Series:
    """Assign only the immutable calendar labels; outside dates are excluded."""
    dates = _dates(values)
    labels = np.select(
        [
            dates.between(DEVELOPMENT_START, DEVELOPMENT_END),
            dates.between(WALK_FORWARD_START, WALK_FORWARD_END),
            dates.between(PSEUDO_HOLDOUT_START, PSEUDO_HOLDOUT_END),
        ],
        [DEVELOPMENT, WALK_FORWARD, PSEUDO_HOLDOUT],
        default="excluded",
    )
    return pd.Series(labels, index=dates.index, name="split")


def assert_calendar_partition(frame: pd.DataFrame) -> None:
    required = {"decision_time", "split"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing split columns: {sorted(missing)}")
    expected = calendar_split(frame["decision_time"])
    actual = frame["split"].astype(str).reset_index(drop=True)
    expected = expected.reset_index(drop=True)
    mismatch = actual.ne(expected)
    if mismatch.any():
        locations = mismatch[mismatch].index[:5].tolist()
        raise ValueError(f"calendar split mismatch at rows {locations}")


def assert_no_pseudo_holdout(frame: pd.DataFrame, *, context: str) -> None:
    if "split" not in frame:
        raise ValueError("split column is required")
    labels = set(frame["split"].dropna().astype(str).unique())
    if PSEUDO_HOLDOUT in labels:
        raise PermissionError(f"{context} may not access {PSEUDO_HOLDOUT}")


def restricted_view(
    frame: pd.DataFrame,
    *,
    allowed_splits: Sequence[str],
    columns: Sequence[str] | None = None,
    context: str,
) -> pd.DataFrame:
    """Return an approved view and refuse pseudo-holdout requests.

    Production research should apply the same split predicate while reading
    Parquet so pseudo-holdout columns are never materialized in the process.
    """
    requested = frozenset(map(str, allowed_splits))
    if not requested or not requested <= ALLOWED_RESEARCH_SPLITS:
        raise PermissionError(f"unapproved split request for {context}: {sorted(requested)}")
    if "split" not in frame:
        raise ValueError("split column is required")
    selected_columns = list(frame.columns if columns is None else columns)
    if "split" not in selected_columns:
        selected_columns.append("split")
    missing = set(selected_columns) - set(frame.columns)
    if missing:
        raise ValueError(f"missing requested columns: {sorted(missing)}")
    out = frame.loc[frame["split"].isin(requested), selected_columns].copy()
    assert_no_pseudo_holdout(out, context=context)
    return out


def read_restricted_parquet(
    path: Path,
    *,
    allowed_splits: Sequence[str],
    columns: Sequence[str],
    context: str,
) -> pd.DataFrame:
    """Read only approved split rows using Parquet predicate pushdown."""
    requested = frozenset(map(str, allowed_splits))
    if not requested or not requested <= ALLOWED_RESEARCH_SPLITS:
        raise PermissionError(f"unapproved split request for {context}: {sorted(requested)}")
    if not columns:
        raise ValueError("at least one column is required")
    # Quote identifiers rather than accepting SQL expressions from callers.
    quoted = [f'"{str(column).replace(chr(34), chr(34) * 2)}"' for column in columns]
    if '"split"' not in quoted:
        quoted.append('"split"')
    placeholders = ",".join("?" for _ in requested)
    query = f"SELECT {','.join(quoted)} FROM read_parquet(?) WHERE split IN ({placeholders})"
    import duckdb

    connection = duckdb.connect()
    try:
        out = connection.execute(query, [str(path), *sorted(requested)]).fetchdf()
    finally:
        connection.close()
    assert_no_pseudo_holdout(out, context=context)
    return out


@dataclass(frozen=True)
class CPCVFold:
    fold_id: int
    test_groups: tuple[int, int]
    train_groups: tuple[int, ...]
    test_dates: tuple[pd.Timestamp, ...]


def cpcv_8_choose_2(session_dates: Iterable[object]) -> tuple[tuple[tuple[pd.Timestamp, ...], ...], tuple[CPCVFold, ...]]:
    dates = tuple(pd.Timestamp(x) for x in sorted(_dates(session_dates).unique()))
    if len(dates) < 8:
        raise ValueError("CPCV 8 choose 2 requires at least eight development sessions")
    chunks = tuple(tuple(dates[i] for i in indices) for indices in np.array_split(np.arange(len(dates)), 8))
    if any(not group for group in chunks):
        raise AssertionError("empty CPCV group")
    folds = []
    for fold_id, held_out in enumerate(combinations(range(8), 2)):
        train_groups = tuple(i for i in range(8) if i not in held_out)
        test_dates = tuple(date for i in held_out for date in chunks[i])
        folds.append(CPCVFold(fold_id, held_out, train_groups, test_dates))
    return chunks, tuple(folds)


def _test_blocks(test_dates: Iterable[object], all_sessions: Iterable[object]) -> list[tuple[int, int]]:
    sessions = list(pd.Timestamp(x) for x in sorted(_dates(all_sessions).unique()))
    lookup = {date: i for i, date in enumerate(sessions)}
    held = sorted({lookup[pd.Timestamp(x)] for x in _dates(test_dates).unique()})
    if not held:
        raise ValueError("test block is empty")
    blocks: list[tuple[int, int]] = []
    start = previous = held[0]
    for index in held[1:]:
        if index != previous + 1:
            blocks.append((start, previous))
            start = index
        previous = index
    blocks.append((start, previous))
    return blocks


def purged_embargoed_train(
    frame: pd.DataFrame,
    *,
    test_dates: Iterable[object],
    all_sessions: Iterable[object],
) -> pd.DataFrame:
    """Remove validation dates, overlapping labels, and one following session."""
    required = {"decision_time", "label_end_at"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing temporal columns: {sorted(missing)}")
    work = frame.copy()
    decision_date = _dates(work["decision_time"])
    test_set = set(pd.Timestamp(x) for x in _dates(test_dates).unique())
    session_list = list(pd.Timestamp(x) for x in sorted(_dates(all_sessions).unique()))
    blocks = _test_blocks(test_set, session_list)
    keep = ~decision_date.isin(test_set)
    decision_time = pd.to_datetime(work["decision_time"], errors="raise")
    label_end = pd.to_datetime(work["label_end_at"], errors="raise")
    if decision_time.dt.tz is None or label_end.dt.tz is None:
        raise ValueError("decision_time and label_end_at must be timezone-aware")
    if decision_time.isna().any() or label_end.isna().any() or label_end.le(decision_time).any():
        raise ValueError("temporal columns require finite label_end_at > decision_time")
    for start_index, end_index in blocks:
        first_date = session_list[start_index]
        first_test_time = decision_time[decision_date.eq(first_date)].min()
        if pd.isna(first_test_time):
            raise ValueError(f"no test decisions for session {first_date.date()}")
        preceding = decision_time < first_test_time
        keep &= ~(preceding & label_end.ge(first_test_time))
        following_index = end_index + 1
        if following_index < len(session_list):
            keep &= ~decision_date.eq(session_list[following_index])
    out = work.loc[keep].copy()
    assert_purge_embargo(out, frame.loc[decision_date.isin(test_set)], all_sessions=all_sessions)
    return out


def assert_purge_embargo(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    all_sessions: Iterable[object],
) -> None:
    """Assert label-end purge and one-full-following-session embargo."""
    required = {"decision_time", "label_end_at"}
    for name, data in (("train", train), ("test", test)):
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{name} missing temporal columns: {sorted(missing)}")
    if test.empty:
        raise ValueError("test frame is empty")
    train_dates = _dates(train["decision_time"])
    test_dates = _dates(test["decision_time"])
    if set(train_dates).intersection(test_dates):
        raise AssertionError("training contains a validation session")
    sessions = list(pd.Timestamp(x) for x in sorted(_dates(all_sessions).unique()))
    train_time = pd.to_datetime(train["decision_time"], errors="raise")
    label_end = pd.to_datetime(train["label_end_at"], errors="raise")
    test_time = pd.to_datetime(test["decision_time"], errors="raise")
    if train_time.dt.tz is None or label_end.dt.tz is None or test_time.dt.tz is None:
        raise ValueError("split timestamps must be timezone-aware")
    if (
        train_time.isna().any() or label_end.isna().any() or test_time.isna().any()
        or label_end.le(train_time).any()
    ):
        raise ValueError("temporal columns require finite label_end_at > decision_time")
    for start_index, end_index in _test_blocks(test_dates, sessions):
        first_date = sessions[start_index]
        first_test_time = test_time[test_dates.eq(first_date)].min()
        preceding = train_time < first_test_time
        if label_end[preceding].ge(first_test_time).any():
            raise AssertionError(f"training label overlaps test block beginning {first_test_time}")
        following_index = end_index + 1
        if following_index < len(sessions):
            embargo_date = sessions[following_index]
            if train_dates.eq(embargo_date).any():
                raise AssertionError(f"embargo session present in training: {embargo_date.date()}")


def split_protocol_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
