"""Deterministic market-data integrity rules for volatility research.

This module deliberately contains no strategy, label, or P&L logic.  Its
functions turn the frozen rules in ``DATA_AUDIT.md`` into reusable, testable
data transformations.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


MARKET_TIMEZONE = ZoneInfo("Asia/Kolkata")
SESSION_OPEN = time(9, 15)
SESSION_CLOSE = time(15, 30, 59, 999999)
MAX_EXPECTED_GAP = pd.Timedelta(seconds=61)


def localize_asia_kolkata(values: pd.Series | Iterable[object]) -> pd.Series:
    """Parse timestamps and return an Asia/Kolkata-aware Series.

    Naive values are interpreted as exchange-local wall-clock times.  Already
    aware values are converted to Asia/Kolkata.  Unparseable values become
    ``NaT`` rather than acquiring a guessed timestamp.
    """

    source = values if isinstance(values, pd.Series) else pd.Series(values)
    parsed = pd.to_datetime(source, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(MARKET_TIMEZONE)
    if pd.api.types.is_datetime64_dtype(parsed.dtype):
        return parsed.dt.tz_localize(MARKET_TIMEZONE)

    # Mixed aware/naive input is uncommon but can occur at ingestion.  Parse
    # each value explicitly so aware values are converted and naive values are
    # treated as exchange-local without depending on the host timezone.
    def convert_one(value: object) -> pd.Timestamp:
        stamp = pd.to_datetime(value, errors="coerce")
        if pd.isna(stamp):
            return pd.NaT
        stamp = pd.Timestamp(stamp)
        if stamp.tzinfo is None:
            return stamp.tz_localize(MARKET_TIMEZONE)
        return stamp.tz_convert(MARKET_TIMEZONE)

    return source.map(convert_one)


def option_available_at(
    frame: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    date_col: str = "Date",
    tick_time_col: str = "tick_time",
) -> pd.Series:
    """Return ``max(datetime, Date + tick_time)`` in Asia/Kolkata.

    Both timestamp representations must parse successfully.  A missing or
    ambiguous availability component produces ``NaT`` and is therefore not
    eligible for a signal or execution.
    """

    observed = localize_asia_kolkata(frame[datetime_col])
    date_text = pd.to_datetime(frame[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    tick_text = frame[tick_time_col].astype("string").str.strip()
    tick = localize_asia_kolkata(date_text + " " + tick_text)
    pair = pd.concat(
        [observed.rename("observed"), tick.rename("tick")], axis=1
    )
    # skipna=False is intentional: absence of either component makes the true
    # information time unverifiable.
    return pair.max(axis=1, skipna=False).rename("available_at")


def session_flags(
    timestamps: pd.Series | Iterable[object],
    *,
    recorded_sessions: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Classify timestamps under the primary 09:15--15:30 session rule.

    Weekend/special sessions are retained as a separate diagnostic stratum and
    never marked as ordinary primary-session observations.  If
    ``recorded_sessions`` is supplied, only dates in that observed calendar
    can be in either session stratum.
    """

    ts = localize_asia_kolkata(timestamps)
    seconds = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    inside = seconds.between(
        SESSION_OPEN.hour * 3600 + SESSION_OPEN.minute * 60,
        SESSION_CLOSE.hour * 3600 + SESSION_CLOSE.minute * 60 + SESSION_CLOSE.second,
    ) & ts.notna()
    if recorded_sessions is None:
        recorded = pd.Series(True, index=ts.index)
    else:
        normalized = localize_asia_kolkata(recorded_sessions).dt.normalize()
        recorded = ts.dt.normalize().isin(normalized.dropna())
    special = ts.dt.dayofweek.ge(5)
    return pd.DataFrame(
        {
            "in_recorded_session": recorded,
            "in_session_hours": inside,
            "is_special_session": recorded & inside & special,
            "is_primary_session": recorded & inside & ~special,
        },
        index=ts.index,
    )


def filter_primary_session(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    recorded_sessions: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Return ordinary recorded-session rows inside 09:15--15:30."""

    flags = session_flags(frame[timestamp_col], recorded_sessions=recorded_sessions)
    return frame.loc[flags["is_primary_session"]].copy()


def spot_integrity_flags(
    frame: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    spot_col: str = "spot_price",
    max_gap: pd.Timedelta | str = MAX_EXPECTED_GAP,
    recorded_sessions: Iterable[object] | None = None,
) -> pd.DataFrame:
    """Flag current-row spot validity and breaks between consecutive bars.

    Gaps are assessed within a local calendar date.  The first observation of
    a date is not a gap.  Rows outside the primary session and nonpositive or
    nonfinite prices are invalid; prices are never forward-filled.
    """

    ts = localize_asia_kolkata(frame[timestamp_col])
    price = pd.to_numeric(frame[spot_col], errors="coerce")
    flags = session_flags(ts, recorded_sessions=recorded_sessions)
    work = pd.DataFrame({"timestamp": ts, "position": np.arange(len(frame))})
    ordered = work.sort_values("timestamp", kind="stable")
    delta = ordered.groupby(ordered["timestamp"].dt.normalize())["timestamp"].diff()
    ordered["gap_from_previous"] = delta.gt(pd.Timedelta(max_gap))
    gap = ordered.sort_values("position")["gap_from_previous"].set_axis(frame.index)
    positive = price.gt(0) & np.isfinite(price)
    result = flags.copy()
    result["spot_is_positive"] = positive
    result["gap_from_previous"] = gap.fillna(False).astype(bool)
    result["spot_row_usable"] = (
        result["is_primary_session"] & positive & ~result["gap_from_previous"]
    )
    return result


def valid_spot_windows(
    starts: pd.Series | Iterable[object],
    ends: pd.Series | Iterable[object],
    spot: pd.DataFrame,
    *,
    timestamp_col: str = "datetime",
    spot_col: str = "spot_price",
    max_gap: pd.Timedelta | str = MAX_EXPECTED_GAP,
    recorded_sessions: Iterable[object] | None = None,
) -> pd.Series:
    """Mark windows containing only valid, contiguous primary-session spots.

    The interval is closed on both ends.  It must contain observations, remain
    on one local date, contain no nonpositive spot, and cross no gap larger
    than ``max_gap``.  This is the primitive to use before constructing either
    lagged features or forward labels.
    """

    start = localize_asia_kolkata(starts).reset_index(drop=True)
    end = localize_asia_kolkata(ends).reset_index(drop=True)
    ts = localize_asia_kolkata(spot[timestamp_col])
    price = pd.to_numeric(spot[spot_col], errors="coerce")
    flags = session_flags(ts, recorded_sessions=recorded_sessions)
    clean = pd.DataFrame({"timestamp": ts, "price": price, "primary": flags["is_primary_session"]})
    clean = clean.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
    stamps = clean["timestamp"]
    bad = (~clean["primary"] | ~np.isfinite(clean["price"]) | clean["price"].le(0)).to_numpy()
    gaps = clean["timestamp"].diff().gt(pd.Timedelta(max_gap))
    same_day = clean["timestamp"].dt.normalize().eq(clean["timestamp"].shift().dt.normalize())
    gap_boundary = (gaps & same_day).to_numpy()
    bad_prefix = np.r_[0, np.cumsum(bad)]
    gap_prefix = np.r_[0, np.cumsum(gap_boundary)]
    output = np.zeros(len(start), dtype=bool)
    for i, (left_stamp, right_stamp) in enumerate(zip(start, end)):
        if pd.isna(left_stamp) or pd.isna(right_stamp) or right_stamp < left_stamp:
            continue
        if left_stamp.normalize() != right_stamp.normalize():
            continue
        # Search the datetime extension array directly.  This is robust to
        # pandas choosing microsecond rather than nanosecond storage units.
        left = int(stamps.searchsorted(left_stamp, side="left"))
        right = int(stamps.searchsorted(right_stamp, side="right"))
        if left == right:
            continue
        has_bad = bad_prefix[right] != bad_prefix[left]
        first_stamp = stamps.iloc[left]
        last_stamp = stamps.iloc[right - 1]
        # A boundary attached to an observation exactly at the window start is
        # outside the interval.  If the first observation is later, however,
        # its preceding gap can overlap the requested window and must count.
        first_gap_position = left + int(first_stamp == left_stamp)
        has_gap = gap_prefix[right] != gap_prefix[first_gap_position]
        uncovered_edge = (
            first_stamp - left_stamp > pd.Timedelta(max_gap)
            or right_stamp - last_stamp > pd.Timedelta(max_gap)
        )
        output[i] = not has_bad and not has_gap and not uncovered_edge
    index = starts.index if isinstance(starts, pd.Series) else pd.RangeIndex(len(output))
    return pd.Series(output, index=index, name="spot_window_usable")


def current_row_iv_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Return side-specific IV flags using current-row fields only."""

    result: dict[str, pd.Series] = {}
    for side in ("call", "put"):
        iv = pd.to_numeric(frame[f"iv_{side}"], errors="coerce")
        converged = frame[f"iv_{side}_converged"].map(
            lambda value: value is True
            or (isinstance(value, (str, np.str_)) and value.strip().lower() == "true")
            or (isinstance(value, (int, np.integer)) and not isinstance(value, bool) and value == 1)
        )
        result[f"{side}_iv_valid"] = iv.gt(0) & np.isfinite(iv)
        result[f"{side}_iv_converged"] = converged.astype(bool)
        result[f"{side}_iv_usable"] = result[f"{side}_iv_valid"] & result[f"{side}_iv_converged"]
    result["all_iv_usable"] = result["call_iv_usable"] & result["put_iv_usable"]
    return pd.DataFrame(result, index=frame.index)


def match_next_snapshot(
    decisions: pd.DataFrame,
    quotes: pd.DataFrame,
    *,
    decision_time_col: str = "decision_time",
    quote_time_col: str = "available_at",
    by: Sequence[str] = (),
    same_session: bool = True,
    primary_session_only: bool = True,
    quote_prefix: str = "execution_",
) -> pd.DataFrame:
    """Attach the first quote strictly after each decision.

    Matching is optionally partitioned by contract columns such as ``expiry``
    and ``strike``.  Exact-time quotes are forbidden.  By default an execution
    cannot cross the local session date, and out-of-hours or special-session
    quotes are ineligible.  Duplicate quote keys at one availability time are
    rejected because there is no deterministic single execution snapshot.
    """

    missing_decision = [column for column in [decision_time_col, *by] if column not in decisions]
    missing_quote = [column for column in [quote_time_col, *by] if column not in quotes]
    if missing_decision or missing_quote:
        raise KeyError(f"missing decision columns={missing_decision}; quote columns={missing_quote}")

    left = decisions.copy()
    right = quotes.copy()
    left[decision_time_col] = localize_asia_kolkata(left[decision_time_col])
    right[quote_time_col] = localize_asia_kolkata(right[quote_time_col])
    right = right.dropna(subset=[quote_time_col])
    if primary_session_only:
        eligible = session_flags(right[quote_time_col])["is_primary_session"]
        right = right.loc[eligible].copy()
    unique_key = [*by, quote_time_col]
    if right.duplicated(unique_key).any():
        raise ValueError(f"quotes are not unique on {unique_key}")

    output = left.copy()
    quote_columns = list(quotes.columns)
    for column in quote_columns:
        output[f"{quote_prefix}{column}"] = pd.NA

    if by:
        grouper: object = by[0] if len(by) == 1 else list(by)
        groups = {
            key if isinstance(key, tuple) else (key,): group.sort_values(quote_time_col)
            for key, group in right.groupby(grouper, dropna=False, sort=False)
        }
    else:
        groups = {(): right.sort_values(quote_time_col)}

    for position, (_, decision) in enumerate(left.iterrows()):
        decision_time = decision[decision_time_col]
        if pd.isna(decision_time):
            continue
        if primary_session_only and not session_flags(pd.Series([decision_time]))[
            "is_primary_session"
        ].iloc[0]:
            continue
        key = tuple(decision[column] for column in by)
        candidates = groups.get(key)
        if candidates is None or candidates.empty:
            continue
        times = candidates[quote_time_col]
        match_position = int(times.searchsorted(decision_time, side="right"))
        if match_position >= len(candidates):
            continue
        match = candidates.iloc[match_position]
        if same_session and match[quote_time_col].normalize() != decision_time.normalize():
            continue
        for column in quote_columns:
            destination = output.columns.get_loc(f"{quote_prefix}{column}")
            output.iat[position, destination] = match[column]

    matched_time_col = f"{quote_prefix}{quote_time_col}"
    output[matched_time_col] = localize_asia_kolkata(output[matched_time_col])
    output[f"{quote_prefix}matched"] = output[matched_time_col].notna()
    return output
