"""Frozen Research V2 economic mapping (research only, no broker interface).

The default CLI is fail-closed.  Real-data evaluation requires an independently
supplied event calendar and leakage authorization.  Pseudo option PnL has a
separate atomic one-shot API and is intentionally unavailable from the CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass, fields
from bisect import bisect_left
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Literal

import duckdb
import numpy as np
import pandas as pd

try:
    from .execution_costs import (
        CostBreakdown, FillPolicy, LiquidityProxy, OptionExecution,
        execution_cost, margin_proxy, settlement_cost,
    )
    from .integrity import option_available_at
    from .research_v2_p1 import append_global_trial, sha256_file
except ImportError:
    from execution_costs import (  # type: ignore
        CostBreakdown, FillPolicy, LiquidityProxy, OptionExecution,
        execution_cost, margin_proxy, settlement_cost,
    )
    from integrity import option_available_at  # type: ignore
    from research_v2_p1 import append_global_trial, sha256_file  # type: ignore


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "STRATEGY_SPEC_V2.md"
SPEC_HASH = ROOT / "STRATEGY_SPEC_V2.sha256"
RISK_ADDENDUM = ROOT / "RISK_POLICY_V2_ADDENDUM.md"
RISK_ADDENDUM_SHA256 = "abf5349f04e239749ff87336a57eeb05b7de79b98d41a0c10afff493e0b75418"
EVENT_CALENDAR = ROOT / "EVENT_CALENDAR_V2.csv"
EVENT_CALENDAR_SHA256 = "35d705b8a16ef3c8f94b4acb2cceae1832a84d4842f586ce47aca4832605fa25"
AUTHORIZATION_NAME = "STRATEGY_PNL_AUTHORIZATION_V2.json"
GLOBAL_TRIALS = ROOT / "TRIALS.csv"
OPTION_SOURCE = ROOT / "data/raw/icharts_option_banknifty_monthly_atm_up_to_5strikes.csv"
P2_DIR = ROOT / "outputs/research_v2/p2"
OUT = ROOT / "outputs/research_v2/strategy"
THRESHOLDS = (1.15, 1.25, 1.35)
DECISION_CLOCKS = {(10, 15), (11, 45), (13, 25)}
FILL_POLICY = FillPolicy(max_primary_lots=1, oi_participation=0.001, unknown_oi_capacity_lots=0)
LIQUIDITY = LiquidityProxy()
CAPITAL = 1_000_000.0


@dataclass
class RiskState:
    cumulative_pnl: float = 0.0
    high_water_pnl: float = 0.0
    iso_week: tuple[int, int] | None = None
    week_pnl: float = 0.0
    current_day: date | None = None
    day_pnl: float = 0.0

    @property
    def drawdown(self) -> float:
        return self.high_water_pnl - self.cumulative_pnl


def pretrade_state_rejection(state: RiskState, day: date) -> str | None:
    iso = day.isocalendar()
    week = (iso.year, iso.week)
    if state.iso_week != week:
        state.iso_week, state.week_pnl = week, 0.0
    if state.current_day != day:
        state.current_day, state.day_pnl = day, 0.0
    if state.day_pnl <= -10_000:
        return "daily_loss_limit"
    if state.week_pnl <= -25_000:
        return "weekly_loss_limit"
    if state.drawdown >= 50_000:
        return "drawdown_suspension"
    return None


def update_risk_state(state: RiskState, day: date, net_pnl: float) -> None:
    if not np.isfinite(net_pnl):
        raise ValueError("risk PnL must be finite")
    pretrade_state_rejection(state, day)
    state.day_pnl += net_pnl
    state.week_pnl += net_pnl
    state.cumulative_pnl += net_pnl
    state.high_water_pnl = max(state.high_water_pnl, state.cumulative_pnl)


def proposed_loss_rejection(state: RiskState, day: date, severe_loss: float) -> str | None:
    """Fail closed when a new position could consume a remaining loss budget."""
    existing = pretrade_state_rejection(state, day)
    if existing is not None:
        return existing
    if not np.isfinite(severe_loss) or severe_loss < 0:
        return "invalid_severe_loss"
    if state.day_pnl - severe_loss <= -10_000:
        return "remaining_daily_loss_budget"
    if state.week_pnl - severe_loss <= -25_000:
        return "remaining_weekly_loss_budget"
    if state.drawdown + severe_loss >= 50_000:
        return "remaining_drawdown_budget"
    return None


@dataclass(frozen=True)
class EventCalendar:
    primary_no_entry_dates: frozenset[date]
    incident_stress_dates: frozenset[date]
    metadata_by_date: dict[date, tuple[dict[str, str], ...]]


def verify_strategy_spec(spec: Path = SPEC, checksum: Path = SPEC_HASH) -> str:
    expected = checksum.read_text(encoding="ascii").strip().split()[0]
    actual = sha256_file(spec)
    if actual != expected:
        raise PermissionError("STRATEGY_SPEC_V2.md checksum mismatch")
    return actual


def verify_risk_addendum(path: Path = RISK_ADDENDUM) -> str:
    actual = sha256_file(path)
    if actual != RISK_ADDENDUM_SHA256:
        raise PermissionError("RISK_POLICY_V2_ADDENDUM.md checksum mismatch")
    return actual


def _relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def verify_run_authorization(
    segment: Literal["development", "validation"],
    event_calendar: Path,
    authorization: Path,
) -> dict[str, str]:
    """Authenticate a frozen, auditor-signed-by-hash pre-PnL inventory.

    The authorization name and its root checksum sidecar are fixed so an
    arbitrary file containing the word "approved" cannot unlock option reads.
    Hashing inventory members is permitted; no tabular prediction or option
    row is opened here.
    """
    authorization = authorization.resolve()
    if authorization.parent != ROOT.resolve() or authorization.name != AUTHORIZATION_NAME:
        raise PermissionError("authorization must be the frozen root authorization")
    checksum = ROOT / f"{AUTHORIZATION_NAME}.sha256"
    if not authorization.exists() or not checksum.exists():
        raise PermissionError("frozen authorization and checksum are required")
    expected_auth = checksum.read_text(encoding="ascii").strip().split()[0].lower()
    if sha256_file(authorization) != expected_auth:
        raise PermissionError("authorization checksum mismatch")
    payload = json.loads(authorization.read_text(encoding="utf-8"))
    if (payload.get("approved") is not True
            or payload.get("scope") != "research_v2_strategy_pnl"
            or segment not in payload.get("authorized_segments", [])
            or payload.get("pseudo_authorized") is not False):
        raise PermissionError("authorization scope does not permit this segment")

    prediction = (P2_DIR / "development_cpcv_oof_predictions.parquet"
                  if segment == "development" else P2_DIR / "walk_forward_predictions.parquet")
    provenance = (P2_DIR / "development_cpcv_oof_manifest.json"
                  if segment == "development" else P2_DIR / "gate.json")
    required_paths = [
        SPEC, RISK_ADDENDUM, event_calendar.resolve(),
        Path(__file__).resolve(), ROOT / "research/execution_costs.py",
        ROOT / "tests/test_research_v2_strategy.py", prediction, provenance,
        OPTION_SOURCE,
    ]
    inventory = payload.get("inventory")
    if not isinstance(inventory, dict):
        raise PermissionError("authorization lacks unified hash inventory")
    verified: dict[str, str] = {}
    for path in required_paths:
        if not path.exists():
            raise PermissionError(f"authorized inventory member absent: {path.name}")
        key = _relative(path)
        actual = sha256_file(path)
        if inventory.get(key) != actual:
            raise PermissionError(f"authorized inventory mismatch: {key}")
        verified[key] = actual
    if verified[_relative(SPEC)] != verify_strategy_spec():
        raise PermissionError("strategy spec inventory mismatch")
    if verified[_relative(RISK_ADDENDUM)] != verify_risk_addendum():
        raise PermissionError("risk addendum inventory mismatch")
    if event_calendar.resolve() != EVENT_CALENDAR.resolve() or verified[_relative(event_calendar)] != EVENT_CALENDAR_SHA256:
        raise PermissionError("event calendar is not the frozen V2 calendar")
    return verified


def _hash_inventory(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            result[name.lstrip("* ")] = digest
    return result


def load_approved_predictions(segment: Literal["development", "validation"]) -> pd.DataFrame:
    """Project only approved timestamp/forecast columns; never target outcomes."""
    if segment == "validation":
        path = P2_DIR / "walk_forward_predictions.parquet"
        gate_path = P2_DIR / "gate.json"
        inventory = _hash_inventory(P2_DIR / "SHA256SUMS")
        if sha256_file(gate_path) != inventory.get("gate.json"):
            raise PermissionError("P2 gate hash mismatch")
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (not gate.get("p2_vs_p0_predictive_gate_pass")
                or not gate.get("provenance_verified")
                or gate.get("pseudo_holdout_accessed")
                or gate.get("strategy_pnl_computed")):
            raise PermissionError("P2 predictive gate is not approved")
        expected = inventory.get(path.name)
        projection = "CAST(decision_time AS VARCHAR) AS decision_time, p2_forecast"
    elif segment == "development":
        path = P2_DIR / "development_cpcv_oof_predictions.parquet"
        manifest_path = P2_DIR / "development_cpcv_oof_manifest.json"
        sums_path = P2_DIR / "development_cpcv_oof_SHA256SUMS"
        if not manifest_path.exists() or not sums_path.exists():
            raise PermissionError("approved seven-model development ledger is absent")
        inventory = _hash_inventory(sums_path)
        manifest_key = "outputs/research_v2/p2/development_cpcv_oof_manifest.json"
        artifact_key = "outputs/research_v2/p2/development_cpcv_oof_predictions.parquet"
        if sha256_file(manifest_path) != inventory.get(manifest_key):
            raise PermissionError("development manifest hash mismatch")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance_ok = (
            manifest.get("artifact") == artifact_key
            and manifest.get("scope") == "development-only selected-P2 CPCV out-of-fold prediction"
            and manifest.get("successful_refits") == manifest.get("expected_refits") == 28
            and manifest.get("component_forecast_count_min") == 7
            and manifest.get("component_forecast_count_max") == 7
            and manifest.get("rows_with_exactly_7_components") == manifest.get("development_rows")
            and manifest.get("coverage_pass") is True
            and manifest.get("seed") == 7319
            and manifest.get("validation_accessed") is False
            and manifest.get("pseudo_holdout_accessed") is False
            and manifest.get("option_data_accessed") is False
            and manifest.get("strategy_pnl_computed") is False
        )
        if not provenance_ok:
            raise PermissionError("development prediction provenance mismatch")
        expected = inventory.get(artifact_key)
        projection = (
            "CAST(decision_time AS VARCHAR) AS decision_time, p2_forecast, "
            "component_forecast_count, split"
        )
    else:
        raise PermissionError("unapproved prediction segment")
    if not path.exists() or sha256_file(path) != expected:
        raise PermissionError("prediction ledger hash mismatch")
    con = duckdb.connect()
    try:
        frame = con.execute(
            f"SELECT {projection} FROM read_parquet(?) ORDER BY decision_time",
            [str(path)],
        ).fetchdf()
    finally:
        con.close()
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True).dt.tz_convert("Asia/Kolkata")
    if segment == "development":
        if (not frame.component_forecast_count.eq(7).all()
                or not frame.split.astype(str).eq("development").all()
                or len(frame) != manifest["development_rows"]):
            raise PermissionError("development ledger coverage mismatch")
        frame = frame[["decision_time", "p2_forecast"]]
    clock = list(zip(frame.decision_time.dt.hour, frame.decision_time.dt.minute))
    frame = frame.loc[[item in DECISION_CLOCKS for item in clock]].copy()
    if frame.empty or not np.isfinite(frame.p2_forecast).all() or not frame.p2_forecast.gt(0).all():
        raise RuntimeError("invalid approved prediction ledger")
    return frame


def load_event_calendar(path: Path) -> tuple[EventCalendar, str]:
    if not path.exists():
        raise PermissionError("predeclared event calendar is required")
    frame = pd.read_csv(path)
    audited = {"date", "event_type", "event_name", "source_url", "announced_at", "primary_no_entry"}
    if set(frame.columns) != audited:
        raise PermissionError("event calendar schema is not the audited schema")
    parsed = pd.to_datetime(frame["date"], errors="raise").dt.date
    text_columns = ["event_type", "event_name", "source_url", "announced_at"]
    if any(frame[column].isna().any() or frame[column].astype(str).str.strip().eq("").any()
           for column in text_columns):
        raise PermissionError("event calendar has unaudited entries")
    known_announcements = frame.announced_at.astype(str).str.strip().ne("unknown")
    announced = pd.to_datetime(frame.loc[known_announcements, "announced_at"], errors="coerce", utc=True)
    if announced.isna().any():
        raise PermissionError("event calendar has invalid announcement timestamps")
    normalized = frame.primary_no_entry.astype(str).str.strip().str.lower()
    if not normalized.isin({"true", "false"}).all():
        raise PermissionError("primary_no_entry must contain only true/false")
    metadata: dict[date, list[dict[str, str]]] = {}
    primary: set[date] = set()
    incidents: set[date] = set()
    for row_index, event_day in enumerate(parsed):
        record = {
            key: str(frame.iloc[row_index][key])
            for key in ("event_type", "event_name", "source_url", "announced_at")
        }
        metadata.setdefault(event_day, []).append(record)
        is_unscheduled_unknown = (
            record["event_type"] == "nse_exchange_halt"
            and record["announced_at"].strip().lower() == "unknown"
        )
        if is_unscheduled_unknown:
            incidents.add(event_day)
        elif normalized.iloc[row_index] == "true":
            primary.add(event_day)
    calendar = EventCalendar(
        primary_no_entry_dates=frozenset(primary),
        incident_stress_dates=frozenset(incidents),
        metadata_by_date={key: tuple(value) for key, value in metadata.items()},
    )
    return calendar, sha256_file(path)


def prepare_option_rows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "datetime", "Date", "tick_time", "expiry", "is_monthly", "strike",
        "spot_price", "call_price", "put_price", "call_oi", "put_oi",
        "iv_call", "iv_put", "iv_call_converged", "iv_put_converged",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"option rows missing {sorted(missing)}")
    out = frame.copy()
    out = out[out.is_monthly.astype(bool)].copy()
    out["available_at"] = option_available_at(out)
    out["expiry"] = pd.to_datetime(out.expiry, errors="coerce").dt.date
    out["strike"] = pd.to_numeric(out.strike, errors="coerce")
    for column in ("spot_price", "call_price", "put_price", "call_oi", "put_oi", "iv_call", "iv_put"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out.dropna(subset=["available_at", "expiry", "strike"]).sort_values(
        ["available_at", "expiry", "strike"]
    ).reset_index(drop=True)


def read_option_rows(start: date, end: date, path: Path = OPTION_SOURCE) -> pd.DataFrame:
    """Read only the calendar interval authorized by the prediction ledger."""
    columns = [
        "datetime", "Date", "tick_time", "expiry", "is_monthly", "strike",
        "spot_price", "call_price", "put_price", "call_oi", "put_oi",
        "iv_call", "iv_put", "iv_call_converged", "iv_put_converged",
    ]
    con = duckdb.connect()
    try:
        query = f"""SELECT {','.join(columns)} FROM read_csv_auto(?, sample_size=20000)
                    WHERE Date >= ? AND Date <= ? AND is_monthly = true"""
        frame = con.execute(query, [str(path), start, end]).fetchdf()
    finally:
        con.close()
    return prepare_option_rows(frame)


def historical_lot_size(day: date) -> int:
    return 15 if day <= date(2024, 12, 24) else 30


def _quality(row: pd.Series, option_type: str) -> bool:
    iv = row[f"iv_{option_type}"]
    converged = row[f"iv_{option_type}_converged"]
    price = row[f"{option_type}_price"]
    oi = row[f"{option_type}_oi"]
    return bool(
        np.isfinite(iv) and iv > 0 and bool(converged)
        and np.isfinite(price) and price > 0 and np.isfinite(oi) and oi >= 0
    )


@dataclass(frozen=True)
class Leg:
    option_type: Literal["call", "put"]
    strike: float
    qty: int
    entry_price: float
    entry_oi: float


def nearest_monthly_chain(snapshot: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    eligible = snapshot[pd.to_datetime(snapshot.expiry).dt.date >= when.date()]
    if eligible.empty:
        return eligible
    expiry = min(eligible.expiry)
    return eligible[eligible.expiry.eq(expiry)].sort_values("strike").drop_duplicates("strike")


def first_snapshot_after(options: pd.DataFrame, when: pd.Timestamp, *, strict: bool) -> pd.DataFrame:
    times = options.loc[
        options.available_at.gt(when) if strict else options.available_at.ge(when), "available_at"
    ]
    times = times[times.dt.normalize().eq(when.normalize())]
    if times.empty:
        return options.iloc[:0].copy()
    first = times.min()
    return options[options.available_at.eq(first)].copy()


def first_complete_entry_snapshot_after(options: pd.DataFrame, when: pd.Timestamp) -> pd.DataFrame:
    """First post-decision snapshot with a valid nearest-expiry ATM pair.

    Completeness is decided from entry fields alone.  Structure-specific
    completeness is checked only after the signal selects that structure, so
    an unused iron-fly wing cannot delay a long-straddle entry.
    """
    candidates = options[
        options.available_at.gt(when)
        & options.available_at.dt.normalize().eq(when.normalize())
    ]
    for timestamp, snapshot in candidates.groupby("available_at", sort=True):
        if not _snapshot_is_synchronous(snapshot):
            continue
        chain = nearest_monthly_chain(snapshot, when)
        if len(chain) < 5:
            continue
        try:
            build_structure(chain, "long_straddle")
        except ValueError:
            continue
        return snapshot.copy()
    return options.iloc[:0].copy()


def _snapshot_is_synchronous(snapshot: pd.DataFrame) -> bool:
    if snapshot.empty:
        return False
    return bool(
        snapshot["datetime"].astype(str).nunique(dropna=False) == 1
        and snapshot["tick_time"].astype(str).nunique(dropna=False) == 1
        and snapshot["spot_price"].nunique(dropna=False) == 1
    )


def select_entry_snapshot(
    options: pd.DataFrame, when: pd.Timestamp, forecast: float, threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str | None, pd.Series | None, str]:
    """Select causally from entry snapshots, requiring only the signaled legs."""
    candidates = options[
        options.available_at.gt(when)
        & options.available_at.dt.normalize().eq(when.normalize())
    ]
    saw_valid_atm = False
    for _, snapshot in candidates.groupby("available_at", sort=True):
        if not _snapshot_is_synchronous(snapshot):
            continue
        chain = nearest_monthly_chain(snapshot, when)
        try:
            _, atm = build_structure(chain, "long_straddle")
        except ValueError:
            continue
        saw_valid_atm = True
        atm_iv = float(np.mean([atm.iv_call, atm.iv_put]))
        iv_variance = atm_iv * atm_iv * 120.0 / (252.0 * 375.0)
        short_ratio, long_ratio = iv_variance / forecast, forecast / iv_variance
        structure = "iron_fly" if short_ratio >= threshold else (
            "long_straddle" if long_ratio >= threshold else None
        )
        if structure is None:
            return snapshot.copy(), chain, None, atm, "threshold_not_crossed"
        try:
            build_structure(chain, structure)
        except ValueError:
            continue
        return snapshot.copy(), chain, structure, atm, "selected"
    reason = "selected_structure_incomplete" if saw_valid_atm else "no_complete_post_decision_snapshot"
    return options.iloc[:0].copy(), options.iloc[:0].copy(), None, None, reason


def index_option_sessions(options: pd.DataFrame) -> dict[date, pd.DataFrame]:
    """Index prepared option rows once, avoiding full-panel scans per decision."""
    return {
        session_day: session.reset_index(drop=True)
        for session_day, session in options.groupby(options.available_at.dt.date, sort=True)
    }


def build_structure(chain: pd.DataFrame, structure: str) -> tuple[list[Leg], pd.Series]:
    if chain.empty:
        raise ValueError("empty nearest-expiry chain")
    spot = float(chain.spot_price.iloc[0])
    ordered = chain.sort_values("strike").reset_index(drop=True)
    atm_index = int((ordered.strike - spot).abs().to_numpy().argmin())
    atm = ordered.iloc[atm_index]
    if not (_quality(atm, "call") and _quality(atm, "put")):
        raise ValueError("invalid ATM quality")
    if structure == "long_straddle":
        return [
            Leg("call", float(atm.strike), 1, float(atm.call_price), float(atm.call_oi)),
            Leg("put", float(atm.strike), 1, float(atm.put_price), float(atm.put_oi)),
        ], atm
    if structure != "iron_fly" or atm_index < 2 or atm_index + 2 >= len(ordered):
        raise ValueError("iron-fly wings unavailable")
    lower, upper = ordered.iloc[atm_index - 2], ordered.iloc[atm_index + 2]
    if not (_quality(lower, "put") and _quality(upper, "call")):
        raise ValueError("invalid iron-fly wing quality")
    return [
        Leg("call", float(atm.strike), -1, float(atm.call_price), float(atm.call_oi)),
        Leg("put", float(atm.strike), -1, float(atm.put_price), float(atm.put_oi)),
        Leg("call", float(upper.strike), 1, float(upper.call_price), float(upper.call_oi)),
        Leg("put", float(lower.strike), 1, float(lower.put_price), float(lower.put_oi)),
    ], atm


def atomic_fill(legs: Iterable[Leg]) -> tuple[bool, str]:
    for leg in legs:
        decision = FILL_POLICY.decide(
            1, open_interest_contracts=int(leg.entry_oi), snapshot_available=True,
            valid_premium=np.isfinite(leg.entry_price) and leg.entry_price > 0,
            allow_partial=False,
        )
        if decision.status != "filled":
            return False, f"{leg.option_type}_{leg.strike}:{decision.reason}"
    return True, "atomic_full_fill"


def defined_risk(legs: list[Leg], lot_size: int, structure: str) -> tuple[float, float]:
    debit = sum(max(0, leg.qty) * leg.entry_price for leg in legs) - sum(
        max(0, -leg.qty) * leg.entry_price for leg in legs
    )
    if structure == "long_straddle":
        loss = sum(leg.entry_price for leg in legs) * lot_size
        return loss, loss
    atm = next(leg.strike for leg in legs if leg.qty < 0)
    widths = [abs(leg.strike - atm) for leg in legs if leg.qty > 0]
    credit = -sum(leg.qty * leg.entry_price for leg in legs)
    loss = max(0.0, max(widths) - credit) * lot_size
    return loss, max(0.0, debit) * lot_size


def _orders(legs: list[Leg], lot_size: int, *, exit_prices: dict[tuple[str, float], float] | None = None):
    result = []
    for leg in legs:
        closing = exit_prices is not None
        side = "buy" if ((leg.qty > 0) != closing) else "sell"
        premium = exit_prices[(leg.option_type, leg.strike)] if closing else leg.entry_price
        result.append(OptionExecution(side, premium, 1, lot_size))
    return result


def uncertainty_cost(cost: CostBreakdown) -> CostBreakdown:
    values = {}
    for field in fields(cost):
        value = getattr(cost, field.name)
        if field.name in {"bid_ask_crossing", "impact_slippage"}:
            values[field.name] = 3.0 * value
        elif field.name == "margin_carry":
            values[field.name] = value
        else:
            values[field.name] = 2.0 * value
    return CostBreakdown(**values)


def _fixed_exit(options: pd.DataFrame, decision_time: pd.Timestamp, entry_time: pd.Timestamp, expiry: date,
                legs: list[Leg]) -> tuple[pd.Timestamp | None, dict[tuple[str, float], float] | None]:
    target = decision_time + pd.Timedelta(minutes=120)
    day_rows = options[
        options.available_at.dt.normalize().eq(entry_time.normalize())
        & options.available_at.ge(target)
        & (options.available_at.dt.time <= pd.Timestamp("15:30").time())
        & options.expiry.eq(expiry)
    ]
    keys = {(leg.option_type, leg.strike) for leg in legs}
    for timestamp, snapshot in day_rows.groupby("available_at", sort=True):
        if not _snapshot_is_synchronous(snapshot):
            continue
        prices = {}
        valid = True
        for option_type, strike in keys:
            row = snapshot[snapshot.strike.eq(strike)]
            if len(row) != 1 or not _quality(row.iloc[0], option_type):
                valid = False
                break
            prices[(option_type, strike)] = float(row.iloc[0][f"{option_type}_price"])
            oi = int(row.iloc[0][f"{option_type}_oi"])
            if FILL_POLICY.decide(1, open_interest_contracts=oi).status != "filled":
                valid = False
                break
        if valid:
            return pd.Timestamp(timestamp), prices
    return None, None


def _entry_costs(legs, day, lot, vol_ratio, carry):
    ordinary = execution_cost(
        _orders(legs, lot), day, volatility_ratio=vol_ratio,
        liquidity=LIQUIDITY, margin_carry=carry,
    )
    return ordinary


def build_session_iv_cache(options: pd.DataFrame) -> dict[date, float]:
    """Build one causal 10:15 ATM-IV value per session in a single pass."""
    cache: dict[date, float] = {}
    for session_day, session in options.groupby(options.available_at.dt.date, sort=True):
        after_clock = session[
            (session.available_at.dt.hour > 10)
            | ((session.available_at.dt.hour == 10) & (session.available_at.dt.minute >= 15))
        ]
        if after_clock.empty:
            continue
        snapshot = after_clock[after_clock.available_at.eq(after_clock.available_at.min())]
        chain = nearest_monthly_chain(snapshot, snapshot.available_at.min())
        if chain.empty:
            continue
        try:
            _, atm = build_structure(chain, "long_straddle")
        except ValueError:
            continue
        cache[session_day] = float(np.mean([atm.iv_call, atm.iv_put]))
    return cache


def prior_session_atm_ivs(cache: dict[date, float], day: date, count: int = 20) -> list[float]:
    """Return at most ``count`` cached sessions strictly before ``day``."""
    days = sorted(cache)
    stop = bisect_left(days, day)
    return [cache[item] for item in days[max(0, stop - count):stop]]


def evaluate_policy(
    predictions: pd.DataFrame,
    options: pd.DataFrame,
    event_dates: EventCalendar | set[date],
    *,
    threshold: float,
    lot_mode: Literal["historical", "constant15"],
    segment: Literal["development", "validation", "pseudo_holdout_previously_exposed"],
    event_counterfactual: bool = False,
) -> pd.DataFrame:
    """Evaluate one frozen sleeve; caller must register the trial first."""
    if threshold not in THRESHOLDS:
        raise ValueError("threshold is not preregistered")
    options = prepare_option_rows(options) if "available_at" not in options else options.copy()
    decisions = predictions.sort_values("decision_time").copy()
    decisions["decision_time"] = pd.to_datetime(decisions.decision_time, utc=True).dt.tz_convert("Asia/Kolkata")
    decision_ids = [f"{segment}:{item.isoformat()}" for item in decisions.decision_time]
    if len(decision_ids) != len(set(decision_ids)):
        raise ValueError("duplicate decision IDs in approved prediction ledger")
    decision_universe = "".join(f"{item}\n" for item in sorted(decision_ids)).encode("utf-8")
    universe_hash = hashlib.sha256(decision_universe).hexdigest()
    expected_decisions = len(decision_ids)
    used_dates: set[date] = set()
    risk_state = RiskState()
    ledger = []
    iv_cache = build_session_iv_cache(options)
    option_sessions = index_option_sessions(options)
    for _, prediction in decisions.iterrows():
        decision_time = prediction.decision_time
        day = decision_time.date()
        decision_id = f"{segment}:{decision_time.isoformat()}"
        if isinstance(event_dates, EventCalendar):
            primary_event = day in event_dates.primary_no_entry_dates
            incident_stress = day in event_dates.incident_stress_dates
            event_records = event_dates.metadata_by_date.get(day, ())
        else:
            primary_event = day in event_dates
            incident_stress = False
            event_records = ()
        decision_base = {
            "segment": segment, "decision_id": decision_id, "decision_time": decision_time,
            "sample_label": ("pseudo-holdout, previously exposed"
                             if segment == "pseudo_holdout_previously_exposed" else segment),
            "expected_decision_count": expected_decisions,
            "decision_universe_sha256": universe_hash,
            "threshold": threshold, "lot_mode": lot_mode,
            "eligible_session": (decision_time.hour, decision_time.minute) in DECISION_CLOCKS,
            "event_day": bool(event_records) or primary_event,
            "primary_no_entry": primary_event,
            "operational_incident_stress": incident_stress,
            "event_types": "|".join(sorted({item["event_type"] for item in event_records})),
            "event_sources": "|".join(sorted({item["source_url"] for item in event_records})),
            "selected_signal": False, "missing_exit": False,
            "exit_treatment": "not_applicable", "turnover_rupees": 0.0,
        }
        if not decision_base["eligible_session"]:
            ledger.append({**decision_base, "status": "no_trade", "reason": "invalid_decision_clock"})
            continue
        if day in used_dates:
            ledger.append({**decision_base, "status": "no_trade", "reason": "first_signal_already_selected"})
            continue
        day_options = option_sessions.get(day, options.iloc[:0])
        forecast = float(prediction.p2_forecast)
        snapshot, chain, structure, atm, selection_reason = select_entry_snapshot(
            day_options, decision_time, forecast, threshold
        )
        if snapshot.empty:
            ledger.append({**decision_base, "status": "no_trade", "reason": selection_reason})
            continue
        assert atm is not None
        atm_iv = float(np.mean([atm.iv_call, atm.iv_put]))
        prior_sessions = prior_session_atm_ivs(iv_cache, day)
        if structure is None:
            ledger.append({**decision_base, "status": "no_trade", "reason": "threshold_not_crossed"})
            continue
        used_dates.add(day)  # the first qualifying signal consumes the session
        base = {
            **decision_base,
            "entry_time": snapshot.available_at.min(), "expiry": min(chain.expiry),
            "structure": structure, "selected_signal": True,
        }
        if primary_event and not event_counterfactual:
            ledger.append({**base, "status": "rejected", "reason": "event_day_no_entry"})
            continue
        state_rejection = pretrade_state_rejection(risk_state, day)
        if state_rejection is not None:
            ledger.append({**base, "status": "rejected", "reason": state_rejection,
                           "week_pnl_before": risk_state.week_pnl,
                           "drawdown_before": risk_state.drawdown})
            continue
        if len(prior_sessions) < 20:
            ledger.append({**base, "status": "rejected", "reason": "fewer_than_20_prior_iv_sessions"})
            continue
        vol_ratio = max(1.0, atm_iv / float(np.median(prior_sessions[-20:])))
        try:
            legs, _ = build_structure(chain, structure)
        except ValueError as exc:
            ledger.append({**base, "status": "rejected", "reason": str(exc)})
            continue
        filled, reason = atomic_fill(legs)
        if not filled:
            ledger.append({**base, "status": "rejected", "reason": reason})
            continue
        lot = 15 if lot_mode == "constant15" else historical_lot_size(day)
        try:
            max_loss, premium_debit = defined_risk(legs, lot, structure)
            # Pretrade uncertainty uses a symmetric close at entry premiums.
            entry_cost = _entry_costs(legs, day, lot, vol_ratio, 0.0)
            synthetic_close = execution_cost(
                _orders(legs, lot, exit_prices={(x.option_type, x.strike): x.entry_price for x in legs}),
                day, volatility_ratio=vol_ratio, liquidity=LIQUIDITY,
            )
            severe_loss = (
                max_loss + uncertainty_cost(entry_cost).total
                + uncertainty_cost(synthetic_close).total
            )
            margin = margin_proxy(
                capital=CAPITAL, spot=float(atm.spot_price), lot_size=lot, lots=1,
                holding_hours=2.0, has_uncovered_short=False,
                premium_debit=premium_debit, defined_max_loss=max_loss,
            )
        except (ArithmeticError, ValueError, TypeError) as exc:
            ledger.append({**base, "status": "rejected",
                           "reason": f"invalid_cost_or_margin_input:{type(exc).__name__}"})
            continue
        budget_rejection = proposed_loss_rejection(risk_state, day, severe_loss)
        if budget_rejection is not None:
            ledger.append({**base, "status": "rejected", "reason": budget_rejection,
                           "defined_max_loss": max_loss, "severe_loss": severe_loss,
                           "day_pnl_before": risk_state.day_pnl,
                           "week_pnl_before": risk_state.week_pnl,
                           "drawdown_before": risk_state.drawdown})
            continue
        if not margin.permitted or severe_loss > 7_500:
            why = "buffered_margin_cap" if not margin.permitted else "trade_stress_loss_cap"
            ledger.append({**base, "status": "rejected", "reason": why,
                           "defined_max_loss": max_loss, "severe_loss": severe_loss,
                           "buffered_margin_proxy": margin.buffered_requirement,
                           "margin_status": "unverified_margin_proxy",
                           "verified_margin_pass": False, "promotion_eligible": False})
            continue
        exit_time, exit_prices = _fixed_exit(
            day_options, decision_time, base["entry_time"], base["expiry"], legs
        )
        carry = margin.carry_cost
        entry_cost = _entry_costs(legs, day, lot, vol_ratio, carry)
        row = {
            **base, "status": "filled", "reason": "", "lot_size": lot,
            "volatility_ratio": vol_ratio, "defined_max_loss": max_loss,
            "severe_loss": severe_loss, "raw_margin_proxy": margin.base_requirement,
            "buffered_margin_proxy": margin.buffered_requirement,
            "buffered_margin": margin.buffered_requirement,
            "margin_status": "unverified_margin_proxy", "verified_margin_pass": False,
            "lot_verification_status": ("unverified_lot_assumption" if lot_mode == "historical"
                                        else "constant15_sensitivity_not_live"),
            "live_eligible": False, "promotion_eligible": False,
            "entry_count": 1, "missing_exit": exit_prices is None,
            "entry_strikes": "|".join(str(x.strike) for x in legs),
        }
        if exit_prices is None:
            adverse_gross = -max_loss if structure == "iron_fly" else -premium_debit
            stressed_close = uncertainty_cost(synthetic_close)
            settlement = settlement_cost(
                trade_date=day, long_intrinsic_points=0.0,
                spot_settlement=float(atm.spot_price), lot_size=lot, lots=1,
                has_unclosed_short=structure == "iron_fly",
            )
            ordinary = CostBreakdown(**{
                field.name: getattr(entry_cost, field.name) + getattr(stressed_close, field.name)
                for field in fields(entry_cost)
            })
            ordinary = CostBreakdown(**{
                **asdict(ordinary),
                "stt": ordinary.stt + settlement.exercise_stt,
                "impact_slippage": ordinary.impact_slippage + settlement.conservative_gap_reserve,
            })
            row.update({"exit_time": pd.NaT, "gross_pnl": adverse_gross,
                        "exit_treatment": "contractual_adverse_missing_exit",
                        "settlement_adverse_cost": settlement.total,
                        "turnover_rupees": 2.0 * sum(abs(x.entry_price * lot) for x in legs),
                        "turnover_treatment": "entry_plus_conservative_synthetic_close"})
        else:
            holding_hours = (exit_time - base["entry_time"]).total_seconds() / 3600.0
            carry = margin.buffered_requirement * 0.08 * holding_hours / (365 * 24)
            ordinary = execution_cost(
                [*_orders(legs, lot), *_orders(legs, lot, exit_prices=exit_prices)],
                day, volatility_ratio=vol_ratio, liquidity=LIQUIDITY, margin_carry=carry,
            )
            gross = sum(
                leg.qty * (exit_prices[(leg.option_type, leg.strike)] - leg.entry_price) * lot
                for leg in legs
            )
            row.update({"exit_time": exit_time, "gross_pnl": gross, "exit_treatment": "observed_fixed_strikes"})
            row.update({
                "turnover_rupees": sum(abs(x.entry_price * lot) for x in legs)
                + sum(abs(exit_prices[(x.option_type, x.strike)] * lot) for x in legs),
                "turnover_treatment": "entry_plus_observed_exit",
            })
        stress2 = ordinary.stressed(2.0)
        uncertain = uncertainty_cost(ordinary)
        row.update({
            "ordinary_cost": ordinary.total, "cost_2x": stress2.total,
            "uncertainty_cost": uncertain.total,
            "ordinary_net": row["gross_pnl"] - ordinary.total,
            "net_2x": row["gross_pnl"] - stress2.total,
            "uncertainty_net": row["gross_pnl"] - uncertain.total,
            **{f"cost_{key}": value for key, value in asdict(ordinary).items()},
        })
        update_risk_state(risk_state, day, float(row["ordinary_net"]))
        row.update({
            "daily_loss_breach": row["ordinary_net"] <= -10_000,
            "weekly_loss_breach": risk_state.week_pnl <= -25_000,
            "drawdown_breach": risk_state.drawdown >= 50_000,
            "week_pnl_after": risk_state.week_pnl,
            "day_pnl_after": risk_state.day_pnl,
            "drawdown_after": risk_state.drawdown,
        })
        ledger.append(row)
    return pd.DataFrame(ledger)


def register_strategy_trial(*, segment: str, threshold: float, lot_mode: str,
                            event_counterfactual: bool, spec_hash: str) -> str:
    trial_id = (
        f"STRATEGY-V2-{segment}-t{threshold:.2f}-{lot_mode}-"
        f"{'eventdiag' if event_counterfactual else 'primary'}-{pd.Timestamp.now(tz='UTC').value}"
    )
    append_global_trial(GLOBAL_TRIALS, {
        "trial_id": trial_id, "hypothesis": "H1|H4",
        "registered_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "status": "running", "structure": "iron_fly|long_straddle",
        "horizon_min": 120, "decision_time": "10:15|11:45|13:25",
        "dte_band": "nearest_recorded_monthly",
        "parameter_grid": f"threshold={threshold};lot_mode={lot_mode};event_counterfactual={event_counterfactual}",
        "outcome_opened": False,
        "notes": f"strategy_spec_sha256={spec_hash}; registry append precedes metrics",
    })
    return trial_id


def complete_strategy_trial(parent_trial_id: str, *, segment: str, threshold: float,
                            lot_mode: str, event_counterfactual: bool,
                            row_count: int, spec_hash: str) -> None:
    """Append an immutable terminal record; never mutate the preregistration."""
    append_global_trial(GLOBAL_TRIALS, {
        "trial_id": f"{parent_trial_id}-COMPLETION", "hypothesis": "H1|H4",
        "registered_at": pd.Timestamp.now(tz="Asia/Kolkata").isoformat(),
        "status": "completed", "structure": "iron_fly|long_straddle",
        "horizon_min": 120, "decision_time": "10:15|11:45|13:25",
        "dte_band": "nearest_recorded_monthly",
        "parameter_grid": f"threshold={threshold};lot_mode={lot_mode};event_counterfactual={event_counterfactual}",
        "outcome_opened": True,
        "notes": (f"completion_for={parent_trial_id};segment={segment};rows={row_count};"
                  f"strategy_spec_sha256={spec_hash};immutable_completion_record"),
    })


def _checked_json_with_sidecar(path: Path) -> tuple[dict, str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.exists() or not sidecar.exists():
        raise PermissionError(f"frozen file or checksum absent: {path.name}")
    expected = sidecar.read_text(encoding="ascii").strip().split()[0].lower()
    actual = sha256_file(path)
    if actual != expected:
        raise PermissionError(f"frozen checksum mismatch: {path.name}")
    return json.loads(path.read_text(encoding="utf-8")), actual


def _load_pseudo_predictions(path: Path) -> pd.DataFrame:
    con = duckdb.connect()
    try:
        frame = con.execute(
            "SELECT CAST(decision_time AS VARCHAR) AS decision_time, p2_forecast "
            "FROM read_parquet(?) ORDER BY decision_time", [str(path)]
        ).fetchdf()
    finally:
        con.close()
    frame["decision_time"] = pd.to_datetime(frame.decision_time, utc=True).dt.tz_convert("Asia/Kolkata")
    clocks = list(zip(frame.decision_time.dt.hour, frame.decision_time.dt.minute))
    frame = frame.loc[[item in DECISION_CLOCKS for item in clocks]].copy()
    if frame.empty or not np.isfinite(frame.p2_forecast).all() or not frame.p2_forecast.gt(0).all():
        raise RuntimeError("invalid pseudo prediction ledger")
    return frame


def run_segment(segment: Literal["development", "validation"], event_calendar: Path,
                leakage_authorization: Path) -> None:
    spec_hash = verify_strategy_spec()
    verify_risk_addendum()
    verify_run_authorization(segment, event_calendar, leakage_authorization)
    OUT.mkdir(parents=True, exist_ok=True)
    # Register every sleeve before opening a real prediction row or option row.
    trial_ids = {
        (threshold, lot_mode, False): register_strategy_trial(
            segment=segment, threshold=threshold, lot_mode=lot_mode,
            event_counterfactual=False, spec_hash=spec_hash,
        )
        for threshold in THRESHOLDS
        for lot_mode in ("historical", "constant15")
    }
    trial_ids[(1.25, "historical", True)] = register_strategy_trial(
        segment=segment, threshold=1.25, lot_mode="historical",
        event_counterfactual=True, spec_hash=spec_hash,
    )
    predictions = load_approved_predictions(segment)
    events, event_hash = load_event_calendar(event_calendar)
    # Warm the causal 20-session IV baseline without expanding decision dates.
    option_start = predictions.decision_time.min().date() - timedelta(days=45)
    options = read_option_rows(option_start, predictions.decision_time.max().date())
    for threshold in THRESHOLDS:
        for lot_mode in ("historical", "constant15"):
            trial_id = trial_ids[(threshold, lot_mode, False)]
            ledger = evaluate_policy(
                predictions, options, events, threshold=threshold,
                lot_mode=lot_mode, segment=segment,
            )
            # Metrics intentionally belong to the authorized runner stage.
            ledger["trial_id"] = trial_id
            ledger["event_calendar_sha256"] = event_hash
            path = OUT / f"{segment}_t{threshold:.2f}_{lot_mode}_ledger.parquet"
            con = duckdb.connect()
            con.register("ledger", ledger)
            con.execute("COPY ledger TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
            con.close()
            complete_strategy_trial(
                trial_id, segment=segment, threshold=threshold, lot_mode=lot_mode,
                event_counterfactual=False, row_count=len(ledger), spec_hash=spec_hash,
            )
    diagnostic = evaluate_policy(
        predictions, options, events, threshold=1.25, lot_mode="historical",
        segment=segment, event_counterfactual=True,
    )
    diagnostic["trial_id"] = trial_ids[(1.25, "historical", True)]
    diagnostic["event_calendar_sha256"] = event_hash
    diagnostic["diagnostic_only"] = True
    diagnostic["promotion_eligible"] = False
    path = OUT / f"{segment}_t1.25_historical_event_counterfactual_ledger.parquet"
    con = duckdb.connect()
    try:
        con.register("ledger", diagnostic)
        con.execute("COPY ledger TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    finally:
        con.close()
    complete_strategy_trial(
        trial_ids[(1.25, "historical", True)], segment=segment, threshold=1.25,
        lot_mode="historical", event_counterfactual=True,
        row_count=len(diagnostic), spec_hash=spec_hash,
    )


def open_pseudo_strategy_once(*, frozen_manifest: Path, authorization: Path,
                              event_calendar: Path, marker: Path) -> None:
    """Consume the pseudo strategy opening atomically before any pseudo scan.

    Deliberately has no CLI route.  An integrator must freeze a manifest of the
    strategy runner, spec, costs, event calendar and thresholds, then receive a
    separate leakage authorization before invoking this function once.
    """
    manifest, manifest_hash = _checked_json_with_sidecar(frozen_manifest)
    auth, _ = _checked_json_with_sidecar(authorization)
    if (manifest.get("scope") != "frozen_research_v2_pseudo_strategy"
            or manifest.get("thresholds") != list(THRESHOLDS)
            or manifest.get("lot_modes") != ["historical", "constant15"]):
        raise PermissionError("pseudo strategy manifest scope mismatch")
    if (auth.get("approved") is not True
            or auth.get("scope") != "research_v2_pseudo_strategy_once"
            or auth.get("pseudo_authorized") is not True
            or auth.get("manifest_sha256") != manifest_hash):
        raise PermissionError("pseudo strategy authorization mismatch")
    inventory = manifest.get("inventory", {})
    pseudo_predictions = P2_DIR / "pseudo_holdout_once_predictions.parquet"
    strategy_gate = ROOT / "STRATEGY_DEV_VALIDATION_GATE_V2.json"
    required = [
        SPEC, RISK_ADDENDUM, event_calendar, Path(__file__).resolve(),
        ROOT / "research/execution_costs.py", ROOT / "tests/test_research_v2_strategy.py",
        pseudo_predictions, OPTION_SOURCE, strategy_gate,
    ]
    for path in required:
        key = _relative(path)
        if not path.exists() or inventory.get(key) != sha256_file(path):
            raise PermissionError(f"pseudo frozen inventory mismatch: {key}")
    if event_calendar.resolve() != EVENT_CALENDAR.resolve() or sha256_file(event_calendar) != EVENT_CALENDAR_SHA256:
        raise PermissionError("pseudo calendar mismatch")
    gate = json.loads(strategy_gate.read_text(encoding="utf-8"))
    if (gate.get("development_strategy_gate_pass") is not True
            or gate.get("validation_strategy_gate_pass") is not True
            or gate.get("strategy_logic_frozen") is not True
            or gate.get("strategy_runner_sha256") != sha256_file(Path(__file__).resolve())):
        raise PermissionError("development and validation strategy gate has not passed")

    # Integrity hashing above does not deserialize either outcome table.  The
    # durable marker is consumed before the first pseudo/option row access.
    marker.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump({
            "status": "CONSUMED_BEFORE_SCAN", "frozen_manifest_sha256": manifest_hash,
            "authorization_sha256": sha256_file(authorization),
            "event_calendar_sha256": sha256_file(event_calendar),
            "opened_at": pd.Timestamp.now(tz="UTC").isoformat(),
        }, handle, indent=2)
        handle.flush(); os.fsync(handle.fileno())

    spec_hash = verify_strategy_spec()
    trial_ids = {
        (threshold, lot_mode, False): register_strategy_trial(
            segment="pseudo_holdout_previously_exposed", threshold=threshold, lot_mode=lot_mode,
            event_counterfactual=False, spec_hash=spec_hash,
        )
        for threshold in THRESHOLDS
        for lot_mode in ("historical", "constant15")
    }
    trial_ids[(1.25, "historical", True)] = register_strategy_trial(
        segment="pseudo_holdout_previously_exposed", threshold=1.25, lot_mode="historical",
        event_counterfactual=True, spec_hash=spec_hash,
    )
    # The first semantic reads of pseudo predictions/options occur only here.
    predictions = _load_pseudo_predictions(pseudo_predictions)
    events, event_hash = load_event_calendar(event_calendar)
    option_start = predictions.decision_time.min().date() - timedelta(days=45)
    options = read_option_rows(option_start, predictions.decision_time.max().date())
    OUT.mkdir(parents=True, exist_ok=True)
    for threshold in THRESHOLDS:
        for lot_mode in ("historical", "constant15"):
            ledger = evaluate_policy(
                predictions, options, events, threshold=threshold,
                lot_mode=lot_mode, segment="pseudo_holdout_previously_exposed",
            )
            ledger["trial_id"] = trial_ids[(threshold, lot_mode, False)]
            ledger["event_calendar_sha256"] = event_hash
            path = OUT / f"pseudo_t{threshold:.2f}_{lot_mode}_ledger.parquet"
            con = duckdb.connect()
            try:
                con.register("ledger", ledger)
                con.execute("COPY ledger TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
            finally:
                con.close()
            complete_strategy_trial(
                trial_ids[(threshold, lot_mode, False)],
                segment="pseudo_holdout_previously_exposed", threshold=threshold,
                lot_mode=lot_mode, event_counterfactual=False,
                row_count=len(ledger), spec_hash=spec_hash,
            )
    diagnostic = evaluate_policy(
        predictions, options, events, threshold=1.25, lot_mode="historical",
        segment="pseudo_holdout_previously_exposed", event_counterfactual=True,
    )
    diagnostic["trial_id"] = trial_ids[(1.25, "historical", True)]
    diagnostic["event_calendar_sha256"] = event_hash
    diagnostic["diagnostic_only"] = True
    diagnostic["promotion_eligible"] = False
    path = OUT / "pseudo_t1.25_historical_event_counterfactual_ledger.parquet"
    con = duckdb.connect()
    try:
        con.register("ledger", diagnostic)
        con.execute("COPY ledger TO ? (FORMAT PARQUET, COMPRESSION ZSTD)", [str(path)])
    finally:
        con.close()
    complete_strategy_trial(
        trial_ids[(1.25, "historical", True)],
        segment="pseudo_holdout_previously_exposed", threshold=1.25,
        lot_mode="historical", event_counterfactual=True,
        row_count=len(diagnostic), spec_hash=spec_hash,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segment", choices=("development", "validation"), required=True)
    parser.add_argument("--event-calendar", type=Path, required=True)
    parser.add_argument("--leakage-authorization", type=Path, required=True)
    args = parser.parse_args()
    run_segment(args.segment, args.event_calendar, args.leakage_authorization)


if __name__ == "__main__":
    main()
