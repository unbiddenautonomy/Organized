"""Fail-closed statistics for authorized Research V2 strategy ledgers.

This module contains no strategy runner and never discovers ledgers.  Callers
must either pass an in-memory ledger or explicitly authorize exact ledger
paths and SHA-256 hashes through :func:`load_authorized_ledgers`.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, rankdata, skew


ANNUAL_TRADING_DAYS = 252
PRIMARY_THRESHOLD = 1.25
PLATEAU_THRESHOLDS = (1.15, 1.25, 1.35)
PSEUDO_SEGMENT = "pseudo_holdout_previously_exposed"
PSEUDO_SAMPLE_LABEL = "pseudo-holdout, previously exposed"
COST_COMPONENTS = (
    "brokerage", "exchange", "stt", "gst", "sebi", "ipft", "stamp",
    "bid_ask_crossing", "impact_slippage", "margin_carry",
)
BASE_REQUIRED = {
    "segment", "sample_label", "trial_id", "decision_id", "decision_time", "eligible_session",
    "expected_decision_count", "decision_universe_sha256",
    "status", "event_day", "missing_exit",
}
FILLED_REQUIRED = {
    "gross_pnl", "ordinary_net", "net_2x", "uncertainty_net", "ordinary_cost",
    "cost_2x", "uncertainty_cost", "buffered_margin", "entry_count", "turnover_rupees",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_authorized_ledgers(
    paths: Iterable[Path], authorization_path: Path, *, allow_pseudo: bool = False
) -> pd.DataFrame:
    """Read only exact, hash-authorized Parquet ledgers; never auto-discover."""
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    expected_scope = "pseudo_strategy_evaluation" if allow_pseudo else "development_validation_strategy_evaluation"
    if authorization.get("authorized") is not True or authorization.get("scope") != expected_scope:
        raise PermissionError(f"authorization scope must be {expected_scope}")
    expected_segments = [PSEUDO_SEGMENT] if allow_pseudo else ["development", "validation"]
    if sorted(map(str, authorization.get("allowed_segments", []))) != sorted(expected_segments):
        raise PermissionError(f"authorization must bind allowed_segments to {expected_segments}")
    recorded = authorization.get("ledger_sha256")
    if not isinstance(recorded, dict) or not recorded:
        raise PermissionError("authorization must bind ledger paths to SHA-256 hashes")
    frames = []
    for input_path in paths:
        path = Path(input_path).resolve()
        key = path.as_posix()
        if "pseudo" in key.lower() and not allow_pseudo:
            raise PermissionError("pseudo strategy ledger is not authorized")
        expected = recorded.get(key)
        if expected is None or sha256_file(path) != str(expected).lower():
            raise PermissionError(f"ledger is absent from authorization or hash mismatched: {path}")
        connection = duckdb.connect()
        try:
            frames.append(connection.execute("SELECT * FROM read_parquet(?)", [str(path)]).fetchdf())
        finally:
            connection.close()
    if not frames:
        raise ValueError("no authorized ledgers supplied")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "segment" not in combined or not set(combined["segment"].astype(str).unique()).issubset(expected_segments):
        raise PermissionError("authorized ledger contains a segment outside its bound scope")
    return combined


def trial_registry_counts(path: Path) -> dict[str, int]:
    """Validate all registry IDs and separate substantive from administration."""
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "trial_id" not in reader.fieldnames:
            raise ValueError("TRIALS.csv lacks trial_id")
        rows = list(reader)
        identifiers = [str(row["trial_id"]) for row in rows]
    if not identifiers or any(not value for value in identifiers):
        raise ValueError("TRIALS.csv is empty or has blank trial IDs")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("TRIALS.csv has duplicate trial IDs")
    administrative = sum(
        identifier.endswith("-COMPLETION") or "completion_for=" in str(row.get("notes", ""))
        for identifier, row in zip(identifiers, rows)
    )
    substantive = len(rows) - administrative
    if substantive < 1:
        raise ValueError("TRIALS.csv has no substantive trials after excluding administrative records")
    return {"substantive_trials": substantive, "administrative_records": administrative,
            "total_records": len(rows)}


def full_trial_count(path: Path) -> int:
    """Return substantive multiplicity for DSR, excluding completion records."""
    return trial_registry_counts(path)["substantive_trials"]


def validate_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(BASE_REQUIRED.difference(ledger.columns))
    if missing:
        raise ValueError(f"ledger missing required columns: {missing}")
    out = ledger.copy()
    if out.empty:
        raise ValueError("ledger is empty")
    out["decision_time"] = pd.to_datetime(out["decision_time"], errors="raise", utc=True)
    if out[["segment", "trial_id", "status"]].isna().any().any():
        raise ValueError("segment, trial_id, and status must be populated")
    if out["sample_label"].isna().any() or out["sample_label"].astype(str).eq("").any():
        raise ValueError("sample_label must be populated")
    pseudo = out["segment"].astype(str).eq(PSEUDO_SEGMENT)
    if pseudo.any() and not out.loc[pseudo, "sample_label"].astype(str).eq(PSEUDO_SAMPLE_LABEL).all():
        raise ValueError(f"pseudo rows require sample_label exactly {PSEUDO_SAMPLE_LABEL!r}")
    if out["segment"].astype(str).eq("pseudo").any():
        raise ValueError(f"noncanonical pseudo segment; use {PSEUDO_SEGMENT}")
    id_is_eventdiag = out["trial_id"].astype(str).str.contains("-eventdiag-", regex=False)
    if "diagnostic_only" not in out:
        if id_is_eventdiag.any():
            raise ValueError("eventdiag trial IDs require an explicit diagnostic_only column")
        out["diagnostic_only"] = False
    else:
        diagnostic_raw = out["diagnostic_only"]
        invalid_diagnostic = diagnostic_raw.dropna().map(
            lambda value: not isinstance(value, (bool, np.bool_))
        )
        if invalid_diagnostic.any():
            raise ValueError("diagnostic_only must be boolean, false, or null for primary rows")
        out["diagnostic_only"] = diagnostic_raw.fillna(False).astype(bool)
    if not out["diagnostic_only"].eq(id_is_eventdiag).all():
        raise ValueError("diagnostic_only must exactly match the eventdiag trial-id designation")
    out["event_counterfactual"] = out["diagnostic_only"]
    if not set(out["status"].astype(str).unique()).issubset({"filled", "rejected", "no_trade"}):
        raise ValueError("status must be filled, rejected, or no_trade")
    if out["decision_id"].isna().any() or out["decision_id"].astype(str).eq("").any():
        raise ValueError("decision_id must be populated")
    out["eligible_session"] = out["eligible_session"].astype(bool)
    if not out["eligible_session"].all():
        raise ValueError("ledger may contain only approved eligible-session decisions")
    if out.duplicated(["segment", "trial_id", "decision_id"]).any():
        raise ValueError("duplicate decision_id within a segment/trial")
    filled = out["status"].eq("filled")
    missing_filled = sorted(FILLED_REQUIRED.difference(out.columns))
    if filled.any() and missing_filled:
        raise ValueError(f"filled ledger missing columns: {missing_filled}")
    if filled.any():
        numeric = out.loc[filled, sorted(FILLED_REQUIRED)].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(float)).all():
            raise ValueError("filled ledger has nonfinite required values")
        out.loc[filled, sorted(FILLED_REQUIRED)] = numeric
        if "exit_treatment" not in out or out.loc[filled, "exit_treatment"].fillna("").astype(str).eq("").any():
            raise ValueError("every filled row must record exit_treatment")
    for flag in ("event_day", "missing_exit"):
        if out[flag].isna().any():
            raise ValueError(f"{flag} must be populated")
        out[flag] = out[flag].astype(bool)
    return out.sort_values("decision_time", kind="stable").reset_index(drop=True)


def decision_universe_sha256(decision_ids: Iterable[str]) -> str:
    identifiers = sorted(set(map(str, decision_ids)))
    payload = "".join(f"{identifier}\n" for identifier in identifiers).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_decision_coverage(ledger: pd.DataFrame) -> dict[str, Any]:
    """Verify absolute pre-evaluation universe commitments and cross-trial coverage."""
    work = validate_ledger(ledger)
    coverage: dict[str, Any] = {}
    for segment, part in work.groupby("segment", sort=True):
        sets = {
            str(trial): frozenset(group["decision_id"].astype(str))
            for trial, group in part.groupby("trial_id", sort=True)
        }
        for trial, group in part.groupby("trial_id", sort=True):
            counts = pd.to_numeric(group["expected_decision_count"], errors="coerce").dropna().unique()
            digests = group["decision_universe_sha256"].dropna().astype(str).unique()
            if len(counts) != 1 or len(digests) != 1:
                raise ValueError(f"decision-universe commitment is not constant for {segment}/{trial}")
            actual_ids = sets[str(trial)]
            actual_digest = decision_universe_sha256(actual_ids)
            if int(counts[0]) != len(actual_ids) or str(digests[0]).lower() != actual_digest:
                raise ValueError(f"absolute decision coverage failed for {segment}/{trial}")
        reference_trial = sorted(sets)[0]
        reference = sets[reference_trial]
        mismatches = {
            trial: {"missing": len(reference - values), "extra": len(values - reference)}
            for trial, values in sets.items() if values != reference
        }
        if mismatches:
            raise ValueError(f"incomplete decision coverage in segment {segment}: {mismatches}")
        coverage[str(segment)] = {
            "eligible_decisions_per_trial": len(reference), "trial_count": len(sets),
            "coverage_pass": True, "absolute_decision_coverage_verified": True,
            "decision_universe_sha256": decision_universe_sha256(reference),
        }
    return coverage


def _daily_returns(group: pd.DataFrame, column: str = "ordinary_net") -> pd.Series:
    values = np.where(group["status"].eq("filled"), group[column].fillna(0.0).astype(float), 0.0)
    days = group["decision_time"].dt.tz_convert("Asia/Kolkata").dt.normalize()
    return pd.Series(values, index=days).groupby(level=0).sum().sort_index()


def annualized_daily_sharpe(daily: Iterable[float]) -> dict[str, Any]:
    """Annualize active-ledger session returns with sqrt(252), sample SD ddof=1."""
    values = np.asarray(list(daily), dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return {"annualized_sharpe": None, "reason": "fewer than two ledger session returns"}
    standard_deviation = float(np.std(values, ddof=1))
    if standard_deviation <= 0:
        return {"annualized_sharpe": None, "reason": "zero daily sample standard deviation"}
    return {
        "annualized_sharpe": float(np.mean(values) / standard_deviation * math.sqrt(ANNUAL_TRADING_DAYS)),
        "reason": None,
        "convention": "Asia/Kolkata ledger sessions; rejected-only sessions are zero; absent sessions omitted; sqrt(252); ddof=1",
    }


def probabilistic_and_deflated_sharpe(daily: Iterable[float], trial_count: int) -> dict[str, Any]:
    """Return PSR(SR>0) and a conservative independent-null deflated PSR."""
    values = np.asarray(list(daily), dtype=float)
    values = values[np.isfinite(values)]
    if trial_count < 1:
        raise ValueError("full trial count must be positive")
    if len(values) < 3 or np.std(values, ddof=1) <= 0:
        return {"estimable": False, "reason": "requires >=3 nonconstant daily returns",
                "trial_count": trial_count, "probabilistic_sharpe": None, "deflated_sharpe": None}
    sr = float(np.mean(values) / np.std(values, ddof=1))
    sample_skew = float(skew(values, bias=False))
    sample_kurtosis = float(kurtosis(values, fisher=False, bias=False)) if len(values) >= 4 else 3.0
    denominator = math.sqrt(max(
        1e-15, 1.0 - sample_skew * sr + ((sample_kurtosis - 1.0) / 4.0) * sr * sr
    ))
    psr = float(norm.cdf(sr * math.sqrt(len(values) - 1) / denominator))
    if trial_count == 1:
        expected_max_z = 0.0
    else:
        euler_gamma = 0.5772156649015329
        expected_max_z = (
            (1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trial_count)
            + euler_gamma * norm.ppf(1.0 - 1.0 / (trial_count * math.e))
        )
    deflated_benchmark_daily = float(expected_max_z / math.sqrt(len(values) - 1))
    dsr = float(norm.cdf(
        (sr - deflated_benchmark_daily) * math.sqrt(len(values) - 1) / denominator
    ))
    return {
        "estimable": True, "reason": None, "trial_count": int(trial_count),
        "daily_sharpe": sr, "annualized_sharpe": sr * math.sqrt(ANNUAL_TRADING_DAYS),
        "probabilistic_sharpe": psr, "deflated_sharpe": dsr,
        "deflated_benchmark_daily_sharpe": deflated_benchmark_daily,
        "skew": sample_skew, "pearson_kurtosis": sample_kurtosis,
        "caveat": "Deflation treats every full-registry trial as an independent null draw; correlated trials reduce effective multiplicity.",
    }


def _historical_loss_cvar95(daily: pd.Series) -> tuple[float | None, float | None]:
    if daily.empty:
        return None, None
    losses = -daily.to_numpy(float)
    tail_count = max(1, math.ceil(0.05 * len(losses)))
    # Stable sorting makes the boundary deterministic; exactly tail_count
    # observations are used, so zero ties cannot dilute expected shortfall.
    ordered = np.sort(losses, kind="stable")[::-1]
    tail = ordered[:tail_count]
    return float(tail[-1]), float(tail.mean())


def block_bootstrap_cvar95_ucb(
    daily: Iterable[float], *, block_length: int = 5, replications: int = 1000, seed: int = 7319
) -> dict[str, Any]:
    """Circular-block bootstrap one-sided 95% UCB for daily loss CVaR95."""
    values = np.asarray(list(daily), dtype=float)
    values = values[np.isfinite(values)]
    if block_length < 1 or replications < 100:
        raise ValueError("block length must be positive and replications at least 100")
    if len(values) < max(20, block_length * 2):
        return {"estimable": False, "reason": "requires at least 20 daily observations",
                "ucb95": None, "block_length": block_length, "replications": replications}
    rng = np.random.default_rng(seed)
    blocks_needed = math.ceil(len(values) / block_length)
    estimates = np.empty(replications)
    for replication in range(replications):
        starts = rng.integers(0, len(values), size=blocks_needed)
        sample = np.concatenate([
            values[(start + np.arange(block_length)) % len(values)] for start in starts
        ])[:len(values)]
        losses = -sample
        tail_count = max(1, math.ceil(0.05 * len(losses)))
        estimates[replication] = np.sort(losses, kind="stable")[::-1][:tail_count].mean()
    return {"estimable": True, "reason": None,
            "ucb95": float(np.quantile(estimates, 0.95, method="higher")),
            "block_length": block_length, "replications": replications, "seed": seed}


def summarize_ledgers(
    ledger: pd.DataFrame, trial_count: int, *, assumption_metadata: dict[str, bool] | None = None
) -> pd.DataFrame:
    work = validate_ledger(ledger)
    validate_decision_coverage(work)
    assumptions = {
        "margin_verified": False, "lot_size_verified": False, "execution_verified": False,
        **(assumption_metadata or {}),
    }
    if set(assumptions) != {"margin_verified", "lot_size_verified", "execution_verified"}:
        raise ValueError("assumption_metadata must contain only margin/lot-size/execution verification flags")
    rows: list[dict[str, Any]] = []
    for (segment, trial_id), group in work.groupby(["segment", "trial_id"], sort=True):
        labels = group["sample_label"].astype(str).unique()
        if len(labels) != 1:
            raise ValueError("sample_label must be constant within segment/trial")
        filled = group[group["status"].eq("filled")].copy()
        rejected = group[group["status"].eq("rejected")]
        no_trade = group[group["status"].eq("no_trade")]
        daily = _daily_returns(group)
        period_index = daily.index.tz_localize(None) if daily.index.tz is not None else daily.index
        weekly = daily.groupby(period_index.to_period("W-FRI")).sum()
        monthly = daily.groupby(period_index.to_period("M")).sum()
        equity = np.r_[0.0, daily.cumsum().to_numpy(float)]
        max_drawdown = float(np.max(np.maximum.accumulate(equity) - equity))
        var95, cvar95 = _historical_loss_cvar95(daily)
        positive_months = monthly.clip(lower=0.0)
        month_concentration = (
            float(positive_months.max() / positive_months.sum()) if positive_months.sum() > 0 else None
        )
        sharpe = annualized_daily_sharpe(daily)
        adjusted = probabilistic_and_deflated_sharpe(daily, trial_count)
        bootstrap = block_bootstrap_cvar95_ucb(daily)
        event = group[group["event_day"]]
        event_filled = event[event["status"].eq("filled")]
        row: dict[str, Any] = {
            "segment": str(segment), "trial_id": str(trial_id),
            "sample_label": labels[0],
            "eligible_decisions": int(len(group)), "filled": int(len(filled)),
            "rejected": int(len(rejected)), "no_trade": int(len(no_trade)),
            "missing_exits": int(filled["missing_exit"].sum()) if not filled.empty else 0,
            "unrecorded_exits": int((
                filled["exit_treatment"].fillna("").astype(str).eq("")
                | (filled["missing_exit"] & filled["exit_treatment"].astype(str).ne(
                    "contractual_adverse_missing_exit"
                ))
            ).sum()) if not filled.empty else 0,
            "gross_total": float(filled["gross_pnl"].sum()) if not filled.empty else 0.0,
            "ordinary_total": float(filled["ordinary_net"].sum()) if not filled.empty else 0.0,
            "net_2x_total": float(filled["net_2x"].sum()) if not filled.empty else 0.0,
            "uncertainty_total": float(filled["uncertainty_net"].sum()) if not filled.empty else 0.0,
            "ordinary_cost_total": float(filled["ordinary_cost"].sum()) if not filled.empty else 0.0,
            "cost_2x_total": float(filled["cost_2x"].sum()) if not filled.empty else 0.0,
            "uncertainty_cost_total": float(filled["uncertainty_cost"].sum()) if not filled.empty else 0.0,
            "worst_day": float(daily.min()), "worst_week": float(weekly.min()),
            "max_drawdown": max_drawdown, "historical_loss_var95": var95,
            "historical_loss_cvar95": cvar95,
            "win_rate": float((filled["ordinary_net"] > 0).mean()) if not filled.empty else None,
            "turnover_rupees": float(filled["turnover_rupees"].sum()) if not filled.empty else 0.0,
            "filled_entry_count": float(filled["entry_count"].sum()) if not filled.empty else 0.0,
            "max_margin": float(filled["buffered_margin"].max()) if not filled.empty else 0.0,
            "max_positive_month_profit_concentration": month_concentration,
            "event_selected": int(len(event)), "event_filled": int(len(event_filled)),
            "event_rejected": int(event["status"].eq("rejected").sum()),
            "event_ordinary_total": float(event_filled["ordinary_net"].sum()) if not event_filled.empty else 0.0,
            "annualized_sharpe": sharpe["annualized_sharpe"], "sharpe_reason": sharpe["reason"],
            "probabilistic_sharpe": adjusted["probabilistic_sharpe"],
            "deflated_sharpe": adjusted["deflated_sharpe"],
            "sharpe_adjustment_reason": adjusted["reason"], "full_trial_count": int(trial_count),
            "cvar95_block_bootstrap_ucb": bootstrap["ucb95"],
            "cvar95_block_bootstrap_estimable": bootstrap["estimable"],
            "cvar95_block_bootstrap_reason": bootstrap["reason"],
            "daily_loss_breaches": int(filled.get("daily_loss_breach", pd.Series(False, index=filled.index)).astype(bool).sum()),
            "weekly_loss_breaches": int(filled.get("weekly_loss_breach", pd.Series(False, index=filled.index)).astype(bool).sum()),
            "drawdown_breaches": int(filled.get("drawdown_breach", pd.Series(False, index=filled.index)).astype(bool).sum()),
            "eligible_sessions": int(group["decision_time"].dt.tz_convert("Asia/Kolkata").dt.normalize().nunique()),
            "historical_size_500_sessions_pass": bool(
                group["decision_time"].dt.tz_convert("Asia/Kolkata").dt.normalize().nunique() >= 500
            ),
            "historical_size_100_fills_pass": bool(len(filled) >= 100),
            "cvar_ucb_20k_pass": bool(bootstrap["estimable"] and bootstrap["ucb95"] <= 20_000),
            "required_new_12m_forward_sample_satisfied": False,
            "forward_sample_reason": "requires a future quarantined approximately 12-month sample; historical rows cannot satisfy it",
            "research_only": True, "verified_backtest": False, "live_eligible": False,
            **assumptions,
        }
        for component in COST_COMPONENTS:
            column = f"cost_{component}"
            row[f"{column}_total"] = float(filled[column].sum()) if column in filled else None
        for metadata in ("threshold", "lot_mode", "diagnostic_only", "event_counterfactual"):
            if metadata in group:
                values = group[metadata].dropna().unique()
                if len(values) != 1:
                    raise ValueError(f"{metadata} must be constant within segment/trial")
                row[metadata] = values[0]
        rows.append(row)
    return pd.DataFrame(rows)


def estimate_cscv_pbo(ledger: pd.DataFrame, *, segment: str, blocks: int = 8) -> dict[str, Any]:
    """Estimate CSCV-style PBO from trial/block ordinary-net returns."""
    work = validate_ledger(ledger)
    part = work[work["segment"].astype(str).eq(segment)].copy()
    variants = sorted(part["trial_id"].astype(str).unique())
    dates = pd.DatetimeIndex(part["decision_time"].dt.normalize().drop_duplicates().sort_values())
    if len(variants) < 2:
        return {"estimable": False, "reason": "requires at least two preregistered strategy variants"}
    if "ordinary_net" not in part:
        return {"estimable": False, "reason": "ordinary_net is required for block returns"}
    if blocks < 6 or blocks % 2 or len(dates) < blocks:
        return {"estimable": False, "reason": "requires an even >=6 time blocks with at least one session each"}
    date_chunks = np.array_split(dates, blocks)
    date_to_block = {date: number for number, chunk in enumerate(date_chunks) for date in chunk}
    part["block"] = part["decision_time"].dt.normalize().map(date_to_block)
    part["block_return"] = np.where(
        part["status"].eq("filled"), part.get("ordinary_net", 0.0).fillna(0.0), 0.0
    )
    matrix = part.groupby(["trial_id", "block"])["block_return"].sum().unstack()
    matrix = matrix.reindex(index=variants, columns=range(blocks))
    if matrix.isna().any().any():
        return {"estimable": False, "reason": "not every variant has observations in every time block"}

    def scores(values: np.ndarray) -> np.ndarray:
        mean = values.mean(axis=1)
        std = values.std(axis=1, ddof=1)
        return np.divide(mean, std, out=np.where(mean > 0, np.inf, np.where(mean < 0, -np.inf, 0.0)), where=std > 0)

    logits: list[float] = []
    degradation = 0
    for train_blocks in itertools.combinations(range(blocks), blocks // 2):
        test_blocks = sorted(set(range(blocks)).difference(train_blocks))
        train_scores = scores(matrix[list(train_blocks)].to_numpy(float))
        selected = int(np.argmax(train_scores))
        test_scores = scores(matrix[test_blocks].to_numpy(float))
        ranks = rankdata(test_scores, method="average")
        relative_rank = float((ranks[selected] - 0.5) / len(variants))
        relative_rank = min(max(relative_rank, 1e-12), 1 - 1e-12)
        logits.append(float(math.log(relative_rank / (1.0 - relative_rank))))
        degradation += int(relative_rank <= 0.5)
    return {
        "estimable": True, "reason": None, "segment": segment, "variants": len(variants),
        "time_blocks": blocks, "combinations": len(logits),
        "pbo": float(degradation / len(logits)), "median_logit_oos_rank": float(np.median(logits)),
        "convention": "equal consecutive session blocks; in-sample block Sharpe selects; OOS block Sharpe ranks",
    }


def plateau_gate(summary: pd.DataFrame, *, segment: str, lot_mode: str = "historical") -> dict[str, Any]:
    required = {"segment", "threshold", "lot_mode", "ordinary_total", "filled"}
    missing = sorted(required.difference(summary.columns))
    if missing:
        return {"pass": False, "reason": f"summary lacks plateau fields: {missing}"}
    rows = summary[(summary["segment"].astype(str) == segment) & (summary["lot_mode"].astype(str) == lot_mode)]
    if "event_counterfactual" in rows:
        rows = rows[~rows["event_counterfactual"].astype(bool)]
    selected = {}
    for threshold in PLATEAU_THRESHOLDS:
        matches = rows[np.isclose(pd.to_numeric(rows["threshold"]), threshold)]
        if len(matches) != 1 or int(matches.iloc[0]["filled"]) <= 0:
            return {"pass": False, "reason": f"requires exactly one nonempty trial at threshold {threshold:.2f}"}
        selected[threshold] = matches.iloc[0]
    primary = selected[PRIMARY_THRESHOLD]
    primary_per_trade = float(primary["ordinary_total"]) / int(primary["filled"])
    primary_sign = np.sign(float(primary["ordinary_total"]))
    checks = {}
    for threshold in (1.15, 1.35):
        neighbor = selected[threshold]
        neighbor_per_trade = float(neighbor["ordinary_total"]) / int(neighbor["filled"])
        same_sign = bool(np.sign(float(neighbor["ordinary_total"])) == primary_sign)
        within = bool(abs(neighbor_per_trade - primary_per_trade) <= 0.25 * max(abs(primary_per_trade), 1e-15))
        checks[f"threshold_{threshold:.2f}"] = {"same_sign": same_sign, "within_25pct_per_trade": within}
    passed = primary_sign != 0 and all(all(value.values()) for value in checks.values())
    return {"pass": bool(passed), "reason": None if passed else "plateau sign/per-trade tolerance failed", "checks": checks}


def primary_tail_gate(summary: pd.DataFrame, *, segment: str) -> dict[str, Any]:
    """Apply frozen observed-sample tail limits to one primary-policy row."""
    required = {
        "segment", "threshold", "lot_mode", "worst_day", "worst_week", "max_drawdown",
        "historical_loss_cvar95", "missing_exits", "unrecorded_exits", "daily_loss_breaches",
        "weekly_loss_breaches", "drawdown_breaches",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        return {"pass": False, "reason": f"summary lacks tail fields: {missing}", "checks": {}}
    rows = summary[(summary["segment"].astype(str) == segment)]
    rows = rows[(rows["lot_mode"].astype(str) == "historical") & np.isclose(
        pd.to_numeric(rows["threshold"]), PRIMARY_THRESHOLD
    )]
    if "event_counterfactual" in rows:
        rows = rows[~rows["event_counterfactual"].astype(bool)]
    if len(rows) != 1:
        return {"pass": False, "reason": "requires exactly one historical primary trial", "checks": {}}
    row = rows.iloc[0]
    checks = {
        "worst_day_at_least_minus_25000": bool(float(row["worst_day"]) >= -25_000),
        "worst_week_at_least_minus_40000": bool(float(row["worst_week"]) >= -40_000),
        "max_drawdown_at_most_50000": bool(float(row["max_drawdown"]) <= 50_000),
        "historical_cvar95_at_most_15000": bool(
            pd.notna(row["historical_loss_cvar95"]) and float(row["historical_loss_cvar95"]) <= 15_000
        ),
        "zero_unrecorded_exits": int(row["unrecorded_exits"]) == 0,
        "zero_risk_breaches": sum(int(row[field]) for field in (
            "daily_loss_breaches", "weekly_loss_breaches", "drawdown_breaches"
        )) == 0,
    }
    passed = all(checks.values())
    return {"pass": passed, "reason": None if passed else "one or more frozen observed tail checks failed",
            "checks": checks}


def tail_sample_sufficiency(summary: pd.DataFrame, *, segment: str) -> dict[str, Any]:
    """Report sample sufficiency separately from observed tail performance."""
    required = {
        "segment", "threshold", "lot_mode", "historical_size_500_sessions_pass",
        "historical_size_100_fills_pass", "cvar_ucb_20k_pass",
        "cvar95_block_bootstrap_estimable", "required_new_12m_forward_sample_satisfied",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        return {"historical_tail_review_sufficient": False,
                "forward_sample_satisfied": False, "reason": f"summary lacks sample fields: {missing}"}
    rows = summary[(summary["segment"].astype(str) == segment)]
    rows = rows[(rows["lot_mode"].astype(str) == "historical") & np.isclose(
        pd.to_numeric(rows["threshold"]), PRIMARY_THRESHOLD
    )]
    if "event_counterfactual" in rows:
        rows = rows[~rows["event_counterfactual"].astype(bool)]
    if len(rows) != 1:
        return {"historical_tail_review_sufficient": False,
                "forward_sample_satisfied": False, "reason": "requires exactly one historical primary trial"}
    row = rows.iloc[0]
    checks = {
        "at_least_500_eligible_sessions": bool(row["historical_size_500_sessions_pass"]),
        "at_least_100_fills": bool(row["historical_size_100_fills_pass"]),
        "block_bootstrap_cvar_ucb_estimable": bool(row["cvar95_block_bootstrap_estimable"]),
        "block_bootstrap_cvar_ucb_at_most_20000": bool(row["cvar_ucb_20k_pass"]),
    }
    return {
        "historical_tail_review_sufficient": all(checks.values()), "historical_checks": checks,
        "forward_sample_satisfied": bool(row["required_new_12m_forward_sample_satisfied"]),
        "reason": "future quarantined approximately 12-month sample remains required",
    }


def development_validation_gate(summary: pd.DataFrame) -> dict[str, Any]:
    """Fail-closed pre-pseudo promotion gate for the frozen primary policy."""
    reasons: list[str] = []
    plateau = {}
    tails = {}
    sample_sufficiency = {}
    for segment in ("development", "validation"):
        plateau[segment] = plateau_gate(summary, segment=segment)
        if not plateau[segment]["pass"]:
            reasons.append(f"{segment}: plateau failed ({plateau[segment]['reason']})")
        tails[segment] = primary_tail_gate(summary, segment=segment)
        if not tails[segment]["pass"]:
            reasons.append(f"{segment}: observed tail gate failed ({tails[segment]['reason']})")
        sample_sufficiency[segment] = tail_sample_sufficiency(summary, segment=segment)
        rows = summary[(summary["segment"].astype(str) == segment)]
        if "lot_mode" not in rows or "threshold" not in rows:
            reasons.append(f"{segment}: primary metadata unavailable")
            continue
        rows = rows[(rows["lot_mode"].astype(str) == "historical") & np.isclose(pd.to_numeric(rows["threshold"]), 1.25)]
        if "event_counterfactual" in rows:
            rows = rows[~rows["event_counterfactual"].astype(bool)]
        if len(rows) != 1:
            reasons.append(f"{segment}: requires exactly one historical primary trial")
            continue
        row = rows.iloc[0]
        if not (float(row["ordinary_total"]) > 0 and float(row["net_2x_total"]) > 0):
            reasons.append(f"{segment}: ordinary and 2x net totals must both be positive")
        concentration = row.get("max_positive_month_profit_concentration")
        if pd.isna(concentration) or float(concentration) > 0.35:
            reasons.append(f"{segment}: monthly profit concentration unavailable or above 35%")
    return {
        "pass": not reasons,
        "verdict": "research_shadow_candidate_on_development_validation" if not reasons else "research_failed",
        "reasons": reasons, "plateau": plateau, "observed_tail_gate": tails,
        "sample_sufficiency": sample_sufficiency,
        "pseudo_strategy_opened": False,
        "verified_backtest": False, "live_eligible": False,
        "caveat": "This research-only gate does not satisfy sample sufficiency, the required new 12-month forward sample, verification, live eligibility, or acceptance.",
    }
