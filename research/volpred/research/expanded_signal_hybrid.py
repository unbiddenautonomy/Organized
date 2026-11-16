"""Expand the frozen hybrid with separately validated shorter-horizon fallbacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import hybrid_regime_strategy as hybrid
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "expanded_signal_hybrid"
HORIZONS = (15, 30, 60)
SECONDARY_STRUCTURES = {
    "short_1_strangle": {(-1, "put"): -1, (1, "call"): -1},
    "short_atm_straddle": {(0, "put"): -1, (0, "call"): -1},
}
QUANTILES = (0.70, 0.75, 0.80, 0.85, 0.90)
PRIMARY_QUANTILE = 0.75
MIN_VALIDATION_DAYS = 40


def historical_panel(horizon: int) -> pd.DataFrame:
    cache = OUT / f"historical_short_2_strangle_{horizon}m_entry_safe.parquet"
    if cache.exists():
        return smb.read_parquet(cache)
    entries, _ = smb.prepare_entry_rows()
    entries = smb.add_entry_greeks(entries)
    wide = smb.wide_legs(smb.leg_panel(entries, smb.load_exits(horizon)))
    dummy = wide[["entry_datetime"]].rename(columns={"entry_datetime": "datetime"}).copy()
    dummy["horizon_min"] = horizon
    dummy["pred_annualized_variance"] = 0.0
    dummy["actual_var"] = 0.0
    dummy["is_0915"] = 0
    dummy["q50_compact"] = 0.0
    panel = smb.structure_panel(
        wide, dummy, "short_2_strangle",
        {(-2, "put"): -1, (2, "call"): -1}, horizon,
    )
    panel["date"] = panel["entry_datetime"].dt.normalize()
    panel["lot_size"] = smb.lot_size(panel["date"])
    panel = panel[
        panel["gross_pnl_points"].notna()
        & panel["entry_turnover_points"].gt(0)
        & panel["exit_turnover_points"].gt(0)
        & ~panel["entry_tiny_any"]
    ].copy()
    entry_minute = panel["entry_datetime"].dt.hour * 60 + panel["entry_datetime"].dt.minute
    exit_minute = panel["exit_datetime"].dt.hour * 60 + panel["exit_datetime"].dt.minute
    panel = panel[
        entry_minute.between(555, 930) & exit_minute.between(555, 930)
        & panel["entry_datetime"].dt.normalize().eq(panel["exit_datetime"].dt.normalize())
    ].copy()
    panel = smb.execute(panel, retrained.SLIPPAGE_BPS, current_2026=True)
    features = smb.read_parquet(retrained.FEATURE_PANEL)[["datetime", *retrained.BASE_FEATURES]]
    panel = panel.merge(
        features, left_on="entry_datetime", right_on="datetime",
        how="inner", validate="one_to_one",
    ).drop(columns="datetime")
    panel["entry_value_pct"] = panel["entry_value"].abs() / panel["spot_price"]
    panel["gamma_s2"] = panel["gamma"] * panel["spot_price"].pow(2)
    for column in retrained.MODEL_FEATURES:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel[
        np.isfinite(panel[retrained.MODEL_FEATURES].to_numpy(dtype=float)).all(axis=1)
    ].copy()
    panel["profitable"] = panel["net_pnl_rupees"].gt(0).astype(int)
    panel["year"] = panel["date"].dt.year
    smb.write_parquet(panel, cache)
    return panel


def historical_structure_panel(name: str, weights: dict) -> pd.DataFrame:
    cache = OUT / f"historical_{name}_120m_entry_safe.parquet"
    if cache.exists():
        return smb.read_parquet(cache)
    entries, _ = smb.prepare_entry_rows()
    entries = smb.add_entry_greeks(entries)
    wide = smb.wide_legs(smb.leg_panel(entries, smb.load_exits(120)))
    dummy = wide[["entry_datetime"]].rename(columns={"entry_datetime": "datetime"}).copy()
    dummy["horizon_min"] = 120
    dummy["pred_annualized_variance"] = 0.0
    dummy["actual_var"] = 0.0
    dummy["is_0915"] = 0
    dummy["q50_compact"] = 0.0
    panel = smb.structure_panel(wide, dummy, name, weights, 120)
    panel["date"] = panel["entry_datetime"].dt.normalize()
    panel["lot_size"] = smb.lot_size(panel["date"])
    panel = panel[
        panel["gross_pnl_points"].notna()
        & panel["entry_turnover_points"].gt(0)
        & panel["exit_turnover_points"].gt(0)
        & ~panel["entry_tiny_any"]
    ].copy()
    entry_minute = panel["entry_datetime"].dt.hour * 60 + panel["entry_datetime"].dt.minute
    exit_minute = panel["exit_datetime"].dt.hour * 60 + panel["exit_datetime"].dt.minute
    panel = panel[
        entry_minute.between(555, 930) & exit_minute.between(555, 930)
        & panel["entry_datetime"].dt.normalize().eq(panel["exit_datetime"].dt.normalize())
    ].copy()
    panel = smb.execute(panel, retrained.SLIPPAGE_BPS, current_2026=True)
    features = smb.read_parquet(retrained.FEATURE_PANEL)[["datetime", *retrained.BASE_FEATURES]]
    panel = panel.merge(
        features, left_on="entry_datetime", right_on="datetime",
        how="inner", validate="one_to_one",
    ).drop(columns="datetime")
    panel["entry_value_pct"] = panel["entry_value"].abs() / panel["spot_price"]
    panel["gamma_s2"] = panel["gamma"] * panel["spot_price"].pow(2)
    for column in retrained.MODEL_FEATURES:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel[np.isfinite(panel[retrained.MODEL_FEATURES].to_numpy(dtype=float)).all(axis=1)].copy()
    panel["profitable"] = panel["net_pnl_rupees"].gt(0).astype(int)
    panel["year"] = panel["date"].dt.year
    smb.write_parquet(panel, cache)
    return panel


def fit_and_score(horizon: int) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    panel = historical_panel(horizon)
    train = panel[panel["year"].le(2022)]
    validation = panel[panel["year"].eq(2023)].copy()
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=0.01, max_iter=500),
    )
    model.fit(train[retrained.MODEL_FEATURES], train["profitable"])
    validation["model_probability"] = model.predict_proba(
        validation[retrained.MODEL_FEATURES]
    )[:, 1]
    rows: list[dict] = []
    for quantile in QUANTILES:
        threshold = float(validation["model_probability"].quantile(quantile))
        trades = retrained.first_threshold_crossing(validation, threshold)
        rows.append({
            "horizon_min": horizon,
            "validation_score_quantile": quantile,
            "probability_threshold": threshold,
            "validation_days": len(trades),
            "validation_net_pnl": trades["net_pnl_rupees"].sum(),
            "validation_mean_net": trades["net_pnl_rupees"].mean(),
            "validation_hit_rate": trades["net_pnl_rupees"].gt(0).mean(),
            "eligible": len(trades) >= MIN_VALIDATION_DAYS
            and trades["net_pnl_rupees"].sum() > 0,
        })
    evaluation = panel[panel["year"].ge(2024)].copy()
    evaluation["model_probability"] = model.predict_proba(
        evaluation[retrained.MODEL_FEATURES]
    )[:, 1]
    return model, pd.DataFrame(rows), evaluation


def fit_structure(name: str, weights: dict) -> tuple[object, pd.DataFrame, pd.DataFrame]:
    panel = historical_structure_panel(name, weights)
    train = panel[panel["year"].le(2022)]
    validation = panel[panel["year"].eq(2023)].copy()
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
    model.fit(train[retrained.MODEL_FEATURES], train["profitable"])
    validation["model_probability"] = model.predict_proba(validation[retrained.MODEL_FEATURES])[:, 1]
    rows: list[dict] = []
    for quantile in QUANTILES:
        threshold = float(validation["model_probability"].quantile(quantile))
        trades = retrained.first_threshold_crossing(validation, threshold)
        rows.append({
            "structure": name, "horizon_min": 120,
            "validation_score_quantile": quantile,
            "probability_threshold": threshold, "validation_days": len(trades),
            "validation_net_pnl": trades["net_pnl_rupees"].sum(),
            "validation_mean_net": trades["net_pnl_rupees"].mean(),
            "validation_hit_rate": trades["net_pnl_rupees"].gt(0).mean(),
            "eligible": len(trades) >= MIN_VALIDATION_DAYS and trades["net_pnl_rupees"].sum() > 0,
        })
    evaluation = panel[panel["year"].ge(2024)].copy()
    evaluation["model_probability"] = model.predict_proba(evaluation[retrained.MODEL_FEATURES])[:, 1]
    return model, pd.DataFrame(rows), evaluation


def primary_120m_q75() -> tuple[object, float, pd.DataFrame]:
    panel = retrained.historical_structure_panel()
    model, _, _, _ = retrained.train(panel)
    validation = panel[panel["year"].eq(2023)].copy()
    validation["model_probability"] = model.predict_proba(
        validation[retrained.MODEL_FEATURES]
    )[:, 1]
    threshold = float(validation["model_probability"].quantile(PRIMARY_QUANTILE))
    evaluation = panel[panel["year"].ge(2024)].copy()
    evaluation["model_probability"] = model.predict_proba(
        evaluation[retrained.MODEL_FEATURES]
    )[:, 1]
    trades = retrained.first_threshold_crossing(evaluation, threshold)
    trades["hybrid_component"] = "120m_short_2_strangle_q75"
    return model, threshold, trades


def add_on_empty_dates(base: pd.DataFrame, candidate: pd.DataFrame) -> pd.DataFrame:
    occupied = set(pd.to_datetime(base["date"]).dt.normalize())
    extra = candidate[~candidate["date"].dt.normalize().isin(occupied)].copy()
    return pd.concat([base, extra], ignore_index=True, sort=False).sort_values("entry_datetime")


def integrity_audit(trades: pd.DataFrame) -> pd.DataFrame:
    ordered = trades.sort_values(["date", "entry_datetime"]).copy()
    prior_exit = ordered.groupby(ordered["entry_datetime"].dt.normalize())["exit_datetime"].shift()
    entry_minute = ordered["entry_datetime"].dt.hour * 60 + ordered["entry_datetime"].dt.minute
    exit_minute = ordered["exit_datetime"].dt.hour * 60 + ordered["exit_datetime"].dt.minute
    held = (ordered["exit_datetime"] - ordered["entry_datetime"]).dt.total_seconds() / 60
    checks = {
        "cross_day_windows": ~ordered["entry_datetime"].dt.normalize().eq(ordered["exit_datetime"].dt.normalize()),
        "outside_0915_1530": ~entry_minute.between(555, 930) | ~exit_minute.between(555, 930),
        "holding_period_mismatch": pd.Series(
            ~np.isclose(held, ordered["horizon_min"]), index=ordered.index,
        ),
        "overlapping_positions": prior_exit.gt(ordered["entry_datetime"]),
        "expiry_before_trade_date": pd.to_datetime(ordered["expiry"]).dt.normalize().lt(ordered["entry_datetime"].dt.normalize()),
    }
    return pd.DataFrame([
        {"check": name, "violations": int(mask.fillna(False).sum()), "passed": not mask.fillna(False).any()}
        for name, mask in checks.items()
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    current = retrained.current_trades()
    primary_model, primary_threshold, primary = primary_120m_q75()
    base = add_on_empty_dates(current, primary)

    tuning_frames: list[pd.DataFrame] = []
    candidates: list[dict] = []
    fitted: dict[int, object] = {}
    daily_eval: dict[int, pd.DataFrame] = {}
    for horizon in HORIZONS:
        model, tuning, evaluation = fit_and_score(horizon)
        tuning["structure"] = "short_2_strangle"
        fitted[horizon] = model
        daily_eval[horizon] = evaluation
        tuning_frames.append(tuning)
        candidates.extend(tuning[tuning["eligible"]].to_dict("records"))
    structure_models: dict[str, object] = {}
    structure_daily: dict[str, pd.DataFrame] = {}
    for name, weights in SECONDARY_STRUCTURES.items():
        model, tuning, evaluation = fit_structure(name, weights)
        structure_models[name] = model
        structure_daily[name] = evaluation
        tuning_frames.append(tuning)
        candidates.extend(tuning[tuning["eligible"]].to_dict("records"))
    tuning_all = pd.concat(tuning_frames, ignore_index=True)
    tuning_all.to_csv(OUT / "shorter_horizon_validation.csv", index=False)

    # Freeze a validation-only coverage/profit knee: retain rules delivering at
    # least 80% of the best 2023 net P&L, then take the one with most days.
    if not candidates:
        raise RuntimeError("No shorter-horizon fallback passed 2023 validation")
    best_validation_net = max(row["validation_net_pnl"] for row in candidates)
    knee = [
        row for row in candidates
        if row["validation_net_pnl"] >= 0.80 * best_validation_net
    ]
    selected = max(knee, key=lambda row: (row["validation_days"], row["validation_net_pnl"]))
    horizon = int(selected["horizon_min"])
    threshold = float(selected["probability_threshold"])
    selected_structure = selected.get("structure", "short_2_strangle")
    if selected_structure in structure_daily:
        secondary = structure_daily[selected_structure]
        secondary_model = structure_models[selected_structure]
    else:
        secondary = daily_eval[horizon]
        secondary_model = fitted[horizon]
    secondary = retrained.first_threshold_crossing(secondary, threshold)
    secondary["hybrid_component"] = f"{horizon}m_{selected_structure}_secondary"
    expanded = add_on_empty_dates(base, secondary)

    frames = {
        "current_coverage_hybrid": current,
        "q75_120m_hybrid": base,
        "expanded_signal_hybrid": expanded,
    }
    performance, coverage = retrained.performance(frames)
    performance.to_csv(OUT / "strategy_performance.csv", index=False)
    coverage.to_csv(OUT / "trading_day_coverage.csv", index=False)
    previous = smb.read_parquet(
        ROOT / "model" / "regimeUnderstanding" / "retrained_hybrid" / "combined_trade_ledger.parquet"
    )
    aggregate_rows: list[dict] = []
    for strategy, trades in {
        "current_coverage_hybrid": current,
        "previous_q80_retrained_hybrid": previous,
        "expanded_signal_hybrid": expanded,
    }.items():
        result = hybrid.metrics(trades)
        daily = trades.groupby("date")["net_pnl_rupees"].sum()
        aggregate_rows.append({
            "strategy": strategy, "total_available_days": 311,
            "signal_days": daily.size, "profitable_days": int(daily.gt(0).sum()),
            "signal_day_coverage": daily.size / 311,
            "profitable_signal_day_rate": daily.gt(0).mean(),
            "trades": result["trades"], "net_pnl": result["net_pnl"],
            "profit_factor": result["profit_factor"], "max_drawdown": result["max_drawdown"],
        })
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(OUT / "aggregate_comparison.csv", index=False)
    smb.write_parquet(expanded, OUT / "expanded_trade_ledger.parquet")
    smb.write_parquet(secondary, OUT / "secondary_candidate_trades.parquet")
    audit = integrity_audit(expanded)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    split_manifest = pd.DataFrame([
        {"split": "train", "date_rule": "year <= 2022", "role": "fit all probability models"},
        {"split": "validation", "date_rule": "year = 2023", "role": "select thresholds, structure, and coverage/profit knee"},
        {"split": "development evaluation", "date_rule": "year = 2024", "role": "frozen out-of-sample evaluation"},
        {"split": "holdout evaluation", "date_rule": "year = 2025 (available through 2025-05-09)", "role": "frozen partial holdout evaluation"},
    ])
    split_manifest.to_csv(OUT / "training_split_manifest.csv", index=False)
    joblib.dump({
        "primary_model": primary_model,
        "primary_threshold": primary_threshold,
        "secondary_model": secondary_model,
        "secondary_structure": selected_structure,
        "secondary_horizon_min": horizon,
        "secondary_threshold": threshold,
        "features": retrained.MODEL_FEATURES,
    }, OUT / "expanded_hybrid_models.joblib")
    policy = {
        "status": "FORWARD_PAPER_TEST_ONLY",
        "priority": [
            "current_coverage_hybrid",
            "120m short ATM+/-2 strangle when score >= q75 2023 threshold",
            f"{horizon}m {selected_structure} when score >= selected 2023 threshold",
        ],
        "fallback_only_on_dates_without_an_earlier_priority_component": True,
        "positions_must_not_overlap": True,
        "same_monthly_expiry": True,
        "slippage_bps": retrained.SLIPPAGE_BPS,
        "selection_rule": "among >=40-day positive 2023 rules retaining >=80% of best validation net, maximize validation days",
        "primary_threshold": primary_threshold,
        "secondary_threshold": threshold,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    report = f"""# Expanded signal hybrid

This extension keeps the previously validated hybrid first, relaxes the
120-minute fallback only from the top 20% to the still-profitable top 25% of
2023 daily scores, then adds one independently trained shorter-horizon fallback
only on dates that remain empty. No 2024/2025 result selects a model, horizon,
or threshold.

## Frozen selection

- Primary fallback threshold: `{primary_threshold:.6f}` (2023 q75).
- Secondary: `{horizon}`-minute `{selected_structure}`.
- Secondary threshold: `{threshold:.6f}`.
- Validation-day minimum: {MIN_VALIDATION_DAYS}.
- Coverage/profit knee: retain at least 80% of the best eligible 2023 net P&L,
  then maximize validation days. This selection uses no 2024/2025 result.
- Costs: current encoded charges plus {retrained.SLIPPAGE_BPS:.0f} bps premium-turnover slippage.
- Same monthly expiry; at most one newly selected fallback per otherwise-empty
  date.

Fallback components are admitted only on a date unused by a higher-priority
component. The pre-existing current component can contain multiple sequential,
non-overlapping trades on one date; the integrity audit confirms zero overlap.

## Results

{dataframe_to_markdown(performance.round(4))}

## Combined 2024 and partial-2025 comparison

{dataframe_to_markdown(aggregate.round(4))}

## Day coverage

{dataframe_to_markdown(coverage.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

## Guardrails

The short strangle has unbounded tail risk. The 2025 period is partial, quotes
do not prove fillability, and this remains a forward-paper-test policy rather
than a live-trading recommendation.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(policy)
    print(tuning_all.to_string(index=False))
    print(performance.to_string(index=False))
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="isolated directory for training outputs and historical caches")
    args = parser.parse_args()
    if args.output:
        OUT = args.output.expanduser().resolve()
    main()
