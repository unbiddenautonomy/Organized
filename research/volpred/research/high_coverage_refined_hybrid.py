"""Combine the refined transition rule with validated fallbacks for coverage.

The refined variance-transition rule has first priority.  The previously
validated expanded hybrid is admitted only on dates without a refined trade.
No 2025 outcome is used to choose either component or the priority order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import expanded_signal_hybrid as expanded
import hybrid_regime_strategy as hybrid
import retrained_short_vol_hybrid as retrained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "high_coverage_refined_hybrid"
REFINED_LEDGER = ROOT / "model" / "regimeUnderstanding" / "refined_variance_transition" / "trade_ledger.parquet"
FALLBACK_LEDGER = ROOT / "model" / "regimeUnderstanding" / "expanded_signal_hybrid" / "expanded_trade_ledger.parquet"
SLIPPAGE_BPS = 25.0
TRADING_DAYS_PER_MONTH = 21.0
AVAILABLE_DAYS = {2024: 235, 2025: 76}


def metrics(frame: pd.DataFrame) -> dict:
    trade = hybrid.metrics(frame)
    daily = frame.groupby("date", as_index=False)["net_pnl_rupees"].sum()
    return {
        **trade,
        "signal_days": int(len(daily)),
        "profitable_days": int(daily["net_pnl_rupees"].gt(0).sum()),
        "profitable_day_rate": float(daily["net_pnl_rupees"].gt(0).mean()) if len(daily) else np.nan,
    }


def combine() -> pd.DataFrame:
    refined = smb.read_parquet(REFINED_LEDGER).copy()
    fallback = smb.read_parquet(FALLBACK_LEDGER).copy()
    refined["hybrid_component"] = "refined_variance_transition_primary"
    fallback["hybrid_component"] = fallback["hybrid_component"].fillna("validated_expanded_fallback")
    occupied = set(pd.to_datetime(refined["date"]).dt.normalize())
    fallback = fallback[~pd.to_datetime(fallback["date"]).dt.normalize().isin(occupied)].copy()
    out = pd.concat([refined, fallback], ignore_index=True, sort=False)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["year"] = out["date"].dt.year
    return out.sort_values(["entry_datetime", "exit_datetime"]).reset_index(drop=True)


def feasibility_frontier() -> pd.DataFrame:
    """Show why forcing six-to-seven months from one rule was rejected."""
    panels = {
        "short_2_strangle": retrained.historical_structure_panel(),
        "short_1_strangle": expanded.historical_structure_panel(
            "short_1_strangle", expanded.SECONDARY_STRUCTURES["short_1_strangle"]
        ),
        "short_atm_straddle": expanded.historical_structure_panel(
            "short_atm_straddle", expanded.SECONDARY_STRUCTURES["short_atm_straddle"]
        ),
    }
    rows = []
    for structure, panel in panels.items():
        train = panel[panel["year"].le(2022)]
        validation = panel[panel["year"].eq(2023)].copy()
        model = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
        model.fit(train[retrained.MODEL_FEATURES], train["profitable"])
        validation["model_probability"] = model.predict_proba(validation[retrained.MODEL_FEATURES])[:, 1]
        for quantile in (0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
            threshold = float(validation["model_probability"].quantile(quantile))
            selected = retrained.first_threshold_crossing(validation, threshold)
            h1 = selected[selected["date"].lt("2023-07-01")]
            h2 = selected[selected["date"].ge("2023-07-01")]
            rows.append({
                "structure": structure,
                "score_quantile": quantile,
                "probability_threshold": threshold,
                "validation_days": int(selected["date"].nunique()),
                "equivalent_trading_months": selected["date"].nunique() / TRADING_DAYS_PER_MONTH,
                "validation_net_pnl": float(selected["net_pnl_rupees"].sum()),
                "validation_hit_rate": float(selected["net_pnl_rupees"].gt(0).mean()),
                "validation_H1_net": float(h1["net_pnl_rupees"].sum()),
                "validation_H2_net": float(h2["net_pnl_rupees"].sum()),
                "target_126_to_147_days": 126 <= selected["date"].nunique() <= 147,
            })
    return pd.DataFrame(rows)


def overlap_audit(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["date", "entry_datetime", "exit_datetime"]).copy()
    previous_exit = ordered.groupby("date")["exit_datetime"].shift()
    entry_minute = ordered["entry_datetime"].dt.hour * 60 + ordered["entry_datetime"].dt.minute
    exit_minute = ordered["exit_datetime"].dt.hour * 60 + ordered["exit_datetime"].dt.minute
    held = (ordered["exit_datetime"] - ordered["entry_datetime"]).dt.total_seconds() / 60
    checks = {
        "cross_day_window": ~ordered["entry_datetime"].dt.normalize().eq(ordered["exit_datetime"].dt.normalize()),
        "outside_0915_1530": ~entry_minute.between(555, 930) | ~exit_minute.between(555, 930),
        "holding_period_mismatch": pd.Series(
            ~np.isclose(held, ordered["horizon_min"]), index=ordered.index
        ),
        "overlapping_positions": previous_exit.gt(ordered["entry_datetime"]),
        "expiry_before_entry": pd.to_datetime(ordered["expiry"]).dt.normalize().lt(ordered["entry_datetime"].dt.normalize()),
    }
    return pd.DataFrame([
        {"check": name, "violations": int(mask.fillna(False).sum()), "passed": not mask.fillna(False).any()}
        for name, mask in checks.items()
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = combine()
    smb.write_parquet(ledger, OUT / "trade_ledger.parquet")

    performance_rows = []
    coverage_rows = []
    for year, period in ((2024, "development_2024"), (2025, "holdout_2025_partial")):
        sample = ledger[ledger["year"].eq(year)]
        result = metrics(sample)
        performance_rows.append({"period": period, **result})
        coverage_rows.append({
            "period": period,
            "available_days": AVAILABLE_DAYS[year],
            "signal_days": result["signal_days"],
            "profitable_days": result["profitable_days"],
            "losing_or_flat_days": result["signal_days"] - result["profitable_days"],
            "signal_day_coverage": result["signal_days"] / AVAILABLE_DAYS[year],
            "equivalent_trading_months": result["signal_days"] / TRADING_DAYS_PER_MONTH,
            "profitable_day_rate": result["profitable_day_rate"],
        })
    combined_result = metrics(ledger)
    performance_rows.append({"period": "combined", **combined_result})
    coverage_rows.append({
        "period": "combined",
        "available_days": sum(AVAILABLE_DAYS.values()),
        "signal_days": combined_result["signal_days"],
        "profitable_days": combined_result["profitable_days"],
        "losing_or_flat_days": combined_result["signal_days"] - combined_result["profitable_days"],
        "signal_day_coverage": combined_result["signal_days"] / sum(AVAILABLE_DAYS.values()),
        "equivalent_trading_months": combined_result["signal_days"] / TRADING_DAYS_PER_MONTH,
        "profitable_day_rate": combined_result["profitable_day_rate"],
    })
    performance = pd.DataFrame(performance_rows)
    coverage = pd.DataFrame(coverage_rows)
    performance.to_csv(OUT / "performance.csv", index=False)
    coverage.to_csv(OUT / "coverage.csv", index=False)

    component = ledger.groupby(["year", "hybrid_component"], as_index=False).agg(
        trades=("date", "size"), signal_days=("date", "nunique"),
        net_pnl=("net_pnl_rupees", "sum"),
    )
    component.to_csv(OUT / "component_mix.csv", index=False)

    cost_rows = []
    for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        stressed = smb.execute(ledger, bps, current_2026=True)
        stressed["year"] = pd.to_datetime(stressed["date"]).dt.year
        for year, period in ((2024, "development_2024"), (2025, "holdout_2025_partial")):
            cost_rows.append({"slippage_bps": bps, "period": period, **metrics(stressed[stressed["year"].eq(year)])})
    cost_sensitivity = pd.DataFrame(cost_rows)
    cost_sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)

    rng = np.random.default_rng(20261013)
    bootstrap_rows = []
    for year, period in ((2024, "development_2024"), (2025, "holdout_2025_partial")):
        values = ledger[ledger["year"].eq(year)].groupby("date")["net_pnl_rupees"].sum().to_numpy(float)
        draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
        bootstrap_rows.append({
            "period": period, "days": len(values), "observed_mean_daily_net": values.mean(),
            "mean_ci_low": np.quantile(draws, 0.025), "mean_ci_high": np.quantile(draws, 0.975),
            "bootstrap_probability_mean_positive": np.mean(draws > 0),
        })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "bootstrap_uncertainty.csv", index=False)

    frontier = feasibility_frontier()
    frontier.to_csv(OUT / "coverage_feasibility_2023.csv", index=False)
    target_frontier = frontier[frontier["target_126_to_147_days"]].copy()
    audit = overlap_audit(ledger)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)

    policy = {
        "status": "FORWARD_PAPER_TEST_ONLY",
        "priority": [
            "refined variance-transition trade",
            "validated expanded hybrid only when the refined rule has no trade that date",
        ],
        "same_monthly_expiry": True,
        "intraday_only": True,
        "no_overlapping_positions": True,
        "slippage_bps": SLIPPAGE_BPS,
        "development_period": "2024; the refined component was selected on 2024",
        "untouched_holdout": "2025-01-01 through 2025-05-09",
        "forced_annual_target_rejected": "All 120m structures at 126-147 validation days lost money in 2023",
        "live_deployment": False,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    report = f"""# High-coverage refined hybrid

The refined variance-transition strategy has first priority. On dates where it
does not trade, the pre-existing validated expanded hybrid may trade. This
raises coverage without lowering a probability threshold after inspecting
2024 or 2025 outcomes.

## Coverage

{dataframe_to_markdown(coverage.round(4))}

Six to seven trading months correspond to approximately 126-147 distinct
trading days. The combined evaluation reaches that absolute range, but the full
2024 calendar year reaches only about five trading months. It is therefore not
valid to describe this as six-to-seven months of coverage per calendar year.

## Net performance after costs

{dataframe_to_markdown(performance[["period", "trades", "signal_days", "profitable_days", "gross_pnl", "cost", "net_pnl", "profit_factor", "max_drawdown"]].round(4))}

## Component mix

{dataframe_to_markdown(component.round(4))}

## Forced-coverage feasibility check

Every tested 120-minute short-volatility structure with 126-147 selected days
lost money in the 2023 validation sample. These rows were rejected rather than
using 2024/2025 results to rescue the threshold.

{dataframe_to_markdown(target_frontier[["structure", "score_quantile", "validation_days", "equivalent_trading_months", "validation_net_pnl", "validation_hit_rate", "validation_H1_net", "validation_H2_net"]].round(4))}

## Bootstrap uncertainty

{dataframe_to_markdown(bootstrap.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}

## Interpretation

The 2024 result is development evidence because the refined component was
selected using 2024. Only the partial-2025 result is an untouched holdout for
the complete priority policy. Quotes do not establish executable fills, and
short strangles retain unbounded tail risk. The policy remains forward-paper
only.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(coverage.to_string(index=False))
    print(performance.to_string(index=False))
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
