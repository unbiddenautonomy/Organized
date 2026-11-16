"""Backtest causal IV-current/RV-current and forecast-convergence rules.

The rule is intentionally separate from the fitted fallback classifiers.  It
uses only values available at entry: trailing 60-minute realized variance,
current ATM IV, and walk-forward IV/RV forecasts for the candidate horizon.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "iv_rv_convergence_backtest"
PANEL = ROOT / "model" / "regimeUnderstanding" / "five_strike_trade_research" / "candidate_panel.parquet"
FEATURES = ROOT / "outputs" / "spot_magnitude_models" / "spot_magnitude_feature_panel.parquet"
ANNUAL_TRADING_MINUTES = 252.0 * 375.0
CURRENT_RV_LOOKBACK_MIN = 60
EQUALITY_BAND = 0.005
SLIPPAGE_BPS = 25.0
MIN_DEVELOPMENT_DAYS = 20

LONG_STRUCTURES = {
    "long_atm_straddle", "long_1_strangle", "long_2_strangle",
    "reverse_iron_fly_1", "reverse_iron_fly_2", "reverse_iron_condor_1_2",
}
SHORT_STRUCTURES = {
    "short_atm_straddle", "short_1_strangle", "short_2_strangle",
    "iron_fly_1", "iron_fly_2", "iron_condor_1_2",
}


def prepare_panel() -> pd.DataFrame:
    panel = smb.read_parquet(PANEL)
    features = smb.read_parquet(FEATURES)[["datetime", "rv_past_60", "iv_change_5"]]
    features = features.drop_duplicates("datetime")
    panel = panel.merge(
        features, left_on="entry_datetime", right_on="datetime",
        how="inner", validate="many_to_one",
    ).drop(columns="datetime")
    panel["rv_current"] = np.sqrt(
        panel["rv_past_60"].clip(lower=0)
        * ANNUAL_TRADING_MINUTES / CURRENT_RV_LOOKBACK_MIN
    )
    panel["iv_current"] = panel["iv_trade_vol"]
    panel["iv_pred"] = panel["iv_current"] * np.exp(panel["q50_compact"])
    panel["future_gap"] = panel["iv_pred"] - panel["predicted_rv"]
    panel["future_near_equal"] = panel["future_gap"].abs().le(EQUALITY_BAND)

    # "Toward RVpred" means the forecast IV move has the required sign, stays
    # on the appropriate side of RVpred (allowing the equality band), and is
    # closer to RVpred than today's unchanged IV would be.
    base_distance = (panel["iv_current"] - panel["predicted_rv"]).abs()
    forecast_distance = panel["future_gap"].abs()
    panel["iv_pred_up_toward_rv_pred"] = (
        panel["iv_pred"].gt(panel["iv_current"])
        & panel["iv_pred"].le(panel["predicted_rv"] + EQUALITY_BAND)
        & forecast_distance.lt(base_distance)
    )
    panel["iv_pred_down_toward_rv_pred"] = (
        panel["iv_pred"].lt(panel["iv_current"])
        & panel["iv_pred"].ge(panel["predicted_rv"] - EQUALITY_BAND)
        & forecast_distance.lt(base_distance)
    )
    panel["long_signal"] = (
        panel["iv_current"].lt(panel["rv_current"])
        & (panel["future_near_equal"] | panel["iv_pred_up_toward_rv_pred"])
    )
    panel["short_signal"] = (
        panel["iv_current"].gt(panel["rv_current"])
        & (panel["future_near_equal"] | panel["iv_pred_down_toward_rv_pred"])
    )
    panel["long_near_signal"] = panel["iv_current"].lt(panel["rv_current"]) & panel["future_near_equal"]
    panel["long_toward_signal"] = panel["iv_current"].lt(panel["rv_current"]) & panel["iv_pred_up_toward_rv_pred"]
    panel["short_near_signal"] = panel["iv_current"].gt(panel["rv_current"]) & panel["future_near_equal"]
    panel["short_toward_signal"] = panel["iv_current"].gt(panel["rv_current"]) & panel["iv_pred_down_toward_rv_pred"]
    panel["long_observed_iv_confirmation"] = panel["iv_change_5"].gt(0)
    panel["short_observed_iv_confirmation"] = panel["iv_change_5"].lt(0)
    panel["convergence_strength"] = base_distance - forecast_distance
    panel["year"] = panel["date"].dt.year
    return panel


def earliest_daily(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    # This is live-reproducible: take the first eligible observation, never the
    # best observation seen later in the same day.
    return (
        frame.sort_values(["entry_datetime", "convergence_strength"], ascending=[True, False])
        .drop_duplicates("date", keep="first")
        .copy()
    )


def metric_row(frame: pd.DataFrame, **labels) -> dict:
    m = smb.metrics(frame)
    return {**labels, **m, "profitable_days": int(frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum())}


def evaluate_grid(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str, str, int], pd.DataFrame]]:
    rows: list[dict] = []
    ledgers: dict[tuple[str, str, int], pd.DataFrame] = {}
    for side, signal_variant, signal, structures in (
        ("long_vol", "combined_or", "long_signal", LONG_STRUCTURES),
        ("long_vol", "near_only", "long_near_signal", LONG_STRUCTURES),
        ("long_vol", "toward_only", "long_toward_signal", LONG_STRUCTURES),
        ("short_vol", "combined_or", "short_signal", SHORT_STRUCTURES),
        ("short_vol", "near_only", "short_near_signal", SHORT_STRUCTURES),
        ("short_vol", "toward_only", "short_toward_signal", SHORT_STRUCTURES),
    ):
        for structure in sorted(structures):
            for horizon in smb.HORIZONS:
                candidates = panel[
                    panel[signal]
                    & panel["structure"].eq(structure)
                    & panel["horizon_min"].eq(horizon)
                ].copy()
                trades = smb.execute(
                    earliest_daily(candidates), SLIPPAGE_BPS, current_2026=True,
                )
                ledgers[(side, signal_variant, structure, horizon)] = trades
                for year, period in ((2024, "development_2024"), (2025, "holdout_2025")):
                    rows.append(metric_row(
                        trades[trades["year"].eq(year)], side=side, signal_variant=signal_variant,
                        structure=structure, horizon_min=horizon, period=period,
                    ))
    return pd.DataFrame(rows), ledgers


def select_on_2024(summary: pd.DataFrame) -> pd.DataFrame:
    dev = summary[
        summary["period"].eq("development_2024")
        & summary["signal_variant"].eq("combined_or")
    ].copy()
    eligible = dev[dev["days"].ge(MIN_DEVELOPMENT_DAYS) & dev["net_pnl"].gt(0)]
    selections = []
    for side in ("long_vol", "short_vol"):
        sample = eligible[eligible["side"].eq(side)]
        if sample.empty:
            sample = dev[dev["side"].eq(side) & dev["days"].ge(MIN_DEVELOPMENT_DAYS)]
        if not sample.empty:
            pick = sample.sort_values(["net_pnl", "profit_factor"], ascending=False).iloc[0].copy()
            pick["passed_development_rule"] = bool(pick["net_pnl"] > 0 and pick["days"] >= MIN_DEVELOPMENT_DAYS)
            selections.append(pick)
    return pd.DataFrame(selections)


def combine_selected(selected: pd.DataFrame, ledgers: dict) -> pd.DataFrame:
    parts = []
    for row in selected.itertuples(index=False):
        part = ledgers[(row.side, "combined_or", row.structure, int(row.horizon_min))].copy()
        part["strategy_side"] = row.side
        parts.append(part)
    if not parts:
        return pd.DataFrame()
    # One position per day. If both sides trigger, the chronologically first
    # signal wins. A same-minute conflict uses greater forecast convergence.
    return (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values(["entry_datetime", "convergence_strength"], ascending=[True, False])
        .drop_duplicates("date", keep="first")
        .sort_values("entry_datetime")
        .reset_index(drop=True)
    )


def benchmark_rows(summary: pd.DataFrame) -> pd.DataFrame:
    return summary[
        summary["signal_variant"].eq("combined_or")
        &
        summary["horizon_min"].eq(120)
        & summary["structure"].isin(["long_atm_straddle", "short_atm_straddle"])
    ].copy()


def integrity_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame([{"check": "nonempty", "violations": 1, "passed": False}])
    entry_min = trades["entry_datetime"].dt.hour * 60 + trades["entry_datetime"].dt.minute
    exit_min = trades["exit_datetime"].dt.hour * 60 + trades["exit_datetime"].dt.minute
    held = (trades["exit_datetime"] - trades["entry_datetime"]).dt.total_seconds() / 60
    checks = {
        "one_trade_per_day": trades.duplicated("date", keep=False),
        "cross_day_window": ~trades["entry_datetime"].dt.normalize().eq(trades["exit_datetime"].dt.normalize()),
        "outside_0915_1530": ~entry_min.between(555, 930) | ~exit_min.between(555, 930),
        "holding_period_mismatch": ~np.isclose(held, trades["horizon_min"]),
        "expiry_before_entry": pd.to_datetime(trades["expiry"]).dt.normalize().lt(trades["entry_datetime"].dt.normalize()),
    }
    return pd.DataFrame([
        {"check": name, "violations": int(pd.Series(mask).fillna(False).sum()),
         "passed": not pd.Series(mask).fillna(False).any()}
        for name, mask in checks.items()
    ])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare_panel()
    summary, ledgers = evaluate_grid(panel)
    selected = select_on_2024(summary)
    combined = combine_selected(selected, ledgers)
    audit = integrity_audit(combined)

    combined_rows = []
    for year, period in ((2024, "development_2024"), (2025, "holdout_2025")):
        combined_rows.append(metric_row(
            combined[combined["year"].eq(year)], strategy="selected_convergence_hybrid", period=period,
        ))
    combined_rows.append(metric_row(
        combined, strategy="selected_convergence_hybrid", period="combined_2024_partial_2025",
    ))
    combined_summary = pd.DataFrame(combined_rows)

    # Use one representative structure to count signal dates without the
    # structure duplication in the candidate panel.
    representative = panel[panel["structure"].eq("long_atm_straddle")]
    signal_coverage = pd.DataFrame([
        {
            "period": period,
            "available_days": int(part["date"].nunique()),
            "long_condition_days": int(part.loc[part["long_signal"], "date"].nunique()),
            "short_condition_days": int(part.loc[part["short_signal"], "date"].nunique()),
            "either_condition_days": int(part.loc[part["long_signal"] | part["short_signal"], "date"].nunique()),
        }
        for period, part in (
            ("development_2024", representative[representative["year"].eq(2024)]),
            ("holdout_2025", representative[representative["year"].eq(2025)]),
            ("combined", representative),
        )
    ])

    summary.to_csv(OUT / "structure_horizon_results.csv", index=False)
    selected.to_csv(OUT / "development_best_diagnostics.csv", index=False)
    benchmark_rows(summary).to_csv(OUT / "fixed_120m_atm_benchmarks.csv", index=False)
    combined_summary.to_csv(OUT / "hybrid_performance.csv", index=False)
    signal_coverage.to_csv(OUT / "signal_coverage.csv", index=False)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)
    smb.write_parquet(combined, OUT / "hybrid_trade_ledger.parquet")

    selected_eval = summary.merge(
        selected[["side", "signal_variant", "structure", "horizon_min", "passed_development_rule"]],
        on=["side", "signal_variant", "structure", "horizon_min"], how="inner",
    )
    dev_sub = summary[
        summary["period"].eq("development_2024")
        & summary["days"].ge(MIN_DEVELOPMENT_DAYS)
    ].copy()
    top_sub = (
        dev_sub.sort_values("net_pnl", ascending=False)
        .drop_duplicates(["side", "signal_variant"])
        [["side", "signal_variant", "structure", "horizon_min"]]
    )
    subcondition_eval = summary.merge(
        top_sub, on=["side", "signal_variant", "structure", "horizon_min"], how="inner",
    )
    subcondition_eval.to_csv(OUT / "subcondition_diagnostics.csv", index=False)
    policy = {
        "status": "RESEARCH_ONLY_NOT_LIVE_DEPLOYED",
        "equality_band_absolute_vol": EQUALITY_BAND,
        "current_rv": "sqrt(trailing_60m_variance * 252 * 375 / 60)",
        "long_condition": "IVcurrent < RVcurrent and (abs(IVpred-RVpred)<=0.005 or IVpred rises toward RVpred)",
        "short_condition": "IVcurrent > RVcurrent and (abs(IVpred-RVpred)<=0.005 or IVpred falls toward RVpred)",
        "selection": "a rule must have >=20 development days and positive 2024 net PnL; no long or short rule passed",
        "holdout": "2025-01-01 through 2025-05-09, untouched by selection",
        "costs": "current encoded 2026 charges plus 25 bps premium-turnover slippage",
        "same_monthly_expiry": True,
        "one_trade_per_day": True,
        "decision": "DO_NOT_ADD_THIS_CONVERGENCE_RULE_TO_LIVE_STRATEGY",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    report = f"""# IV/RV forecast-convergence backtest

## Tested rule

Current realized volatility is the annualized trailing 60-minute volatility:

`RVcurrent = sqrt(sum(last 60 intraday log returns^2) * 252 * 375 / 60)`.

The future forecasts use `IVpred = IVcurrent * exp(predicted log-IV change)`
and the horizon-matched walk-forward `RVpred`. "Nearly equal" means
`abs(IVpred - RVpred) <= {EQUALITY_BAND:.3f}` (0.5 volatility points).

- Long-vol signal: `IVcurrent < RVcurrent` and future IV is nearly equal to
  predicted RV, or IV is forecast to rise toward predicted RV.
- Short-vol signal: `IVcurrent > RVcurrent` and future IV is nearly equal to
  predicted RV, or IV is forecast to fall toward predicted RV.

"Toward" requires the IV forecast to have the correct sign, remain on the
appropriate side of predicted RV (within the equality tolerance is allowed),
and reduce the distance to predicted RV relative to leaving IV unchanged.

Every entry uses only timestamp-available data. It takes the first eligible
signal of the day, keeps the entry strikes fixed, uses one monthly expiry, and
exits after the selected intraday horizon no later than 15:30.

## Data split

- Development/selection: 2024.
- Untouched holdout: 2025 through 9 May, the end of forecast-aligned data.
- The forecasts themselves are walk-forward out-of-sample predictions.

The forecast-aligned panel has no 2023 rows, so the structure and horizon for
this new rule cannot be selected on the repository's usual 2023 validation
period. The 2024-selected hybrid is therefore exploratory, while the 2025
figures are its clean holdout check.

## Condition coverage

{dataframe_to_markdown(signal_coverage)}

## Best 2024 diagnostics for the requested OR rule

No cell passed the required positive-development-P&L rule. The table therefore
shows the least-bad cell on each side, not an approved strategy.

{dataframe_to_markdown(selected[["side", "structure", "horizon_min", "days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown", "passed_development_rule"]].round(4))}

## Selected rules in development and holdout

{dataframe_to_markdown(selected_eval[["side", "structure", "horizon_min", "period", "days", "profitable_days", "net_pnl", "mean_net", "hit_rate", "profit_factor", "max_drawdown"]].round(4))}

## Forced one-trade-per-day diagnostic

This combines the two least-bad cells only to quantify the consequence of
forcing the hypothesis into a hybrid. It is not selected for deployment.

{dataframe_to_markdown(combined_summary.round(4))}

## Near-equality versus moving-toward diagnostics

Each row is the best 2024 cell with at least 20 days for that subcondition;
the adjacent 2025 row is untouched holdout performance of the same cell.

{dataframe_to_markdown(subcondition_eval[["side", "signal_variant", "structure", "horizon_min", "period", "days", "profitable_days", "net_pnl", "hit_rate", "profit_factor"]].round(4))}

## Fixed 120-minute ATM benchmarks

{dataframe_to_markdown(benchmark_rows(summary)[["side", "structure", "period", "days", "profitable_days", "net_pnl", "mean_net", "hit_rate", "profit_factor", "max_drawdown"]].round(4))}

## Integrity checks

{dataframe_to_markdown(audit)}

## Interpretation limit

This is a historical, cost-stressed screen, not an instruction to place live
orders. Monthly-expiry short-vol trades retain large gap and tail risk, and
the data use quoted option prices rather than executable bid/ask fills.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("SELECTED")
    print(selected_eval.to_string(index=False))
    print("HYBRID")
    print(combined_summary.to_string(index=False))
    print("COVERAGE")
    print(signal_coverage.to_string(index=False))
    print("BENCHMARKS")
    print(benchmark_rows(summary).to_string(index=False))
    print("AUDIT")
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
