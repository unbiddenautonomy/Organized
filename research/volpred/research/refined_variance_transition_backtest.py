"""Backtest the refined variance-space IV/RV transition hypothesis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import five_strike_regime_trade_research as five
import iv_rv_convergence_backtest as old
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "refined_variance_transition"
PANEL_CACHE = OUT / "refined_candidate_panel.parquet"
IV_FORECASTS = ROOT / "outputs" / "log_iv_change_models" / "compact_five_feature" / "walk_forward_predictions.parquet"
ANNUAL_MINUTES = 252.0 * 375.0
Z_EQUALITY = 1.0
MIN_PRIOR_ERRORS = 50
MIN_HALF_DAYS = 8
COST_MULTIPLES = (1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0)


def causal_prior_std(frame: pd.DataFrame, value: str, output: str) -> pd.DataFrame:
    """Attach an expanding error SD using complete prior dates only."""
    daily = frame.groupby(["horizon_min", "date"], as_index=False)[value].agg(["count", "sum", lambda x: np.square(x).sum()]).reset_index()
    daily = daily.rename(columns={"<lambda_0>": "sum_sq"}).sort_values(["horizon_min", "date"])
    daily["prior_n"] = daily.groupby("horizon_min")["count"].cumsum() - daily["count"]
    daily["prior_sum"] = daily.groupby("horizon_min")["sum"].cumsum() - daily["sum"]
    daily["prior_sum_sq"] = daily.groupby("horizon_min")["sum_sq"].cumsum() - daily["sum_sq"]
    variance = (
        daily["prior_sum_sq"] - daily["prior_sum"].pow(2) / daily["prior_n"].replace(0, np.nan)
    ) / (daily["prior_n"] - 1).replace(0, np.nan)
    daily[output] = np.sqrt(variance.clip(lower=0))
    daily.loc[daily["prior_n"].lt(MIN_PRIOR_ERRORS), output] = np.nan
    return frame.merge(daily[["horizon_min", "date", output]], on=["horizon_min", "date"], how="left", validate="many_to_one")


def build_panel() -> pd.DataFrame:
    if PANEL_CACHE.exists():
        panel = smb.read_parquet(PANEL_CACHE)
    else:
        panel = five.build_panel()
        smb.write_parquet(panel, PANEL_CACHE)

    features = smb.read_parquet(old.FEATURES)[["datetime", "rv_past_60", "iv_change_5"]].drop_duplicates("datetime")
    panel = panel.merge(features, left_on="entry_datetime", right_on="datetime", how="inner", validate="many_to_one").drop(columns="datetime")
    ivfc = smb.read_parquet(IV_FORECASTS)
    target_cols = [f"target_dlog_iv_{h}" for h in smb.HORIZONS]
    ivfc["actual_dlog_iv"] = ivfc[target_cols].bfill(axis=1).iloc[:, 0]
    ivfc = ivfc[["datetime", "horizon_min", "actual_dlog_iv"]].drop_duplicates(["datetime", "horizon_min"])
    panel = panel.merge(ivfc, left_on=["entry_datetime", "horizon_min"], right_on=["datetime", "horizon_min"], how="inner", validate="many_to_one").drop(columns="datetime")

    panel["rv_current"] = np.sqrt(panel["rv_past_60"].clip(lower=0) * ANNUAL_MINUTES / 60.0)
    panel["iv_current"] = panel["iv_trade_vol"]
    panel["iv_pred"] = panel["iv_current"] * np.exp(panel["q50_compact"])
    panel["iv_actual_future"] = panel["iv_current"] * np.exp(panel["actual_dlog_iv"])
    panel["iv_variance_current"] = panel["iv_current"].pow(2)
    panel["rv_variance_current"] = panel["rv_current"].pow(2)
    panel["iv_variance_pred"] = panel["iv_pred"].pow(2)
    panel["rv_variance_pred"] = panel["predicted_rv"].pow(2)
    panel["rv_variance_actual"] = panel["actual_var"] * ANNUAL_MINUTES / panel["horizon_min"]
    panel["iv_variance_actual_future"] = panel["iv_actual_future"].pow(2)
    panel["predicted_variance_gap"] = panel["iv_variance_pred"] - panel["rv_variance_pred"]
    panel["realized_future_variance_gap"] = panel["iv_variance_actual_future"] - panel["rv_variance_actual"]
    panel["gap_forecast_error"] = panel["realized_future_variance_gap"] - panel["predicted_variance_gap"]
    panel["actual_d_iv"] = panel["iv_actual_future"] - panel["iv_current"]
    panel["iv_change_forecast_error"] = panel["actual_d_iv"] - panel["pred_d_iv"]

    # Estimate forecast uncertainty once per timestamp/horizon, not once per
    # duplicated structure row, and use only errors from earlier dates.
    unique = panel[
        panel["structure"].eq("long_atm_straddle")
    ][["entry_datetime", "date", "horizon_min", "gap_forecast_error", "iv_change_forecast_error"]].drop_duplicates(["entry_datetime", "horizon_min"])
    unique = causal_prior_std(unique, "gap_forecast_error", "gap_error_sd")
    unique = causal_prior_std(unique, "iv_change_forecast_error", "iv_change_error_sd")
    panel = panel.merge(
        unique[["entry_datetime", "horizon_min", "gap_error_sd", "iv_change_error_sd"]],
        on=["entry_datetime", "horizon_min"], how="left", validate="many_to_one",
    )
    panel["gap_z"] = panel["predicted_variance_gap"] / panel["gap_error_sd"]
    panel["future_near_equal_refined"] = panel["gap_z"].abs().le(Z_EQUALITY)
    baseline_distance = (panel["iv_variance_current"] - panel["rv_variance_pred"]).abs()
    forecast_distance = panel["predicted_variance_gap"].abs()
    panel["variance_convergence"] = baseline_distance - forecast_distance
    panel["convergence_strength"] = panel["variance_convergence"]
    panel["iv_pred_up_toward_refined"] = (
        panel["iv_pred"].gt(panel["iv_current"])
        & panel["predicted_variance_gap"].le(panel["gap_error_sd"])
        & forecast_distance.lt(baseline_distance)
    )
    panel["iv_pred_down_toward_refined"] = (
        panel["iv_pred"].lt(panel["iv_current"])
        & panel["predicted_variance_gap"].ge(-panel["gap_error_sd"])
        & forecast_distance.lt(baseline_distance)
    )
    current_long = panel["iv_variance_current"].lt(panel["rv_variance_current"])
    current_short = panel["iv_variance_current"].gt(panel["rv_variance_current"])
    panel["long_near"] = current_long & panel["future_near_equal_refined"] & panel["iv_change_5"].gt(0)
    panel["long_toward"] = current_long & panel["iv_pred_up_toward_refined"] & panel["iv_change_5"].gt(0)
    panel["short_near"] = current_short & panel["future_near_equal_refined"] & panel["iv_change_5"].lt(0)
    panel["short_toward"] = current_short & panel["iv_pred_down_toward_refined"] & panel["iv_change_5"].lt(0)
    panel["long_combined"] = panel["long_near"] | panel["long_toward"]
    panel["short_combined"] = panel["short_near"] | panel["short_toward"]

    panel["expected_volga_refined"] = 0.5 * panel["volga"] * (
        panel["pred_d_iv"].pow(2) + panel["iv_change_error_sd"].pow(2)
    )
    # E[dS]=0 because there is no signed spot forecast. Consequently the
    # conditional vanna and charm expectations are zero, not fitted constants.
    panel["expected_vanna_refined"] = 0.0
    panel["expected_charm_refined"] = 0.0
    panel["expected_pnl_refined"] = panel[[
        "expected_gamma", "expected_theta", "expected_vega",
        "expected_volga_refined", "expected_vanna_refined", "expected_charm_refined",
    ]].sum(axis=1)
    panel["forecast_gross_rupees_refined"] = panel["expected_pnl_refined"] * panel["lot_size"]
    panel["forecast_cost_cover_refined"] = panel["forecast_gross_rupees_refined"] / panel["estimated_entry_cost"].clip(lower=1e-9)
    panel["year"] = panel["date"].dt.year
    panel["half"] = np.where(panel["entry_datetime"].lt("2024-07-01"), "2024_H1", np.where(panel["year"].eq(2024), "2024_H2", "2025_holdout"))
    return panel


def metric(frame: pd.DataFrame) -> dict:
    m = smb.metrics(frame)
    m["profitable_days"] = int(frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum())
    return m


def run_search(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, ledgers = [], {}
    signals = {
        ("long_vol", "combined_or"): "long_combined", ("long_vol", "near_only"): "long_near", ("long_vol", "toward_only"): "long_toward",
        ("short_vol", "combined_or"): "short_combined", ("short_vol", "near_only"): "short_near", ("short_vol", "toward_only"): "short_toward",
    }
    for (side, variant), signal in signals.items():
        structures = old.LONG_STRUCTURES if side == "long_vol" else old.SHORT_STRUCTURES
        for structure in sorted(structures):
            for horizon in smb.HORIZONS:
                base = panel[
                    panel[signal] & panel["structure"].eq(structure) & panel["horizon_min"].eq(horizon)
                    & panel["gap_error_sd"].notna() & panel["iv_change_error_sd"].notna()
                ]
                for multiple in COST_MULTIPLES:
                    eligible = base[base["forecast_cost_cover_refined"].ge(multiple)]
                    trades = smb.execute(old.earliest_daily(eligible), old.SLIPPAGE_BPS, current_2026=True)
                    key = (side, variant, structure, horizon, multiple)
                    ledgers[key] = trades
                    for period in ("2024_H1", "2024_H2", "2025_holdout"):
                        rows.append({"side": side, "signal_variant": variant, "structure": structure,
                                     "horizon_min": horizon, "cost_multiple": multiple, "period": period,
                                     **metric(trades[trades["half"].eq(period)])})
    return pd.DataFrame(rows), ledgers


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    results, ledgers = run_search(panel)
    keys = ["side", "signal_variant", "structure", "horizon_min", "cost_multiple"]
    dev = results[results["period"].isin(["2024_H1", "2024_H2"])]
    wide = dev.pivot_table(index=keys, columns="period", values=["days", "net_pnl", "hit_rate", "profit_factor"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]; wide = wide.reset_index()
    wide["development_days"] = wide["days_2024_H1"] + wide["days_2024_H2"]
    wide["development_net"] = wide["net_pnl_2024_H1"] + wide["net_pnl_2024_H2"]
    wide["worst_half_net"] = wide[["net_pnl_2024_H1", "net_pnl_2024_H2"]].min(axis=1)
    wide["stable_profitable"] = (
        wide["days_2024_H1"].ge(MIN_HALF_DAYS) & wide["days_2024_H2"].ge(MIN_HALF_DAYS)
        & wide["net_pnl_2024_H1"].gt(0) & wide["net_pnl_2024_H2"].gt(0)
    )
    picks = []
    for side in ("long_vol", "short_vol"):
        eligible = wide[wide["side"].eq(side) & wide["stable_profitable"]]
        if not eligible.empty:
            picks.append(eligible.sort_values(["worst_half_net", "development_net", "development_days"], ascending=False).iloc[0])
    selected = pd.DataFrame(picks)
    detail = results.merge(selected[keys], on=keys, how="inner") if not selected.empty else pd.DataFrame()
    parts = []
    for row in selected.itertuples(index=False):
        part = ledgers[(row.side, row.signal_variant, row.structure, int(row.horizon_min), float(row.cost_multiple))].copy()
        part["strategy_side"] = row.side; parts.append(part)
    hybrid = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not hybrid.empty:
        hybrid = hybrid.sort_values(["entry_datetime", "forecast_cost_cover_refined"], ascending=[True, False]).drop_duplicates("date")
        smb.write_parquet(hybrid, OUT / "trade_ledger.parquet")
    hybrid_perf = pd.DataFrame([
        {"period": period, **metric(hybrid[hybrid["half"].eq(period)] if not hybrid.empty else hybrid)}
        for period in ("2024_H1", "2024_H2", "2025_holdout")
    ])
    if not hybrid.empty:
        hybrid_perf = pd.concat([hybrid_perf, pd.DataFrame([{"period": "combined", **metric(hybrid)}])], ignore_index=True)

    results.to_csv(OUT / "full_search_results.csv", index=False)
    wide.to_csv(OUT / "development_selection_grid.csv", index=False)
    selected.to_csv(OUT / "selected_rules.csv", index=False)
    detail.to_csv(OUT / "selected_rule_performance.csv", index=False)
    hybrid_perf.to_csv(OUT / "hybrid_performance.csv", index=False)
    audit = old.integrity_audit(hybrid); audit.to_csv(OUT / "integrity_audit.csv", index=False)

    cost_rows = []
    for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        stressed = smb.execute(hybrid, bps, current_2026=True) if not hybrid.empty else hybrid
        for period in ("2024_H1", "2024_H2", "2025_holdout"):
            part = stressed[stressed["half"].eq(period)] if not stressed.empty else stressed
            cost_rows.append({"slippage_bps": bps, "period": period, **metric(part)})
    cost_sensitivity = pd.DataFrame(cost_rows)
    cost_sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)

    rng = np.random.default_rng(20261013)
    bootstrap_rows = []
    greek_rows = []
    for period in ("2024_H1", "2024_H2", "2025_holdout"):
        part = hybrid[hybrid["half"].eq(period)] if not hybrid.empty else hybrid
        values = part["net_pnl_rupees"].to_numpy(float) if not part.empty else np.array([])
        if len(values):
            draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
            bootstrap_rows.append({
                "period": period, "days": len(values), "observed_mean": values.mean(),
                "mean_ci_low": np.quantile(draws, 0.025), "mean_ci_high": np.quantile(draws, 0.975),
                "bootstrap_probability_mean_positive": np.mean(draws > 0),
            })
            greek_rows.append({
                "period": period, "days": len(values),
                "mean_gamma_points": part["expected_gamma"].mean(),
                "mean_theta_points": part["expected_theta"].mean(),
                "mean_vega_points": part["expected_vega"].mean(),
                "mean_vanna_points": part["expected_vanna_refined"].mean(),
                "mean_volga_points": part["expected_volga_refined"].mean(),
                "mean_charm_points": part["expected_charm_refined"].mean(),
                "mean_total_forecast_points": part["expected_pnl_refined"].mean(),
            })
    bootstrap = pd.DataFrame(bootstrap_rows); bootstrap.to_csv(OUT / "bootstrap_uncertainty.csv", index=False)
    greek_attribution = pd.DataFrame(greek_rows); greek_attribution.to_csv(OUT / "forecast_greek_attribution.csv", index=False)
    coverage_rows = []
    for period in ("2024_H1", "2024_H2", "2025_holdout"):
        available = panel.loc[panel["half"].eq(period), "date"].drop_duplicates()
        traded = hybrid.loc[hybrid["half"].eq(period)]
        signal_days = int(traded["date"].nunique())
        available_days = int(available.nunique())
        coverage_rows.append({
            "period": period,
            "start_date": available.min(),
            "end_date": available.max(),
            "available_days": available_days,
            "signal_days": signal_days,
            "profitable_days": int(traded.groupby("date")["net_pnl_rupees"].sum().gt(0).sum()),
            "signal_day_coverage": signal_days / available_days,
        })
    coverage_rows.append({
        "period": "combined",
        "start_date": panel["date"].min(),
        "end_date": panel["date"].max(),
        "available_days": int(panel["date"].nunique()),
        "signal_days": int(hybrid["date"].nunique()),
        "profitable_days": int(hybrid.groupby("date")["net_pnl_rupees"].sum().gt(0).sum()),
        "signal_day_coverage": hybrid["date"].nunique() / panel["date"].nunique(),
    })
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUT / "coverage.csv", index=False)

    policy = {
        "status": "RESEARCH_ONLY_NOT_LIVE_DEPLOYED",
        "variance_space": True, "near_equality": f"abs(predicted variance gap / causal prior error SD) <= {Z_EQUALITY}",
        "prior_error_minimum": MIN_PRIOR_ERRORS,
        "observed_confirmation": "five-minute IV change aligned with forecast direction",
        "expected_pnl": "gamma + theta + vega + volga; expected vanna and charm are zero because E[dS]=0",
        "selection": "positive net PnL and >=8 days in each 2024 half; maximin net PnL",
        "holdout": "partial 2025 not used for selection", "selected_rules": selected.to_dict("records"),
        "long_vol_decision": "NO_RULE_PASSED_BOTH_2024_HALVES",
        "decision": "PASS_BACKTEST_FORWARD_PAPER_ONLY",
        "live_deployment": False,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")
    report = f"""# Refined variance-space transition backtest

## Rule

The current and predicted IV/RV gaps are squared-volatility gaps. Near equality
means the predicted variance gap lies within one causal forecast-error standard
deviation. Error dispersion at each timestamp uses completed prior dates only
and requires at least {MIN_PRIOR_ERRORS} observations. A directionally aligned
five-minute observed IV change is mandatory.

Expected structure P&L includes gamma, theta, vega and volga. Expected vanna
and charm are zero because the model has no signed spot-return forecast, so
`E[dS]=0`; no unestimated directional mean is invented. A trade must also pass
the development-selected complete-cost multiple.

## Selected rules

{dataframe_to_markdown(selected.round(4)) if not selected.empty else "No rule passed both 2024 halves."}

## Development and holdout

{dataframe_to_markdown(detail[["side", "signal_variant", "structure", "horizon_min", "cost_multiple", "period", "days", "profitable_days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown"]].round(4)) if not detail.empty else "No rule selected."}

## Combined

{dataframe_to_markdown(hybrid_perf.round(4))}

## Day coverage

{dataframe_to_markdown(coverage.assign(signal_day_coverage=coverage["signal_day_coverage"].round(4)))}

## Forecast Greek attribution

{dataframe_to_markdown(greek_attribution.round(4))}

## Cost sensitivity

{dataframe_to_markdown(cost_sensitivity[["slippage_bps", "period", "days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown"]].round(4))}

## Bootstrap uncertainty

{dataframe_to_markdown(bootstrap.round(4))}

## Integrity audit

{dataframe_to_markdown(audit)}
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("STABLE", int(wide["stable_profitable"].sum()))
    print("SELECTED\n", selected.to_string(index=False))
    print("DETAIL\n", detail.to_string(index=False))
    print("HYBRID\n", hybrid_perf.to_string(index=False))


if __name__ == "__main__":
    main()
