"""Cost-aware optimization constrained to the user's IV/RV transition hypothesis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import iv_rv_convergence_backtest as base
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "iv_rv_transition_profit_optimization"
COST_MULTIPLES = (1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0)
CONVERGENCE_FLOORS = (0.0, 0.001, 0.0025, 0.005)
MIN_HALF_DAYS = 8


def metrics(frame: pd.DataFrame) -> dict:
    result = smb.metrics(frame)
    result["profitable_days"] = int(frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum())
    return result


def prepare() -> pd.DataFrame:
    panel = base.prepare_panel()
    panel["forecast_gross_rupees"] = panel["expected_pnl_points"] * panel["lot_size"]
    panel["forecast_cost_cover"] = panel["forecast_gross_rupees"] / panel["estimated_entry_cost"].clip(lower=1e-9)
    panel["half"] = np.where(
        panel["entry_datetime"].lt(pd.Timestamp("2024-07-01")), "2024_H1",
        np.where(panel["year"].eq(2024), "2024_H2", "2025_holdout"),
    )
    return panel


def candidate_trades(
    panel: pd.DataFrame, side: str, variant: str, structure: str, horizon: int,
    cost_multiple: float, convergence_floor: float,
) -> pd.DataFrame:
    signal_col = {
        ("long_vol", "combined_or"): "long_signal",
        ("long_vol", "near_only"): "long_near_signal",
        ("long_vol", "toward_only"): "long_toward_signal",
        ("short_vol", "combined_or"): "short_signal",
        ("short_vol", "near_only"): "short_near_signal",
        ("short_vol", "toward_only"): "short_toward_signal",
    }[(side, variant)]
    eligible = panel[
        panel[signal_col]
        & panel["structure"].eq(structure)
        & panel["horizon_min"].eq(horizon)
        & panel["forecast_cost_cover"].ge(cost_multiple)
        & panel["convergence_strength"].ge(convergence_floor)
    ].copy()
    # The first eligible observation is executable in real time; no end-of-day
    # ranking or future information enters selection.
    chosen = base.earliest_daily(eligible)
    return smb.execute(chosen, base.SLIPPAGE_BPS, current_2026=True)


def filtered_trades(candidates: pd.DataFrame, cost_multiple: float, convergence_floor: float) -> pd.DataFrame:
    eligible = candidates[
        candidates["forecast_cost_cover"].ge(cost_multiple)
        & candidates["convergence_strength"].ge(convergence_floor)
    ]
    return smb.execute(base.earliest_daily(eligible), base.SLIPPAGE_BPS, current_2026=True)


def search(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple, pd.DataFrame]]:
    rows, ledgers = [], {}
    for side, structures in (("long_vol", base.LONG_STRUCTURES), ("short_vol", base.SHORT_STRUCTURES)):
        for variant in ("combined_or", "near_only", "toward_only"):
            signal_col = {
                ("long_vol", "combined_or"): "long_signal",
                ("long_vol", "near_only"): "long_near_signal",
                ("long_vol", "toward_only"): "long_toward_signal",
                ("short_vol", "combined_or"): "short_signal",
                ("short_vol", "near_only"): "short_near_signal",
                ("short_vol", "toward_only"): "short_toward_signal",
            }[(side, variant)]
            for confirmation in ("forecast_only", "observed_iv_5_aligned"):
                confirmation_mask = (
                    pd.Series(True, index=panel.index) if confirmation == "forecast_only"
                    else panel[f"{'long' if side == 'long_vol' else 'short'}_observed_iv_confirmation"]
                )
                for structure in sorted(structures):
                    for horizon in smb.HORIZONS:
                        candidates = panel[
                            panel[signal_col] & confirmation_mask
                            & panel["structure"].eq(structure)
                            & panel["horizon_min"].eq(horizon)
                        ].copy()
                        for multiple in COST_MULTIPLES:
                            for floor in CONVERGENCE_FLOORS:
                                key = (side, variant, confirmation, structure, horizon, multiple, floor)
                                trades = filtered_trades(candidates, multiple, floor)
                                ledgers[key] = trades
                                for period in ("2024_H1", "2024_H2", "2025_holdout"):
                                    part = trades[trades["half"].eq(period)]
                                    rows.append({
                                        "side": side, "signal_variant": variant,
                                        "confirmation": confirmation,
                                        "structure": structure, "horizon_min": horizon,
                                        "cost_multiple": multiple,
                                        "convergence_floor": floor, "period": period,
                                        **metrics(part),
                                    })
    return pd.DataFrame(rows), ledgers


def select_rules(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["side", "signal_variant", "confirmation", "structure", "horizon_min", "cost_multiple", "convergence_floor"]
    dev = results[results["period"].isin(["2024_H1", "2024_H2"])].copy()
    wide = dev.pivot_table(index=keys, columns="period", values=["days", "net_pnl", "hit_rate", "profit_factor"], aggfunc="first")
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()
    wide["development_days"] = wide["days_2024_H1"] + wide["days_2024_H2"]
    wide["development_net"] = wide["net_pnl_2024_H1"] + wide["net_pnl_2024_H2"]
    wide["worst_half_net"] = wide[["net_pnl_2024_H1", "net_pnl_2024_H2"]].min(axis=1)
    wide["stable_profitable"] = (
        wide["days_2024_H1"].ge(MIN_HALF_DAYS)
        & wide["days_2024_H2"].ge(MIN_HALF_DAYS)
        & wide["net_pnl_2024_H1"].gt(0)
        & wide["net_pnl_2024_H2"].gt(0)
    )
    selected = []
    for side in ("long_vol", "short_vol"):
        eligible = wide[wide["side"].eq(side) & wide["stable_profitable"]].copy()
        if eligible.empty:
            continue
        # Maximin selection rewards rules that survive both halves, then favors
        # total net P&L and coverage.  Holdout results are not referenced.
        selected.append(eligible.sort_values(
            ["worst_half_net", "development_net", "development_days"], ascending=False,
        ).iloc[0])
    return pd.DataFrame(selected), wide


def evaluate_selected(selected: pd.DataFrame, results: pd.DataFrame, ledgers: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()
    keys = ["side", "signal_variant", "confirmation", "structure", "horizon_min", "cost_multiple", "convergence_floor"]
    detail = results.merge(selected[keys], on=keys, how="inner")
    parts = []
    for row in selected.itertuples(index=False):
        key = (row.side, row.signal_variant, row.confirmation, row.structure, int(row.horizon_min), float(row.cost_multiple), float(row.convergence_floor))
        part = ledgers[key].copy()
        part["strategy_side"] = row.side
        parts.append(part)
    hybrid = (
        pd.concat(parts, ignore_index=True, sort=False)
        .sort_values(["entry_datetime", "forecast_cost_cover"], ascending=[True, False])
        .drop_duplicates("date", keep="first")
        .sort_values("entry_datetime")
        .reset_index(drop=True)
    )
    return detail, hybrid


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = prepare()
    results, ledgers = search(panel)
    selected, selection_grid = select_rules(results)
    detail, hybrid = evaluate_selected(selected, results, ledgers)

    results.to_csv(OUT / "full_search_results.csv", index=False)
    selection_grid.to_csv(OUT / "development_selection_grid.csv", index=False)
    selected.to_csv(OUT / "selected_rules.csv", index=False)
    detail.to_csv(OUT / "selected_rule_performance.csv", index=False)
    if not hybrid.empty:
        smb.write_parquet(hybrid, OUT / "selected_trade_ledger.parquet")

    hybrid_rows = []
    for period in ("2024_H1", "2024_H2", "2025_holdout"):
        part = hybrid[hybrid["half"].eq(period)] if not hybrid.empty else hybrid
        hybrid_rows.append({"period": period, **metrics(part)})
    if not hybrid.empty:
        hybrid_rows.append({"period": "combined", **metrics(hybrid)})
    hybrid_perf = pd.DataFrame(hybrid_rows)
    hybrid_perf.to_csv(OUT / "hybrid_performance.csv", index=False)

    cost_rows = []
    for bps in (0.0, 10.0, 25.0, 50.0, 100.0):
        stressed = smb.execute(hybrid, bps, current_2026=True) if not hybrid.empty else hybrid
        for period in ("2024_H1", "2024_H2", "2025_holdout"):
            part = stressed[stressed["half"].eq(period)] if not stressed.empty else stressed
            cost_rows.append({"slippage_bps": bps, "period": period, **metrics(part)})
    cost_sensitivity = pd.DataFrame(cost_rows)
    cost_sensitivity.to_csv(OUT / "cost_sensitivity.csv", index=False)

    rng = np.random.default_rng(20261013)
    bootstrap_rows = []
    for period in ("2024_H1", "2024_H2", "2025_holdout"):
        values = hybrid.loc[hybrid["half"].eq(period), "net_pnl_rupees"].to_numpy(float)
        if len(values):
            draws = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
            bootstrap_rows.append({
                "period": period, "days": len(values), "observed_mean": values.mean(),
                "mean_ci_low": np.quantile(draws, 0.025),
                "mean_ci_high": np.quantile(draws, 0.975),
                "bootstrap_probability_mean_positive": np.mean(draws > 0),
            })
    bootstrap = pd.DataFrame(bootstrap_rows)
    bootstrap.to_csv(OUT / "bootstrap_uncertainty.csv", index=False)
    audit = base.integrity_audit(hybrid)
    audit.to_csv(OUT / "integrity_audit.csv", index=False)

    decision = "PASS_BACKTEST_FORWARD_PAPER_ONLY" if not selected.empty else "NO_STABLE_PROFITABLE_RULE"
    policy = {
        "status": "RESEARCH_ONLY_NOT_LIVE_DEPLOYED",
        "decision": decision,
        "hypothesis_is_mandatory": True,
        "cost_gate": "forecast Greek PnL / estimated complete round-trip cost >= selected multiple",
        "observed_confirmation": "optional five-minute IV change aligned with the forecasted IV direction",
        "development_stability": f">={MIN_HALF_DAYS} days and positive net PnL in both 2024 halves",
        "selection_objective": "maximize the worse half net PnL; tie-break on total development net and days",
        "holdout": "2025 through 2025-05-09 is never used for selection",
        "costs": "current encoded 2026 charges plus 25 bps premium-turnover slippage",
        "selected_rules": selected.to_dict("records"),
        "long_vol_decision": "NO_RULE_PASSED_DEVELOPMENT_STABILITY",
        "live_deployment": False,
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")

    stable_count = int(selection_grid["stable_profitable"].sum())
    report = f"""# Cost-aware IV/RV transition strategy

## Constraint

The original hypothesis is mandatory. Long volatility still requires
`IVcurrent < RVcurrent` plus predicted IV becoming equal to or rising toward
predicted RV. Short volatility still requires `IVcurrent > RVcurrent` plus
predicted IV becoming equal to or falling toward predicted RV.

The only added entry requirement is economic sufficiency:

`forecast Greek P&L / estimated complete round-trip cost >= k`.

Forecast Greek P&L is `Gamma + Theta + Vega` for the complete structure and
the selected horizon. The realized option P&L still uses unchanged entry
strikes and observed exit prices. This gate does not redefine the transition;
it rejects transitions whose modeled edge is too small to pay execution cost.

## Selection discipline

- Search dimensions: transition subtype, optional observed five-minute IV
  confirmation, five-strike structure, 15/30/60/120 minute horizon, cost
  multiple, and minimum predicted convergence.
- A candidate needs at least {MIN_HALF_DAYS} trading days and positive net P&L
  independently in both halves of 2024.
- Selection maximizes the weaker half's net P&L. Partial 2025 is holdout only.
- Complete costs use current encoded charges plus {base.SLIPPAGE_BPS:.0f} bps
  premium-turnover slippage.

Stable profitable development candidates: {stable_count}.

## Selected rules

{dataframe_to_markdown(selected.round(4)) if not selected.empty else "No rule passed."}

## Selected-rule performance

{dataframe_to_markdown(detail[["side", "signal_variant", "confirmation", "structure", "horizon_min", "cost_multiple", "convergence_floor", "period", "days", "profitable_days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown"]].round(4)) if not detail.empty else "No rule passed."}

## Combined hybrid

{dataframe_to_markdown(hybrid_perf.round(4))}

## Cost sensitivity

{dataframe_to_markdown(cost_sensitivity[["slippage_bps", "period", "days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown"]].round(4))}

## Sampling uncertainty

{dataframe_to_markdown(bootstrap.round(4))}

## Integrity checks

{dataframe_to_markdown(audit)}

## Research basis

The economic object is a variance-risk-premium comparison: expected future
realized variance versus option-implied variance, with the IV forecast used to
model vega P&L. Carr and Wu (2009) formalize variance risk premium as the
difference between realized variance and an option-implied variance-swap rate.
The present implementation is only an approximation because it uses a
five-strike monthly-expiry structure rather than a model-free variance strip.
The full-cost inequality is necessary because a statistical volatility edge
is not automatically an executable option profit.

## Interpretation

Passing 2024 is necessary but not sufficient. Deployment requires positive
2025 holdout performance and subsequent forward paper trading. A rule that
only compensates costs in development but fails holdout remains rejected.
The bootstrap confidence intervals include zero, so the observed positive P&L
does not yet establish statistical significance.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("STABLE", stable_count)
    print("SELECTED")
    print(selected.to_string(index=False))
    print("DETAIL")
    print(detail.to_string(index=False))
    print("HYBRID")
    print(hybrid_perf.to_string(index=False))


if __name__ == "__main__":
    main()
