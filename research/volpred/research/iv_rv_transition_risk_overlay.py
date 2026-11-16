"""Causal take-profit/stop-loss overlay for cost-covered IV/RV transitions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import iv_rv_convergence_backtest as hypothesis
import iv_rv_transition_profit_optimization as cost_model
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "iv_rv_transition_profit_optimization" / "risk_overlay"
FINAL_HORIZON = 120
CHECKPOINTS = (15, 30, 60, 120)
COST_MULTIPLES = (1.0, 1.5, 2.0, 3.0, 5.0)
TAKE_PROFITS = (0.10, 0.20, 0.30, 0.50, np.inf)
STOP_LOSSES = (0.25, 0.50, 0.75, 1.00, np.inf)
MIN_HALF_DAYS = 8
STRUCTURES = {
    "long_vol": ("long_atm_straddle", "long_1_strangle", "long_2_strangle"),
    "short_vol": ("short_atm_straddle", "short_1_strangle", "short_2_strangle"),
}


def build_paths(panel: pd.DataFrame, structure: str) -> pd.DataFrame:
    pieces = []
    for horizon in CHECKPOINTS:
        raw = panel[
            panel["structure"].eq(structure)
            & panel["horizon_min"].eq(horizon)
        ].copy()
        executed = smb.execute(raw, hypothesis.SLIPPAGE_BPS, current_2026=True)
        pieces.append(executed[[
            "entry_datetime", "date", "horizon_min", "gross_pnl_points",
            "net_pnl_rupees", "gross_pnl_rupees", "total_cost", "exit_datetime",
        ]])
    return pd.concat(pieces, ignore_index=True)


def apply_overlay(entries: pd.DataFrame, paths: pd.DataFrame, take_profit: float, stop_loss: float) -> pd.DataFrame:
    if entries.empty:
        return entries.copy()
    path = paths.merge(
        entries[["entry_datetime", "entry_value", "half", "forecast_cost_cover"]],
        on="entry_datetime", how="inner", validate="many_to_one",
    )
    path["gross_return_on_premium"] = path["gross_pnl_points"] / path["entry_value"].abs().clip(lower=1e-9)
    path["trigger"] = (
        path["gross_return_on_premium"].ge(take_profit)
        | path["gross_return_on_premium"].le(-stop_loss)
        | path["horizon_min"].eq(FINAL_HORIZON)
    )
    chosen = (
        path[path["trigger"]]
        .sort_values(["entry_datetime", "horizon_min"])
        .drop_duplicates("entry_datetime", keep="first")
        .copy()
    )
    return chosen


def metric(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"trades": 0, "days": 0, "net_pnl": 0.0, "mean_net": np.nan,
                "hit_rate": np.nan, "profit_factor": np.nan, "max_drawdown": np.nan,
                "profitable_days": 0, "mean_exit_min": np.nan}
    pnl = frame["net_pnl_rupees"]
    daily = frame.groupby("date")["net_pnl_rupees"].sum().sort_index()
    cumulative = daily.cumsum()
    drawdown = cumulative.cummax() - cumulative
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return {
        "trades": len(frame), "days": daily.size, "net_pnl": pnl.sum(),
        "mean_net": pnl.mean(), "hit_rate": pnl.gt(0).mean(),
        "profit_factor": wins / losses if losses > 0 else np.inf,
        "max_drawdown": drawdown.max(), "profitable_days": int(daily.gt(0).sum()),
        "mean_exit_min": frame["horizon_min"].mean(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = cost_model.prepare()
    rows, ledgers = [], {}
    for side, structures in STRUCTURES.items():
        for variant in ("combined_or", "near_only", "toward_only"):
            signal_col = {
                ("long_vol", "combined_or"): "long_signal",
                ("long_vol", "near_only"): "long_near_signal",
                ("long_vol", "toward_only"): "long_toward_signal",
                ("short_vol", "combined_or"): "short_signal",
                ("short_vol", "near_only"): "short_near_signal",
                ("short_vol", "toward_only"): "short_toward_signal",
            }[(side, variant)]
            for structure in structures:
                final = panel[
                    panel[signal_col]
                    & panel["structure"].eq(structure)
                    & panel["horizon_min"].eq(FINAL_HORIZON)
                ].copy()
                paths = build_paths(panel, structure)
                for multiple in COST_MULTIPLES:
                    entries = hypothesis.earliest_daily(final[final["forecast_cost_cover"].ge(multiple)])
                    for take_profit in TAKE_PROFITS:
                        for stop_loss in STOP_LOSSES:
                            key = (side, variant, structure, multiple, take_profit, stop_loss)
                            trades = apply_overlay(entries, paths, take_profit, stop_loss)
                            ledgers[key] = trades
                            for period in ("2024_H1", "2024_H2", "2025_holdout"):
                                rows.append({
                                    "side": side, "signal_variant": variant, "structure": structure,
                                    "cost_multiple": multiple, "take_profit": take_profit,
                                    "stop_loss": stop_loss, "period": period,
                                    **metric(trades[trades["half"].eq(period)]),
                                })
    results = pd.DataFrame(rows)
    keys = ["side", "signal_variant", "structure", "cost_multiple", "take_profit", "stop_loss"]
    dev = results[results["period"].isin(["2024_H1", "2024_H2"])]
    wide = dev.pivot_table(index=keys, columns="period", values=["days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown"], aggfunc="first")
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide["development_net"] = wide["net_pnl_2024_H1"] + wide["net_pnl_2024_H2"]
    wide["worst_half_net"] = wide[["net_pnl_2024_H1", "net_pnl_2024_H2"]].min(axis=1)
    wide["development_days"] = wide["days_2024_H1"] + wide["days_2024_H2"]
    wide["stable_profitable"] = (
        wide["days_2024_H1"].ge(MIN_HALF_DAYS) & wide["days_2024_H2"].ge(MIN_HALF_DAYS)
        & wide["net_pnl_2024_H1"].gt(0) & wide["net_pnl_2024_H2"].gt(0)
    )
    selected_rows = []
    for side in ("long_vol", "short_vol"):
        eligible = wide[wide["side"].eq(side) & wide["stable_profitable"]]
        if not eligible.empty:
            selected_rows.append(eligible.sort_values(
                ["worst_half_net", "development_net", "development_days"], ascending=False,
            ).iloc[0])
    selected = pd.DataFrame(selected_rows)
    detail = results.merge(selected[keys], on=keys, how="inner") if not selected.empty else pd.DataFrame()

    parts = []
    for row in selected.itertuples(index=False):
        key = (row.side, row.signal_variant, row.structure, float(row.cost_multiple), float(row.take_profit), float(row.stop_loss))
        part = ledgers[key].copy(); part["strategy_side"] = row.side; parts.append(part)
    hybrid = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not hybrid.empty:
        hybrid = hybrid.sort_values(["entry_datetime", "forecast_cost_cover"], ascending=[True, False]).drop_duplicates("date")
        smb.write_parquet(hybrid, OUT / "selected_trade_ledger.parquet")
    hybrid_perf = pd.DataFrame([
        {"period": period, **metric(hybrid[hybrid["half"].eq(period)] if not hybrid.empty else hybrid)}
        for period in ("2024_H1", "2024_H2", "2025_holdout")
    ])
    if not hybrid.empty:
        hybrid_perf = pd.concat([hybrid_perf, pd.DataFrame([{"period": "combined", **metric(hybrid)}])], ignore_index=True)

    results.to_csv(OUT / "full_overlay_search.csv", index=False)
    wide.to_csv(OUT / "development_selection_grid.csv", index=False)
    selected.to_csv(OUT / "selected_rules.csv", index=False)
    detail.to_csv(OUT / "selected_rule_performance.csv", index=False)
    hybrid_perf.to_csv(OUT / "hybrid_performance.csv", index=False)

    policy = {
        "status": "RESEARCH_ONLY_NOT_LIVE_DEPLOYED",
        "hypothesis_is_mandatory": True,
        "entry_cost_gate": "forecast Greek PnL >= selected multiple times complete cost",
        "risk_checks_minutes": list(CHECKPOINTS),
        "take_profit_and_stop_loss_units": "gross option PnL divided by absolute entry premium",
        "selection_uses": "2024 H1 and H2 only",
        "holdout": "partial 2025, never used for selection",
        "selected_rules": selected.to_dict("records"),
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, default=str), encoding="utf-8")
    report = f"""# Risk-managed IV/RV transition strategy

The user's long/short transition hypothesis remains a mandatory first gate.
Entries must also forecast enough Gamma + Theta + Vega P&L to cover a selected
multiple of complete costs. The risk overlay checks the unchanged option
strikes after 15, 30, and 60 minutes and otherwise exits after 120 minutes.

Take-profit and stop-loss levels are fractions of the absolute entry premium.
All rule choices use 2024 only and must have at least {MIN_HALF_DAYS} days plus
positive net P&L in both halves. Partial 2025 is holdout only.

## Selected rules

{dataframe_to_markdown(selected.round(4)) if not selected.empty else "No rule passed."}

## Development and holdout

{dataframe_to_markdown(detail[["side", "signal_variant", "structure", "cost_multiple", "take_profit", "stop_loss", "period", "days", "profitable_days", "net_pnl", "hit_rate", "profit_factor", "max_drawdown", "mean_exit_min"]].round(4)) if not detail.empty else "No rule passed."}

## Combined performance

{dataframe_to_markdown(hybrid_perf.round(4))}

The holdout must be positive by a meaningful margin before considering forward
paper deployment. Short-vol tail exposure remains even with checkpoint stops,
because prices can gap between observations.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print("STABLE", int(wide["stable_profitable"].sum()))
    print("SELECTED\n", selected.to_string(index=False))
    print("DETAIL\n", detail.to_string(index=False))
    print("HYBRID\n", hybrid_perf.to_string(index=False))


if __name__ == "__main__":
    main()
