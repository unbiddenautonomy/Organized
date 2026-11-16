"""Intersect the IV/RV transition hypothesis with pre-2024 profitability gates."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import iv_rv_convergence_backtest as hypothesis
import iv_rv_transition_profit_optimization as profit
import retrained_short_vol_hybrid as trained
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "iv_rv_transition_profit_optimization" / "pre2024_model_gate"
ARTIFACT = ROOT / "model" / "regimeUnderstanding" / "expanded_signal_hybrid" / "expanded_hybrid_models.joblib"


def scored_panel() -> tuple[pd.DataFrame, dict]:
    panel = profit.prepare()
    artifact = joblib.load(ARTIFACT)
    features = smb.read_parquet(trained.FEATURE_PANEL)[["datetime", *trained.BASE_FEATURES]].drop_duplicates("datetime")
    panel = panel.merge(features, left_on="entry_datetime", right_on="datetime", how="inner", validate="many_to_one").drop(columns="datetime")
    panel["entry_value_pct"] = panel["entry_value"].abs() / panel["spot_price"]
    panel["gamma_s2"] = panel["gamma"] * panel["spot_price"].pow(2)
    for col in artifact["features"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce")
    finite = np.isfinite(panel[artifact["features"]].to_numpy(float)).all(axis=1)
    panel["model_probability"] = np.nan
    primary = finite & panel["structure"].eq("short_2_strangle") & panel["horizon_min"].eq(120)
    secondary = finite & panel["structure"].eq("short_atm_straddle") & panel["horizon_min"].eq(120)
    panel.loc[primary, "model_probability"] = artifact["primary_model"].predict_proba(panel.loc[primary, artifact["features"]])[:, 1]
    panel.loc[secondary, "model_probability"] = artifact["secondary_model"].predict_proba(panel.loc[secondary, artifact["features"]])[:, 1]
    return panel, artifact


def metric(frame: pd.DataFrame) -> dict:
    m = smb.metrics(frame)
    m["profitable_days"] = int(frame.groupby("date")["net_pnl_rupees"].sum().gt(0).sum())
    return m


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel, artifact = scored_panel()
    specs = [
        ("short_2_strangle", float(artifact["primary_threshold"])),
        ("short_atm_straddle", float(artifact["secondary_threshold"])),
    ]
    rows, ledgers = [], []
    for structure, threshold in specs:
        base = panel[
            panel["short_signal"] & panel["structure"].eq(structure)
            & panel["horizon_min"].eq(120) & panel["model_probability"].ge(threshold)
        ].copy()
        for gate_name, candidates in (
            ("hypothesis_plus_pre2024_model", base),
            ("hypothesis_model_plus_cost1", base[base["forecast_cost_cover"].ge(1.0)]),
        ):
            trades = smb.execute(hypothesis.earliest_daily(candidates), hypothesis.SLIPPAGE_BPS, current_2026=True)
            trades["gate"] = gate_name; trades["component"] = structure; ledgers.append(trades)
            for period, mask in (
                ("2024", trades["year"].eq(2024)),
                ("2025_holdout", trades["year"].eq(2025)),
                ("combined", pd.Series(True, index=trades.index)),
            ):
                rows.append({"structure": structure, "gate": gate_name, "probability_threshold": threshold,
                             "period": period, **metric(trades[mask])})
    results = pd.DataFrame(rows)

    # Fixed priority decided before evaluation: two-strike strangle first, ATM
    # straddle only on dates without a primary signal. Both keep the hypothesis
    # and pre-2024 model threshold mandatory; cost1 is an economic identity.
    primary = ledgers[1].copy()
    secondary = ledgers[3].copy()
    hybrid = pd.concat([
        primary,
        secondary[~secondary["date"].isin(primary["date"])],
    ], ignore_index=True, sort=False).sort_values("entry_datetime")
    hybrid_rows = []
    for period, part in (("2024", hybrid[hybrid["year"].eq(2024)]),
                         ("2025_holdout", hybrid[hybrid["year"].eq(2025)]),
                         ("combined", hybrid)):
        hybrid_rows.append({"period": period, **metric(part)})
    hybrid_perf = pd.DataFrame(hybrid_rows)
    results.to_csv(OUT / "component_performance.csv", index=False)
    hybrid_perf.to_csv(OUT / "hybrid_performance.csv", index=False)
    smb.write_parquet(hybrid, OUT / "hybrid_trade_ledger.parquet")
    policy = {
        "status": "RESEARCH_ONLY_NOT_LIVE_DEPLOYED",
        "mandatory_transition": "IVcurrent > RVcurrent and (IVpred approximately RVpred or IVpred falls toward RVpred)",
        "mandatory_cost_gate": "forecast Greek PnL >= complete estimated cost",
        "primary": {"structure": "short_2_strangle", "threshold": artifact["primary_threshold"]},
        "secondary": {"structure": "short_atm_straddle", "threshold": artifact["secondary_threshold"]},
        "model_training": "through 2022",
        "threshold_selection": "2023",
        "evaluation": "2024 and partial 2025",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    report = f"""# IV/RV transition with pre-2024 profitability gates

The user's short-vol transition remains mandatory. A trade must additionally
pass a profitability-model threshold trained through 2022 and fixed on 2023,
and forecast Gamma + Theta + Vega P&L must cover complete estimated cost.
No 2024 or 2025 result selects these thresholds.

## Components

{dataframe_to_markdown(results.round(4))}

## Fixed-priority hybrid

{dataframe_to_markdown(hybrid_perf.round(4))}

The primary trade is the 120-minute short two-strike OTM strangle. The ATM
short straddle is used only on a date without a primary signal. All legs use
one monthly expiry and unchanged entry strikes.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(results.to_string(index=False)); print(hybrid_perf.to_string(index=False))


if __name__ == "__main__":
    main()
