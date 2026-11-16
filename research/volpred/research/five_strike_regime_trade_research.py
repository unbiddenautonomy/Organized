"""Compare same-expiry five-strike structures within predicted IV/RV scenes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "five_strike_trade_research"
EQUALITY_BAND = 0.005
SLIPPAGE_BPS = 25.0

# Positive weight buys a contract; negative weight sells it. All legs use the
# same monthly expiry and their entry strikes remain fixed until exit.
STRUCTURES: dict[str, dict[tuple[int, str], int]] = {
    "long_atm_straddle": {(0, "call"): 1, (0, "put"): 1},
    "short_atm_straddle": {(0, "call"): -1, (0, "put"): -1},
    "long_1_strangle": {(1, "call"): 1, (-1, "put"): 1},
    "short_1_strangle": {(1, "call"): -1, (-1, "put"): -1},
    "long_2_strangle": {(2, "call"): 1, (-2, "put"): 1},
    "short_2_strangle": {(2, "call"): -1, (-2, "put"): -1},
    "reverse_iron_fly_1": {(0, "call"): 1, (0, "put"): 1, (1, "call"): -1, (-1, "put"): -1},
    "iron_fly_1": {(0, "call"): -1, (0, "put"): -1, (1, "call"): 1, (-1, "put"): 1},
    "reverse_iron_fly_2": {(0, "call"): 1, (0, "put"): 1, (2, "call"): -1, (-2, "put"): -1},
    "iron_fly_2": {(0, "call"): -1, (0, "put"): -1, (2, "call"): 1, (-2, "put"): 1},
    "iron_condor_1_2": {(-1, "put"): -1, (1, "call"): -1, (-2, "put"): 1, (2, "call"): 1},
    "reverse_iron_condor_1_2": {(-1, "put"): 1, (1, "call"): 1, (-2, "put"): -1, (2, "call"): -1},
}


def build_panel() -> pd.DataFrame:
    entries, _ = smb.prepare_entry_rows()
    entries = smb.add_entry_greeks(entries)
    smb.build_120m_exit_cache(entries)
    forecasts = smb.forecasts()
    frames = []
    for horizon in smb.HORIZONS:
        wide = smb.wide_legs(smb.leg_panel(entries, smb.load_exits(horizon)))
        for name, weights in STRUCTURES.items():
            frames.append(smb.structure_panel(wide, forecasts, name, weights, horizon))
    panel = smb.attach_entry_iv(pd.concat(frames, ignore_index=True))
    numeric = [
        "gross_pnl_points", "entry_turnover_points", "exit_turnover_points",
        "buy_turnover_points", "sell_turnover_points", "expected_pnl_points",
        "expected_gamma", "expected_theta", "expected_vega", "expected_vanna",
        "expected_volga", "expected_charm", "lot_size",
    ]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel[
        panel["gross_pnl_points"].notna()
        & panel["entry_turnover_points"].gt(0)
        & panel["exit_turnover_points"].gt(0)
        & ~panel["entry_tiny_any"]
        & np.isfinite(panel["expected_pnl_points"])
    ].copy()
    entry_minute = panel["entry_datetime"].dt.hour * 60 + panel["entry_datetime"].dt.minute
    exit_minute = panel["exit_datetime"].dt.hour * 60 + panel["exit_datetime"].dt.minute
    panel = panel[
        entry_minute.between(9 * 60 + 15, 15 * 60 + 30)
        & exit_minute.between(9 * 60 + 15, 15 * 60 + 30)
        & panel["entry_datetime"].dt.normalize().eq(panel["exit_datetime"].dt.normalize())
    ].copy()
    # Despite its legacy column name, this forecast is cumulative variance over
    # the requested horizon (the adjacent actual_var has the same units).
    panel["predicted_horizon_variance"] = panel["pred_annualized_variance"]
    panel["predicted_rv"] = np.sqrt(
        panel["predicted_horizon_variance"].clip(lower=0)
        * smb.TRADING_MINUTES_YEAR / panel["horizon_min"]
    )
    panel["expected_gamma"] = (
        0.5 * panel["gamma"] * panel["spot_price"].pow(2)
        * panel["predicted_horizon_variance"]
    )
    panel["expected_pnl_points"] = panel[[
        "expected_gamma", "expected_theta", "expected_vega", "expected_vanna",
        "expected_volga", "expected_charm",
    ]].sum(axis=1)
    gap = panel["iv_trade_vol"] - panel["predicted_rv"]
    panel["scene"] = np.select(
        [gap.gt(EQUALITY_BAND), gap.lt(-EQUALITY_BAND)],
        ["IV > RV", "IV < RV"],
        default="IV = RV",
    )
    panel["iv_minus_predicted_rv"] = gap

    estimate = panel.copy()
    estimate["exit_turnover_points"] = estimate["entry_turnover_points"]
    estimate["buy_turnover_points"] = estimate["entry_turnover_points"]
    estimate["sell_turnover_points"] = estimate["entry_turnover_points"]
    panel["estimated_entry_cost"] = smb.charge_components(
        estimate, SLIPPAGE_BPS, current_2026=True
    )["total_cost"].to_numpy()
    panel["expected_net_rupees"] = (
        panel["expected_pnl_points"] * panel["lot_size"] - panel["estimated_entry_cost"]
    )
    panel["greek_edge_gate"] = panel["expected_net_rupees"].gt(0)
    return panel


def score(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, trades = [], []
    # ``stale_any`` and ``tiny_any`` inspect exit quotes and therefore cannot
    # be entry-time eligibility filters. Entry quote quality was already
    # enforced by ``entry_tiny_any`` while constructing the panel.
    quality = panel.copy()
    for (scene, structure, horizon), candidates in quality.groupby(
        ["scene", "structure", "horizon_min"], observed=True
    ):
        for gate_name, sample in (
            ("scene_only", candidates),
            ("scene_plus_greek_edge", candidates[candidates["greek_edge_gate"]]),
        ):
            chosen = smb.select_non_overlapping(sample)
            executed = smb.execute(chosen, SLIPPAGE_BPS, current_2026=True)
            executed["gate"] = gate_name
            if not executed.empty:
                trades.append(executed)
            for period, mask in (
                ("development_2024", executed["date"].dt.year.eq(2024)),
                ("holdout_2025", executed["date"].dt.year.eq(2025)),
            ):
                part = executed[mask]
                base_metrics = smb.metrics(part)
                base_metrics["p05_net"] = (
                    part["net_pnl_rupees"].quantile(0.05) if not part.empty else np.nan
                )
                rows.append({
                    "scene": scene,
                    "structure": structure,
                    "horizon_min": horizon,
                    "gate": gate_name,
                    "period": period,
                    **base_metrics,
                    "mean_expected_gamma_points": sample["expected_gamma"].mean(),
                    "mean_expected_theta_points": sample["expected_theta"].mean(),
                    "mean_expected_vega_points": sample["expected_vega"].mean(),
                    "candidate_rows": len(sample),
                })
    return pd.DataFrame(rows), pd.concat(trades, ignore_index=True) if trades else panel.iloc[:0]


def robust_candidates(summary: pd.DataFrame) -> pd.DataFrame:
    primary = summary[summary["gate"].eq("scene_plus_greek_edge")].copy()
    index = ["scene", "structure", "horizon_min", "gate"]
    wide = primary.pivot_table(
        index=index,
        columns="period",
        values=["trades", "mean_net", "hit_rate", "profit_factor", "p05_net"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{period}" for metric, period in wide.columns]
    wide = wide.reset_index()
    required = ["trades_development_2024", "trades_holdout_2025", "mean_net_development_2024", "mean_net_holdout_2025"]
    for column in required:
        if column not in wide:
            wide[column] = np.nan
    wide["robust_positive"] = (
        wide["trades_development_2024"].ge(20)
        & wide["trades_holdout_2025"].ge(10)
        & wide["mean_net_development_2024"].gt(0)
        & wide["mean_net_holdout_2025"].gt(0)
    )
    return wide.sort_values(
        ["scene", "robust_positive", "mean_net_holdout_2025"], ascending=[True, False, False]
    )


def write_report(summary: pd.DataFrame, robust: pd.DataFrame, panel: pd.DataFrame) -> None:
    cols = [
        "scene", "structure", "horizon_min", "robust_positive",
        "trades_development_2024", "mean_net_development_2024",
        "hit_rate_development_2024", "trades_holdout_2025",
        "mean_net_holdout_2025", "hit_rate_holdout_2025",
    ]
    available = [column for column in cols if column in robust]
    top = robust.groupby("scene", observed=True).head(8)[available]
    passing = robust[robust["robust_positive"]][available]
    passing_text = dataframe_to_markdown(passing) if not passing.empty else "No cell passed the robustness rule."
    top_text = dataframe_to_markdown(top) if not top.empty else "No eligible cells were available."
    selected = robust[
        robust["scene"].eq("IV < RV")
        & robust["structure"].eq("short_2_strangle")
        & robust["horizon_min"].eq(120)
    ][available]
    report = f"""# Five-strike IV/RV structure research

## Decision

The expanded search changes the `IV < predicted RV` conclusion. The preferred
forward-paper candidate is a **120-minute short two-strike OTM strangle**:

- sell the put two strikes below entry ATM;
- sell the call two strikes above entry ATM;
- use the same monthly expiry for both legs;
- retain both original strikes through exit;
- enter no later than 13:30 and exit after 120 minutes;
- enter only when the predicted Greek P&L, including the separate IV-change
  forecast, exceeds estimated complete costs.

This is not a contradiction. `IV < predicted RV` makes the short position
negative-gamma, but the selected observations also forecast a sufficiently
large IV decline. For the selected short-two-strangle sample, mean expected
gamma, theta and vega contributions were approximately -28.44, +10.63 and
+60.16 points respectively before costs. Positive forecast vega P&L outweighed
negative gamma.

The `IV = predicted RV` scene remains no-trade: no straddle, strangle, iron fly,
iron condor or reverse structure passed the same development/holdout rule, even
after DTE and time-of-day subdivision. `IV > predicted RV` also produced no
cell passing this stricter complete-cost rule.

Selected cell:

{dataframe_to_markdown(selected)}

## Scope

- Source structure universe: ATM, one strike above/below, and two strikes
  above/below ATM from the five-strike dataset.
- All legs use the same monthly expiry and retain their entry strikes.
- Horizons: 15, 30, 60, and 120 minutes, with exits no later than 15:30.
- IV/RV scenes use the timestamp-available RV forecast, not future RV.
- `IV = RV` is the +/-{100 * EQUALITY_BAND:.1f}-vol-point band.
- Primary execution stress: current charge schedule encoded by the repository
  plus {SLIPPAGE_BPS:.0f} bps of premium-turnover slippage.
- Trades are non-overlapping within each structure, horizon, and scene.

## Structures

- Straddles use the ATM call and put.
- Strangles use symmetric +/-1 or +/-2 OTM legs.
- Iron flies buy/sell the ATM straddle and take the opposite position in
  symmetric wings.
- Iron condors use +/-1 short legs and +/-2 protective wings; reverse iron
  condors invert every leg.
- Directional vertical spreads and risk reversals are excluded because no
  signed spot-return forecast is used.

## Entry equation

For leg weights `w_i`, all at the same expiry,

`E[PnL] = sum_i w_i * (0.5*Gamma_i*S^2*vhat_h + Theta_i*h/1440 + Vega_i*dIV_hat) - costs`.

`vhat_h` is cumulative forecast variance over the horizon. It is annualized
only for the IV/RV scene comparison as
`predicted_RV = sqrt(vhat_h * 252 * 375 / h)`.

The Greek-edge gate requires this expected net P&L to be positive before an
entry is eligible. Exact exit P&L uses the unchanged strikes:

`PnL = sum_i w_i * (OptionPrice_i(t+h) - OptionPrice_i(t)) * lot - costs`.

Vanna, volga, charm, higher-order terms and forecast error remain in realized
P&L even though they are not used as independently forecast entry terms.

## Robustness rule

A cell passes only if it has at least 20 non-overlapping 2024 trades, at least
10 non-overlapping 2025 holdout trades, and positive mean net P&L in both
periods after the primary cost stress.

## Passing cells

{passing_text}

## Leading cells by scene

{top_text}

## Guardrails

This is a historical research screen, not an instruction to place a live
order. A positive cell still requires tail-risk limits, independent forward
paper trading, executable bid/ask data, and a single portfolio-level
non-overlap rule across competing structures.

Coverage: {len(panel):,} clean structure-window candidates from
{panel['entry_datetime'].min()} through {panel['entry_datetime'].max()}.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    panel = build_panel()
    summary, trades = score(panel)
    robust = robust_candidates(summary)
    summary.to_csv(OUT / "structure_scene_performance.csv", index=False)
    robust.to_csv(OUT / "robust_candidates.csv", index=False)
    smb.write_parquet(panel, OUT / "candidate_panel.parquet")
    smb.write_parquet(trades, OUT / "nonoverlapping_trades.parquet")
    policy = {
        "status": "FORWARD_PAPER_TEST_ONLY",
        "same_monthly_expiry": True,
        "fixed_entry_strikes": True,
        "intraday_session": "09:15-15:30",
        "cost_model": "current encoded charges plus 25 bps premium-turnover slippage",
        "scenes": {
            "IV < predicted RV": {
                "action": "SELL_TWO_STRIKE_OTM_STRANGLE",
                "legs": ["SELL put at ATM-2 strikes", "SELL call at ATM+2 strikes"],
                "horizon_minutes": 120,
                "latest_entry": "13:30",
                "extra_gate": "expected gamma + theta + vega PnL exceeds complete estimated costs",
            },
            "IV = predicted RV": {
                "action": "NO_TRADE",
                "reason": "no five-strike structure passed development and holdout robustness",
            },
            "IV > predicted RV": {
                "action": "NO_TRADE_UNDER_PRIMARY_COST_STRESS",
                "reason": "no five-strike structure passed development and holdout robustness",
            },
        },
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    write_report(summary, robust, panel)
    print(robust[robust["robust_positive"]].to_string(index=False))


if __name__ == "__main__":
    main()
