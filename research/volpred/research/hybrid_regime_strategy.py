"""Combine validated IV/RV and directional signals into one intraday portfolio."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "hybrid_strategy"
VOL_TRADES = ROOT / "model" / "regimeUnderstanding" / "five_strike_trade_research" / "nonoverlapping_trades.parquet"
DIRECTION_TRADES = ROOT / "model" / "regimeUnderstanding" / "directional_debit_spreads" / "nonoverlapping_trades.parquet"


def signal_streams() -> dict[str, pd.DataFrame]:
    vol = smb.read_parquet(VOL_TRADES)
    direction = smb.read_parquet(DIRECTION_TRADES)

    low_short = vol[
        vol["scene"].eq("IV < RV")
        & vol["structure"].eq("short_2_strangle")
        & vol["horizon_min"].eq(120)
        & vol["gate"].eq("scene_plus_greek_edge")
    ].copy()
    low_short["signal_name"] = "iv_lt_rv_short_2_strangle"

    low_direction = direction[
        direction["evaluation_scene"].eq("IV < RV")
        & direction["family"].eq("naked_atm")
        & direction["horizon_min"].eq(120)
        & direction["signal_mode"].eq("forecast_sign")
        & direction["gate"].eq("sign_plus_expected_edge")
    ].copy()
    low_direction["signal_name"] = np.where(
        low_direction["structure"].eq("long_atm_call"),
        "iv_lt_rv_long_atm_call", "iv_lt_rv_long_atm_put",
    )

    # This stream is intentionally included only in the forced two-scene
    # comparison. It did not pass the development/holdout robustness rule.
    high_short = vol[
        vol["scene"].eq("IV > RV")
        & vol["structure"].eq("short_2_strangle")
        & vol["horizon_min"].eq(120)
        & vol["gate"].eq("scene_plus_greek_edge")
    ].copy()
    high_short["signal_name"] = "iv_gt_rv_short_2_strangle_unvalidated"

    return {
        "low_short": low_short,
        "low_direction": low_direction,
        "high_short": high_short,
    }


def portfolio_nonoverlap(candidates: pd.DataFrame) -> pd.DataFrame:
    """Choose the largest expected edge at a timestamp and allow one open trade."""
    if candidates.empty:
        return candidates.copy()
    ranked = candidates.sort_values(
        ["entry_datetime", "expected_net_rupees"], ascending=[True, False]
    )
    chosen: list[int] = []
    next_free = pd.Timestamp.min
    for idx, row in ranked.iterrows():
        if row["entry_datetime"] >= next_free:
            chosen.append(idx)
            next_free = row["exit_datetime"]
    return ranked.loc[chosen].sort_values("entry_datetime").copy()


def metrics(frame: pd.DataFrame) -> dict:
    result = smb.metrics(frame)
    pnl = frame["net_pnl_rupees"] if not frame.empty else pd.Series(dtype=float)
    result["median_net"] = pnl.median() if not frame.empty else np.nan
    result["p05_net"] = pnl.quantile(0.05) if not frame.empty else np.nan
    result["best_trade"] = pnl.max() if not frame.empty else np.nan
    result["worst_trade"] = pnl.min() if not frame.empty else np.nan
    return result


def backtest() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    streams = signal_streams()
    definitions = {
        "conservative_hybrid": [streams["low_short"], streams["low_direction"]],
        "forced_two_scene_hybrid": [
            streams["low_short"], streams["low_direction"], streams["high_short"]
        ],
        "iv_lt_rv_short_only": [streams["low_short"]],
        "iv_lt_rv_direction_only": [streams["low_direction"]],
        "iv_gt_rv_short_only_unvalidated": [streams["high_short"]],
    }
    ledgers: list[pd.DataFrame] = []
    rows: list[dict] = []
    mix_rows: list[dict] = []
    for strategy, parts in definitions.items():
        candidates = pd.concat(parts, ignore_index=True, sort=False)
        selected = portfolio_nonoverlap(candidates)
        selected["hybrid_strategy"] = strategy
        ledgers.append(selected)
        for year, period in ((2024, "development_2024"), (2025, "holdout_2025")):
            sample = selected[selected["date"].dt.year.eq(year)]
            rows.append({"strategy": strategy, "period": period, **metrics(sample)})
            for signal, group in sample.groupby("signal_name", observed=True):
                mix_rows.append({
                    "strategy": strategy, "period": period, "signal_name": signal,
                    "trades": len(group), "net_pnl": group["net_pnl_rupees"].sum(),
                    "mean_net": group["net_pnl_rupees"].mean(),
                    "hit_rate": group["net_pnl_rupees"].gt(0).mean(),
                })
    return pd.DataFrame(rows), pd.DataFrame(mix_rows), pd.concat(ledgers, ignore_index=True, sort=False)


def write_report(summary: pd.DataFrame, mix: pd.DataFrame) -> None:
    show = summary[[
        "strategy", "period", "trades", "net_pnl", "mean_net", "median_net",
        "hit_rate", "profit_factor", "p05_net", "max_drawdown",
    ]].round(4)
    conservative_mix = mix[mix["strategy"].eq("conservative_hybrid")].round(4)
    report = f"""# Hybrid IV/RV strategy backtest

## Simple trading rules

### IV below predicted RV

1. If the 120-minute signed-return forecast is positive and its complete
   expected Greek P&L is positive after costs, buy the ATM call.
2. If that return forecast is negative, buy the ATM put instead.
3. A 120-minute short two-strike OTM strangle is also eligible when its
   gamma + theta + vega forecast is positive after costs.
4. If both signals occur together, take only the one with the larger expected
   net rupee edge. Keep the entry strikes unchanged and exit after 120 minutes.

### IV above predicted RV

The intuitive trade is to sell the call two strikes above ATM and sell the put
two strikes below ATM for 120 minutes. The backtest does **not** validate this
rule: it lost in 2024. It is included in `forced_two_scene_hybrid` only to show
the effect of forcing a trade in this scene.

### IV approximately equal to predicted RV

Do not trade. No tested structure passed both evaluation periods.

All trades use one monthly expiry, fixed entry strikes, a latest entry of
13:30, one open portfolio position, and an exit no later than 15:30. Results
include the current encoded charge schedule and 25 bps premium-turnover
slippage.

## Results

{dataframe_to_markdown(show)}

`conservative_hybrid` uses only the two IV<RV streams.
`forced_two_scene_hybrid` adds the unvalidated IV>RV short strangle.

## Conservative-hybrid trade mix

{dataframe_to_markdown(conservative_mix)}

## Interpretation

The comparison is a portfolio backtest, not a sum of independent strategy
totals. When signals overlap, only the earliest eligible trade is retained and
simultaneous signals compete on expected net edge. The component streams were
previously selected in the same reused 2024-2025 research period, so this is
forward-paper evidence rather than an untouched test.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary, mix, ledger = backtest()
    summary.to_csv(OUT / "hybrid_performance.csv", index=False)
    mix.to_csv(OUT / "hybrid_trade_mix.csv", index=False)
    smb.write_parquet(ledger, OUT / "hybrid_trade_ledger.parquet")
    write_report(summary, mix)
    policy = {
        "status": "FORWARD_PAPER_TEST_ONLY",
        "recommended_variant": "conservative_hybrid",
        "iv_lt_predicted_rv": "choose the larger positive expected edge between a directional ATM long option and the 120-minute short two-strike OTM strangle",
        "iv_eq_predicted_rv": "no trade",
        "iv_gt_predicted_rv": "no validated live trade; short two-strike OTM strangle is research-only",
        "same_monthly_expiry": True,
        "horizon_minutes": 120,
        "latest_entry": "13:30",
        "portfolio_rule": "one open trade; simultaneous candidates ranked by expected net rupees",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
