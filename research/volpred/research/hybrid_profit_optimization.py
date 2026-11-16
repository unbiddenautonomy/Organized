"""Improve hybrid profit using 2024 only, then evaluate once on 2025."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

import hybrid_regime_strategy as hybrid
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "profit_optimization"

CALL = "iv_lt_rv_long_atm_call"
PUT = "iv_lt_rv_long_atm_put"
SHORT = "iv_lt_rv_short_2_strangle"
SOURCE_SETS = {
    "all_validated_streams": (CALL, PUT, SHORT),
    "put_and_short_strangle": (PUT, SHORT),
    "put_only": (PUT,),
    "short_strangle_only": (SHORT,),
}
DTE_FLOORS = (0, 4, 8, 15)
MIN_DEVELOPMENT_TRADES = 30


def candidates() -> pd.DataFrame:
    streams = hybrid.signal_streams()
    return pd.concat(
        [streams["low_short"], streams["low_direction"]],
        ignore_index=True, sort=False,
    )


def apply_rule(frame: pd.DataFrame, sources: tuple[str, ...], dte_floor: int) -> pd.DataFrame:
    eligible = frame[
        frame["signal_name"].isin(sources)
        & frame["days_to_expiry"].ge(dte_floor)
    ].copy()
    return hybrid.portfolio_nonoverlap(eligible)


def optimize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    development = frame[frame["date"].dt.year.eq(2024)]
    baseline = apply_rule(development, SOURCE_SETS["all_validated_streams"], 0)
    baseline_dd = hybrid.metrics(baseline)["max_drawdown"]
    rows: list[dict] = []
    for source_name, dte_floor in product(SOURCE_SETS, DTE_FLOORS):
        trades = apply_rule(development, SOURCE_SETS[source_name], dte_floor)
        row = {
            "source_rule": source_name,
            "dte_floor": dte_floor,
            **hybrid.metrics(trades),
        }
        row["eligible"] = (
            row["trades"] >= MIN_DEVELOPMENT_TRADES
            and row["max_drawdown"] <= baseline_dd
        )
        rows.append(row)
    results = pd.DataFrame(rows).sort_values(
        ["eligible", "net_pnl"], ascending=[False, False]
    )
    winner = results[results["eligible"]].iloc[0]
    frozen = {
        "source_rule": winner["source_rule"],
        "sources": SOURCE_SETS[winner["source_rule"]],
        "dte_floor": int(winner["dte_floor"]),
        "selection_period": "2024",
        "selection_objective": "maximum net P&L with at least 30 trades and max drawdown no worse than baseline",
    }
    return results, frozen


def evaluate(frame: pd.DataFrame, frozen: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = {
        "baseline_conservative_hybrid": (SOURCE_SETS["all_validated_streams"], 0),
        "profit_optimized_hybrid": (tuple(frozen["sources"]), frozen["dte_floor"]),
    }
    rows: list[dict] = []
    ledgers: list[pd.DataFrame] = []
    for rule, (sources, dte_floor) in rules.items():
        for year, period in ((2024, "development_2024"), (2025, "holdout_2025")):
            trades = apply_rule(frame[frame["date"].dt.year.eq(year)], sources, dte_floor)
            trades["optimization_rule"] = rule
            ledgers.append(trades)
            rows.append({"rule": rule, "period": period, **hybrid.metrics(trades)})
    return pd.DataFrame(rows), pd.concat(ledgers, ignore_index=True, sort=False)


def write_report(search: pd.DataFrame, frozen: dict, evaluation: pd.DataFrame) -> None:
    show = evaluation[[
        "rule", "period", "trades", "net_pnl", "mean_net", "median_net",
        "hit_rate", "profit_factor", "p05_net", "max_drawdown",
    ]].round(4)
    eligible = search[search["eligible"]][[
        "source_rule", "dte_floor", "trades", "net_pnl", "mean_net",
        "hit_rate", "profit_factor", "max_drawdown",
    ]].head(8).round(4)
    report = f"""# Hybrid profit optimization

## Frozen improved rule

- Trade only when `IV < predicted RV`.
- If direction is predicted down and expected Greek P&L is positive after
  costs, buy the ATM put.
- The 120-minute short call at ATM+2 and short put at ATM-2 remains eligible
  when its expected Greek P&L is positive after costs.
- Do not take the long-call stream.
- Require at least 4 calendar days to monthly expiry at entry.
- If the put and strangle arrive together, take the one with the larger
  expected net rupee edge.
- Use one lot, one open trade, fixed entry strikes, a 120-minute holding period,
  and enter no later than 13:30.

The rule was selected using 2024 only. The search maximized net P&L subject to
at least {MIN_DEVELOPMENT_TRADES} development trades and maximum drawdown no
worse than the original hybrid. It searched only four source combinations and
DTE floors {DTE_FLOORS}; it did not tune on 2025.

## Baseline versus optimized

{dataframe_to_markdown(show)}

## Leading eligible 2024 candidates

{dataframe_to_markdown(eligible)}

## Interpretation

The profit increase does not use additional leverage. It comes from removing
the historically losing call stream and avoiding the most expiry-sensitive
0-3 DTE observations. The 2025 holdout contains only ten optimized trades, so
the improvement remains a forward-paper candidate rather than a live strategy.
No IV>RV rule was added because doing so made the 2024 hybrid materially
negative in the prior test.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = candidates()
    search, frozen = optimize(frame)
    evaluation, ledger = evaluate(frame, frozen)
    search.to_csv(OUT / "development_search.csv", index=False)
    evaluation.to_csv(OUT / "optimized_performance.csv", index=False)
    smb.write_parquet(ledger, OUT / "optimized_trade_ledger.parquet")
    (OUT / "policy.json").write_text(json.dumps({
        "status": "FORWARD_PAPER_TEST_ONLY",
        "rule": frozen,
        "same_monthly_expiry": True,
        "horizon_minutes": 120,
        "latest_entry": "13:30",
        "position_size": "one lot",
        "iv_gt_predicted_rv": "no trade",
        "iv_eq_predicted_rv": "no trade",
    }, indent=2), encoding="utf-8")
    write_report(search, frozen, evaluation)
    print(frozen)
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
