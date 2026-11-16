"""Increase distinct trading days while preserving positive one-lot economics."""

from __future__ import annotations

import json
from itertools import product
from pathlib import Path

import pandas as pd

import hybrid_profit_optimization as profit
import hybrid_regime_strategy as hybrid
import structure_model_backtest as smb
from atm_netpremium_correlation import dataframe_to_markdown


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "regimeUnderstanding" / "coverage_optimization"
MIN_DEVELOPMENT_DAYS = 50
DTE_CHOICES = (0, 4, 8, 15, 999)  # 999 disables that stream.
CALL_MODES = ("off", "near_expiry_only", "all")


def candidate_frame() -> pd.DataFrame:
    streams = hybrid.signal_streams()
    return pd.concat(
        [streams["low_short"], streams["low_direction"]],
        ignore_index=True, sort=False,
    )


def apply_coverage_rule(
    frame: pd.DataFrame, call_mode: str, put_dte_floor: int, short_dte_floor: int
) -> pd.DataFrame:
    mask = (
        frame["signal_name"].eq(profit.PUT) & frame["days_to_expiry"].ge(put_dte_floor)
    ) | (
        frame["signal_name"].eq(profit.SHORT) & frame["days_to_expiry"].ge(short_dte_floor)
    )
    if call_mode == "near_expiry_only":
        mask |= frame["signal_name"].eq(profit.CALL) & frame["days_to_expiry"].lt(4)
    elif call_mode == "all":
        mask |= frame["signal_name"].eq(profit.CALL)
    elif call_mode != "off":
        raise ValueError(f"Unknown call mode: {call_mode}")
    return hybrid.portfolio_nonoverlap(frame[mask].copy())


def optimize(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    development = frame[frame["date"].dt.year.eq(2024)]
    baseline = profit.apply_rule(
        development, profit.SOURCE_SETS["all_validated_streams"], 0
    )
    baseline_drawdown = hybrid.metrics(baseline)["max_drawdown"]
    rows: list[dict] = []
    for call_mode, put_floor, short_floor in product(CALL_MODES, DTE_CHOICES, DTE_CHOICES):
        trades = apply_coverage_rule(development, call_mode, put_floor, short_floor)
        row = {
            "call_mode": call_mode,
            "put_dte_floor": put_floor,
            "short_dte_floor": short_floor,
            **hybrid.metrics(trades),
        }
        row["eligible"] = (
            row["days"] >= MIN_DEVELOPMENT_DAYS
            and row["net_pnl"] > 0
            and row["max_drawdown"] <= baseline_drawdown
        )
        rows.append(row)
    search = pd.DataFrame(rows).sort_values(
        ["eligible", "net_pnl"], ascending=[False, False]
    )
    winner = search[search["eligible"]].iloc[0]
    frozen = {
        "call_mode": winner["call_mode"],
        "put_dte_floor": int(winner["put_dte_floor"]),
        "short_dte_floor": int(winner["short_dte_floor"]),
        "selection_period": "2024",
        "minimum_development_trading_days": MIN_DEVELOPMENT_DAYS,
        "objective": "maximum net P&L among rules meeting the coverage and drawdown constraints",
    }
    return search, frozen


def evaluate(frame: pd.DataFrame, frozen: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    rules = {
        "profit_optimized_baseline": ("off", 4, 4),
        "coverage_optimized_hybrid": (
            frozen["call_mode"], frozen["put_dte_floor"], frozen["short_dte_floor"]
        ),
    }
    rows: list[dict] = []
    ledgers: list[pd.DataFrame] = []
    for rule, settings in rules.items():
        for year, period in ((2024, "development_2024"), (2025, "holdout_2025")):
            sample = frame[frame["date"].dt.year.eq(year)]
            trades = apply_coverage_rule(sample, *settings)
            trades["coverage_rule"] = rule
            ledgers.append(trades)
            rows.append({"rule": rule, "period": period, **hybrid.metrics(trades)})
    return pd.DataFrame(rows), pd.concat(ledgers, ignore_index=True, sort=False)


def write_report(search: pd.DataFrame, evaluation: pd.DataFrame) -> None:
    results = evaluation[[
        "rule", "period", "trades", "days", "net_pnl", "mean_net",
        "hit_rate", "profit_factor", "max_drawdown", "p05_net",
    ]].round(4)
    leaders = search[search["eligible"]][[
        "call_mode", "put_dte_floor", "short_dte_floor", "trades", "days",
        "net_pnl", "hit_rate", "profit_factor", "max_drawdown",
    ]].head(10).round(4)
    report = f"""# Hybrid trading-day coverage optimization

## Frozen higher-coverage rule

Trade only in `IV < predicted RV`, with one monthly-expiry position at a time:

- buy the ATM put when the downside forecast and expected-P&L gate agree, but
  only with at least 4 calendar days to expiry;
- sell the ATM-2 put and ATM+2 call when the short-strangle expected-P&L gate
  is positive, including during 0-3 DTE;
- buy the ATM call only during 0-3 DTE when its directional expected-P&L gate
  is positive;
- if signals coincide, choose the highest expected net rupee edge;
- keep entry strikes fixed, hold 120 minutes, and enter no later than 13:30.

The search used 2024 only. A rule needed at least {MIN_DEVELOPMENT_DAYS} distinct
development trading days, positive net P&L, and drawdown no worse than the
original conservative hybrid. Position size remained one lot.

## Coverage and P&L

{dataframe_to_markdown(results)}

## Leading eligible development rules

{dataframe_to_markdown(leaders)}

## Limitations

The near-expiry call component added development-period days but generated no
eligible 2025 holdout call trades. Its contribution is therefore not validated
yet. The 2025 sample runs only through the available May data. Forcing trades
in IV=RV or IV>RV was not used because it reduced historical robustness.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = candidate_frame()
    search, frozen = optimize(frame)
    evaluation, ledger = evaluate(frame, frozen)
    search.to_csv(OUT / "development_coverage_search.csv", index=False)
    evaluation.to_csv(OUT / "coverage_performance.csv", index=False)
    smb.write_parquet(ledger, OUT / "coverage_trade_ledger.parquet")
    (OUT / "policy.json").write_text(json.dumps({
        "status": "FORWARD_PAPER_TEST_ONLY",
        "rule": frozen,
        "scene": "IV < predicted RV",
        "same_monthly_expiry": True,
        "position_size": "one lot",
        "horizon_minutes": 120,
        "latest_entry": "13:30",
        "iv_eq_or_gt_predicted_rv": "no trade",
    }, indent=2), encoding="utf-8")
    write_report(search, evaluation)
    print(frozen)
    print(evaluation.to_string(index=False))


if __name__ == "__main__":
    main()
