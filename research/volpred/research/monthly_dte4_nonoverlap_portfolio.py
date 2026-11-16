from __future__ import annotations

"""Exploratory, strictly non-overlapping portfolio of DTE>=4 shadow signals.

The incumbent models are inputs only.  Policy selection for expiry cycle k uses
realized results from cycles k-6,...,k-1; it never uses cycle k outcomes.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import monthly_dte4_research as research


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs" / "monthly_dte4_research"
OUT = ROOT / "outputs" / "monthly_dte4_nonoverlap"
LOOKBACK_CYCLES = 6
MIN_VALIDATION_TRADES = 5
PRIMARY_BPS = 25.0


POLICIES = pd.DataFrame([
    {"policy": "combined_all", "candidate": "combined_greeks_short", "min_dte": 4, "max_dte": 45},
    {"policy": "combined_4_14", "candidate": "combined_greeks_short", "min_dte": 4, "max_dte": 15},
    {"policy": "combined_15_45", "candidate": "combined_greeks_short", "min_dte": 15, "max_dte": 45},
    {"policy": "combined_both", "candidate": "combined_greeks_both", "min_dte": 4, "max_dte": 45},
    {"policy": "iv_both", "candidate": "iv_change_vega_both", "min_dte": 4, "max_dte": 45},
])


def load_predictions() -> pd.DataFrame:
    con = duckdb.connect()
    frame = con.execute(
        "SELECT * FROM read_parquet(?)", [str(BASE / "oos_predictions.parquet")]
    ).fetchdf()
    con.close()
    return frame


def policy_trade_panel(predictions: pd.DataFrame) -> pd.DataFrame:
    raw = research.build_trades(predictions)
    primary = research.attach_costs(raw, PRIMARY_BPS)
    frames = []
    for row in POLICIES.itertuples(index=False):
        z = primary[
            primary["candidate"].eq(row.candidate)
            & primary["days_to_expiry"].ge(row.min_dte)
            & primary["days_to_expiry"].lt(row.max_dte)
        ].copy()
        z["policy"] = row.policy
        frames.append(z)
    return pd.concat(frames, ignore_index=True)


def rolling_scheduler(predictions: pd.DataFrame, policy_trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cycles = list(pd.Series(predictions["outer_test_cycle"].unique()).sort_values())
    chosen_trades, decisions = [], []
    for pos, cycle in enumerate(cycles):
        if pos < LOOKBACK_CYCLES:
            decisions.append({"test_cycle": cycle, "selected_policy": "warmup_no_trade",
                              "validation_net": np.nan, "validation_trades": 0})
            continue
        validation_cycles = cycles[pos - LOOKBACK_CYCLES:pos]
        history = policy_trades[policy_trades["outer_test_cycle"].isin(validation_cycles)]
        scores = history.groupby("policy").agg(
            validation_trades=("net_pnl_rupees", "size"),
            validation_net=("net_pnl_rupees", "sum"),
            validation_mean=("net_pnl_rupees", "mean"),
        )
        eligible = scores[
            scores["validation_trades"].ge(MIN_VALIDATION_TRADES)
            & scores["validation_net"].gt(0)
        ].sort_values(["validation_net", "validation_mean"], ascending=False)
        if eligible.empty:
            decisions.append({"test_cycle": cycle, "selected_policy": "no_trade",
                              "validation_net": 0.0, "validation_trades": 0})
            continue
        policy = eligible.index[0]
        decision = eligible.iloc[0]
        decisions.append({"test_cycle": cycle, "selected_policy": policy,
                          "validation_net": decision.validation_net,
                          "validation_trades": int(decision.validation_trades)})
        chosen_trades.append(
            policy_trades[
                policy_trades["outer_test_cycle"].eq(cycle)
                & policy_trades["policy"].eq(policy)
            ]
        )
    trades = pd.concat(chosen_trades, ignore_index=True) if chosen_trades else policy_trades.iloc[:0].copy()
    # The daily panel has one entry and one same-day exit. These assertions make
    # the non-overlap and no-overnight contract executable rather than narrative.
    assert not trades["date"].duplicated().any(), "more than one portfolio trade on a date"
    assert trades["datetime"].dt.normalize().eq(trades["exit_datetime"].dt.normalize()).all(), "overnight trade"
    assert ((trades["exit_datetime"].dt.hour * 60 + trades["exit_datetime"].dt.minute) >= 915).all(), "exit before 15:15"
    return trades, pd.DataFrame(decisions)


def evaluate(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    perf_rows, infer_rows = [], []
    base_cols = [c for c in trades.columns if c not in {
        "brokerage", "stt", "exchange", "sebi", "ipft", "stamp", "gst", "slippage",
        "total_cost", "buy_turnover", "sell_turnover", "gross_pnl_rupees", "net_pnl_rupees", "cost_bps",
    }]
    raw = trades[base_cols].copy()
    for bps in (10.0, 25.0, 50.0):
        costed = research.attach_costs(raw, bps)
        perf_rows.append({"portfolio": "rolling_six_cycle_selector", "cost_bps": bps,
                          **research.performance(costed)})
        infer_rows.append({"portfolio": "rolling_six_cycle_selector", "cost_bps": bps,
                           **research.cycle_inference(costed)})
    return pd.DataFrame(perf_rows), pd.DataFrame(infer_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    POLICIES.to_csv(OUT / "policy_registry.csv", index=False)
    predictions = load_predictions()
    policy_trades = policy_trade_panel(predictions)
    trades, decisions = rolling_scheduler(predictions, policy_trades)
    decisions.to_csv(OUT / "cycle_policy_decisions.csv", index=False)
    research.write_parquet(trades, OUT / "portfolio_trades_before_cost_reprice.parquet")
    perf, inference = evaluate(trades)
    perf.to_csv(OUT / "performance.csv", index=False)
    inference.to_csv(OUT / "cycle_inference.csv", index=False)
    primary = research.attach_costs(trades, PRIMARY_BPS)
    primary["profitable"] = primary["net_pnl_rupees"].gt(0)
    primary.to_csv(OUT / "trade_ledger_25bps.csv", index=False)
    yearly = primary.groupby(primary["datetime"].dt.year.rename("year")).agg(
        trades=("net_pnl_rupees", "size"), profitable=("profitable", "sum"),
        total_net=("net_pnl_rupees", "sum"),
    ).reset_index()
    primary_perf = perf[perf["cost_bps"].eq(PRIMARY_BPS)].iloc[0]
    primary_inf = inference[inference["cost_bps"].eq(PRIMARY_BPS)].iloc[0]
    lines = [
        "# Non-overlapping DTE >= 4 portfolio",
        "",
        "The incumbent models were not modified. This is an exploratory scheduler built after the first session-end study.",
        "",
        "## Mathematical policy",
        "",
        f"For expiry cycle k, score every registered policy using only cycles k-{LOOKBACK_CYCLES} through k-1. A policy is eligible when it produced at least {MIN_VALIDATION_TRADES} validation trades and positive aggregate validation net P&L at {PRIMARY_BPS:.0f} bps. Select the eligible policy with maximum validation net P&L; break ties by mean net P&L. If none qualifies, do not trade.",
        "",
        "At most one policy is active in an expiry cycle, at most one position is opened per day, and every position exits on the entry date at or after 15:15.",
        "",
        "## Result",
        "",
        perf.to_string(index=False),
        "",
        inference.to_string(index=False),
        "",
        "## Year stability at 25 bps",
        "",
        yearly.to_string(index=False),
        "",
        "## Decision",
        "",
        ("Promising but not validated: net P&L is positive, but the cycle confidence interval still includes zero."
         if primary_perf.total_net > 0 and primary_inf.ci_low <= 0 else
         "Rejected under the current evidence."),
        "",
        "This result is exploratory rather than an untouched confirmation because the policy family was expanded after reviewing the first study. It must not replace the production model without a frozen forward shadow period.",
    ]
    (OUT / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
