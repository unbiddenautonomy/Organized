from __future__ import annotations

import heapq
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "analysis_outputs" / "three_iv_rv_conditions" / "three_rule_predictions.csv.gz"
OUTCOMES = ROOT / "analysis_outputs" / "confidence_threshold_optimization_cap3" / "candidate_outcomes.csv.gz"
ENTRY_PRICES = (
    ROOT.parent / "tools"
    / "artifacts"
    / "current_pipeline_test_predictions_with_pnl.csv"
)
OUTPUT = ROOT / "analysis_outputs" / "test_period_final_routing_above75_event0"

ABOVE_PROBABILITY_THRESHOLD = 0.75
NEAR_GAP_DECIMAL = 0.005
EVENT_SCORE_THRESHOLD = 0.0
MAX_OPEN_POSITIONS = 3
LOT_SIZE = 30
ANNUALIZATION_DAYS = 252


def explicit_cost(entry_price: float, exit_price: float) -> float:
    entry_turnover = entry_price * LOT_SIZE
    exit_turnover = exit_price * LOT_SIZE
    turnover = entry_turnover + exit_turnover
    stt = entry_turnover * 0.0015
    nse = turnover * 0.0003553
    sebi = turnover * 0.000001
    stamp = exit_turnover * 0.00003
    gst = 0.18 * (nse + sebi)
    return stt + nse + sebi + stamp + gst


def apply_cap(candidates: pd.DataFrame) -> pd.DataFrame:
    open_exits: list[int] = []
    accepted: list[int] = []
    for index, row in candidates.sort_values("signal_datetime", kind="stable").iterrows():
        signal_ns = int(pd.Timestamp(row["signal_datetime"]).value)
        while open_exits and open_exits[0] <= signal_ns:
            heapq.heappop(open_exits)
        if len(open_exits) >= MAX_OPEN_POSITIONS:
            continue
        heapq.heappush(open_exits, int(pd.Timestamp(row["exit_datetime"]).value))
        accepted.append(index)
    return candidates.loc[accepted].sort_values("signal_datetime", kind="stable").reset_index(drop=True)


def annualized_ratios(daily_pnl: pd.Series) -> tuple[float | None, float | None]:
    values = daily_pnl.astype(float).to_numpy()
    if len(values) < 2:
        return None, None
    sample_std = float(np.std(values, ddof=1))
    sharpe = math.sqrt(ANNUALIZATION_DAYS) * float(np.mean(values)) / sample_std if sample_std else None
    downside_deviation = float(np.sqrt(np.mean(np.minimum(values, 0.0) ** 2)))
    sortino = (
        math.sqrt(ANNUALIZATION_DAYS) * float(np.mean(values)) / downside_deviation
        if downside_deviation
        else None
    )
    return sharpe, sortino


def drawdown_details(ledger: pd.DataFrame, pnl_column: str) -> dict:
    ordered = ledger.sort_values(["exit_datetime", "signal_datetime"], kind="stable").reset_index(drop=True)
    pnl = ordered[pnl_column].astype(float).to_numpy()
    equity = np.concatenate(([0.0], np.cumsum(pnl)))
    running_peak = np.maximum.accumulate(equity)
    drawdown = equity - running_peak
    trough_index = int(np.argmin(drawdown))
    peak_index = int(np.argmax(equity[: trough_index + 1]))
    peak_value = float(equity[peak_index])
    trough_value = float(equity[trough_index])
    recovery_candidates = np.flatnonzero(equity[trough_index + 1 :] >= peak_value)
    recovery_index = (
        int(trough_index + 1 + recovery_candidates[0]) if len(recovery_candidates) else None
    )

    first_date = pd.Timestamp(ordered.iloc[0]["signal_datetime"]).date()
    peak_date = (
        pd.Timestamp(ordered.iloc[peak_index - 1]["exit_datetime"]).date()
        if peak_index > 0
        else first_date
    )
    trough_date = (
        pd.Timestamp(ordered.iloc[trough_index - 1]["exit_datetime"]).date()
        if trough_index > 0
        else first_date
    )
    recovery_date = (
        pd.Timestamp(ordered.iloc[recovery_index - 1]["exit_datetime"]).date()
        if recovery_index is not None and recovery_index > 0
        else None
    )
    trade_dates = pd.to_datetime(ordered["signal_datetime"]).dt.date
    decline_mask = (trade_dates > peak_date) & (trade_dates <= trough_date)
    if peak_index == 0:
        decline_mask = trade_dates <= trough_date
    decline_active_days = int(trade_dates.loc[decline_mask].nunique())

    return {
        "maximum_drawdown_inr": float(drawdown[trough_index]),
        "maximum_drawdown_magnitude_inr": float(-drawdown[trough_index]),
        "peak_equity_inr": peak_value,
        "trough_equity_inr": trough_value,
        "peak_date": str(peak_date),
        "trough_date": str(trough_date),
        "trades_peak_to_trough": int(trough_index - peak_index),
        "active_trade_days_peak_to_trough": decline_active_days,
        "calendar_days_peak_to_trough": int((trough_date - peak_date).days),
        "recovered": recovery_index is not None,
        "recovery_date": str(recovery_date) if recovery_date is not None else None,
        "trades_peak_to_recovery": int(recovery_index - peak_index) if recovery_index is not None else None,
        "calendar_days_peak_to_recovery": (
            int((recovery_date - peak_date).days) if recovery_date is not None else None
        ),
    }


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS, parse_dates=["datetime", "expiry"], low_memory=False)
    predictions = predictions.loc[predictions["base_eligible"].astype(bool)].copy()
    outcomes = pd.read_csv(
        OUTCOMES,
        parse_dates=["signal_datetime", "exit_datetime"],
        low_memory=False,
    )
    fields = predictions[
        [
            "datetime",
            "row_id",
            "pred_no120_gate_iv_minus_rv",
            "pred_no120_event_score",
            "pred_convergence_probability_classifier",
            "pred_strategy_rule_iv_above_rv_iv_declines_to_rv",
        ]
    ]
    frame = outcomes.merge(
        fields,
        left_on="signal_datetime",
        right_on="datetime",
        how="inner",
        validate="one_to_one",
    )

    prices = pd.read_csv(
        ENTRY_PRICES,
        usecols=["datetime", "expiry", "row_id", "entry_straddle_price"],
        parse_dates=["datetime", "expiry"],
        low_memory=False,
    )
    nearest = prices.groupby("datetime")["expiry"].transform("min")
    prices = prices.loc[prices["expiry"].eq(nearest), ["row_id", "entry_straddle_price"]]
    prices = prices.drop_duplicates("row_id", keep="last")
    frame = frame.merge(prices, on="row_id", how="left", validate="one_to_one")
    if frame["entry_straddle_price"].isna().any():
        raise RuntimeError("Missing entry prices for one or more candidate outcomes")

    gap = frame["pred_no120_gate_iv_minus_rv"].astype(float)
    score = frame["pred_no120_event_score"].astype(float)
    probability = frame["pred_convergence_probability_classifier"].astype(float)
    convergence_rule = frame["pred_strategy_rule_iv_above_rv_iv_declines_to_rv"].astype(bool)
    above = (gap > NEAR_GAP_DECIMAL) & convergence_rule & (probability > ABOVE_PROBABILITY_THRESHOLD)
    near = (gap.abs() <= NEAR_GAP_DECIMAL) & (score >= EVENT_SCORE_THRESHOLD)
    below = (gap < -NEAR_GAP_DECIMAL) & (score >= EVENT_SCORE_THRESHOLD)
    qualified = frame.loc[above | near | below].copy()
    qualified["route"] = np.select(
        [above.loc[qualified.index], near.loc[qualified.index], below.loc[qualified.index]],
        ["iv_above_rv_convergence", "iv_near_rv_sharp_rv_decline", "iv_below_rv_sharp_rv_decline"],
        default="unrouted",
    )
    qualified["entry_straddle_price"] = qualified["entry_straddle_price"].astype(float)
    qualified["exit_straddle_price"] = (
        qualified["entry_straddle_price"] - qualified["pnl"].astype(float) / LOT_SIZE
    )
    qualified["explicit_costs_inr"] = [
        explicit_cost(entry, exit_)
        for entry, exit_ in qualified[["entry_straddle_price", "exit_straddle_price"]].itertuples(index=False, name=None)
    ]
    qualified["gross_pnl_inr"] = qualified["pnl"].astype(float)
    qualified["net_pnl_inr"] = qualified["gross_pnl_inr"] - qualified["explicit_costs_inr"]

    ledger = apply_cap(qualified)
    eligible_dates = pd.Index(sorted(pd.to_datetime(predictions["datetime"]).dt.normalize().unique()))
    ledger["trade_date"] = pd.to_datetime(ledger["signal_datetime"]).dt.normalize()
    daily = ledger.groupby("trade_date").agg(
        gross_pnl_inr=("gross_pnl_inr", "sum"),
        explicit_costs_inr=("explicit_costs_inr", "sum"),
        net_pnl_inr=("net_pnl_inr", "sum"),
        trades=("trade_date", "size"),
    ).reindex(eligible_dates, fill_value=0.0)
    daily.index.name = "date"
    gross_sharpe, gross_sortino = annualized_ratios(daily["gross_pnl_inr"])
    net_sharpe, net_sortino = annualized_ratios(daily["net_pnl_inr"])

    net_values = ledger["net_pnl_inr"].astype(float)
    daily_net = daily["net_pnl_inr"].astype(float)
    start_date = eligible_dates.min().date()
    end_date = eligible_dates.max().date()
    exit_reason_counts = Counter(ledger["exit_reason"])
    exit_reason_pnl = ledger.groupby("exit_reason").agg(
        trades=("exit_reason", "size"),
        gross_pnl_inr=("gross_pnl_inr", "sum"),
        net_pnl_inr=("net_pnl_inr", "sum"),
    ).reset_index().to_dict(orient="records")
    ledger["net_result"] = np.select(
        [ledger["net_pnl_inr"] > 0, ledger["net_pnl_inr"] < 0],
        ["green", "red"],
        default="flat",
    )
    exit_reason_result_counts = {
        reason: {result: int(count) for result, count in group["net_result"].value_counts().items()}
        for reason, group in ledger.groupby("exit_reason", sort=True)
    }
    branch_counts = qualified["route"].value_counts().to_dict()
    executed_branch_counts = ledger["route"].value_counts().to_dict()

    summary = {
        "evaluation_window": {
            "start": str(start_date),
            "end": str(end_date),
            "calendar_days_inclusive": int((end_date - start_date).days + 1),
            "eligible_market_days": int(len(eligible_dates)),
            "active_trade_days": int((daily["trades"] > 0).sum()),
        },
        "routing": {
            "iv_above_rv": "gap > 0.005, convergence rule true, probability > 0.75",
            "iv_near_rv": "abs(gap) <= 0.005, RV-decline event score >= 0",
            "iv_below_rv": "gap < -0.005, RV-decline event score >= 0",
        },
        "risk": {
            "profit_target_inr": 1500.0,
            "stop_loss_inr": -750.0,
            "maximum_open_positions": MAX_OPEN_POSITIONS,
            "maximum_holding_minutes": 120,
            "lot_size": LOT_SIZE,
        },
        "signals": {
            "qualified": int(len(qualified)),
            "qualified_by_route": {key: int(value) for key, value in branch_counts.items()},
            "executed": int(len(ledger)),
            "executed_by_route": {key: int(value) for key, value in executed_branch_counts.items()},
            "skipped_at_three_position_cap": int(len(qualified) - len(ledger)),
        },
        "trade_outcomes_after_costs": {
            "green_trades": int((net_values > 0).sum()),
            "red_trades": int((net_values < 0).sum()),
            "flat_trades": int((net_values == 0).sum()),
            "win_rate_pct": float((net_values > 0).mean() * 100),
        },
        "day_outcomes_after_costs_all_eligible_days": {
            "green_days": int((daily_net > 0).sum()),
            "red_days": int((daily_net < 0).sum()),
            "flat_or_no_trade_days": int((daily_net == 0).sum()),
        },
        "pnl": {
            "gross_pnl_inr": float(ledger["gross_pnl_inr"].sum()),
            "explicit_costs_inr": float(ledger["explicit_costs_inr"].sum()),
            "net_pnl_inr": float(ledger["net_pnl_inr"].sum()),
        },
        "risk_adjusted_daily_all_eligible_days": {
            "annualized_sharpe_gross": gross_sharpe,
            "annualized_sortino_gross": gross_sortino,
            "annualized_sharpe_after_costs": net_sharpe,
            "annualized_sortino_after_costs": net_sortino,
            "risk_free_or_target_daily_pnl": 0.0,
            "sharpe_std_convention": "sample standard deviation",
            "sortino_downside_convention": "sqrt(mean(min(daily_pnl, 0)^2))",
            "annualization_days": ANNUALIZATION_DAYS,
        },
        "maximum_drawdown_after_costs": drawdown_details(ledger, "net_pnl_inr"),
        "maximum_drawdown_gross": drawdown_details(ledger, "gross_pnl_inr"),
        "exit_reasons": {
            "counts": {key: int(value) for key, value in exit_reason_counts.items()},
            "green_red_after_costs": exit_reason_result_counts,
            "pnl": exit_reason_pnl,
        },
        "accounting": "Explicit statutory/exchange costs, zero brokerage, no slippage",
        "evaluation_status": "Reused historical evaluation period; thresholds were selected using related historical results, so this is not an untouched holdout.",
    }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUTPUT / "executed_trades.csv", index=False)
    daily.reset_index().to_csv(OUTPUT / "daily_pnl.csv", index=False)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
