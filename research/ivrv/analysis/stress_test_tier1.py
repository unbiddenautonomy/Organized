from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "analysis_outputs" / "test_period_final_routing_above75_event0"
OUTPUT = ROOT / "analysis_outputs" / "stress_test_2026-10-27"
TRADES_PATH = SOURCE / "executed_trades.csv"
DAILY_PATH = SOURCE / "daily_pnl.csv"
SEED = 20261027
N_SIMULATIONS = 10_000
ANNUALIZATION = 252


def sharpe(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    return float(values.mean() / sd * math.sqrt(ANNUALIZATION)) if sd > 0 else math.nan


def max_drawdown(values: np.ndarray) -> float:
    equity = np.concatenate(([0.0], np.cumsum(np.asarray(values, dtype=float))))
    return float(np.max(np.maximum.accumulate(equity) - equity))


def longest_losing_streak(values: np.ndarray) -> int:
    best = current = 0
    for value in values:
        if value < 0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def daily_from_trades(trades: pd.DataFrame, pnl: np.ndarray, dates: pd.DatetimeIndex) -> np.ndarray:
    series = pd.Series(pnl, index=trades["trade_date"]).groupby(level=0).sum()
    return series.reindex(dates, fill_value=0.0).to_numpy(dtype=float)


def block_bootstrap(values: np.ndarray, block: int, rng: np.random.Generator) -> dict:
    n = len(values)
    blocks_needed = math.ceil(n / block)
    sharpes = np.empty(N_SIMULATIONS)
    drawdowns = np.empty(N_SIMULATIONS)
    totals = np.empty(N_SIMULATIONS)
    for simulation in range(N_SIMULATIONS):
        starts = rng.integers(0, n - block + 1, blocks_needed)
        path = np.concatenate([values[start : start + block] for start in starts])[:n]
        sharpes[simulation] = sharpe(path)
        drawdowns[simulation] = max_drawdown(path)
        totals[simulation] = path.sum()
    return {
        "block_days": block,
        "sharpe_p05_p50_p95": np.percentile(sharpes, [5, 50, 95]).tolist(),
        "max_drawdown_p05_p50_p95_p99_inr": np.percentile(drawdowns, [5, 50, 95, 99]).tolist(),
        "net_pnl_p05_p50_p95_inr": np.percentile(totals, [5, 50, 95]).tolist(),
        "probability_net_pnl_below_zero": float(np.mean(totals < 0)),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    trades = pd.read_csv(TRADES_PATH, parse_dates=["signal_datetime", "exit_datetime"])
    daily = pd.read_csv(DAILY_PATH, parse_dates=["date"])
    trades["trade_date"] = trades["signal_datetime"].dt.normalize()
    eligible_dates = pd.DatetimeIndex(daily["date"])
    gross = trades["gross_pnl_inr"].to_numpy(dtype=float)
    cost = trades["explicit_costs_inr"].to_numpy(dtype=float)
    baseline_trade = gross - cost
    baseline_daily = daily["net_pnl_inr"].to_numpy(dtype=float)
    baseline = {
        "trades": int(len(trades)),
        "days": int(len(daily)),
        "net_pnl_inr": float(baseline_trade.sum()),
        "sharpe": sharpe(baseline_daily),
        "max_drawdown_inr": max_drawdown(baseline_trade),
        "longest_losing_streak_trades": longest_losing_streak(baseline_trade),
    }

    cost_rows = []
    for multiplier in [1.0, 1.5, 2.0, 3.0]:
        for slippage in [0.0, 25.0, 50.0, 100.0, 150.0]:
            stressed = gross - multiplier * cost - slippage
            stressed_daily = daily_from_trades(trades, stressed, eligible_dates)
            cost_rows.append({
                "cost_multiplier": multiplier,
                "slippage_per_trade_inr": slippage,
                "net_pnl_inr": float(stressed.sum()),
                "sharpe": sharpe(stressed_daily),
                "win_rate_pct": float(np.mean(stressed > 0) * 100),
            })
    cost_sweep = pd.DataFrame(cost_rows)
    cost_sweep.to_csv(OUTPUT / "tier1_cost_slippage_sweep.csv", index=False)

    thresholds = []
    for multiplier in [1.0, 1.5, 2.0, 3.0]:
        breakeven = float((gross.sum() - multiplier * cost.sum()) / len(trades))
        zero_slip_sharpe = sharpe(daily_from_trades(trades, gross - multiplier * cost, eligible_dates))
        if zero_slip_sharpe < 3:
            below_three = 0.0
        else:
            low, high = 0.0, max(breakeven + 500.0, 500.0)
            for _ in range(60):
                middle = (low + high) / 2
                value = sharpe(daily_from_trades(trades, gross - multiplier * cost - middle, eligible_dates))
                if value >= 3:
                    low = middle
                else:
                    high = middle
            below_three = high
        thresholds.append({
            "cost_multiplier": multiplier,
            "breakeven_slippage_per_trade_inr": breakeven,
            "slippage_where_sharpe_falls_below_3_inr": below_three,
        })
    threshold_frame = pd.DataFrame(thresholds)
    threshold_frame.to_csv(OUTPUT / "tier1_cost_thresholds.csv", index=False)

    bootstrap = []
    for block in [5, 7, 10]:
        bootstrap.append(block_bootstrap(baseline_daily, block, np.random.default_rng(SEED + block)))

    remove_rows = []
    for number in [1, 3, 5]:
        keep = np.ones(len(trades), dtype=bool)
        keep[np.argsort(baseline_trade)[-number:]] = False
        stressed = baseline_trade[keep]
        stressed_trades = trades.loc[keep].copy()
        stressed_daily = daily_from_trades(stressed_trades, stressed, eligible_dates)
        remove_rows.append({
            "test": f"remove_top_{number}_trades",
            "removed": number,
            "net_pnl_inr": float(stressed.sum()),
            "sharpe": sharpe(stressed_daily),
        })
    top_day_count = int(math.ceil(len(daily) * 0.05))
    keep_days = np.ones(len(daily), dtype=bool)
    keep_days[np.argsort(baseline_daily)[-top_day_count:]] = False
    reduced_daily = baseline_daily.copy()
    reduced_daily[~keep_days] = 0.0
    remove_rows.append({
        "test": "remove_top_5pct_eligible_days",
        "removed": top_day_count,
        "net_pnl_inr": float(reduced_daily.sum()),
        "sharpe": sharpe(reduced_daily),
    })
    remove_best = pd.DataFrame(remove_rows)
    remove_best.to_csv(OUTPUT / "tier1_remove_best.csv", index=False)

    rng = np.random.default_rng(SEED + 100)
    shuffled_drawdowns = np.empty(N_SIMULATIONS)
    shuffled_streaks = np.empty(N_SIMULATIONS, dtype=int)
    for simulation in range(N_SIMULATIONS):
        path = rng.permutation(baseline_trade)
        shuffled_drawdowns[simulation] = max_drawdown(path)
        shuffled_streaks[simulation] = longest_losing_streak(path)
    realized_dd = max_drawdown(baseline_trade)
    realized_streak = longest_losing_streak(baseline_trade)
    trade_order = {
        "realized_max_drawdown_inr": realized_dd,
        "max_drawdown_p05_p50_p95_p99_inr": np.percentile(shuffled_drawdowns, [5, 50, 95, 99]).tolist(),
        "realized_drawdown_percentile_lower_is_luckier": float(np.mean(shuffled_drawdowns <= realized_dd) * 100),
        "realized_longest_losing_streak": realized_streak,
        "losing_streak_p05_p50_p95_p99": np.percentile(shuffled_streaks, [5, 50, 95, 99]).tolist(),
        "realized_streak_percentile": float(np.mean(shuffled_streaks <= realized_streak) * 100),
    }

    fill_rows = []
    rng = np.random.default_rng(SEED + 200)
    winner_weights = np.where(baseline_trade > 0, 5.0, 1.0)
    winner_weights /= winner_weights.sum()
    for fraction in [0.10, 0.20, 0.30]:
        drop_count = int(round(len(trades) * fraction))
        for method in ["random", "winner_weighted_5_to_1"]:
            totals = np.empty(N_SIMULATIONS)
            for simulation in range(N_SIMULATIONS):
                probabilities = None if method == "random" else winner_weights
                dropped = rng.choice(len(trades), size=drop_count, replace=False, p=probabilities)
                totals[simulation] = baseline_trade.sum() - baseline_trade[dropped].sum()
            fill_rows.append({
                "drop_fraction": fraction,
                "method": method,
                "dropped_trades": drop_count,
                "net_pnl_p05_inr": float(np.percentile(totals, 5)),
                "net_pnl_p50_inr": float(np.percentile(totals, 50)),
                "net_pnl_p95_inr": float(np.percentile(totals, 95)),
                "probability_net_below_zero": float(np.mean(totals < 0)),
            })
    fill_miss = pd.DataFrame(fill_rows)
    fill_miss.to_csv(OUTPUT / "tier1_fill_miss.csv", index=False)

    gate_row = cost_sweep.loc[
        cost_sweep["cost_multiplier"].eq(2.0)
        & cost_sweep["slippage_per_trade_inr"].eq(50.0)
    ].iloc[0]
    conservative_bootstrap_p05 = min(item["sharpe_p05_p50_p95"][0] for item in bootstrap)
    top_five_net = float(remove_best.loc[remove_best["test"].eq("remove_top_5_trades"), "net_pnl_inr"].iloc[0])
    review_rows = [
        {
            "test": "Baseline",
            "baseline": "Current routed ledger",
            "stressed_value": f"Net ₹{baseline['net_pnl_inr']:.2f}; Sharpe {baseline['sharpe']:.3f}",
            "criterion": "Reference",
            "result": "REFERENCE",
        },
        {
            "test": "2x costs + ₹50 slippage",
            "baseline": f"Net ₹{baseline['net_pnl_inr']:.2f}; Sharpe {baseline['sharpe']:.3f}",
            "stressed_value": f"Net ₹{gate_row.net_pnl_inr:.2f}; Sharpe {gate_row.sharpe:.3f}",
            "criterion": "Net > 0 and Sharpe > 3",
            "result": "PASS" if gate_row.net_pnl_inr > 0 and gate_row.sharpe > 3 else "FAIL",
        },
        {
            "test": "Block bootstrap",
            "baseline": f"Sharpe {baseline['sharpe']:.3f}",
            "stressed_value": f"Worst block-size 5th percentile Sharpe {conservative_bootstrap_p05:.3f}",
            "criterion": "5th percentile Sharpe > 1.5",
            "result": "PASS" if conservative_bootstrap_p05 > 1.5 else "FAIL",
        },
        {
            "test": "Remove top 5 trades",
            "baseline": f"Net ₹{baseline['net_pnl_inr']:.2f}",
            "stressed_value": f"Net ₹{top_five_net:.2f}",
            "criterion": "Net > 0",
            "result": "PASS" if top_five_net > 0 else "FAIL",
        },
        {
            "test": "Trade-order Monte Carlo",
            "baseline": f"Realized drawdown ₹{realized_dd:.2f}",
            "stressed_value": f"Realized drawdown percentile {trade_order['realized_drawdown_percentile_lower_is_luckier']:.2f}%",
            "criterion": "Not below 1st percentile",
            "result": "PASS" if trade_order["realized_drawdown_percentile_lower_is_luckier"] >= 1.0 else "FAIL",
        },
    ]
    review = pd.DataFrame(review_rows)
    review.to_csv(OUTPUT / "single_page_review.csv", index=False)

    payload = {
        "tier": 1,
        "source_files": [str(TRADES_PATH), str(DAILY_PATH)],
        "random_seed": SEED,
        "simulations_per_test": N_SIMULATIONS,
        "baseline": baseline,
        "cost_thresholds": thresholds,
        "block_bootstrap": bootstrap,
        "remove_best": remove_rows,
        "trade_order_monte_carlo": trade_order,
        "adverse_fill_miss": fill_rows,
        "tier_1_gate_passed": bool(gate_row.net_pnl_inr > 0 and gate_row.sharpe > 3),
    }
    (OUTPUT / "tier1_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
