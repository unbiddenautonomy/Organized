from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd

import backtest_user_csv_current_paper_rules as engine


OUTPUT = Path(__file__).resolve().parent / 'analysis_outputs' / 'confidence_threshold_optimization_cap3'


def precompute_outcomes(predictions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    eligible = predictions.loc[predictions["base_eligible"]].copy()
    eligible = eligible.sort_values("datetime", kind="stable").reset_index(drop=True)
    paths = {}
    for key, group in prices.groupby(["dte_flag", "expiry", "strike"], sort=False):
        ordered = group.sort_values("datetime", kind="stable")
        paths[(key[0], pd.Timestamp(key[1]), float(key[2]))] = (
            ordered["datetime"].to_numpy(dtype="datetime64[ns]"),
            ordered["straddle_price"].to_numpy(dtype=float),
        )

    outcomes = []
    for row in eligible.itertuples():
        key = (row.dte_flag, pd.Timestamp(row.expiry), float(row.entry_strike))
        stamps, premiums = paths[key]
        entry64 = np.datetime64(row.datetime)
        start = int(np.searchsorted(stamps, entry64, side="left"))
        if start >= len(stamps) or stamps[start] != entry64:
            raise RuntimeError(f"Missing entry price for {row.datetime} {key}")
        entry_price = float(premiums[start])
        cutoff = pd.Timestamp(row.datetime).normalize() + pd.Timedelta(54899, unit="s")
        horizon = pd.Timestamp(row.datetime) + pd.Timedelta(minutes=engine.HORIZON_MINUTES)
        deadline = min(cutoff, horizon)
        end = int(np.searchsorted(stamps, np.datetime64(deadline), side="right"))
        last_price = entry_price
        exit_stamp = deadline
        exit_reason = "session_square_off_151430" if cutoff <= horizon else "time_exit_120m"
        pnl = 0.0
        for index in range(start, end):
            stamp = pd.Timestamp(stamps[index])
            last_price = float(premiums[index])
            pnl = (entry_price - last_price) * engine.LOT_SIZE
            elapsed = (stamp - pd.Timestamp(row.datetime)).total_seconds() / 60.0
            if stamp >= cutoff:
                exit_stamp, exit_reason = stamp, "session_square_off_151430"
                break
            if pnl >= engine.TARGET_INR:
                exit_stamp, exit_reason = stamp, "profit_target"
                break
            if pnl <= engine.STOP_INR:
                exit_stamp, exit_reason = stamp, "stop_loss"
                break
            if elapsed >= engine.HORIZON_MINUTES:
                exit_stamp, exit_reason = stamp, "time_exit_120m"
                break
        else:
            pnl = (entry_price - last_price) * engine.LOT_SIZE
        outcomes.append({
            "signal_datetime": pd.Timestamp(row.datetime),
            "exit_datetime": pd.Timestamp(exit_stamp),
            "probability": float(row.pred_convergence_probability_classifier),
            "pnl": float(pnl),
            "exit_reason": exit_reason,
        })
    return pd.DataFrame(outcomes)


def score_threshold(outcomes: pd.DataFrame, threshold: float) -> dict:
    candidates = outcomes.loc[outcomes["probability"] > threshold]
    open_exits: list[int] = []
    accepted = []
    for row in candidates.itertuples():
        signal_ns = int(pd.Timestamp(row.signal_datetime).value)
        while open_exits and open_exits[0] <= signal_ns:
            heapq.heappop(open_exits)
        if len(open_exits) >= 3:
            continue
        heapq.heappush(open_exits, int(pd.Timestamp(row.exit_datetime).value))
        accepted.append(row)
    if not accepted:
        return {
            "confidence_threshold_strict_gt": threshold, "qualified_signals": len(candidates),
            "trades": 0, "gross_pnl_inr": 0.0, "win_rate_pct": None,
            "profit_factor": None, "maximum_drawdown_inr": 0.0,
            "active_trade_days": 0, "signals_skipped_at_position_cap": len(candidates),
        }
    frame = pd.DataFrame([row._asdict() for row in accepted])
    pnl = frame["pnl"].astype(float)
    exit_equity = frame.groupby("exit_datetime", sort=True)["pnl"].sum().cumsum()
    drawdown = exit_equity - exit_equity.cummax().clip(lower=0)
    gains = float(pnl[pnl > 0].sum())
    losses = float(-pnl[pnl < 0].sum())
    return {
        "confidence_threshold_strict_gt": float(threshold),
        "qualified_signals": int(len(candidates)),
        "trades": int(len(frame)),
        "gross_pnl_inr": float(pnl.sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100),
        "profit_factor": gains / losses if losses else None,
        "maximum_drawdown_inr": float(drawdown.min()),
        "active_trade_days": int(pd.to_datetime(frame["signal_datetime"]).dt.date.nunique()),
        "signals_skipped_at_position_cap": int(len(candidates) - len(frame)),
    }


def main() -> None:
    engine.TARGET_INR = 1500.0
    engine.STOP_INR = -750.0
    predictions = engine.load_established_test_predictions()
    predictions["Date"] = predictions["datetime"].dt.strftime("%Y-%m-%d")
    prices = engine.load_relevant_prices(predictions)
    outcomes = precompute_outcomes(predictions, prices)

    thresholds = sorted(set([0.0, *outcomes["probability"].astype(float).tolist()]))
    results = pd.DataFrame([score_threshold(outcomes, value) for value in thresholds])
    results = results.sort_values(
        ["gross_pnl_inr", "confidence_threshold_strict_gt"], ascending=[False, False]
    ).reset_index(drop=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(OUTPUT / "candidate_outcomes.csv.gz", index=False, compression="gzip")
    results.to_csv(OUTPUT / "all_exact_threshold_results.csv", index=False)
    results.head(20).to_csv(OUTPUT / "top_20_thresholds.csv", index=False)
    payload = {
        "objective": "maximize historical gross P&L with maximum three concurrent positions",
        "evaluation_status": "threshold optimized on a reused held-out diagnostic; requires untouched validation",
        "evaluation_window": {"start": engine.START_DATE, "end": engine.END_DATE},
        "eligible_trading_days": int(predictions.loc[predictions["base_eligible"], "datetime"].dt.date.nunique()),
        "risk": {
            "profit_target_inr": engine.TARGET_INR,
            "stop_loss_inr": engine.STOP_INR,
            "maximum_open_positions": 3,
            "lot_size": engine.LOT_SIZE,
            "maximum_holding_minutes": engine.HORIZON_MINUTES,
        },
        "exact_thresholds_evaluated": int(len(results)),
        "best": results.iloc[0].to_dict(),
        "top_10": results.head(10).to_dict(orient="records"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
