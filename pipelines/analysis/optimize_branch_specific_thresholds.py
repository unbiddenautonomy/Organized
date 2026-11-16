from __future__ import annotations

import heapq
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent / 'analysis_outputs'
THREE = ROOT / "three_iv_rv_conditions"
OUT = ROOT / "branch_specific_thresholds"
ABOVE_THRESHOLD = 0.8005355169472045


def select(indices: np.ndarray, signal_ns: np.ndarray, exit_ns: np.ndarray, pnl: np.ndarray) -> tuple[list[int], float]:
    heap: list[int] = []
    accepted: list[int] = []
    total = 0.0
    for index in indices:
        while heap and heap[0] <= signal_ns[index]:
            heapq.heappop(heap)
        if len(heap) >= 3:
            continue
        heapq.heappush(heap, int(exit_ns[index]))
        accepted.append(int(index))
        total += float(pnl[index])
    return accepted, total


def metrics(frame: pd.DataFrame, accepted: list[int], qualified: int) -> dict:
    chosen = frame.iloc[accepted].copy()
    values = chosen["pnl"].astype(float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if len(chosen):
        equity = chosen.groupby("exit_datetime", sort=True)["pnl"].sum().cumsum()
        drawdown = float((equity - equity.cummax().clip(lower=0)).min())
    else:
        drawdown = 0.0
    return {
        "qualified_signals": int(qualified),
        "executed_trades": int(len(chosen)),
        "skipped_at_three_position_cap": int(qualified - len(chosen)),
        "gross_pnl_inr": float(values.sum()),
        "win_rate_pct": float((values > 0).mean() * 100) if len(values) else None,
        "profit_factor": gains / losses if losses else None,
        "maximum_drawdown_inr": drawdown,
        "active_trade_days": int(chosen["signal_datetime"].dt.date.nunique()) if len(chosen) else 0,
    }


def main() -> None:
    predictions = pd.read_csv(THREE / "three_rule_predictions.csv.gz", parse_dates=["datetime"])
    predictions = predictions.loc[predictions["base_eligible"].astype(bool)].copy()
    outcomes = pd.read_csv(
        ROOT / "confidence_threshold_optimization_cap3" / "candidate_outcomes.csv.gz",
        parse_dates=["signal_datetime", "exit_datetime"],
    )
    fields = predictions[[
        "datetime", "pred_convergence_probability_classifier",
        "pred_strategy_rule_iv_above_rv_iv_declines_to_rv",
        "pred_strategy_rule_iv_rv_within_half_point",
        "pred_strategy_rule_iv_below_rv_fast_rv_decline",
    ]]
    frame = outcomes.merge(fields, left_on="signal_datetime", right_on="datetime", validate="one_to_one")
    near = frame["pred_strategy_rule_iv_rv_within_half_point"].astype(bool).to_numpy()
    above = (~near) & frame["pred_strategy_rule_iv_above_rv_iv_declines_to_rv"].astype(bool).to_numpy()
    below = (~near) & frame["pred_strategy_rule_iv_below_rv_fast_rv_decline"].astype(bool).to_numpy()
    probability = frame["pred_convergence_probability_classifier"].astype(float).to_numpy()
    signal_ns = frame["signal_datetime"].astype("int64").to_numpy()
    exit_ns = frame["exit_datetime"].astype("int64").to_numpy()
    pnl = frame["pnl"].astype(float).to_numpy()
    fixed_above = above & (probability > ABOVE_THRESHOLD)

    def branch_scan(branch: np.ndarray) -> tuple[float, dict]:
        thresholds = sorted(set([0.0, *probability[branch].tolist(), 1.0]))
        best = None
        for threshold in thresholds:
            mask = branch & (probability > threshold)
            accepted, _ = select(np.flatnonzero(mask), signal_ns, exit_ns, pnl)
            result = metrics(frame, accepted, int(mask.sum()))
            candidate = (result["gross_pnl_inr"], threshold, result)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        return float(best[1]), best[2]

    near_standalone_threshold, near_standalone = branch_scan(near)
    below_standalone_threshold, below_standalone = branch_scan(below)

    near_thresholds = sorted(set([0.0, *probability[near].tolist(), 1.0]))
    below_thresholds = sorted(set([0.0, *probability[below].tolist(), 1.0]))
    best_joint = None
    joint_rows = []
    for near_threshold in near_thresholds:
        near_mask = near & (probability > near_threshold)
        for below_threshold in below_thresholds:
            mask = fixed_above | near_mask | (below & (probability > below_threshold))
            accepted, total = select(np.flatnonzero(mask), signal_ns, exit_ns, pnl)
            candidate = (total, near_threshold + below_threshold)
            if best_joint is None or candidate > best_joint[0]:
                result = metrics(frame, accepted, int(mask.sum()))
                best_joint = (candidate, float(near_threshold), float(below_threshold), accepted, result)
        joint_rows.append({
            "near_threshold": near_threshold,
            "best_gross_pnl_at_this_near_threshold": best_joint[0][0],
        })

    _, best_near, best_below, accepted, joint_metrics = best_joint
    ledger = frame.iloc[accepted].copy()
    ledger["branch"] = np.select(
        [near[accepted], above[accepted], below[accepted]],
        ["near_equal", "above_rv", "below_rv"], default="none"
    )
    attribution = ledger.groupby("branch")["pnl"].agg(["count", "sum"]).reset_index().to_dict(orient="records")
    OUT.mkdir(parents=True, exist_ok=True)
    ledger.to_csv(OUT / "best_joint_portfolio_trades.csv", index=False)
    pd.DataFrame(joint_rows).to_csv(OUT / "near_threshold_profile.csv", index=False)
    payload = {
        "objective": "maximize historical gross P&L under one shared maximum-three-position cap",
        "evaluation_status": "optimized on reused historical evaluation data; requires forward paper validation",
        "fixed_above_rv_threshold_strict_gt": ABOVE_THRESHOLD,
        "standalone_branch_optima": {
            "near_equal": {"threshold_strict_gt": near_standalone_threshold, **near_standalone},
            "below_rv": {"threshold_strict_gt": below_standalone_threshold, **below_standalone},
        },
        "best_joint_thresholds": {
            "above_rv_strict_gt": ABOVE_THRESHOLD,
            "near_equal_strict_gt": best_near,
            "below_rv_strict_gt": best_below,
        },
        "best_joint_portfolio": joint_metrics,
        "pnl_attribution": attribution,
        "note": "Near-equal has no dedicated probability model and below-RV uses a separate event model; these cutoffs apply the convergence classifier as a ranking score and are empirical, not calibrated branch probabilities.",
    }
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
