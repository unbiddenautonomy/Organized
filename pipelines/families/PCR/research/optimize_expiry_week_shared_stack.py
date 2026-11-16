"""Optimize the shared expiry-week stack using pre-test data only.

This v2 run keeps brokerage at exactly zero. Statutory exchange charges and the
two-point straddle slippage assumption remain included. Model families retain
the leakage-controlled selection procedure from run_expiry_week_shared_stack.
Only calendar 2023 is used to select the extension buffer and week-specific
gate thresholds. January 2024 onward is evaluation-only.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_expiry_week_shared_stack as base


OUTPUT = HERE / "output" / "expiry_week_shared_stack_v2_zero_brokerage"
MODEL_ROOT = OUTPUT / "models"
BUFFERS = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0)
DECAY_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
WIN_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
TAIL_LIMITS = (0.08, 0.10, 0.125, 0.15, 0.20, 0.25)


def materialize_with_buffer(frame: pd.DataFrame, buffer: float) -> pd.DataFrame:
    result = frame.copy()
    result["horizon_minutes"] = np.where(
        result["predicted_pnl_120_points"]
        > result["predicted_pnl_60_points"] + buffer,
        120,
        60,
    )
    result = base.materialize_selected_outcome(result)
    result["brokerage_rupees"] = 0.0
    # Base total_cost_rupees consists only of statutory charges and slippage.
    result["total_cost_rupees"] = (
        result["cost_rupees"] + result["slippage_rupees"] + result["brokerage_rupees"]
    )
    return result


def drawdown(net: pd.Series) -> float:
    equity = net.cumsum()
    return float((equity - equity.cummax().clip(lower=0.0)).min()) if len(equity) else 0.0


def fast_one_open(signals: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Equivalent to the production overlap rule without pandas row iteration."""
    ordered = signals.sort_index()
    entries = ordered.index.to_numpy(dtype="datetime64[ns]")
    exits = pd.to_datetime(ordered["exit_datetime"]).to_numpy(dtype="datetime64[ns]")
    keep: list[int] = []
    active_until = np.datetime64("1677-09-22", "ns")
    for position in range(len(ordered)):
        if entries[position] >= active_until:
            keep.append(position)
            active_until = exits[position]
    return ordered.iloc[keep], len(ordered) - len(keep)


def optimize_group(rows: pd.DataFrame, group: str) -> tuple[dict[str, float], dict[str, float]]:
    candidates: list[dict[str, float]] = []
    minimum_trades = 8 if group == "0" else 12
    win_floor = 65.0 if group == "0" else 58.0
    for decay_q in DECAY_QUANTILES:
        decay_gate = max(0.0, float(rows["predicted_decay_60_points"].quantile(decay_q)))
        for win_q in WIN_QUANTILES:
            win_gate = float(rows["win_probability"].quantile(win_q))
            for tail_limit in TAIL_LIMITS:
                signals = rows.loc[
                    rows["predicted_decay_60_points"].gt(decay_gate)
                    & rows["win_probability"].ge(win_gate)
                    & rows["large_loss_probability"].le(tail_limit)
                ]
                trades, rejected = fast_one_open(signals)
                if len(trades) < minimum_trades:
                    continue
                wins = float((trades["net_rupees"] > 0).mean())
                net = float(trades["net_rupees"].sum())
                dd = drawdown(trades.groupby(trades.index.normalize())["net_rupees"].sum())
                # Joint objective rewards both profitability and hit rate while
                # penalizing drawdown. Every term is computed on 2023 only.
                score = net - 0.35 * abs(dd) + 75.0 * (100.0 * wins - win_floor)
                candidates.append({
                    "group": group,
                    "decay_quantile": decay_q,
                    "win_quantile": win_q,
                    "minimum_expected_points": decay_gate,
                    "minimum_win_probability": win_gate,
                    "maximum_large_loss_probability": tail_limit,
                    "calibration_trades": len(trades),
                    "calibration_win_rate_pct": 100.0 * wins,
                    "calibration_net_rupees": net,
                    "calibration_max_drawdown_rupees": dd,
                    "overlap_rejections": rejected,
                    "score": score,
                    "meets_win_floor": 100.0 * wins >= win_floor,
                })
    if not candidates:
        raise RuntimeError(f"No viable threshold candidates for group {group}")
    table = pd.DataFrame(candidates)
    eligible = table.loc[table["meets_win_floor"] & table["calibration_net_rupees"].gt(0)]
    if eligible.empty:
        eligible = table
    winner = eligible.sort_values(
        ["score", "calibration_net_rupees", "calibration_win_rate_pct"],
        ascending=False,
    ).iloc[0]
    threshold = {
        "entry_quantile": float(winner["decay_quantile"]),
        "win_quantile": float(winner["win_quantile"]),
        "minimum_expected_points": float(winner["minimum_expected_points"]),
        "minimum_win_probability": float(winner["minimum_win_probability"]),
        "maximum_large_loss_probability": float(winner["maximum_large_loss_probability"]),
    }
    return threshold, winner.to_dict()


def optimize_2023(calibration_predictions: pd.DataFrame) -> tuple[float, dict[str, dict[str, float]], pd.DataFrame]:
    buffer_rows: list[dict[str, object]] = []
    payloads: dict[float, tuple[dict[str, dict[str, float]], list[dict[str, float]]]] = {}
    for buffer in BUFFERS:
        calibration = materialize_with_buffer(calibration_predictions, buffer)
        thresholds: dict[str, dict[str, float]] = {}
        group_rows: list[dict[str, float]] = []
        for group in ("0", "1", "2", "3_4"):
            threshold, selected_row = optimize_group(
                calibration.loc[calibration["calibration_group"].eq(group)], group
            )
            thresholds[group] = threshold
            group_rows.append(selected_row)
        gated = base.apply_gates(calibration, thresholds)
        trades, rejected = fast_one_open(gated.loc[gated["all_gates_pass"]])
        net = float(trades["net_rupees"].sum())
        win_rate = 100.0 * float((trades["net_rupees"] > 0).mean())
        dd = drawdown(trades.groupby(trades.index.normalize())["net_rupees"].sum())
        score = net - 0.35 * abs(dd) + 75.0 * (win_rate - 60.0)
        buffer_rows.append({
            "extension_buffer_points": buffer,
            "trades": len(trades),
            "win_rate_pct": win_rate,
            "net_rupees": net,
            "max_drawdown_rupees": dd,
            "overlap_rejections": rejected,
            "score": score,
        })
        payloads[buffer] = (thresholds, group_rows)
    comparison = pd.DataFrame(buffer_rows).sort_values(
        ["score", "net_rupees", "win_rate_pct"], ascending=False
    )
    winner_buffer = float(comparison.iloc[0]["extension_buffer_points"])
    thresholds, group_rows = payloads[winner_buffer]
    group_table = pd.DataFrame(group_rows)
    group_table["extension_buffer_points"] = winner_buffer
    search = pd.concat(
        [comparison.assign(record_type="buffer_comparison"), group_table.assign(record_type="selected_group_threshold")],
        ignore_index=True,
        sort=False,
    )
    return winner_buffer, thresholds, search


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table, features = base.build_learning_table()

    selected: dict[str, str] = {}
    selection_parts = []
    selected["decay"], part = base.select_regressor(table, features, ["net_points_60"], "straddle_decay")
    selection_parts.append(part)
    selected["holding"], part = base.select_regressor(table, features, ["net_points_60", "net_points_120"], "holding_period")
    selection_parts.append(part)
    selected["win"], part = base.select_classifier(table, features, "win_60", "win_probability")
    selection_parts.append(part)
    selected["tail"], part = base.select_classifier(table, features, "tail_event", "tail_risk")
    selection_parts.append(part)
    selection = pd.concat(selection_parts, ignore_index=True)
    models = base.fit_final_models(table, features, selected)

    calibration = base.add_predictions(
        table.loc[(table.index >= base.THRESHOLD_START) & (table.index < base.TEST_START)],
        features,
        models,
    )
    extension_buffer, thresholds, search = optimize_2023(calibration)

    test = base.add_predictions(table.loc[table.index >= base.TEST_START], features, models)
    test = materialize_with_buffer(test, extension_buffer)
    test = base.apply_gates(test, thresholds)
    gated = test.loc[test["all_gates_pass"]]
    trades, overlap_rejections = base.enforce_one_open(gated)

    observed_dates = pd.DatetimeIndex(sorted(test.index.normalize().unique()))
    daily = (
        trades.groupby(trades.index.normalize())["net_rupees"].sum()
        .reindex(observed_dates, fill_value=0.0).rename_axis("date").reset_index()
    )
    summary = base.aggregate_metrics(trades, daily)
    summary.update({
        "brokerage_rupees": 0.0,
        "overlap_rejections": overlap_rejections,
        "test_start": str(test.index.min()),
        "test_end": str(test.index.max()),
        "extension_buffer_points": extension_buffer,
    })
    trades = trades.copy()
    trades["year"] = trades.index.year
    expiry = base.grouped_metrics(trades, "calibration_group")
    yearly = base.grouped_metrics(trades, "year")
    funnel = pd.DataFrame([
        {"stage": "test candidates", "rows": len(test)},
        {"stage": "decay pass", "rows": int(test["decay_pass"].sum())},
        {"stage": "win pass", "rows": int(test["win_pass"].sum())},
        {"stage": "tail pass", "rows": int(test["tail_pass"].sum())},
        {"stage": "all gates", "rows": len(gated)},
        {"stage": "executed after one-open rule", "rows": len(trades)},
    ])

    selection.to_csv(OUTPUT / "model_family_selection_2021.csv", index=False)
    search.to_csv(OUTPUT / "threshold_optimization_2023.csv", index=False)
    pd.DataFrame([summary]).to_csv(OUTPUT / "summary.csv", index=False)
    trades.reset_index().to_csv(OUTPUT / "trades.csv", index=False)
    daily.to_csv(OUTPUT / "daily_pnl.csv", index=False)
    expiry.to_csv(OUTPUT / "expiry_week_metrics.csv", index=False)
    yearly.to_csv(OUTPUT / "yearly_metrics.csv", index=False)
    funnel.to_csv(OUTPUT / "gate_funnel.csv", index=False)
    test.reset_index().to_csv(OUTPUT / "all_test_candidates.csv.gz", index=False, compression="gzip")
    (OUTPUT / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    (OUTPUT / "run_metadata.json").write_text(json.dumps({
        "architecture": "shared models with 2023-optimized expiry-week gates",
        "brokerage_rupees": 0.0,
        "statutory_charges_included": True,
        "slippage_points_per_straddle": 2.0,
        "selected_models": selected,
        "features": features,
        "horizons": base.HORIZONS,
        "extension_buffer_points": extension_buffer,
        "large_loss_points": base.LARGE_LOSS_POINTS,
        "family_selection_period": "2021",
        "classifier_probability_calibration_period": "2022",
        "threshold_and_buffer_optimization_period": "2023",
        "test_start": "2024-01-01",
        "thresholds": thresholds,
        "optimization_objective": "net PnL - 0.35*abs(drawdown) + 75*(win_rate_pct - target)",
    }, indent=2), encoding="utf-8")

    old_root, old_model_root = base.ROOT, base.MODEL_ROOT
    base.ROOT, base.MODEL_ROOT = OUTPUT, MODEL_ROOT
    try:
        base.write_report(summary, expiry, yearly, funnel, selected)
        base.package_models(models, selected, features, thresholds, selection, summary)
    finally:
        base.ROOT, base.MODEL_ROOT = old_root, old_model_root

    report = (OUTPUT / "report.md").read_text(encoding="utf-8")
    report = report.replace(
        "## Frozen architecture",
        "## Zero-brokerage assumption\n\nBrokerage is fixed at **Rs 0.00**. Statutory charges and two straddle points of slippage remain included.\n\n## Frozen architecture",
    ).replace(
        "Only the entry thresholds vary by expiry bucket; weeks 3 and 4 share a calibration group.",
        f"Only the entry thresholds vary by expiry bucket; weeks 3 and 4 share a calibration group. The {extension_buffer:g}-point holding extension buffer and all thresholds were selected using 2023 only.",
    )
    (OUTPUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
