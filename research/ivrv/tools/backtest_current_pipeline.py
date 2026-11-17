from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipelines.ivrv.model import FixedStrikeInferencePipeline


ROOT = Path(__file__).resolve().parents[1] / "v2"  # research/v2: datasets and model research
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
PATH_OUTCOMES = ARTIFACTS / "test_inference_trades_with_paths.csv"
COMBINED_FEATURES = (
    ROOT
    / "research/combined_atm_iv_rv_convergence_v2/artifacts/combined_atm_iv_convergence_v2_features.csv"
)
RV_FEATURES = (
    ROOT
    / "research/fixed_strike_iv_rv_convergence/artifacts/fixed_strike_convergence_features.csv"
)
LEG_FEATURES = ROOT / "research/fixed_strike_iv_120m/artifacts/fixed_strike_iv_120m_dataset.csv"

COMMON_TEST_START = pd.Timestamp("2024-09-19 12:56:59")
COMMON_TEST_END = pd.Timestamp("2025-05-09 13:21:59")
KEYS = ["datetime", "dte_flag", "expiry", "entry_strike"]

PREDICTIONS_OUTPUT = ARTIFACTS / "current_pipeline_test_predictions_with_pnl.csv"
TRADES_OUTPUT = ARTIFACTS / "current_pipeline_classifier_trades.csv"
SUMMARY_OUTPUT = ARTIFACTS / "current_pipeline_pnl_summary.csv"
REPORT_OUTPUT = ARTIFACTS / "CURRENT_PIPELINE_BACKTEST_REPORT.md"

# Project-configured exchange/statutory schedule; brokerage is intentionally zero.
STT_SELL_PREMIUM_RATE = 0.0015
NSE_TRANSACTION_RATE = 0.0003553
SEBI_TURNOVER_RATE = 0.000001
STAMP_BUY_PREMIUM_RATE = 0.00003
GST_RATE = 0.18


def source_columns(features: set[str]) -> set[str]:
    columns = set(features)
    for name in list(features):
        if name.startswith("abs_spot_return_"):
            columns.add(name.removeprefix("abs_"))
            columns.discard(name)
        elif name.startswith("log_rv_trailing_") or name.startswith("log_rq_trailing_"):
            columns.add(name.removeprefix("log_"))
            columns.discard(name)
    return columns


def read_and_align(path: Path, columns: set[str], candidates: pd.DataFrame, label: str) -> pd.DataFrame:
    requested = sorted(columns | set(KEYS))
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(requested) - set(header))
    if missing:
        raise ValueError(f"{label} source is missing columns: {missing[:12]}")
    source = pd.read_csv(path, usecols=requested, parse_dates=["datetime"], low_memory=False)
    if source.duplicated(KEYS).any():
        duplicates = int(source.duplicated(KEYS, keep=False).sum())
        raise ValueError(f"{label} source contains {duplicates} duplicate identity rows")
    aligned = candidates[KEYS + ["_candidate_order"]].merge(
        source, on=KEYS, how="left", sort=False, validate="one_to_one", indicator=True
    )
    unmatched = int(aligned["_merge"].ne("both").sum())
    if unmatched:
        raise ValueError(f"{label} source failed to match {unmatched} test rows")
    return aligned.drop(columns="_merge").sort_values("_candidate_order", kind="stable").reset_index(drop=True)


def sequential_trades(frame: pd.DataFrame, signal: pd.Series) -> pd.DataFrame:
    eligible = frame.loc[signal.astype(bool)].sort_values("datetime", kind="stable")
    selected: list[int] = []
    available_at = pd.Timestamp.min
    for index, row in eligible.iterrows():
        if row["datetime"] >= available_at:
            selected.append(index)
            available_at = row["stop_only_exit_datetime"]
    return eligible.loc[selected].copy().reset_index(drop=True)


def maximum_drawdown(values: pd.Series) -> float:
    equity = values.astype(float).cumsum()
    running_peak = np.maximum.accumulate(np.r_[0.0, equity.to_numpy()])
    drawdown = np.r_[0.0, equity.to_numpy()] - running_peak
    return float(drawdown.min())


def longest_losing_streak(values: pd.Series) -> int:
    longest = current = 0
    for value in values.astype(float):
        current = current + 1 if value < 0 else 0
        longest = max(longest, current)
    return longest


def metrics(values: pd.Series) -> dict[str, float | int]:
    values = values.astype(float)
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    return {
        "trades": int(len(values)),
        "total_pnl": float(values.sum()),
        "mean_pnl": float(values.mean()),
        "median_pnl": float(values.median()),
        "win_rate_pct": float((values > 0).mean() * 100.0),
        "profit_factor": gains / losses if losses else np.inf,
        "worst_trade": float(values.min()),
        "best_trade": float(values.max()),
        "maximum_drawdown": maximum_drawdown(values),
        "longest_losing_streak": longest_losing_streak(values),
    }


def add_costs(trades: pd.DataFrame) -> pd.DataFrame:
    result = trades.copy()
    gross = result["stop_only_pnl_per_lot"].astype(float)
    lots = result["lot_size"].astype(float)
    entry_turnover = result["entry_straddle_price"].astype(float) * lots
    exit_premium = result["entry_straddle_price"].astype(float) - gross / lots
    exit_turnover = exit_premium * lots
    total_turnover = entry_turnover + exit_turnover
    result["configured_brokerage"] = 0.0
    result["configured_stt"] = entry_turnover * STT_SELL_PREMIUM_RATE
    result["configured_nse_transaction_charges"] = total_turnover * NSE_TRANSACTION_RATE
    result["configured_sebi_charges"] = total_turnover * SEBI_TURNOVER_RATE
    result["configured_stamp_duty"] = exit_turnover * STAMP_BUY_PREMIUM_RATE
    result["configured_gst"] = GST_RATE * (
        result["configured_nse_transaction_charges"] + result["configured_sebi_charges"]
    )
    charge_columns = [
        "configured_brokerage",
        "configured_stt",
        "configured_nse_transaction_charges",
        "configured_sebi_charges",
        "configured_stamp_duty",
        "configured_gst",
    ]
    result["configured_explicit_costs"] = result[charge_columns].sum(axis=1)
    result["net_pnl_configured_costs"] = gross - result["configured_explicit_costs"]
    result["net_pnl_configured_costs_slip_0.5pt"] = (
        result["net_pnl_configured_costs"] - 4.0 * 0.5 * lots
    )
    result["net_pnl_configured_costs_slip_1pt"] = (
        result["net_pnl_configured_costs"] - 4.0 * lots
    )
    return result


def money(value: float) -> str:
    return f"₹{value:,.2f}"


def main() -> None:
    pipeline = FixedStrikeInferencePipeline()

    forbidden_prefixes = ("target_", "exit_", "label_")
    forbidden_exact = {"label_available", "training_row_eligible", "matched_exit_timestamp", "exit_datetime"}
    for name, payload in pipeline.artifacts.items():
        bad = [
            feature
            for feature in payload.get("features", [])
            if feature.startswith(forbidden_prefixes) or feature in forbidden_exact
        ]
        if bad:
            raise AssertionError(f"Leakage fields in {name}: {bad}")

    path_frame = pd.read_csv(
        PATH_OUTCOMES,
        parse_dates=["datetime", "exit_datetime", "stop_only_exit_datetime", "bracket_exit_datetime"],
        low_memory=False,
    )
    candidates = path_frame.loc[
        path_frame["datetime"].between(COMMON_TEST_START, COMMON_TEST_END, inclusive="both")
    ].copy()
    candidates = candidates.sort_values("row_id", kind="stable").reset_index(drop=True)
    candidates["_candidate_order"] = np.arange(len(candidates))
    if len(candidates) != 7423:
        raise AssertionError(f"Expected 7,423 common held-out rows, found {len(candidates):,}")

    combined_model_features: set[str] = set()
    leg_model_features: set[str] = set()
    for name, payload in pipeline.artifacts.items():
        if name.startswith("combined_"):
            combined_model_features.update(payload.get("features", []))
        elif name.startswith(("call_", "put_")):
            leg_model_features.update(payload.get("features", []))
    combined_model_features -= {"rv_pred_120m", "entry_gap_to_rv_pred_combined"}
    rv_model_features = set(pipeline.artifacts["rv_xgboost"]["features"])

    combined = read_and_align(
        COMBINED_FEATURES,
        combined_model_features | {"entry_atm_iv_combined"},
        candidates,
        "combined",
    )
    rv = read_and_align(RV_FEATURES, source_columns(rv_model_features), candidates, "RV")
    legs = read_and_align(
        LEG_FEATURES,
        leg_model_features | {"entry_iv_call", "entry_iv_put"},
        candidates,
        "leg",
    )

    predictions = pipeline.predict(combined, legs, rv)
    path_columns = [
        "exit_datetime",
        "label_short_straddle_pnl",
        "quality_trainable",
        "lot_size",
        "horizon_pnl_per_lot",
        "row_id",
        "call_price",
        "put_price",
        "entry_straddle_price",
        "Date",
        "path_observations",
        "stop_only_exit_datetime",
        "stop_only_exit_reason",
        "stop_only_pnl_per_lot",
        "stop_only_holding_minutes",
        "bracket_exit_datetime",
        "bracket_exit_reason",
        "bracket_pnl_per_lot",
        "bracket_holding_minutes",
    ]
    predictions = pd.concat(
        [predictions.reset_index(drop=True), candidates[path_columns].reset_index(drop=True)], axis=1
    )
    predictions.to_csv(PREDICTIONS_OUTPUT, index=False)

    baseline = sequential_trades(predictions, pd.Series(True, index=predictions.index))
    classifier = sequential_trades(predictions, predictions["pred_converged_classifier"])
    classifier = add_costs(classifier)
    classifier.to_csv(TRADES_OUTPUT, index=False)

    scenarios = {
        "all_eligible_gross": baseline["stop_only_pnl_per_lot"],
        "classifier_gross": classifier["stop_only_pnl_per_lot"],
        "classifier_configured_costs_no_brokerage": classifier["net_pnl_configured_costs"],
        "classifier_costs_plus_0.5pt_per_fill": classifier[
            "net_pnl_configured_costs_slip_0.5pt"
        ],
        "classifier_costs_plus_1pt_per_fill": classifier[
            "net_pnl_configured_costs_slip_1pt"
        ],
    }
    summary = pd.DataFrame(
        [{"scenario": name, **metrics(values)} for name, values in scenarios.items()]
    )
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    gross = summary.loc[summary.scenario.eq("classifier_gross")].iloc[0]
    net = summary.loc[
        summary.scenario.eq("classifier_configured_costs_no_brokerage")
    ].iloc[0]
    total_costs = float(classifier["configured_explicit_costs"].sum())
    report = f"""# Current retrained pipeline: leakage-controlled held-out backtest

## Evaluation contract

- Common model test window: {COMMON_TEST_START} through {COMMON_TEST_END}.
- Aligned candidate rows: {len(predictions):,}.
- Signal fixed before scoring: saved classifier probability >= 0.5.
- Trade: short entry-strike call and put, one historical lot.
- Exit: first observed one-minute P&L <= -₹750, otherwise exactly 120 minutes.
- Only one position at a time; the next entry is allowed after the causal prior exit timestamp.
- All forecasts were regenerated from the currently active joblibs. Old prediction columns were not used.

## Primary result

| Metric | Gross | Configured charges, zero brokerage |
|---|---:|---:|
| Trades | {int(gross.trades)} | {int(net.trades)} |
| Total P&L | **{money(gross.total_pnl)}** | **{money(net.total_pnl)}** |
| Mean/trade | {money(gross.mean_pnl)} | {money(net.mean_pnl)} |
| Median/trade | {money(gross.median_pnl)} | {money(net.median_pnl)} |
| Win rate | {gross.win_rate_pct:.2f}% | {net.win_rate_pct:.2f}% |
| Profit factor | {gross.profit_factor:.2f} | {net.profit_factor:.2f} |
| Maximum drawdown | {money(gross.maximum_drawdown)} | {money(net.maximum_drawdown)} |
| Worst trade | {money(gross.worst_trade)} | {money(net.worst_trade)} |
| Best trade | {money(gross.best_trade)} | {money(net.best_trade)} |

Configured exchange/statutory costs total **{money(total_costs)}**. Brokerage is zero.

## Leakage controls and interpretation

- Every saved feature contract was checked for target, exit, label, eligibility, and matched-exit fields.
- Model inputs were loaded only from entry-time feature tables and aligned by timestamp, DTE flag, expiry, and fixed entry strike.
- Realized option P&L and minute paths were attached only after predictions were generated.
- The common start is the latest held-out start among the component models, excluding the previously invalid early rows.
- Training used chronological splits with a 120-minute purge.

There is no detected row-level look-ahead leakage. However, the feature-removal decision was previously evaluated on this same test block. This makes the result a **reused-test diagnostic**, not a new untouched confirmatory result. A clean confirmation requires rebuilding features for option data after 9 May 2025 and freezing the model and trading rule before scoring it.
"""
    REPORT_OUTPUT.write_text(report, encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"\nWrote {PREDICTIONS_OUTPUT}")
    print(f"Wrote {TRADES_OUTPUT}")
    print(f"Wrote {SUMMARY_OUTPUT}")
    print(f"Wrote {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
