from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
PARENT_ARTIFACTS = ROOT.parent / "artifacts"


FEATURE_COLUMNS = [
    "row_id", "days_to_expiry", "entry_spot_price", "entry_future_price", "entry_straddle_price",
    "entry_iv_call", "entry_iv_put", "entry_delta_call", "entry_delta_put", "entry_vega_call",
    "entry_vega_put", "entry_gamma_call", "entry_gamma_put", "entry_theta_call", "entry_theta_put",
    "surface_skew_call", "surface_skew_put", "surface_smile_call", "surface_smile_put",
    "change_5m_entry_spot_price", "change_15m_entry_spot_price", "change_30m_entry_spot_price",
    "change_60m_entry_spot_price", "change_5m_entry_straddle_price", "change_15m_entry_straddle_price",
    "change_30m_entry_straddle_price", "change_60m_entry_straddle_price",
    "change_15m_entry_iv_call", "change_15m_entry_iv_put", "change_30m_entry_iv_call",
    "change_30m_entry_iv_put", "change_60m_entry_iv_call", "change_60m_entry_iv_put",
]


def profit_factor(pnl: pd.Series) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    return float(gains / losses) if losses > 0 else np.nan


def metrics(frame: pd.DataFrame) -> dict:
    pnl = frame["stop_only_pnl_per_lot"]
    return {
        "trades": int(len(frame)), "net_pnl": float(pnl.sum()), "mean_pnl": float(pnl.mean()),
        "win_rate": float((pnl > 0).mean()), "loss_rate": float((pnl < 0).mean()),
        "stop_rate": float(frame["stop_only_exit_reason"].eq("stop_loss").mean()),
        "profit_factor": profit_factor(pnl),
    }


def enrich() -> pd.DataFrame:
    trades = pd.read_csv(ARTIFACTS / "non_overlapping_stop_only_trades.csv", parse_dates=["datetime", "exit_datetime"])
    available = pd.read_csv(PARENT_ARTIFACTS / "model_features.csv", nrows=0).columns
    columns = [column for column in FEATURE_COLUMNS if column in available]
    features = pd.read_csv(PARENT_ARTIFACTS / "model_features.csv", usecols=columns)
    frame = trades.merge(features, on="row_id", how="left", suffixes=("", "_feature"), validate="one_to_one")

    fixed_columns = ["row_id", "exit_spot_price", "exit_call_price", "exit_put_price"]
    fixed = pd.read_csv(PARENT_ARTIFACTS / "fixed_strike_iv_120m_dataset.csv", usecols=fixed_columns)
    frame = frame.merge(fixed, on="row_id", how="left", suffixes=("", "_source"), validate="one_to_one")

    frame["red_trade"] = frame["stop_only_pnl_per_lot"] < 0
    frame["minutes_after_open"] = frame["datetime"].dt.hour * 60 + frame["datetime"].dt.minute - 555
    frame["iv_rv_spread"] = frame["iv_current"] - frame["rv_current"]
    frame["iv_rv_ratio"] = frame["iv_current"] / frame["rv_current"]
    frame["call_put_iv_gap"] = (frame["entry_iv_call"] - frame["entry_iv_put"]).abs()
    frame["entry_premium_per_lot"] = frame["entry_straddle_price"] * frame["lot_size"]
    frame["straddle_pct_spot"] = frame["entry_straddle_price"] / frame["entry_spot_price"]
    frame["short_theta_carry_per_lot"] = -(frame["entry_theta_call"] + frame["entry_theta_put"]) * frame["lot_size"]
    frame["vega_per_lot"] = (frame["entry_vega_call"].abs() + frame["entry_vega_put"].abs()) * frame["lot_size"]
    frame["gamma_loss_for_1pct_move"] = (
        0.5 * (frame["entry_gamma_call"].abs() + frame["entry_gamma_put"].abs())
        * (0.01 * frame["entry_spot_price"]) ** 2 * frame["lot_size"]
    )
    frame["delta_imbalance"] = (frame["entry_delta_call"] + frame["entry_delta_put"]).abs()
    for minutes in (5, 15, 30, 60):
        spot_change = f"change_{minutes}m_entry_spot_price"
        if spot_change in frame:
            frame[f"abs_spot_move_{minutes}m_over_premium"] = frame[spot_change].abs() / frame["entry_straddle_price"]
        straddle_change = f"change_{minutes}m_entry_straddle_price"
        if straddle_change in frame:
            frame[f"straddle_change_{minutes}m_pct"] = frame[straddle_change] / frame["entry_straddle_price"]
    frame["actual_abs_spot_move"] = (frame["exit_spot_price"] - frame["entry_spot_price"]).abs()
    frame["actual_spot_move_over_premium"] = frame["actual_abs_spot_move"] / frame["entry_straddle_price"]
    frame["horizon_call_pnl_per_lot"] = (frame["entry_call_price"] - frame["exit_call_price"]) * frame["lot_size"]
    frame["horizon_put_pnl_per_lot"] = (frame["entry_put_price"] - frame["exit_put_price"]) * frame["lot_size"]
    return frame


def comparison(frame: pd.DataFrame, entry_features: list[str]) -> pd.DataFrame:
    rows = []
    red = frame[frame["red_trade"]]
    green = frame[~frame["red_trade"]]
    for feature in entry_features:
        a, b = red[feature].dropna(), green[feature].dropna()
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0)
        rows.append({
            "feature": feature, "red_count": len(a), "green_count": len(b),
            "red_median": a.median(), "green_median": b.median(),
            "red_mean": a.mean(), "green_mean": b.mean(),
            "standardized_mean_difference_red_minus_green": (a.mean() - b.mean()) / pooled if pooled > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(
        "standardized_mean_difference_red_minus_green", key=lambda x: x.abs(), ascending=False
    )


def binned_results(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        valid = frame.dropna(subset=[feature]).copy()
        try:
            valid["bin"] = pd.qcut(valid[feature], 4, duplicates="drop")
        except ValueError:
            continue
        for interval, group in valid.groupby("bin", observed=True):
            row = metrics(group)
            row.update({"feature": feature, "bin": str(interval), "feature_min": group[feature].min(),
                        "feature_max": group[feature].max()})
            rows.append(row)
    return pd.DataFrame(rows)


def search_filters(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    split = int(len(frame) * 0.70)
    development = frame.iloc[:split].copy()
    holdout = frame.iloc[split:].copy()
    baseline_dev = metrics(development)
    baseline_holdout = metrics(holdout)
    rows = []
    for feature in features:
        series = development[feature].dropna()
        if series.nunique() < 5:
            continue
        for quantile in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            threshold = float(series.quantile(quantile))
            for operator in ("<=", ">="):
                dev_mask = development[feature].le(threshold) if operator == "<=" else development[feature].ge(threshold)
                hold_mask = holdout[feature].le(threshold) if operator == "<=" else holdout[feature].ge(threshold)
                dev_kept, hold_kept = development[dev_mask], holdout[hold_mask]
                if len(dev_kept) < 0.40 * len(development) or len(hold_kept) < 10:
                    continue
                dev_metrics, hold_metrics = metrics(dev_kept), metrics(hold_kept)
                rows.append({
                    "feature": feature, "operator": operator, "threshold": threshold,
                    "development_retained_fraction": len(dev_kept) / len(development),
                    "development_loss_rate": dev_metrics["loss_rate"],
                    "development_loss_rate_change": dev_metrics["loss_rate"] - baseline_dev["loss_rate"],
                    "development_mean_pnl": dev_metrics["mean_pnl"],
                    "development_profit_factor": dev_metrics["profit_factor"],
                    "holdout_trades": len(hold_kept), "holdout_retained_fraction": len(hold_kept) / len(holdout),
                    "holdout_loss_rate": hold_metrics["loss_rate"],
                    "holdout_loss_rate_change": hold_metrics["loss_rate"] - baseline_holdout["loss_rate"],
                    "holdout_mean_pnl": hold_metrics["mean_pnl"],
                    "holdout_profit_factor": hold_metrics["profit_factor"],
                })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["development_loss_rate", "development_mean_pnl"], ascending=[True, False])


def main() -> None:
    frame = enrich().sort_values("datetime").reset_index(drop=True)
    entry_features = [
        "minutes_after_open", "days_to_expiry", "iv_rv_spread", "iv_rv_ratio", "rv_current",
        "iv_current", "call_put_iv_gap", "entry_premium_per_lot", "straddle_pct_spot",
        "short_theta_carry_per_lot", "vega_per_lot", "gamma_loss_for_1pct_move", "delta_imbalance",
        "abs_spot_move_5m_over_premium", "abs_spot_move_15m_over_premium",
        "abs_spot_move_30m_over_premium", "abs_spot_move_60m_over_premium",
        "straddle_change_15m_pct", "straddle_change_30m_pct", "straddle_change_60m_pct",
        "surface_skew_call", "surface_skew_put", "surface_smile_call", "surface_smile_put",
    ]
    entry_features = [feature for feature in entry_features if feature in frame]
    frame.to_csv(ARTIFACTS / "red_trade_enriched.csv", index=False)
    comparison(frame, entry_features).to_csv(ARTIFACTS / "red_vs_green_entry_features.csv", index=False)
    binned_results(frame, entry_features).to_csv(ARTIFACTS / "entry_feature_quartiles.csv", index=False)
    filters = search_filters(frame, entry_features)
    filters.to_csv(ARTIFACTS / "entry_filter_holdout_results.csv", index=False)

    red = frame[frame["red_trade"]]
    green = frame[~frame["red_trade"]]
    outcome = pd.DataFrame([
        {"group": "red", "trades": len(red), "stop_rate": red.stop_only_exit_reason.eq("stop_loss").mean(),
         "median_abs_spot_move": red.actual_abs_spot_move.median(),
         "median_spot_move_over_premium": red.actual_spot_move_over_premium.median(),
         "median_holding_minutes": red.stop_only_holding_minutes.median(),
         "mean_pnl": red.stop_only_pnl_per_lot.mean()},
        {"group": "non_red", "trades": len(green), "stop_rate": green.stop_only_exit_reason.eq("stop_loss").mean(),
         "median_abs_spot_move": green.actual_abs_spot_move.median(),
         "median_spot_move_over_premium": green.actual_spot_move_over_premium.median(),
         "median_holding_minutes": green.stop_only_holding_minutes.median(),
         "mean_pnl": green.stop_only_pnl_per_lot.mean()},
    ])
    outcome.to_csv(ARTIFACTS / "red_trade_outcome_properties.csv", index=False)
    print(outcome.to_string(index=False))
    print("\nTop entry differences")
    print(comparison(frame, entry_features).head(12).to_string(index=False))
    print("\nTop development-selected filters with holdout results")
    print(filters.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
