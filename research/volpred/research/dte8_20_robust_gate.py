"""Frozen robust paper gate for same-session DTE 8-20 ATM straddles.

The model is deliberately small: one Huber regression per DTE band, refit at
the start of each month using only prior observations.  It forecasts net P&L
as a return on entry premium and trades above the training-only 60th
percentile.  The same selected timestamps are then repriced under all cost
stress cases so costs cannot change the signal retrospectively.
"""

from __future__ import annotations

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from dte4_20_pnl_model import FEATURES, panel
from frozen_near_expiry_strategy import bootstrap_total


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/dte8_20_robust_gate"
ARTIFACT = ROOT / "model/dte4_20_paper/dte8_20_huber_gate.joblib"
MIN_TRAIN = 100
THRESHOLD_QUANTILE = 0.60
BANDS = {"8-14": (8.0, 15.0), "15-20": (15.0, 21.0)}


def estimator():
    return make_pipeline(
        RobustScaler(),
        HuberRegressor(epsilon=1.35, alpha=0.01, max_iter=2000),
    )


def walk_forward(frame: pd.DataFrame) -> pd.DataFrame:
    outputs = []
    for band, (low, high) in BANDS.items():
        data = frame[(frame.dte >= low) & (frame.dte < high)].sort_values("datetime")
        months = data.datetime.dt.to_period("M")
        for month in months.drop_duplicates().sort_values():
            test = data[months.eq(month)]
            train = data[data.datetime < test.datetime.min()]
            if len(train) < MIN_TRAIN:
                continue
            model = estimator().fit(train[FEATURES], train.net_return)
            train_prediction = model.predict(train[FEATURES])
            threshold = max(0.0, float(np.quantile(train_prediction, THRESHOLD_QUANTILE)))
            result = test.copy()
            result["dte_band"] = band
            result["predicted_net_return"] = model.predict(test[FEATURES])
            result["train_threshold"] = threshold
            result["train_size"] = len(train)
            result["signal"] = result.predicted_net_return > threshold
            outputs.append(result)
    return pd.concat(outputs, ignore_index=True)


def performance(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (band, spread), group in trades.groupby(["dte_band", "half_spread_bps"]):
        daily = group.groupby("date").net_pnl.sum()
        curve = daily.cumsum()
        lo, hi, p = bootstrap_total(group)
        rows.append({
            "dte_band": band,
            "half_spread_bps": spread,
            "trades": len(group),
            "net_pnl": group.net_pnl.sum(),
            "mean_trade": group.net_pnl.mean(),
            "hit_rate": (group.net_pnl > 0).mean(),
            "worst_trade": group.net_pnl.min(),
            "max_drawdown": (curve.cummax() - curve).max(),
            "ci_low": lo,
            "ci_high": hi,
            "p_net_le_zero": p,
        })
    return pd.DataFrame(rows)


def freeze_artifact(frame: pd.DataFrame) -> dict:
    models = {}
    for band, (low, high) in BANDS.items():
        train = frame[(frame.dte >= low) & (frame.dte < high)].sort_values("datetime")
        model = estimator().fit(train[FEATURES], train.net_return)
        threshold = max(0.0, float(np.quantile(model.predict(train[FEATURES]), THRESHOLD_QUANTILE)))
        models[band] = {
            "model": model,
            "threshold": threshold,
            "dte_min": low,
            "dte_max_exclusive": high,
            "training_rows": len(train),
        }
    artifact = {
        "models": models,
        "features": FEATURES,
        "status": "PAPER_ONLY",
        "entry_start": "09:45",
        "entry_end": "09:59",
        "exit_rule": "same-session at or after 15:15",
        "half_spread_bps": 25.0,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "trained_through": str(frame.datetime.max()),
        "research_status": "exploratory_candidate_requires_forward_validation",
    }
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    return artifact


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    primary = panel(25.0)
    oos = walk_forward(primary)
    selected = oos[oos.signal].copy()
    keys = selected[["datetime", "dte_band", "predicted_net_return", "train_threshold", "train_size"]]
    stressed = []
    for spread in (10.0, 25.0, 50.0, 100.0):
        stressed.append(panel(spread).merge(keys, on="datetime", how="inner", validate="one_to_one"))
    trades = pd.concat(stressed, ignore_index=True)
    stats = performance(trades)
    artifact = freeze_artifact(primary[(primary.dte >= 8) & (primary.dte < 21)])
    oos.to_csv(OUT / "all_walk_forward_predictions.csv", index=False)
    trades.to_csv(OUT / "selected_trades_cost_stress.csv", index=False)
    stats.to_csv(OUT / "performance.csv", index=False)
    (OUT / "artifact_metadata.json").write_text(json.dumps({
        k: v for k, v in artifact.items() if k not in {"models"}
    }, indent=2), encoding="utf-8")
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
