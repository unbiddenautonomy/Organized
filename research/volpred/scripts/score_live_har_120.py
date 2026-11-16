"""Match matured live HAR forecasts to realized 120-minute spot variance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "runtime" / "predictions.jsonl"
BARS = ROOT / "runtime" / "live_spot_bars.csv"
OUTPUT = ROOT / "runtime" / "har_120_live_scores.csv"


def main() -> None:
    if not PREDICTIONS.exists() or not BARS.exists():
        print(json.dumps({"status": "waiting_for_predictions_and_bars", "scored": 0}))
        return
    rows = []
    for line in PREDICTIONS.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            forecast = record.get("forecasts", {}).get("120", {})
            if forecast.get("available") and forecast.get("algorithm") == "har":
                rows.append({
                    "datetime": record["bar_timestamp"],
                    "forecast": forecast["future_variance"],
                    "forecast_q90": forecast["future_variance_q90"],
                })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    pred = pd.DataFrame(rows)
    if pred.empty:
        print(json.dumps({"status": "waiting_for_120m_predictions", "scored": 0}))
        return
    pred["datetime"] = pd.to_datetime(pred["datetime"], errors="coerce").dt.tz_localize(None)
    pred = pred.dropna().drop_duplicates("datetime", keep="last").sort_values("datetime")
    bars = pd.read_csv(BARS, parse_dates=["datetime"]).dropna().drop_duplicates("datetime", keep="last")
    if bars["datetime"].dt.tz is not None:
        bars["datetime"] = bars["datetime"].dt.tz_localize(None)
    bars = bars.sort_values("datetime").set_index("datetime")
    bars["return_sq"] = np.log(bars["spot_price"] / bars["spot_price"].shift(1)).pow(2)
    scored = []
    for row in pred.itertuples(index=False):
        expected = pd.date_range(row.datetime + pd.Timedelta(minutes=1), periods=120, freq="min")
        future = bars.reindex(expected)
        if future["return_sq"].notna().all():
            scored.append({
                "datetime": row.datetime, "forecast": row.forecast,
                "forecast_q90": row.forecast_q90,
                "actual": float(future["return_sq"].sum()),
            })
    frame = pd.DataFrame(scored)
    if frame.empty:
        print(json.dumps({"status": "waiting_for_mature_120m_windows", "scored": 0,
                          "predictions": len(pred)}))
        return
    frame["actual_over_forecast"] = frame["actual"] / frame["forecast"]
    frame["q90_exceedance"] = frame["actual"] > frame["forecast_q90"]
    frame.to_csv(OUTPUT, index=False)
    summary = {
        "status": "monitoring" if len(frame) < 30 else "minimum_sample_available",
        "scored": len(frame), "days": int(frame["datetime"].dt.date.nunique()),
        "mean_actual_over_forecast": float(frame["actual_over_forecast"].mean()),
        "q90_hit_rate": float(frame["q90_exceedance"].mean()),
    }
    if len(frame) >= 30:
        x = np.column_stack([np.ones(len(frame)), np.log(frame["forecast"])])
        y = np.log(frame["actual"])
        beta = np.linalg.pinv(x.T @ x) @ x.T @ y
        residual = y - x @ beta
        covariance = np.linalg.pinv(x.T @ x) * float(residual @ residual / max(len(frame) - 2, 1))
        delta = beta - np.array([0.0, 1.0])
        wald = float(delta @ np.linalg.pinv(covariance) @ delta)
        summary.update({"mz_intercept": float(beta[0]), "mz_slope": float(beta[1]),
                        "mz_pvalue_provisional": float(stats.chi2.sf(wald, 2))})
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
