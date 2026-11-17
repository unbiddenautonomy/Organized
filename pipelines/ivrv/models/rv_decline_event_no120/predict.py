"""Standalone batch inference for the no-120-minute-input RV/IV model."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Required so joblib can resolve model_components.SeparateTargetRegressor.
import model_components  # noqa: F401


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    def add(name: str, values) -> None:
        result[name] = values.replace([np.inf, -np.inf], np.nan)

    iv = pd.to_numeric(result.get("entry_atm_iv_combined"), errors="coerce")
    for horizon in (5, 15, 30, 60):
        rv_name = f"rv_trailing_{horizon}m"
        if rv_name not in result:
            continue
        rv = pd.to_numeric(result[rv_name], errors="coerce")
        add(f"engineered_iv_minus_rv_{horizon}m", iv - rv)
        add(f"engineered_iv_to_rv_ratio_{horizon}m", iv / rv.where(rv.abs() > 1e-8))
        add(f"engineered_log_rv_{horizon}m", np.log(rv.clip(lower=1e-8)))
    add("engineered_log_entry_iv", np.log(iv.clip(lower=1e-8)))

    available = [h for h in (5, 15, 30, 60) if f"rv_trailing_{h}m" in result]
    for short, long in zip(available[:-1], available[1:]):
        add(
            f"engineered_rv_term_slope_{short}m_{long}m",
            pd.to_numeric(result[f"rv_trailing_{short}m"], errors="coerce")
            - pd.to_numeric(result[f"rv_trailing_{long}m"], errors="coerce"),
        )
    if "entry_straddle_price" in result and "spot_price" in result:
        spot = pd.to_numeric(result["spot_price"], errors="coerce")
        add(
            "engineered_straddle_to_spot",
            pd.to_numeric(result["entry_straddle_price"], errors="coerce") / spot.where(spot.abs() > 1e-8),
        )
    if "call_oi" in result and "put_oi" in result:
        call_oi = pd.to_numeric(result["call_oi"], errors="coerce")
        put_oi = pd.to_numeric(result["put_oi"], errors="coerce")
        add("engineered_put_call_oi_ratio", put_oi / call_oi.where(call_oi.abs() > 1e-8))
        total_oi = put_oi + call_oi
        add("engineered_put_call_oi_imbalance", (put_oi - call_oi) / total_oi.where(total_oi.abs() > 1e-8))
    return result


def numeric_matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    matrix = frame.reindex(columns=features).apply(pd.to_numeric, errors="coerce")
    return matrix.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict sharper RV decline versus IV over the next 120 minutes")
    parser.add_argument("input_csv", help="CSV containing the raw entry-time feature columns")
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--model", default=str(Path(__file__).with_name("winner.joblib")))
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = bundle["features"]
    if any("120m" in name.lower() for name in features):
        raise RuntimeError("Model bundle unexpectedly contains a 120-minute input feature")

    frame = add_engineered_features(pd.read_csv(args.input_csv, low_memory=False))
    missing = sorted(set(features) - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required columns; first missing: {missing[:10]}")

    components = bundle["model"].predict(numeric_matrix(frame, features))
    predicted_rv_change = components[:, 0]
    predicted_iv_change = components[:, 1]
    settings = bundle["target_definition"]
    rv_margin = -float(settings["minimum_rv_drop"]) - predicted_rv_change
    relative_margin = (
        predicted_iv_change - predicted_rv_change - float(settings["minimum_rv_drop_beyond_iv"])
    )
    score = np.minimum(rv_margin, relative_margin)

    identity = [name for name in ("datetime", "entry_strike", "expiry") if name in frame]
    output = frame[identity].copy()
    output["model_score"] = score
    output["predicted_event"] = score >= float(bundle["threshold"])
    output["predicted_rv_change_120m"] = predicted_rv_change
    output["predicted_iv_change_120m"] = predicted_iv_change
    output.to_csv(args.output, index=False)
    print(f"Wrote {len(output):,} predictions to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
