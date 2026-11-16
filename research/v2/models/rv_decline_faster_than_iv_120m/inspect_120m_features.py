from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect 120-minute feature coefficients in the refined Elastic Net model")
    parser.add_argument("--model", default="artifacts_oot_2026/winner.joblib")
    parser.add_argument("--output", default="artifacts_oot_2026/feature_relevance_120m.csv")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    features = list(bundle["features"])
    rv_pipeline = bundle["model"].rv_model_
    iv_pipeline = bundle["model"].iv_model_
    rv_coef = np.asarray(rv_pipeline.named_steps["model"].coef_, float)
    iv_coef = np.asarray(iv_pipeline.named_steps["model"].coef_, float)
    rows = []
    for index, feature in enumerate(features):
        if "120m" not in feature:
            continue
        rows.append({
            "feature": feature,
            "rv_change_standardized_coefficient": float(rv_coef[index]),
            "iv_change_standardized_coefficient": float(iv_coef[index]),
            "rv_absolute_coefficient": float(abs(rv_coef[index])),
            "iv_absolute_coefficient": float(abs(iv_coef[index])),
            "combined_absolute_coefficient": float(abs(rv_coef[index]) + abs(iv_coef[index])),
        })
    output = pd.DataFrame(rows).sort_values("combined_absolute_coefficient", ascending=False)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
