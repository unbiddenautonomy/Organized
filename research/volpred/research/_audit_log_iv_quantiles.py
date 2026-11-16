import duckdb
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
p = ROOT / "outputs" / "log_iv_change_models" / "walk_forward_predictions.parquet"
df = duckdb.connect().execute("select * from read_parquet(?)", [p]).fetchdf()
rows = []
for subset, data in (("intraday", df), ("0915", df[df["is_0915"].eq(1)])):
    for h, part in data.groupby("horizon_min"):
        y = part[f"target_dlog_iv_{h}"].to_numpy()
        ordered = (part["q10"] <= part["q50"]) & (part["q50"] <= part["q90"])
        covered = (y >= np.minimum(part["q10"], part["q90"])) & (y <= np.maximum(part["q10"], part["q90"]))
        rows.append({
            "subset": subset, "horizon_min": h, "n": len(part),
            "q10_q50_q90_ordered_rate": ordered.mean(),
            "empirical_10_90_coverage": covered.mean(),
            "mean_band_width": (part["q90"] - part["q10"]).mean(),
        })
out = pd.DataFrame(rows)
out.to_csv(ROOT / "outputs" / "log_iv_change_models" / "quantile_band_calibration.csv", index=False)
print(out.to_string(index=False))
