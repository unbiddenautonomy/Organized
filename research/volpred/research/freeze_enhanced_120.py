"""Freeze the already-evaluated Enhanced 120-minute candidate for future testing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np

from enhanced_rv_model import (
    ENHANCED_FEATURES, EPS, OUT, TRAIN_DAYS, bounds, clipped, fit_model,
    smear_tables,
)
from validate_120_rv_model import tune


ROOT = Path(__file__).resolve().parents[1]
PANEL = OUT / "enhanced_spot_panel.parquet"
DEST = ROOT / "model" / "candidates" / "enhanced_rv_120_frozen_20261012.joblib"
META = ROOT / "model" / "candidates" / "enhanced_rv_120_frozen_20261012.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if DEST.exists() or META.exists():
        raise FileExistsError("Frozen candidate already exists; refusing to overwrite it.")
    panel = duckdb.connect().execute("SELECT * FROM read_parquet(?)", [str(PANEL)]).fetchdf()
    target = "future_var_120"
    required = ENHANCED_FEATURES + [target, "future_abs_log_return_120"]
    data = panel[np.isfinite(panel[required]).all(axis=1) & panel[target].gt(0)].copy()
    alpha = tune(data, ENHANCED_FEATURES, target)
    final_days = data["date"].drop_duplicates().sort_values().iloc[-TRAIN_DAYS:]
    final = data[data["date"].isin(final_days)].copy()
    low, high = bounds(final, ENHANCED_FEATURES)
    final.loc[:, ENHANCED_FEATURES] = clipped(final, ENHANCED_FEATURES, low, high)
    model, residuals = fit_model(final, ENHANCED_FEATURES, target, alpha)
    global_smear, bucket_smear, bucket_q90 = smear_tables(final, residuals)
    standardized_abs = final["future_abs_log_return_120"] / np.sqrt(final[target].clip(lower=EPS))
    artifact = {
        "model": model, "features": ENHANCED_FEATURES, "horizon_min": 120,
        "algorithm": "enhanced_frozen_candidate", "alpha": alpha,
        "clip_lower": low.to_dict(), "clip_upper": high.to_dict(),
        "global_smear": global_smear, "bucket_smear": bucket_smear,
        "bucket_residual_q90": bucket_q90,
        "absolute_return_multiplier": {
            "mean": float(standardized_abs.mean()),
            "q80": float(standardized_abs.quantile(0.80)),
            "q90": float(standardized_abs.quantile(0.90)),
        },
        "frozen": True,
    }
    DEST.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, DEST)
    metadata = {
        "status": "FROZEN_NOT_DEPLOYED", "created": "2026-10-12",
        "artifact": str(DEST.relative_to(ROOT)), "artifact_sha256": sha256(DEST),
        "source_panel": str(PANEL.relative_to(ROOT)), "source_panel_sha256": sha256(PANEL),
        "training_start": str(final["date"].min().date()),
        "training_end": str(final["date"].max().date()),
        "training_days": int(final["date"].nunique()), "training_rows": len(final),
        "alpha": alpha, "features": ENHANCED_FEATURES,
        "deployment_note": "This artifact is excluded from the live registry. HAR remains deployed.",
    }
    META.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
