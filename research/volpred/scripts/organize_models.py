"""Build the canonical, self-contained model registry under ``model/``.

The training programs keep their original outputs in research directories.  This
script copies every serialized estimator into a stable family directory, copies
the corresponding manifests, extracts fitted StandardScaler steps for audit and
reuse, and writes a checksum-backed inventory.
"""

from __future__ import annotations

import __main__
import csv
import hashlib
import importlib
import shutil
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "model"
OUTPUTS = ROOT / "outputs"

SOURCES = {
    "atm_iv_change": OUTPUTS / "atm_iv_change_models",
    "atm_iv_change_simple_5f": OUTPUTS / "atm_iv_change_models" / "simple_five_feature",
    "log_iv_change": OUTPUTS / "log_iv_change_models",
    "log_iv_change_compact_5f": OUTPUTS / "log_iv_change_models" / "compact_five_feature",
    "realized_variance": OUTPUTS / "rv_prediction_models",
    "spot_magnitude": OUTPUTS / "spot_magnitude_models",
    "volatility_extension": OUTPUTS / "volatility_specific_models",
    "enhanced_rv": OUTPUTS / "enhanced_rv_models",
}


def install_pickle_compatibility() -> None:
    """Expose classes saved from training scripts that ran as ``__main__``."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "research"))
    modules = (
        importlib.import_module("research.simple_iv_change_models"),
        importlib.import_module("research.log_iv_change_prediction"),
        importlib.import_module("research.compact_log_iv_quantile"),
    )
    # Older artifacts referenced modules by their pre-package top-level names.
    # Map those names to the organized package modules while re-serializing.
    sys.modules["simple_iv_change_models"] = modules[0]
    sys.modules["log_iv_change_prediction"] = modules[1]
    sys.modules["compact_log_iv_quantile"] = modules[2]
    sys.modules["atm_iv_change_prediction"] = importlib.import_module("research.atm_iv_change_prediction")
    for class_name in ("FastQuantileRegressor", "TrainingWinsorizer"):
        for module in modules:
            if hasattr(module, class_name):
                setattr(__main__, class_name, getattr(module, class_name))
                break


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def scaler_from(model):
    steps = getattr(model, "named_steps", None)
    if not steps:
        return None
    for step in steps.values():
        if type(step).__name__.endswith("Scaler"):
            return step
    return None


def is_production_candidate(family: str, filename: str) -> bool:
    if family in {"realized_variance", "volatility_extension", "enhanced_rv"}:
        return True
    if family == "spot_magnitude":
        return "lightgbm" not in filename
    return False


def main() -> None:
    install_pickle_compatibility()
    REGISTRY.mkdir(exist_ok=True)
    rows: list[dict[str, object]] = []

    for family, source_dir in SOURCES.items():
        family_dir = REGISTRY / family
        artifact_dir = family_dir / "artifacts"
        scaler_dir = family_dir / "scalers"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        scaler_dir.mkdir(parents=True, exist_ok=True)

        for manifest in source_dir.glob("*manifest*.csv"):
            shutil.copy2(manifest, family_dir / manifest.name)

        for source in sorted(source_dir.glob("*.joblib")):
            destination = artifact_dir / source.name
            payload = joblib.load(source)
            # Re-serialize after compatibility loading so estimators created by a
            # training script executed as __main__ become portable imports.
            joblib.dump(payload, destination)
            model = payload.get("model") if isinstance(payload, dict) else payload
            features = payload.get("features", []) if isinstance(payload, dict) else []
            horizon = payload.get("horizon_min", "") if isinstance(payload, dict) else ""
            fitted_scaler = scaler_from(model)
            scaler_rel = ""
            if fitted_scaler is not None:
                scaler_path = scaler_dir / f"{source.stem}.scaler.joblib"
                joblib.dump(fitted_scaler, scaler_path)
                scaler_rel = scaler_path.relative_to(ROOT).as_posix()

            rows.append(
                {
                    "family": family,
                    "model_file": destination.relative_to(ROOT).as_posix(),
                    "source_file": source.relative_to(ROOT).as_posix(),
                    "horizon_min": horizon,
                    "estimator_type": type(model).__name__,
                    "feature_count": len(features),
                    "features": "|".join(map(str, features)),
                    "scaler": "embedded_and_extracted" if fitted_scaler else "not_required",
                    "scaler_file": scaler_rel,
                    "production_candidate": is_production_candidate(family, source.name),
                    "sha256": digest(destination),
                }
            )

    fields = list(rows[0])
    with (REGISTRY / "inventory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Organized {len(rows)} models in {REGISTRY}")
    print(f"Extracted {sum(row['scaler'] == 'embedded_and_extracted' for row in rows)} fitted scalers")


if __name__ == "__main__":
    main()
