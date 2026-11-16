"""Rebuild the short-straddle paper model/scaler bundle reproducibly."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = Path(__file__).resolve().parent
MODELS = BUNDLE / "models"
SCALERS = BUNDLE / "scalers"

SPECS = (
    {
        "name": "enhanced_rv_15m_model",
        "source": "model/enhanced_rv/artifacts/enhanced_rv_15m_model.joblib",
        "feature_module": "feature_engineering/enhanced_rv_15m_model.py",
        "role": "live 15-minute realized-variance forecast",
    },
    {
        "name": "enhanced_rv_30m_model",
        "source": "model/enhanced_rv/artifacts/enhanced_rv_30m_model.joblib",
        "feature_module": "feature_engineering/enhanced_rv_30m_model.py",
        "role": "live 30-minute realized-variance forecast",
    },
    {
        "name": "enhanced_rv_60m_model",
        "source": "model/enhanced_rv/artifacts/enhanced_rv_60m_model.joblib",
        "feature_module": "feature_engineering/enhanced_rv_60m_model.py",
        "role": "live 60-minute realized-variance forecast",
    },
    {
        "name": "enhanced_rv_120m_model",
        "source": "model/enhanced_rv/artifacts/enhanced_rv_120m_model.joblib",
        "feature_module": "feature_engineering/enhanced_rv_120m_model.py",
        "role": "live 120-minute realized-variance forecast",
    },
    {
        "name": "dte4_7_pnl_gate",
        "source": "model/dte4_20_paper/dte4_7_pnl_gate.joblib",
        "feature_module": "feature_engineering/dte4_7_pnl_gate.py",
        "role": "active 4-7 DTE ATM short-straddle decision gate",
    },
    {
        "name": "dte8_20_huber_gate",
        "source": "model/dte4_20_paper/dte8_20_huber_gate.joblib",
        "feature_module": "feature_engineering/dte8_20_huber_gate.py",
        "role": "active 8-14 DTE ATM short-straddle decision gate",
    },
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pipeline_scaler(pipeline):
    for step_name in ("standardscaler", "robustscaler"):
        if step_name in pipeline.named_steps:
            return pipeline.named_steps[step_name]
    raise ValueError(f"no supported scaler found in {pipeline.named_steps.keys()}")


def main() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    SCALERS.mkdir(parents=True, exist_ok=True)
    records = []

    for spec in SPECS:
        source = ROOT / spec["source"]
        destination = MODELS / source.name
        shutil.copy2(source, destination)
        artifact = joblib.load(source)

        scaler_paths = []
        if "model" in artifact:
            scaler_path = SCALERS / f"{spec['name']}.scaler.joblib"
            joblib.dump(pipeline_scaler(artifact["model"]), scaler_path)
            scaler_paths.append(scaler_path.relative_to(BUNDLE).as_posix())
        else:
            for band, band_model in artifact["models"].items():
                safe_band = band.replace("-", "_")
                scaler_path = SCALERS / f"{spec['name']}_{safe_band}.scaler.joblib"
                joblib.dump(pipeline_scaler(band_model["model"]), scaler_path)
                scaler_paths.append(scaler_path.relative_to(BUNDLE).as_posix())

        records.append(
            {
                **spec,
                "bundled_model": destination.relative_to(BUNDLE).as_posix(),
                "sha256": sha256(destination),
                "source_sha256": sha256(source),
                "features": list(artifact["features"]),
                "scalers": scaler_paths,
                "paper_only": True,
            }
        )

    manifest = {
        "bundle": "BANKNIFTY short-straddle paper trading",
        "model_count": len(records),
        "models": records,
        "excluded": {
            "model/dte4_20_paper/multi_structure_selector.joblib": (
                "Active at DTE 15-20 for short strangles; outside short-straddle scope."
            )
        },
    }
    (BUNDLE / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Built {len(records)} model artifacts and {sum(len(x['scalers']) for x in records)} scalers")


if __name__ == "__main__":
    main()

