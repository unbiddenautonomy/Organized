from __future__ import annotations

from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for name in (
        "entry_surface_csv", "full_chain_csv", "entries_csv", "dataset_csv", "model_features_csv",
        "model_targets_csv", "feature_manifest_json", "quality_report_json",
    ):
        config["data"][name] = str((path.parent / config["data"][name]).resolve())
    config["project_dir"] = str(path.parent)
    return config

