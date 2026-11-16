"""Write local IVRV trainer configs that point at external feature tables.

The generated YAML files live in ignored runtime. Source configs and deployed
models are never changed by this helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
V2 = REPO / "research" / "v2"
DEFAULT_OUTPUT = REPO / "runtime" / "v2-training-configs"

JOBS = {
    "combined": (
        V2 / "models/combined_atm_iv_rv_convergence_v2_120m/config_fullchain.yaml",
        "combined_atm_iv_rv_convergence_v2/artifacts_fullchain",
        {
            "feature_csv": "combined_atm_iv_convergence_v2_features_fullchain.csv",
            "feature_manifest": "combined_atm_iv_convergence_v2_features_fullchain_column_manifest.json",
        },
    ),
    "rv": (
        V2 / "models/fixed_strike_iv_to_rv_convergence_120m/rv_improvement_full_config.yaml",
        "fixed_strike_iv_rv_convergence/artifacts",
        {
            "feature_csv": "fixed_strike_convergence_features.csv",
            "feature_manifest": "fixed_strike_convergence_features_column_manifest.json",
        },
    ),
    "event": (
        V2 / "models/rv_decline_faster_than_iv_120m/config.yaml",
        "combined_atm_iv_rv_convergence_v2/artifacts",
        {
            "feature_csv": "combined_atm_iv_convergence_v2_features.csv",
            "feature_manifest": "combined_atm_iv_convergence_v2_features_column_manifest.json",
        },
    ),
    "fixed_iv": (
        V2 / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/config.yaml",
        "fixed_strike_iv_120m/artifacts",
        {
            "features_csv": "model_features.csv",
            "targets_csv": "model_targets.csv",
            "feature_manifest_json": "feature_manifest.json",
        },
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-root", type=Path, required=True,
        help="directory containing the combined_atm_iv_rv_convergence_v2, "
        "fixed_strike_iv_rv_convergence, and fixed_strike_iv_120m folders",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    feature_root = args.feature_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    missing: list[Path] = []
    configs: dict[str, dict] = {}
    for name, (template, subdir, inputs) in JOBS.items():
        config = yaml.safe_load(template.read_text(encoding="utf-8"))
        for key, filename in inputs.items():
            path = feature_root / subdir / filename
            if not path.is_file():
                missing.append(path)
            config["data"][key] = str(path)
        configs[name] = config
    if missing:
        parser.error("missing feature inputs:\n" + "\n".join(str(p) for p in missing))
    output.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        path = output / f"{name}.yaml"
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
