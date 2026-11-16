"""Verify copied artifacts, scaler extraction and feature-module contracts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np


BUNDLE = Path(__file__).resolve().parent
ROOT = BUNDLE.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path):
    package_name = "short_straddle_bundle_features"
    package_path = BUNDLE / "feature_engineering"
    if package_name not in __import__("sys").modules:
        package_spec = importlib.util.spec_from_file_location(
            package_name,
            package_path / "__init__.py",
            submodule_search_locations=[str(package_path)],
        )
        package = importlib.util.module_from_spec(package_spec)
        __import__("sys").modules[package_name] = package
        package_spec.loader.exec_module(package)
    module_name = f"{package_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def assert_scalers_equal(expected, actual) -> None:
    if type(expected) is not type(actual):
        raise AssertionError(f"scaler type mismatch: {type(expected)} != {type(actual)}")
    for attribute in ("mean_", "scale_", "center_", "var_"):
        left = getattr(expected, attribute, None)
        right = getattr(actual, attribute, None)
        if left is None and right is None:
            continue
        if left is None or right is None or not np.allclose(left, right, equal_nan=True):
            raise AssertionError(f"scaler attribute mismatch: {attribute}")


def main() -> None:
    manifest = json.loads((BUNDLE / "manifest.json").read_text(encoding="utf-8"))
    scaler_count = 0
    for record in manifest["models"]:
        source = ROOT / record["source"]
        bundled = BUNDLE / record["bundled_model"]
        if sha256(source) != sha256(bundled) or sha256(bundled) != record["sha256"]:
            raise AssertionError(f"artifact hash mismatch: {record['name']}")

        artifact = joblib.load(bundled)
        module = load_module(BUNDLE / record["feature_module"])
        if list(module.FEATURE_NAMES) != list(artifact["features"]):
            raise AssertionError(f"feature contract mismatch: {record['name']}")

        expected_scalers = []
        if "model" in artifact:
            expected_scalers.append(next(iter(record["scalers"])))
            pipelines = [artifact["model"]]
        else:
            pipelines = [artifact["models"][band]["model"] for band in artifact["models"]]
            expected_scalers = record["scalers"]
        if len(pipelines) != len(expected_scalers):
            raise AssertionError(f"scaler count mismatch: {record['name']}")

        for pipeline, relative_scaler in zip(pipelines, expected_scalers, strict=True):
            embedded = next(
                pipeline.named_steps[name]
                for name in ("standardscaler", "robustscaler")
                if name in pipeline.named_steps
            )
            extracted = joblib.load(BUNDLE / relative_scaler)
            assert_scalers_equal(embedded, extracted)
            scaler_count += 1

    print(f"Verified {manifest['model_count']} artifacts, {scaler_count} scalers, and all feature contracts")


if __name__ == "__main__":
    main()

