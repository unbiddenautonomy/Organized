from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


SOURCE = Path("artifacts/frozen_models_2026_holdout")
DESTINATION = Path("model_releases")
METRICS = json.loads((SOURCE / "metrics.json").read_text(encoding="utf-8"))
TRAINING_CODE = [
    Path("retrain_frozen_models_2026.py"),
    Path("train_high_win_rate_models.py"),
    Path("train_high_win_rate_hybrid.py"),
    Path("train_fixed_atm_horizons.py"),
    Path("package_model_releases.py"),
]


def safe_json(value):
    if isinstance(value, float):
        if value != value:
            return None
        if value == float("inf"):
            return "Infinity"
        if value == float("-inf"):
            return "-Infinity"
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [safe_json(item) for item in value]
    return value


def dump_pipeline_parts(pipeline, folder: Path, prefix: str = "") -> dict:
    names = {"pipeline": None, "preprocessor": None, "estimator": None, "scaler": None}
    pipeline_name = f"{prefix}pipeline.joblib"
    joblib.dump(pipeline, folder / pipeline_name)
    names["pipeline"] = pipeline_name
    if isinstance(pipeline, Pipeline):
        estimator_name = f"{prefix}estimator.joblib"
        joblib.dump(pipeline.steps[-1][1], folder / estimator_name)
        names["estimator"] = estimator_name
        if len(pipeline.steps) > 1:
            preprocessor = Pipeline(pipeline.steps[:-1])
            preprocessor_name = f"{prefix}preprocessor.joblib"
            joblib.dump(preprocessor, folder / preprocessor_name)
            names["preprocessor"] = preprocessor_name
        scaler = pipeline.named_steps.get("scale")
        scaler_name = f"{prefix}scaler.joblib"
        if scaler is not None:
            joblib.dump(scaler, folder / scaler_name)
            names["scaler"] = scaler_name
        else:
            joblib.dump(
                {
                    "required": False,
                    "reason": "The released estimator is tree-based and does not use feature scaling.",
                },
                folder / scaler_name,
            )
            names["scaler"] = scaler_name
    else:
        estimator_name = f"{prefix}estimator.joblib"
        joblib.dump(pipeline, folder / estimator_name)
        names["estimator"] = estimator_name
    return names


def write_requirements(folder: Path) -> None:
    (folder / "requirements.txt").write_text(
        "\n".join(
            [
                "python>=3.13,<3.14",
                "pandas==2.3.3",
                "numpy==2.5.2",
                "scikit-learn==1.9.0",
                "lightgbm==4.7.0",
                "joblib==1.5.3",
                "duckdb>=1.4,<2",
                "",
            ]
        ),
        encoding="utf-8",
    )


def copy_training_code(folder: Path) -> None:
    code = folder / "training_code"
    code.mkdir(exist_ok=True)
    for source in TRAINING_CODE:
        shutil.copy2(source, code / source.name)


def write_classifier_inference(folder: Path) -> None:
    (folder / "predict.py").write_text(
        '''from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(frame: pd.DataFrame, model_path: str | Path = "model.joblib") -> pd.DataFrame:
    artifact = joblib.load(model_path)
    missing = [name for name in artifact["features"] if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    probability = artifact["model"].predict_proba(frame[artifact["features"]])[:, 1]
    threshold = float(artifact["probability_threshold"])
    return pd.DataFrame({
        "win_probability": probability,
        "enter_trade": probability >= threshold,
    }, index=frame.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(pd.read_csv(args.features_csv), args.model).to_csv(args.output, index=False)
''',
        encoding="utf-8",
    )


def write_hybrid_inference(folder: Path) -> None:
    (folder / "predict.py").write_text(
        '''from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd


def predict(frame: pd.DataFrame, model_path: str | Path = "model.joblib") -> pd.DataFrame:
    artifact = joblib.load(model_path)
    missing = [name for name in artifact["features"] if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing}")
    values = frame[artifact["features"]]
    probability = artifact["classifier"].predict_proba(values)[:, 1]
    predicted_edge = artifact["regressor"].predict(values)
    enter = (
        (probability >= float(artifact["probability_threshold"]))
        & (predicted_edge >= float(artifact["predicted_net_edge_threshold"]))
    )
    return pd.DataFrame({
        "win_probability": probability,
        "predicted_net_edge_points": predicted_edge,
        "enter_trade": enter,
    }, index=frame.index)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("features_csv")
    parser.add_argument("--model", default="model.joblib")
    parser.add_argument("--output", default="predictions.csv")
    args = parser.parse_args()
    predict(pd.read_csv(args.features_csv), args.model).to_csv(args.output, index=False)
''',
        encoding="utf-8",
    )


def classifier_model_card(release_name: str, result: dict, artifact: dict) -> str:
    metrics = result["metrics_2026"]
    return f"""# {release_name}

## Purpose

Predict whether a fixed-signal-ATM BANKNIFTY short strangle will produce positive net premium decay over {result['horizon_minutes']} minutes after two points of modeled round-trip cost.

## Frozen configuration

- Model: {result['model']}
- Horizon: {result['horizon_minutes']} minutes
- Win-probability threshold: {result['probability_threshold']:.2f}
- Entry strike: ATM at signal time and fixed through exit
- Execution: next one-minute full-chain quote after the signal snapshot
- Development data end: {METRICS['development_end']}
- Unseen holdout: {METRICS['holdout_start']} through {METRICS['holdout_end']}
- Feature count: {len(artifact['features'])}

## 2026 unseen holdout

- Trades: {metrics['trades']}
- Win rate: {metrics['win_rate_after_slippage']:.2%}
- Net points: {metrics['net_points_after_2pt_slippage']:.2f}
- Profit factor: {metrics['profit_factor']}
- Maximum drawdown points: {metrics['maximum_drawdown_points']}
- 95% Wilson win-rate lower bound: {metrics['wilson_win_rate_lower_95']:.2%}

## Preprocessing

Use `pipeline.joblib` or `model.joblib` for inference. The fitted preprocessor is provided separately in `preprocessor.joblib`. `scaler.joblib` contains the fitted scaler when one is used; for tree models it records that scaling is not required.

## Limitations

- Statutory charges and bid/ask spreads are not included.
- Results use minute-close execution with two points of modeled round-trip cost.
- Exact fixed-strike quote availability can reduce sample coverage.
- The 2026 source uses synthetic futures when a futures quote is unavailable.
- This artifact is research-only and is not approved for live deployment.
"""


def hybrid_model_card(release_name: str, result: dict, artifact: dict) -> str:
    metrics = result["metrics_2026"]
    return f"""# {release_name}

## Purpose

Combine a 90-minute positive-outcome classifier with a net-decay regressor. A trade requires both the frozen win-probability and predicted-edge gates.

## Frozen configuration

- Classifier: Extra Trees classifier
- Regressor: LightGBM Huber regressor
- Horizon: 90 minutes
- Win-probability threshold: {artifact['probability_threshold']:.2f}
- Predicted net-edge threshold: {artifact['predicted_net_edge_threshold']:.1f} points
- Entry strike: ATM at signal time and fixed through exit
- Execution: next one-minute full-chain quote after the signal snapshot
- Development data end: {METRICS['development_end']}
- Unseen holdout: {METRICS['holdout_start']} through {METRICS['holdout_end']}
- Feature count: {len(artifact['features'])}

## 2026 unseen holdout

- Trades: {metrics['trades']}
- Net points: {metrics['net_points_after_2pt_slippage']:.2f}
- Result: No qualifying trades; win rate and profit factor are undefined.

## Preprocessing

The classifier and regressor pipelines are embedded in `model.joblib`. Their fitted preprocessors and estimators are also supplied separately. Neither tree model requires feature scaling; the scaler artifacts record this explicitly.

## Limitations

- The frozen hybrid did not trigger in the unseen 2026 holdout.
- Statutory charges and bid/ask spreads are not included.
- Exact fixed-strike quote availability can reduce sample coverage.
- The 2026 source uses synthetic futures when a futures quote is unavailable.
- This artifact is research-only and is not approved for live deployment.
"""


def checksums(folder: Path) -> None:
    rows = []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS.txt":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(folder).as_posix()}")
    (folder / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def package_classifier(result: dict) -> dict:
    horizon = result["horizon_minutes"]
    release_name = f"banknifty_fixed_atm_{horizon}m_{result['model']}"
    folder = DESTINATION / release_name
    folder.mkdir(parents=True, exist_ok=True)
    source_artifact = SOURCE / result["artifact"]
    shutil.copy2(source_artifact, folder / "model.joblib")
    artifact = joblib.load(source_artifact)
    parts = dump_pipeline_parts(artifact["model"], folder)
    feature_manifest = {
        "features": artifact["features"],
        "feature_count": len(artifact["features"]),
        "target": "positive fixed-strike net decay after two points of cost",
        "horizon_minutes": horizon,
        "probability_threshold": artifact["probability_threshold"],
    }
    (folder / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(safe_json(result), indent=2), encoding="utf-8")
    shutil.copy2(SOURCE / f"classifier_2026_trades_{horizon}m.csv", folder / "test_trades_2026.csv")
    shutil.copy2(SOURCE / "REPORT.md", folder / "EVALUATION_REPORT.md")
    write_classifier_inference(folder)
    write_requirements(folder)
    copy_training_code(folder)
    (folder / "MODEL_CARD.md").write_text(classifier_model_card(release_name, result, artifact), encoding="utf-8")
    checksums(folder)
    return {"release": release_name, "folder": str(folder), "parts": parts}


def package_hybrid(result: dict) -> dict:
    release_name = "banknifty_fixed_atm_90m_high_win_rate_hybrid"
    folder = DESTINATION / release_name
    folder.mkdir(parents=True, exist_ok=True)
    source_artifact = SOURCE / "frozen_hybrid_90m.joblib"
    shutil.copy2(source_artifact, folder / "model.joblib")
    artifact = joblib.load(source_artifact)
    classifier_parts = dump_pipeline_parts(artifact["classifier"], folder, "classifier_")
    regressor_parts = dump_pipeline_parts(artifact["regressor"], folder, "regressor_")
    feature_manifest = {
        "features": artifact["features"],
        "feature_count": len(artifact["features"]),
        "target": "positive fixed-strike outcome plus predicted net-decay gate",
        "horizon_minutes": 90,
        "probability_threshold": artifact["probability_threshold"],
        "predicted_net_edge_threshold": artifact["predicted_net_edge_threshold"],
    }
    (folder / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2), encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(safe_json(result), indent=2), encoding="utf-8")
    shutil.copy2(SOURCE / "hybrid_2026_trades_90m.csv", folder / "test_trades_2026.csv")
    shutil.copy2(SOURCE / "REPORT.md", folder / "EVALUATION_REPORT.md")
    write_hybrid_inference(folder)
    write_requirements(folder)
    copy_training_code(folder)
    (folder / "MODEL_CARD.md").write_text(hybrid_model_card(release_name, result, artifact), encoding="utf-8")
    checksums(folder)
    return {
        "release": release_name,
        "folder": str(folder),
        "classifier_parts": classifier_parts,
        "regressor_parts": regressor_parts,
    }


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    releases = [package_classifier(result) for result in METRICS["classifier_results"]]
    releases.append(package_hybrid(METRICS["hybrid_result"]))
    index = {
        "source_report": str(SOURCE / "REPORT.md"),
        "development_end": METRICS["development_end"],
        "holdout_start": METRICS["holdout_start"],
        "holdout_end": METRICS["holdout_end"],
        "releases": releases,
    }
    (DESTINATION / "release_index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    readme = [
        "# BANKNIFTY Fixed ATM Model Releases",
        "",
        "Each subfolder is a self-contained research release with its full model artifact, fitted preprocessing, estimator, scaler metadata, inference code, training code, feature manifest, model card, metrics, test trades, requirements, and SHA-256 checksums.",
        "",
        "## Releases",
        "",
    ]
    readme.extend(f"- `{row['release']}`" for row in releases)
    readme.extend(["", "See each `MODEL_CARD.md` before using an artifact.", ""])
    (DESTINATION / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
