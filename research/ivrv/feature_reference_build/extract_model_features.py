import hashlib
import json
import sys
from pathlib import Path

import joblib
import yaml


WORKSPACE = Path(__file__).resolve().parents[3]
INFERENCE = WORKSPACE / "pipelines" / "ivrv"
CONFIG = INFERENCE / "config.yaml"
EVENT_PACKAGE = INFERENCE / "models" / "rv_decline_event_no120"
OUTPUT = WORKSPACE / "runtime" / "feature_reference" / "model_features.json"


MODEL_DETAILS = {
    "rv_xgboost": {
        "algorithm": "XGBoost regressor",
        "output": "Calibrated annualized forward 120 minute realized volatility",
        "preprocessing": "Log variance inverse transform and artifact calibration",
    },
    "combined_direct_lightgbm": {
        "algorithm": "LightGBM regressor",
        "output": "Combined IV change over 120 minutes",
        "preprocessing": "Native missing value handling",
    },
    "combined_error_correction_ridge": {
        "algorithm": "Ridge regression pipeline",
        "output": "Combined IV change over 120 minutes",
        "preprocessing": "Median imputer and RobustScaler embedded in pipeline",
    },
    "combined_monotonic_xgboost": {
        "algorithm": "Monotonic XGBoost regressor",
        "output": "Combined IV change over 120 minutes",
        "preprocessing": "Native tree preprocessing",
    },
    "combined_three_model_ensemble": {
        "algorithm": "Weighted Ridge XGBoost LightGBM ensemble",
        "output": "Combined IV change over 120 minutes",
        "preprocessing": "Component specific preprocessing; weights 0.0 0.1 0.9",
    },
    "combined_convergence_classifier": {
        "algorithm": "LightGBM binary classifier",
        "output": "Probability that combined IV moves closer to predicted RV",
        "preprocessing": "Native missing value handling",
    },
    "combined_quantile_lightgbm": {
        "algorithm": "Three LightGBM quantile regressors",
        "output": "Q10 Q50 and Q90 combined IV change",
        "preprocessing": "Native missing value handling",
    },
    "combined_ngboost": {
        "algorithm": "NGBoost probabilistic regressor",
        "output": "Mean and scale of combined IV change",
        "preprocessing": "Embedded median imputer and target standardization",
    },
    "call_hist_gradient_boosting": {
        "algorithm": "Histogram gradient boosting pipeline",
        "output": "Call IV 120 minute log ratio",
        "preprocessing": "Median imputer embedded in pipeline",
    },
    "put_hist_gradient_boosting": {
        "algorithm": "Histogram gradient boosting pipeline",
        "output": "Put IV 120 minute log ratio",
        "preprocessing": "Median imputer embedded in pipeline",
    },
    "call_lightgbm_q10": {
        "algorithm": "LightGBM quantile regressor",
        "output": "Q10 call IV 120 minute log ratio",
        "preprocessing": "Native missing value handling",
    },
    "call_lightgbm_q50": {
        "algorithm": "LightGBM quantile regressor",
        "output": "Q50 call IV 120 minute log ratio",
        "preprocessing": "Native missing value handling",
    },
    "call_lightgbm_q90": {
        "algorithm": "LightGBM quantile regressor",
        "output": "Q90 call IV 120 minute log ratio",
        "preprocessing": "Native missing value handling",
    },
    "put_lightgbm_q10": {
        "algorithm": "LightGBM quantile regressor",
        "output": "Q10 put IV 120 minute log ratio",
        "preprocessing": "Native missing value handling",
    },
    "put_lightgbm_q50": {
        "algorithm": "LightGBM quantile regressor",
        "output": "Q50 put IV 120 minute log ratio",
        "preprocessing": "Native missing value handling",
    },
    "put_lightgbm_q90": {
        "algorithm": "LightGBM quantile regressor",
        "output": "Q90 put IV 120 minute log ratio",
        "preprocessing": "Native missing value handling",
    },
    "rv_iv_sharp_decline_event": {
        "algorithm": "Two target ElasticNet regression bundle",
        "output": "Signed score for the sharp forward RV decline event",
        "preprocessing": "Median imputer and StandardScaler in each target pipeline",
    },
}


def digest(features: list[str]) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()[:12]


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    models = []
    contracts: dict[str, dict] = {}

    for key, relative in config["models"].items():
        artifact = (INFERENCE / relative).resolve()
        payload = joblib.load(artifact)
        features = list(payload["features"])
        feature_hash = digest(features)
        contract_id = {
            168: "RV 168",
            151: "Combined 151",
            143: "Leg 143",
        }[len(features)]
        contracts.setdefault(
            contract_id,
            {"id": contract_id, "feature_count": len(features), "hash": feature_hash, "features": features, "models": []},
        )
        if contracts[contract_id]["hash"] != feature_hash:
            raise RuntimeError(f"Contract collision for {contract_id}")
        contracts[contract_id]["models"].append(key)
        models.append(
            {
                "key": key,
                "artifact": str(artifact.relative_to(WORKSPACE)),
                "contract": contract_id,
                "feature_count": len(features),
                **MODEL_DETAILS[key],
            }
        )

    sys.path.insert(0, str(EVENT_PACKAGE))
    event_payload = joblib.load(EVENT_PACKAGE / "winner.joblib")
    event_features = list(event_payload["features"])
    event_contract = "Event 168"
    contracts[event_contract] = {
        "id": event_contract,
        "feature_count": len(event_features),
        "hash": digest(event_features),
        "features": event_features,
        "models": ["rv_iv_sharp_decline_event"],
    }
    models.append(
        {
            "key": "rv_iv_sharp_decline_event",
            "artifact": str((EVENT_PACKAGE / "winner.joblib").relative_to(WORKSPACE)),
            "contract": event_contract,
            "feature_count": len(event_features),
            **MODEL_DETAILS["rv_iv_sharp_decline_event"],
        }
    )

    result = {
        "document_date": "27 October 2026",
        "config": str(CONFIG.relative_to(WORKSPACE)),
        "models": models,
        "contracts": [contracts[name] for name in ("RV 168", "Combined 151", "Leg 143", "Event 168")],
        "event_metadata": {
            "lookback_minutes": event_payload.get("lookback_minutes"),
            "threshold": event_payload.get("threshold"),
            "target_definition": event_payload.get("target_definition"),
            "engineered_features": event_payload.get("engineered_features", []),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
