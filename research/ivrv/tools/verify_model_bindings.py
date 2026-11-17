from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import joblib
import yaml


BASE = Path(__file__).resolve().parent
REPO = BASE.parents[2]
CONFIG = REPO / "pipelines" / "ivrv" / "config.yaml"
OUTPUT = BASE / "artifacts" / "ACTIVE_MODEL_BINDINGS.json"
RETRAINED_SOURCES = {
    "rv_xgboost": REPO / "research" / "v2"
    / "models/fixed_strike_iv_to_rv_convergence_120m/artifacts_rv_improved_full_ablation_no_120m/xgboost.joblib",
    "combined_direct_lightgbm": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_direct_lightgbm.joblib",
    "combined_error_correction_ridge": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_error_correction_ridge.joblib",
    "combined_monotonic_xgboost": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_monotonic_xgboost.joblib",
    "combined_three_model_ensemble": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_three_model_ensemble.joblib",
    "combined_convergence_classifier": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_convergence_classifier.joblib",
    "combined_quantile_lightgbm": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_quantile_lightgbm.joblib",
    "combined_ngboost": REPO / "research" / "v2"
    / "models/combined_atm_iv_rv_convergence_v2_120m/artifacts_ablation_no_120m/combined_ngboost.joblib",
    "call_hist_gradient_boosting": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/call_hist_gradient_boosting.joblib",
    "put_hist_gradient_boosting": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/put_hist_gradient_boosting.joblib",
    "call_lightgbm_q10": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/call_lightgbm_q10.joblib",
    "call_lightgbm_q50": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/call_lightgbm_q50.joblib",
    "call_lightgbm_q90": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/call_lightgbm_q90.joblib",
    "put_lightgbm_q10": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/put_lightgbm_q10.joblib",
    "put_lightgbm_q50": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/put_lightgbm_q50.joblib",
    "put_lightgbm_q90": BASE.parent
    / "research/fixed_strike_iv_120m/models/predict_fixed_atm_strike_iv_120m/artifacts_inference_60m/put_lightgbm_q90.joblib",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_120m_feature(name: str) -> bool:
    return (
        (name.endswith("120m") or "_120m_" in name)
        and name != "rv_pred_120m"
        and not name.startswith("entry_gap_to_rv_pred_")
    )


def main() -> None:
    raw = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    records = []
    for name, configured in raw["models"].items():
        path = (BASE / configured).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if "backup_20261022" in str(path).lower():
            raise AssertionError(f"{name} points to a backup artifact: {path}")
        payload = joblib.load(path)
        features = list(payload.get("features", []))
        active_hash = sha256(path)
        retrained_source = RETRAINED_SOURCES.get(name)
        retrained_hash_match = None
        if retrained_source is not None:
            retrained_hash_match = retrained_source.is_file() and active_hash == sha256(retrained_source)
            if not retrained_hash_match:
                raise AssertionError(f"{name} does not match its retrained source artifact")
            forbidden = [feature for feature in features if historical_120m_feature(feature)]
            if forbidden:
                raise AssertionError(f"{name} still contains historical 120m features: {forbidden}")
        records.append(
            {
                "name": name,
                "resolved_path": str(path),
                "sha256": active_hash,
                "feature_count": len(features),
                "retrained_canonical_model": retrained_source is not None,
                "retrained_source_hash_match": retrained_hash_match,
            }
        )
    result = {
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(CONFIG),
        "all_paths_exist": True,
        "backup_paths_referenced": False,
        "retrained_model_bindings_verified": len(RETRAINED_SOURCES),
        "models": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
