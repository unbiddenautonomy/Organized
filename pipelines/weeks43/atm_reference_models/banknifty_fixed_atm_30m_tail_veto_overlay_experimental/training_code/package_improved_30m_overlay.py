from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "artifacts" / "2026_tail_veto_overlay"
RELEASE = ROOT / "model_releases" / "banknifty_fixed_atm_30m_tail_veto_overlay_experimental"


PREDICT_CODE = '''from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


class TailVetoPredictor:
    """Stateful sequential inference; process rows in timestamp order."""

    def __init__(self, model_path: str | Path = "model.joblib") -> None:
        artifact = joblib.load(model_path)
        self.artifact = artifact
        self.history = deque(
            [float(x) for x in artifact["rolling_score_history"]],
            maxlen=int(artifact["rolling_score_window"]),
        )

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        entry_features = self.artifact["entry_features"]
        veto_features = self.artifact["veto_features"]
        missing = [x for x in entry_features + veto_features if x not in frame.columns]
        if missing:
            raise ValueError(f"Missing features: {sorted(set(missing))}")
        entry_probability = self.artifact["entry_model"].predict_proba(
            frame[entry_features]
        )[:, 1]
        tail_probability = self.artifact["tail_veto_model"].predict_proba(
            frame[veto_features]
        )[:, 1]
        veto_threshold, veto = [], []
        for score in tail_probability:
            threshold = float(np.quantile(np.asarray(self.history), self.artifact["veto_percentile"]))
            veto_threshold.append(threshold)
            veto.append(bool(score >= threshold))
            self.history.append(float(score))
        candidate = entry_probability >= float(self.artifact["entry_threshold"])
        return pd.DataFrame(
            {
                "entry_probability": entry_probability,
                "tail_probability": tail_probability,
                "tail_veto_threshold": veto_threshold,
                "tail_veto": veto,
                "enter_trade": candidate & ~np.asarray(veto),
            },
            index=frame.index,
        )
'''


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RELEASE.mkdir(parents=True, exist_ok=True)
    (RELEASE / "training_code").mkdir(exist_ok=True)
    entry = joblib.load(ROOT / "artifacts/frozen_models_2026_holdout/frozen_classifier_30m.joblib")
    veto = joblib.load(ROOT / "artifacts/tail_risk_veto/tail_veto_30m.joblib")
    composite = {
        "entry_model": entry["model"],
        "tail_veto_model": veto["model"],
        "entry_features": entry["features"],
        "veto_features": veto["features"],
        "entry_threshold": float(entry["probability_threshold"]),
        "veto_percentile": 0.80,
        "rolling_score_window": int(veto["rolling_score_window"]),
        "rolling_score_history": veto["rolling_score_history"],
        "horizon_minutes": 30,
        "strike_rule": "ATM at signal time; same CE/PE strike held through t+30m",
        "status": "experimental; 2026 overlay evaluation is diagnostic, not fresh holdout",
    }
    joblib.dump(composite, RELEASE / "model.joblib")
    joblib.dump(entry["model"], RELEASE / "entry_pipeline.joblib")
    joblib.dump(veto["model"], RELEASE / "tail_veto_pipeline.joblib")
    joblib.dump(
        {
            "entry_imputer": entry["model"].named_steps["imputer"],
            "tail_imputers": {
                name: estimator.named_steps["imputer"]
                for name, estimator in veto["model"].named_estimators_.items()
            },
        },
        RELEASE / "preprocessor.joblib",
    )
    joblib.dump(
        veto["model"].named_estimators_["logistic"].named_steps["scale"],
        RELEASE / "scaler.joblib",
    )
    joblib.dump(
        {
            "entry_estimator": entry["model"].named_steps["model"],
            "tail_estimators": veto["model"].named_estimators_,
        },
        RELEASE / "estimators.joblib",
    )
    shutil.copy2(SOURCE / "metrics.json", RELEASE / "metrics.json")
    shutil.copy2(SOURCE / "REPORT.md", RELEASE / "EVALUATION_REPORT.md")
    shutil.copy2(SOURCE / "trades_before_veto.csv", RELEASE / "trades_before_veto.csv")
    shutil.copy2(SOURCE / "trades_after_veto.csv", RELEASE / "trades_after_veto.csv")
    (RELEASE / "feature_manifest.json").write_text(
        json.dumps({"entry_features": entry["features"], "tail_veto_features": veto["features"]}, indent=2),
        encoding="utf-8",
    )
    (RELEASE / "scaler_metadata.json").write_text(
        json.dumps({"entry_model": "no scaler (tree model)", "tail_logistic_component": "fitted StandardScaler in scaler.joblib", "tail_tree_components": "no scaler"}, indent=2),
        encoding="utf-8",
    )
    (RELEASE / "predict.py").write_text(PREDICT_CODE, encoding="utf-8")
    (RELEASE / "requirements.txt").write_text(
        "joblib\nnumpy\npandas\nscikit-learn\n", encoding="utf-8"
    )
    card = f'''# BANKNIFTY fixed-ATM 30-minute tail-veto overlay (experimental)

## Goal

Increase the win rate and reduce tail losses of the 30-minute fixed-ATM short-strangle entry model without changing the strike between signal time and t+30 minutes.

## Construction

- Entry model: frozen Extra Trees classifier, probability threshold 0.55.
- Risk overlay: soft-voting tail-loss classifier (logistic, Extra Trees, histogram gradient boosting).
- Veto rule: reject a candidate when its tail-risk score is at or above the rolling 80th percentile of the prior 1,000 scores.
- Timing: the current score is appended only after the decision, so the threshold is backward-looking.
- Horizon: 30 minutes.
- Strike: signal-time ATM CE/PE strike, fixed through exit.

## Diagnostic comparison on 2026

- Base: 163 trades, 52.76% wins, +216.9 net points, 1.16 profit factor, -196.1 max drawdown.
- Overlay: 152 trades, 55.92% wins, +287.85 net points, 1.24 profit factor, -176.45 max drawdown.

## Leakage and validation status

The sequential veto calculation does not use future scores. However, 2026 had already been inspected before this overlay evaluation, so this comparison is **not a fresh holdout** and must not be treated as confirmed out-of-sample improvement. Validate once on a later untouched period before deployment.

## Files

- `model.joblib`: complete composite artifact and rolling-state seed.
- `entry_pipeline.joblib`, `tail_veto_pipeline.joblib`: fitted component pipelines.
- `preprocessor.joblib`, `scaler.joblib`, `estimators.joblib`: separated fitted components.
- `predict.py`: stateful sequential inference helper.
- `feature_manifest.json`, `scaler_metadata.json`: input and scaling documentation.
- `metrics.json`, trade CSVs, and `EVALUATION_REPORT.md`: diagnostic evidence.

Research-only; costs model two points of round-trip slippage and excludes some real-world charges/spread effects.
'''
    (RELEASE / "MODEL_CARD.md").write_text(card, encoding="utf-8")
    for name in [
        "evaluate_2026_tail_veto_overlay.py", "train_tail_risk_veto.py",
        "retrain_frozen_models_2026.py", "train_fixed_atm_horizons.py",
        "train_high_win_rate_models.py", "package_improved_30m_overlay.py",
    ]:
        shutil.copy2(ROOT / name, RELEASE / "training_code" / name)
    files = sorted(p for p in RELEASE.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    (RELEASE / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(path)}  {path.relative_to(RELEASE).as_posix()}" for path in files) + "\n",
        encoding="utf-8",
    )
    print(RELEASE)


if __name__ == "__main__":
    main()
