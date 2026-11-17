"""Minimal loader for this packaged model."""
from pathlib import Path
import joblib
import pandas as pd

HERE = Path(__file__).resolve().parent
BUNDLE = joblib.load(HERE / "model_bundle.joblib")

def predict(frame: pd.DataFrame):
    x = frame[BUNDLE["features"]]
    model = BUNDLE["model"]
    if "calibrator" not in BUNDLE:
        return model.predict(x)
    import numpy as np
    raw = model.predict_proba(x)[:, 1]
    eps = 1e-6
    score = np.log(np.clip(raw, eps, 1 - eps) / np.clip(1 - raw, eps, 1 - eps))
    return BUNDLE["calibrator"].predict_proba(score.reshape(-1, 1))[:, 1]
