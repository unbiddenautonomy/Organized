from __future__ import annotations

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
