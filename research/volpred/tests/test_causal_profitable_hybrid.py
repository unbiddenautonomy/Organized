import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))

import causal_profitable_hybrid as strategy


def test_walk_forward_training_labels_are_strictly_prior():
    dates = pd.bdate_range("2020-01-02", periods=150)
    frame = pd.DataFrame({
        "datetime": dates + pd.Timedelta(hours=9, minutes=45),
        "exit_datetime": dates + pd.Timedelta(hours=15, minutes=20),
        "date": dates.normalize(),
        "dte": 10.0,
        "net_return": np.sin(np.arange(len(dates)) / 7.0) / 100.0,
    })
    for index, feature in enumerate(strategy.FEATURES):
        if feature == "dte":
            continue
        frame[feature] = np.cos(np.arange(len(dates)) / (index + 2.0))

    scored = strategy.model_walk_forward(frame, "ridge10", 0.60)

    assert not scored.empty
    assert (
        pd.to_datetime(scored["max_training_exit_datetime"])
        < scored["datetime"]
    ).all()
    assert scored["train_rows"].ge(strategy.MIN_TRAIN).all()
