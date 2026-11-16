from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from regime_understanding import (
    SESSION_CLOSE_MINUTE,
    assign_regimes,
    black_greeks,
    regime_episodes,
)


def test_black_greeks_are_finite_for_atm_option():
    values = black_greeks(
        np.array([45000.0]), np.array([45000.0]), np.array([10 / 365]),
        np.array([0.20]), np.array([45000.0]), True,
    )
    assert all(np.isfinite(value[0]) for value in values.values())
    assert values["gamma"][0] > 0
    assert values["vega"][0] > 0
    assert values["theta"][0] < 0


def test_regime_episode_breaks_on_state_and_session():
    times = pd.to_datetime([
        "2022-01-03 09:15", "2022-01-03 09:16", "2022-01-03 09:17",
        "2022-01-04 09:15",
    ])
    frame = pd.DataFrame({
        "datetime": times,
        "date": times.normalize(),
        "expiry": pd.to_datetime(["2022-01-27"] * 4),
        "quality_ok": [True] * 4,
    })
    thresholds = {}
    for greek in ("gamma", "vega", "theta", "vanna", "volga", "charm"):
        frame[greek] = [0.0, 0.0, 2.0, 2.0]
        thresholds[greek] = {"low_max": 0.5, "high_min": 1.5}
    episodes = regime_episodes(assign_regimes(frame, thresholds))
    gamma = episodes[episodes["greek"].eq("gamma")].reset_index(drop=True)
    assert gamma["regime"].tolist() == ["low", "high", "high"]
    assert gamma["duration_minutes"].tolist() == [2.0, 1.0, 1.0]


def test_latest_intraday_entry_depends_on_horizon():
    close = pd.Timestamp("2022-01-03") + pd.Timedelta(minutes=SESSION_CLOSE_MINUTE)
    expected = {15: "15:15", 30: "15:00", 60: "14:30", 120: "13:30", 180: "12:30"}
    for horizon, clock in expected.items():
        assert (close - pd.Timedelta(minutes=horizon)).strftime("%H:%M") == clock
