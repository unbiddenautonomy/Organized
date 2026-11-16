from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class SelectionPolicy:
    """Conservative paired-sample champion selection policy."""

    min_observations: int = 100
    window_observations: int = 500
    min_relative_improvement: float = 0.02
    min_direction_improvement: float = 0.0
    switch_cooldown_minutes: int = 60
    max_model_age_minutes: int = 1_440

    def validate(self) -> None:
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")
        if self.window_observations < self.min_observations:
            raise ValueError("window_observations must be >= min_observations")
        if not 0 <= self.min_relative_improvement < 1:
            raise ValueError("min_relative_improvement must be in [0, 1)")


def metrics(errors: list[float], directions: list[bool]) -> dict[str, float | int]:
    count = len(errors)
    return {
        "observations": count,
        "mae": sum(abs(value) for value in errors) / count if count else float("nan"),
        "rmse": sqrt(sum(value * value for value in errors) / count) if count else float("nan"),
        "direction_accuracy": sum(directions) / count if count else float("nan"),
    }
