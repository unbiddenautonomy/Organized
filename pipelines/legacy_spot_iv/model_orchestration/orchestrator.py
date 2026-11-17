from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .policy import SelectionPolicy, metrics
from .registry import ModelRegistry, parse_time, utc_now


UTC = timezone.utc


class ModelOrchestrator:
    """Records shadow forecasts, settles labels, and selects a safe champion."""

    def __init__(self, registry: ModelRegistry, database: str | Path, policy: SelectionPolicy | None = None):
        self.registry = registry
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or SelectionPolicy()
        self.policy.validate()
        with closing(self._connect()) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS forecasts (
                model_id TEXT NOT NULL, target TEXT NOT NULL, horizon INTEGER NOT NULL,
                prediction_time TEXT NOT NULL, due_time TEXT NOT NULL, current_value REAL NOT NULL,
                prediction REAL NOT NULL, actual REAL, settled_at TEXT,
                PRIMARY KEY(model_id, target, horizon, prediction_time))""")
            db.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database)
        connection.row_factory = sqlite3.Row
        return connection

    def record_prediction(self, model_id: str, target: str, horizon: int, prediction_time: datetime, current_value: float, prediction: float) -> None:
        self.registry.model(model_id)
        stamp = prediction_time.astimezone(UTC)
        due = stamp + timedelta(minutes=horizon)
        with closing(self._connect()) as db:
            db.execute("INSERT OR IGNORE INTO forecasts VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)",
                       (model_id, target, horizon, stamp.isoformat(), due.isoformat(), float(current_value), float(prediction)))
            db.commit()

    def settle(self, target: str, observed_at: datetime, actual: float, tolerance_seconds: int = 30) -> int:
        """Settle only forecasts whose due timestamp matches this observation."""
        stamp = observed_at.astimezone(UTC)
        lower = (stamp - timedelta(seconds=tolerance_seconds)).isoformat()
        upper = (stamp + timedelta(seconds=tolerance_seconds)).isoformat()
        with closing(self._connect()) as db:
            result = db.execute("""UPDATE forecasts SET actual=?, settled_at=?
                WHERE target=? AND actual IS NULL AND due_time BETWEEN ? AND ?""",
                (float(actual), stamp.isoformat(), target, lower, upper))
            count = result.rowcount
            db.commit()
            return count

    def _paired_rows(self, target: str, horizon: int, left: str, right: str) -> list[sqlite3.Row]:
        with closing(self._connect()) as db:
            return db.execute("""SELECT a.prediction_time, a.current_value, a.prediction AS left_prediction,
                    b.prediction AS right_prediction, a.actual
                FROM forecasts a JOIN forecasts b
                  ON a.target=b.target AND a.horizon=b.horizon AND a.prediction_time=b.prediction_time
                WHERE a.target=? AND a.horizon=? AND a.model_id=? AND b.model_id=?
                  AND a.actual IS NOT NULL AND b.actual IS NOT NULL
                ORDER BY a.prediction_time DESC LIMIT ?""",
                (target, horizon, left, right, self.policy.window_observations)).fetchall()

    @staticmethod
    def _score(rows: list[sqlite3.Row], side: str) -> dict[str, float | int]:
        errors = [float(row[f"{side}_prediction"]) - float(row["actual"]) for row in rows]
        directions = [
            (float(row[f"{side}_prediction"]) - float(row["current_value"]))
            * (float(row["actual"]) - float(row["current_value"])) > 0
            for row in rows
        ]
        return metrics(errors, directions)

    def evaluate(self, target: str, horizon: int, now: datetime | None = None) -> dict:
        now = (now or utc_now()).astimezone(UTC)
        candidates = [model for model in self.registry.models() if model.status == "ready"]
        fresh = [model for model in candidates if now - parse_time(model.registered_at) <= timedelta(minutes=self.policy.max_model_age_minutes)]
        champion_id = self.registry.champion(target, horizon)
        if not fresh:
            return {"action": "hold", "reason": "no_fresh_models", "champion": champion_id}
        if champion_id is None or champion_id not in {model.model_id for model in fresh}:
            preferred = next((m for m in fresh if m.cadence == "eod"), fresh[0])
            self.registry.promote(target, horizon, preferred.model_id, {"reason": "bootstrap_or_stale_champion"}, now)
            return {"action": "promote", "champion": preferred.model_id, "reason": "bootstrap_or_stale_champion"}

        last_switch = self.registry.last_switch_at(target, horizon)
        if last_switch and now - last_switch < timedelta(minutes=self.policy.switch_cooldown_minutes):
            return {"action": "hold", "champion": champion_id, "reason": "cooldown"}

        best_id, best_report = champion_id, None
        for model in fresh:
            if model.model_id == champion_id:
                continue
            rows = self._paired_rows(target, horizon, champion_id, model.model_id)
            if len(rows) < self.policy.min_observations:
                continue
            champion_metrics = self._score(rows, "left")
            candidate_metrics = self._score(rows, "right")
            relative = (champion_metrics["mae"] - candidate_metrics["mae"]) / max(champion_metrics["mae"], 1e-12)
            direction_gain = candidate_metrics["direction_accuracy"] - champion_metrics["direction_accuracy"]
            if relative >= self.policy.min_relative_improvement and direction_gain >= self.policy.min_direction_improvement:
                if best_report is None or relative > best_report["relative_mae_improvement"]:
                    best_id, best_report = model.model_id, {
                        "paired_observations": len(rows), "relative_mae_improvement": relative,
                        "direction_accuracy_gain": direction_gain, "champion_metrics": champion_metrics,
                        "candidate_metrics": candidate_metrics,
                    }
        if best_id != champion_id:
            self.registry.promote(target, horizon, best_id, best_report, now)
            return {"action": "promote", "champion": best_id, "previous": champion_id, "evidence": best_report}
        return {"action": "hold", "champion": champion_id, "reason": "insufficient_superiority"}

    def route(self, target: str, horizon: int) -> str:
        champion = self.registry.champion(target, horizon)
        if champion is None:
            raise RuntimeError(f"no champion for {target}:{horizon}; call evaluate first")
        return champion
