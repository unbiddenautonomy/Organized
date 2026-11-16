from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile


UTC = timezone.utc


def utc_now() -> datetime:
    return datetime.now(UTC)


def parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


@dataclass(frozen=True)
class ModelVersion:
    model_id: str
    cadence: str
    artifact_uri: str
    trained_through: str
    registered_at: str
    status: str = "ready"


class ModelRegistry:
    """Small JSON registry with atomic writes; replace with a DB at multi-host scale."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"version": 1, "models": {}, "champions": {}, "switches": []})

    def _read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, self.path)

    def register(self, model: ModelVersion) -> None:
        if model.cadence not in {"intraday", "eod"}:
            raise ValueError("cadence must be intraday or eod")
        state = self._read()
        state["models"][model.model_id] = asdict(model)
        self._write(state)

    def model(self, model_id: str) -> ModelVersion:
        return ModelVersion(**self._read()["models"][model_id])

    def models(self) -> list[ModelVersion]:
        return [ModelVersion(**value) for value in self._read()["models"].values()]

    def champion(self, target: str, horizon: int) -> str | None:
        return self._read()["champions"].get(f"{target}:{horizon}")

    def promote(self, target: str, horizon: int, model_id: str, reason: dict, at: datetime | None = None) -> None:
        state = self._read()
        if model_id not in state["models"]:
            raise KeyError(model_id)
        key = f"{target}:{horizon}"
        previous = state["champions"].get(key)
        if previous == model_id:
            return
        timestamp = (at or utc_now()).astimezone(UTC).isoformat()
        state["champions"][key] = model_id
        state["switches"].append({"target": target, "horizon": horizon, "from": previous, "to": model_id, "at": timestamp, "reason": reason})
        self._write(state)

    def last_switch_at(self, target: str, horizon: int) -> datetime | None:
        matching = [row for row in self._read()["switches"] if row["target"] == target and row["horizon"] == horizon]
        return parse_time(matching[-1]["at"]) if matching else None
