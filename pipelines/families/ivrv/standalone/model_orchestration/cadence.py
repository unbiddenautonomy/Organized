from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path


@dataclass(frozen=True)
class TrainingJob:
    cadence: str
    command: tuple[str, ...]
    working_directory: Path

    def run(self) -> subprocess.CompletedProcess[str]:
        if self.cadence not in {"intraday", "eod"}:
            raise ValueError("unknown cadence")
        return subprocess.run(self.command, cwd=self.working_directory, text=True, capture_output=True, check=False)


def is_due(cadence: str, now: datetime, last_success: datetime | None, intraday_minutes: int = 30, eod_after: time = time(15, 45)) -> bool:
    if last_success is None:
        return True
    if cadence == "intraday":
        return (now - last_success).total_seconds() >= intraday_minutes * 60
    if cadence == "eod":
        return now.time() >= eod_after and last_success.date() < now.date()
    raise ValueError("unknown cadence")
