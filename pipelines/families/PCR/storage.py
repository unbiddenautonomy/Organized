"""Atomic snapshots and append-only journal owned exclusively by PCR."""
import json
import math
import os
import time
import uuid
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, "item"):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def read_text(path):
    for attempt in range(8):
        try:
            return Path(path).read_text(encoding="utf-8")
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.005 * 2**attempt)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(clean(value), allow_nan=False, indent=2), encoding="utf-8")
    # Windows briefly denies replacement while a dashboard reader has the
    # destination open. Preserve atomicity and retry that transient condition.
    for attempt in range(8):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(.005 * 2**attempt)


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(clean(value), allow_nan=False) + "\n")
        handle.flush()


def read_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    for line in read_text(path).splitlines():
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # Interrupted final journal write, never invent an event.
    return result
