"""Crash-safe JSON files, JSON-lines journals and single-writer file locks.

Windows briefly refuses to replace or read a file while a dashboard reader
holds it open, so replacements and reads retry that transient PermissionError.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid

_RETRIES = 8


def _retry_delay(attempt: int) -> float:
    return min(0.01 * 2 ** attempt, 0.2)


def json_clean(value: Any) -> Any:
    """Make numpy scalars, dates and non-finite floats JSON-safe (NaN/inf become None)."""
    if isinstance(value, dict):
        return {str(k): json_clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clean(v) for v in value]
    if hasattr(value, "item"):
        return json_clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def atomic_write_json(path: str | Path, value: Any, *, prepare: Callable[[Any], Any] | None = None,
                      trailing_newline: bool = False, fsync: bool = False) -> None:
    """Replace `path` with a whole JSON document (indent 2), or leave it untouched on failure.

    `prepare` transforms the value first (for example json_clean).
    `trailing_newline` and `fsync` keep the exact bytes and durability some
    callers' files already have.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(prepare(value) if prepare else value, handle, indent=2, allow_nan=False)
            if trailing_newline:
                handle.write("\n")
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        for attempt in range(_RETRIES):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == _RETRIES - 1:
                    raise
                time.sleep(_retry_delay(attempt))
    finally:
        if temporary.exists():
            temporary.unlink()


def read_text(path: str | Path) -> str:
    """Read UTF-8 text, retrying a transient Windows sharing violation."""
    for attempt in range(_RETRIES):
        try:
            return Path(path).read_text(encoding="utf-8")
        except PermissionError:
            if attempt == _RETRIES - 1:
                raise
            time.sleep(_retry_delay(attempt))
    raise AssertionError("unreachable")


def read_json(path: str | Path, default: Any = None) -> Any:
    """Parsed JSON, or `default` when the file is missing or unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def append_jsonl(path: str | Path, value: Any, *, prepare: Callable[[Any], Any] | None = None) -> None:
    """Append one compact JSON line and flush it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(prepare(value) if prepare else value, allow_nan=False) + "\n")
        handle.flush()


def read_jsonl(path: str | Path) -> list[Any]:
    """All complete records; a torn final line from an interrupted write is skipped."""
    path = Path(path)
    if not path.exists():
        return []
    records = []
    for line in read_text(path).splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@contextmanager
def file_lock(path: str | Path, *, blocking: bool = False):
    """Hold an OS lock on `path` (created if needed) for the `with` block.

    Non-blocking by default: a lock already held by another process raises OSError.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        handle.seek(0)
        if not handle.read(1):
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
