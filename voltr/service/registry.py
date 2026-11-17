"""Which pipelines the service runs and where their code lives: read from config/service.json.

    "host":      module providing the session config loader and the main dashboard
    "pipelines": {name: {"adapter": "package.module:Class", "lock": path, "status_file": path?}}

Paths are relative to the repository root. Adding a pipeline means adding one
entry here (see docs/ADDING_A_PIPELINE.md); nothing in voltr changes.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]


def service_settings(live_config_path: str | Path) -> dict[str, Any]:
    """The service.json next to the session config, or {} when absent."""
    path = Path(live_config_path).resolve().with_name("service.json")
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def import_object(spec: str) -> Any:
    """Import "package.module:attribute" (or just a module)."""
    module, _, attribute = spec.partition(":")
    loaded = importlib.import_module(module)
    return getattr(loaded, attribute) if attribute else loaded


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def host(settings: dict[str, Any]) -> Any:
    return import_object(settings["host"])


def pipelines(settings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return dict(settings.get("pipelines", {}))
