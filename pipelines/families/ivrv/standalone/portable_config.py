"""Standalone-engine settings. Path and adapter helpers now live in voltr.core.machine."""

from __future__ import annotations

import json
import os
from pathlib import Path

# Until the pipelines move under the repository root (refactor step 9), make
# the shared `voltr` package importable when this folder is the entry point.
import sys as _sys
from pathlib import Path as _Path
_REPO = next(p for p in _Path(__file__).resolve().parents if (p / "voltr" / "__init__.py").is_file())
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))

from voltr.core.machine import local_ipv4_adapters, relative_to_config, resolve_interface  # noqa: E402,F401


def load_source_config(path: Path) -> dict:
    """Load standalone engine settings relative to their own JSON file."""
    path = path.resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if os.environ.get("VOLTR_DB_PASSWORD"):
        config["db_password"] = os.environ["VOLTR_DB_PASSWORD"]
    for key in ("contract_file", "spot_model_dir", "iv_model_dir", "pnl_model_dir", "hmm_model_dir",
                "expanded_hybrid_model_file", "rv120_model_file", "intraday_training_database",
                "intraday_artifact_dir", "eod_artifact_dir"):
        if config.get(key):
            config[key] = str(relative_to_config(config[key], path))
    config["interface_ip"] = resolve_interface(config.get("interface_ip"), environment="VOLTR_FO_INTERFACE")
    config["cm_interface_ip"] = resolve_interface(
        config.get("cm_interface_ip", config["interface_ip"]), environment="VOLTR_CM_INTERFACE"
    )
    return config
