"""Legacy engine settings: paths resolve against config.json; the DB password may come from VOLTR_DB_PASSWORD."""

from __future__ import annotations

import os
from pathlib import Path

from voltr.core.machine import load_engine_config

PATH_KEYS = ("contract_file", "spot_model_dir", "iv_model_dir", "pnl_model_dir", "hmm_model_dir",
             "expanded_hybrid_model_file", "rv120_model_file", "intraday_training_database",
             "intraday_artifact_dir", "eod_artifact_dir")


def load_source_config(path: Path) -> dict:
    """Load the legacy engine settings relative to their own JSON file."""
    config = load_engine_config(path, PATH_KEYS)
    if os.environ.get("VOLTR_DB_PASSWORD"):
        config["db_password"] = os.environ["VOLTR_DB_PASSWORD"]
    return config
