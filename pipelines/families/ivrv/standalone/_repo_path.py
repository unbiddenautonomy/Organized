"""Temporary: make the shared `voltr` package importable from this folder.

Removed in refactor step 10, when this engine moves under pipelines/ and runs
from the repository root.
"""
import sys
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "voltr" / "__init__.py").is_file())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
