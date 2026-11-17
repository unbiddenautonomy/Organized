"""Independent PCR paper-trading extension. No broker order API is used."""
# Until the pipelines move under the repository root (refactor step 9), make
# the shared `voltr` package importable when this folder is the entry point.
import sys as _sys
from pathlib import Path as _Path
_REPO = next(p for p in _Path(__file__).resolve().parents if (p / "voltr" / "__init__.py").is_file())
if str(_REPO) not in _sys.path:
    _sys.path.insert(0, str(_REPO))
