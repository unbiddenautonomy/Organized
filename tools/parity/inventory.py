"""List every gitignored file in the repo with its size and SHA-256.

Git does not carry these files (contract masters, research artifacts,
backtest replays, datasets), so every refactor step is checked against this
list: the same contents must still exist, at the same path unless the step
says it moves them. Python caches are left out.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SKIP = ("__pycache__/", ".pytest_cache/", ".venv/", "runtime/parity")


def ignored_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "--others", "--ignored", "--exclude-standard", "-z"],
                         cwd=REPO, capture_output=True, check=True).stdout.decode("utf-8")
    return sorted(p for p in out.split("\0") if p and not any(s in p for s in SKIP) and not p.endswith(".pyc"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build() -> dict:
    files = {p: {"bytes": (REPO / p).stat().st_size, "sha256": sha256(REPO / p)} for p in ignored_files()}
    return {"count": len(files), "bytes": sum(f["bytes"] for f in files.values()), "files": files}


if __name__ == "__main__":
    data = build()
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    text = json.dumps(data, indent=1, sort_keys=True)
    if target:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    print(f"{data['count']} ignored files, {data['bytes'] / 2**20:.1f} MiB")
