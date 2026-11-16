"""Export and verify the committed portable source snapshot.

The archive contains exactly the tracked files at HEAD. Local data links,
environments, logs, credentials and generated results stay outside the ZIP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT.parent / "VolTr-source-refactored.zip"


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha256(stream) -> str:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    try:
        inside_repository = os.path.commonpath((ROOT, output)) == str(ROOT)
    except ValueError:  # A different Windows drive is necessarily outside.
        inside_repository = False
    if inside_repository:
        parser.error("place the archive outside the repository")
    if output.exists():
        parser.error(f"output exists; choose another path: {output}")
    if git("status", "--porcelain=v1", "--untracked-files=normal").strip():
        parser.error("commit or remove untracked source files before exporting HEAD")

    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "archive", "--format=zip", "--output", str(output), "HEAD"],
        cwd=ROOT, check=True,
    )
    tracked = {name.decode("utf-8") for name in git("ls-files", "-z").split(b"\0") if name}
    with zipfile.ZipFile(output) as archive:
        archived = {name for name in archive.namelist() if not name.endswith("/")}
        if archived != tracked:
            raise RuntimeError(f"archive mismatch: missing={tracked - archived}, extra={archived - tracked}")
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP CRC mismatch: {bad}")
        models = json.loads(archive.read("setup/model_manifest.json"))["files"]
        for name, expected in models.items():
            with archive.open(name) as model_file:
                actual = sha256(model_file)
            if actual.lower() != expected.lower():
                raise RuntimeError(f"model hash mismatch: {name}")

    with output.open("rb") as source:
        digest = sha256(source)
    checksum_file = output.with_suffix(output.suffix + ".sha256.txt")
    checksum_file.write_text(f"{digest}  {output.name}\n", encoding="ascii")
    print(json.dumps({
        "archive": str(output),
        "checksum_file": str(checksum_file),
        "commit": git("rev-parse", "--short", "HEAD").decode().strip(),
        "files": len(tracked),
        "model_hashes_verified": len(models),
        "bytes": output.stat().st_size,
        "sha256": digest,
    }, indent=2))


if __name__ == "__main__":
    main()
