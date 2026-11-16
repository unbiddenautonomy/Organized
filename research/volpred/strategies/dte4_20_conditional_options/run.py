"""Single entry point for validating or rebuilding this strategy package."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(relative: str) -> None:
    subprocess.run([sys.executable, str(ROOT / relative)], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--research", action="store_true")
    args = parser.parse_args()
    if not args.validate and not args.research:
        parser.error("choose --validate or --research")
    if args.research:
        run("research/dte4_20_structure_research.py")
        run("research/dte4_20_multi_structure_selector.py")
    if args.validate:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)
        for file in ("app/trade_puncher.py", "research/dte4_20_structure_research.py",
                     "research/dte4_20_multi_structure_selector.py"):
            subprocess.run([sys.executable, "-m", "py_compile", str(ROOT / file)], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()

