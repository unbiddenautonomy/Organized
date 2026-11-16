"""Reference rerun or recorded-input replay, always in a separate output folder."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    # Reference mode reproduces the earlier study including historical charges,
    # sparse source minutes and flagged synthetic exits. No reoptimization.
    config = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    raw_path = Path(os.environ.get('VOLTR_PCR_RAW_CACHE', config["reference_raw_cache"])).expanduser()
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    if not raw_path.is_file():
        raise FileNotFoundError(f"Historical raw option cache unavailable: {raw_path}")
    sys.path.insert(0, str(ROOT / "research"))
    import run_target_3000_stop_1500_max3 as runner
    runner.INPUT = ROOT / "reference"
    runner.OUTPUT = args.output
    runner.engine.RAW_CACHE = raw_path
    runner.main()


if __name__ == "__main__":
    main()
