from __future__ import annotations

import argparse
import json

from fixed_strike_iv.config import load_config
from fixed_strike_iv.pipeline import attach_exact_exits, build_entries, validate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build exact fixed-entry-strike 120-minute IV labels")
    parser.add_argument("stage", choices=["entries", "exits", "validate", "all"])
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--start-date", default=None, help="Optional YYYY-MM-DD entry cutoff, useful for smoke tests")
    parser.add_argument("--max-days", type=int, default=None, help="Optional number of entry dates")
    args = parser.parse_args()
    config = load_config(args.config)
    results = {}
    if args.stage in {"entries", "all"}:
        results["entries"] = build_entries(config, start_date=args.start_date, max_days=args.max_days)
    if args.stage in {"exits", "all"}:
        results["exits"] = attach_exact_exits(config)
    if args.stage in {"validate", "all"}:
        results["validate"] = validate_dataset(config)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()

