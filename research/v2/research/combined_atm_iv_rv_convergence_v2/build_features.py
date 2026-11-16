from __future__ import annotations

import argparse
import json

from feature_pipeline import build_feature_table, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V2 combined call-put fixed-strike ATM-IV features")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()
    result = build_feature_table(load_config(args.config), output=args.output, max_days=args.max_days)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
