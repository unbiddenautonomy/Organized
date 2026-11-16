from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
PREDICTIONS = ROOT / "analysis_outputs" / "three_iv_rv_conditions" / "three_rule_predictions.csv.gz"
OUTPUT = ROOT / "analysis_outputs" / "stress_test_2026-10-27" / "replay_prices_exact.csv.gz"


def main() -> None:
    sys.path.insert(0, str(ROOT))
    import backtest_user_csv_current_paper_rules as source

    predictions = pd.read_csv(PREDICTIONS, parse_dates=["datetime", "expiry"], low_memory=False)
    predictions = predictions.loc[predictions["base_eligible"].astype(bool)].copy()
    predictions["Date"] = predictions["datetime"].dt.strftime("%Y-%m-%d")
    prices = source.load_relevant_prices(predictions)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(OUTPUT, index=False, compression="gzip")
    print({"rows": len(prices), "output": str(OUTPUT)})


if __name__ == "__main__":
    main()
