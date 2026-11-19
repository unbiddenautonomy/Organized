"""Read closed IV/RV paper positions without changing the trading engine."""

from __future__ import annotations

import json
from pathlib import Path


def closed_paper_positions(base_dir: Path, config: dict) -> dict:
    short_path = base_dir / str(config.get("paper_trade_events_log", "logs/paper_trade_events.jsonl"))
    long_path = short_path.with_name(f"{short_path.stem}_long{short_path.suffix}")
    positions: list[dict] = []
    realized = {"SHORT_STRADDLE": 0.0, "LONG_STRADDLE": 0.0}
    for side, path in (("SHORT_STRADDLE", short_path), ("LONG_STRADDLE", long_path)):
        if not path.is_file():
            continue
        with path.open(encoding="utf-8") as events:
            for line in events:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if event.get("event_type") != "paper_trade_exit":
                    continue
                row = dict(event)
                row["side"] = side
                positions.append(row)
                realized[side] += float(row.get("realized_pnl_inr") or 0.0)
    positions.sort(key=lambda row: (row.get("exit_datetime") or "", row.get("position_id") or ""), reverse=True)
    return {"positions": positions, "count": len(positions),
            "realized_pnl_inr": realized}
