import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from pipelines.ivrv.paper_history import closed_paper_positions


class ClosedPaperPositionsTest(unittest.TestCase):
    def test_reads_all_exit_reasons_and_separate_realized_totals(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            logs = root / "logs"
            logs.mkdir()
            short = [
                {"event_type": "paper_trade_entry", "position_id": "short-1"},
                {"event_type": "paper_trade_exit", "position_id": "short-1",
                 "exit_datetime": "2026-11-17 10:00:00", "exit_reason": "profit_target",
                 "realized_pnl_inr": 1500.0},
                {"event_type": "paper_trade_exit", "position_id": "short-2",
                 "exit_datetime": "2026-11-17 12:00:00", "exit_reason": "stop_loss",
                 "realized_pnl_inr": -750.0},
            ]
            long = [{"event_type": "paper_trade_exit", "position_id": "long-1",
                     "exit_datetime": "2026-11-17 11:00:00", "exit_reason": "time_exit_120m",
                     "realized_pnl_inr": 200.0}]
            (logs / "paper_trade_events.jsonl").write_text(
                "\n".join(json.dumps(row) for row in short) + "\ninvalid-json\n", encoding="utf-8")
            (logs / "paper_trade_events_long.jsonl").write_text(
                "\n".join(json.dumps(row) for row in long), encoding="utf-8")
            result = closed_paper_positions(root, {})
            self.assertEqual(result["count"], 3)
            self.assertEqual([row["position_id"] for row in result["positions"]],
                             ["short-2", "long-1", "short-1"])
            self.assertEqual(result["realized_pnl_inr"],
                             {"SHORT_STRADDLE": 750.0, "LONG_STRADDLE": 200.0})


if __name__ == "__main__":
    unittest.main()
