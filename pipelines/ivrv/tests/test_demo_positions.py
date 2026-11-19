from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import csv
from io import StringIO
import unittest

from pipelines.ivrv.demo_positions import DemoPositionLedger


class DemoLedgerTest(unittest.TestCase):
    def test_skipped_position_drawdown_horizon_restart_and_csv(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            ledger = DemoPositionLedger(path, 30)
            entry = datetime(2026, 11, 17, 11, 0)
            args = dict(signal_time=entry, entry_time=entry, expiry="2026-11-23",
                        strike=50000, probability=0.8, threshold=0.7,
                        paper_skip_reason="maximum_open_positions_reached",
                        call_token=1, put_token=2, call_price=200, put_price=200,
                        target=1500, stop=-750)
            self.assertFalse(ledger.consider(**args, paper_entered=True))
            self.assertFalse(ledger.consider(**{**args, "probability": 0.7}, paper_entered=False))
            self.assertTrue(ledger.consider(**args, paper_entered=False))
            self.assertFalse(ledger.consider(**args, paper_entered=False))

            def mark(minutes, leg_price):
                now = entry + timedelta(minutes=minutes)
                second = minutes * 60
                ledger.mark(now, {1: (leg_price, second), 2: (leg_price, second)}, second)

            mark(10, 195)   # +300
            mark(20, 205)   # -300, drawdown -600 from peak
            self.assertEqual(ledger.snapshot()["positions"][0]["max_drawdown_inr"], -600)
            ledger = DemoPositionLedger(path, 30)
            mark(120, 197.5)  # +150 at the two-hour mark
            row = ledger.snapshot()["positions"][0]
            self.assertEqual(row["status"], "CONVERGED")
            self.assertEqual(row["horizon_pnl_inr"], 150)
            self.assertEqual(row["max_drawdown_inr"], -600)
            parsed = list(csv.DictReader(StringIO(ledger.csv_bytes().decode("utf-8-sig"))))
            self.assertEqual(parsed[0]["max_drawdown_inr"], "-600.0")
            self.assertEqual(parsed[0]["converged"], "True")
            self.assertTrue(ledger.save_session_csv("2026-11-17").is_file())

    def test_session_end_does_not_claim_convergence(self):
        with TemporaryDirectory() as directory:
            ledger = DemoPositionLedger(Path(directory) / "demo.json", 30)
            entry = datetime(2026, 11, 17, 14, 0)
            ledger.consider(signal_time=entry, entry_time=entry, expiry="2026-11-23",
                            strike=50000, probability=0.8, threshold=0.7,
                            paper_entered=False, paper_skip_reason="slots_full",
                            call_token=1, put_token=2, call_price=200, put_price=200,
                            target=1500, stop=-750)
            ledger.close_session(datetime(2026, 11, 17, 15, 15))
            row = ledger.snapshot()["positions"][0]
            self.assertEqual(row["status"], "HORIZON_UNAVAILABLE")
            self.assertIsNone(row["converged"])

    def test_long_demo_uses_long_pnl_direction(self):
        with TemporaryDirectory() as directory:
            ledger = DemoPositionLedger(Path(directory) / "demo.json", 30)
            entry = datetime(2026, 11, 17, 10, 0)
            ledger.consider(signal_time=entry, entry_time=entry, expiry="2026-11-23",
                            strike=50000, probability=0.8, threshold=0.7,
                            paper_entered=False, paper_skip_reason="slots_full",
                            call_token=1, put_token=2, call_price=200, put_price=200,
                            target=1500, stop=-750, side="LONG_STRADDLE")
            ledger.mark(entry + timedelta(hours=2), {1: (210, 7200), 2: (210, 7200)}, 7200)
            row = ledger.snapshot()["positions"][0]
            self.assertEqual(row["horizon_pnl_inr"], 600)
            self.assertTrue(row["converged"])


if __name__ == "__main__":
    unittest.main()
