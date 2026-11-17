import json
import tempfile
import unittest
from pathlib import Path

from pipelines.legacy_spot_iv.live_retraining import LiveRetrainingManager


class LiveRetrainingTests(unittest.TestCase):
    def test_matured_udp_derived_rows_publish_and_shadow_predict(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            config={"intraday_retraining_enabled":True,"intraday_retraining_minutes":120,
                    "intraday_retraining_minimum_rows":5,"intraday_retraining_validation_fraction":.25,
                    "intraday_retraining_minimum_improvement":-10.0,
                    "intraday_training_database":"training.db","intraday_artifact_dir":"models",
                    "intraday_shadow_prediction_file":"shadow.jsonl"}
            manager=LiveRetrainingManager(config,root)
            try:
                base=30_000_000
                for offset in range(12):
                    minute=base+offset
                    event={"exchange_minute":f"2026-09-11 10:{offset:02d}:00","current_spot":100+offset,
                           "predictions":{"5":{"predicted_spot":106+offset}}}
                    # Use a stable artificial minute for deterministic due-time settlement.
                    manager._minute=lambda _event,m=minute:m
                    manager.on_spot_prediction(event)
                if manager.worker: manager.worker.join(timeout=5)
                manager._train_all(base+11)
                self.assertIn(("spot",5),manager.bundles)
                manager._write_shadow("spot",5,"2026-09-11 10:12:00",base+12,112,118)
            finally:
                manager.close()
            rows=[json.loads(line) for line in (root/"shadow.jsonl").read_text().splitlines()]
            self.assertTrue(rows)
            self.assertTrue((root/"models"/"CURRENT").exists())

    def test_eod_models_publish_separately(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); config={"intraday_retraining_enabled":True,"intraday_retraining_minimum_rows":3,
                "eod_retraining_enabled":True,"eod_retraining_minimum_rows":3,"intraday_retraining_minimum_improvement":-10,
                "intraday_training_database":"training.db","intraday_artifact_dir":"intraday","eod_artifact_dir":"eod",
                "intraday_shadow_prediction_file":"shadow.jsonl","retraining_live_output_file":"outputs.jsonl"}
            manager=LiveRetrainingManager(config,root)
            try:
                base=40_000_000
                for offset in range(10):
                    manager._minute=lambda _event,m=base+offset:m
                    manager.on_iv_prediction({"exchange_minute":f"2026-09-11 10:{offset:02d}:00","current_iv":.10+offset/1000,
                                              "predictions":{"5":{"predicted_iv":.11+offset/1000}}})
                if manager.worker: manager.worker.join(timeout=5)
                self.assertTrue(manager._train_all(base+9,"eod"))
                self.assertIn(("iv",5),manager.eod_bundles)
                self.assertTrue((root/"eod"/"CURRENT").exists())
            finally: manager.close()


if __name__ == "__main__":
    unittest.main()
