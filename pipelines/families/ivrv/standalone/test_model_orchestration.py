import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from model_orchestration import ModelOrchestrator, ModelRegistry, SelectionPolicy
from model_orchestration.registry import ModelVersion


UTC = timezone.utc


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.registry = ModelRegistry(root / "registry.json")
        self.now = datetime(2026, 9, 11, 5, 0, tzinfo=UTC)
        for model_id, cadence in (("eod-v1", "eod"), ("intra-v1", "intraday")):
            self.registry.register(ModelVersion(model_id, cadence, f"models/{model_id}", "2026-09-10T10:00:00+00:00", self.now.isoformat()))
        policy = SelectionPolicy(min_observations=3, window_observations=10, min_relative_improvement=.02, switch_cooldown_minutes=0)
        self.service = ModelOrchestrator(self.registry, root / "scores.db", policy)

    def tearDown(self):
        self.temp.cleanup()

    def test_bootstraps_eod_then_promotes_better_paired_candidate(self):
        report = self.service.evaluate("spot", 5, self.now)
        self.assertEqual(report["champion"], "eod-v1")
        for index, actual in enumerate((101.0, 102.0, 103.0)):
            stamp = self.now + timedelta(minutes=index * 10)
            self.service.record_prediction("eod-v1", "spot", 5, stamp, 100, actual + 5)
            self.service.record_prediction("intra-v1", "spot", 5, stamp, 100, actual + .1)
            self.assertEqual(self.service.settle("spot", stamp + timedelta(minutes=5), actual), 2)
        report = self.service.evaluate("spot", 5, self.now + timedelta(minutes=40))
        self.assertEqual(report["action"], "promote")
        self.assertEqual(self.service.route("spot", 5), "intra-v1")

    def test_unmatured_forecast_is_not_scored(self):
        self.service.evaluate("iv", 15, self.now)
        self.service.record_prediction("eod-v1", "iv", 15, self.now, .2, .19)
        self.assertEqual(self.service.settle("iv", self.now + timedelta(minutes=14), .18), 0)


if __name__ == "__main__":
    unittest.main()
