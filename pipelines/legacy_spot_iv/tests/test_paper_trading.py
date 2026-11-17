import unittest

import numpy as np

from voltr.market.stats import nan_window_stats as _nan_window_stats
from pipelines.legacy_spot_iv.paper_trading.engine import PaperTradingEngine, bep_stress_margin_per_lot, combined_short_mtm


class PaperRiskFormulaTest(unittest.TestCase):
    def test_nan_window_stats_keeps_empty_columns_without_runtime_warning(self):
        mean, std = _nan_window_stats(np.asarray([[1.0, np.nan], [3.0, np.nan]]))
        self.assertEqual((mean[0], std[0]), (2.0, 1.0))
        self.assertTrue(np.isnan(mean[1]))
        self.assertTrue(np.isnan(std[1]))

    def test_combined_short_mtm_uses_entry_credit_and_exit_debit(self):
        self.assertEqual(combined_short_mtm(500.0, 400.0, 30, 2), 6000.0)
        self.assertEqual(combined_short_mtm(500.0, 650.0, 30, 1), -4500.0)

    def test_bep_and_margin_reserve(self):
        lower, upper, stress_loss, margin = bep_stress_margin_per_lot(
            spot=48000, strike=48000, entry_credit=500, lot_size=30,
            stress_pct=0.02, configured_margin=10000, buffer=1.25,
        )
        self.assertEqual((lower, upper), (47500, 48500))
        self.assertEqual(stress_loss, 13800)
        self.assertEqual(margin, 17250)

    def test_forecast_is_paired_with_actual_at_target_minute(self):
        rows = []
        keys = set()
        PaperTradingEngine._record_forecasts(
            rows, keys, 100, 48000.0, {"5": {"predicted_spot": 48100.0}}, "predicted_spot"
        )
        PaperTradingEngine._settle_forecasts(rows, 104, 48080.0)
        self.assertEqual(rows[0]["status"], "pending")
        PaperTradingEngine._settle_forecasts(rows, 105, 48090.0)
        self.assertEqual(rows[0]["status"], "matured")
        self.assertEqual(rows[0]["error"], 10.0)

    def test_selects_best_supported_horizon_that_fits_before_close(self):
        class FakeAdapter:
            def score(self, _spot, _iv):
                return {str(h): {"expected_pnl": float(h * 10), "profit_probability": .7,
                    "pnl_q10": -100.0, "mae_q90": 500.0, "hmm_in_distribution": True}
                    for h in (5, 15, 30, 45, 60)}
        engine = object.__new__(PaperTradingEngine)
        engine.config = {"paper_max_predicted_iv_change": 0.00025, "paper_force_exit_minute": 915}
        engine.horizons = (5, 15, 30, 45, 60)
        engine.pnl_adapter = FakeAdapter()
        spot = {"predictions": {str(h): {"predicted_spot": 48000 + h} for h in engine.horizons}}
        iv = {"predictions": {
            "5": {"predicted_iv_change": 0.0}, "15": {"predicted_iv_change": -0.001},
            "30": {"predicted_iv_change": -0.002}, "45": {"predicted_iv_change": -0.001},
            "60": {"predicted_iv_change": -0.004},
        }}
        engine.latest_pnl_forecasts = engine.pnl_adapter.score(spot, iv)
        # 30m has the best model score among horizons that still fit before close.
        selected = engine._select_horizon(spot, iv, 885)
        self.assertIsNotNone(selected)
        self.assertEqual(selected[0], 30)

    def test_ten_minute_confirmation_interpolates_five_and_fifteen(self):
        event = {"predictions": {
            "5": {"predicted_spot": 48050.0},
            "15": {"predicted_spot": 48150.0},
        }}
        self.assertEqual(PaperTradingEngine._interpolated_ten_minute(event, "predicted_spot"), 48100.0)

    def test_entry_waits_for_exact_spot_iv_timestamp_pair(self):
        engine = object.__new__(PaperTradingEngine)
        engine.enabled = True
        engine.position = None
        engine.latest_spot_prediction = {"exchange_minute": "2026-09-07 14:36:00"}
        engine.latest_iv_prediction = {"exchange_minute": "2026-09-07 14:35:00"}
        engine._consider_entry()  # must return before reading any trade/risk state

    def test_stacked_forecast_updates_once_for_each_synchronized_pair(self):
        class FakeAdapter:
            def __init__(self): self.calls = 0
            def score(self, _spot, _iv):
                self.calls += 1
                return {"5": {"hmm_state": 1}}
        engine = object.__new__(PaperTradingEngine)
        engine.pnl_adapter = FakeAdapter()
        engine.latest_stacked_minute = None
        engine.latest_pnl_forecasts = {}
        engine.latest_spot_prediction = {"exchange_minute": "2026-09-07 14:36:00"}
        engine.latest_iv_prediction = {"exchange_minute": "2026-09-07 14:35:00"}
        engine._update_stacked_forecasts()
        self.assertEqual(engine.pnl_adapter.calls, 0)
        engine.latest_iv_prediction = {"exchange_minute": "2026-09-07 14:36:00"}
        engine._update_stacked_forecasts()
        engine._update_stacked_forecasts()
        self.assertEqual(engine.pnl_adapter.calls, 1)
        self.assertEqual(engine.latest_stacked_minute, PaperTradingEngine._event_minute(engine.latest_iv_prediction))


if __name__ == "__main__":
    unittest.main()
