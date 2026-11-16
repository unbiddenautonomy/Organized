import math
import unittest

from banknifty_120m_live_inference import training_greeks


class ExpandedHybridLiveInferenceTest(unittest.TestCase):
    def test_training_greeks_have_short_structure_signs(self):
        call = training_greeks(48000, 48000, 10 / 365, 0.20, 48000, True)
        put = training_greeks(48000, 48000, 10 / 365, 0.20, 48000, False)
        self.assertGreater(call["gamma"], 0)
        self.assertGreater(call["vega"], 0)
        self.assertLess(call["theta_day"], 0)
        short_gamma = -call["gamma"] - put["gamma"]
        short_vega = -call["vega"] - put["vega"]
        short_theta = -call["theta_day"] - put["theta_day"]
        self.assertLess(short_gamma, 0)
        self.assertLess(short_vega, 0)
        self.assertGreater(short_theta, 0)

    def test_training_greeks_use_absolute_volatility_vega_units(self):
        result = training_greeks(48000, 48000, 10 / 365, 0.20, 48000, True)
        self.assertGreater(result["vega"], 1000)
        self.assertTrue(all(math.isfinite(value) for value in result.values()))


if __name__ == "__main__":
    unittest.main()
