from __future__ import annotations

import unittest

import numpy as np

from tcn_union.evaluation.evaluate import _metric_summary


class MapeMetricTests(unittest.TestCase):
    def test_mape_is_percent_of_each_true_value(self) -> None:
        truth = np.array([[10.0, 20.0, 2.0], [20.0, 40.0, 4.0]])
        prediction = np.array([[11.0, 18.0, 1.5], [22.0, 40.0, 5.0]])
        metrics = _metric_summary(truth, prediction, np.array([10.0, 20.0, 2.0]))
        self.assertAlmostEqual(metrics["H2"]["MAPE_percent"], 10.0)
        self.assertAlmostEqual(metrics["NH3"]["MAPE_percent"], 5.0)
        self.assertAlmostEqual(metrics["NO2"]["MAPE_percent"], 25.0)
        self.assertAlmostEqual(metrics["macro_MAPE_percent"], 40.0 / 3.0)

    def test_mape_is_null_when_any_target_is_zero(self) -> None:
        truth = np.array([[0.0, 20.0, 2.0], [10.0, 40.0, 4.0]])
        prediction = np.array([[1.0, 22.0, 3.0], [9.0, 38.0, 3.0]])
        metrics = _metric_summary(truth, prediction, np.array([10.0, 20.0, 2.0]))
        self.assertIsNone(metrics["H2"]["MAPE_percent"])
        self.assertIn("zero", metrics["H2"]["MAPE_null_reason"])
        self.assertIsNotNone(metrics["NH3"]["MAPE_percent"])
        self.assertIsNone(metrics["macro_MAPE_percent"])


if __name__ == "__main__":
    unittest.main()
