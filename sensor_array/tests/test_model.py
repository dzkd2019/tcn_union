"""新模型的最小单元测试：形状初值、双指数、MLP 残差与保存/加载。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sensor_array import load_experiments
from sensor_array.core.double_exponential import DoubleExponentialArrayModel
from sensor_array.core.mlp_per_gas import PerGasMLPDoubleExponentialModel
from sensor_array.core.shapes import fit_experiment_shapes

INDICES = [0, 1, 5, 7, 11, 14]


class ModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        all_experiments = load_experiments()
        cls.experiments = [all_experiments[index] for index in INDICES]
        cls.shapes = fit_experiment_shapes(cls.experiments)
        cls.double = DoubleExponentialArrayModel().fit(cls.experiments, shape_init=cls.shapes)
        cls.per_gas = PerGasMLPDoubleExponentialModel(epochs=2, device="cpu").fit(
            cls.experiments, skeleton=cls.double
        )
        cls.test = cls.experiments[0]

    def test_shapes_are_finite_and_positive(self) -> None:
        amplitude, tau, delay = self.shapes
        self.assertEqual(amplitude.shape, (len(self.experiments), 3))
        self.assertTrue(np.isfinite(amplitude).all())
        self.assertTrue(np.isfinite(tau).all())
        self.assertTrue(np.isfinite(delay).all())
        self.assertTrue((amplitude > 0).all())
        self.assertTrue((tau > 0).all())
        self.assertTrue((delay >= 0).all())
        self.assertTrue((delay <= 2.0).all())

    def test_double_prediction_shape_and_sign(self) -> None:
        prediction = self.double.predict_relative(self.test.concentration[None, :], self.test.time)
        self.assertEqual(prediction.shape, (1, len(self.test.time), 3))
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.all(prediction[0, :, 2] <= 0.0))
        response = self.double.predict_response(
            self.test.concentration[None, :], self.test.time, self.test.initial
        )
        self.assertTrue(np.allclose(response, self.test.initial[None, None, :] * (1.0 + prediction)))

    def test_prefix_prediction_matches_full_curve(self) -> None:
        concentration = self.test.concentration[None, :]
        for model in (self.double, self.per_gas):
            full = model.predict_relative(concentration, self.test.time)
            prefix = model.predict_relative(concentration, self.test.time[:20])
            self.assertTrue(np.allclose(prefix, full[:, :20]))

    def test_per_gas_adds_residual(self) -> None:
        baseline = self.double.predict_relative(self.test.concentration[None, :], self.test.time)
        combined = self.per_gas.predict_relative(self.test.concentration[None, :], self.test.time)
        self.assertEqual(combined.shape, baseline.shape)
        self.assertTrue(np.isfinite(combined).all())
        self.assertGreater(float(np.max(np.abs(combined - baseline))), 0.0)

    def test_save_load_round_trip(self) -> None:
        concentration = self.test.concentration[None, :]
        time = self.test.time
        expected_double = self.double.predict_relative(concentration, time)
        expected_per_gas = self.per_gas.predict_relative(concentration, time)
        with tempfile.TemporaryDirectory() as directory:
            double_path = Path(directory) / "double.json"
            per_gas_path = Path(directory) / "per_gas.json"
            self.double.save(double_path)
            self.per_gas.save(per_gas_path)
            loaded_double = DoubleExponentialArrayModel.load(double_path)
            loaded_per_gas = PerGasMLPDoubleExponentialModel.load(per_gas_path)
        self.assertLess(float(np.max(np.abs(loaded_double.predict_relative(concentration, time) - expected_double))), 1e-10)
        self.assertLess(float(np.max(np.abs(loaded_per_gas.predict_relative(concentration, time) - expected_per_gas))), 1e-10)



if __name__ == "__main__":
    unittest.main()