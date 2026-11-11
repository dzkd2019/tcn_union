"""低浓度损失加权的单元测试。"""
from __future__ import annotations

import unittest

import numpy as np

from tcn_union.training.train import _low_conc_powers, _low_conc_weights


class LowConcentrationWeightTest(unittest.TestCase):
    def test_zero_power_returns_unit_weights(self) -> None:
        target = np.asarray([[8000.0, 10.0, 2.0], [32000.0, 50.0, 10.0]])
        weights = _low_conc_weights(target, 0.0)
        np.testing.assert_array_equal(weights, np.ones_like(target, dtype=np.float32))

    def test_lower_concentration_gets_larger_weight(self) -> None:
        target = np.asarray(
            [[8000.0, 10.0, 2.0], [16000.0, 20.0, 4.0], [32000.0, 50.0, 10.0]]
        )
        weights = _low_conc_weights(target, 1.0)
        self.assertEqual(weights.shape, target.shape)
        # 每个气体：低浓度行的权重 >= 高浓度行
        self.assertTrue((weights[0] >= weights[1]).all())
        self.assertTrue((weights[1] >= weights[2]).all())
        self.assertTrue((weights >= 0.25).all() and (weights <= 4.0).all())
        # 低浓度行被放大，高浓度行被压小
        self.assertTrue((weights[0] > 1.0).all())
        self.assertTrue((weights[2] < 1.0).all())

    def test_negative_power_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _low_conc_weights(np.ones((2, 3)), -0.1)

    def test_per_gas_powers(self) -> None:
        self.assertEqual(_low_conc_powers("0,0,1.0"), (0.0, 0.0, 1.0))
        self.assertEqual(_low_conc_powers(0.5), (0.5, 0.5, 0.5))
        target = np.asarray([[8000.0, 10.0, 2.0], [32000.0, 50.0, 10.0]])
        weights = _low_conc_weights(target, "0,0,1")
        np.testing.assert_array_equal(weights[:, :2], np.ones((2, 2), dtype=np.float32))
        self.assertGreater(weights[0, 2], weights[1, 2])

    def test_invalid_per_gas_powers_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _low_conc_powers("0,-1,1")
        with self.assertRaises(ValueError):
            _low_conc_powers("0,1")


if __name__ == "__main__":
    unittest.main()
