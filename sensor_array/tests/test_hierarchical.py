"""分层随机效应生成器的最小测试。"""
from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from sensor_array.core.hierarchical import HierarchicalRandomEffectsModel


class _DummyForward:
    def predict_relative(self, labels, time):
        labels = np.atleast_2d(labels)
        time = np.asarray(time, dtype=float)
        amplitude = 0.01 * labels[:, :1] / 1000.0 + 0.1
        shape = 1.0 - np.exp(-time[None, :, None] / 10.0)
        signs = np.array([1.0, 1.0, -1.0])
        return signs[None, None, :] * amplitude[:, None, :] * shape


class HierarchicalTest(unittest.TestCase):
    def test_sampling_shape_and_scale(self) -> None:
        times = np.arange(51) * 0.2
        rng = np.random.default_rng(0)
        experiments = []
        for index in range(8):
            concentration = np.array([8000.0 + 2000 * index, 10.0 + 5 * index, 2.0 + index])
            baseline = _DummyForward().predict_relative(concentration[None, :], times)[0]
            gain = np.exp(rng.normal(0.0, 0.02, 3))
            experiments.append(
                SimpleNamespace(
                    time=times,
                    concentration=concentration,
                    relative=baseline * gain[None, :],
                )
            )
        model = HierarchicalRandomEffectsModel().fit(
            _DummyForward(), experiments, seconds=10
        )
        self.assertEqual(model.effects_.shape, (8, 9))
        response = model.sample(
            _DummyForward(),
            np.array([[12000.0, 20.0, 4.0]]),
            np.array([[1000.0, 500.0, 100.0]]),
            seconds=10,
            rng=np.random.default_rng(1),
        )
        self.assertEqual(response.shape, (1, 51, 3))
        self.assertTrue(np.all(np.isfinite(response)))
        self.assertTrue(np.all(response > 0.0))


if __name__ == "__main__":
    unittest.main()
