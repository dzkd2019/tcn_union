"""无 Ridge 的 TCN 集评估模块单元测试（CPU 合成数据）。"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.evaluation.ensemble_evaluate import (
    _available_seeds,
    _clip_to_development_range,
    _mean_predictions,
    _tcn_predictions,
)


class MeanPredictionsTest(unittest.TestCase):
    def test_mean_predictions_average_seeds(self) -> None:
        rng = np.random.default_rng(3)
        predictions = [rng.uniform(1.0, 10.0, size=(5, 3)) for _ in range(3)]
        expected = np.mean(np.stack(predictions), axis=0)
        np.testing.assert_allclose(_mean_predictions(predictions), expected)


class ClipTest(unittest.TestCase):
    def test_clip_to_development_range_uses_per_gas_bounds(self) -> None:
        y_dev = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0]])
        prediction = np.array([[-1.0, 5.0, 40.0], [20.0, 10.0, -5.0]])
        clipped = _clip_to_development_range(prediction, y_dev)
        np.testing.assert_allclose(
            clipped, np.array([[0.0, 5.0, 30.0], [10.0, 10.0, 0.0]])
        )


class CheckpointLoadingTest(unittest.TestCase):
    def test_missing_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                _tcn_predictions(
                    [Path(tmp)], [], 10.0, [20260923], torch.device("cpu")
                )

    def test_available_seeds_supports_per_seed_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            window_dir = Path(tmp) / "10s"
            window_dir.mkdir()
            (window_dir / "seed_20260923.pt").touch()
            self.assertEqual(
                _available_seeds(Path(tmp), 10.0, [20260923, 20260924]),
                [20260923],
            )
            self.assertEqual(_available_seeds(Path(tmp), 10.0, [20260924]), [])


if __name__ == "__main__":
    unittest.main()
