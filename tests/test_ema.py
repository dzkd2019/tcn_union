from __future__ import annotations

import unittest

import numpy as np
import torch

from tcn_union.training.train import _train_network


class EmaTrainingTest(unittest.TestCase):
    def test_ema_training_produces_finite_weights(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=(8, 6, 51)).astype(np.float32)
        y = rng.normal(size=(8, 3)).astype(np.float32)
        model, stats, epochs, score, prediction = _train_network(
            x,
            y,
            None,
            None,
            y,
            2,
            123,
            torch.device("cpu"),
            False,
            ema_decay=0.9,
        )
        self.assertEqual(epochs, 2)
        self.assertIsNone(score)
        self.assertIsNone(prediction)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
        self.assertTrue(np.isfinite(stats["x_mean"]).all())

    def test_invalid_ema_decay_rejected(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(4, 6, 51)).astype(np.float32)
        y = rng.normal(size=(4, 3)).astype(np.float32)
        with self.assertRaises(ValueError):
            _train_network(
                x,
                y,
                None,
                None,
                y,
                1,
                123,
                torch.device("cpu"),
                False,
                ema_decay=1.0,
            )


if __name__ == "__main__":
    unittest.main()
