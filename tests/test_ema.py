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


class GasLossWeightTest(unittest.TestCase):
    def _run(self, weights: tuple[float, float, float] | None):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(6, 6, 51)).astype(np.float32)
        y = rng.normal(size=(6, 3)).astype(np.float32)
        return _train_network(
            x,
            y,
            None,
            None,
            y,
            1,
            123,
            torch.device("cpu"),
            False,
            gas_loss_weights=weights,
        )

    def test_valid_gas_weights_train_finite(self) -> None:
        model, stats, epochs, score, prediction = self._run((1.0, 1.0, 2.0))
        self.assertEqual(epochs, 1)
        self.assertIsNone(score)
        self.assertIsNone(prediction)
        for parameter in model.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
        self.assertTrue(np.isfinite(stats["x_mean"]).all())

    def test_invalid_gas_weights_rejected(self) -> None:
        for weights in ((1.0, 1.0), (-1.0, 1.0, 1.0), (0.0, 0.0, 0.0)):
            with self.assertRaises(ValueError):
                self._run(weights)


if __name__ == "__main__":
    unittest.main()
