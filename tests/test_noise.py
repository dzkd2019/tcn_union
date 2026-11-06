from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tcn_union.noise import NoiseConfig, estimate_noise_sigma, noisy_features
from tcn_union.train import _read_manifest


def _rows(count: int = 6) -> list[dict]:
    rows = _read_manifest(Path("artifacts/splits/manifest.csv"))
    return [row for row in rows if row["split"] == "development"][:count]


class NoiseTest(unittest.TestCase):
    def test_sigma_positive_and_three_channels(self) -> None:
        sigma = estimate_noise_sigma(_rows(), 10.0)
        self.assertEqual(sigma.shape, (3,))
        self.assertTrue(np.isfinite(sigma).all())
        self.assertTrue(np.all(sigma >= 0.0))

    def test_noisy_features_are_deterministic_and_finite(self) -> None:
        rows = _rows()
        config = NoiseConfig(scale=0.1, copies=2)
        first_x, first_y = noisy_features(rows, 10.0, "raw_delta", config, 123)
        second_x, second_y = noisy_features(rows, 10.0, "raw_delta", config, 123)
        self.assertEqual(first_x.shape, (len(rows) * 2, 6, 51))
        self.assertEqual(first_y.shape, (len(rows) * 2, 3))
        self.assertTrue(np.isfinite(first_x).all())
        np.testing.assert_array_equal(first_x, second_x)
        np.testing.assert_array_equal(first_y, second_y)

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NoiseConfig(scale=-1.0)
        with self.assertRaises(ValueError):
            NoiseConfig(copies=0)


if __name__ == "__main__":
    unittest.main()
