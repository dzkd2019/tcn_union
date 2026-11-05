from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.train import _read_manifest
from tcn_union.vae import VAEConfig, fit_vae, synthesize_features


def _rows(count: int = 6) -> list[dict]:
    rows = _read_manifest(Path("artifacts/splits/manifest.csv"))
    return [row for row in rows if row["split"] == "development"][:count]


class VaeTest(unittest.TestCase):
    def test_sampling_shape_and_range(self) -> None:
        rows = _rows()
        config = VAEConfig(
            epochs=2,
            batch_size=3,
            components=4,
            latent_dim=2,
            hidden_dim=8,
        )
        vae = fit_vae(rows, config, 123, torch.device("cpu"))
        responses, labels = vae.sample(4, 456)
        self.assertEqual(responses.shape, (4, 3, 51))
        self.assertEqual(labels.shape, (4, 3))
        self.assertTrue(np.isfinite(responses).all())
        self.assertTrue(np.isfinite(labels).all())
        train = np.stack(
            [np.asarray(row["_response"], dtype=np.float64)[:51].T for row in rows]
        )
        self.assertTrue(np.all(responses >= train.min(axis=(0, 2))[None, :, None] - 1e-6))
        self.assertTrue(np.all(responses <= train.max(axis=(0, 2))[None, :, None] + 1e-6))

    def test_synthetic_features_shape(self) -> None:
        config = VAEConfig(
            epochs=2,
            batch_size=3,
            components=4,
            latent_dim=2,
            hidden_dim=8,
        )
        features, labels = synthesize_features(
            _rows(), 2, config, 7, torch.device("cpu")
        )
        self.assertEqual(features.shape, (2, 6, 51))
        self.assertEqual(labels.shape, (2, 3))
        self.assertTrue(np.isfinite(features).all())

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VAEConfig(seconds=7.0)
        with self.assertRaises(ValueError):
            VAEConfig(components=0)
        with self.assertRaises(ValueError):
            VAEConfig(beta=-1.0)


if __name__ == "__main__":
    unittest.main()
