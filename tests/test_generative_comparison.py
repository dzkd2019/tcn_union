from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.augmentation.direct_vae import DirectVAEConfig, fit_direct_vae
from tcn_union.augmentation.pca_generator import PCAConfig, fit_pca
from tcn_union.evaluation.generative_metrics import evaluate_generation
from tcn_union.training.train import _read_manifest


def _rows(count: int = 8) -> list[dict]:
    rows = _read_manifest(Path("artifacts/splits/manifest.csv"))
    return [row for row in rows if row["split"] == "development"][:count]


class GenerativeBaselineTest(unittest.TestCase):
    def test_pca_sample_shape_and_determinism(self) -> None:
        rows = _rows()
        generator = fit_pca(rows, PCAConfig(components=4), 123)
        first = generator.sample(4, 7)
        second = generator.sample(4, 7)
        self.assertEqual(first[0].shape, (4, 3, 51))
        self.assertEqual(first[1].shape, (4, 3))
        np.testing.assert_array_equal(first[0], second[0])
        self.assertTrue(np.isfinite(first[0]).all())

    def test_direct_vae_sample_shape(self) -> None:
        rows = _rows(6)
        config = DirectVAEConfig(
            epochs=1,
            batch_size=3,
            latent_dim=2,
            hidden_dim=8,
        )
        generator = fit_direct_vae(rows, config, 123, torch.device("cpu"))
        responses, labels = generator.sample(3, 4)
        self.assertEqual(responses.shape, (3, 3, 51))
        self.assertEqual(labels.shape, (3, 3))
        self.assertTrue(np.isfinite(responses).all())

    def test_metrics_identical_samples(self) -> None:
        rng = np.random.default_rng(0)
        real = rng.normal(size=(8, 3, 51))
        report = evaluate_generation(real, real.copy())
        self.assertAlmostEqual(report["macro"]["wasserstein_normalized"], 0.0, places=8)
        self.assertAlmostEqual(report["macro"]["acf_distance"], 0.0, places=8)
        self.assertAlmostEqual(report["macro"]["pairwise_distance_ratio"], 1.0, places=8)


if __name__ == "__main__":
    unittest.main()
