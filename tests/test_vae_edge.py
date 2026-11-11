"""VAE 边缘案例条件采样与边缘组合选择的单元测试。"""
from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.training.gan_train import _edge_case_labels
from tcn_union.training.train import _read_manifest

ROOT = Path(__file__).resolve().parents[1]


def _rows(count: int = 6) -> list[dict]:
    rows = _read_manifest(ROOT / "artifacts/splits/manifest.csv")
    return [row for row in rows if row["split"] == "development"][:count]


class EdgeCaseAugmentationTest(unittest.TestCase):
    def _fit_tiny_vae(self, rows: list[dict]):
        config = VAEConfig(
            epochs=2,
            batch_size=3,
            components=4,
            latent_dim=2,
            hidden_dim=8,
        )
        return fit_vae(rows, config, 123, torch.device("cpu"))

    def test_condition_labels_are_reproduced(self) -> None:
        rows = _rows()
        vae = self._fit_tiny_vae(rows)
        labels = np.repeat(
            np.asarray([rows[0]["_labels"]], dtype=np.float32), 4, axis=0
        )
        responses, out_labels = vae.sample(4, 456, condition_labels=labels)
        self.assertEqual(responses.shape, (4, 3, 51))
        np.testing.assert_allclose(out_labels, labels, rtol=0.0, atol=1e-5)
        self.assertTrue(np.isfinite(responses).all())

    def test_invalid_condition_labels_rejected(self) -> None:
        rows = _rows()
        vae = self._fit_tiny_vae(rows)
        with self.assertRaises(ValueError):
            vae.sample(4, 1, condition_labels=np.ones((3, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            vae.sample(
                2,
                1,
                condition_labels=np.asarray(
                    [[-1.0, 10.0, 2.0], [8000.0, 20.0, 4.0]], dtype=np.float32
                ),
            )

    def test_edge_case_labels_keep_only_extreme_levels(self) -> None:
        rows = [
            {"_labels": np.array([8000.0, 10.0, 2.0])},
            {"_labels": np.array([16000.0, 20.0, 4.0])},
            {"_labels": np.array([32000.0, 50.0, 10.0])},
        ]
        edge = _edge_case_labels(rows)
        self.assertEqual(edge.shape, (2, 3))
        np.testing.assert_array_equal(edge[0], [8000.0, 10.0, 2.0])
        np.testing.assert_array_equal(edge[1], [32000.0, 50.0, 10.0])


if __name__ == "__main__":
    unittest.main()
