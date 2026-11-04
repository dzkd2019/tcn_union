from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.augment import (
    CutoutConfig,
    CutoutDataset,
    apply_cutout,
    make_cutout_context,
)
from tcn_union.data import load_run, make_features


def _row(index: int) -> dict:
    time, response, labels = load_run(Path(f"data/{index}.csv"))
    return {"file": f"data/{index}.csv", "_time": time, "_response": response, "_labels": labels}


class CutoutTest(unittest.TestCase):
    def test_cutout_keeps_boundaries(self) -> None:
        _, response, _ = load_run(Path("data/0.csv"))
        prefix = response[:51]
        result = apply_cutout(prefix, 10, 5)
        np.testing.assert_allclose(result[:10], prefix[:10])
        np.testing.assert_allclose(result[15:], prefix[15:])
        self.assertTrue(np.isfinite(result).all())

    def test_dataset_is_deterministic_and_does_not_mutate_source(self) -> None:
        rows = [_row(0), _row(1), _row(2)]
        original = [row["_response"].copy() for row in rows]
        context = make_cutout_context(rows, 10.0, "raw_delta", CutoutConfig(), 1234)
        dataset = CutoutDataset(context)
        dataset.epoch = 3
        first, label_first = dataset[0]
        second, label_second = dataset[0]
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(label_first, label_second))
        dataset.epoch = 4
        third, _ = dataset[0]
        self.assertFalse(torch.equal(first, third))
        for row, before in zip(rows, original, strict=True):
            np.testing.assert_array_equal(row["_response"], before)

    def test_label_is_unchanged(self) -> None:
        rows = [_row(0), _row(1)]
        context = make_cutout_context(rows, 10.0, "raw_delta", CutoutConfig(), 7)
        dataset = CutoutDataset(context)
        dataset.epoch = 1
        _, label = dataset[0]
        np.testing.assert_allclose(label.numpy(), rows[0]["_labels"].astype(np.float32))

    def test_features_are_causal_and_have_expected_shape(self) -> None:
        rows = [_row(0)]
        context = make_cutout_context(rows, 10.0, "raw_delta", CutoutConfig(), 99)
        dataset = CutoutDataset(context)
        dataset.epoch = 1
        features, _ = dataset[0]
        self.assertEqual(tuple(features.shape), (6, 51))
        self.assertTrue(np.isfinite(features.numpy()).all())
        clean = make_features(rows[0]["_time"], rows[0]["_response"], 10.0, "raw_delta")
        self.assertEqual(clean.shape, features.shape)

    def test_zero_probability_returns_clean_features(self) -> None:
        rows = [_row(0)]
        context = make_cutout_context(
            rows, 10.0, "raw_delta", CutoutConfig(probability=0.0), 5
        )
        dataset = CutoutDataset(context)
        dataset.epoch = 1
        features, _ = dataset[0]
        clean = make_features(rows[0]["_time"], rows[0]["_response"], 10.0, "raw_delta")
        np.testing.assert_allclose(features.numpy(), clean)

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CutoutConfig(probability=1.5)
        with self.assertRaises(ValueError):
            CutoutConfig(max_seconds=0.0)
        with self.assertRaises(ValueError):
            make_cutout_context([_row(0)], 5.0, "raw_delta", CutoutConfig(max_seconds=6.0), 1)


if __name__ == "__main__":
    unittest.main()
