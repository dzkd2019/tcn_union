from __future__ import annotations

import unittest

import numpy as np

from tcn_union.augment import (
    AugmentConfig,
    AugmentedDataset,
    cutout,
    make_context,
    prepend_constant_hold,
    warp_time,
)
from tcn_union.data import load_run, make_features


def _row(index: int) -> dict:
    time, response, labels = load_run(__import__("pathlib").Path(f"data/{index}.csv"))
    return {"file": f"data/{index}.csv", "_time": time, "_response": response, "_labels": labels}


class AugmentTest(unittest.TestCase):
    def test_warp_stays_inside_prefix(self) -> None:
        time, response, _ = load_run(__import__("pathlib").Path("data/0.csv"))
        mask = time <= 10.0 + 1e-9
        warped = warp_time(time[mask], response[mask], 0.2)
        self.assertEqual(warped.shape, response[mask].shape)
        self.assertTrue(np.isfinite(warped).all())
        np.testing.assert_allclose(warped[0], response[mask][0])
        np.testing.assert_allclose(warped[-1], response[mask][-1])

    def test_prefix_hold_uses_only_real_samples(self) -> None:
        _, response, _ = load_run(__import__("pathlib").Path("data/0.csv"))
        prefix = response[:51]
        held = prepend_constant_hold(prefix, 30)
        np.testing.assert_allclose(held[:30], prefix[:30])
        np.testing.assert_allclose(held[30:], np.broadcast_to(prefix[29], held[30:].shape))

    def test_cutout_keeps_boundaries(self) -> None:
        _, response, _ = load_run(__import__("pathlib").Path("data/0.csv"))
        prefix = response[:51]
        result = cutout(prefix, 10, 5)
        np.testing.assert_allclose(result[:10], prefix[:10])
        np.testing.assert_allclose(result[15:], prefix[15:])
        self.assertTrue(np.isfinite(result).all())

    def test_dataset_is_deterministic_and_does_not_mutate_source(self) -> None:
        rows = [_row(0), _row(1), _row(2)]
        original = [row["_response"].copy() for row in rows]
        config = AugmentConfig("jitter", noise=0.002, transient=0.05, warp=0.1)
        context = make_context(rows, 10.0, "raw_delta", config, 1234)
        dataset = AugmentedDataset(context)
        dataset.epoch = 3
        first, label_first = dataset[0]
        second, label_second = dataset[0]
        self.assertTrue(np.array_equal(first.numpy(), second.numpy()))
        self.assertTrue(np.array_equal(label_first.numpy(), label_second.numpy()))
        dataset.epoch = 4
        third, _ = dataset[0]
        self.assertFalse(np.array_equal(first.numpy(), third.numpy()))
        for row, before in zip(rows, original, strict=True):
            np.testing.assert_array_equal(row["_response"], before)

    def test_non_mixup_label_is_unchanged(self) -> None:
        rows = [_row(0), _row(1)]
        config = AugmentConfig("jitter", noise=0.002, transient=0.05)
        context = make_context(rows, 10.0, "raw_delta", config, 7)
        dataset = AugmentedDataset(context)
        dataset.epoch = 1
        _, label = dataset[0]
        np.testing.assert_allclose(label.numpy(), rows[0]["_labels"].astype(np.float32))

    def test_mixup_label_is_between_parents(self) -> None:
        rows = [_row(0), _row(1), _row(2), _row(3)]
        config = AugmentConfig("mix", mixup_alpha=0.2, mixup_probability=1.0)
        context = make_context(rows, 10.0, "raw_delta", config, 11)
        dataset = AugmentedDataset(context)
        dataset.epoch = 1
        labels = np.stack([rows[i]["_labels"] for i in range(len(rows))])
        low = labels.min(axis=0)
        high = labels.max(axis=0)
        for index in range(len(rows)):
            _, label = dataset[index]
            value = label.numpy()
            self.assertTrue(np.all(value >= low - 1e-5))
            self.assertTrue(np.all(value <= high + 1e-5))

    def test_clean_and_augmented_features_are_causal(self) -> None:
        rows = [_row(0)]
        config = AugmentConfig("jitter", noise=0.002, transient=0.05, warp=0.1)
        context = make_context(rows, 10.0, "raw_delta", config, 99)
        dataset = AugmentedDataset(context)
        dataset.epoch = 1
        features, _ = dataset[0]
        self.assertEqual(tuple(features.shape), (6, 51))
        self.assertTrue(np.isfinite(features.numpy()).all())
        clean = make_features(rows[0]["_time"], rows[0]["_response"], 10.0, "raw_delta")
        self.assertEqual(clean.shape, features.shape)

    def test_invalid_config_rejected(self) -> None:
        with self.assertRaises(ValueError):
            AugmentConfig("bad", probability=1.5)
        with self.assertRaises(ValueError):
            AugmentConfig("bad", mixup_alpha=0.2, mixup_probability=0.0)


if __name__ == "__main__":
    unittest.main()
