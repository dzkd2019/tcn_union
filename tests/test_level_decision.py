"""离散浓度档位决策规则的单元测试。"""
from __future__ import annotations

import unittest

import numpy as np

from tcn_union.evaluation.level_decision import (
    level_matches,
    levels_from_labels,
    snap_to_levels,
)


class LevelDecisionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = np.array(
            [[8000.0, 10.0, 2.0], [16000.0, 20.0, 4.0], [8000.0, 30.0, 2.0]]
        )
        self.levels = levels_from_labels(self.labels)

    def test_levels_are_sorted_unique_per_gas(self) -> None:
        self.assertEqual(len(self.levels), 3)
        np.testing.assert_array_equal(self.levels[0], [8000.0, 16000.0])
        np.testing.assert_array_equal(self.levels[1], [10.0, 20.0, 30.0])
        np.testing.assert_array_equal(self.levels[2], [2.0, 4.0])

    def test_snap_always_uses_nearest_level(self) -> None:
        prediction = np.array([[9000.0, 26.0, 3.2], [15000.0, 12.0, 2.1]])
        snapped = snap_to_levels(prediction, self.levels)
        np.testing.assert_array_equal(
            snapped, np.array([[8000.0, 30.0, 4.0], [16000.0, 10.0, 2.0]])
        )

    def test_tolerance_only_snaps_close_predictions(self) -> None:
        prediction = np.array([[8400.0, 10.0, 2.0], [9200.0, 10.0, 2.0]])
        snapped = snap_to_levels(prediction, self.levels, tolerance=0.1)
        # 8400 距 8000 为 5%：吸附；9200 距 8000 为 15%：保留连续值。
        np.testing.assert_array_equal(snapped[:, 0], np.array([8000.0, 9200.0]))

    def test_level_matches_counts_correct_levels(self) -> None:
        truth = np.array([[8000.0, 10.0, 2.0], [16000.0, 30.0, 4.0]])
        prediction = np.array([[8100.0, 11.0, 2.2], [15000.0, 20.0, 3.9]])
        matches = level_matches(truth, prediction, self.levels)
        np.testing.assert_array_equal(
            matches, np.array([[True, True, True], [True, False, True]])
        )

    def test_invalid_shapes_rejected(self) -> None:
        with self.assertRaises(ValueError):
            levels_from_labels(np.zeros((3, 2)))
        with self.assertRaises(ValueError):
            snap_to_levels(np.zeros((2, 2)), self.levels)


if __name__ == "__main__":
    unittest.main()
