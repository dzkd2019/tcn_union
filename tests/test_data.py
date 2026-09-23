from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tcn_union.data import load_prediction_prefix, load_run, make_features

ROOT = Path(__file__).resolve().parents[1]


class DataContractTests(unittest.TestCase):
    def test_real_run_prefix_shapes_and_zero_delta(self) -> None:
        time, response, labels = load_run(ROOT / "data/1.csv")
        self.assertEqual((len(time), response.shape, labels.shape), (299, (299, 3), (3,)))
        self.assertEqual(make_features(time, response, 5.0, "raw_delta").shape, (6, 26))
        self.assertEqual(make_features(time, response, 10.0, "raw_delta").shape, (6, 51))
        for seconds, length in ((5.0, 26), (10.0, 51)):
            delta = make_features(time, response, seconds, "delta")
            self.assertEqual(delta.shape, (3, length))
            np.testing.assert_array_equal(delta[:, 0], np.zeros(3, dtype=np.float32))

    def test_prediction_prefix_ignores_extra_label_columns(self) -> None:
        source = (ROOT / "data/1.csv").read_text(encoding="utf-8-sig").splitlines()[:28]
        altered = source.copy()
        for index in range(1, len(altered)):
            cells = altered[index].split(",")
            cells[4:] = ["999", "888", "777"]
            altered[index] = ",".join(cells)
        with tempfile.TemporaryDirectory() as directory:
            original_path = Path(directory) / "original.csv"
            altered_path = Path(directory) / "altered.csv"
            original_path.write_text("\n".join(source) + "\n", encoding="utf-8")
            altered_path.write_text("\n".join(altered) + "\n", encoding="utf-8")
            original = load_prediction_prefix(original_path)
            changed = load_prediction_prefix(altered_path)
        np.testing.assert_array_equal(original[0], changed[0])
        np.testing.assert_array_equal(original[1], changed[1])

    def test_prediction_reader_stops_after_requested_window(self) -> None:
        source = (ROOT / "data/1.csv").read_text(encoding="utf-8-sig").splitlines()[:27]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prefix.csv"
            path.write_text(
                "\n".join(source + ["5.2,not-a-number,NaN,invalid"]) + "\n",
                encoding="utf-8",
            )
            time, response = load_prediction_prefix(path, 5.0)
        features = make_features(time, response, 5.0, "delta")
        self.assertEqual(features.shape, (3, 26))

    def test_training_loader_rejects_missing_concentration_columns(self) -> None:
        source = (ROOT / "data/1.csv").read_text(encoding="utf-8-sig").splitlines()[:4]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unlabeled.csv"
            path.write_text("\n".join(source).split(",conc1")[0] + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected header"):
                load_run(path)

    def test_prefix_shorter_than_target_fails(self) -> None:
        time, response, _ = load_run(ROOT / "data/1.csv")
        with self.assertRaisesRegex(ValueError, "prefix is unavailable"):
            make_features(time[time <= 4.0], response[time <= 4.0], 5.0, "delta")


if __name__ == "__main__":
    unittest.main()
