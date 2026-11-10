from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from tcn_union.core.data import load_run, make_features
from tcn_union.core.model import GasTCN
from tcn_union.training.train import fit_scalers

ROOT = Path(__file__).resolve().parents[1]


class CausalityTests(unittest.TestCase):
    def test_future_sensor_values_cannot_change_five_second_prediction(self) -> None:
        time, response, _ = load_run(ROOT / "data/1.csv")
        changed = response.copy()
        changed[time > 5.0] += np.array([10000.0, -5000.0, 3000.0])
        original_input = make_features(time, response, 5.0, "raw_delta")
        changed_input = make_features(time, changed, 5.0, "raw_delta")
        np.testing.assert_array_equal(original_input, changed_input)
        torch.manual_seed(123)
        model = GasTCN(6).eval()
        with torch.no_grad():
            original_output = model(torch.from_numpy(original_input).unsqueeze(0))
            changed_output = model(torch.from_numpy(changed_input).unsqueeze(0))
        torch.testing.assert_close(original_output, changed_output, rtol=0.0, atol=0.0)

    def test_holdout_perturbations_do_not_enter_training_scalers(self) -> None:
        time, response, label = load_run(ROOT / "data/1.csv")
        training_x = np.stack(
            [
                make_features(time, response, 5.0, "raw_delta"),
                make_features(time, response * 1.01, 5.0, "raw_delta"),
            ]
        )
        training_y = np.stack((label, label + np.array([100.0, 1.0, 0.1]))).astype(np.float32)
        all_x = np.concatenate((training_x, make_features(time, response, 5.0, "raw_delta")[None, :, :]))
        all_y = np.concatenate((training_y, label[None, :].astype(np.float32)))
        train_indices = np.array([0, 1])
        first = fit_scalers(all_x[train_indices], all_y[train_indices])
        all_x[2] += 1e6
        all_y[2] += 1e6
        second = fit_scalers(all_x[train_indices], all_y[train_indices])
        for name in first:
            np.testing.assert_array_equal(first[name], second[name])


if __name__ == "__main__":
    unittest.main()
