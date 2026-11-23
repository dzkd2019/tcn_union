"""少一路传感器时的浓度反演评估。

模型仍用三路响应训练；评估时只在 MAP 残差中删掉一路通道，模拟部署时某一路失效。
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from ..analysis.estimation import (
    concentration_bounds,
    estimate_noise,
    map_estimate,
    start_points,
)
from ..core.data import GASES, Experiment, load_experiments
from ..core.double_exponential import DoubleExponentialArrayModel
from ..core.mlp_per_gas import PerGasMLPDoubleExponentialModel
from ..core.shapes import fit_experiment_shapes
from ..paths import EVALUATION, ensure

CONDITIONS = (
    ("完整", (0, 1, 2)),
    ("缺 H₂", (1, 2)),
    ("缺 NH₃", (0, 2)),
    ("缺 NO₂", (0, 1)),
)


def _sigma(model, train: list[Experiment]) -> np.ndarray:
    time = train[0].time
    concentration = np.stack([item.concentration for item in train])
    relative = np.stack([item.relative for item in train])
    sigma_model = np.sqrt(
        np.mean((model.predict_relative(concentration, time) - relative) ** 2, axis=(0, 1))
    )
    noise = estimate_noise(train)
    return np.sqrt(noise**2 + sigma_model**2)


def run(folds: int = 62, window: float = 10.0, max_iter: int = 600) -> None:
    ensure()
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    rows: list[dict] = []
    for index, test in enumerate(experiments[:folds]):
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        shape_init = tuple(shapes[k][train_indices] for k in range(3))
        double = DoubleExponentialArrayModel().fit(train, shape_init=shape_init)
        combined = PerGasMLPDoubleExponentialModel(epochs=max_iter).fit(train, skeleton=double)
        sigma = _sigma(combined, train)
        bounds = concentration_bounds(train)
        starts = start_points(train)
        mask = test.time <= window
        for label, channels in CONDITIONS:
            estimate = map_estimate(
                combined,
                test.time[mask],
                test.relative[mask],
                sigma,
                bounds,
                starts,
                channels=np.asarray(channels, dtype=int),
            )
            for j, gas in enumerate(GASES):
                theta = float(estimate["theta"][j])
                rows.append(
                    {
                        "file": test.file_id,
                        "condition": label,
                        "tmax": window,
                        "gas": gas,
                        "theta_hat": theta,
                        "theta_true": float(test.concentration[j]),
                        "rel_error": theta / test.concentration[j] - 1.0,
                        "abs_rel_error": abs(theta / test.concentration[j] - 1.0),
                        "success": estimate["success"],
                    }
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(EVALUATION / "degradation.csv", index=False)
    summary = (
        frame.groupby(["condition", "tmax", "gas"], sort=False)
        .agg(
            median_abs_rel_error=("abs_rel_error", "median"),
            p90_abs_rel_error=("abs_rel_error", lambda values: values.quantile(0.9)),
            median_rel_error=("rel_error", "median"),
            success_rate=("success", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(EVALUATION / "degradation_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=62)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--max-iter", type=int, default=600)
    args = parser.parse_args()
    run(folds=args.folds, window=args.window, max_iter=args.max_iter)
