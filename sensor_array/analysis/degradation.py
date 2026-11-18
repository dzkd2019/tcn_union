"""留一传感器降级实验：固定时间窗，依次屏蔽一路传感器后估计三气体浓度。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.data import GASES, load_experiments
from .estimation import _fit_fold, concentration_bounds, map_estimate
from ..core.grey_box import fit_experiment_shapes
from ..paths import ANALYSIS, ensure

WINDOW = 10.0
DROP_SIGMA = 1e6


def main() -> None:
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    low, high = concentration_bounds(experiments)
    rows = []
    for i, experiment in enumerate(experiments):
        train = [item for j, item in enumerate(experiments) if j != i]
        indices = [j for j in range(len(experiments)) if j != i]
        model, sigma_model = _fit_fold(train, tuple(part[indices] for part in shapes))
        sigma_full = np.sqrt(1e-10 + sigma_model**2)
        mask = experiment.time <= WINDOW
        for drop in ("none",) + GASES:
            sigma = sigma_full.copy()
            if drop != "none":
                sigma[GASES.index(drop)] = DROP_SIGMA
            result = map_estimate(
                model,
                experiment.relative[mask],
                experiment.time[mask],
                sigma,
                low,
                high,
            )
            truth = experiment.concentration
            for gas, estimate in zip(GASES, np.exp(result["theta"])):
                rows.append(
                    {
                        "file": experiment.file_id,
                        "drop": drop,
                        "gas": gas,
                        "estimate": estimate,
                        "truth": experiment.concentration[GASES.index(gas)],
                        "abs_rel_error": abs(estimate / experiment.concentration[GASES.index(gas)] - 1.0),
                        "success": result["success"],
                    }
                )
        print(f"降级折 {i + 1}/{len(experiments)} (file {experiment.file_id})", flush=True)
    frame = pd.DataFrame(rows)
    ensure()
    frame.to_csv(ANALYSIS / "degradation.csv", index=False, encoding="utf-8")
    print(frame.groupby(["drop", "gas"])["abs_rel_error"].median().unstack() * 100.0)


if __name__ == "__main__":
    main()
