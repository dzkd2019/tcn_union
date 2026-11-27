"""汇总 no2_guard：逐折配对、seed 级 OOF 与 3-seed 集成。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tcn_union.experiments.forward_augmentation import _metric_values
from tcn_union.training.gan_train import _read_manifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "artifacts/augmentation/no2_guard/full"
METRICS = [
    "macro_three_sigma",
    "macro_mape",
    "H2_three_sigma",
    "NH3_three_sigma",
    "NO2_three_sigma",
]


def _truth_by_fold(path: Path) -> np.ndarray:
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    return np.concatenate(
        [
            np.stack([row["_labels"] for row in development if int(row["fold"]) == fold])
            for fold in range(5)
        ]
    ).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT))
    args = parser.parse_args()
    path = Path(args.path)
    folds = pd.read_csv(path / "folds.csv")
    arrays = np.load(path / "predictions.npz")
    truth = _truth_by_fold(ROOT)
    seeds = sorted(folds["seed"].unique())
    configs = sorted(folds["config"].unique())

    rows = []
    for arm in ("vae", "guard"):
        for seed in seeds:
            prediction = np.mean(
                [arrays[f"{arm}|{seed}|{config}"] for config in configs], axis=0
            )
            rows.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "n": len(truth),
                    **_metric_values(prediction, truth),
                }
            )
    seed_rows = pd.DataFrame(rows)
    seed_rows.to_csv(path / "oof_seed.csv", index=False)

    ensemble_rows = []
    for arm in ("vae", "guard"):
        prediction = np.mean(
            [arrays[f"{arm}|{seed}|{config}"] for seed in seeds for config in configs],
            axis=0,
        )
        ensemble_rows.append(
            {"arm": arm, "n": len(truth), **_metric_values(prediction, truth)}
        )
    ensemble = pd.DataFrame(ensemble_rows)
    ensemble.to_csv(path / "ensemble.csv", index=False)

    paired = folds.pivot_table(
        index=["fold", "seed", "config"], columns="arm", values="macro_three_sigma"
    ).reset_index()
    paired["guard_minus_vae"] = paired["guard"] - paired["vae"]
    paired.to_csv(path / "paired.csv", index=False)
    macro = paired["guard_minus_vae"].to_numpy()
    print(seed_rows[["arm", "seed", *METRICS]].to_string(index=False))
    print()
    print(ensemble[["arm", *METRICS]].to_string(index=False))
    print()
    if np.any(macro != 0):
        p_value = wilcoxon(macro).pvalue
        print(
            f"paired folds={len(macro)} improved={(macro < 0).sum()} "
            f"mean={macro.mean():+.3f} pp median={np.median(macro):+.3f} pp "
            f"p={p_value:.4g}"
        )
    else:
        print("paired differences all zero")


if __name__ == "__main__":
    main()
