"""汇总组合使用：seed 级 OOF 与多个 seed 平均；NH₃/NO₂ 与 PCA-VAE 一致。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tcn_union.experiments.full_protocol_augmentation import _metric_values
from tcn_union.training.gan_train import _read_manifest

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "artifacts/augmentation/no2_guard/full"
REFERENCE = ROOT / "artifacts/augmentation/protocols/full/predictions.npz"
METRICS = [
    "macro_three_sigma",
    "macro_mape",
    "H2_three_sigma",
    "NH3_three_sigma",
    "NO2_three_sigma",
]


def _truth() -> np.ndarray:
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    return np.concatenate(
        [
            np.stack([row["_labels"] for row in development if int(row["fold"]) == fold])
            for fold in sorted({int(row["fold"]) for row in development})
        ]
    ).astype(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(DEFAULT))
    args = parser.parse_args()
    path = Path(args.path)
    folds = pd.read_csv(path / "folds.csv")
    arrays = np.load(path / "predictions.npz")
    truth = _truth()
    reference = np.load(REFERENCE)["vae_seed_mean"][:, 1:]
    seeds = sorted(folds["seed"].unique())
    configs = sorted(folds["config"].unique())

    def _prediction(seed_list: list[int]) -> np.ndarray:
        prediction = np.mean(
            [arrays[f"guard|{seed}|{config}"] for seed in seed_list for config in configs],
            axis=0,
        )
        prediction = prediction.copy()
        prediction[:, 1:] = reference
        return prediction

    rows = []
    for seed in seeds:
        rows.append(
            {
                "arm": "guard",
                "seed": seed,
                "n": len(truth),
                **_metric_values(_prediction([seed]), truth),
            }
        )
    seed_rows = pd.DataFrame(rows)
    seed_rows.to_csv(path / "oof_seed.csv", index=False)

    seed_mean = pd.DataFrame(
        [{"arm": "guard", "n": len(truth), **_metric_values(_prediction(seeds), truth)}]
    )
    seed_mean.to_csv(path / "seed_mean.csv", index=False)

    print(seed_rows[["arm", "seed", *METRICS]].to_string(index=False))
    print()
    print(seed_mean[["arm", *METRICS]].to_string(index=False))


if __name__ == "__main__":
    main()
