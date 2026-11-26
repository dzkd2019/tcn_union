"""汇总方案 1 的 6 checkpoint 集成与配对比较。"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tcn_union.experiments.full_protocol_augmentation import _metric_values
from tcn_union.experiments.old_protocol_augmentation import _write_csv

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    run = args.run if args.run.is_absolute() else ROOT / args.run
    data = np.load(run / "predictions.npz")
    truth = data["truth"]
    labels_min = truth.min(axis=0)
    labels_max = truth.max(axis=0)
    arms = sorted({key.split("|")[0] for key in data.files if "|" in key})
    rows = []
    for arm in arms:
        keys = [key for key in data.files if key.startswith(f"{arm}|")]
        prediction = np.mean([data[key] for key in keys], axis=0)
        prediction = np.clip(prediction, labels_min, labels_max)
        rows.append({"arm": arm, "n_checkpoints": len(keys), **_metric_values(prediction, truth)})
    _write_csv(run / "ensemble.csv", rows)
    print(pd.DataFrame(rows).to_string(index=False))

    folds = pd.read_csv(run / "folds.csv")
    pivot = folds.pivot_table(
        index=["seed", "config", "fold"],
        columns="arm",
        values="macro_three_sigma",
    )
    pairs = [
        ("hier", "real"),
        ("hier", "vae"),
        ("hier+vae", "real"),
        ("hier+vae", "vae"),
        ("hier+vae", "hier"),
    ]
    paired = []
    for candidate, baseline in pairs:
        if candidate not in pivot or baseline not in pivot:
            continue
        diff = (pivot[candidate] - pivot[baseline]).dropna()
        statistic, pvalue = wilcoxon(diff) if len(diff) >= 6 else (np.nan, np.nan)
        paired.append(
            {
                "candidate": candidate,
                "baseline": baseline,
                "n": int(len(diff)),
                "mean_diff": float(diff.mean()),
                "median_diff": float(diff.median()),
                "improved": int((diff < 0).sum()),
                "p": float(pvalue),
            }
        )
    _write_csv(run / "paired.csv", paired)
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == "__main__":
    main()
