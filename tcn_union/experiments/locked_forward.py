"""锁定 12 条的一次性 TCN 评估。

OOF 结果只用于选候选臂与固定训练轮数；本脚本不再用锁定集做早停或选模型。
每个 seed 在 50 条开发集上重训，锁定 12 条只前向一次。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from tcn_union.core.data import make_features
from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.experiments.forward_augmentation import (
    ARMS,
    BASE,
    DEFAULT_SEEDS,
    DEFAULT_WINDOWS,
    _arm_arrays,
    _forward_synthetic,
    _metric_values,
    _pca_vae_synthetic,
    _real_features,
    _real_scaler,
)
from tcn_union.training.gan_train import MODE, _read_manifest
from tcn_union.training.train import (
    _restore_target,
    _train_network,
)

from sensor_array.core.data import load_experiments


def _fixed_epochs(rows: list[dict], window: int, arm: str, seed: int) -> int:
    values = [
        int(row["epochs"])
        for row in rows
        if int(row["window"]) == window
        and row["arm"] == arm
        and int(row["seed"]) == seed
    ]
    if not values:
        raise ValueError(f"没有 {window}s/{arm}/seed={seed} 的 OOF 轮数")
    return int(np.median(values))


def _predict(
    model,
    stats: dict[str, np.ndarray],
    x: np.ndarray,
    device,
) -> np.ndarray:
    x_z = (x - stats["x_mean"][None, :, None]) / stats["x_std"][None, :, None]
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(x_z).to(device)).cpu().numpy()
    prediction = prediction * stats["y_std"][None, :] + stats["y_mean"][None, :]
    return _restore_target(prediction, "log,none,log")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有记录可写入 {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    windows: tuple[int, ...],
) -> None:
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"未知 arm: {arm}")
    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(BASE.parents[0] / "splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    locked = [row for row in rows if row["split"] == "test"]
    if len(development) != 50 or len(locked) != 12:
        raise ValueError(
            f"划分应为 50/12，实际 {len(development)}/{len(locked)}"
        )
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    time_axis = experiments[0].time
    folds = list(csv.DictReader((BASE / "folds.csv").open(encoding="utf-8")))
    locked_truth = np.stack([row["_labels"] for row in locked]).astype(np.float32)

    records: list[dict] = []
    for window in windows:
        x_dev, y_dev = _real_features(development, window)
        x_locked, _ = _real_features(locked, window)
        scaler = _real_scaler(x_dev, y_dev)
        train_experiments = [
            experiments[int(Path(row["file"]).stem)] for row in development
        ]
        for seed in seeds:
            forward = _forward_synthetic(
                development, train_experiments, time_axis, window, seed, device
            )
            pca_vae = _pca_vae_synthetic(development, window, seed + 100, device)
            for arm in arms:
                x_train, y_train = _arm_arrays(arm, x_dev, y_dev, forward, pca_vae)
                epochs = _fixed_epochs(folds, window, arm, seed)
                model, stats, _, _, _ = _train_network(
                    x_train,
                    y_train,
                    None,
                    None,
                    y_train,
                    epochs,
                    seed,
                    device,
                    False,
                    validation_metric="three_sigma",
                    target_transform="log,none,log",
                    per_gas_head=True,
                    low_conc_power="1.0",
                    scaler_stats=scaler,
                )
                prediction = _predict(model, stats, x_locked, device)
                metrics = _metric_values(prediction, locked_truth)
                record = {
                    "window": window,
                    "arm": arm,
                    "seed": seed,
                    "n_dev": len(x_dev),
                    "n_train": len(x_train),
                    "epochs": epochs,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
                records.append(record)
                print(
                    f"{window}s {arm:14s} seed={seed} n={len(x_train):3d} "
                    f"epochs={epochs:3d} macro3σ={metrics['macro_three_sigma']:.3f} "
                    f"macroMAPE={metrics['macro_mape']:.3f}",
                    flush=True,
                )

    _write_csv(BASE / "locked.csv", records)
    summary: list[dict] = []
    for window in windows:
        for arm in arms:
            selected = [
                row
                for row in records
                if row["window"] == window and row["arm"] == arm
            ]
            item: dict[str, float | int | str] = {
                "window": window,
                "arm": arm,
                "seeds": len(selected),
            }
            for name in ("macro_three_sigma", "macro_mape", "macro_median", "macro_p90"):
                values = np.array([row[name] for row in selected], dtype=float)
                item[f"{name}_mean"] = round(float(values.mean()), 6)
                item[f"{name}_sd"] = (
                    round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0
                )
            summary.append(item)
    _write_csv(BASE / "locked_summary.csv", summary)
    print(f"Wrote {BASE / 'locked.csv'} and {BASE / 'locked_summary.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    args = parser.parse_args()
    run(tuple(args.arms), tuple(args.seeds), tuple(args.windows))


if __name__ == "__main__":
    main()
