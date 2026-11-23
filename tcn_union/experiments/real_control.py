"""增强臂的“双倍真实样本”对照。

增强臂每折有 80 条训练样本、约 10 个 batch/epoch；仅真实臂只有 40 条、5 个
batch。把真实样本重复一次得到 n=80 的对照，用来区分“更多更新/更大样本数”
与“合成数据本身”的收益。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.experiments.forward_augmentation import (
    BASE,
    DEFAULT_SEEDS,
    DEFAULT_WINDOWS,
    _metric_values,
    _real_features,
    _real_scaler,
)
from tcn_union.training.gan_train import _read_manifest
from tcn_union.training.train import _train_network


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有记录可写入 {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(seeds: tuple[int, ...], windows: tuple[int, ...], epochs: int) -> None:
    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(BASE.parents[0] / "splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    records: list[dict] = []
    for window in windows:
        for fold in range(5):
            train_rows = [row for row in development if int(row["fold"]) != fold]
            val_rows = [row for row in development if int(row["fold"]) == fold]
            x_real, y_real = _real_features(train_rows, window)
            x_val, y_val = _real_features(val_rows, window)
            scaler = _real_scaler(x_real, y_real)
            x_train = np.concatenate((x_real, x_real), axis=0)
            y_train = np.concatenate((y_real, y_real), axis=0)
            for seed in seeds:
                _, _, epochs_used, _, prediction = _train_network(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    y_real,
                    epochs,
                    seed + 1000 * fold,
                    device,
                    True,
                    validation_metric="three_sigma",
                    target_transform="log,none,log",
                    per_gas_head=True,
                    low_conc_power="1.0",
                    scaler_stats=scaler,
                )
                if prediction is None:
                    raise RuntimeError("训练没有产生验证预测")
                metrics = _metric_values(prediction, y_val)
                record = {
                    "window": window,
                    "fold": fold,
                    "seed": seed,
                    "arm": "real_dup",
                    "n_train": len(x_train),
                    "epochs": epochs_used,
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
                records.append(record)
                print(
                    f"{window}s fold={fold} seed={seed} epochs={epochs_used} "
                    f"macro3σ={metrics['macro_three_sigma']:.3f} "
                    f"macroMAPE={metrics['macro_mape']:.3f}",
                    flush=True,
                )
    _write_csv(BASE / "real_dup.csv", records)
    summary: list[dict] = []
    for window in windows:
        selected = [row for row in records if row["window"] == window]
        item: dict[str, float | int] = {"window": window, "arm": "real_dup"}
        for name in ("macro_three_sigma", "macro_mape", "macro_median", "macro_p90"):
            values = np.array([row[name] for row in selected], dtype=float)
            item[f"{name}_mean"] = round(float(values.mean()), 6)
            item[f"{name}_sd"] = round(float(values.std(ddof=1)), 6)
        summary.append(item)
    _write_csv(BASE / "real_dup_summary.csv", summary)
    print(f"Wrote {BASE / 'real_dup.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--windows", nargs="+", type=int, default=list(DEFAULT_WINDOWS))
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    run(tuple(args.seeds), tuple(args.windows), args.epochs)


if __name__ == "__main__":
    main()
