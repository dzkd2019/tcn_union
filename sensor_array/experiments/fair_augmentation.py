"""P5：等曝光 / 等真实预算的合成数据公平对照。

只使用 50 条开发实验与固定五折；12 条锁定测试不参与。三个臂：

- real：每步 8 条真实，150 epoch；
- real+synthetic-equal：每步 8 真实 + 8 合成（batch 16），150 epoch；
- pretrain+finetune-equal：100 epoch 合成预训练 + 150 epoch 真实微调。

5 s 与 10 s 分别用对应长度的前缀训练一个模型（26 / 51 点），使两个窗口的
误差都在训练分布内；两个窗口上的三臂协议相同。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
from scipy import stats

from tcn_union.cli.split import FOLDS, TEST_IDS

from ..core.data import GASES, load_experiments
from ..core.grey_box import fit_experiment_shapes
from ..core.hybrid import HybridArrayModel
from ..paths import AUGMENTATION, ensure
from .inverse_augmentation import (
    SYNTHETIC,
    TIME,
    _features,
    _train,
)

BATCH = 8
EPOCHS = 150
PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 150
FINETUNE_LR = 1e-3
TMAX = (5.0, 10.0)
ARMS = ("real", "real+synthetic-equal", "pretrain+finetune-equal")
METRICS = (
    "mape_macro",
    "median_are_macro",
    "nmae_macro",
    "mape_H2",
    "median_are_H2",
    "nmae_H2",
    "mape_NH3",
    "median_are_NH3",
    "nmae_NH3",
    "mape_NO2",
    "median_are_NO2",
    "nmae_NO2",
)


def _prepare_fold(experiments, by_id, shapes, fold: int, seed: int) -> dict:
    """构造一折的真实/合成训练数据与 5 s、10 s 测试前缀。"""
    dev_ids = sorted(set(range(62)) - TEST_IDS)
    train_ids = sorted(set(dev_ids) - FOLDS[fold])
    test_ids = sorted(FOLDS[fold])
    train = [by_id[identifier] for identifier in train_ids]
    shape_indices = [
        index
        for index, experiment in enumerate(experiments)
        if experiment.file_id in train_ids
    ]
    shape_init = tuple(part[shape_indices] for part in shapes)
    model = HybridArrayModel().fit(train, shape_init=shape_init)

    concentration = np.stack([experiment.concentration for experiment in train])
    low, high = concentration.min(axis=0), concentration.max(axis=0)
    rng = np.random.default_rng(seed + fold)
    synthetic_concentration = np.exp(
        rng.uniform(np.log(low), np.log(high), size=(SYNTHETIC, 3))
    )
    synthetic = model.predict_relative(synthetic_concentration, TIME)
    residual = np.stack(
        [experiment.relative[: len(TIME)] for experiment in train]
    ) - model.predict_relative(concentration, TIME)
    synthetic = synthetic + residual[rng.integers(0, len(residual), SYNTHETIC)]

    mean = np.log(concentration).mean(axis=0)
    scale = np.log(concentration).std(axis=0)
    real_x = np.stack([_features(experiment.relative[: len(TIME)]) for experiment in train])
    real_y = (np.log(concentration) - mean) / scale
    synthetic_x = np.stack([_features(curve) for curve in synthetic])
    synthetic_y = (np.log(synthetic_concentration) - mean) / scale
    x_mean = real_x.mean(axis=(0, 2), keepdims=True)
    x_std = real_x.std(axis=(0, 2), keepdims=True) + 1e-8
    real_x = (real_x - x_mean) / x_std
    synthetic_x = (synthetic_x - x_mean) / x_std

    test_y = np.stack([by_id[identifier].concentration for identifier in test_ids])
    test_x_raw = np.stack(
        [_features(by_id[identifier].relative[: len(TIME)]) for identifier in test_ids]
    )
    test_x = (test_x_raw - x_mean) / x_std
    return {
        "train": train,
        "test_ids": test_ids,
        "concentration": concentration,
        "real_x": real_x,
        "real_y": real_y,
        "synthetic_x": synthetic_x,
        "synthetic_y": synthetic_y,
        "test_x": test_x,
        "test_y": test_y,
        "mean": mean,
        "scale": scale,
        "label_range": np.ptp(concentration, axis=0),
    }


def _metrics(
    model,
    data: dict,
    tmax: float,
) -> dict:
    """在一个前缀上评估三气体浓度误差。"""
    count = int(np.searchsorted(TIME, tmax, side="right"))
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(data["test_x"][:, :, :count])).numpy()
    prediction = np.exp(prediction * data["scale"] + data["mean"])
    truth = data["test_y"]
    relative = np.abs(prediction / truth - 1.0)
    absolute = np.abs(prediction - truth)
    result = {
        "mape_macro": float(relative.mean() * 100.0),
        "median_are_macro": float(np.median(relative) * 100.0),
        "nmae_macro": float((absolute / data["label_range"]).mean() * 100.0),
    }
    for index, gas in enumerate(GASES):
        result[f"mape_{gas}"] = float(relative[:, index].mean() * 100.0)
        result[f"median_are_{gas}"] = float(np.median(relative[:, index]) * 100.0)
        result[f"nmae_{gas}"] = float(
            (absolute[:, index] / data["label_range"][index]).mean() * 100.0
        )
    return result


def _budget() -> pd.DataFrame:
    """记录每个臂的真实/合成曝光与更新步数，便于核对公平性。"""
    return pd.DataFrame(
        [
            {
                "arm": "real",
                "real_per_batch": BATCH,
                "synthetic_per_batch": 0,
                "pretrain_epochs": 0,
                "finetune_epochs": EPOCHS,
                "real_samples": BATCH * 10 * EPOCHS,
                "synthetic_samples": 0,
                "real_updates": 10 * EPOCHS,
                "synthetic_updates": 0,
                "optimizer_updates": 10 * EPOCHS,
            },
            {
                "arm": "real+synthetic-equal",
                "real_per_batch": BATCH,
                "synthetic_per_batch": BATCH,
                "pretrain_epochs": 0,
                "finetune_epochs": EPOCHS,
                "real_samples": BATCH * 10 * EPOCHS,
                "synthetic_samples": BATCH * 10 * EPOCHS,
                "real_updates": 10 * EPOCHS,
                "synthetic_updates": 10 * EPOCHS,
                "optimizer_updates": 10 * EPOCHS,
            },
            {
                "arm": "pretrain+finetune-equal",
                "real_per_batch": BATCH,
                "synthetic_per_batch": BATCH,
                "pretrain_epochs": PRETRAIN_EPOCHS,
                "finetune_epochs": FINETUNE_EPOCHS,
                "real_samples": BATCH * 10 * FINETUNE_EPOCHS,
                "synthetic_samples": BATCH * 10 * PRETRAIN_EPOCHS,
                "real_updates": 10 * FINETUNE_EPOCHS,
                "synthetic_updates": 10 * PRETRAIN_EPOCHS,
                "optimizer_updates": 10 * (PRETRAIN_EPOCHS + FINETUNE_EPOCHS),
            },
        ]
    )


def run_seed(seed: int) -> None:
    experiments = load_experiments()
    by_id = {experiment.file_id: experiment for experiment in experiments}
    shapes = fit_experiment_shapes(experiments)
    rows = []
    for fold, _ in enumerate(FOLDS):
        data = _prepare_fold(experiments, by_id, shapes, fold, seed)
        for tmax in TMAX:
            count = int(np.searchsorted(TIME, tmax, side="right"))
            real_x = data["real_x"][:, :, :count]
            synthetic_x = data["synthetic_x"][:, :, :count]
            models = {
                "real": _train(
                    real_x, data["real_y"], None, None, seed + fold
                ),
                "real+synthetic-equal": _train(
                    real_x,
                    data["real_y"],
                    synthetic_x,
                    data["synthetic_y"],
                    seed + fold,
                    batch_size=2 * BATCH,
                ),
            }
            pretrained = _train(
                synthetic_x,
                data["synthetic_y"],
                None,
                None,
                seed + fold,
                epochs=PRETRAIN_EPOCHS,
            )
            models["pretrain+finetune-equal"] = _train(
                real_x,
                data["real_y"],
                None,
                None,
                seed + fold,
                epochs=FINETUNE_EPOCHS,
                lr=FINETUNE_LR,
                model=pretrained,
            )
            for arm, model in models.items():
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "arm": arm,
                        "tmax": tmax,
                        **_metrics(model, data, tmax),
                    }
                )
        print(f"fold {fold} done", flush=True)
    frame = pd.DataFrame(rows)
    ensure()
    frame.to_csv(
        AUGMENTATION / f"fair_comparison_seed{seed}.csv",
        index=False,
        encoding="utf-8",
    )
    print(frame.groupby(["tmax", "arm"])[["mape_macro", "median_are_macro"]].agg(["mean", "std"]))


def combine() -> None:
    paths = sorted(AUGMENTATION.glob("fair_comparison_seed*.csv"))
    if not paths:
        raise FileNotFoundError("没有 fair_comparison_seed*.csv，请先跑 --seed")
    frame = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    frame.to_csv(AUGMENTATION / "fair_comparison.csv", index=False, encoding="utf-8")
    _budget().to_csv(AUGMENTATION / "fair_budget.csv", index=False, encoding="utf-8")

    rows = []
    for tmax in TMAX:
        baseline = frame[
            (frame["arm"] == "real") & (frame["tmax"] == tmax)
        ].set_index(["seed", "fold"])
        for arm in ARMS:
            subset = frame[(frame["arm"] == arm) & (frame["tmax"] == tmax)].set_index(
                ["seed", "fold"]
            )
            for metric in METRICS:
                values = subset[metric]
                paired = baseline.loc[values.index, metric] if arm != "real" else values
                delta = values - paired
                if arm == "real" or len(delta) < 2 or np.allclose(delta, 0.0):
                    p_value = float("nan")
                    low = high = float(delta.mean())
                else:
                    p_value = float(stats.ttest_rel(values, paired).pvalue)
                    sem = float(stats.sem(delta))
                    low, high = stats.t.interval(
                        0.95, len(delta) - 1, loc=float(delta.mean()), scale=sem
                    ) if sem > 0 else (float(delta.mean()), float(delta.mean()))
                rows.append(
                    {
                        "tmax": tmax,
                        "arm": arm,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "delta_mean": float(delta.mean()),
                        "ci_low": float(low),
                        "ci_high": float(high),
                        "p_vs_real": p_value,
                        "n_pairs": int(len(delta)),
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(AUGMENTATION / "fair_summary.csv", index=False, encoding="utf-8")
    print(summary[summary["metric"].isin(["mape_macro", "median_are_macro"])].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="P5 等曝光/等预算合成数据对照")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，写入独立 CSV")
    parser.add_argument("--combine", action="store_true", help="合并各种子并计算配对检验")
    arguments = parser.parse_args()
    if arguments.combine:
        combine()
    else:
        run_seed(arguments.seed)


if __name__ == "__main__":
    main()
