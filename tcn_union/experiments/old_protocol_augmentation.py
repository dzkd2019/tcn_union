"""旧 TCN 协议下的数据增强来源对照（VAE-only / 正模型-only / 正模型+VAE / 仅真实）。

协议尽量贴近旧 gan_train.py 的最终 p10 设置：

* 固定 manifest 的 50 条开发集与 5 折；每折 40 条训练、10 条验证；
  锁定 12 条不参与本脚本。
* 所有臂共用旧 TCN 训练协议：validation_metric="three_sigma"、
  target_transform="log,none,log"、per_gas_head=True、
  low_conc_power=1.0、最多 500 epoch、早停 50。
* 增强臂都只用该折 40 条训练集拟合生成器；合成总数固定为 3×40=120。
  vae 为 120 条 PCA-VAE；forward 为 120 条正模型；forward+vae
  为 60 条正模型 + 60 条 PCA-VAE。三臂总合成数相同，避免曝光量混杂。
* 不加 0.1× 测量噪声、边缘条件复制或两套低浓度配置，因为这些属于
  旧“组合方案”，会把“增强来源”与“额外视图/重采样”混在一起。旧完整
  方案的数字（10.22% 单 seed、7.47% 六 checkpoint 集成）只作为外部参照。
* 网络初始化/打乱 seed 用 base seed；生成器 seed 用 seed + 1000 * fold，
  与旧 cross_validate 一致。每个臂各自拟合 scaler（旧 _train_network
  默认行为），不共享真实折 scaler。

输出 artifacts/augmentation/protocols/old/：

* folds.csv：逐 seed/折/臂的指标与最佳 epoch；
* oof.csv：50 条开发集的 OOF 指标；
* summary.csv：3 seed 的 OOF 均值/标准差；
* predictions.npz：OOF 预测与真实标签，供配对复算；
* run_config.json：实际运行参数。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.core.data import make_features
from tcn_union.core.device import resolve_device
from tcn_union.experiments.forward_augmentation import (
    _metric_values,
    _real_features,
)
from tcn_union.training.gan_train import MODE, _read_manifest
from tcn_union.training.train import _train_network

from sensor_array.core.data import load_experiments
from sensor_array.core.double_exponential import DoubleExponentialArrayModel
from sensor_array.core.mlp_per_gas import PerGasMLPDoubleExponentialModel

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/protocols/old"
ARMS = ("real", "vae", "forward", "forward+vae")
DEFAULT_SEEDS = (20260923, 20260924, 20260925)
EPOCHS = 500
SYNTHETIC_FACTOR = 3


def _synthetic_concentration(
    labels: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """前一半复制训练标签（循环取），其余在训练档位网格内随机组合。"""
    n_copy = count // 2
    copied = labels[np.arange(n_copy) % len(labels)]
    levels = [np.unique(labels[:, axis]) for axis in range(3)]
    random_part = np.column_stack(
        [rng.choice(levels[axis], count - n_copy) for axis in range(3)]
    )
    return np.concatenate((copied, random_part), axis=0)


def _initial_for_synthetic(
    rows: list[dict], count: int, rng: np.random.Generator
) -> np.ndarray:
    """前一半同浓度真实 R0（循环取），其余从训练折 R0 经验分布抽。"""
    initial = np.stack([row["_response"][0] for row in rows])
    n_copy = count // 2
    same = initial[np.arange(n_copy) % len(initial)]
    random_index = rng.integers(0, len(initial), size=count - n_copy)
    return np.concatenate((same, initial[random_index]), axis=0)


def _forward_synthetic(
    rows: list[dict],
    experiments: list,
    time: np.ndarray,
    seconds: int,
    seed: int,
    device,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """只用该折训练行拟合双指数+三路 MLP，生成 count 条合成特征。"""
    skeleton = DoubleExponentialArrayModel().fit(experiments)
    model = PerGasMLPDoubleExponentialModel(
        device=str(device), epochs=600, random_state=seed
    ).fit(experiments, skeleton=skeleton)
    labels = np.stack([row["_labels"] for row in rows]).astype(np.float64)
    rng = np.random.default_rng(seed)
    concentration = _synthetic_concentration(labels, count, rng)
    initial = _initial_for_synthetic(rows, count, rng)
    relative = model.predict_relative(concentration, time)
    # 历史实现：H2 合成通道被置零，旧 forward 臂的 H2 曲线实际是平线。
    # 该结果只作历史对照；修正后的完整协议见 full_protocol_augmentation.py。
    relative[:, 0, :] = 0.0
    response = initial[:, None, :] * (1.0 + relative)
    features = np.stack(
        [make_features(time, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), concentration.astype(np.float32)


def _vae_synthetic(
    rows: list[dict], seconds: int, seed: int, device, count: int
) -> tuple[np.ndarray, np.ndarray]:
    """只用该折训练行拟合 PCA-VAE，生成 count 条合成特征。"""
    config = VAEConfig(seconds=float(seconds))
    model = fit_vae(rows, config, seed, device)
    responses, labels = model.sample(count, seed + 1)
    times = np.arange(round(seconds / 0.2) + 1) * 0.2
    features = np.stack(
        [make_features(times, response.T, seconds, MODE) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_real: np.ndarray,
    seed: int,
    device,
    epochs: int,
) -> tuple[np.ndarray, int]:
    """旧 p10 训练协议；不传 scaler，让每个臂按自己的训练数组拟合。"""
    _, _, epochs_used, _, prediction = _train_network(
        x_train,
        y_train,
        x_val,
        y_val,
        y_real,
        epochs,
        seed,
        device,
        True,
        validation_metric="three_sigma",
        target_transform="log,none,log",
        per_gas_head=True,
        low_conc_power="1.0",
    )
    if prediction is None:
        raise RuntimeError("训练没有产生验证预测")
    return prediction, epochs_used


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"没有记录可写入 {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(
    folds: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    seconds: int = 10,
    epochs: int = EPOCHS,
) -> None:
    if folds < 1 or folds > 5:
        raise ValueError("folds 必须在 1–5 之间")
    if seconds < 2 or seconds > 10:
        raise ValueError("seconds 必须在 2–10 之间")
    device = resolve_device("cuda")
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    if len(development) != 50:
        raise ValueError(f"开发集应为 50 条，实际 {len(development)}")
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    time_axis = experiments[0].time
    BASE.mkdir(parents=True, exist_ok=True)
    synthetic_count = SYNTHETIC_FACTOR * 40

    fold_rows: list[dict] = []
    oof_rows: list[dict] = []
    predictions: dict[str, list[np.ndarray]] = {}
    truth: list[np.ndarray] = []
    started = time.time()

    for fold in range(folds):
        train_rows = [row for row in development if int(row["fold"]) != fold]
        val_rows = [row for row in development if int(row["fold"]) == fold]
        train_experiments = [
            experiments[int(Path(row["file"]).stem)] for row in train_rows
        ]
        x_real, y_real = _real_features(train_rows, seconds)
        x_val, y_val = _real_features(val_rows, seconds)
        truth.append(y_val)

        for seed in seeds:
            aug_seed = seed + 1000 * fold
            x_forward, y_forward = _forward_synthetic(
                train_rows,
                train_experiments,
                time_axis,
                seconds,
                aug_seed,
                device,
                synthetic_count,
            )
            x_vae, y_vae = _vae_synthetic(
                train_rows, seconds, aug_seed, device, synthetic_count
            )
            half = synthetic_count // 2
            arm_arrays = {
                "real": (x_real, y_real),
                "vae": (
                    np.concatenate((x_real, x_vae), axis=0),
                    np.concatenate((y_real, y_vae), axis=0),
                ),
                "forward": (
                    np.concatenate((x_real, x_forward), axis=0),
                    np.concatenate((y_real, y_forward), axis=0),
                ),
                "forward+vae": (
                    np.concatenate((x_real, x_forward[:half], x_vae[:half]), axis=0),
                    np.concatenate((y_real, y_forward[:half], y_vae[:half]), axis=0),
                ),
            }
            print(
                f"fold={fold} seed={seed} forward={len(y_forward)} vae={len(y_vae)}",
                flush=True,
            )
            for arm, (x_train, y_train) in arm_arrays.items():
                prediction, epochs_used = _train(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    y_real,
                    seed,
                    device,
                    epochs,
                )
                metrics = _metric_values(prediction, y_val)
                fold_rows.append(
                    {
                        "window": seconds,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "n_train": len(x_train),
                        "epochs": epochs_used,
                        **{key: round(value, 6) for key, value in metrics.items()},
                    }
                )
                predictions.setdefault(f"{arm}_s{seed}", []).append(prediction)
                print(
                    f"  {arm:12s} n={len(x_train):3d} epochs={epochs_used:3d} "
                    f"macro3σ={metrics['macro_three_sigma']:.3f} "
                    f"macroMAPE={metrics['macro_mape']:.3f}",
                    flush=True,
                )

    truth_array = np.concatenate(truth, axis=0).astype(np.float32)
    for key in predictions:
        predictions[key] = [np.concatenate(predictions[key], axis=0).astype(np.float32)]

    for seed in seeds:
        for arm in ARMS:
            key = f"{arm}_s{seed}"
            metrics = _metric_values(predictions[key][0], truth_array)
            oof_rows.append(
                {
                    "window": seconds,
                    "seed": seed,
                    "arm": arm,
                    "n": len(truth_array),
                    **{name: round(value, 6) for name, value in metrics.items()},
                }
            )

    summary_rows: list[dict] = []
    for arm in ARMS:
        selected = [row for row in oof_rows if row["arm"] == arm]
        summary: dict[str, float | int | str] = {
            "window": seconds,
            "arm": arm,
            "seeds": len(selected),
        }
        for name in ("macro_three_sigma", "macro_mape", "macro_median", "macro_p90"):
            values = np.array([row[name] for row in selected], dtype=float)
            summary[f"{name}_mean"] = round(float(values.mean()), 6)
            summary[f"{name}_sd"] = (
                round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0
            )
        summary_rows.append(summary)

    _write_csv(BASE / "folds.csv", fold_rows)
    _write_csv(BASE / "oof.csv", oof_rows)
    _write_csv(BASE / "summary.csv", summary_rows)
    np.savez_compressed(
        BASE / "predictions.npz",
        truth=truth_array,
        **{key: value[0] for key, value in predictions.items()},
    )
    config = {
        "folds": folds,
        "seeds": list(seeds),
        "window": seconds,
        "epochs": epochs,
        "arms": list(ARMS),
        "synthetic_count_per_arm": synthetic_count,
        "synthetic_ratio": SYNTHETIC_FACTOR,
        "noise": False,
        "edge_copies": 0,
        "low_conc_configs": ["1.0"],
        "elapsed_seconds": round(time.time() - started, 1),
        "mode": MODE,
        "device": str(device),
    }
    (BASE / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {BASE} in {config['elapsed_seconds']} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(args.seeds),
        seconds=args.seconds,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
