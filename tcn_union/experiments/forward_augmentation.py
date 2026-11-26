"""正模型、PCA-VAE 与二者混合的逆 TCN 数据增强对比。

协议（对应 sensor_array/docs/plan_20261123.md 第 5 节）：

* 固定 manifest 的 50 条开发集与 5 折；每折 40 条训练、10 条验证；
  锁定 12 条不参与本脚本，也不参与合成、早停或选型。
* 每折只用该折 40 条真实训练集拟合正模型（双指数骨架 + 三路独立 MLP）
  和 PCA-VAE；合成样本不接触验证折。
* 四臂：real、forward、pca_vae、forward+pca_vae。单臂合成数与真实训练数
  1:1；混合臂为 40 条正模型 + 40 条 PCA-VAE。
* 所有臂共用真实训练折拟合的 scaler、TCN 结构、训练轮数、早停指标
  （验证集宏 3σ）、逐气体输出头和低浓度加权。
* 每个折/窗口先训练四个臂，使同折比较共享同一批真实样本与同一验证折。

输出目录为 artifacts/augmentation/protocols/forward/：

* folds.csv：每 seed、窗口、折、臂的逐折指标；
* oof.csv：每 seed、窗口、臂在 50 条开发集上的 OOF 指标；
* summary.csv：3 seed 的 OOF 均值/标准差；
* predictions.npz：OOF 预测与真实标签，便于配对复算；
* run_config.json：实际运行参数。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.core.data import make_features
from tcn_union.core.device import resolve_device
from tcn_union.training.gan_train import MODE, _read_manifest
from tcn_union.training.train import _train_network, fit_scalers

from sensor_array.core.data import load_experiments
from sensor_array.core.double_exponential import DoubleExponentialArrayModel
from sensor_array.core.mlp_per_gas import PerGasMLPDoubleExponentialModel

GASES = ("H2", "NH3", "NO2")
ARMS = ("real", "forward", "pca_vae", "forward+pca_vae")
DEFAULT_SEEDS = (20260923, 20260924, 20260925)
DEFAULT_WINDOWS = (5, 10)
EPOCHS = 500
BASE = ROOT / "artifacts/augmentation/protocols/forward"


def _metric_values(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """逐气体 3σ/MAPE/中位绝对相对误差/p90，并取宏平均。"""
    out: dict[str, float] = {}
    for axis, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, axis] - truth[:, axis]) / np.maximum(
            np.abs(truth[:, axis]), 1e-9
        )
        absolute = np.abs(relative)
        out[f"{gas}_three_sigma"] = float(3.0 * np.std(relative, ddof=1))
        out[f"{gas}_mape"] = float(np.mean(absolute))
        out[f"{gas}_median"] = float(np.median(absolute))
        out[f"{gas}_p90"] = float(np.quantile(absolute, 0.9))
    for key in ("three_sigma", "mape", "median", "p90"):
        out[f"macro_{key}"] = float(np.mean([out[f"{gas}_{key}"] for gas in GASES]))
    return out


def _real_scaler(x_real: np.ndarray, y_real: np.ndarray) -> dict[str, np.ndarray]:
    """真实训练折的 scaler；所有臂共用，避免合成标签改变尺度。"""
    y_fit = y_real.copy()
    y_fit[:, 0] = np.log(np.maximum(y_fit[:, 0], 1e-6))
    y_fit[:, 2] = np.log(np.maximum(y_fit[:, 2], 1e-6))
    return fit_scalers(x_real, y_fit)


def _real_features(rows: list[dict], seconds: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [make_features(row["_time"], row["_response"], seconds, MODE) for row in rows]
    )
    y = np.stack([row["_labels"] for row in rows]).astype(np.float32)
    return x.astype(np.float32), y


def _synthetic_concentration(
    labels: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """一半复制训练标签，一半在训练档位网格内随机组合。"""
    n = len(labels)
    n_copy = n // 2
    levels = [np.unique(labels[:, axis]) for axis in range(3)]
    random_part = np.column_stack(
        [rng.choice(levels[axis], n - n_copy) for axis in range(3)]
    )
    return np.concatenate((labels[:n_copy], random_part), axis=0)


def _initial_for_synthetic(
    rows: list[dict], n: int, rng: np.random.Generator
) -> np.ndarray:
    """一半同浓度真实 R0，一半从训练折 R0 经验分布抽。"""
    initial = np.stack([row["_response"][0] for row in rows])
    n_copy = n // 2
    same = initial[:n_copy]
    random_index = rng.integers(0, len(initial), size=n - n_copy)
    return np.concatenate((same, initial[random_index]), axis=0)


def _forward_synthetic(
    train_rows: list[dict],
    train_experiments: list,
    time: np.ndarray,
    seconds: int,
    seed: int,
    device,
) -> tuple[np.ndarray, np.ndarray]:
    """只用 40 条训练实验拟合正模型，生成 40 条合成特征。"""
    skeleton = DoubleExponentialArrayModel().fit(train_experiments)
    forward = PerGasMLPDoubleExponentialModel(
        device=str(device), epochs=600, random_state=seed
    ).fit(train_experiments, skeleton=skeleton)
    labels = np.stack([row["_labels"] for row in train_rows]).astype(np.float64)
    rng = np.random.default_rng(seed)
    concentration = _synthetic_concentration(labels, rng)
    initial = _initial_for_synthetic(train_rows, len(concentration), rng)
    relative = forward.predict_relative(concentration, time)
    relative[:, 0, :] = 0.0
    response = initial[:, None, :] * (1.0 + relative)
    features = np.stack(
        [make_features(time, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), concentration.astype(np.float32)


def _pca_vae_synthetic(
    train_rows: list[dict], seconds: int, seed: int, device
) -> tuple[np.ndarray, np.ndarray]:
    """复用第一阶段 PCA-VAE 配置，只在该折 40 条上拟合。"""
    config = VAEConfig(seconds=float(seconds))
    model = fit_vae(train_rows, config, seed, device)
    responses, labels = model.sample(len(train_rows), seed + 1)
    times = np.arange(round(seconds / 0.2) + 1) * 0.2
    features = np.stack(
        [make_features(times, response.T, seconds, MODE) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _arm_arrays(
    arm: str,
    x_real: np.ndarray,
    y_real: np.ndarray,
    forward: tuple[np.ndarray, np.ndarray],
    pca_vae: tuple[np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x_parts = [x_real]
    y_parts = [y_real]
    if arm in ("forward", "forward+pca_vae"):
        x_parts.append(forward[0])
        y_parts.append(forward[1])
    if arm in ("pca_vae", "forward+pca_vae"):
        x_parts.append(pca_vae[0])
        y_parts.append(pca_vae[1])
    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def _train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_real: np.ndarray,
    scaler: dict[str, np.ndarray],
    seed: int,
    device,
    epochs: int,
) -> tuple[np.ndarray, int]:
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
        scaler_stats=scaler,
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
    windows: tuple[int, ...] = DEFAULT_WINDOWS,
    arms: tuple[str, ...] = ARMS,
    epochs: int = EPOCHS,
) -> None:
    if folds < 1 or folds > 5:
        raise ValueError("folds 必须在 1–5 之间")
    if epochs < 1:
        raise ValueError("epochs 必须为正")
    for arm in arms:
        if arm not in ARMS:
            raise ValueError(f"未知 arm: {arm}")
    device = resolve_device("cuda")
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    if len(development) != 50:
        raise ValueError(f"开发集应为 50 条，实际 {len(development)}")
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    time_axis = experiments[0].time
    BASE.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    oof_rows: list[dict] = []
    predictions: dict[str, list[np.ndarray]] = {}
    truth: dict[int, list[np.ndarray]] = {}
    started = time.time()

    for window in windows:
        for fold in range(folds):
            train_rows = [row for row in development if int(row["fold"]) != fold]
            val_rows = [row for row in development if int(row["fold"]) == fold]
            train_experiments = [
                experiments[int(Path(row["file"]).stem)] for row in train_rows
            ]
            x_real, y_real = _real_features(train_rows, window)
            x_val, y_val = _real_features(val_rows, window)
            scaler = _real_scaler(x_real, y_real)
            fold_pred: dict[tuple[int, str], np.ndarray] = {}

            # 正模型与 PCA-VAE 都只拟合该折训练集；两者都不看验证折。
            for seed in seeds:
                forward = _forward_synthetic(
                    train_rows, train_experiments, time_axis, window, seed, device
                )
                pca_vae = _pca_vae_synthetic(train_rows, window, seed + 100, device)
                print(
                    f"window={window}s fold={fold} seed={seed} "
                    f"forward={len(forward[1])} pca={len(pca_vae[1])}",
                    flush=True,
                )
                for arm in arms:
                    x_train, y_train = _arm_arrays(arm, x_real, y_real, forward, pca_vae)
                    prediction, epochs_used = _train(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        y_real,
                        scaler,
                        seed + 1000 * fold,
                        device,
                        epochs,
                    )
                    metrics = _metric_values(prediction, y_val)
                    row = {
                        "window": window,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "n_train": len(x_train),
                        "epochs": epochs_used,
                        **{key: round(value, 6) for key, value in metrics.items()},
                    }
                    fold_rows.append(row)
                    fold_pred[(seed, arm)] = prediction
                    print(
                        f"  {arm:14s} n={len(x_train):3d} epochs={epochs_used:3d} "
                        f"macro3σ={metrics['macro_three_sigma']:.3f} "
                        f"macroMAPE={metrics['macro_mape']:.3f}",
                        flush=True,
                    )

            for (seed, arm), value in fold_pred.items():
                key = f"{arm}_s{seed}_w{window}"
                predictions.setdefault(key, []).append(value)
            truth.setdefault(window, []).append(y_val)

    # 按 seed/arm/window 拼成 50 条 OOF，计算汇总指标。
    for window in windows:
        truth[window] = [np.concatenate(truth[window], axis=0).astype(np.float32)]
    for key in predictions:
        predictions[key] = [np.concatenate(predictions[key], axis=0).astype(np.float32)]

    for window in windows:
        for seed in seeds:
            for arm in arms:
                key = f"{arm}_s{seed}_w{window}"
                metrics = _metric_values(predictions[key][0], truth[window][0])
                oof_rows.append(
                    {
                        "window": window,
                        "seed": seed,
                        "arm": arm,
                        "n": len(predictions[key][0]),
                        **{name: round(value, 6) for name, value in metrics.items()},
                    }
                )

    summary_rows: list[dict] = []
    for window in windows:
        for arm in arms:
            selected = [row for row in oof_rows if row["window"] == window and row["arm"] == arm]
            summary: dict[str, float | int | str] = {
                "window": window,
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
        **{key: value[0] for key, value in predictions.items()},
        **{f"truth_w{window}": truth[window][0] for window in windows},
    )
    config = {
        "folds": folds,
        "seeds": list(seeds),
        "windows": list(windows),
        "arms": list(arms),
        "epochs": epochs,
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
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS))
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(args.seeds),
        windows=tuple(args.windows),
        arms=tuple(args.arms),
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
