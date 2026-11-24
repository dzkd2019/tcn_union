"""完整旧 TCN 协议下的四臂增强对照（real / VAE / forward / forward+VAE）。

相对 old_protocol_augmentation.py，本脚本补上旧最终 p10 方案里被省略的
三件事，并把它们固定成同一套协议：

* 0.1×MAD 测量噪声视图（每折 +40 条；所有增强臂都加，real 臂保持干净）；
* 边缘浓度条件复制（每个边缘浓度组合复制 4 次；只对生成器可用，
  正模型与 VAE 各自按条件生成）；
* 两套低浓度配置 1.0 与 0,0.3,1.0，各 3 seed，共 6 个 checkpoint；
  预测先按 6 个 checkpoint 平均，再裁剪到开发集逐气体浓度范围。

四臂的合成总量仍保持同量级：每臂基础合成 120 条（3×40），
forward+vae 为 60 条正模型 + 60 条 VAE；边缘复制按来源等分。
real 臂只使用真实训练折，不加噪声/合成/边缘，作为干净基线。
锁定 12 条不参与本脚本。

输出目录 artifacts/full_protocol_augmentation/：

* folds.csv：逐 fold/seed/config/arm 的指标与最佳 epoch；
* oof_seed.csv：每个 seed 的 2 配置平均 OOF；
* oof.csv：6 checkpoint 平均后的 OOF；
* summary.csv：3 seed 的均值/标准差与 6 checkpoint 集成；
* paired.csv：候选臂相对 real 的逐 seed/折配对统计；
* predictions.npz：OOF 预测、真实标签与文件号；
* summary.md：自动生成的简表；
* run_config.json：实际运行参数。
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from tcn_union.augmentation.noise import NoiseConfig, noisy_features
from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.core.data import make_features
from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.experiments.old_protocol_augmentation import (
    _initial_for_synthetic,
    _metric_values,
    _real_features,
    _synthetic_concentration,
    _write_csv,
)
from tcn_union.training.gan_train import MODE, _edge_case_labels, _read_manifest
from tcn_union.training.train import _train_network, fit_scalers

from sensor_array.core.data import load_experiments
from sensor_array.core.double_exponential import DoubleExponentialArrayModel
from sensor_array.core.mlp_per_gas import PerGasMLPDoubleExponentialModel

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/full_protocol_augmentation"
ARMS = ("real", "vae", "forward", "forward+vae", "forward_pretrain", "forward_unseen", "forward_h2", "forward_seen")
DEFAULT_SEEDS = (20260923, 20260924, 20260925)
DEFAULT_CONFIGS = ("1.0", "0,0.3,1.0")
EPOCHS = 500
PRETRAIN_EPOCHS = 200
SYNTHETIC_FACTOR = 3
NOISE_SCALE = 0.1
EDGE_COPIES = 4
PAIR_METRICS = (
    "macro_three_sigma",
    "macro_mape",
    "macro_median",
    "macro_p90",
    "H2_three_sigma",
    "NH3_three_sigma",
    "NO2_three_sigma",
)


def _times(seconds: int) -> np.ndarray:
    return np.arange(round(seconds / 0.2) + 1) * 0.2


def _grid_combinations(labels: np.ndarray) -> np.ndarray:
    levels = [np.unique(labels[:, axis]) for axis in range(3)]
    return np.array(list(product(*levels)), dtype=np.float64)


def _synthetic_concentration_excluding(
    labels: np.ndarray,
    excluded: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """一半复制训练标签，一半使用训练档位网格内、非验证折的新组合。"""
    n_copy = count // 2
    copied = labels[np.arange(n_copy) % len(labels)]
    grid = _grid_combinations(labels)
    train_set = {tuple(row) for row in labels.tolist()}
    excluded_set = {tuple(row) for row in excluded.tolist()}
    allowed = np.array(
        [row for row in grid if tuple(row) not in train_set and tuple(row) not in excluded_set],
        dtype=np.float64,
    )
    if not len(allowed):
        return _synthetic_concentration(labels, count, rng)
    reps = int(np.ceil((count - n_copy) / len(allowed)))
    order = []
    for _ in range(reps):
        order.append(allowed[rng.permutation(len(allowed))])
    new_part = np.concatenate(order, axis=0)[: count - n_copy]
    return np.concatenate((copied, new_part), axis=0)


def _unseen_concentrations(
    labels: np.ndarray,
    excluded: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """在训练档位网格内，循环采样既非训练也非验证折的浓度组合。"""
    grid = _grid_combinations(labels)
    train_set = {tuple(row) for row in labels.tolist()}
    excluded_set = {tuple(row) for row in excluded.tolist()}
    allowed = np.array(
        [row for row in grid if tuple(row) not in train_set and tuple(row) not in excluded_set],
        dtype=np.float64,
    )
    if not len(allowed):
        return _synthetic_concentration(labels, count, rng)
    reps = int(np.ceil(count / len(allowed)))
    parts = [allowed[rng.permutation(len(allowed))] for _ in range(reps)]
    return np.concatenate(parts, axis=0)[:count]


def _training_concentrations(
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """从训练标签中有放回抽样，不产生新的浓度组合。"""
    index = rng.integers(0, len(labels), size=count)
    return labels[index]


def _random_initial(rows: list[dict], count: int, rng: np.random.Generator) -> np.ndarray:
    initial = np.stack([row["_response"][0] for row in rows])
    index = rng.integers(0, len(initial), size=count)
    return initial[index]


def _forward_features(
    model: PerGasMLPDoubleExponentialModel,
    labels: np.ndarray,
    initial: np.ndarray,
    seconds: int,
) -> tuple[np.ndarray, np.ndarray]:
    """用正模型生成绝对响应，再转成 TCN 前缀特征。"""
    times = _times(seconds)
    relative = model.predict_relative(labels, times)
    response = initial[:, None, :] * (1.0 + relative)
    features = np.stack(
        [make_features(times, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _vae_features(
    model,
    count: int,
    seed: int,
    seconds: int,
    condition_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    responses, labels = model.sample(
        count, seed, condition_labels=condition_labels
    )
    times = _times(seconds)
    features = np.stack(
        [make_features(times, response.T, seconds, MODE) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _initial_for_labels(
    rows: list[dict], labels: np.ndarray
) -> np.ndarray:
    """按标签查真实 R0；查不到时从训练折 R0 经验分布抽。"""
    lookup = {
        tuple(np.asarray(row["_labels"], dtype=np.float64)): row["_response"][0]
        for row in rows
    }
    values = np.stack([row["_response"][0] for row in rows])
    rng = np.random.default_rng(0)
    initial = []
    for label in labels:
        key = tuple(np.asarray(label, dtype=np.float64))
        if key in lookup:
            initial.append(lookup[key])
        else:
            initial.append(values[rng.integers(0, len(values))])
    return np.stack(initial)


def _fit_generators(
    train_rows: list[dict],
    train_experiments: list,
    seed: int,
    device,
    need_forward: bool,
    need_vae: bool,
):
    skeleton = None
    forward = None
    vae = None
    if need_forward:
        skeleton = DoubleExponentialArrayModel().fit(train_experiments)
        forward = PerGasMLPDoubleExponentialModel(
            device=str(device), epochs=600, random_state=seed
        ).fit(train_experiments, skeleton=skeleton)
    if need_vae:
        vae = fit_vae(train_rows, VAEConfig(seconds=10.0), seed, device)
    return forward, vae


def _synthetic_for_arm(
    arm: str,
    forward_base: tuple[np.ndarray, np.ndarray],
    vae_base: tuple[np.ndarray, np.ndarray],
    edge_labels: np.ndarray,
    edge_forward: tuple[np.ndarray, np.ndarray],
    edge_vae: tuple[np.ndarray, np.ndarray],
    forward_unseen_base: tuple[np.ndarray, np.ndarray],
    forward_seen_base: tuple[np.ndarray, np.ndarray],
    synthetic_count: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """按臂拼基础合成与边缘复制，返回 (x, y, synthetic, edge)。"""
    if arm == "vae":
        x = np.concatenate((vae_base[0], edge_vae[0]), axis=0)
        y = np.concatenate((vae_base[1], edge_vae[1]), axis=0)
        return x, y, len(vae_base[1]), len(edge_vae[1])
    if arm in {"forward", "forward_pretrain"}:
        x = np.concatenate((forward_base[0], edge_forward[0]), axis=0)
        y = np.concatenate((forward_base[1], edge_forward[1]), axis=0)
        return x, y, len(forward_base[1]), len(edge_forward[1])
    if arm in {"forward_unseen", "forward_h2"}:
        x = np.concatenate((forward_unseen_base[0], edge_forward[0]), axis=0)
        y = np.concatenate((forward_unseen_base[1], edge_forward[1]), axis=0)
        return x, y, len(forward_unseen_base[1]), len(edge_forward[1])
    if arm == "forward_seen":
        x = np.concatenate((forward_seen_base[0], edge_forward[0]), axis=0)
        y = np.concatenate((forward_seen_base[1], edge_forward[1]), axis=0)
        return x, y, len(forward_seen_base[1]), len(edge_forward[1])
    if arm == "forward+vae":
        half = synthetic_count // 2
        n_edge_half = len(edge_labels) * (EDGE_COPIES // 2)
        x = np.concatenate(
            (
                forward_base[0][:half],
                vae_base[0][:half],
                edge_forward[0][:n_edge_half],
                edge_vae[0][:n_edge_half],
            ),
            axis=0,
        )
        y = np.concatenate(
            (
                forward_base[1][:half],
                vae_base[1][:half],
                edge_forward[1][:n_edge_half],
                edge_vae[1][:n_edge_half],
            ),
            axis=0,
        )
        return x, y, half * 2, n_edge_half * 2
    raise ValueError(f"未知增强臂 {arm}")


def _train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_real: np.ndarray,
    seed: int,
    device,
    epochs: int,
    low_conc_power: str,
    scaler_stats: dict[str, np.ndarray] | None = None,
    sample_weight: np.ndarray | None = None,
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
        low_conc_power=low_conc_power,
        scaler_stats=scaler_stats,
    )
    if prediction is None:
        raise RuntimeError("训练没有产生验证预测")
    return prediction, epochs_used


def _train_pretrain(
    x_pretrain: np.ndarray,
    y_pretrain: np.ndarray,
    x_finetune: np.ndarray,
    y_finetune: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_real: np.ndarray,
    seed: int,
    device,
    epochs: int,
    low_conc_power: str,
    scaler_stats: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    """先在正模型合成曲线上预训练，再用真实+噪声+边缘数据微调。"""
    model, _, _, _, _ = _train_network(
        x_pretrain,
        y_pretrain,
        None,
        None,
        y_pretrain,
        PRETRAIN_EPOCHS,
        seed,
        device,
        False,
        target_transform="log,none,log",
        per_gas_head=True,
        low_conc_power=low_conc_power,
    )
    _, _, epochs_used, _, prediction = _train_network(
        x_finetune,
        y_finetune,
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
        low_conc_power=low_conc_power,
        initial_state=model.state_dict(),
        scaler_stats=scaler_stats,
    )
    if prediction is None:
        raise RuntimeError("预训练微调没有产生验证预测")
    return prediction, epochs_used


def _clip(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.clip(
        prediction,
        labels.min(axis=0),
        labels.max(axis=0),
    )


def _transformed_labels(labels: np.ndarray) -> np.ndarray:
    """与训练目标一致：H2/NO2 取 log，NH3 保持原空间。"""
    transformed = np.asarray(labels, dtype=np.float64).copy()
    transformed[:, 0] = np.log(np.maximum(transformed[:, 0], 1e-6))
    transformed[:, 2] = np.log(np.maximum(transformed[:, 2], 1e-6))
    return transformed


def _bootstrap_ci(
    values: np.ndarray, seed: int = 0, draws: int = 10000
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_rows(
    fold_rows: list[dict], metrics: tuple[str, ...], arms: tuple[str, ...]
) -> list[dict]:
    """候选臂相对 real 的逐 seed/折配对差；负值表示候选更好。"""
    if "real" not in arms:
        return []
    rows = []
    for candidate in ("vae", "forward", "forward+vae", "forward_pretrain", "forward_unseen", "forward_h2", "forward_seen"):
        if candidate not in arms:
            continue
        for metric in metrics:
            pair = []
            for seed in sorted({row["seed"] for row in fold_rows}):
                for fold in sorted({row["fold"] for row in fold_rows}):
                    base = next(
                        row
                        for row in fold_rows
                        if row["arm"] == "real"
                        and row["seed"] == seed
                        and row["fold"] == fold
                    )
                    cand = next(
                        row
                        for row in fold_rows
                        if row["arm"] == candidate
                        and row["seed"] == seed
                        and row["fold"] == fold
                    )
                    pair.append(cand[metric] - base[metric])
            values = np.asarray(pair, dtype=float)
            if np.allclose(values, 0.0):
                p_value = 1.0
            else:
                p_value = float(wilcoxon(values, alternative="two-sided").pvalue)
            low, high = _bootstrap_ci(values)
            rows.append(
                {
                    "candidate": candidate,
                    "reference": "real",
                    "metric": metric,
                    "n": len(values),
                    "wins": int(np.sum(values < 0.0)),
                    "median_delta": round(float(np.median(values)), 6),
                    "mean_delta": round(float(values.mean()), 6),
                    "ci_low": round(low, 6),
                    "ci_high": round(high, 6),
                    "wilcoxon_p": round(p_value, 8),
                }
            )
    return rows


def _summary_markdown(summary_rows: list[dict]) -> str:
    lines = [
        "# 完整旧协议四臂对照",
        "",
        "| 臂 | 3 seed 宏 3σ | 3 seed 宏 MAPE | 6 checkpoint 宏 3σ | 6 checkpoint 宏 MAPE |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {row['macro_three_sigma_mean']:.3f} ± "
            f"{row['macro_three_sigma_sd']:.3f} | "
            f"{row['macro_mape_mean']:.3f} | "
            f"{row['ensemble_macro_three_sigma']:.3f} | "
            f"{row['ensemble_macro_mape']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(
    folds: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    seconds: int = 10,
    epochs: int = EPOCHS,
    configs: tuple[str, ...] = DEFAULT_CONFIGS,
    arms: tuple[str, ...] = ARMS,
    output: Path | str = BASE,
    real_scaler: bool = False,
) -> None:
    if folds < 1 or folds > 5:
        raise ValueError("folds 必须在 1–5 之间")
    if seconds < 2 or seconds > 10:
        raise ValueError("seconds 必须在 2–10 之间")
    if not set(arms) <= set(ARMS):
        raise ValueError(f"arms 必须是 {ARMS} 的子集")
    if not configs:
        raise ValueError("至少需要一个低浓度配置")

    out = Path(output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(ROOT / "artifacts/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    if len(development) != 50:
        raise ValueError(f"开发集应为 50 条，实际 {len(development)}")
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    y_dev = np.stack([row["_labels"] for row in development]).astype(np.float32)
    synthetic_count = SYNTHETIC_FACTOR * 40
    need_forward = bool({"forward", "forward+vae", "forward_pretrain", "forward_unseen", "forward_h2", "forward_seen"} & set(arms))
    need_vae = bool({"vae", "forward+vae"} & set(arms))

    fold_rows: list[dict] = []
    fold_predictions: dict[tuple[str, int, int, int], np.ndarray] = {}
    fold_truth: dict[int, np.ndarray] = {}
    truth_parts: list[np.ndarray] = []
    file_ids: list[int] = []
    started = time.time()

    for fold in range(folds):
        train_rows = [row for row in development if int(row["fold"]) != fold]
        val_rows = [row for row in development if int(row["fold"]) == fold]
        train_experiments = [
            experiments[int(Path(row["file"]).stem)] for row in train_rows
        ]
        x_real, y_real = _real_features(train_rows, seconds)
        x_val, y_val = _real_features(val_rows, seconds)
        scaler_stats = (
            fit_scalers(x_real, _transformed_labels(y_real)) if real_scaler else None
        )
        fold_truth[fold] = y_val
        truth_parts.append(y_val)
        file_ids.extend(int(Path(row["file"]).stem) for row in val_rows)

        for seed in seeds:
            aug_seed = seed + 1000 * fold
            forward, vae = _fit_generators(
                train_rows,
                train_experiments,
                aug_seed,
                device,
                need_forward,
                need_vae,
            )
            labels = np.stack([row["_labels"] for row in train_rows]).astype(
                np.float64
            )
            rng = np.random.default_rng(aug_seed)
            val_labels = np.stack([row["_labels"] for row in val_rows]).astype(np.float64)
            base_labels = _synthetic_concentration_excluding(
                labels, val_labels, synthetic_count, rng
            )
            base_initial = _initial_for_synthetic(
                train_rows, synthetic_count, rng
            )
            forward_base = (
                _forward_features(forward, base_labels, base_initial, seconds)
                if forward is not None
                else (np.empty((0, 0)), np.empty((0, 3)))
            )
            if {"forward_unseen", "forward_h2"} & set(arms) and forward is not None:
                unseen_labels = _unseen_concentrations(
                    labels, val_labels, synthetic_count, rng
                )
                unseen_initial = _random_initial(train_rows, synthetic_count, rng)
                forward_unseen_base = _forward_features(
                    forward, unseen_labels, unseen_initial, seconds
                )
            else:
                forward_unseen_base = (np.empty((0, 0)), np.empty((0, 3)))
            if "forward_seen" in arms and forward is not None:
                seen_labels = _training_concentrations(
                    labels, synthetic_count, rng
                )
                seen_initial = _initial_for_labels(train_rows, seen_labels)
                forward_seen_base = _forward_features(
                    forward, seen_labels, seen_initial, seconds
                )
            else:
                forward_seen_base = (np.empty((0, 0)), np.empty((0, 3)))
            vae_base = (
                _vae_features(vae, synthetic_count, aug_seed + 1, seconds)
                if vae is not None
                else (np.empty((0, 0)), np.empty((0, 3)))
            )

            edge_labels = _edge_case_labels(train_rows)
            if edge_labels.size:
                repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
                edge_initial = _initial_for_labels(train_rows, repeated)
                edge_forward = (
                    _forward_features(forward, repeated, edge_initial, seconds)
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_vae = (
                    _vae_features(
                        vae,
                        len(repeated),
                        aug_seed + 2,
                        seconds,
                        condition_labels=repeated,
                    )
                    if vae is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
            else:
                edge_forward = (np.empty((0, 0)), np.empty((0, 3)))
                edge_vae = (np.empty((0, 0)), np.empty((0, 3)))

            x_noise, y_noise = noisy_features(
                train_rows,
                seconds,
                MODE,
                NoiseConfig(scale=NOISE_SCALE, copies=1),
                aug_seed + 17,
            )
            arm_arrays: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
            arm_arrays["real"] = (x_real, y_real, 0, 0)
            sample_weights: dict[str, np.ndarray | None] = {"real": None}
            pretrain_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for arm in arms:
                if arm == "real":
                    continue
                x_syn, y_syn, n_syn, n_edge = _synthetic_for_arm(
                    arm,
                    forward_base,
                    vae_base,
                    edge_labels,
                    edge_forward,
                    edge_vae,
                    forward_unseen_base,
                    forward_seen_base,
                    synthetic_count,
                )
                if arm == "forward_pretrain":
                    arm_arrays[arm] = (
                        np.concatenate((x_real, x_noise, edge_forward[0]), axis=0),
                        np.concatenate((y_real, y_noise, edge_forward[1]), axis=0),
                        n_syn,
                        n_edge,
                    )
                    pretrain_arrays[arm] = (x_syn, y_syn)
                else:
                    arm_arrays[arm] = (
                        np.concatenate((x_real, x_noise, x_syn), axis=0),
                        np.concatenate((y_real, y_noise, y_syn), axis=0),
                        n_syn,
                        n_edge,
                    )
                    sample_weights[arm] = None

            if "forward_h2" in arms:
                n_clean = len(y_real) + len(y_noise)
                weights = np.ones((len(y_real) + len(y_noise) + len(edge_forward[1]), 3), dtype=np.float32)
                if len(edge_forward[1]):
                    weights[-len(edge_forward[1]):, 1:] = 0.0
                # forward_unseen 合成部分：只给 H2 头监督
                n_h2 = arm_arrays["forward_h2"][2]
                start = n_clean
                weights[start:start + n_h2, 1:] = 0.0
                weights[start + n_h2:, 1:] = 0.0
                sample_weights["forward_h2"] = weights

            for arm in arms:
                x_train, y_train, n_syn, n_edge = arm_arrays[arm]
                for config_index, power in enumerate(configs):
                    if arm == "forward_pretrain":
                        prediction, epochs_used = _train_pretrain(
                            *pretrain_arrays[arm],
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            y_real,
                            seed,
                            device,
                            epochs,
                            power,
                            scaler_stats=scaler_stats,
                        )
                    else:
                        prediction, epochs_used = _train(
                            x_train,
                            y_train,
                            x_val,
                            y_val,
                            y_real,
                            seed,
                            device,
                            epochs,
                            power,
                            scaler_stats=scaler_stats,
                        )
                    pred_clip = _clip(prediction, y_dev)
                    metrics = _metric_values(pred_clip, y_val)
                    row = {
                        "window": seconds,
                        "fold": fold,
                        "seed": seed,
                        "config": power,
                        "arm": arm,
                        "n_train": len(x_train),
                        "n_noise": len(y_noise) if arm != "real" else 0,
                        "n_synthetic": n_syn,
                        "n_edge": n_edge,
                        "epochs": epochs_used,
                        **{
                            key: round(value, 6)
                            for key, value in metrics.items()
                        },
                    }
                    fold_rows.append(row)
                    fold_predictions[(arm, seed, fold, config_index)] = prediction
                    print(
                        f"fold={fold} seed={seed} arm={arm:12s} cfg={power:9s} "
                        f"n={len(x_train):3d} ep={epochs_used:3d} "
                        f"3σ={metrics['macro_three_sigma']:.3f} "
                        f"MAPE={metrics['macro_mape']:.3f}",
                        flush=True,
                    )

    # 逐 seed/折先平均两套低浓度配置，用于配对统计。
    fold_seed_rows: list[dict] = []
    for arm in arms:
        for seed in seeds:
            for fold in range(folds):
                preds = [
                    fold_predictions[(arm, seed, fold, index)]
                    for index in range(len(configs))
                ]
                metrics = _metric_values(
                    _clip(np.mean(preds, axis=0), y_dev), fold_truth[fold]
                )
                fold_seed_rows.append(
                    {
                        "window": seconds,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "checkpoints": len(configs),
                        **{
                            key: round(value, 6)
                            for key, value in metrics.items()
                        },
                    }
                )

    # 同一 seed 内先平均两套低浓度配置，再作为该 seed 的 OOF。
    oof_seed_rows: list[dict] = []
    seed_predictions: dict[tuple[str, int], np.ndarray] = {}
    for arm in arms:
        for seed in seeds:
            parts = []
            for fold in range(folds):
                preds = [
                    fold_predictions[(arm, seed, fold, index)]
                    for index in range(len(configs))
                ]
                parts.append(np.mean(preds, axis=0))
            prediction = _clip(np.concatenate(parts, axis=0), y_dev)
            truth = np.concatenate(truth_parts, axis=0)
            metrics = _metric_values(prediction, truth)
            oof_seed_rows.append(
                {
                    "window": seconds,
                    "seed": seed,
                    "arm": arm,
                    "configs": ";".join(configs),
                    "n": len(truth),
                    **{
                        key: round(value, 6)
                        for key, value in metrics.items()
                    },
                }
            )
            seed_predictions[(arm, seed)] = prediction

    # 6 checkpoint（2 配置 × 3 seed）逐折平均，再做 OOF 与裁剪。
    oof_rows: list[dict] = []
    ensemble_predictions: dict[str, np.ndarray] = {}
    for arm in arms:
        parts = []
        for fold in range(folds):
            preds = [
                fold_predictions[(arm, seed, fold, index)]
                for seed in seeds
                for index in range(len(configs))
            ]
            parts.append(np.mean(preds, axis=0))
        prediction = _clip(np.concatenate(parts, axis=0), y_dev)
        truth = np.concatenate(truth_parts, axis=0)
        metrics = _metric_values(prediction, truth)
        oof_rows.append(
            {
                "window": seconds,
                "arm": arm,
                "checkpoints": len(seeds) * len(configs),
                "n": len(truth),
                **{key: round(value, 6) for key, value in metrics.items()},
            }
        )
        ensemble_predictions[arm] = prediction

    summary_rows: list[dict] = []
    for arm in arms:
        selected = [row for row in oof_seed_rows if row["arm"] == arm]
        summary: dict[str, float | int | str] = {
            "window": seconds,
            "arm": arm,
            "seeds": len(selected),
            "configs": len(configs),
        }
        for name in (
            "macro_three_sigma",
            "macro_mape",
            "macro_median",
            "macro_p90",
        ):
            values = np.array([row[name] for row in selected], dtype=float)
            summary[f"{name}_mean"] = round(float(values.mean()), 6)
            summary[f"{name}_sd"] = (
                round(float(values.std(ddof=1)), 6) if len(values) > 1 else 0.0
            )
        ensemble = next(row for row in oof_rows if row["arm"] == arm)
        summary["ensemble_macro_three_sigma"] = ensemble["macro_three_sigma"]
        summary["ensemble_macro_mape"] = ensemble["macro_mape"]
        summary["ensemble_macro_median"] = ensemble["macro_median"]
        summary["ensemble_macro_p90"] = ensemble["macro_p90"]
        summary_rows.append(summary)

    paired_rows = _paired_rows(fold_seed_rows, PAIR_METRICS, arms)

    _write_csv(out / "folds.csv", fold_rows)
    _write_csv(out / "fold_seed.csv", fold_seed_rows)
    _write_csv(out / "oof_seed.csv", oof_seed_rows)
    _write_csv(out / "oof.csv", oof_rows)
    _write_csv(out / "summary.csv", summary_rows)
    if paired_rows:
        _write_csv(out / "paired.csv", paired_rows)

    truth = np.concatenate(truth_parts, axis=0).astype(np.float32)
    payload = {
        "truth": truth,
        "file_id": np.asarray(file_ids, dtype=np.int32),
    }
    for (arm, seed), prediction in seed_predictions.items():
        payload[f"{arm}_s{seed}"] = prediction.astype(np.float32)
    for arm, prediction in ensemble_predictions.items():
        payload[f"{arm}_ensemble"] = prediction.astype(np.float32)
    np.savez_compressed(out / "predictions.npz", **payload)

    (out / "summary.md").write_text(
        _summary_markdown(summary_rows), encoding="utf-8"
    )
    noise_count = next(
        (row["n_noise"] for row in fold_rows if row["arm"] != "real"), 0
    )
    config = {
        "folds": folds,
        "seeds": list(seeds),
        "window": seconds,
        "epochs": epochs,
        "arms": list(arms),
        "low_conc_configs": list(configs),
        "synthetic_count_per_arm": synthetic_count,
        "synthetic_ratio": SYNTHETIC_FACTOR,
        "noise_scale": NOISE_SCALE,
        "noise_count_per_augmented_arm": noise_count,
        "edge_copies": EDGE_COPIES,
        "pretrain_epochs": PRETRAIN_EPOCHS,
        "real_scaler": real_scaler,
        "clip_to_development_range": True,
        "elapsed_seconds": round(time.time() - started, 1),
        "mode": MODE,
        "device": str(device),
    }
    (out / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out} in {config['elapsed_seconds']} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--configs", nargs="+", default=list(DEFAULT_CONFIGS)
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS))
    parser.add_argument("--output", default=str(BASE))
    parser.add_argument(
        "--real-scaler",
        action="store_true",
        help="用真实训练折拟合 scaler，合成样本只参与参数更新",
    )
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(args.seeds),
        seconds=args.seconds,
        epochs=args.epochs,
        configs=tuple(args.configs),
        arms=tuple(args.arms),
        output=args.output,
        real_scaler=args.real_scaler,
    )


if __name__ == "__main__":
    main()
