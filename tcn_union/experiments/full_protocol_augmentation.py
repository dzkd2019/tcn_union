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

输出目录 artifacts/augmentation/protocols/full/：

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
import os
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
BASE = ROOT / "artifacts/augmentation/protocols/full"
ARMS = (
    "real",
    "vae",
    "forward",
    "forward+vae",
    "forward_cond",
    "forward_cond+vae",
    "forward_cond_weighted",
    "forward_cond_h2",
    "forward_resid",
    "forward_resid+vae",
    "forward_nn",
    "forward_nn+vae",
    "forward_pretrain",
    "forward_pretrain_no_h2",
    "forward_pretrain_h2",
    "forward_cond_pretrain",
    "forward_unseen",
    "forward_h2",
    "forward_seen",
    "forward_prob",
    "forward_prob_joint",
    "forward_prob_joint10",
)
DEFAULT_SEEDS = (20260923, 20260924, 20260925)
DEFAULT_CONFIGS = ("1.0", "0,0.3,1.0")
EPOCHS = 500
PRETRAIN_EPOCHS = 200
SYNTHETIC_FACTOR = 3
JOINT_SAMPLE = os.environ.get("FORWARD_JOINT_SAMPLE", "normal")
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


def _forward_residual_features(
    model: PerGasMLPDoubleExponentialModel,
    rows: list[dict],
    labels: np.ndarray,
    initial: np.ndarray,
    seconds: int,
    rng: np.random.Generator,
    alpha: float = 1.0,
    nearest: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """正模型均值曲线 + 训练折经验残差 bootstrap。

    残差来自训练折真实曲线与该折正模型预测之差，只加在合成曲线上，
    用来补足正模型缺失的实验间形状差异。nearest=True 时按对数浓度
    距离选择最近的真实实验作为残差供体，避免跨浓度移植形状。"""
    times = _times(seconds)
    relative = model.predict_relative(labels, times)
    if nearest:
        train_labels = np.stack([row["_labels"] for row in rows]).astype(np.float64)
        scale = train_labels.std(axis=0)
        scale[scale == 0] = 1.0
        target = np.log(labels) / scale
        source = np.log(train_labels) / scale
        distance = ((target[:, None, :] - source[None, :, :]) ** 2).sum(axis=2)
        donor_index = distance.argmin(axis=1)
    else:
        donor_index = rng.integers(0, len(rows), size=len(labels))
    donor_labels = np.stack([rows[i]["_labels"] for i in donor_index]).astype(
        np.float64
    )
    donor_response = np.stack([rows[i]["_response"] for i in donor_index])
    donor_response = donor_response[:, : len(times), :]
    donor_initial = donor_response[:, 0, :]
    donor_relative = donor_response / donor_initial[:, None, :] - 1.0
    donor_model = model.predict_relative(donor_labels, times)
    relative = relative + alpha * (donor_relative - donor_model)
    response = initial[:, None, :] * (1.0 + relative)
    features = np.stack(
        [make_features(times, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _prob_design(values: np.ndarray) -> np.ndarray:
    u = np.log(values)
    return np.column_stack(
        (
            np.ones(len(u)),
            u,
            u**2,
            u[:, [0]] * u[:, [1]],
            u[:, [0]] * u[:, [2]],
            u[:, [1]] * u[:, [2]],
        )
    )


def _forward_prob_features(
    model: PerGasMLPDoubleExponentialModel,
    rows: list[dict],
    labels: np.ndarray,
    initial: np.ndarray,
    seconds: int,
    rng: np.random.Generator,
    components: int = 3,
    ridge: float = 1e-2,
    shrink: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """正模型均值 + 训练折残差的低秩条件分布采样。

    与直接 bootstrap 某个真实实验的残差不同，这里先把训练折残差按气体
    做 PCA，再让每个主成分得分对 log 浓度做二次岭回归；合成样本的条件
    均值取回归预测，波动取回归残差协方差。这样既能补实验间形状差异，
    又不会把某个高浓度实验的残差整条移植到低浓度组合上。
    """
    times = _times(seconds)
    train_labels = np.stack([row["_labels"] for row in rows]).astype(np.float64)
    train_response = np.stack(
        [np.asarray(row["_response"])[: len(times)] for row in rows]
    )
    train_initial = train_response[:, 0, :]
    train_relative = train_response / train_initial[:, None, :] - 1.0
    train_prediction = model.predict_relative(train_labels, times)
    residual = train_relative - train_prediction

    design = _prob_design(train_labels)
    design_mean = design[:, 1:].mean(axis=0)
    design_scale = design[:, 1:].std(axis=0)
    design_scale[design_scale == 0.0] = 1.0
    design[:, 1:] = (design[:, 1:] - design_mean) / design_scale
    target_design = _prob_design(labels)
    target_design[:, 1:] = (target_design[:, 1:] - design_mean) / design_scale

    sampled = np.empty((len(labels), len(times), 3), dtype=np.float64)
    for gas in range(3):
        values = residual[:, :, gas]
        mean = values.mean(axis=0)
        centered = values - mean[None, :]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        count = min(components, len(rows) - 1, len(times))
        basis = vt[:count]
        scores = centered @ basis.T

        penalty = ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.solve(
            design.T @ design + penalty, design.T @ scores
        )
        fitted = design @ coefficients
        error = scores - fitted
        covariance = np.atleast_2d(np.cov(error, rowvar=False))
        covariance = 0.5 * (covariance + covariance.T)
        covariance += shrink * np.trace(covariance) / count * np.eye(count)
        covariance += 1e-10 * np.eye(count)

        predicted = target_design @ coefficients
        noise = rng.multivariate_normal(
            np.zeros(count), covariance, size=len(labels)
        )
        reconstruction = mean[None, :] + (predicted + noise) @ basis
        low = values.min(axis=0)
        high = values.max(axis=0)
        sampled[:, :, gas] = np.clip(reconstruction, low, high)

    relative = model.predict_relative(labels, times) + sampled
    response = initial[:, None, :] * (1.0 + relative)
    features = np.stack(
        [make_features(times, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _forward_prob_joint_features(
    model: PerGasMLPDoubleExponentialModel,
    rows: list[dict],
    labels: np.ndarray,
    initial: np.ndarray,
    seconds: int,
    rng: np.random.Generator,
    components: int = 5,
    ridge: float = 1e-2,
    shrink: float = 0.2,
    sample: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    """正模型均值 + 三气体联合残差的低秩条件分布采样。

    逐气体独立采样会破坏实验内三通道残差的相关性；这里把 3×T 的残差
    展平后做一次 PCA，使合成曲线保留跨气体共享的实验状态。
    """
    times = _times(seconds)
    train_labels = np.stack([row["_labels"] for row in rows]).astype(np.float64)
    train_response = np.stack(
        [np.asarray(row["_response"])[: len(times)] for row in rows]
    )
    train_initial = train_response[:, 0, :]
    train_relative = train_response / train_initial[:, None, :] - 1.0
    train_prediction = model.predict_relative(train_labels, times)
    residual = (train_relative - train_prediction).reshape(len(rows), -1)

    design = _prob_design(train_labels)
    design_mean = design[:, 1:].mean(axis=0)
    design_scale = design[:, 1:].std(axis=0)
    design_scale[design_scale == 0.0] = 1.0
    design[:, 1:] = (design[:, 1:] - design_mean) / design_scale
    target_design = _prob_design(labels)
    target_design[:, 1:] = (target_design[:, 1:] - design_mean) / design_scale

    mean = residual.mean(axis=0)
    centered = residual - mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(components, len(rows) - 1, residual.shape[1])
    basis = vt[:count]
    scores = centered @ basis.T

    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ scores)
    error = scores - design @ coefficients
    covariance = np.atleast_2d(np.cov(error, rowvar=False))
    covariance = 0.5 * (covariance + covariance.T)
    covariance += shrink * np.trace(covariance) / count * np.eye(count)
    covariance += 1e-10 * np.eye(count)

    predicted = target_design @ coefficients
    noise = rng.multivariate_normal(np.zeros(count), covariance, size=len(labels))
    if sample == "bootstrap":
        distance = ((target_design[:, None, :] - design[None, :, :]) ** 2).sum(axis=2)
        nearest = np.argsort(distance, axis=1)[:, :5]
        pick = nearest[np.arange(len(labels)), rng.integers(0, 5, size=len(labels))]
        noise = error[pick]
    reconstruction = mean[None, :] + (predicted + noise) @ basis
    reconstruction = reconstruction.reshape(len(labels), len(times), 3)
    low = residual.reshape(len(rows), len(times), 3).min(axis=0)
    high = residual.reshape(len(rows), len(times), 3).max(axis=0)
    sampled = np.clip(reconstruction, low[None, :, :], high[None, :, :])

    relative = model.predict_relative(labels, times) + sampled
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



def _initial_for_conditional(
    rows: list[dict], labels: np.ndarray, ridge: float = 1e-3
) -> np.ndarray:
    """按浓度条件预测初始绝对响应 R0；训练标签命中时用真实值。

    随机抽 R0 会让 NH3 的原始通道与浓度标签不一致（NH3 的 R0 与浓度
    Spearman 相关 0.98）。这里对 log(R0) 拟合二次多项式，再对未见过
    的浓度组合做条件预测，作为正模型的边界条件。
    """
    train_labels = np.stack(
        [np.asarray(row["_labels"], dtype=np.float64) for row in rows]
    )
    initial = np.stack([row["_response"][0] for row in rows])
    lookup = {
        tuple(np.asarray(row["_labels"], dtype=np.float64)): row["_response"][0]
        for row in rows
    }

    def design(values: np.ndarray) -> np.ndarray:
        u = np.log(values)
        return np.column_stack(
            (
                np.ones(len(u)),
                u,
                u**2,
                u[:, [0]] * u[:, [1]],
                u[:, [0]] * u[:, [2]],
                u[:, [1]] * u[:, [2]],
            )
        )

    x = design(train_labels)
    x_mean = x[:, 1:].mean(axis=0)
    x_scale = x[:, 1:].std(axis=0)
    x[:, 1:] = (x[:, 1:] - x_mean) / x_scale
    penalty = ridge * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ np.log(initial))

    predicted = np.exp(design(labels) @ coefficients)
    limits = np.stack(
        (initial.min(axis=0) * 0.8, initial.max(axis=0) * 1.2)
    )
    predicted = np.clip(predicted, limits[0], limits[1])
    result = predicted.copy()
    for index, label in enumerate(labels):
        key = tuple(np.asarray(label, dtype=np.float64))
        if key in lookup:
            result[index] = lookup[key]
    return result


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
    forward_cond_base: tuple[np.ndarray, np.ndarray],
    vae_base: tuple[np.ndarray, np.ndarray],
    edge_labels: np.ndarray,
    edge_forward: tuple[np.ndarray, np.ndarray],
    edge_forward_cond: tuple[np.ndarray, np.ndarray],
    edge_forward_resid: tuple[np.ndarray, np.ndarray],
    edge_forward_nn: tuple[np.ndarray, np.ndarray],
    edge_forward_prob: tuple[np.ndarray, np.ndarray],
    edge_forward_prob_joint: tuple[np.ndarray, np.ndarray],
    edge_forward_prob_joint10: tuple[np.ndarray, np.ndarray],
    edge_vae: tuple[np.ndarray, np.ndarray],
    forward_unseen_base: tuple[np.ndarray, np.ndarray],
    forward_seen_base: tuple[np.ndarray, np.ndarray],
    forward_prob_base: tuple[np.ndarray, np.ndarray],
    forward_prob_joint_base: tuple[np.ndarray, np.ndarray],
    forward_prob_joint10_base: tuple[np.ndarray, np.ndarray],
    forward_resid_base: tuple[np.ndarray, np.ndarray],
    forward_nn_base: tuple[np.ndarray, np.ndarray],
    synthetic_count: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """按臂拼基础合成与边缘复制，返回 (x, y, synthetic, edge)。"""
    if arm == "vae":
        x = np.concatenate((vae_base[0], edge_vae[0]), axis=0)
        y = np.concatenate((vae_base[1], edge_vae[1]), axis=0)
        return x, y, len(vae_base[1]), len(edge_vae[1])
    if arm in {"forward", "forward_pretrain", "forward_pretrain_no_h2", "forward_pretrain_h2"}:
        x = np.concatenate((forward_base[0], edge_forward[0]), axis=0)
        y = np.concatenate((forward_base[1], edge_forward[1]), axis=0)
        return x, y, len(forward_base[1]), len(edge_forward[1])
    if arm in {"forward_cond", "forward_cond_pretrain"}:
        x = np.concatenate(
            (forward_cond_base[0], edge_forward_cond[0]), axis=0
        )
        y = np.concatenate(
            (forward_cond_base[1], edge_forward_cond[1]), axis=0
        )
        return x, y, len(forward_cond_base[1]), len(edge_forward_cond[1])
    if arm == "forward_resid":
        x = np.concatenate((forward_resid_base[0], edge_forward_resid[0]), axis=0)
        y = np.concatenate((forward_resid_base[1], edge_forward_resid[1]), axis=0)
        return x, y, len(forward_resid_base[1]), len(edge_forward_resid[1])
    if arm == "forward_nn":
        x = np.concatenate((forward_nn_base[0], edge_forward_nn[0]), axis=0)
        y = np.concatenate((forward_nn_base[1], edge_forward_nn[1]), axis=0)
        return x, y, len(forward_nn_base[1]), len(edge_forward_nn[1])
    if arm == "forward_nn+vae":
        half = synthetic_count // 2
        n_edge_half = len(edge_labels) * (EDGE_COPIES // 2)
        x = np.concatenate(
            (
                forward_nn_base[0][:half],
                vae_base[0][:half],
                edge_forward_nn[0][:n_edge_half],
                edge_vae[0][:n_edge_half],
            ),
            axis=0,
        )
        y = np.concatenate(
            (
                forward_nn_base[1][:half],
                vae_base[1][:half],
                edge_forward_nn[1][:n_edge_half],
                edge_vae[1][:n_edge_half],
            ),
            axis=0,
        )
        return x, y, half * 2, n_edge_half * 2
    if arm == "forward_resid+vae":
        half = synthetic_count // 2
        n_edge_half = len(edge_labels) * (EDGE_COPIES // 2)
        x = np.concatenate(
            (
                forward_resid_base[0][:half],
                vae_base[0][:half],
                edge_forward_resid[0][:n_edge_half],
                edge_vae[0][:n_edge_half],
            ),
            axis=0,
        )
        y = np.concatenate(
            (
                forward_resid_base[1][:half],
                vae_base[1][:half],
                edge_forward_resid[1][:n_edge_half],
                edge_vae[1][:n_edge_half],
            ),
            axis=0,
        )
        return x, y, half * 2, n_edge_half * 2
    if arm in {"forward_cond+vae", "forward_cond_weighted", "forward_cond_h2"}:
        half = synthetic_count // 2
        n_edge_half = len(edge_labels) * (EDGE_COPIES // 2)
        x = np.concatenate(
            (
                forward_cond_base[0][:half],
                vae_base[0][:half],
                edge_forward_cond[0][:n_edge_half],
                edge_vae[0][:n_edge_half],
            ),
            axis=0,
        )
        y = np.concatenate(
            (
                forward_cond_base[1][:half],
                vae_base[1][:half],
                edge_forward_cond[1][:n_edge_half],
                edge_vae[1][:n_edge_half],
            ),
            axis=0,
        )
        return x, y, half * 2, n_edge_half * 2
    if arm in {"forward_unseen", "forward_h2"}:
        x = np.concatenate((forward_unseen_base[0], edge_forward[0]), axis=0)
        y = np.concatenate((forward_unseen_base[1], edge_forward[1]), axis=0)
        return x, y, len(forward_unseen_base[1]), len(edge_forward[1])
    if arm == "forward_seen":
        x = np.concatenate((forward_seen_base[0], edge_forward[0]), axis=0)
        y = np.concatenate((forward_seen_base[1], edge_forward[1]), axis=0)
        return x, y, len(forward_seen_base[1]), len(edge_forward[1])
    if arm == "forward_prob":
        x = np.concatenate((forward_prob_base[0], edge_forward_prob[0]), axis=0)
        y = np.concatenate((forward_prob_base[1], edge_forward_prob[1]), axis=0)
        return x, y, len(forward_prob_base[1]), len(edge_forward_prob[1])
    if arm == "forward_prob_joint":
        x = np.concatenate(
            (forward_prob_joint_base[0], edge_forward_prob_joint[0]), axis=0
        )
        y = np.concatenate(
            (forward_prob_joint_base[1], edge_forward_prob_joint[1]), axis=0
        )
        return x, y, len(forward_prob_joint_base[1]), len(edge_forward_prob_joint[1])
    if arm == "forward_prob_joint10":
        x = np.concatenate(
            (forward_prob_joint10_base[0], edge_forward_prob_joint10[0]), axis=0
        )
        y = np.concatenate(
            (forward_prob_joint10_base[1], edge_forward_prob_joint10[1]), axis=0
        )
        return x, y, len(forward_prob_joint10_base[1]), len(edge_forward_prob_joint10[1])
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
        sample_weight=sample_weight,
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
    pretrain_weight: np.ndarray | None = None,
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
        sample_weight=pretrain_weight,
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
    for candidate in (
        "vae",
        "forward",
        "forward+vae",
        "forward_cond",
        "forward_cond+vae",
        "forward_cond_weighted",
        "forward_cond_h2",
        "forward_resid",
        "forward_resid+vae",
        "forward_nn",
        "forward_nn+vae",
        "forward_pretrain",
        "forward_pretrain_no_h2",
        "forward_pretrain_h2",
        "forward_cond_pretrain",
        "forward_unseen",
        "forward_h2",
        "forward_seen",
        "forward_prob",
        "forward_prob_joint",
        "forward_prob_joint10",
    ):
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
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    if len(development) != 50:
        raise ValueError(f"开发集应为 50 条，实际 {len(development)}")
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    y_dev = np.stack([row["_labels"] for row in development]).astype(np.float32)
    synthetic_count = SYNTHETIC_FACTOR * 40
    need_forward = bool({"forward", "forward+vae", "forward_cond", "forward_cond+vae", "forward_cond_weighted", "forward_cond_h2", "forward_resid", "forward_resid+vae", "forward_nn", "forward_nn+vae", "forward_pretrain", "forward_pretrain_no_h2", "forward_pretrain_h2", "forward_cond_pretrain", "forward_unseen", "forward_h2", "forward_seen", "forward_prob", "forward_prob_joint", "forward_prob_joint10"} & set(arms))
    need_vae = bool({"vae", "forward+vae", "forward_cond+vae", "forward_cond_weighted", "forward_cond_h2", "forward_resid+vae", "forward_nn+vae"} & set(arms))

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
            if {"forward_cond", "forward_cond+vae", "forward_cond_weighted", "forward_cond_h2", "forward_cond_pretrain"} & set(arms) and forward is not None:
                forward_cond_base = _forward_features(
                    forward,
                    base_labels,
                    _initial_for_conditional(train_rows, base_labels),
                    seconds,
                )
            else:
                forward_cond_base = (np.empty((0, 0)), np.empty((0, 3)))
            if {"forward_nn", "forward_nn+vae"} & set(arms) and forward is not None:
                forward_nn_base = _forward_residual_features(
                    forward,
                    train_rows,
                    base_labels,
                    _initial_for_labels(train_rows, base_labels),
                    seconds,
                    rng,
                    nearest=True,
                )
            else:
                forward_nn_base = (np.empty((0, 0)), np.empty((0, 3)))
            if {"forward_resid", "forward_resid+vae"} & set(arms) and forward is not None:
                forward_resid_base = _forward_residual_features(
                    forward,
                    train_rows,
                    base_labels,
                    _initial_for_labels(train_rows, base_labels),
                    seconds,
                    rng,
                )
            else:
                forward_resid_base = (np.empty((0, 0)), np.empty((0, 3)))
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
            if "forward_prob" in arms and forward is not None:
                prob_labels = _training_concentrations(
                    labels, synthetic_count, rng
                )
                prob_initial = _initial_for_labels(train_rows, prob_labels)
                forward_prob_base = _forward_prob_features(
                    forward, train_rows, prob_labels, prob_initial, seconds, rng
                )
            else:
                forward_prob_base = (np.empty((0, 0)), np.empty((0, 3)))
            if {"forward_prob_joint", "forward_prob_joint10"} & set(arms) and forward is not None:
                joint_labels = _training_concentrations(
                    labels, synthetic_count, rng
                )
                joint_initial = _initial_for_labels(train_rows, joint_labels)
                forward_prob_joint_base = _forward_prob_joint_features(
                    forward, train_rows, joint_labels, joint_initial, seconds, rng
                )
                forward_prob_joint10_base = _forward_prob_joint_features(
                    forward,
                    train_rows,
                    joint_labels,
                    joint_initial,
                    seconds,
                    rng,
                    components=10,
                    sample=JOINT_SAMPLE,
                )
            else:
                forward_prob_joint_base = (np.empty((0, 0)), np.empty((0, 3)))
                forward_prob_joint10_base = (np.empty((0, 0)), np.empty((0, 3)))
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
                edge_forward_cond = (
                    _forward_features(
                        forward,
                        repeated,
                        _initial_for_conditional(train_rows, repeated),
                        seconds,
                    )
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_forward_resid = (
                    _forward_residual_features(
                        forward,
                        train_rows,
                        repeated,
                        _initial_for_labels(train_rows, repeated),
                        seconds,
                        rng,
                    )
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_forward_nn = (
                    _forward_residual_features(
                        forward,
                        train_rows,
                        repeated,
                        _initial_for_labels(train_rows, repeated),
                        seconds,
                        rng,
                        nearest=True,
                    )
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_forward_prob = (
                    _forward_prob_features(
                        forward,
                        train_rows,
                        repeated,
                        _initial_for_labels(train_rows, repeated),
                        seconds,
                        rng,
                    )
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_forward_prob_joint = (
                    _forward_prob_joint_features(
                        forward,
                        train_rows,
                        repeated,
                        _initial_for_labels(train_rows, repeated),
                        seconds,
                        rng,
                    )
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_forward_prob_joint10 = (
                    _forward_prob_joint_features(
                        forward,
                        train_rows,
                        repeated,
                        _initial_for_labels(train_rows, repeated),
                        seconds,
                        rng,
                        components=10,
                    )
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
                edge_forward_cond = (np.empty((0, 0)), np.empty((0, 3)))
                edge_forward_resid = (np.empty((0, 0)), np.empty((0, 3)))
                edge_forward_nn = (np.empty((0, 0)), np.empty((0, 3)))
                edge_forward_prob = (np.empty((0, 0)), np.empty((0, 3)))
                edge_forward_prob_joint = (np.empty((0, 0)), np.empty((0, 3)))
                edge_forward_prob_joint10 = (np.empty((0, 0)), np.empty((0, 3)))
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
            pretrain_weights: dict[str, np.ndarray | None] = {
                "forward_pretrain_no_h2": None,
                "forward_pretrain_h2": None,
            }
            pretrain_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for arm in arms:
                if arm == "real":
                    continue
                x_syn, y_syn, n_syn, n_edge = _synthetic_for_arm(
                    arm,
                    forward_base,
                    forward_cond_base,
                    vae_base,
                    edge_labels,
                    edge_forward,
                    edge_forward_cond,
                    edge_forward_resid,
                    edge_forward_nn,
                    edge_forward_prob,
                    edge_forward_prob_joint,
                    edge_forward_prob_joint10,
                    edge_vae,
                    forward_unseen_base,
                    forward_seen_base,
                    forward_prob_base,
                    forward_prob_joint_base,
                    forward_prob_joint10_base,
                    forward_resid_base,
                    forward_nn_base,
                    synthetic_count,
                )
                if arm in {"forward_pretrain", "forward_pretrain_no_h2", "forward_pretrain_h2", "forward_cond_pretrain"}:
                    edge_arm = (
                        edge_forward_cond
                        if arm == "forward_cond_pretrain"
                        else edge_forward
                    )
                    arm_arrays[arm] = (
                        np.concatenate((x_real, x_noise, edge_arm[0]), axis=0),
                        np.concatenate((y_real, y_noise, edge_arm[1]), axis=0),
                        n_syn,
                        n_edge,
                    )
                    pretrain_arrays[arm] = (x_syn, y_syn)
                    if arm in {"forward_pretrain_no_h2", "forward_pretrain_h2"}:
                        weights = np.ones((len(y_syn), 3), dtype=np.float32)
                        if arm == "forward_pretrain_no_h2":
                            weights[:, 0] = 0.0
                        else:
                            weights[:, 1:] = 0.0
                        pretrain_weights[arm] = weights
                else:
                    arm_arrays[arm] = (
                        np.concatenate((x_real, x_noise, x_syn), axis=0),
                        np.concatenate((y_real, y_noise, y_syn), axis=0),
                        n_syn,
                        n_edge,
                    )
                    sample_weights[arm] = None

            if {"forward_cond_weighted", "forward_cond_h2"} & set(arms):
                # 条件正模型样本只监督 H2，NH3/NO2 交给 VAE；
                # weighted 臂对 NH3/NO2 保留 0.25 弱正则，h2 臂完全关闭。
                n_clean = len(y_real) + len(y_noise)
                half = synthetic_count // 2
                n_edge_half = len(edge_labels) * (EDGE_COPIES // 2)
                for arm in ("forward_cond_weighted", "forward_cond_h2"):
                    if arm not in arms:
                        continue
                    weights = np.ones(
                        (n_clean + 2 * half + 2 * n_edge_half, 3), dtype=np.float32
                    )
                    weak = 0.25 if arm == "forward_cond_weighted" else 0.0
                    weights[n_clean:n_clean + half, 1:] = weak
                    edge_start = n_clean + 2 * half
                    weights[edge_start:edge_start + n_edge_half, 1:] = weak
                    sample_weights[arm] = weights

            if "forward_h2" in arms:
                n_clean = len(y_real) + len(y_noise)
                n_h2 = arm_arrays["forward_h2"][2]
                n_edge_h2 = arm_arrays["forward_h2"][3]
                weights = np.ones((n_clean + n_h2 + n_edge_h2, 3), dtype=np.float32)
                weights[n_clean:, 1:] = 0.0
                sample_weights["forward_h2"] = weights

            for arm in arms:
                x_train, y_train, n_syn, n_edge = arm_arrays[arm]
                for config_index, power in enumerate(configs):
                    if arm in {"forward_pretrain", "forward_pretrain_no_h2", "forward_pretrain_h2", "forward_cond_pretrain"}:
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
                            pretrain_weight=pretrain_weights.get(arm),
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
                            sample_weight=sample_weights.get(arm),
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
