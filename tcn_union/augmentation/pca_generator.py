"""条件 PCA 生成基线：PCA 系数 + 条件均值 + 高斯残差。

只用于与 GAN / VAE / PCA-VAE 对比，不进入最终模型。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tcn_union.core.data import TIME_STEP_SECONDS, make_features


@dataclass(frozen=True)
class PCAConfig:
    seconds: float = 10.0
    components: int = 6

    def __post_init__(self) -> None:
        if self.seconds not in {5.0, 10.0}:
            raise ValueError("PCA 只支持 5 秒或 10 秒窗口")
        if self.components < 1:
            raise ValueError("PCA 主成分数必须为正")


class TrainedPCA:
    def __init__(
        self,
        means: np.ndarray,
        components: np.ndarray,
        coefficient_std: np.ndarray,
        mean_coef: np.ndarray,
        residual_std: np.ndarray,
        labels_z: np.ndarray,
        label_mean: np.ndarray,
        label_std: np.ndarray,
        output_min: np.ndarray,
        output_max: np.ndarray,
    ) -> None:
        self.means = means
        self.components = components
        self.coefficient_std = coefficient_std
        self.mean_coef = mean_coef
        self.residual_std = residual_std
        self.labels_z = labels_z
        self.label_mean = label_mean
        self.label_std = label_std
        self.output_min = output_min
        self.output_max = output_max

    def sample(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        if count < 1:
            raise ValueError("PCA 采样数量必须为正")
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(self.labels_z), size=count)
        condition = self.labels_z[indices]
        residual = rng.normal(0.0, 1.0, size=(count, 3, self.components.shape[1]))
        residual = residual * self.residual_std[None, :, :]
        design = np.concatenate((np.ones((count, 1)), condition), axis=1)
        base = (design @ self.mean_coef).reshape(count, 3, self.components.shape[1])
        coefficients = (base + residual) * self.coefficient_std[None, :, :]
        responses = np.empty((count, 3, self.components.shape[2]), dtype=np.float64)
        for gas in range(3):
            responses[:, gas, :] = self.means[gas][None, :] + coefficients[:, gas, :] @ self.components[gas]
        responses = np.clip(
            responses,
            self.output_min[None, :, None],
            self.output_max[None, :, None],
        )
        labels = condition * self.label_std[None, :] + self.label_mean[None, :]
        return responses, labels


def _prefix_arrays(rows: list[dict[str, Any]], seconds: float) -> tuple[np.ndarray, np.ndarray]:
    expected = round(seconds / TIME_STEP_SECONDS) + 1
    responses = []
    labels = []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= seconds + 1e-9
        if int(mask.sum()) != expected:
            raise ValueError("训练前缀长度与 PCA 窗口不一致")
        responses.append(response[mask].T)
        labels.append(np.asarray(row["_labels"], dtype=np.float64))
    return np.stack(responses), np.stack(labels).astype(np.float32)


def fit_pca(
    rows: list[dict[str, Any]],
    config: PCAConfig,
    seed: int,
) -> TrainedPCA:
    responses, labels = _prefix_arrays(rows, config.seconds)
    n, _, length = responses.shape
    components = min(config.components, n - 1, length)
    means, bases, coefficients = [], [], []
    for gas in range(3):
        flat = responses[:, gas, :]
        mean = flat.mean(axis=0)
        centered = flat - mean[None, :]
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:components]
        coefficient = centered @ basis.T
        means.append(mean)
        bases.append(basis)
        coefficients.append(coefficient)
    means_array = np.stack(means)
    components_array = np.stack(bases)
    coefficient_array = np.stack(coefficients, axis=1)  # [n,3,K]

    label_mean = labels.mean(axis=0).astype(np.float32)
    label_std = labels.std(axis=0).astype(np.float32)
    label_std[label_std < 1e-8] = 1.0
    labels_z = ((labels - label_mean[None, :]) / label_std[None, :]).astype(np.float32)
    coefficient_std = coefficient_array.std(axis=0).astype(np.float32)
    coefficient_std[coefficient_std < 1e-8] = 1.0
    coefficient_z = coefficient_array / coefficient_std[None, :, :]
    design = np.concatenate((np.ones((n, 1)), labels_z), axis=1)
    mean_coef = np.linalg.lstsq(design, coefficient_z.reshape(n, -1), rcond=None)[0]
    base = (design @ mean_coef).reshape(n, 3, components)
    residual = coefficient_z - base
    residual_std = residual.std(axis=0).astype(np.float32)
    residual_std[residual_std < 1e-8] = 1.0

    return TrainedPCA(
        means=means_array,
        components=components_array,
        coefficient_std=coefficient_std,
        mean_coef=mean_coef,
        residual_std=residual_std,
        labels_z=labels_z,
        label_mean=label_mean,
        label_std=label_std,
        output_min=responses.min(axis=(0, 2)).astype(np.float32),
        output_max=responses.max(axis=(0, 2)).astype(np.float32),
    )


def synthesize_features(
    rows: list[dict[str, Any]],
    count: int,
    config: PCAConfig,
    seed: int,
    mode: str = "raw_delta",
) -> tuple[np.ndarray, np.ndarray]:
    generator = fit_pca(rows, config, seed)
    responses, labels = generator.sample(count, seed + 1)
    times = np.arange(round(config.seconds / TIME_STEP_SECONDS) + 1) * TIME_STEP_SECONDS
    features = np.stack(
        [make_features(times, response.T, config.seconds, mode) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)