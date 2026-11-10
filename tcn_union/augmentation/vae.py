"""基于 PCA 响应基的条件 VAE，用于小样本响应曲线扩充。

直接对 51×3 原始曲线做 GAN 在 50 条样本上很容易生成不自然形状。这里改为：

1. 只用训练折真实曲线拟合每气体 PCA 响应基，生成曲线被限制在真实响应张成的
   低维子空间里；
2. 用训练标签拟合 PCA 系数的条件均值；
3. 条件 VAE 只学习条件均值周围的残差分布；
4. 采样时 = 条件均值 + VAE 残差，再反 PCA 回原始响应。

这样生成曲线天然平滑、接近真实响应形态，同时保留标签条件和样本间变化。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tcn_union.core.data import TIME_STEP_SECONDS, make_features
from tcn_union.training.train import set_seed


@dataclass(frozen=True)
class VAEConfig:
    """PCA-VAE 的默认配置。"""

    seconds: float = 10.0
    components: int = 6
    latent_dim: int = 4
    hidden_dim: int = 64
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta: float = 0.1
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.seconds not in {5.0, 10.0}:
            raise ValueError("VAE 只支持 5 秒或 10 秒窗口")
        if self.components < 1 or self.latent_dim < 1 or self.hidden_dim < 1:
            raise ValueError("VAE 维度必须为正")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("VAE epochs 必须为正，batch_size 至少为 2")
        if self.beta < 0.0:
            raise ValueError("VAE beta 不能为负")


class _Encoder(nn.Module):
    def __init__(self, config: VAEConfig, coefficient_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(coefficient_dim + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, coefficient: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(torch.cat((coefficient, condition), dim=1))
        return self.mu(hidden), self.logvar(hidden)


class _Decoder(nn.Module):
    def __init__(self, config: VAEConfig, coefficient_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, coefficient_dim),
        )

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((latent, condition), dim=1))


class TrainedVAE:
    """训练好的 PCA-VAE 及反变换所需统计量。"""

    def __init__(
        self,
        decoder: _Decoder,
        config: VAEConfig,
        means: np.ndarray,
        components: np.ndarray,
        coefficient_std: np.ndarray,
        mean_coef: np.ndarray,
        labels_z: np.ndarray,
        label_mean: np.ndarray,
        label_std: np.ndarray,
        output_min: np.ndarray,
        output_max: np.ndarray,
    ) -> None:
        self.decoder = decoder
        self.config = config
        self.means = means
        self.components = components
        self.coefficient_std = coefficient_std
        self.mean_coef = mean_coef
        self.labels_z = labels_z
        self.label_mean = label_mean
        self.label_std = label_std
        self.output_min = output_min
        self.output_max = output_max

    def sample(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """采样合成响应和对应浓度标签。"""
        if count < 1:
            raise ValueError("VAE 采样数量必须为正")
        rng = torch.Generator().manual_seed(seed)
        self.decoder.eval()
        device = next(self.decoder.parameters()).device
        with torch.no_grad():
            latent = torch.randn(count, self.config.latent_dim, generator=rng).to(device)
            indices = torch.randint(0, len(self.labels_z), (count,), generator=rng)
            condition = torch.from_numpy(self.labels_z[indices.numpy()]).to(device)
            residual = self.decoder(latent, condition).cpu().numpy().reshape(count, 3, -1)
        labels = condition.cpu().numpy() * self.label_std[None, :] + self.label_mean[None, :]
        design = np.concatenate((np.ones((count, 1)), condition.cpu().numpy()), axis=1)
        base = (design @ self.mean_coef).reshape(count, 3, -1)
        coefficients = (base + residual) * self.coefficient_std[None, :, :]
        responses = np.empty((count, 3, self.components.shape[2]), dtype=np.float64)
        for gas in range(3):
            responses[:, gas, :] = self.means[gas][None, :] + coefficients[:, gas, :] @ self.components[gas]
        responses = np.clip(
            responses,
            self.output_min[None, :, None],
            self.output_max[None, :, None],
        )
        return responses, labels

    def component(self) -> int:
        return self.components.shape[1]


def _prefix_arrays(rows: list[dict[str, Any]], seconds: float) -> tuple[np.ndarray, np.ndarray]:
    expected = round(seconds / TIME_STEP_SECONDS) + 1
    responses = []
    labels = []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= seconds + 1e-9
        if int(mask.sum()) != expected:
            raise ValueError("训练前缀长度与 VAE 窗口不一致")
        responses.append(response[mask].T)
        labels.append(np.asarray(row["_labels"], dtype=np.float64))
    return np.stack(responses), np.stack(labels).astype(np.float32)


def fit_vae(
    rows: list[dict[str, Any]],
    config: VAEConfig,
    seed: int,
    device: torch.device,
) -> TrainedVAE:
    """只用给定训练行拟合 PCA-VAE。"""
    if len(rows) < 2:
        raise ValueError("VAE 至少需要两个训练实验")
    set_seed(seed)
    responses, labels = _prefix_arrays(rows, config.seconds)
    n, _, length = responses.shape
    components = min(config.components, n - 1, length)
    if components < 1:
        raise ValueError("样本太少，无法拟合 PCA 响应基")

    means = []
    bases = []
    coefficients = []
    # 每个气体单独做 PCA：H2/NH3/NO2 的幅值和形态差异太大，混在一起不合适。
    for gas in range(3):
        flat = responses[:, gas, :]
        mean = flat.mean(axis=0)          # 平均响应曲线
        centered = flat - mean[None, :]   # 每条曲线相对平均形状的差异
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:components]           # 前 K 个主方向
        coefficient = centered @ basis.T  # 每条曲线压缩成 K 个系数
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
    coefficient_z = coefficient_array / coefficient_std[None, None, :]
    # 用三个浓度标签线性预测 PCA 系数，得到“标签决定的平均趋势”。
    design = np.concatenate((np.ones((n, 1)), labels_z), axis=1)
    mean_coef = np.linalg.lstsq(design, coefficient_z.reshape(n, -1), rcond=None)[0]
    base = (design @ mean_coef).reshape(n, 3, components)
    # VAE 只学标签解释不了的残差：个体差异、噪声和未观测因素。
    residual = coefficient_z - base

    coefficient_dim = 3 * components
    encoder = _Encoder(config, coefficient_dim).to(device)
    decoder = _Decoder(config, coefficient_dim).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(residual.reshape(n, -1).astype(np.float32)),
            torch.from_numpy(labels_z),
        ),
        batch_size=min(config.batch_size, n),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    for _ in range(config.epochs):
        for batch_residual, batch_label in loader:
            batch_residual = batch_residual.to(device)
            batch_label = batch_label.to(device)
            # 编码器输出隐变量分布；重参数化让采样仍可反向传播。
            mu, logvar = encoder(batch_residual, batch_label)
            std = torch.exp(0.5 * logvar)
            noise = torch.randn_like(std)
            latent = mu + std * noise
            # 解码器重建残差；KL 约束隐变量分布接近标准正态。
            reconstructed = decoder(latent, batch_label)
            reconstruction = ((reconstructed - batch_residual) ** 2).mean()
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction + config.beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()

    output_min = responses.min(axis=(0, 2)).astype(np.float32)
    output_max = responses.max(axis=(0, 2)).astype(np.float32)
    decoder.eval()
    return TrainedVAE(
        decoder=decoder,
        config=config,
        means=means_array,
        components=components_array,
        coefficient_std=coefficient_std,
        mean_coef=mean_coef,
        labels_z=labels_z,
        label_mean=label_mean,
        label_std=label_std,
        output_min=output_min,
        output_max=output_max,
    )


def synthesize_features(
    rows: list[dict[str, Any]],
    count: int,
    config: VAEConfig,
    seed: int,
    device: torch.device,
    mode: str = "raw_delta",
) -> tuple[np.ndarray, np.ndarray]:
    """拟合 PCA-VAE 并返回下游 TCN 可用的特征。"""
    vae = fit_vae(rows, config, seed, device)
    responses, labels = vae.sample(count, seed + 1)
    times = np.arange(round(config.seconds / TIME_STEP_SECONDS) + 1) * TIME_STEP_SECONDS
    features = np.stack(
        [make_features(times, response.T, config.seconds, mode) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)
