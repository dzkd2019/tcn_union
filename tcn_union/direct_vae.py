"""直接对 51×3 响应曲线训练的条件 VAE 基线。

与 PCA-VAE 的区别：不经过 PCA，解码器直接输出完整曲线。
只用于对比实验。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TIME_STEP_SECONDS, make_features
from .train import set_seed


@dataclass(frozen=True)
class DirectVAEConfig:
    seconds: float = 10.0
    latent_dim: int = 4
    hidden_dim: int = 64
    epochs: int = 300
    batch_size: int = 16
    learning_rate: float = 1e-3
    beta: float = 0.1
    max_grad_norm: float = 10.0

    def __post_init__(self) -> None:
        if self.seconds not in {5.0, 10.0}:
            raise ValueError("direct VAE 只支持 5 秒或 10 秒窗口")
        if self.latent_dim < 1 or self.hidden_dim < 1:
            raise ValueError("direct VAE 维度必须为正")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("direct VAE epochs 必须为正，batch_size 至少为 2")
        if self.beta < 0.0:
            raise ValueError("direct VAE beta 不能为负")


class _Encoder(nn.Module):
    def __init__(self, config: DirectVAEConfig, curve_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(curve_dim + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.logvar = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, curve: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(torch.cat((curve, condition), dim=1))
        return self.mu(hidden), self.logvar(hidden)


class _Decoder(nn.Module):
    def __init__(self, config: DirectVAEConfig, curve_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.latent_dim + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, curve_dim),
        )

    def forward(self, latent: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((latent, condition), dim=1))


class TrainedDirectVAE:
    def __init__(
        self,
        decoder: _Decoder,
        config: DirectVAEConfig,
        curve_mean: np.ndarray,
        curve_std: np.ndarray,
        label_mean: np.ndarray,
        label_std: np.ndarray,
        labels_z: np.ndarray,
        output_min: np.ndarray,
        output_max: np.ndarray,
    ) -> None:
        self.decoder = decoder
        self.config = config
        self.curve_mean = curve_mean
        self.curve_std = curve_std
        self.label_mean = label_mean
        self.label_std = label_std
        self.labels_z = labels_z
        self.output_min = output_min
        self.output_max = output_max

    def sample(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = torch.Generator().manual_seed(seed)
        self.decoder.eval()
        device = next(self.decoder.parameters()).device
        with torch.no_grad():
            latent = torch.randn(count, self.config.latent_dim, generator=rng).to(device)
            indices = torch.randint(0, len(self.labels_z), (count,), generator=rng)
            condition = torch.from_numpy(self.labels_z[indices.numpy()]).to(device)
            curve = self.decoder(latent, condition).cpu().numpy()
        curve = curve.reshape(count, 3, -1) * self.curve_std[None, :, None] + self.curve_mean[None, :, None]
        curve = np.clip(curve, self.output_min[None, :, None], self.output_max[None, :, None])
        labels = condition.cpu().numpy() * self.label_std[None, :] + self.label_mean[None, :]
        return curve, labels


def _prefix_arrays(rows: list[dict[str, Any]], seconds: float) -> tuple[np.ndarray, np.ndarray]:
    expected = round(seconds / TIME_STEP_SECONDS) + 1
    responses, labels = [], []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= seconds + 1e-9
        if int(mask.sum()) != expected:
            raise ValueError("训练前缀长度与 direct VAE 窗口不一致")
        responses.append(response[mask].T)
        labels.append(np.asarray(row["_labels"], dtype=np.float64))
    return np.stack(responses), np.stack(labels).astype(np.float32)


def fit_direct_vae(
    rows: list[dict[str, Any]],
    config: DirectVAEConfig,
    seed: int,
    device: torch.device,
) -> TrainedDirectVAE:
    if len(rows) < 2:
        raise ValueError("direct VAE 至少需要两个训练实验")
    set_seed(seed)
    responses, labels = _prefix_arrays(rows, config.seconds)
    n, _, length = responses.shape
    curve_mean = responses.mean(axis=(0, 2)).astype(np.float32)
    curve_std = responses.std(axis=(0, 2)).astype(np.float32)
    curve_std[curve_std < 1e-8] = 1.0
    curves = torch.from_numpy(
        ((responses - curve_mean[None, :, None]) / curve_std[None, :, None])
        .reshape(n, -1)
        .astype(np.float32)
    ).to(device)
    label_mean = labels.mean(axis=0).astype(np.float32)
    label_std = labels.std(axis=0).astype(np.float32)
    label_std[label_std < 1e-8] = 1.0
    labels_z = ((labels - label_mean[None, :]) / label_std[None, :]).astype(np.float32)
    conditions = torch.from_numpy(labels_z).to(device)

    encoder = _Encoder(config, 3 * length).to(device)
    decoder = _Decoder(config, 3 * length).to(device)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=config.learning_rate,
    )
    loader = DataLoader(
        TensorDataset(curves, conditions),
        batch_size=min(config.batch_size, n),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    for _ in range(config.epochs):
        for batch_curve, batch_label in loader:
            mu, logvar = encoder(batch_curve, batch_label)
            std = torch.exp(0.5 * logvar)
            latent = mu + std * torch.randn_like(std)
            reconstructed = decoder(latent, batch_label)
            reconstruction = ((reconstructed - batch_curve) ** 2).mean()
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = reconstruction + config.beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(decoder.parameters()),
                config.max_grad_norm,
            )
            optimizer.step()
    decoder.eval()
    return TrainedDirectVAE(
        decoder=decoder,
        config=config,
        curve_mean=curve_mean,
        curve_std=curve_std,
        label_mean=label_mean,
        label_std=label_std,
        labels_z=labels_z,
        output_min=responses.min(axis=(0, 2)).astype(np.float32),
        output_max=responses.max(axis=(0, 2)).astype(np.float32),
    )


def synthesize_features(
    rows: list[dict[str, Any]],
    count: int,
    config: DirectVAEConfig,
    seed: int,
    device: torch.device,
    mode: str = "raw_delta",
) -> tuple[np.ndarray, np.ndarray]:
    generator = fit_direct_vae(rows, config, seed, device)
    responses, labels = generator.sample(count, seed + 1)
    times = np.arange(round(config.seconds / TIME_STEP_SECONDS) + 1) * TIME_STEP_SECONDS
    features = np.stack(
        [make_features(times, response.T, config.seconds, mode) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)
