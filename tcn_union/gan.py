"""条件 GAN：生成与浓度标签对应的 10 秒原始响应片段。

这里的目标不是“生成任意曲线”，而是在 **训练折内部** 学习
``(浓度标签) -> (三路响应前缀)`` 的条件分布，用来补充小样本。
生成器只看到训练折；验证折和测试折不参与 GAN 拟合。

训练加入两个物理先验：
- 平滑惩罚：抑制高频锯齿；
- 范围惩罚：把生成值拉回训练集逐通道的观测范围。
它们不能保证完全符合传感器物理，只能减少明显不合理的曲线。
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
class GANConfig:
    """小样本条件 GAN 的默认配置。"""

    seconds: float = 10.0
    noise_dim: int = 32
    hidden_dim: int = 64
    control_points: int = 11
    epochs: int = 300
    batch_size: int = 16
    lr: float = 1e-4
    n_critic: int = 5
    lambda_gp: float = 10.0
    lambda_smooth: float = 1e-3
    lambda_range: float = 1e-3
    max_grad_norm: float = 100.0

    def __post_init__(self) -> None:
        if self.seconds not in {5.0, 10.0}:
            raise ValueError("GAN 只支持 5 秒或 10 秒窗口")
        if self.epochs < 1 or self.batch_size < 2:
            raise ValueError("GAN epochs 必须为正，batch_size 至少为 2")
        if self.noise_dim < 1 or self.hidden_dim < 1:
            raise ValueError("GAN 隐层维度必须为正")
        if self.control_points < 2 or self.control_points > 51:
            raise ValueError("control_points 必须在 [2, 51] 内")
        if self.n_critic < 1:
            raise ValueError("n_critic 至少为 1")
        if self.lambda_gp < 0.0 or self.lambda_smooth < 0.0 or self.lambda_range < 0.0:
            raise ValueError("GAN 损失权重不能为负")


class _Generator(nn.Module):
    def __init__(self, config: GANConfig, length: int) -> None:
        super().__init__()
        self.length = length
        self.control_points = config.control_points
        self.net = nn.Sequential(
            nn.Linear(config.noise_dim + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, 3 * self.control_points),
        )

    def forward(self, noise: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """生成低分辨率控制点，再线性插值成完整响应曲线。"""
        output = self.net(torch.cat((noise, condition), dim=1))
        control = output.view(-1, 3, self.control_points)
        return torch.nn.functional.interpolate(
            control, size=self.length, mode="linear", align_corners=True
        )


class _Critic(nn.Module):
    def __init__(self, config: GANConfig, length: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, config.hidden_dim, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, length)
            flattened = self.features(dummy).flatten(1).shape[1]
        self.condition = nn.Linear(3, config.hidden_dim)
        self.output = nn.Linear(flattened + config.hidden_dim, 1)

    def forward(self, response: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        features = self.features(response).flatten(1)
        return self.output(torch.cat((features, self.condition(condition)), dim=1))


def _prefix_arrays(rows: list[dict[str, Any]], seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """从训练行提取固定长度响应前缀和标签。"""
    responses = []
    labels = []
    expected = round(seconds / TIME_STEP_SECONDS) + 1
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= seconds + 1e-9
        if int(mask.sum()) != expected:
            raise ValueError("训练前缀长度与 GAN 窗口不一致")
        responses.append(response[mask].T)
        labels.append(np.asarray(row["_labels"], dtype=np.float64))
    return np.stack(responses), np.stack(labels).astype(np.float32)


def _gradient_penalty(
    critic: _Critic,
    real: torch.Tensor,
    fake: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, 1, device=real.device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).detach().requires_grad_(True)
    score = critic(interpolated, condition)
    gradient = torch.autograd.grad(
        outputs=score.sum(),
        inputs=interpolated,
        create_graph=True,
        retain_graph=True,
    )[0]
    norms = gradient.flatten(1).norm(2, dim=1)
    return ((norms - 1.0) ** 2).mean()


class TrainedGAN:
    """训练好的生成器及其标准化统计量。"""

    def __init__(
        self,
        generator: _Generator,
        config: GANConfig,
        x_mean: np.ndarray,
        x_std: np.ndarray,
        x_min: np.ndarray,
        x_max: np.ndarray,
        y_mean: np.ndarray,
        y_std: np.ndarray,
        labels_z: np.ndarray,
    ) -> None:
        self.generator = generator
        self.config = config
        self.x_mean = x_mean
        self.x_std = x_std
        self.x_min = x_min
        self.x_max = x_max
        self.y_mean = y_mean
        self.y_std = y_std
        self.labels_z = labels_z

    def sample(self, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """采样 ``count`` 条响应和对应的训练标签。"""
        if count < 1:
            raise ValueError("GAN 采样数量必须为正")
        generator = torch.Generator().manual_seed(seed)
        self.generator.eval()
        device = next(self.generator.parameters()).device
        with torch.no_grad():
            noise = torch.randn(count, self.config.noise_dim, generator=generator).to(device)
            indices = torch.randint(
                0, len(self.labels_z), (count,), generator=generator
            )
            condition = torch.from_numpy(self.labels_z[indices.numpy()]).to(device)
            generated = self.generator(noise, condition).cpu().numpy()
        responses = generated * self.x_std[None, :, None] + self.x_mean[None, :, None]
        responses = np.clip(responses, self.x_min[None, :, None], self.x_max[None, :, None])
        labels = condition.cpu().numpy() * self.y_std[None, :] + self.y_mean[None, :]
        return responses.astype(np.float64), labels.astype(np.float64)


def fit_gan(
    rows: list[dict[str, Any]],
    config: GANConfig,
    seed: int,
    device: torch.device,
) -> TrainedGAN:
    """只用给定训练行拟合条件 WGAN-GP。"""
    if len(rows) < 2:
        raise ValueError("GAN 至少需要两个训练实验")
    set_seed(seed)
    responses, labels = _prefix_arrays(rows, config.seconds)
    length = responses.shape[2]
    x_mean = responses.mean(axis=(0, 2)).astype(np.float32)
    x_std = responses.std(axis=(0, 2)).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0
    x_min = responses.min(axis=(0, 2)).astype(np.float32)
    x_max = responses.max(axis=(0, 2)).astype(np.float32)
    y_mean = labels.mean(axis=0).astype(np.float32)
    y_std = labels.std(axis=0).astype(np.float32)
    y_std[y_std < 1e-8] = 1.0

    real_x = torch.from_numpy(
        ((responses - x_mean[None, :, None]) / x_std[None, :, None]).astype(np.float32)
    ).to(device)
    real_y = torch.from_numpy(((labels - y_mean[None, :]) / y_std[None, :]).astype(np.float32)).to(device)
    real_min = torch.from_numpy(((x_min - x_mean) / x_std).astype(np.float32)).to(device)
    real_max = torch.from_numpy(((x_max - x_mean) / x_std).astype(np.float32)).to(device)

    generator = _Generator(config, length).to(device)
    critic = _Critic(config, length).to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=config.lr, betas=(0.5, 0.999))
    opt_c = torch.optim.Adam(critic.parameters(), lr=config.lr, betas=(0.5, 0.999))
    loader = DataLoader(
        TensorDataset(real_x, real_y),
        batch_size=min(config.batch_size, len(rows)),
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )

    for _ in range(config.epochs):
        for batch_x, batch_y in loader:
            batch_size = batch_x.shape[0]
            for _ in range(config.n_critic):
                noise = torch.randn(batch_size, config.noise_dim, device=device)
                condition = batch_y[torch.randperm(batch_size, device=device)]
                fake = generator(noise, condition).detach()
                loss_c = (
                    critic(fake, condition).mean()
                    - critic(batch_x, batch_y).mean()
                    + config.lambda_gp * _gradient_penalty(critic, batch_x, fake, condition)
                )
                opt_c.zero_grad(set_to_none=True)
                loss_c.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), config.max_grad_norm)
                opt_c.step()

            noise = torch.randn(batch_size, config.noise_dim, device=device)
            condition = batch_y[torch.randperm(batch_size, device=device)]
            fake = generator(noise, condition)
            smoothness = ((fake[:, :, 2:] - 2.0 * fake[:, :, 1:-1] + fake[:, :, :-2]) ** 2).mean()
            out_of_range = (
                torch.relu(fake - real_max[None, :, None]).mean()
                + torch.relu(real_min[None, :, None] - fake).mean()
            )
            loss_g = (
                -critic(fake, condition).mean()
                + config.lambda_smooth * smoothness
                + config.lambda_range * out_of_range
            )
            opt_g.zero_grad(set_to_none=True)
            loss_g.backward()
            nn.utils.clip_grad_norm_(generator.parameters(), config.max_grad_norm)
            opt_g.step()

    return TrainedGAN(
        generator=generator,
        config=config,
        x_mean=x_mean,
        x_std=x_std,
        x_min=x_min,
        x_max=x_max,
        y_mean=y_mean,
        y_std=y_std,
        labels_z=((labels - y_mean[None, :]) / y_std[None, :]).astype(np.float32),
    )


def synthesize_features(
    rows: list[dict[str, Any]],
    count: int,
    config: GANConfig,
    seed: int,
    device: torch.device,
    mode: str = "raw_delta",
) -> tuple[np.ndarray, np.ndarray]:
    """拟合 GAN 并返回下游 TCN 可直接使用的 ``(特征, 标签)``。"""
    gan = fit_gan(rows, config, seed, device)
    responses, labels = gan.sample(count, seed + 1)
    times = np.arange(round(config.seconds / TIME_STEP_SECONDS) + 1) * TIME_STEP_SECONDS
    features = np.stack(
        [make_features(times, response.T, config.seconds, mode) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)
