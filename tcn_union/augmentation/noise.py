"""基于真实测量噪声估计的轻量输入增强。

早期 cutout 实验里的噪声强度是按初始响应比例人为设定的，远大于真实测量
噪声，因此破坏了初值-标签关系。这里改为从训练折的平滑残差估计噪声：
对每条响应做二阶差分，用 MAD 估计高斯噪声标准差，再按倍率采样。

噪声只加到训练折的原始响应上；验证/测试保持原始曲线。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tcn_union.core.data import make_features


@dataclass(frozen=True)
class NoiseConfig:
    """噪声增强参数。"""

    scale: float = 1.0
    copies: int = 1

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale < 0.0:
            raise ValueError("noise scale 必须为非负有限值")
        if self.copies < 1:
            raise ValueError("noise copies 必须为正")


def estimate_noise_sigma(rows: list[dict[str, Any]], seconds: float) -> np.ndarray:
    """用二阶差分 MAD 估计三路响应的测量噪声标准差。"""
    sigmas = []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        prefix = response[time <= seconds + 1e-9]
        # 平滑曲线的二阶差分主要来自测量噪声；用 MAD 估计比直接 std 更抗离群点。
        second = np.diff(prefix, n=2, axis=0)
        if second.shape[0] == 0:
            continue
        median = np.median(second, axis=0)
        mad = np.median(np.abs(second - median[None, :]), axis=0)
        # 高斯白噪声的二阶差分方差为 6σ²，MAD 到标准差的换算系数为 0.6745。
        sigmas.append(mad / 0.6745 / np.sqrt(6.0))
    if not sigmas:
        raise ValueError("没有足够长的训练前缀来估计噪声")
    sigma = np.median(np.stack(sigmas), axis=0)
    if not np.isfinite(sigma).all() or np.any(sigma < 0.0):
        raise ValueError("噪声估计无效")
    return sigma.astype(np.float64)


def noisy_features(
    rows: list[dict[str, Any]],
    seconds: float,
    mode: str,
    config: NoiseConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """对每条训练响应加高斯噪声并返回特征和标签。"""
    # 每折重新估计噪声尺度；同一个 seed 下噪声视图可复现。
    rng = np.random.default_rng(seed)
    sigma = estimate_noise_sigma(rows, seconds) * float(config.scale)
    features = []
    labels = []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= seconds + 1e-9
        prefix = response[mask]
        prefix_time = time[mask]
        for _ in range(config.copies):
            # 只在原始响应上加噪声，标签保持不变。
            noisy = prefix + rng.normal(0.0, sigma[None, :], size=prefix.shape)
            features.append(make_features(prefix_time, noisy, seconds, mode))
            labels.append(np.asarray(row["_labels"], dtype=np.float64))
    return np.stack(features).astype(np.float32), np.stack(labels).astype(np.float32)
