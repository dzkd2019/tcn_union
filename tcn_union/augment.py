"""训练阶段专用的原始响应 cutout 增强。

最终模型只使用一种增强：在 10 秒响应内部随机挖掉一小段，用区间两端的
线性桥替换。增强只作用于训练折；验证、测试和推理始终使用原始曲线。
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import TIME_STEP_SECONDS, make_features

# 最终模型使用的 cutout 参数：每次训练都挖掉最多 2 秒。
CUTOUT_PROBABILITY = 1.0
CUTOUT_MAX_SECONDS = 2.0


@dataclass(frozen=True)
class CutoutConfig:
    """cutout 的两个参数，默认值就是最终模型使用的值。"""

    probability: float = CUTOUT_PROBABILITY
    max_seconds: float = CUTOUT_MAX_SECONDS

    def __post_init__(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("cutout 概率必须在 [0, 1] 内")
        if not np.isfinite(self.max_seconds) or self.max_seconds <= 0.0:
            raise ValueError("cutout 最大时长必须为正的有限值")


@dataclass(frozen=True)
class CutoutContext:
    """一折训练数据及其 cutout 设置。"""

    rows: Sequence[dict[str, Any]]
    seconds: float
    mode: str
    config: CutoutConfig
    base_seed: int

    def __post_init__(self) -> None:
        if self.seconds not in {5.0, 10.0}:
            raise ValueError("cutout 只支持 5 秒和 10 秒前缀")
        if self.mode not in {"raw_delta", "delta"}:
            raise ValueError("cutout 只支持 raw_delta 和 delta 模式")
        if len(self.rows) == 0:
            raise ValueError("cutout 至少需要一个训练实验")
        if self.config.max_seconds > self.seconds + 1e-9:
            raise ValueError("cutout 最大时长不能超过训练窗口")


def apply_cutout(responses: np.ndarray, start: int, length: int) -> np.ndarray:
    """把 ``[start, start+length)`` 区间替换为两端响应的线性桥。"""
    if responses.ndim != 2 or responses.shape[1] != 3:
        raise ValueError("responses 必须是 [时间, 3]")
    if start < 1 or length < 1 or start + length >= responses.shape[0]:
        raise ValueError("cutout 必须保留区间两侧的边界采样点")

    result = responses.copy()
    left = result[start - 1]
    right = result[start + length]
    weights = (np.arange(1, length + 1, dtype=np.float64) / (length + 1))[:, None]
    result[start : start + length] = left[None, :] * (1.0 - weights) + right[None, :] * weights
    return result


class CutoutDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """每个 epoch 在线生成一个 cutout 视图的数据集。"""

    def __init__(self, context: CutoutContext) -> None:
        self.context = context
        self.epoch = 0
        self._times = []
        self._responses = []
        self._labels = []
        for row in context.rows:
            time = np.asarray(row["_time"], dtype=np.float64)
            response = np.asarray(row["_response"], dtype=np.float64)
            mask = time <= context.seconds + 1e-9
            self._times.append(time[mask])
            self._responses.append(response[mask])
            self._labels.append(np.asarray(row["_labels"], dtype=np.float64))

        expected = round(context.seconds / TIME_STEP_SECONDS) + 1
        if any(len(time) != expected for time in self._times):
            raise ValueError("训练前缀与请求窗口不一致")

    def __len__(self) -> int:
        return len(self.context.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        config = self.context.config
        seed = (
            int(self.context.base_seed)
            + 1_000_003 * int(self.epoch)
            + 7919 * int(index)
        ) % (2**32)
        rng = np.random.default_rng(seed)

        responses = self._responses[index]
        # 先抽一次是否做 cutout；最终模型 probability=1.0，因此一定增强。
        if rng.random() >= config.probability:
            features = make_features(
                self._times[index], responses, self.context.seconds, self.context.mode
            )
        else:
            max_length = max(1, round(config.max_seconds / TIME_STEP_SECONDS))
            max_length = min(max_length, responses.shape[0] - 2)
            length = int(rng.integers(1, max_length + 1))
            start = int(rng.integers(1, responses.shape[0] - length))
            features = make_features(
                self._times[index],
                apply_cutout(responses, start, length),
                self.context.seconds,
                self.context.mode,
            )
        label = self._labels[index].astype(np.float32)
        return torch.from_numpy(features), torch.from_numpy(label)


def make_cutout_context(
    rows: Sequence[dict[str, Any]],
    seconds: float,
    mode: str,
    config: CutoutConfig,
    base_seed: int,
) -> CutoutContext:
    """构造一折的 cutout 上下文。"""
    return CutoutContext(
        rows=rows,
        seconds=seconds,
        mode=mode,
        config=config,
        base_seed=base_seed,
    )
