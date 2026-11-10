"""把连续浓度预测映射到实验设计中的离散浓度档位（决策规则）。

本项目的 62 个实验只使用固定的浓度档位（H2 4 档、NH3 5 档、NO2 5 档）。
若部署端的浓度就是这些标定档位，可以在回归输出后取最近档位作为最终判断；
该模块提供档位提取、最近档位映射和可选的容差规则，供评估脚本复用。
"""
from __future__ import annotations

import numpy as np

from tcn_union.evaluation.evaluate import AXES


def levels_from_labels(y_train: np.ndarray) -> tuple[np.ndarray, ...]:
    """从开发集标签提取每个气体的升序离散档位。

    参数：`y_train` 形状为 `[runs,3]` 的原始浓度。
    返回：长度 3 的元组，逐气体为 `np.unique` 后的档位数组。
    """
    if y_train.ndim != 2 or y_train.shape[1] != len(AXES):
        raise ValueError("y_train 必须是 [runs,3] 的浓度矩阵")
    return tuple(np.unique(y_train[:, index]) for index in range(len(AXES)))


def snap_to_levels(
    prediction: np.ndarray,
    levels: tuple[np.ndarray, ...],
    tolerance: float | None = None,
) -> np.ndarray:
    """把连续预测映射到最近的离散档位。

    参数：`prediction` 形状 `[runs,3]`；`levels` 为 `levels_from_labels` 的结果；
    `tolerance` 为 None 时总是取最近档位，否则只有相对距离
    ``|pred-level|/|level| <= tolerance`` 的预测才被吸附，其余保留连续值。
    返回：与输入同形状的映射结果。
    """
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(levels):
        raise ValueError("prediction 必须是 [runs,3] 且与 levels 数量一致")
    output = values.copy()
    for index, level in enumerate(levels):
        level = np.asarray(level, dtype=np.float64)
        if level.ndim != 1 or level.size == 0:
            raise ValueError("levels 中每个气体必须是非空一维数组")
        nearest = level[np.abs(values[:, index, None] - level[None, :]).argmin(axis=1)]
        if tolerance is None:
            output[:, index] = nearest
        else:
            distance = np.abs(values[:, index] - nearest) / np.maximum(
                np.abs(nearest), 1e-9
            )
            output[:, index] = np.where(distance <= tolerance, nearest, values[:, index])
    return output


def level_matches(
    truth: np.ndarray, prediction: np.ndarray, levels: tuple[np.ndarray, ...]
) -> np.ndarray:
    """判断吸附后的档位是否与真实标签一致，返回 `[runs,3]` 布尔矩阵。"""
    snapped = snap_to_levels(prediction, levels)
    truth = np.asarray(truth, dtype=np.float64)
    return np.isclose(snapped, truth, rtol=0.0, atol=1e-9)
