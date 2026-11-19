"""浓度特征、岭回归与逐实验形状初值。

这些函数只服务于双指数骨架的初始化，不包含任何响应模型。
"""
from __future__ import annotations

from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares

from .data import RESPONSE_SIGNS, Experiment

DELAY_BOUNDS = (0.0, 2.0)
TAU_BOUNDS = (0.1, 120.0)
SHAPE_STARTS = (0.0, 0.3)


def monomial_features(u: np.ndarray, degree: int) -> np.ndarray:
    """构造 [1, 一次项, ..., degree 次项] 的多项式特征，u 形状为 (N, k)。"""
    columns = [np.ones(len(u))]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(u.shape[1]), order):
            columns.append(np.prod(u[:, indices], axis=1))
    return np.column_stack(columns)


def ridge_least_squares(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """带岭惩罚的最小二乘；截距不惩罚。"""
    penalty = np.sqrt(ridge) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.lstsq(
        np.vstack([x, penalty]),
        np.r_[y, np.zeros(x.shape[1])],
        rcond=None,
    )[0]


def _shape_curve(time: np.ndarray, amplitude: float, tau: float, delay: float) -> np.ndarray:
    return amplitude * (1.0 - np.exp(-np.clip(time - delay, 0.0, None) / tau))


def _fit_one_shape(time: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """拟合单条有符号相对响应，返回 (A, tau, delay)。"""
    best = None
    for start_delay in SHAPE_STARTS:
        result = least_squares(
            lambda p: _shape_curve(time, p[0], p[1], p[2]) - target,
            x0=[max(float(target[-1]), 1e-4), 10.0, start_delay],
            bounds=(
                [0.0, TAU_BOUNDS[0], DELAY_BOUNDS[0]],
                [10.0, TAU_BOUNDS[1], DELAY_BOUNDS[1]],
            ),
            max_nfev=2000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    return float(best.x[0]), float(best.x[1]), float(best.x[2])


def fit_experiment_shapes(
    experiments: list[Experiment],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐实验拟合 (A, tau, delay)，用作双指数联合拟合的初值。

    返回三个 (N,3) 数组；逐实验拟合彼此独立，留一验证可预先算好再按折取子集。
    """
    time = experiments[0].time
    shapes = np.empty((len(experiments), 3, 3))
    for i, experiment in enumerate(experiments):
        relative = experiment.relative
        for j in range(3):
            target = RESPONSE_SIGNS[j] * relative[:, j]
            shapes[i, j] = _fit_one_shape(time, target)
    return shapes[:, :, 0], shapes[:, :, 1], shapes[:, :, 2]
