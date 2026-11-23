"""由 5 s / 10 s 窗口反演三气体浓度。

逆问题在 log 浓度空间求解，使用残差加权最小二乘、对数正态先验和边界软墙。
Laplace 协方差只作为不确定度诊断，不参与模型选择。
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from ..core.data import GASES, REFERENCE_CONCENTRATION, Experiment

NOISE_FLOOR = 1e-5
PRIOR_SIGMA = 1.5
WALL = 0.15


def estimate_noise(experiments: list[Experiment]) -> np.ndarray:
    """用相邻三点二阶差分估计各通道的相对响应噪声。"""
    relative = np.stack([experiment.relative for experiment in experiments])
    second = relative[:, :-2] - 2.0 * relative[:, 1:-1] + relative[:, 2:]
    mad = np.median(np.abs(second - np.median(second, axis=(0, 1))), axis=(0, 1))
    return np.maximum(1.4826 * mad / np.sqrt(6.0), NOISE_FLOOR)


def concentration_bounds(
    experiments: list[Experiment], margin: float = 0.10
) -> np.ndarray:
    """设计域在 log 浓度空间的边界，形状 (3,2)。"""
    concentration = np.stack([experiment.concentration for experiment in experiments])
    low = np.log(concentration.min(axis=0))
    high = np.log(concentration.max(axis=0))
    span = high - low
    return np.column_stack([low - margin * np.maximum(span, 1e-6), high + margin * np.maximum(span, 1e-6)])


def start_points(experiments: list[Experiment], n: int = 3) -> list[np.ndarray]:
    """确定性地给出若干 log 浓度起点，覆盖设计域中心与两端。"""
    concentration = np.stack([experiment.concentration for experiment in experiments])
    quantiles = (0.5,) if n == 1 else np.linspace(0.2, 0.8, n)
    starts = [np.median(concentration, axis=0)]
    for q in quantiles:
        starts.append(np.percentile(concentration, 100.0 * q, axis=0))
    unique = []
    for point in starts:
        point = np.clip(point, 1e-9, None)
        if not any(np.allclose(point, other) for other in unique):
            unique.append(point)
    return unique


def _residual_vector(
    model,
    log_theta: np.ndarray,
    time: np.ndarray,
    relative: np.ndarray,
    sigma: np.ndarray,
    bounds: np.ndarray,
    channels: np.ndarray,
) -> np.ndarray:
    prediction = model.predict_relative(np.exp(log_theta)[None, :], time)[0]
    data = (prediction[:, channels] - relative[:, channels]) / sigma[channels][None, :]
    prior = (log_theta - np.log(REFERENCE_CONCENTRATION)) / PRIOR_SIGMA
    span = bounds[:, 1] - bounds[:, 0]
    wall_width = np.maximum(np.minimum(WALL, 0.25 * span), 1e-9)
    wall = np.zeros(3)
    for j in range(3):
        if log_theta[j] < bounds[j, 0] + wall_width[j]:
            wall[j] = (bounds[j, 0] + wall_width[j] - log_theta[j]) / wall_width[j]
        elif log_theta[j] > bounds[j, 1] - wall_width[j]:
            wall[j] = (log_theta[j] - bounds[j, 1] + wall_width[j]) / wall_width[j]
    return np.r_[data.ravel(), prior, wall]


def map_estimate(
    model,
    time: np.ndarray,
    relative: np.ndarray,
    sigma: np.ndarray,
    bounds: np.ndarray,
    starts: list[np.ndarray] | None = None,
    channels: np.ndarray | None = None,
) -> dict:
    """在给定时间窗口上做 MAP 估计并返回 Laplace 标准误。

    ``channels`` 指定参与反演的传感器通道；默认三路全部参与。
    """
    time = np.asarray(time, dtype=float)
    relative = np.asarray(relative, dtype=float)
    if channels is None:
        channels = np.arange(relative.shape[1])
    channels = np.asarray(channels, dtype=int)
    sigma = np.asarray(sigma, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    if starts is None:
        starts = [np.exp((bounds[:, 0] + bounds[:, 1]) / 2.0)]
    best = None
    span = bounds[:, 1] - bounds[:, 0]
    wall_width = np.maximum(np.minimum(WALL, 0.25 * span), 1e-9)
    for start in starts:
        log_start = np.clip(
            np.log(np.asarray(start, dtype=float)),
            bounds[:, 0] + wall_width,
            bounds[:, 1] - wall_width,
        )
        result = least_squares(
            lambda value: _residual_vector(model, value, time, relative, sigma, bounds, channels),
            x0=log_start,
            bounds=(bounds[:, 0], bounds[:, 1]),
            max_nfev=300,
            x_scale=0.2,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    n_data = relative[:, channels].size
    dof = max(n_data - 3, 1)
    covariance = np.linalg.pinv(best.jac.T @ best.jac) * (2.0 * best.cost / dof)
    se_log = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "theta": np.exp(best.x),
        "log_theta": best.x,
        "se_log": se_log,
        "se": np.exp(best.x) * se_log,
        "cost": float(best.cost),
        "success": bool(best.success),
        "nfev": int(best.nfev),
    }
