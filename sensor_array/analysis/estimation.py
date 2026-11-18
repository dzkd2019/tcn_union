"""恒浓度 MAP 估计、Laplace 区间、CV+ conformal 与自适应停止。

估计器把一段相对响应 y_obs(t) 映射回浓度 c：

    theta = log c
    theta_hat = argmin_theta  sum_{j,t} ((y_model(t,c)-y_obs(t))/sigma_eff_j)^2
                              + ||theta-theta_prior||^2 / prior_sigma^2

其中 sigma_eff_j = sqrt(sigma_meas_j^2 + sigma_model_j^2)，sigma_model 是训练折
上混合模型残差的 RMS。这样做是因为测量噪声极低（H2 通道只有 1e-5），若只用
测量噪声，似然会把 H2 的模型偏差当成信号，反而扭曲估计。

Laplace 区间用数值 Jacobian 的 Gauss-Newton Hessian 求 log 浓度的协方差；
CV+ conformal 用 62 折的折外误差校准区间；自适应停止则找最早满足目标相对
半宽的窗长。
"""
from __future__ import annotations

import argparse
from itertools import product

import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr

from ..core.data import Experiment, load_experiments
from ..core.hybrid import HybridArrayModel
from .identifiability import estimate_noise
from ..core.grey_box import fit_experiment_shapes
from ..paths import ANALYSIS, ensure

WINDOWS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0)
ALPHAS = (0.1, 0.2)
STOP_TARGETS = (0.05, 0.10, 0.20, 0.30)
PRIOR_SIGMA = 1.0
DECISION_ALPHA = 0.1
COVARIANCE_SHRINKAGE = 0.1
Z90 = 1.6448536269514722
PROFILE_DELTA = 3.84
PROFILE_MAX_STEP = 2.0
LOSS_RATIOS = (1.0, 3.0, 10.0)
LOSS_TOLERANCES = (0.05, 0.10, 0.20)


def concentration_bounds(experiments: list[Experiment]) -> tuple[np.ndarray, np.ndarray]:
    concentration = np.stack([experiment.concentration for experiment in experiments])
    return concentration.min(axis=0), concentration.max(axis=0)


def residual_covariance(
    residuals: np.ndarray,
    noise: np.ndarray | None = None,
    shrinkage: float = COVARIANCE_SHRINKAGE,
) -> np.ndarray:
    """把折外残差 (N,T,3) 收缩成 (T*3,T*3) 的误差协方差。

    残差跨时间高度相关，直接用逐点方差会把一条平滑的模型偏差当成独立噪声。
    这里对方差-相关矩阵做对角收缩，并可叠加测量噪声；行顺序与
    (T,3).ravel() 的 C 顺序一致。
    """
    residuals = np.asarray(residuals, dtype=float)
    if residuals.ndim != 3:
        raise ValueError("residuals 应为 (N,T,3)")
    n_samples, n_time, n_gas = residuals.shape
    if n_samples < 2:
        raise ValueError("估计协方差至少需要两条残差曲线")
    flat = residuals.reshape(n_samples, n_time * n_gas)
    covariance = np.cov(flat, rowvar=False, bias=True)
    variance = np.maximum(np.diag(covariance), 0.0)
    scale = np.sqrt(variance)
    denominator = np.maximum(np.outer(scale, scale), np.finfo(float).tiny)
    correlation = covariance / denominator
    correlation = (1.0 - shrinkage) * correlation + shrinkage * np.eye(n_time * n_gas)
    covariance = correlation * np.outer(scale, scale)
    if noise is not None:
        covariance = covariance + np.diag(np.tile(np.asarray(noise, dtype=float), n_time))
    return 0.5 * (covariance + covariance.T)


def covariance_cholesky(covariance: np.ndarray) -> np.ndarray:
    """对称化并加抖动后做 Cholesky 分解，返回下三角 L。"""
    covariance = np.asarray(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("covariance 应为方阵")
    covariance = 0.5 * (covariance + covariance.T)
    scale = max(float(np.trace(covariance)) / len(covariance), 1e-12)
    jitter = scale * 1e-10
    identity = np.eye(len(covariance))
    for _ in range(8):
        try:
            return np.linalg.cholesky(covariance + jitter * identity)
        except np.linalg.LinAlgError:
            jitter *= 10.0
    raise np.linalg.LinAlgError("协方差矩阵无法做 Cholesky 分解")


def _whiten(cholesky: np.ndarray, values: np.ndarray) -> np.ndarray:
    """用 L^{-1} 白化残差或 Jacobian。"""
    return solve_triangular(cholesky, values, lower=True, check_finite=False)


def _window_covariance(covariance: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """取 (T,3) 时间窗对应的协方差子块。"""
    indices = (3 * np.flatnonzero(mask)[:, None] + np.arange(3)).ravel()
    return covariance[np.ix_(indices, indices)]


def _start_points(low: np.ndarray, high: np.ndarray) -> list[np.ndarray]:
    """域中心 + 八个角点，作为多起点优化的初值。"""
    center = np.log((low + high) / 2.0)
    corners = [
        np.log(np.array([high[0] if bit else low[0] for bit in bits]))
        for bits in product((0, 1), repeat=3)
    ]
    return [center] + corners


def _screen_starts(
    model: HybridArrayModel,
    relative: np.ndarray,
    time: np.ndarray,
    sigma: np.ndarray,
    starts: list[np.ndarray],
    prior_center: np.ndarray,
    prior_sigma: float,
    keep: int = 2,
    cholesky: np.ndarray | None = None,
) -> list[np.ndarray]:
    """先用一次前向残差给多起点排序，只对最好的两个起点做优化。"""
    costs = []
    for start in starts:
        prediction = model.predict_relative(np.exp(start)[None, :], time)[0]
        if cholesky is None:
            residual = ((prediction - relative) / sigma[None, :]).ravel()
        else:
            residual = _whiten(cholesky, (prediction - relative).ravel())
        costs.append(
            float(np.sum(residual**2) + np.sum(((start - prior_center) / prior_sigma) ** 2))
        )
    order = np.argsort(costs)[:keep]
    return [starts[index] for index in order]


def _residual_vector(
    model: HybridArrayModel,
    relative: np.ndarray,
    time: np.ndarray,
    sigma: np.ndarray | None,
    theta: np.ndarray,
    prior_center: np.ndarray,
    prior_sigma: float,
    log_low: np.ndarray,
    log_high: np.ndarray,
    wall: float,
    cholesky: np.ndarray | None,
) -> np.ndarray:
    """MAP 的加权数据残差 + 先验 + 软墙，供优化与剖面共用。"""
    prediction = model.predict_relative(np.exp(theta)[None, :], time)[0]
    if cholesky is None:
        data_residual = ((prediction - relative) / sigma[None, :]).ravel()
    else:
        data_residual = _whiten(cholesky, (prediction - relative).ravel())
    return np.r_[
        data_residual,
        (theta - prior_center) / prior_sigma,
        np.maximum(log_low - theta, 0.0) / wall,
        np.maximum(theta - log_high, 0.0) / wall,
    ]


def map_estimate(
    model: HybridArrayModel,
    relative: np.ndarray,
    time: np.ndarray,
    sigma: np.ndarray | None,
    low: np.ndarray,
    high: np.ndarray,
    starts: list[np.ndarray] | None = None,
    prior_sigma: float = PRIOR_SIGMA,
    wall: float = 0.15,
    covariance: np.ndarray | None = None,
) -> dict:
    """对一条相对响应曲线做 MAP 估计，返回 log 浓度与 Laplace 协方差。

    设计域用软墙约束：越界残差按 ``(theta - bound) / wall`` 增长，
    避免硬边界把估计值钉在边界上、制造虚假的零误差。
    """
    time = np.asarray(time, dtype=float)
    relative = np.asarray(relative, dtype=float)
    if covariance is not None:
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (relative.size, relative.size):
            raise ValueError("covariance 形状应与 (T,3) 展平后的长度一致")
        cholesky = covariance_cholesky(covariance)
        sigma_weights = None
    else:
        if sigma is None:
            raise ValueError("sigma 与 covariance 至少提供一个")
        sigma_weights = np.asarray(sigma, dtype=float)
        cholesky = None
    if starts is None:
        starts = _start_points(low, high)
    log_low, log_high = np.log(low), np.log(high)
    prior_center = 0.5 * (log_low + log_high)
    starts = _screen_starts(
        model,
        relative,
        time,
        sigma_weights,
        starts,
        prior_center,
        prior_sigma,
        cholesky=cholesky,
    )

    def residual(theta: np.ndarray) -> np.ndarray:
        return _residual_vector(
            model,
            relative,
            time,
            sigma_weights,
            theta,
            prior_center,
            prior_sigma,
            log_low,
            log_high,
            wall,
            cholesky,
        )

    best = None
    for start in starts:
        result = least_squares(
            residual,
            np.clip(start, log_low, log_high),
            x_scale="jac",
            max_nfev=100,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    covariance, scale = laplace_covariance(
        model,
        best.x,
        relative,
        time,
        sigma_weights,
        prior_sigma,
        cholesky=cholesky,
    )
    return {
        "theta": best.x,
        "cov": covariance,
        "se": np.sqrt(np.diag(covariance)),
        "overdispersion": scale,
        "cost": float(best.cost),
        "success": bool(best.success),
    }


def laplace_covariance(
    model: HybridArrayModel,
    theta: np.ndarray,
    relative: np.ndarray,
    time: np.ndarray,
    sigma: np.ndarray | None,
    prior_sigma: float = PRIOR_SIGMA,
    cholesky: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """数值 Jacobian 的高斯-牛顿 Hessian 的逆。

    标量 sigma 分支按残差方差做过度离散修正；协方差分支已把模型偏差
    与测量噪声计入协方差，不再额外放大。
    """
    step = 1e-4
    jacobian = np.empty((relative.size, 3))
    for k in range(3):
        delta = np.zeros(3)
        delta[k] = step
        plus = model.predict_relative(np.exp(theta + delta)[None, :], time)[0]
        minus = model.predict_relative(np.exp(theta - delta)[None, :], time)[0]
        jacobian[:, k] = ((plus - minus) / (2.0 * step)).ravel()
    if cholesky is None:
        if sigma is None:
            raise ValueError("标量分支需要 sigma")
        weighted = jacobian / sigma[None, :]
    else:
        weighted = _whiten(cholesky, jacobian)
    hessian = weighted.T @ weighted + np.eye(3) / prior_sigma**2
    covariance = np.linalg.inv(hessian)
    if cholesky is None:
        residual = (model.predict_relative(np.exp(theta)[None, :], time)[0] - relative) / sigma[None, :]
        dof = max(residual.size - 3, 1)
        scale = max(float(np.sum(residual**2) / dof), 1.0)
    else:
        scale = 1.0
    return covariance * scale, scale


def profile_likelihood(
    model: HybridArrayModel,
    relative: np.ndarray,
    time: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    theta_true: np.ndarray,
    prior_sigma: float = PRIOR_SIGMA,
    wall: float = 0.15,
) -> list[dict]:
    """标量 MAP 的三气体惩罚剖面区间（ΔSSR=3.84，约 95%）。

    固定一个 log 浓度，优化另外两个；剖面区间很窄且常排除真值时，
    说明模型偏差主导，线性化/剖面区间不能作为可信度声明。
    """
    time = np.asarray(time, dtype=float)
    relative = np.asarray(relative, dtype=float)
    result = map_estimate(model, relative, time, sigma, low, high)
    theta_hat = result["theta"]
    base_cost = float(result["cost"])
    log_low, log_high = np.log(low), np.log(high)
    prior_center = 0.5 * (log_low + log_high)

    def profile_cost(gas_index: int, value: float) -> float:
        others = [index for index in range(3) if index != gas_index]

        def residual(other: np.ndarray) -> np.ndarray:
            theta = np.empty(3)
            theta[gas_index] = value
            theta[others] = other
            return _residual_vector(
                model,
                relative,
                time,
                sigma,
                theta,
                prior_center,
                prior_sigma,
                log_low,
                log_high,
                wall,
                None,
            )

        fit = least_squares(
            residual,
            theta_hat[others],
            x_scale="jac",
            max_nfev=60,
        )
        return float(fit.cost)

    def interval(gas_index: int) -> tuple[float, float, bool]:
        def delta(offset: float) -> float:
            cost = profile_cost(gas_index, float(theta_hat[gas_index] + offset))
            return 2.0 * (cost - base_cost) - PROFILE_DELTA

        def side(direction: float) -> tuple[float, bool]:
            step = 0.005
            previous = 0.0
            while step <= PROFILE_MAX_STEP:
                current = direction * step
                if delta(current) >= 0.0:
                    try:
                        return float(brentq(delta, previous, current)), True
                    except ValueError:
                        return float(current), False
                previous = current
                step *= 2.0
            return float(direction * step), False

        left, left_ok = side(-1.0)
        right, right_ok = side(1.0)
        return left, right, bool(left_ok and right_ok)

    rows = []
    for gas_index, gas in enumerate(("H2", "NH3", "NO2")):
        left, right, converged = interval(gas_index)
        low_interval = float(theta_hat[gas_index] + left)
        high_interval = float(theta_hat[gas_index] + right)
        center = float(np.exp(theta_hat[gas_index]))
        width_rel = float((np.exp(high_interval) - np.exp(low_interval)) / center)
        laplace_half = Z90 * float(result["se"][gas_index])
        laplace_width = float(
            (
                np.exp(theta_hat[gas_index] + laplace_half)
                - np.exp(theta_hat[gas_index] - laplace_half)
            )
            / center
        )
        rows.append(
            {
                "gas": gas,
                "theta_hat": float(theta_hat[gas_index]),
                "theta_true": float(theta_true[gas_index]),
                "se": float(result["se"][gas_index]),
                "profile_low": low_interval,
                "profile_high": high_interval,
                "width_log": float(high_interval - low_interval),
                "width_rel": width_rel,
                "laplace_width_rel": laplace_width,
                "delta_ssr_truth": float(
                    2.0
                    * (profile_cost(gas_index, float(theta_true[gas_index])) - base_cost)
                ),
                "covers_truth": bool(
                    low_interval <= float(theta_true[gas_index]) <= high_interval
                ),
                "converged": converged,
            }
        )
    return rows


def profile_likelihood_table(
    experiments: list[Experiment],
    noise: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对 62 个外层折做剖面似然，汇总覆盖率与区间宽度。"""
    if noise is None:
        noise = estimate_noise(experiments)
    low, high = concentration_bounds(experiments)
    shapes = fit_experiment_shapes(experiments)
    records = []
    for index, test in enumerate(experiments):
        print(f"剖面似然折 {index + 1}/{len(experiments)} (file {test.file_id})", flush=True)
        train_indices = [i for i in range(len(experiments)) if i != index]
        model, sigma_model = _fit_fold(
            [experiments[i] for i in train_indices],
            _subset_shapes(shapes, train_indices),
        )
        sigma_eff = np.sqrt(noise**2 + sigma_model**2)
        mask = test.time <= 10.0
        rows = profile_likelihood(
            model,
            test.relative[mask],
            test.time[mask],
            sigma_eff,
            low,
            high,
            np.log(test.concentration),
        )
        for row in rows:
            records.append({"file": test.file_id, **row})
    frame = pd.DataFrame(records)
    summary = (
        frame.groupby("gas", sort=False)
        .agg(
            n=("file", "size"),
            coverage=("covers_truth", "mean"),
            median_width_rel=("width_rel", "median"),
            median_width_log=("width_log", "median"),
            median_laplace_width_rel=("laplace_width_rel", "median"),
            median_delta_ssr_truth=("delta_ssr_truth", "median"),
            converged=("converged", "sum"),
        )
        .reset_index()
    )
    summary["width_ratio"] = (
        summary["median_width_rel"] / summary["median_laplace_width_rel"]
    )
    return frame, summary


def _subset_shapes(shapes, indices: list[int]):
    return tuple(part[indices] for part in shapes)


def _fit_fold(
    train: list[Experiment],
    shape_init,
) -> tuple[HybridArrayModel, np.ndarray]:
    """拟合训练折的混合模型，并返回各通道残差 RMS（模型误差尺度）。"""
    model = HybridArrayModel().fit(train, shape_init=shape_init)
    relative = np.stack([experiment.relative for experiment in train])
    concentration = np.stack([experiment.concentration for experiment in train])
    residual = relative - model.predict_relative(concentration, train[0].time)
    sigma_model = np.sqrt(np.mean(residual**2, axis=(0, 1)))
    return model, sigma_model


def run_logo_estimation(
    experiments: list[Experiment],
    windows: tuple[float, ...] = WINDOWS,
    noise: np.ndarray | None = None,
) -> pd.DataFrame:
    """62 折留一：每折拟合模型，再在多个时间窗上估计留出实验的浓度。"""
    if noise is None:
        noise = estimate_noise(experiments)
    low, high = concentration_bounds(experiments)
    starts = _start_points(low, high)
    shapes = fit_experiment_shapes(experiments)
    records = []
    for index, test in enumerate(experiments):
        print(f"估计折 {index + 1}/{len(experiments)} (file {test.file_id})", flush=True)
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        shape_init = _subset_shapes(shapes, train_indices)
        model, sigma_model = _fit_fold(train, shape_init)
        sigma_eff = np.sqrt(noise**2 + sigma_model**2)
        theta_true = np.log(test.concentration)
        for tmax in windows:
            mask = test.time <= tmax
            result = map_estimate(
                model,
                test.relative[mask],
                test.time[mask],
                sigma_eff,
                low,
                high,
                starts=starts,
            )
            for j, gas in enumerate(("H2", "NH3", "NO2")):
                error = float(result["theta"][j] - theta_true[j])
                records.append(
                    {
                        "file": test.file_id,
                        "tmax": tmax,
                        "gas": gas,
                        "theta_hat": float(result["theta"][j]),
                        "theta_true": float(theta_true[j]),
                        "se": float(result["se"][j]),
                        "rel_error": float(np.exp(error) - 1.0),
                        "abs_rel_error": float(abs(np.exp(error) - 1.0)),
                        "success": result["success"],
                    }
                )
    return pd.DataFrame(records)


def _nested_residual_matrix(
    experiments: list[Experiment],
    train_indices: list[int],
    shapes,
    folds: int,
) -> np.ndarray:
    """对训练折内部再做 K 折 CV，返回 (N,T,3) 的内层折外残差。

    外层留出文件不参与任何内层训练或预测；内层每折的残差只来自
    未见过该内层验证文件的模型。返回行顺序与 train_indices 一致。
    """
    if folds < 2:
        raise ValueError("内层折数至少为 2")
    positions = {index: position for position, index in enumerate(train_indices)}
    residual = np.empty(
        (len(train_indices), experiments[0].relative.shape[0], 3), dtype=float
    )
    for fold in np.array_split(np.asarray(train_indices), folds):
        held_out = {int(index) for index in fold}
        inner_indices = [index for index in train_indices if index not in held_out]
        model, _ = _fit_fold(
            [experiments[index] for index in inner_indices],
            _subset_shapes(shapes, inner_indices),
        )
        for index in fold:
            index = int(index)
            experiment = experiments[index]
            prediction = model.predict_relative(experiment.concentration[None], experiment.time)[0]
            residual[positions[index]] = prediction - experiment.relative
    return residual


def run_bias_comparison(
    experiments: list[Experiment],
    windows: tuple[float, ...] = (10.0,),
    noise: np.ndarray | None = None,
    shrinkage: float = COVARIANCE_SHRINKAGE,
    nested_folds: int = 0,
) -> pd.DataFrame:
    """在 LOGO 折上比较标量 sigma 与残差协方差两种加权。

    协方差取自 artifacts/analysis/residual_matrix.npz 的折外残差，并在每折
    排除当前留出文件，避免把测试文件的偏差结构带入权重。这是 P4 主结果。
    当 nested_folds>0 时，额外对每个训练折做内层 K 折 CV，用内层折外残差
    估计协方差；外层留出文件完全不参与内层模型，因而协方差是嵌套折外的。
    """
    if noise is None:
        noise = estimate_noise(experiments)
    residual_data = np.load(ANALYSIS / "residual_matrix.npz")
    residual = residual_data["residual"]
    residual_files = residual_data["file_id"]
    residual_index = {int(file_id): index for index, file_id in enumerate(residual_files)}
    low, high = concentration_bounds(experiments)
    starts = _start_points(low, high)
    shapes = fit_experiment_shapes(experiments)
    weightings = ["scalar", "covariance"]
    if nested_folds:
        weightings.append("covariance-nested")
    rows = []
    for index, test in enumerate(experiments):
        print(f"偏差对比折 {index + 1}/{len(experiments)} (file {test.file_id})", flush=True)
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        shape_init = _subset_shapes(shapes, train_indices)
        model, sigma_model = _fit_fold(train, shape_init)
        sigma_eff = np.sqrt(noise**2 + sigma_model**2)
        keep = np.array(
            [int(file_id) != test.file_id for file_id in residual_files],
            dtype=bool,
        )
        full_covariance = residual_covariance(residual[keep], noise=noise, shrinkage=shrinkage)
        nested_covariance = None
        if nested_folds:
            nested_residual = _nested_residual_matrix(experiments, train_indices, shapes, nested_folds)
            nested_covariance = residual_covariance(
                nested_residual, noise=noise, shrinkage=shrinkage
            )
        theta_true = np.log(test.concentration)
        for tmax in windows:
            mask = test.time <= tmax
            time = test.time[mask]
            observed = test.relative[mask]
            covariance = _window_covariance(full_covariance, mask)
            nested_window = None if nested_covariance is None else _window_covariance(nested_covariance, mask)
            for weighting in weightings:
                if weighting == "scalar":
                    weight_covariance = None
                    weight_sigma = sigma_eff
                elif weighting == "covariance":
                    weight_covariance = covariance
                    weight_sigma = None
                else:
                    weight_covariance = nested_window
                    weight_sigma = None
                result = map_estimate(
                    model,
                    observed,
                    time,
                    weight_sigma,
                    low,
                    high,
                    starts=starts,
                    covariance=weight_covariance,
                )
                for gas_index, gas in enumerate(("H2", "NH3", "NO2")):
                    error = float(result["theta"][gas_index] - theta_true[gas_index])
                    rows.append(
                        {
                            "weighting": weighting,
                            "file": test.file_id,
                            "tmax": float(tmax),
                            "gas": gas,
                            "theta_hat": float(result["theta"][gas_index]),
                            "theta_true": float(theta_true[gas_index]),
                            "se": float(result["se"][gas_index]),
                            "abs_rel_error": float(abs(np.exp(error) - 1.0)),
                            "covered": bool(abs(error) <= Z90 * float(result["se"][gas_index])),
                            "overdispersion": float(result["overdispersion"]),
                            "success": bool(result["success"]),
                        }
                    )
    return pd.DataFrame(rows)


def summarize_bias(frame: pd.DataFrame) -> pd.DataFrame:
    """按加权方式/窗口/气体汇总偏差对比。"""
    return (
        frame.groupby(["weighting", "tmax", "gas"], sort=False)
        .agg(
            n=("file", "nunique"),
            median_abs_rel_error=("abs_rel_error", "median"),
            p90_abs_rel_error=("abs_rel_error", lambda values: values.quantile(0.9)),
            coverage_laplace90=("covered", "mean"),
            median_se=("se", "median"),
            median_overdispersion=("overdispersion", "median"),
        )
        .reset_index()
    )


def add_cvplus(frame: pd.DataFrame) -> pd.DataFrame:
    """对每个 tmax/气体，用其余折的 |误差|/se 分数构造 CV+ 区间。"""
    frame = frame.copy()
    frame["score"] = (
        (frame["theta_hat"] - frame["theta_true"]).abs() / frame["se"].clip(lower=1e-6)
    )
    rows = []
    for (tmax, gas), group in frame.groupby(["tmax", "gas"], sort=False):
        scores_all = group["score"].to_numpy()
        for _, row in group.iterrows():
            other = scores_all[group["file"].to_numpy() != row["file"]]
            for alpha in ALPHAS:
                level = min(1.0, np.ceil((len(other) + 1) * (1.0 - alpha)) / len(other))
                q = float(np.quantile(other, level, method="higher"))
                half = q * float(row["se"])
                rows.append(
                    {
                        "file": int(row["file"]),
                        "tmax": float(tmax),
                        "gas": gas,
                        "alpha": alpha,
                        "q": q,
                        "half_log": half,
                        "width_rel": float(np.exp(half) - 1.0),
                        "covered": bool(abs(row["theta_hat"] - row["theta_true"]) <= half),
                        "abs_rel_error": float(row["abs_rel_error"]),
                    }
                )
    return pd.DataFrame(rows)


def _conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """有限样本 conformal 分位数；空数组返回 nan。"""
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    level = min(1.0, np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


def _group_table(experiments: list[Experiment]) -> pd.DataFrame:
    """文件对应的 NH3 档位与 8 条采集块（块号取起始文件号）。"""
    return pd.DataFrame(
        {
            "file": [experiment.file_id for experiment in experiments],
            "NH3": [float(experiment.concentration[1]) for experiment in experiments],
            "block": [(experiment.file_id // 8) * 8 for experiment in experiments],
        }
    )


def group_conformal(
    frame: pd.DataFrame,
    experiments: list[Experiment],
    alpha: float = DECISION_ALPHA,
) -> pd.DataFrame:
    """按 NH3 档位/采集块分别校准 CV+，并与全局分位数并列报告。

    组内分位数只用同组其他文件的分数；组内不足两条时回退到全局分位数。
    这是描述性校准，不参与模型选择。
    """
    frame = frame.copy()
    frame["score"] = (
        (frame["theta_hat"] - frame["theta_true"]).abs() / frame["se"].clip(lower=1e-6)
    )
    frame = frame.merge(_group_table(experiments), on="file", how="left")
    records = []
    for group_type in ("NH3", "block"):
        for (tmax, gas), subset in frame.groupby(["tmax", "gas"], sort=False):
            scores_all = subset["score"].to_numpy()
            for _, row in subset.iterrows():
                file_id = int(row["file"])
                other = subset["score"].to_numpy()[subset["file"].to_numpy() != file_id]
                q_global = _conformal_quantile(other, alpha)
                group_value = row[group_type]
                same_group = subset[
                    (subset[group_type] == group_value) & (subset["file"] != file_id)
                ]["score"].to_numpy()
                q_group = _conformal_quantile(same_group, alpha) if same_group.size >= 2 else q_global
                for calibration, q in (("global", q_global), ("group", q_group)):
                    half = q * float(row["se"])
                    records.append(
                        {
                            "group_type": group_type,
                            "group": group_value,
                            "file": file_id,
                            "tmax": float(tmax),
                            "gas": gas,
                            "calibration": calibration,
                            "alpha": alpha,
                            "q": q,
                            "covered": bool(abs(row["theta_hat"] - row["theta_true"]) <= half),
                            "width_rel": float(np.exp(half) - 1.0),
                        }
                    )
    long = pd.DataFrame(records)
    summary = (
        long.groupby(
            ["group_type", "group", "tmax", "gas", "calibration"],
            sort=False,
        )
        .agg(
            n=("file", "size"),
            coverage=("covered", "mean"),
            median_width_rel=("width_rel", "median"),
            p90_width_rel=("width_rel", lambda values: values.quantile(0.9)),
            q_median=("q", "median"),
        )
        .reset_index()
    )
    overall = (
        long[long["calibration"] == "group"]
        .groupby(["group_type", "tmax", "gas"], sort=False)
        .agg(
            n=("file", "size"),
            coverage=("covered", "mean"),
            median_width_rel=("width_rel", "median"),
            p90_width_rel=("width_rel", lambda values: values.quantile(0.9)),
            q_median=("q", "median"),
        )
        .reset_index()
    )
    overall["calibration"] = "group"
    overall["group"] = "overall"

    baseline = long[long["calibration"] == "global"].drop_duplicates(
        ["file", "tmax", "gas", "calibration"]
    )
    base = (
        baseline.groupby(["tmax", "gas"], sort=False)
        .agg(
            n=("file", "size"),
            coverage=("covered", "mean"),
            median_width_rel=("width_rel", "median"),
            p90_width_rel=("width_rel", lambda values: values.quantile(0.9)),
            q_median=("q", "median"),
        )
        .reset_index()
    )
    base["calibration"] = "global"
    base["group_type"] = "all"
    base["group"] = "overall"
    columns = [
        "group_type",
        "group",
        "tmax",
        "gas",
        "calibration",
        "n",
        "coverage",
        "median_width_rel",
        "p90_width_rel",
        "q_median",
    ]
    return pd.concat(
        [summary, overall[columns], base[columns]], ignore_index=True
    )[columns]


def stopping_table(conformal: pd.DataFrame) -> pd.DataFrame:
    """对每个目标相对半宽，找最早满足全部三气体的窗长。"""
    rows = []
    files = sorted(conformal["file"].unique())
    for file_id in files:
        for target in STOP_TARGETS:
            subset = conformal[(conformal["file"] == file_id) & (conformal["alpha"] == 0.1)]
            chosen = None
            for tmax, group in subset.groupby("tmax", sort=True):
                if len(group) < 3:
                    continue
                if float(group["width_rel"].max()) <= target:
                    chosen = group
                    break
            if chosen is None:
                chosen = subset[subset["tmax"] == subset["tmax"].max()]
            rows.append(
                {
                    "file": int(file_id),
                    "target": target,
                    "stop_time": float(chosen["tmax"].iloc[0]),
                    "reached": bool(float(chosen["width_rel"].max()) <= target),
                    "width_at_stop": float(chosen["width_rel"].max()),
                    "error_at_stop": float(chosen["abs_rel_error"].max()),
                }
            )
    return pd.DataFrame(rows)


def decision_stopping(
    frame: pd.DataFrame,
    conformal: pd.DataFrame,
    experiments: list[Experiment],
    alpha: float = DECISION_ALPHA,
    decision_time: float = 10.0,
) -> pd.DataFrame:
    """以“浓度是否超过中位阈值”为下游决策，找最早可决定的窗口。

    90% CV+ 区间完全落在阈值一侧才算决定；否则继续观测。阈值取 62 条
    实验的逐气体中位数，使两类样本大致平衡，仅用于描述性决策分析。
    """
    thresholds = np.median(
        np.stack([experiment.concentration for experiment in experiments]), axis=0
    )
    truth = {
        experiment.file_id: experiment.concentration for experiment in experiments
    }
    estimates = frame[["file", "tmax", "gas", "theta_hat", "theta_true"]]
    merged = conformal[conformal["alpha"] == alpha].merge(
        estimates, on=["file", "tmax", "gas"], how="inner"
    )
    rows = []
    for (file_id, gas), subset in merged.groupby(["file", "gas"], sort=False):
        gas_index = ("H2", "NH3", "NO2").index(gas)
        threshold = float(thresholds[gas_index])
        true_side = int(truth[int(file_id)][gas_index] > threshold)
        subset = subset.sort_values("tmax")
        chosen = None
        decision_side = -1
        for _, row in subset.iterrows():
            lower = float(np.exp(row["theta_hat"] - row["half_log"]))
            upper = float(np.exp(row["theta_hat"] + row["half_log"]))
            side = 1 if lower > threshold else (0 if upper < threshold else -1)
            if side >= 0:
                chosen = row
                decision_side = side
                break
        if chosen is None:
            chosen = subset.iloc[-1]
            decision_side = -1
        decided = decision_side >= 0
        correct = bool(decided and decision_side == true_side)
        stop_time = float(chosen["tmax"])
        rows.append(
            {
                "file": int(file_id),
                "gas": gas,
                "threshold": threshold,
                "true_side": true_side,
                "stop_time": stop_time,
                "decision_side": decision_side,
                "decided": decided,
                "correct": correct,
                "width_at_stop": float(chosen["width_rel"]),
                "estimate": float(np.exp(chosen["theta_hat"])),
                "lower": float(np.exp(chosen["theta_hat"] - chosen["half_log"])),
                "upper": float(np.exp(chosen["theta_hat"] + chosen["half_log"])),
                "decided_by_target": bool(decided and stop_time <= decision_time),
            }
        )
    return pd.DataFrame(rows)


def summarize_decisions(table: pd.DataFrame, decision_time: float = 10.0) -> pd.DataFrame:
    """按气体汇总决策停止：决定数、误判数、10 s 内决定数与停止时间。"""
    summary = (
        table.groupby("gas", sort=False)
        .agg(
            n=("file", "size"),
            decided=("decided", "sum"),
            correct=("correct", "sum"),
            median_stop=("stop_time", "median"),
            median_width=("width_at_stop", "median"),
            decided_by_target=("decided_by_target", "sum"),
        )
        .reset_index()
    )
    within = table[table["decided_by_target"]].groupby("gas", sort=False)["correct"].sum()
    summary["correct_by_target"] = (
        summary["gas"].map(within).fillna(0).astype(int)
    )
    summary["false"] = summary["decided"] - summary["correct"]
    summary["false_rate"] = np.where(
        summary["decided"] > 0, summary["false"] / summary["decided"], np.nan
    )
    summary["false_by_target"] = summary["decided_by_target"] - summary["correct_by_target"]
    summary["false_rate_by_target"] = np.where(
        summary["decided_by_target"] > 0,
        summary["false_by_target"] / summary["decided_by_target"],
        np.nan,
    )
    summary["target"] = decision_time
    return summary


def loss_stopping(
    frame: pd.DataFrame,
    conformal: pd.DataFrame,
    experiments: list[Experiment],
    alpha: float = DECISION_ALPHA,
    decision_time: float = 10.0,
    ratios: tuple[float, ...] = LOSS_RATIOS,
    tolerances: tuple[float, ...] = LOSS_TOLERANCES,
) -> pd.DataFrame:
    """按下游期望损失决定是否停止，扫描代价比与风险容忍度。

    对 90% CV+ 半宽 half_log 取 log-normal 近似，σ_log=half_log/Z90；
    阈值仍取逐气体中位。C_FP=1，C_FN=ratio。区间只用于近似概率，
    由于 P4 已显示协方差/共形区间偏保守，这是描述性分析。
    """
    thresholds = np.median(
        np.stack([experiment.concentration for experiment in experiments]), axis=0
    )
    truth = {
        experiment.file_id: experiment.concentration for experiment in experiments
    }
    estimates = frame[["file", "tmax", "gas", "theta_hat", "theta_true"]]
    merged = conformal[conformal["alpha"] == alpha].merge(
        estimates, on=["file", "tmax", "gas"], how="inner"
    )
    rows = []
    for (file_id, gas), subset in merged.groupby(["file", "gas"], sort=False):
        gas_index = ("H2", "NH3", "NO2").index(gas)
        threshold = float(thresholds[gas_index])
        true_side = int(truth[int(file_id)][gas_index] > threshold)
        log_threshold = float(np.log(threshold))
        subset = subset.sort_values("tmax")
        for ratio in ratios:
            for tolerance in tolerances:
                chosen = None
                for _, row in subset.iterrows():
                    sigma_log = max(float(row["half_log"]) / Z90, 1e-9)
                    p_above = float(
                        ndtr((float(row["theta_hat"]) - log_threshold) / sigma_log)
                    )
                    loss_above = 1.0 - p_above
                    loss_below = ratio * p_above
                    side = 1 if loss_above <= loss_below else 0
                    risk = min(loss_above, loss_below)
                    if risk <= tolerance:
                        chosen = (row, side, risk, p_above)
                        break
                if chosen is None:
                    row = subset.iloc[-1]
                    sigma_log = max(float(row["half_log"]) / Z90, 1e-9)
                    p_above = float(
                        ndtr((float(row["theta_hat"]) - log_threshold) / sigma_log)
                    )
                    loss_above = 1.0 - p_above
                    loss_below = ratio * p_above
                    side = 1 if loss_above <= loss_below else 0
                    chosen = (row, side, min(loss_above, loss_below), p_above)
                    decided = False
                else:
                    decided = True
                row, side, risk, p_above = chosen
                realized = 0.0
                if side == 1 and true_side == 0:
                    realized = 1.0
                elif side == 0 and true_side == 1:
                    realized = float(ratio)
                rows.append(
                    {
                        "file": int(file_id),
                        "gas": gas,
                        "threshold": threshold,
                        "cost_ratio": float(ratio),
                        "tolerance": float(tolerance),
                        "true_side": true_side,
                        "stop_time": float(row["tmax"]),
                        "decision_side": side,
                        "risk": float(risk),
                        "p_above": p_above,
                        "decided": bool(decided),
                        "correct": bool(decided and side == true_side),
                        "false_positive": bool(decided and side == 1 and true_side == 0),
                        "false_negative": bool(decided and side == 0 and true_side == 1),
                        "realized_loss": realized,
                        "width_at_stop": float(row["width_rel"]),
                        "decided_by_target": bool(decided and float(row["tmax"]) <= decision_time),
                    }
                )
    return pd.DataFrame(rows)


def summarize_losses(table: pd.DataFrame, decision_time: float = 10.0) -> pd.DataFrame:
    """按气体/代价比/容忍度汇总停止结果。"""
    summary = (
        table.groupby(["gas", "cost_ratio", "tolerance"], sort=False)
        .agg(
            n=("file", "size"),
            decided=("decided", "sum"),
            correct=("correct", "sum"),
            false_positive=("false_positive", "sum"),
            false_negative=("false_negative", "sum"),
            undecided=("decided", lambda values: int((~values).sum())),
            median_stop=("stop_time", "median"),
            median_risk=("risk", "median"),
            decided_by_target=("decided_by_target", "sum"),
        )
        .reset_index()
    )
    within = table[table["decided_by_target"]].groupby(
        ["gas", "cost_ratio", "tolerance"], sort=False
    )["correct"].sum()
    index = pd.MultiIndex.from_frame(summary[["gas", "cost_ratio", "tolerance"]])
    summary["correct_by_target"] = within.reindex(index).fillna(0).to_numpy(dtype=int)
    summary["false"] = summary["decided"] - summary["correct"]
    summary["false_rate"] = np.where(
        summary["decided"] > 0, summary["false"] / summary["decided"], np.nan
    )
    summary["false_by_target"] = summary["decided_by_target"] - summary["correct_by_target"]
    summary["false_rate_by_target"] = np.where(
        summary["decided_by_target"] > 0,
        summary["false_by_target"] / summary["decided_by_target"],
        np.nan,
    )
    losses = table[table["decided"]].groupby(
        ["gas", "cost_ratio", "tolerance"], sort=False
    )["realized_loss"].sum()
    summary["loss_decided"] = losses.reindex(index).fillna(0.0).to_numpy()
    summary["target"] = decision_time
    return summary


def error_budget(
    experiments: list[Experiment],
    windows: tuple[float, ...] = (5.0, 10.0, 20.0, 60.0),
    noise: np.ndarray | None = None,
    repeats: int = 2,
    frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """把真实误差与“同模型合成数据”“灰盒合成数据”的误差放在一起。

    - hybrid：用混合模型生成无模型偏差的曲线，误差只剩噪声与估计器；
    - grey：用一阶灰盒生成曲线、用混合模型估计，误差包含灰盒-混合的模型偏差；
    - real：真实曲线，误差包含全部因素。
    """
    if noise is None:
        noise = estimate_noise(experiments)
    low, high = concentration_bounds(experiments)
    starts = _start_points(low, high)
    shapes = fit_experiment_shapes(experiments)
    rng = np.random.default_rng(0)
    real_errors = {}
    if frame is not None:
        for _, row in frame[frame["tmax"].isin(windows)].iterrows():
            real_errors[(int(row["file"]), float(row["tmax"]), row["gas"])] = float(row["abs_rel_error"])
    rows = []
    for index, test in enumerate(experiments):
        print(f"误差预算折 {index + 1}/{len(experiments)} (file {test.file_id})", flush=True)
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        shape_init = _subset_shapes(shapes, train_indices)
        model, sigma_model = _fit_fold(train, shape_init)
        sigma_eff = np.sqrt(noise**2 + sigma_model**2)
        theta_true = np.log(test.concentration)
        for tmax in windows:
            mask = test.time <= tmax
            time = test.time[mask]
            observed = test.relative[mask]
            sources = {
                "real": observed,
                "hybrid": model.predict_relative(test.concentration[None], time)[0],
                "grey": model.grey.predict_relative(test.concentration[None], time)[0],
            }
            for source, curve in sources.items():
                if source == "real" and real_errors:
                    for j, gas in enumerate(("H2", "NH3", "NO2")):
                        rows.append(
                            {
                                "file": test.file_id,
                                "tmax": float(tmax),
                                "source": source,
                                "gas": gas,
                                "abs_rel_error": real_errors[(test.file_id, float(tmax), gas)],
                            }
                        )
                    continue
                for repeat in range(repeats if source != "real" else 1):
                    sample = curve + rng.normal(scale=noise[None, :], size=curve.shape) if source != "real" else curve
                    result = map_estimate(model, sample, time, sigma_eff, low, high, starts=starts)
                    for j, gas in enumerate(("H2", "NH3", "NO2")):
                        rows.append(
                            {
                                "file": test.file_id,
                                "tmax": float(tmax),
                                "source": source,
                                "gas": gas,
                                "abs_rel_error": float(abs(np.exp(result["theta"][j] - theta_true[j]) - 1.0)),
                            }
                        )
    return pd.DataFrame(rows)


def run_p4() -> None:
    """P4：偏差协方差 MAP、分组共形、几何/损失停止与剖面似然。"""
    experiments = load_experiments()
    noise = estimate_noise(experiments)
    print("相对噪声（H2, NH3, NO2）:", np.array2string(noise, precision=2))
    bias = run_bias_comparison(experiments, windows=(10.0,), noise=noise)
    bias_summary = summarize_bias(bias)

    estimates = pd.read_csv(ANALYSIS / "estimates.csv")
    conformal = pd.read_csv(ANALYSIS / "conformal.csv")
    groups = group_conformal(estimates, experiments)
    decisions = decision_stopping(estimates, conformal, experiments)
    decision_summary = summarize_decisions(decisions)
    losses = loss_stopping(estimates, conformal, experiments)
    loss_summary = summarize_losses(losses)
    profiles, profile_summary = profile_likelihood_table(experiments, noise=noise)

    ensure()
    bias.to_csv(ANALYSIS / "bias_estimates.csv", index=False, encoding="utf-8")
    bias_summary.to_csv(ANALYSIS / "bias_summary.csv", index=False, encoding="utf-8")
    groups.to_csv(ANALYSIS / "coverage_by_group.csv", index=False, encoding="utf-8")
    decisions.to_csv(ANALYSIS / "stopping_decision.csv", index=False, encoding="utf-8")
    decision_summary.to_csv(
        ANALYSIS / "stopping_decision_summary.csv", index=False, encoding="utf-8"
    )
    losses.to_csv(ANALYSIS / "stopping_loss.csv", index=False, encoding="utf-8")
    loss_summary.to_csv(
        ANALYSIS / "stopping_loss_summary.csv", index=False, encoding="utf-8"
    )
    profiles.to_csv(ANALYSIS / "profile_likelihood.csv", index=False, encoding="utf-8")
    profile_summary.to_csv(
        ANALYSIS / "profile_likelihood_summary.csv", index=False, encoding="utf-8"
    )

    print("\nP4 标量 vs 协方差（10 s 中位绝对相对误差 / %）:")
    print(
        (bias_summary.pivot(index="weighting", columns="gas", values="median_abs_rel_error") * 100)
        .round(2)
        .to_string()
    )
    print("\nP4 标量 vs 协方差（Laplace 90% 覆盖率）:")
    print(
        (bias_summary.pivot(index="weighting", columns="gas", values="coverage_laplace90") * 100)
        .round(1)
        .to_string()
    )
    print("\nP4 分组覆盖（10 s，按组内文件加权）:")
    ten = groups[groups["tmax"] == 10.0]
    rows = []
    for group_type in ("NH3", "block"):
        part = ten[
            (ten["group_type"] == group_type)
            & (ten["calibration"] == "group")
            & (ten["group"] != "overall")
        ]
        for gas, table in part.groupby("gas"):
            rows.append(
                {
                    "group_type": group_type,
                    "gas": gas,
                    "coverage": float(np.average(table["coverage"], weights=table["n"])),
                }
            )
    baseline = ten[(ten["group_type"] == "all") & (ten["calibration"] == "global")]
    for _, row in baseline.iterrows():
        rows.append({"group_type": "global", "gas": row["gas"], "coverage": row["coverage"]})
    print(
        (pd.DataFrame(rows).pivot(index="group_type", columns="gas", values="coverage") * 100)
        .round(1)
        .to_string()
    )
    print("\nP4 决策停止（10 s 目标）:")
    print(decision_summary.to_string(index=False))
    print("\nP4 损失停止（C_FN/C_FP=3，容忍度 0.10）:")
    selected = loss_summary[
        (loss_summary["cost_ratio"] == 3.0) & (loss_summary["tolerance"] == 0.10)
    ]
    print(selected.to_string(index=False))
    print("\nP4 剖面似然（ΔSSR=3.84）:")
    print(profile_summary.to_string(index=False))


def run_nested_bias_validation(folds: int = 5) -> None:
    """P4 补充：用内层 K 折残差估计协方差，检查非嵌套近似的影响。"""
    experiments = load_experiments()
    noise = estimate_noise(experiments)
    frame = run_bias_comparison(
        experiments, windows=(10.0,), noise=noise, nested_folds=folds
    )
    summary = summarize_bias(frame)
    ensure()
    frame.to_csv(ANALYSIS / "bias_nested_estimates.csv", index=False, encoding="utf-8")
    summary.to_csv(ANALYSIS / "bias_nested_summary.csv", index=False, encoding="utf-8")

    print("\n嵌套协方差验证（10 s 中位绝对相对误差 / %）:")
    print(
        (summary.pivot(index="weighting", columns="gas", values="median_abs_rel_error") * 100)
        .round(2)
        .to_string()
    )
    print("\n嵌套协方差验证（Laplace 90% 覆盖率 / %）:")
    print(
        (summary.pivot(index="weighting", columns="gas", values="coverage_laplace90") * 100)
        .round(1)
        .to_string()
    )
    print("\n嵌套协方差验证（中位 se）:")
    print(summary.pivot(index="weighting", columns="gas", values="median_se").round(4).to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="MAP 估计、共形校准与自适应停止")
    parser.add_argument(
        "--part",
        choices=("estimator", "p4", "p4-nested"),
        default="estimator",
        help="estimator：62 折 MAP/Laplace/CV+/停止/误差预算；p4：协方差 MAP/分组共形/决策与损失停止/剖面似然；p4-nested：嵌套协方差验证",
    )
    arguments = parser.parse_args()

    experiments = load_experiments()
    noise = estimate_noise(experiments)
    if arguments.part == "p4":
        run_p4()
        return
    if arguments.part == "p4-nested":
        run_nested_bias_validation()
        return

    print("相对噪声（H2, NH3, NO2）:", np.array2string(noise, precision=2))
    frame = run_logo_estimation(experiments, noise=noise)
    conformal = add_cvplus(frame)
    stopping = stopping_table(conformal)
    budget = error_budget(experiments, noise=noise, frame=frame)

    ensure()
    frame.to_csv(ANALYSIS / "estimates.csv", index=False, encoding="utf-8")
    conformal.to_csv(ANALYSIS / "conformal.csv", index=False, encoding="utf-8")
    stopping.to_csv(ANALYSIS / "stopping.csv", index=False, encoding="utf-8")
    budget.to_csv(ANALYSIS / "error_budget.csv", index=False, encoding="utf-8")

    point = frame.groupby(["tmax", "gas"])["abs_rel_error"].median().unstack()
    print("\n中位绝对相对误差 (%):")
    print((point * 100).round(2).to_string())
    coverage = conformal[conformal["alpha"] == 0.1].groupby(["tmax", "gas"])["covered"].mean().unstack()
    print("\nCV+ 90% 覆盖率:")
    print((coverage * 100).round(1).to_string())
    print("\n自适应停止（目标 10%）:")
    summary = stopping[stopping["target"] == 0.10].groupby("reached").agg(
        n=("file", "size"), median_stop=("stop_time", "median"), median_error=("error_at_stop", "median")
    )
    print(summary.round(3).to_string())
    print("\n误差预算（中位绝对相对误差, %）:")
    print((budget.groupby(["tmax", "source", "gas"])["abs_rel_error"].median().unstack() * 100).round(2).to_string())


if __name__ == "__main__":
    main()
