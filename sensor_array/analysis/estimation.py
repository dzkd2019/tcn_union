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

from itertools import product

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from ..core.data import Experiment, load_experiments
from ..core.hybrid import HybridArrayModel
from .identifiability import estimate_noise
from ..core.grey_box import fit_experiment_shapes
from ..paths import ANALYSIS, ensure

WINDOWS = (2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0)
ALPHAS = (0.1, 0.2)
STOP_TARGETS = (0.05, 0.10, 0.20, 0.30)
PRIOR_SIGMA = 1.0


def concentration_bounds(experiments: list[Experiment]) -> tuple[np.ndarray, np.ndarray]:
    concentration = np.stack([experiment.concentration for experiment in experiments])
    return concentration.min(axis=0), concentration.max(axis=0)


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
) -> list[np.ndarray]:
    """先用一次前向残差给多起点排序，只对最好的两个起点做优化。"""
    costs = []
    for start in starts:
        prediction = model.predict_relative(np.exp(start)[None, :], time)[0]
        residual = (prediction - relative) / sigma[None, :]
        costs.append(
            float(np.sum(residual**2) + np.sum(((start - prior_center) / prior_sigma) ** 2))
        )
    order = np.argsort(costs)[:keep]
    return [starts[index] for index in order]


def map_estimate(
    model: HybridArrayModel,
    relative: np.ndarray,
    time: np.ndarray,
    sigma: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    starts: list[np.ndarray] | None = None,
    prior_sigma: float = PRIOR_SIGMA,
    wall: float = 0.15,
) -> dict:
    """对一条相对响应曲线做 MAP 估计，返回 log 浓度与 Laplace 协方差。

    设计域用软墙约束：越界残差按 ``(theta - bound) / wall`` 增长，
    避免硬边界把估计值钉在边界上、制造虚假的零误差。
    """
    time = np.asarray(time, dtype=float)
    relative = np.asarray(relative, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if starts is None:
        starts = _start_points(low, high)
    log_low, log_high = np.log(low), np.log(high)
    prior_center = 0.5 * (log_low + log_high)
    starts = _screen_starts(model, relative, time, sigma, starts, prior_center, prior_sigma)

    def residual(theta: np.ndarray) -> np.ndarray:
        prediction = model.predict_relative(np.exp(theta)[None, :], time)[0]
        return np.r_[
            ((prediction - relative) / sigma[None, :]).ravel(),
            (theta - prior_center) / prior_sigma,
            np.maximum(log_low - theta, 0.0) / wall,
            np.maximum(theta - log_high, 0.0) / wall,
        ]

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
        model, best.x, relative, time, sigma, prior_sigma
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
    sigma: np.ndarray,
    prior_sigma: float = PRIOR_SIGMA,
) -> tuple[np.ndarray, float]:
    """数值 Jacobian 的高斯-牛顿 Hessian 的逆，按残差方差做过度离散修正。"""
    step = 1e-4
    jacobian = np.empty((relative.size, 3))
    for k in range(3):
        delta = np.zeros(3)
        delta[k] = step
        plus = model.predict_relative(np.exp(theta + delta)[None, :], time)[0]
        minus = model.predict_relative(np.exp(theta - delta)[None, :], time)[0]
        jacobian[:, k] = ((plus - minus) / (2.0 * step)).ravel()
    weighted = jacobian / sigma[None, :]
    hessian = weighted.T @ weighted + np.eye(3) / prior_sigma**2
    covariance = np.linalg.inv(hessian)
    residual = (model.predict_relative(np.exp(theta)[None, :], time)[0] - relative) / sigma[None, :]
    dof = max(residual.size - 3, 1)
    scale = max(float(np.sum(residual**2) / dof), 1.0)
    return covariance * scale, scale


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


def main() -> None:
    experiments = load_experiments()
    noise = estimate_noise(experiments)
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
