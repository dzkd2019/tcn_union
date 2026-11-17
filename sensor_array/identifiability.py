"""浓度可辨识性分析：Fisher 信息、CRLB 与时间窗信息量。

对每条实验、每个时间窗，把绝对输出 R(t)=R(0)[1+y(t)] 对 log 浓度（以及可选的
log R(0)）求数值 Jacobian，按测量噪声归一化后得到 Fisher 信息矩阵：

    F = J^T Σ^{-1} J

CRLB 为 F^{-1} 对角线的平方根，近似等于相对浓度的标准差。条件数、浓度-基线
相关系数和最弱特征向量用来判断哪些方向不可辨识。

用法：
    .venv/bin/python -m sensor_array.identifiability
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from .data import Experiment, load_experiments
from .model import FirstOrderArrayModel

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
WINDOWS = (2.0, 5.0, 10.0, 20.0, 60.0)
# H2 通道在数据中几乎无逐点噪声（二阶差分 MAD 接近 0），直接用会使 FIM 奇异。
# 取 1e-5 的相对噪声下限，相当于把“该通道被平滑过”当作未知噪声处理。
NOISE_FLOOR = 1e-5


def estimate_noise(experiments: list[Experiment]) -> np.ndarray:
    """用二阶差分 MAD 估计三路相对噪声，并施加下限。

    平滑曲线的二阶差分主要来自测量噪声；高斯白噪声的二阶差分方差为 6σ²，
    MAD 到标准差的换算系数为 0.6745。该方法与
    `tcn_union/augmentation/noise.py::estimate_noise_sigma` 一致。
    """
    estimates = []
    for experiment in experiments:
        second = np.diff(experiment.response, n=2, axis=0)
        spread = np.median(np.abs(second - np.median(second, axis=0)), axis=0)
        estimates.append(spread / 0.6745 / np.sqrt(6.0) / experiment.initial)
    return np.maximum(np.median(np.stack(estimates), axis=0), NOISE_FLOOR)


def _jacobian(
    model: FirstOrderArrayModel,
    experiment: Experiment,
    tmax: float,
    noise_rel: np.ndarray,
    with_baseline: bool,
) -> np.ndarray:
    mask = experiment.time <= tmax
    time = experiment.time[mask]
    concentration = experiment.concentration
    initial = experiment.initial
    sigma = noise_rel * initial
    n_parameter = 6 if with_baseline else 3
    base = np.r_[np.log(concentration), np.log(initial)][:n_parameter]

    def response(theta: np.ndarray) -> np.ndarray:
        c = np.exp(theta[:3])
        r0 = np.exp(theta[3:]) if with_baseline else initial
        relative = model.predict_relative(c[None, :], time)[0]
        return r0[None, :] * (1.0 + relative)

    jacobian = np.empty((int(mask.sum()) * 3, n_parameter))
    for k in range(n_parameter):
        step = np.zeros(n_parameter)
        step[k] = 1e-4
        jacobian[:, k] = ((response(base + step) - response(base - step)) / 2e-4).ravel()
    return jacobian / np.tile(sigma, int(mask.sum()))[:, None]


def fisher_information(
    model: FirstOrderArrayModel,
    experiment: Experiment,
    tmax: float,
    noise_rel: np.ndarray,
    with_baseline: bool = False,
) -> dict:
    """返回单条实验、单个时间窗的 FIM 及其派生量。"""
    jacobian = _jacobian(model, experiment, tmax, noise_rel, with_baseline)
    fim = jacobian.T @ jacobian
    eigenvalues = np.linalg.eigvalsh(fim)
    covariance = np.linalg.pinv(fim)
    crlb = np.sqrt(np.diag(covariance)) * 100.0
    max_corr = np.nan
    if with_baseline:
        scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
        corr = covariance / np.maximum(scale, 1e-12)
        max_corr = float(np.max(np.abs(corr[:3, 3:])))
    return {
        "cond": float(eigenvalues[-1] / max(eigenvalues[0], 1e-12)),
        "crlb": crlb[:3],
        "max_corr_c_r0": max_corr,
        "eigenvalues": eigenvalues,
    }


def analyze(
    experiments: list[Experiment],
    model: FirstOrderArrayModel,
    windows: tuple[float, ...] = WINDOWS,
) -> pd.DataFrame:
    """对全部实验与时间窗计算 CRLB，返回长表。"""
    noise_rel = estimate_noise(experiments)
    rows = []
    for experiment in experiments:
        for tmax in windows:
            for with_baseline, mode in ((False, "c"), (True, "c+R0")):
                result = fisher_information(model, experiment, tmax, noise_rel, with_baseline)
                rows.append(
                    {
                        "file_id": experiment.file_id,
                        "h2": experiment.concentration[0],
                        "nh3": experiment.concentration[1],
                        "no2": experiment.concentration[2],
                        "tmax": tmax,
                        "mode": mode,
                        "cond": result["cond"],
                        "crlb_h2": result["crlb"][0],
                        "crlb_nh3": result["crlb"][1],
                        "crlb_no2": result["crlb"][2],
                        "max_corr_c_r0": result["max_corr_c_r0"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    experiments = load_experiments()
    model = FirstOrderArrayModel.load(ARTIFACTS / "model.json")
    noise_rel = estimate_noise(experiments)
    print("相对噪声（H2, NH3, NO2）:", np.array2string(noise_rel, precision=2))
    frame = analyze(experiments, model)
    ARTIFACTS.mkdir(exist_ok=True)
    frame.to_csv(ARTIFACTS / "identifiability.csv", index=False, encoding="utf-8")

    summary = (
        frame.groupby(["mode", "tmax"])[["crlb_h2", "crlb_nh3", "crlb_no2", "cond"]]
        .median()
        .reset_index()
    )
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\n写入 {ARTIFACTS / 'identifiability.csv'}（{len(frame)} 行）")


if __name__ == "__main__":
    main()
