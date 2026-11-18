"""混合模型折外残差诊断：增益/形状分解、PCA、设计变量相关与分块漂移。

只读取 LOGO 的折外预测（artifacts/logo/logo_predictions.npz），不重新拟合模型。
归一化使用与 LOGO 相同的训练折尺度：对第 i 条实验，scale 取其余 61 条
relative 响应的 RMS。PCA 输入是未归一化的残差 (62, 299)，因此方差占比反映
绝对误差结构；逐文件指标则除以各自通道的训练折尺度。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core.data import GASES, Experiment, load_experiments
from ..core.grey_box import fit_experiment_shapes
from ..paths import ANALYSIS, LOGO, ensure

EARLY = 5.0
LATE = 20.0
MODEL = "hybrid"


def _fold_scale(experiments: list[Experiment]) -> np.ndarray:
    """每条实验的训练折尺度（留出该条后其余实验的 RMS），形状 (N,3)。"""
    relative = np.stack([experiment.relative for experiment in experiments])
    total = np.sum(relative**2, axis=(0, 1))
    count = (len(experiments) - 1) * relative.shape[1]
    return np.sqrt(
        (total[None, :] - np.sum(relative**2, axis=1)) / count
    )


def load_predictions(name: str = MODEL) -> dict[int, np.ndarray]:
    """读取折外预测，键为文件序号。"""
    with np.load(LOGO / "logo_predictions.npz") as archive:
        return {
            int(key.split("_")[-1]): archive[key]
            for key in archive.files
            if key.startswith(f"{name}_")
        }


def _knn_distance(concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对数浓度空间里到最近 1/3 个训练点的距离（留一）。"""
    u = np.log(concentration)
    distance = np.sqrt(((u[:, None, :] - u[None, :, :]) ** 2).sum(axis=-1))
    np.fill_diagonal(distance, np.inf)
    ordered = np.sort(distance, axis=1)
    return ordered[:, 0], ordered[:, :3].mean(axis=1)


def _decompose(residual: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把残差投影到预测曲线上：增益分量 + 正交形状分量。"""
    alpha = float(np.dot(residual, prediction) / max(np.dot(prediction, prediction), 1e-18))
    gain = alpha * prediction
    return gain, residual - gain


def _pca(residual: np.ndarray, time: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按通道做 SVD，返回前三个时间模式与得分。"""
    mode_rows, score_rows = [], []
    for index, gas in enumerate(GASES):
        matrix = residual[:, :, index]
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        variance = singular**2 / np.sum(singular**2)
        for component in range(3):
            score = left[:, component] * singular[component]
            for file_index in range(matrix.shape[0]):
                score_rows.append(
                    {
                        "file": file_index,
                        "gas": gas,
                        "component": component + 1,
                        "score": float(score[file_index]),
                    }
                )
            for t, value in zip(time, right[component]):
                mode_rows.append(
                    {
                        "gas": gas,
                        "component": component + 1,
                        "variance_ratio": float(variance[component]),
                        "time": float(t),
                        "shape": float(value),
                    }
                )
    return pd.DataFrame(mode_rows), pd.DataFrame(score_rows)


def build_table(experiments: list[Experiment], name: str = MODEL) -> pd.DataFrame:
    """逐实验 × 通道的残差指标表。"""
    predictions = load_predictions(name)
    truth = np.stack([experiment.relative for experiment in experiments])
    prediction = np.stack([predictions[experiment.file_id] for experiment in experiments])
    residual = prediction - truth
    time = experiments[0].time
    scale = _fold_scale(experiments)
    amplitude, tau, delay = fit_experiment_shapes(experiments)
    concentration = np.stack([experiment.concentration for experiment in experiments])
    knn1, knn3 = _knn_distance(concentration)
    _, scores = _pca(residual, time)
    scores = scores.pivot_table(index=["file", "gas"], columns="component", values="score")
    scores.columns = [f"score_pc{int(column)}" for column in scores.columns]
    scores = scores.reset_index()

    early = time <= EARLY
    late = time >= LATE
    rows = []
    for index, experiment in enumerate(experiments):
        for gas_index, gas in enumerate(GASES):
            error = residual[index, :, gas_index]
            predicted = prediction[index, :, gas_index]
            observed = truth[index, :, gas_index]
            gain, shape = _decompose(error, predicted)
            energy = float(np.sum(error**2))
            steps = np.abs(np.diff(observed))
            step_index = int(np.argmax(steps))
            rows.append(
                {
                    "file": experiment.file_id,
                    "gas": gas,
                    "h2": experiment.concentration[0],
                    "nh3": experiment.concentration[1],
                    "no2": experiment.concentration[2],
                    "block": f"NH3={experiment.concentration[1]:g}",
                    "r0": float(experiment.initial[gas_index]),
                    "nrmse": float(np.sqrt(np.mean(error**2)) / scale[index, gas_index]),
                    "gain_share": float(np.sum(gain**2) / max(energy, 1e-18)),
                    "shape_share": float(np.sum(shape**2) / max(energy, 1e-18)),
                    "gain_rms": float(np.sqrt(np.mean(gain**2)) / scale[index, gas_index]),
                    "shape_rms": float(np.sqrt(np.mean(shape**2)) / scale[index, gas_index]),
                    "early_rms": float(np.sqrt(np.mean(error[early] ** 2)) / scale[index, gas_index]),
                    "late_rms": float(np.sqrt(np.mean(error[late] ** 2)) / scale[index, gas_index]),
                    "end_err": float((predicted[-1] - observed[-1]) / max(abs(observed[-1]), 1e-9)),
                    "slope": float(np.polyfit(time[-51:], observed[-51:], 1)[0]),
                    "knn1": float(knn1[index]),
                    "knn3": float(knn3[index]),
                    "amplitude": float(amplitude[index, gas_index]),
                    "tau": float(tau[index, gas_index]),
                    "delay": float(delay[index, gas_index]),
                    "max_step": float(steps[step_index]),
                    "step_time": float(time[step_index + 1]),
                    "energy": energy,
                }
            )
    frame = pd.DataFrame(rows)
    frame["energy_share"] = frame["energy"] / frame.groupby("gas")["energy"].transform("sum")
    return frame.merge(scores, on=["file", "gas"], how="left")


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    """残差特征与设计变量的 Spearman 相关（逐通道）。"""
    features = ["nrmse", "gain_rms", "shape_rms", "score_pc1", "score_pc2", "score_pc3"]
    variables = [
        "file",
        "h2",
        "nh3",
        "no2",
        "r0",
        "slope",
        "knn1",
        "amplitude",
        "tau",
        "delay",
    ]
    rows = []
    for gas, group in frame.groupby("gas", sort=False):
        for feature in features:
            for variable in variables:
                values = group[[feature, variable]].dropna()
                if values[variable].nunique() < 3:
                    continue
                rho, p = spearmanr(values[variable], values[feature])
                rows.append(
                    {
                        "gas": gas,
                        "feature": feature,
                        "variable": variable,
                        "rho": float(rho),
                        "p": float(p),
                        "n": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def block_table(frame: pd.DataFrame) -> pd.DataFrame:
    """NH3 采集块内的误差水平与块内随文件序号的趋势。"""
    rows = []
    for (gas, block), group in frame.groupby(["gas", "block"], sort=False):
        rho, p = spearmanr(group["file"], group["nrmse"]) if group["file"].nunique() > 2 else (np.nan, np.nan)
        rows.append(
            {
                "gas": gas,
                "block": block,
                "n": int(len(group)),
                "file_first": int(group["file"].min()),
                "file_last": int(group["file"].max()),
                "nrmse_median": float(group["nrmse"].median()),
                "shape_rms_median": float(group["shape_rms"].median()),
                "rho_within_block": float(rho),
                "p_within_block": float(p),
            }
        )
    return pd.DataFrame(rows)


def region_table(frame: pd.DataFrame) -> pd.DataFrame:
    """设计边角区域的残差能量份额与误差水平（区域可重叠）。"""
    regions = {
        "H2=32000": frame["h2"] == 32000,
        "H2=24000": frame["h2"] == 24000,
        "NH3=10": frame["nh3"] == 10,
        "NH3=50": frame["nh3"] == 50,
        "NO2=2": frame["no2"] == 2,
        "NO2=10": frame["no2"] == 10,
    }
    rows = []
    for label, mask in regions.items():
        for gas, group in frame.groupby("gas", sort=False):
            subset = group[mask.loc[group.index]]
            rows.append(
                {
                    "region": label,
                    "gas": gas,
                    "n": int(len(subset)),
                    "energy_share": float(subset["energy_share"].sum()),
                    "nrmse_median": float(subset["nrmse"].median()),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    ensure()
    experiments = load_experiments()
    frame = build_table(experiments)
    truth = np.stack([experiment.relative for experiment in experiments])
    predictions = load_predictions()
    prediction = np.stack([predictions[experiment.file_id] for experiment in experiments])
    scale = _fold_scale(experiments)
    np.savez(
        ANALYSIS / "residual_matrix.npz",
        time=experiments[0].time,
        residual=prediction - truth,
        scale=scale,
        file_id=np.array([experiment.file_id for experiment in experiments]),
    )
    modes, scores = _pca(prediction - truth, experiments[0].time)
    correlations = correlation_table(frame)
    blocks = block_table(frame)
    regions = region_table(frame)

    frame.to_csv(ANALYSIS / "residual_diagnostics.csv", index=False, encoding="utf-8")
    correlations.to_csv(ANALYSIS / "residual_correlations.csv", index=False, encoding="utf-8")
    modes.to_csv(ANALYSIS / "residual_pca_modes.csv", index=False, encoding="utf-8")
    scores.to_csv(ANALYSIS / "residual_pca_scores.csv", index=False, encoding="utf-8")
    blocks.to_csv(ANALYSIS / "residual_blocks.csv", index=False, encoding="utf-8")
    regions.to_csv(ANALYSIS / "residual_regions.csv", index=False, encoding="utf-8")

    print("逐通道中位指标")
    print(
        frame.groupby("gas")[["nrmse", "gain_share", "shape_share"]]
        .median()
        .round(3)
        .to_string()
    )
    print("\nPCA 方差占比")
    print(
        modes.groupby(["gas", "component"])["variance_ratio"]
        .first()
        .unstack()
        .round(3)
        .to_string()
    )
    print("\n相关显著的项（p < 0.01）")
    print(
        correlations[correlations["p"] < 0.01]
        .sort_values(["gas", "feature", "p"])
        .round(3)
        .to_string(index=False)
    )
    print("\n区域能量份额")
    print(regions.pivot(index="region", columns="gas", values="energy_share").round(3).to_string())


if __name__ == "__main__":
    main()
