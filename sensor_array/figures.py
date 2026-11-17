"""生成报告用图（PNG + SVG）。

运行：``.venv/bin/python -m sensor_array.figures``
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import GASES, RESPONSE_SIGNS, Experiment, load_experiments
from .model import FirstOrderArrayModel, fit_experiment_shapes

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FIGURES = Path(__file__).resolve().parent / "figures"

GAS_LABEL = {"H2": "H$_2$", "NH3": "NH$_3$", "NO2": "NO$_2$"}
GAS_COLORS = {"H2": "#C0392B", "NH3": "#E69F00", "NO2": "#0072B2"}
MODEL_COLORS = {
    "mean_curve": "#8C8C8C",
    "direct_ridge": "#E69F00",
    "two_stage": "#CC79A7",
    "tcn": "#6A4C93",
    "grey_box": "#009E73",
    "hybrid": "#4C72B0",
}
MODEL_LABELS = {
    "mean_curve": "均值曲线",
    "direct_ridge": "直接岭回归",
    "two_stage": "两阶段回归",
    "tcn": "TCN",
    "grey_box": "灰盒",
    "hybrid": "混合（本文）",
}
MODEL_ORDER = ["mean_curve", "direct_ridge", "two_stage", "tcn", "grey_box", "hybrid"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "figure.dpi": 120,
        }
    )


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=300)
    figure.savefig(FIGURES / f"{name}.svg")
    plt.close(figure)
    print("saved", FIGURES / f"{name}.png")


def _letter(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9.5)


def figure_data_overview(experiments: list[Experiment]) -> None:
    time = experiments[0].time
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 5.6))
    files = np.array([experiment.file_id for experiment in experiments])
    initial = np.stack([experiment.initial for experiment in experiments])
    relative = np.stack([experiment.relative for experiment in experiments])
    ax = axes[0, 0]
    for index, gas in enumerate(GASES):
        ax.semilogy(files, initial[:, index] / initial[0, index], color=GAS_COLORS[gas], lw=1.4, label=GAS_LABEL[gas])
    ax.set_xlabel("实验序号")
    ax.set_ylabel("初始响应 / 首个实验值")
    ax.legend(frameon=False, ncol=3)
    _letter(ax, "(a) 初始基线随实验序号漂移")
    for ax, gas, index in zip(axes.flat[1:], GASES, range(3)):
        curves = relative[:, :, index] * 100.0
        ax.plot(time, curves.T, color="#B0B0B0", lw=0.45, alpha=0.55)
        ax.plot(time, np.median(curves, axis=0), color=GAS_COLORS[gas], lw=1.8, label="中位数")
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("相对响应 / %")
        ax.legend(frameon=False, loc="best")
        _letter(ax, f"({'bcd'[index]}) {GAS_LABEL[gas]}（62 条曲线）")
    figure.tight_layout()
    _save(figure, "fig01_data_overview")


def figure_design_coverage(experiments: list[Experiment]) -> None:
    concentration = np.stack([experiment.concentration for experiment in experiments])
    files = np.array([experiment.file_id for experiment in experiments])
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    ax = axes[0, 0]
    ax.plot(files, concentration[:, 1], "o", ms=3.5, color=GAS_COLORS["NH3"])
    ax.set_xlabel("实验序号")
    ax.set_ylabel("NH$_3$ 浓度")
    _letter(ax, "(a) 采集顺序与 NH$_3$ 档位完全对应")

    scatter = axes[0, 1].scatter(concentration[:, 0], concentration[:, 2], c=concentration[:, 1], cmap="viridis", s=26, edgecolor="white", linewidth=0.4)
    axes[0, 1].set_xlabel("H$_2$ 浓度")
    axes[0, 1].set_ylabel("NO$_2$ 浓度")
    figure.colorbar(scatter, ax=axes[0, 1], label="NH$_3$")
    _letter(axes[0, 1], "(b) H$_2$–NO$_2$ 平面")

    scatter = axes[1, 0].scatter(concentration[:, 0], concentration[:, 1], c=concentration[:, 2], cmap="plasma", s=26, edgecolor="white", linewidth=0.4)
    axes[1, 0].set_xlabel("H$_2$ 浓度")
    axes[1, 0].set_ylabel("NH$_3$ 浓度")
    figure.colorbar(scatter, ax=axes[1, 0], label="NO$_2$")
    _letter(axes[1, 0], "(c) H$_2$–NH$_3$ 平面")

    scatter = axes[1, 1].scatter(concentration[:, 1], concentration[:, 2], c=concentration[:, 0], cmap="magma", s=26, edgecolor="white", linewidth=0.4)
    axes[1, 1].set_xlabel("NH$_3$ 浓度")
    axes[1, 1].set_ylabel("NO$_2$ 浓度")
    figure.colorbar(scatter, ax=axes[1, 1], label="H$_2$")
    _letter(axes[1, 1], "(d) NH$_3$–NO$_2$ 平面")
    figure.tight_layout()
    _save(figure, "fig02_design_coverage")


def figure_shape_fit(experiments: list[Experiment], shapes) -> None:
    amplitude, tau, delay = shapes
    relative = np.stack([experiment.relative for experiment in experiments])
    time = experiments[0].time
    index = 30
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))
    ax = axes[0]
    for gas_index, gas in enumerate(GASES):
        prediction = _first_order(time, amplitude[index, gas_index], tau[index, gas_index], delay[index, gas_index])
        target = RESPONSE_SIGNS[gas_index] * relative[index, :, gas_index]
        ax.plot(time, target / amplitude[index, gas_index], color=GAS_COLORS[gas], lw=0.9, alpha=0.55, ls="--")
        ax.plot(time, prediction / amplitude[index, gas_index], color=GAS_COLORS[gas], lw=1.6, label=GAS_LABEL[gas])
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("符号校正响应 / 拟合幅值")
    ax.set_ylim(0.0, 1.12)
    ax.legend(frameon=False, ncol=3, loc="lower right")
    _letter(ax, f"(a) 实验 {index}：实测（虚线）与一阶模型（实线）")

    errors = []
    for gas_index, gas in enumerate(GASES):
        prediction = _first_order(time[None, :], amplitude[:, gas_index, None], tau[:, gas_index, None], delay[:, gas_index, None])
        target = RESPONSE_SIGNS[gas_index] * relative[:, :, gas_index]
        nrmse = np.sqrt(np.mean((prediction - target) ** 2, axis=1)) / np.abs(amplitude[:, gas_index])
        errors.append(nrmse * 100.0)
    ax = axes[1]
    positions = np.arange(3)
    parts = ax.boxplot(errors, positions=positions, widths=0.5, patch_artist=True, showfliers=False)
    for patch, gas in zip(parts["boxes"], GASES):
        patch.set_facecolor(GAS_COLORS[gas])
        patch.set_alpha(0.35)
        patch.set_edgecolor(GAS_COLORS[gas])
    for median in parts["medians"]:
        median.set_color("black")
        median.set_linewidth(1.0)
    for position, values, gas in zip(positions, errors, GASES):
        jitter = np.random.default_rng(0).normal(0, 0.045, size=values.size)
        ax.plot(position + jitter, values, ".", ms=1.8, color=GAS_COLORS[gas], alpha=0.45)
    ax.set_xticks(positions, [GAS_LABEL[gas] for gas in GASES])
    ax.set_ylabel("逐实验拟合 nRMSE / %")
    _letter(ax, "(b) 一阶+延迟的逐实验拟合误差")
    figure.tight_layout()
    _save(figure, "fig03_shape_fit")


def _first_order(
    time: np.ndarray,
    amplitude: np.ndarray,
    tau: np.ndarray,
    delay: np.ndarray,
) -> np.ndarray:
    elapsed = np.maximum(np.asarray(time) - delay, 0.0)
    return amplitude * (1.0 - np.exp(-elapsed / tau))


def figure_logo_comparison(metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 3.3))
    for ax, gas in zip(axes, list(GASES) + ["macro"]):
        key = f"nrmse_{gas}"
        values = []
        for model in MODEL_ORDER:
            values.append(metrics.loc[metrics["model"] == model, key].to_numpy())
        parts = ax.boxplot(values, positions=np.arange(len(MODEL_ORDER)), widths=0.58, patch_artist=True, showfliers=False)
        for patch, model in zip(parts["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.35)
            patch.set_edgecolor(MODEL_COLORS[model])
        for median in parts["medians"]:
            median.set_color("black")
            median.set_linewidth(1.0)
        rng = np.random.default_rng(0)
        for position, model, array in zip(np.arange(len(MODEL_ORDER)), MODEL_ORDER, values):
            ax.plot(position + rng.normal(0, 0.045, size=array.size), array, ".", ms=1.5, color=MODEL_COLORS[model], alpha=0.4)
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[model] for model in MODEL_ORDER], rotation=25, ha="right")
        ax.set_title(GAS_LABEL[gas] if gas in GAS_LABEL else "宏平均", fontsize=9.5, loc="left")
        if gas == "H2":
            ax.set_ylabel("曲线 nRMSE")
    figure.suptitle("留一实验交叉验证：各模型的相对响应曲线误差（越低越好）", fontsize=10.5)
    figure.tight_layout()
    _save(figure, "fig04_logo_comparison")


def figure_gain_tau(experiments: list[Experiment], shapes, model: FirstOrderArrayModel) -> None:
    amplitude, tau, _ = shapes
    concentration = np.stack([experiment.concentration for experiment in experiments])
    figure, axes = plt.subplots(2, 2, figsize=(8.6, 6.6))
    ax = axes[0, 0]
    for index, gas in enumerate(GASES):
        predicted = model.gain(concentration)[:, index]
        ax.loglog(np.abs(amplitude[:, index]), np.abs(predicted), "o", ms=3.0, color=GAS_COLORS[gas], alpha=0.7, label=GAS_LABEL[gas])
    limits = np.array([5e-4, 1.5])
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("逐实验拟合增益")
    ax.set_ylabel("模型增益")
    ax.legend(frameon=False)
    _letter(ax, "(a) 稳态增益一致性（全量拟合）")

    ax = axes[0, 1]
    for index, gas in enumerate(GASES):
        predicted = model.tau(concentration)[:, index]
        ax.loglog(tau[:, index], predicted, "o", ms=3.0, color=GAS_COLORS[gas], alpha=0.7, label=GAS_LABEL[gas])
    limits = np.array([1.0, 80.0])
    ax.plot(limits, limits, color="black", lw=0.8, ls="--")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("逐实验拟合时间常数 / s")
    ax.set_ylabel("模型时间常数 / s")
    ax.legend(frameon=False)
    _letter(ax, "(b) 时间常数一致性（全量拟合）")

    ax = axes[1, 0]
    h2_grid = np.array([8000.0, 16000.0, 24000.0, 32000.0])
    for no2 in [2.0, 6.0, 10.0]:
        grid = np.column_stack([h2_grid, np.full(4, 30.0), np.full(4, no2)])
        ax.plot(h2_grid, model.gain(grid)[:, 0] * 100.0, "-o", ms=3, lw=1.2, label=f"NO$_2$={int(no2)}")
        mask = (concentration[:, 0] >= 8000) & (concentration[:, 1] == 30) & (concentration[:, 2] == no2)
        ax.plot(concentration[mask, 0], np.abs(amplitude[mask, 0]) * 100.0, "s", ms=3.5, mfc="none", color="#333333")
    ax.set_xscale("log")
    ax.set_xlabel("H$_2$ 浓度")
    ax.set_ylabel("H$_2$ 增益 / %")
    ax.legend(frameon=False)
    _letter(ax, "(c) H$_2$ 增益受 NO$_2$ 抑制（NH$_3$=30）")

    ax = axes[1, 1]
    nh3_grid = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    for h2 in [8000.0, 16000.0, 24000.0]:
        grid = np.column_stack([np.full(5, h2), nh3_grid, np.full(5, 6.0)])
        ax.plot(nh3_grid, model.gain(grid)[:, 1] * 100.0, "-o", ms=3, lw=1.2, label=f"H$_2$={int(h2)}")
        mask = (concentration[:, 0] == h2) & (concentration[:, 2] == 6)
        ax.plot(concentration[mask, 1], np.abs(amplitude[mask, 1]) * 100.0, "s", ms=3.5, mfc="none", color="#333333")
    ax.set_xlabel("NH$_3$ 浓度")
    ax.set_ylabel("NH$_3$ 增益 / %")
    ax.legend(frameon=False)
    _letter(ax, "(d) NH$_3$ 增益的浓度与 H$_2$ 交叉影响（NO$_2$=6）")
    figure.tight_layout()
    _save(figure, "fig05_gain_tau")


def figure_prediction_examples(experiments: list[Experiment], metrics: pd.DataFrame, predictions: dict[str, np.ndarray]) -> None:
    time = experiments[0].time
    relative = np.stack([experiment.relative for experiment in experiments])
    concentration = np.stack([experiment.concentration for experiment in experiments])
    files = np.array([experiment.file_id for experiment in experiments])
    scores = metrics.loc[metrics["model"] == "grey_box", "nrmse_macro"].to_numpy()
    order = np.argsort(scores)
    picks = [int(order[0]), int(order[len(order) // 2]), int(order[-1])]
    row_labels = ["最好", "中等", "最差"]
    figure, axes = plt.subplots(3, 3, figsize=(9.8, 6.9), sharex=True)
    for row, index in enumerate(picks):
        h2, nh3, no2 = concentration[index]
        for column, gas in enumerate(GASES):
            ax = axes[row, column]
            prediction = predictions[f"grey_box_{files[index]}"]
            ax.plot(time, relative[index, :, column] * 100.0, color="black", lw=1.0, label="实测")
            ax.plot(time, prediction[:, column] * 100.0, color=GAS_COLORS[gas], lw=1.4, ls="--", label="预测")
            if row == 0:
                ax.set_title(GAS_LABEL[gas], fontsize=9.5)
            if column == 0:
                ax.set_ylabel(
                    f"{row_labels[row]}\n文件 {files[index]} · {scores[index] * 100:.1f}%\n"
                    f"H$_2$={h2:.0f}, NH$_3$={nh3:.0f}, NO$_2$={no2:.0f}",
                    fontsize=7.0,
                )
            if row == 2:
                ax.set_xlabel("时间 / s")
    axes[0, 2].legend(frameon=False, loc="lower right")
    figure.suptitle("灰盒模型留一实验预测示例（虚线为预测，纵轴标签为留出实验的宏 nRMSE）", fontsize=10.5, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    _save(figure, "fig06_prediction_examples")


def figure_absolute(experiments: list[Experiment], model: FirstOrderArrayModel) -> None:
    time = experiments[0].time
    metrics = pd.read_csv(ARTIFACTS / "logo_metrics.csv")
    scores = metrics.loc[metrics["model"] == "grey_box", "nrmse_macro"].to_numpy()
    index = int(np.argsort(scores)[len(scores) // 2])
    experiment = experiments[index]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharex=True)
    prediction = model.predict_response(experiment.concentration[None], time, experiment.initial[None])[0]
    for column, gas in enumerate(GASES):
        ax = axes[column]
        ax.plot(time, experiment.response[:, column], color="black", lw=1.0, label="实测")
        ax.plot(time, prediction[:, column], color=GAS_COLORS[gas], lw=1.4, ls="--", label="预测")
        ax.set_title(GAS_LABEL[gas], fontsize=9.5)
        ax.set_xlabel("时间 / s")
        if column == 0:
            ax.set_ylabel("传感器输出")
        ax.legend(frameon=False)
    figure.suptitle(
        f"由浓度与初始基线重建绝对输出（文件 {experiment.file_id}："
        f"H$_2$={experiment.concentration[0]:.0f}, NH$_3$={experiment.concentration[1]:.0f}, "
        f"NO$_2$={experiment.concentration[2]:.0f}）",
        fontsize=10.5,
    )
    figure.tight_layout()
    _save(figure, "fig07_absolute_response")


def main() -> None:
    _style()
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    model = FirstOrderArrayModel.load(ARTIFACTS / "model.json")
    metrics = pd.read_csv(ARTIFACTS / "logo_metrics.csv")
    predictions = {key: value for key, value in np.load(ARTIFACTS / "logo_predictions.npz").items()}
    figure_data_overview(experiments)
    figure_design_coverage(experiments)
    figure_shape_fit(experiments, shapes)
    figure_logo_comparison(metrics)
    figure_gain_tau(experiments, shapes, model)
    figure_prediction_examples(experiments, metrics, predictions)
    figure_absolute(experiments, model)


if __name__ == "__main__":
    main()
