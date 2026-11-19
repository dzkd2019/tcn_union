"""当前模型的评估图：数据设计、曲线示例、留一误差、下游浓度误差与区间校准。

所有图写入 sensor_array/figures/model/，同时输出 PNG（300 dpi）与 SVG。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.data import GASES, load_experiments
from ..paths import CALIBRATION, EVALUATION
from .style import (
    GAS_COLORS,
    GAS_LABEL,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    _save,
    _style,
)


def _curve_examples(experiments, curve: pd.DataFrame, predictions: dict) -> None:
    macro = curve[curve["model"] == "double"].set_index("file")["nrmse_macro"]
    median_file = int(macro.sub(macro.median()).abs().idxmin())
    worst_file = int(macro.idxmax())
    selected = [worst_file, median_file]
    figure, axes = plt.subplots(2, 3, figsize=(10.6, 5.0), layout="constrained")
    for row, file_id in enumerate(selected):
        experiment = next(item for item in experiments if item.file_id == file_id)
        for column, gas in enumerate(GASES):
            ax = axes[row, column]
            ax.plot(experiment.time, experiment.relative[:, column], color="#333333", lw=1.1, label="实测")
            for model in MODEL_ORDER:
                prediction = predictions[f"{file_id}_{model}"]
                ax.plot(experiment.time, prediction[:, column], color=MODEL_COLORS[model], lw=0.9, label=MODEL_LABELS[model])
            if row == 0:
                ax.set_title(f"({chr(97 + column)}) {GAS_LABEL[gas]}")
            if column == 0:
                ax.set_ylabel(f"file {file_id}\n相对响应")
            if row == 1:
                ax.set_xlabel("时间 / s")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    _save(figure, "fig02_curve_examples")


def _curve_error(curve: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 3.7), layout="constrained")
    ax = axes[0]
    positions = []
    data = []
    colors = []
    spacing = 1.9
    for gas_index, gas in enumerate(GASES):
        base = gas_index * spacing
        for model_index, model in enumerate(MODEL_ORDER):
            positions.append(base + 0.55 + (model_index - 0.5) * 0.32)
            data.append(100.0 * curve.loc[curve["model"] == model, f"nrmse_{gas}"].to_numpy())
            colors.append(MODEL_COLORS[model])
    box = ax.boxplot(data, positions=positions, widths=0.27, patch_artist=True, showfliers=False)
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([index * spacing + 0.55 for index in range(len(GASES))])
    ax.set_xticklabels([GAS_LABEL[gas] for gas in GASES])
    ax.set_yscale("log")
    ax.set_ylabel("逐文件曲线 nRMSE / %")
    ax.set_title("(a) 留一曲线误差")
    ax.grid(axis="y", alpha=0.25, lw=0.5)

    ax = axes[1]
    x = 100.0 * curve[curve["model"] == "double"].set_index("file")["nrmse_macro"]
    y = 100.0 * curve[curve["model"] == "mlp_per_gas"].set_index("file")["nrmse_macro"]
    ax.scatter(x, y, s=14, color=MODEL_COLORS["mlp_per_gas"], alpha=0.75, linewidths=0)
    low = min(x.min(), y.min()) * 0.8
    high = max(x.max(), y.max()) * 1.2
    ax.plot([low, high], [low, high], color="#555555", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_xlabel("双指数骨架宏平均 nRMSE / %")
    ax.set_ylabel("三路独立 MLP 宏平均 nRMSE / %")
    improved = int((y < x).sum())
    ax.text(0.05, 0.92, f"{improved}/{len(x)} 折改善", transform=ax.transAxes, fontsize=8)
    ax.set_title("(b) 逐文件宏平均")
    handles = [
        plt.Line2D([], [], color=MODEL_COLORS[model], lw=6, label=MODEL_LABELS[model])
        for model in MODEL_ORDER
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    _save(figure, "fig03_curve_error")


def _downstream(downstream: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.4), layout="constrained")
    panels = ((10.0, False, "(a) 10 s 中位绝对相对误差"), (5.0, False, "(b) 5 s 中位绝对相对误差"), (10.0, True, "(c) 10 s p90 绝对相对误差"))
    width = 0.30
    x = np.arange(len(GASES))
    for ax, (window, p90, title) in zip(axes, panels):
        maxima = []
        for model_index, model in enumerate(MODEL_ORDER):
            values = []
            for gas in GASES:
                column = "p90_abs_rel_error" if p90 else "median_abs_rel_error"
                row = downstream[
                    (downstream["model"] == model)
                    & (downstream["gas"] == gas)
                    & (downstream["tmax"] == window)
                ]
                values.append(100.0 * float(row[column].iloc[0]))
            values = np.array(values)
            maxima.append(values.max())
            offset = (model_index - (len(MODEL_ORDER) - 1) / 2.0) * width
            ax.bar(x + offset, values, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
            for xi, value in zip(x + offset, values):
                ax.text(xi, value + 0.2, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([GAS_LABEL[gas] for gas in GASES])
        ax.set_ylabel("绝对相对误差 / %")
        ax.set_title(title)
        ax.set_ylim(0, max(maxima) * 1.18)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
    _save(figure, "fig04_downstream")


def _interval_calibration(summary: pd.DataFrame) -> None:
    methods = ["laplace90", "conformal_all", "conformal_NH3", "conformal_block"]
    labels = ["原始 Laplace 90%", "全局共形", "NH₃ 档位共形", "采集块共形"]
    colors = ["#777777", "#4C72B0", "#E69F00", "#009E73"]
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), layout="constrained")
    x = np.arange(len(GASES))
    width = 0.2
    panels = (
        ("development", "coverage", "(a) 开发集覆盖率", 100.0),
        ("development", "median_width_rel", "(b) 开发集中位区间宽度", 100.0),
        ("locked", "coverage", "(c) 锁定集覆盖率", 100.0),
    )
    for ax, (split, column, title, scale) in zip(axes, panels):
        tops = []
        for method_index, method in enumerate(methods):
            values = []
            for gas in GASES:
                row = summary[
                    (summary["method"] == method)
                    & (summary["split"] == split)
                    & (summary["gas"] == gas)
                    & (summary["tmax"] == 10.0)
                ]
                values.append(scale * float(row[column].iloc[0]))
            values = np.array(values)
            tops.append(values.max())
            offset = (method_index - (len(methods) - 1) / 2.0) * width
            ax.bar(x + offset, values, width, color=colors[method_index], label=labels[method_index])
        ax.set_xticks(x)
        ax.set_xticklabels([GAS_LABEL[gas] for gas in GASES])
        ax.set_title(title)
        if column == "coverage":
            ax.axhline(90.0, color="#333333", ls=":", lw=0.8)
            ax.set_ylim(0, 105)
            ax.set_ylabel("覆盖率 / %")
        else:
            ax.set_ylim(0, max(tops) * 1.18)
            ax.set_ylabel("中位宽度 / %")
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    figure.suptitle("P4 区间校准：Laplace 欠覆盖，全局/NH₃ 共形约 90%，采集块偏低", fontsize=10)
    _save(figure, "fig05_interval_calibration")


def _design(experiments) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), layout="constrained")
    ax = axes[0]
    concentration = np.stack([experiment.concentration for experiment in experiments])
    scatter = ax.scatter(
        concentration[:, 0],
        concentration[:, 1],
        c=concentration[:, 2],
        cmap="viridis",
        s=28,
        edgecolor="white",
        linewidth=0.4,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("H₂ 浓度 / 原始单位")
    ax.set_ylabel("NH₃ 浓度 / 原始单位")
    ax.set_title("(a) 62 条实验的设计点")
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("NO₂ 浓度 / 原始单位")

    ax = axes[1]
    experiment = experiments[0]
    for gas, column in zip(GASES, range(3)):
        ax.plot(experiment.time, experiment.relative[:, column], color=GAS_COLORS[gas], lw=1.0, label=GAS_LABEL[gas])
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("相对响应")
    ax.set_title(f"(b) file {experiment.file_id} 的原始曲线")
    ax.legend(frameon=False)
    _save(figure, "fig01_design_and_data")


def main() -> None:
    _style()
    experiments = load_experiments()
    _design(experiments)
    curve = pd.read_csv(EVALUATION / "curve_metrics.csv")
    downstream = pd.read_csv(EVALUATION / "downstream_summary.csv")
    predictions = dict(np.load(EVALUATION / "predictions.npz"))
    _curve_examples(experiments, curve, predictions)
    _curve_error(curve)
    _downstream(downstream)
    calibration = CALIBRATION / "summary.csv"
    if calibration.exists():
        _interval_calibration(pd.read_csv(calibration))
    print("saved figures to", "sensor_array/figures/model")


if __name__ == "__main__":
    main()
