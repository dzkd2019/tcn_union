"""当前模型的评估图：数据设计、曲线示例、代表性曲线、留一误差、下游浓度误差与区间校准。

所有图写入 sensor_array/figures/model/，同时输出 PNG（300 dpi）与 SVG。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


def _representative_curves(experiments, curve: pd.DataFrame, predictions: dict) -> None:
    """五个代表实验的实测曲线与两个正模型生成曲线。

    三个案例按当前模型的宏 nRMSE 取最优、中位、最差；另外两个是设计边界：
    file 4（NH3 最低档、H2/NO2 居中）与 file 61（NH3/NO2 同时最高）。
    """
    macro = curve[curve["model"] == "mlp_per_gas"].set_index("file")["nrmse_macro"]
    best_file = int(macro.idxmin())
    median_file = int(macro.sub(macro.median()).abs().idxmin())
    worst_file = int(macro.idxmax())
    cases = [
        (best_file, "最优"),
        (median_file, "中位"),
        (4, r"$\mathrm{NH}_3$ 低端"),
        (61, r"$\mathrm{NH}_3$/$\mathrm{NO}_2$ 高端"),
        (worst_file, "最差"),
    ]
    figure = plt.figure(figsize=(11.4, 8.8), layout="constrained")
    grid = figure.add_gridspec(len(cases), 4, width_ratios=[0.62, 1, 1, 1])
    axes = np.empty((len(cases), len(GASES)), dtype=object)
    first_axis = None
    for row, (file_id, tag) in enumerate(cases):
        experiment = next(item for item in experiments if item.file_id == file_id)
        header = figure.add_subplot(grid[row, 0])
        header.axis("off")
        header.text(
            0.95,
            0.5,
            f"file {file_id}\n{tag} · 宏 {100.0 * macro[file_id]:.2f}%\n"
            fr"$\mathrm{{H}}_2$ {experiment.concentration[0]:.0f}" "\n"
            fr"$\mathrm{{NH}}_3$ {experiment.concentration[1]:.0f} · $\mathrm{{NO}}_2$ {experiment.concentration[2]:.0f}",
            ha="right",
            va="center",
            fontsize=7.8,
            linespacing=1.45,
        )
        for column, gas in enumerate(GASES):
            ax = figure.add_subplot(grid[row, column + 1], sharex=first_axis)
            if first_axis is None:
                first_axis = ax
            axes[row, column] = ax
            ax.plot(experiment.time, experiment.relative[:, column], color="#333333", lw=1.05, label="实测")
            for model in MODEL_ORDER:
                prediction = predictions[f"{file_id}_{model}"]
                ax.plot(
                    experiment.time,
                    prediction[:, column],
                    color=MODEL_COLORS[model],
                    lw=0.85,
                    label=MODEL_LABELS[model],
                )
            ax.grid(axis="y", alpha=0.15, lw=0.4)
            ax.set_xlim(experiment.time[0], experiment.time[-1])
            if row == 0:
                ax.set_title(f"({chr(97 + column)}) {GAS_LABEL[gas]}")
            if row == len(cases) - 1:
                ax.set_xlabel("时间 / s")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    figure.suptitle("正模型生成曲线与实测对比：五个代表实验", fontsize=10.5)
    _save(figure, "fig06_representative_curves")


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
    labels = ["原始 Laplace 90%", "全局共形", r"$\mathrm{NH}_3$ 档位共形", "采集块共形"]
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
    figure.suptitle(r"P4 区间校准：Laplace 欠覆盖，全局/$\mathrm{NH}_3$ 共形约 90%，采集块偏低", fontsize=10)
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
    ax.set_xlabel(r"$\mathrm{H}_2$ 浓度 / 原始单位")
    ax.set_ylabel(r"$\mathrm{NH}_3$ 浓度 / 原始单位")
    ax.set_title("(a) 62 条实验的设计点")
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$\mathrm{NO}_2$ 浓度 / 原始单位")

    ax = axes[1]
    experiment = experiments[0]
    for gas, column in zip(GASES, range(3)):
        ax.plot(experiment.time, experiment.relative[:, column], color=GAS_COLORS[gas], lw=1.0, label=GAS_LABEL[gas])
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("相对响应")
    ax.set_title(f"(b) file {experiment.file_id} 的原始曲线")
    ax.legend(frameon=False)
    _save(figure, "fig01_design_and_data")



def _architecture() -> None:
    """双指数骨架 + 三路 MLP 残差的推理架构图（不依赖拟合结果）。"""
    figure, ax = plt.subplots(figsize=(9.4, 4.1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    blue = "#4C72B0"
    green = "#009E73"
    ink = "#222222"

    def box(x, y, w, h, text, edge, face, fontsize=9.0):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.07,rounding_size=0.12",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color=ink)
        return patch

    def arrow(start, end, color=ink, style="-|>", linestyle="-", lw=1.0):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle=style,
                mutation_scale=10,
                linewidth=lw,
                color=color,
                linestyle=linestyle,
                shrinkA=2,
                shrinkB=2,
            )
        )

    box(
        0.15,
        1.55,
        1.75,
        1.55,
        "浓度输入\n$\\mathbf{c}=(c_{H_2},c_{NH_3},c_{NO_2})$\n时间 $t$",
        "#666666",
        "#F5F5F5",
        8.6,
    )
    box(
        2.55,
        3.05,
        3.55,
        1.30,
        "双指数骨架\n"
        r"$y_j^{\mathrm{skel}}=s_j A_j(c)\left[f_j(c)\left(1-e^{-t/\tau^{f}_j}\right)"
        r"+\left(1-f_j(c)\right)\left(1-e^{-t/\tau^{s}_j}\right)\right]$",
        blue,
        "#EAF0F8",
        8.4,
    )
    box(
        2.55,
        0.55,
        3.55,
        1.25,
        "三路 MLP 残差（每路独立）\n"
        r"$x=[\log c,\,t,\,y^{\mathrm{skel}}]$" "\n"
        r"$7\to64\to64\to1\;\Rightarrow\; r_j(t,c)$",
        green,
        "#E6F4EF",
        8.4,
    )
    box(
        7.35,
        1.75,
        2.45,
        1.15,
        "三路传感器输出\n$R_j(t)=R_j(0)\\,[1+y_j(t,c)]$\n$j=H_2,NH_3,NO_2$",
        "#666666",
        "#F5F5F5",
        8.6,
    )

    ax.scatter([6.55], [2.55], s=150, color="white", edgecolor=ink, linewidth=1.0, zorder=5)
    ax.text(6.55, 2.55, "+", ha="center", va="center", fontsize=13, color=ink, zorder=6)

    arrow((1.90, 2.65), (2.55, 3.45), color=blue)
    arrow((1.90, 2.05), (2.55, 1.50), color=green)
    arrow((4.33, 3.05), (4.33, 1.80), color=green, linestyle="--")
    ax.text(4.43, 2.42, "$y^{\\mathrm{skel}}$", fontsize=8.2, color=green, va="center")
    arrow((6.10, 3.50), (6.55, 2.78), color=blue)
    arrow((6.10, 1.65), (6.55, 2.32), color=green)
    ax.text(6.30, 3.28, "$y^{\\mathrm{skel}}$", fontsize=7.8, color=blue)
    ax.text(6.30, 1.78, "$r$", fontsize=8.2, color=green)
    arrow((6.80, 2.55), (7.35, 2.55), color=ink)
    ax.text(7.06, 2.70, "$y$", fontsize=8.4, color=ink, ha="center")

    arrow((8.58, 0.95), (8.58, 1.75), color="#666666")
    ax.text(8.68, 1.30, "$R_j(0)$", fontsize=8.2, color="#444444", va="center")

    ax.text(0.15, 4.55, "传感器阵列动态响应建模：可解释骨架 + 学习式残差", fontsize=10.5, color=ink)
    _save(figure, "fig07_model_architecture")


def main() -> None:
    _style()
    experiments = load_experiments()
    _design(experiments)
    curve = pd.read_csv(EVALUATION / "curve_metrics.csv")
    downstream = pd.read_csv(EVALUATION / "downstream_summary.csv")
    predictions = dict(np.load(EVALUATION / "predictions.npz"))
    _curve_examples(experiments, curve, predictions)
    _representative_curves(experiments, curve, predictions)
    _architecture()
    _curve_error(curve)
    _downstream(downstream)
    calibration = CALIBRATION / "summary.csv"
    if calibration.exists():
        _interval_calibration(pd.read_csv(calibration))
    print("saved figures to", "sensor_array/figures/model")


if __name__ == "__main__":
    main()
