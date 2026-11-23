"""当前模型的评估图：数据设计、留一曲线误差、模型架构与数据问题。

所有图写入 sensor_array/figures/model/，同时输出 PNG（300 dpi）与 SVG。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..core.data import GASES, load_experiments
from ..paths import EVALUATION
from .style import (
    GAS_COLORS,
    GAS_LABEL,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_ORDER,
    _save,
    _style,
)


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


def _concentration_design(experiments) -> None:
    h2_levels = sorted({item.concentration[0] for item in experiments})
    nh3_levels = sorted({item.concentration[1] for item in experiments})
    no2_levels = sorted({item.concentration[2] for item in experiments})
    counts = np.zeros((len(h2_levels), len(nh3_levels), len(no2_levels)), dtype=int)
    for item in experiments:
        i = h2_levels.index(item.concentration[0])
        j = nh3_levels.index(item.concentration[1])
        k = no2_levels.index(item.concentration[2])
        counts[i, j, k] += 1

    figure = plt.figure(figsize=(11.0, 5.6), layout="constrained")
    outer = figure.add_gridspec(2, 1, height_ratios=(1.0, 0.85))
    top = outer[0].subgridspec(1, 4, wspace=0.32)
    bottom = outer[1].subgridspec(1, 2, wspace=0.28)

    for index, level in enumerate(h2_levels):
        ax = figure.add_subplot(top[0, index])
        values = counts[index]
        ax.imshow(values, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for row in range(len(nh3_levels)):
            for column in range(len(no2_levels)):
                value = int(values[row, column])
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if value else "#B0B0B0",
                )
        ax.set_xticks(range(len(no2_levels)))
        ax.set_xticklabels([f"{value:.0f}" for value in no2_levels])
        ax.set_yticks(range(len(nh3_levels)))
        ax.set_yticklabels([f"{value:.0f}" for value in nh3_levels])
        ax.set_title(rf"$\mathrm{{H}}_2$={level:.0f}")
        ax.set_xlabel(r"$\mathrm{NO}_2$ 浓度")
        if index == 0:
            ax.set_ylabel(r"$\mathrm{NH}_3$ 浓度")

    panels = (
        counts.sum(axis=1),
        counts.sum(axis=2),
        r"$\mathrm{H}_2$–$\mathrm{NO}_2$ 平面：跨 $\mathrm{NH}_3$ 的组合数",
        r"$\mathrm{H}_2$–$\mathrm{NH}_3$ 平面：跨 $\mathrm{NO}_2$ 的组合数",
        r"$\mathrm{NO}_2$ 浓度",
        r"$\mathrm{NH}_3$ 浓度",
    )
    for index in range(2):
        values = panels[index]
        ax = figure.add_subplot(bottom[0, index])
        image = ax.imshow(values, cmap="Blues", vmin=0, vmax=5, aspect="auto")
        for row in range(len(h2_levels)):
            for column in range(5):
                value = int(values[row, column])
                ax.text(
                    column,
                    row,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value >= 3 else "#333333",
                )
        ax.set_xticks(range(5))
        ax.set_xticklabels([f"{value:.0f}" for value in (no2_levels if index == 0 else nh3_levels)])
        ax.set_yticks(range(len(h2_levels)))
        ax.set_yticklabels([f"{value:.0f}" for value in h2_levels])
        ax.set_xlabel(panels[4 + index])
        if index == 0:
            ax.set_ylabel(r"$\mathrm{H}_2$ 浓度")
        ax.set_title(panels[2 + index], fontsize=8.5)

    colorbar = figure.colorbar(image, ax=figure.axes[-2:], orientation="horizontal", shrink=0.55, pad=0.08)
    colorbar.set_label(r"该平面内有数据的组合数")
    _save(figure, "fig09_concentration_design")


def _nh3_initial_drift(experiments) -> None:
    files = np.array([item.file_id for item in experiments])
    concentration = np.array([item.concentration[1] for item in experiments])
    initial = np.array([item.initial[1] for item in experiments])
    correlation_file = float(np.corrcoef(files, initial)[0, 1])
    correlation_concentration = float(np.corrcoef(concentration, initial)[0, 1])
    fold = float(initial.max() / initial.min())

    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), layout="constrained")
    ax = axes[0]
    scatter = ax.scatter(
        files,
        initial,
        c=concentration,
        cmap="viridis",
        s=28,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel("实验序号")
    ax.set_ylabel(r"$\mathrm{NH}_3$ 初始响应 $R(0)$ / 原始单位")
    ax.set_title("(a) 随实验序号漂移")
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(
        0.03,
        0.97,
        f"Pearson $r$ = {correlation_file:.3f}\n最大/最小 = {fold:.2f}×",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label(r"$\mathrm{NH}_3$ 浓度")

    ax = axes[1]
    scatter = ax.scatter(
        concentration,
        initial,
        c=files,
        cmap="plasma",
        s=28,
        edgecolor="white",
        linewidth=0.35,
    )
    ax.set_xlabel(r"$\mathrm{NH}_3$ 浓度 / 原始单位")
    ax.set_ylabel(r"$\mathrm{NH}_3$ 初始响应 $R(0)$ / 原始单位")
    ax.set_title("(b) 与浓度同步上升")
    ax.grid(alpha=0.25, lw=0.5)
    ax.text(
        0.03,
        0.97,
        f"Pearson $r$ = {correlation_concentration:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85),
    )
    colorbar = figure.colorbar(scatter, ax=ax, pad=0.02)
    colorbar.set_label("实验序号")
    _save(figure, "fig10_nh3_initial_drift")




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
    _concentration_design(experiments)
    _nh3_initial_drift(experiments)
    _architecture()
    curve = pd.read_csv(EVALUATION / "curve_metrics.csv")
    _curve_error(curve)
    print("saved figures to", "sensor_array/figures/model")


if __name__ == "__main__":
    main()
