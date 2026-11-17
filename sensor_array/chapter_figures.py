"""章节图 fig08–fig16（PNG + SVG）。

运行：``.venv/bin/python -m sensor_array.chapter_figures``
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import GASES, load_experiments
from .figures import (
    ARTIFACTS,
    GAS_COLORS,
    GAS_LABEL,
    MODEL_COLORS,
    MODEL_LABELS,
    _letter,
    _save,
    _style,
)
from .hybrid import HybridArrayModel, bspline_basis
from .model import FirstOrderArrayModel, fit_experiment_shapes

MODEL_ORDER = ["mean_curve", "direct_ridge", "two_stage", "tcn", "grey_box", "hybrid"]
MODEL_COLORS = {**MODEL_COLORS, "hybrid": "#4C72B0"}
MODEL_LABELS = {**MODEL_LABELS, "grey_box": "灰盒", "hybrid": "混合（本文）"}


def figure_hybrid(experiments, model: HybridArrayModel) -> None:
    sensitivity = pd.read_csv(ARTIFACTS / "hybrid_sensitivity.csv")
    aggregate = sensitivity.groupby("n_basis")["nrmse_macro"].agg(["mean", "std"])
    time = experiments[0].time
    index = 21
    experiment = experiments[index]
    relative = experiment.relative

    figure, axes = plt.subplots(2, 3, figsize=(9.6, 5.8))
    ax = axes[0, 0]
    ax.errorbar(
        aggregate.index,
        aggregate["mean"] * 100.0,
        yerr=aggregate["std"] / np.sqrt(len(sensitivity)) * 100.0,
        color="#4C72B0",
        marker="o",
        ms=3.5,
        lw=1.2,
        capsize=2.5,
        label="均值 ± 标准误",
    )
    ax.axhline(3.47, color=MODEL_COLORS["grey_box"], ls="--", lw=1.0, label="灰盒 3.47%")
    ax.axhline(2.59, color=MODEL_COLORS["direct_ridge"], ls=":", lw=1.0, label="岭回归 2.59%")
    ax.set_xlabel("残差时间基数量 K")
    ax.set_ylabel("LOGO 宏 nRMSE / %")
    ax.set_ylim(2.45, 3.65)
    ax.legend(frameon=False)
    _letter(ax, "(a) K 的边际收益")

    ax = axes[0, 1]
    basis = bspline_basis(time, model.n_basis, model.spline_degree, model.t_max)
    for k in range(basis.shape[1]):
        ax.plot(time, basis[:, k], lw=1.1, label=f"$\\phi_{k + 1}$")
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("残差时间基")
    ax.legend(frameon=False, ncol=2, fontsize=6.5)
    _letter(ax, "(b) 在 t=0 处为零的 B 样条基")

    ax = axes[0, 2]
    prediction = model.predict_relative(experiment.concentration[None], time)[0]
    grey = model.grey.predict_relative(experiment.concentration[None], time)[0]
    for gas_index, gas in enumerate(GASES):
        ax.plot(time, (prediction[:, gas_index] - grey[:, gas_index]) * 100.0, color=GAS_COLORS[gas], lw=1.2, label=GAS_LABEL[gas])
    ax.set_xlabel("时间 / s")
    ax.set_ylabel("混合修正 / %")
    ax.legend(frameon=False)
    _letter(ax, f"(c) 文件 {experiment.file_id} 的残差修正")

    for gas_index, gas in enumerate(GASES):
        ax = axes[1, gas_index]
        ax.plot(time, relative[:, gas_index] * 100.0, color="black", lw=1.1, label="实测")
        ax.plot(time, grey[:, gas_index] * 100.0, color=MODEL_COLORS["grey_box"], lw=1.2, ls="--", label="灰盒")
        ax.plot(time, prediction[:, gas_index] * 100.0, color=MODEL_COLORS["hybrid"], lw=1.3, ls="--", label="混合")
        ax.set_xlabel("时间 / s")
        ax.set_title(GAS_LABEL[gas], fontsize=9.0)
        if gas_index == 0:
            ax.set_ylabel("相对响应 / %")
            ax.legend(frameon=False)
    figure.suptitle(f"混合模型：一阶骨架 + 低秩残差（示例为文件 {experiment.file_id} 的全量拟合）", fontsize=10.5)
    figure.tight_layout()
    _save(figure, "fig08_hybrid_model")


def figure_crlb(estimate: pd.DataFrame | None) -> None:
    table = pd.read_csv(ARTIFACTS / "identifiability.csv")
    table = table[table["mode"] == "c"]
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    ax = axes[0]
    for gas, column in zip(GASES, ("crlb_h2", "crlb_nh3", "crlb_no2")):
        group = table.groupby("tmax")[column]
        median, p90 = group.median(), group.quantile(0.9)
        ax.plot(median.index, median.to_numpy(), "-o", ms=3, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
        ax.plot(p90.index, p90.to_numpy(), "--", lw=0.9, color=GAS_COLORS[gas], alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
    ax.minorticks_off()
    ax.set_xlabel("时间窗上限 / s")
    ax.set_ylabel("CRLB（相对浓度标准差）/ %")
    ax.legend(frameon=False, title="实线中位 / 虚线 p90")
    _letter(ax, "(a) 噪声下限：CRLB")

    ax = axes[1]
    if estimate is not None and not estimate.empty:
        for gas in GASES:
            subset = estimate[estimate["gas"] == gas]
            group = subset.groupby("tmax")["se"]
            median, p90 = group.median(), group.quantile(0.9)
            ax.plot(median.index, (np.exp(median.to_numpy()) - 1.0) * 100.0, "-o", ms=3, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
            ax.plot(p90.index, (np.exp(p90.to_numpy()) - 1.0) * 100.0, "--", lw=0.9, color=GAS_COLORS[gas], alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
    ax.minorticks_off()
    ax.set_xlabel("时间窗上限 / s")
    ax.set_ylabel("Laplace 后验标准差 / %")
    ax.legend(frameon=False, title="实线中位 / 虚线 p90")
    _letter(ax, "(b) 计入模型偏差后的实际不确定性")
    figure.tight_layout()
    _save(figure, "fig09_crlb_window")


def figure_identifiability_map(experiments) -> None:
    table = pd.read_csv(ARTIFACTS / "identifiability.csv")
    table = table[(table["mode"] == "c") & (table["tmax"] == 5.0)]
    concentration = np.stack([experiment.concentration for experiment in experiments])
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    specs = [
        (0, "crlb_h2", 0, 2, "H$_2$", "NO$_2$"),
        (1, "crlb_nh3", 1, 0, "NH$_3$", "H$_2$"),
        (2, "crlb_no2", 2, 1, "NO$_2$", "NH$_3$"),
    ]
    for ax, (gas_index, column, x_index, color_index, xlabel, color_label) in zip(axes, specs):
        scatter = ax.scatter(
            concentration[:, x_index],
            table[column].to_numpy(),
            c=concentration[:, color_index],
            cmap="viridis",
            s=24,
            edgecolor="white",
            linewidth=0.4,
        )
        ax.set_yscale("log")
        ax.axhline(1.0, color="#888888", ls=":", lw=0.9)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("CRLB / %")
        figure.colorbar(scatter, ax=ax, label=color_label)
        _letter(ax, f"({'abc'[gas_index]}) {GAS_LABEL[GASES[gas_index]]}（5 s）")
    figure.suptitle("5 秒窗的可辨识性地图：越靠上越难估计（虚线为 1%）", fontsize=10.0)
    figure.tight_layout()
    _save(figure, "fig10_identifiability_map")


def figure_error_map(estimate: pd.DataFrame, experiments) -> None:
    data = estimate[estimate["tmax"] == 10.0].copy()
    data["value"] = np.exp(data["theta_true"])
    data["error"] = data["abs_rel_error"] * 100.0
    lookup = {
        experiment.file_id: experiment.concentration
        for experiment in experiments
    }
    data["h2"] = [lookup[file_id][0] for file_id in data["file"]]
    data["nh3"] = [lookup[file_id][1] for file_id in data["file"]]
    data["no2"] = [lookup[file_id][2] for file_id in data["file"]]
    specs = [("H2", "NO2"), ("NH3", "H2"), ("NO2", "NH3")]
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, (gas, other) in zip(axes, specs):
        subset = data[data["gas"] == gas]
        pivot = subset.pivot_table(
            index="value", columns=other.lower(), values="error", aggfunc="median"
        )
        values = np.ma.masked_invalid(pivot.to_numpy())
        image = ax.imshow(values, cmap="magma_r", vmin=0.0, vmax=25.0, aspect="auto")
        ax.set_xticks(range(pivot.shape[1]), [f"{value:g}" for value in pivot.columns])
        ax.set_yticks(range(pivot.shape[0]), [f"{value:g}" for value in pivot.index])
        ax.set_xlabel(GAS_LABEL[other])
        ax.set_ylabel(GAS_LABEL[gas])
        for row in range(pivot.shape[0]):
            for column in range(pivot.shape[1]):
                value = pivot.to_numpy()[row, column]
                if np.isfinite(value):
                    ax.text(
                        column,
                        row,
                        f"{value:.1f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color="white" if value > 12.5 else "black",
                    )
        _letter(ax, f"({'abc'[GASES.index(gas)]}) {GAS_LABEL[gas]}（行：{GAS_LABEL[gas]}，列：{GAS_LABEL[other]}）")
    figure.suptitle("10 秒窗的误差地图：设计域边角的误差明显更大", fontsize=10.0)
    figure.tight_layout()
    figure.subplots_adjust(right=0.88)
    colorbar_axis = figure.add_axes((0.9, 0.18, 0.015, 0.64))
    figure.colorbar(image, cax=colorbar_axis, label="中位绝对相对误差 / %")
    _save(figure, "fig11_error_map")


def figure_early_error(estimate: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.2), sharey=True)
    for ax, quantile, title in ((axes[0], 0.5, "(a) 中位数"), (axes[1], 0.9, "(b) p90")):
        for gas in GASES:
            group = estimate[estimate["gas"] == gas].groupby("tmax")["abs_rel_error"].quantile(quantile)
            ax.plot(group.index, group.to_numpy() * 100.0, "-o", ms=3, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
        ax.minorticks_off()
        ax.axhline(5.0, color="#888888", ls=":", lw=0.8)
        ax.axhline(10.0, color="#888888", ls=":", lw=0.8)
        ax.set_xlabel("时间窗上限 / s")
        ax.legend(frameon=False)
        _letter(ax, title)
    axes[0].set_ylabel("绝对相对误差 / %")
    figure.suptitle("早期浓度估计：中位误差随窗长下降，NO$_2$ 收敛最慢；p90 仍受模型偏差限制", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig12_early_error")


def figure_coverage(conformal: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.8, 3.0))
    ax = axes[0]
    for alpha in (0.1, 0.2):
        group = conformal[conformal["alpha"] == alpha].groupby("tmax")["covered"].mean()
        ax.plot(group.index, group.to_numpy() * 100.0, "-o", ms=3, label=f"名义 {100 * (1 - alpha):.0f}%")
        ax.axhline(100 * (1 - alpha), color="#888888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
    ax.minorticks_off()
    ax.set_ylim(60, 100)
    ax.set_xlabel("时间窗上限 / s")
    ax.set_ylabel("CV+ 实际覆盖率 / %")
    ax.legend(frameon=False)
    _letter(ax, "(a) 覆盖率")

    ax = axes[1]
    for gas in GASES:
        subset = conformal[(conformal["alpha"] == 0.1) & (conformal["gas"] == gas)]
        width = subset.groupby("tmax")["width_rel"].median()
        ax.plot(width.index, width.to_numpy() * 100.0, "-o", ms=3, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
    for target in (5.0, 10.0, 20.0):
        ax.axhline(target, color="#888888", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
    ax.minorticks_off()
    ax.set_xlabel("时间窗上限 / s")
    ax.set_ylabel("90% 区间宽度（中位）/ %")
    ax.legend(frameon=False)
    _letter(ax, "(b) 区间宽度")
    figure.suptitle("不确定性：覆盖率接近名义值，但区间宽度由少数偏差大的实验决定", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig13_coverage")


def figure_stopping(stopping: pd.DataFrame) -> None:
    targets = sorted(stopping["target"].unique())
    reached = stopping.groupby("target")["reached"].agg(["sum", "size"])
    median = stopping[stopping["reached"]].groupby("target")["stop_time"].median()
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    ax = axes[0]
    ax.bar(range(len(targets)), [100.0 * reached.loc[target, "sum"] / reached.loc[target, "size"] for target in targets], color="#4C72B0")
    for position, target in enumerate(targets):
        ax.text(position, 102.0, f"{int(reached.loc[target, 'sum'])}/62", ha="center", fontsize=7)
    ax.set_xticks(range(len(targets)), [f"{100 * target:.0f}%" for target in targets])
    ax.set_ylim(0, 115)
    ax.set_xlabel("目标区间宽度")
    ax.set_ylabel("达到目标的实验比例 / %")
    _letter(ax, "(a) 能否给出足够窄的区间")

    ax = axes[1]
    values = [median.get(target, np.nan) for target in targets]
    ax.bar(range(len(targets)), values, color="#4C72B0")
    for position, value in enumerate(values):
        if np.isfinite(value):
            ax.text(position, value + 1.5, f"{value:.0f} s", ha="center", fontsize=7)
    ax.set_xticks(range(len(targets)), [f"{100 * target:.0f}%" for target in targets])
    ax.set_xlabel("目标区间宽度")
    ax.set_ylabel("中位停止时间 / s")
    ax.set_ylim(0, float(np.nanmax(values)) * 1.25)
    _letter(ax, "(b) 达到目标所需时间")
    figure.suptitle("自适应停止：10% 宽度目标只在约四分之一的实验上可实现", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig14_stopping")


def figure_degradation() -> None:
    data = pd.read_csv(ARTIFACTS / "degradation.csv")
    median = data.groupby(["drop", "gas"])["abs_rel_error"].median().unstack() * 100.0
    order = ["none", "H2", "NH3", "NO2"]
    labels = ["完整", "去 H$_2$", "去 NH$_3$", "去 NO$_2$"]
    positions = np.arange(len(order))
    figure, ax = plt.subplots(figsize=(5.6, 3.2))
    width = 0.25
    for offset, gas in enumerate(GASES):
        ax.bar(positions + (offset - 1) * width, median.loc[order, gas], width, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("中位绝对相对误差 / %")
    ax.legend(frameon=False, ncol=3)
    ax.set_title("留一传感器降级（10 s 窗）", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig15_degradation")


def figure_budget() -> None:
    data = pd.read_csv(ARTIFACTS / "error_budget.csv")
    sources = [("real", "实测曲线", "#444444"), ("hybrid", "混合生成", "#4C72B0"), ("grey", "灰盒生成", "#009E73")]
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, gas in zip(axes, GASES):
        subset = data[data["gas"] == gas]
        for source, label, color in sources:
            group = subset[subset["source"] == source].groupby("tmax")["abs_rel_error"].median()
            ax.plot(group.index, group.to_numpy() * 100.0, "-o", ms=3, color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([5, 10, 20, 60], ["5", "10", "20", "60"])
        ax.minorticks_off()
        ax.set_xlabel("时间窗上限 / s")
        _letter(ax, f"({'abc'[GASES.index(gas)]}) {GAS_LABEL[gas]}")
    axes[0].set_ylabel("中位绝对相对误差 / %")
    axes[0].legend(frameon=False)
    figure.suptitle("误差预算：同一估计器在模型自生成曲线上的误差远低于实测，说明模型偏差是主因", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig16_error_budget")


def main() -> None:
    _style()
    experiments = load_experiments()
    model = HybridArrayModel.load(ARTIFACTS / "hybrid.json")
    estimate = None
    path = ARTIFACTS / "estimates.csv"
    if path.exists():
        estimate = pd.read_csv(path)
    figure_hybrid(experiments, model)
    figure_crlb(estimate)
    figure_identifiability_map(experiments)
    if estimate is not None:
        figure_error_map(estimate, experiments)
        figure_early_error(estimate)
        figure_coverage(pd.read_csv(ARTIFACTS / "conformal.csv"))
        figure_stopping(pd.read_csv(ARTIFACTS / "stopping.csv"))
    if (ARTIFACTS / "degradation.csv").exists():
        figure_degradation()
    if (ARTIFACTS / "error_budget.csv").exists():
        figure_budget()


if __name__ == "__main__":
    main()
