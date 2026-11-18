"""章节图 fig08–fig27（PNG + SVG）。

运行：``.venv/bin/python -m sensor_array.plotting.chapter``
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core.data import GASES, load_experiments
from .style import (
    GAS_COLORS,
    GAS_LABEL,
    MODEL_COLORS,
    MODEL_LABELS,
    _letter,
    _save,
    _style,
)
from ..core.hybrid import HybridArrayModel, bspline_basis
from ..core.grey_box import FirstOrderArrayModel, fit_experiment_shapes
from ..paths import ANALYSIS, AUGMENTATION, MODELS, VALIDATION

MODEL_ORDER = ["mean_curve", "direct_ridge", "two_stage", "tcn", "grey_box", "hybrid"]
MODEL_COLORS = {**MODEL_COLORS, "hybrid": "#4C72B0"}
MODEL_LABELS = {**MODEL_LABELS, "grey_box": "灰盒", "hybrid": "混合（本文）"}

PROTOCOL_ORDER = ["logo", "level", "block"]
PROTOCOL_COLORS = {"logo": "#444444", "level": "#E69F00", "block": "#7B3294"}
PROTOCOL_LABELS = {"logo": "留一实验", "level": "留一档位", "block": "留一采集块"}
DOWNSTREAM_MODELS = ["grey_box", "hybrid", "direct_ridge"]


def figure_hybrid(experiments, model: HybridArrayModel) -> None:
    sensitivity = pd.read_csv(ANALYSIS / "hybrid_sensitivity.csv")
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
    table = pd.read_csv(ANALYSIS / "identifiability.csv")
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
    table = pd.read_csv(ANALYSIS / "identifiability.csv")
    table = table[(table["mode"] == "c") & (table["tmax"] == 5.0)]
    concentration = np.stack([experiment.concentration for experiment in experiments])
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.3))
    specs = [
        (0, "crlb_h2", 0, 2, "H₂", "NO₂"),
        (1, "crlb_nh3", 1, 0, "NH₃", "H₂"),
        (2, "crlb_no2", 2, 1, "NO₂", "NH₃"),
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
    figure.suptitle("早期浓度估计：中位误差随窗长下降，NO₂ 收敛最慢；p90 仍受模型偏差限制", fontsize=9.5)
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
    data = pd.read_csv(ANALYSIS / "degradation.csv")
    order = ["none", "H2", "NH3", "NO2"]
    labels = {"none": "完整", "H2": "缺 H₂", "NH3": "缺 NH₃", "NO2": "缺 NO₂"}
    colors = {"none": "#444444", **GAS_COLORS}
    figure, axes = plt.subplots(1, 3, figsize=(9.0, 3.0), sharey=True)
    for ax, gas in zip(axes, GASES):
        subset = data[data["gas"] == gas]
        for drop in order:
            group = subset[subset["drop"] == drop].groupby("tmax")["abs_rel_error"].median() * 100.0
            ax.plot(group.index, group.to_numpy(), "-o", ms=3, lw=1.1, color=colors[drop], label=labels[drop])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
        ax.set_yticks([2, 5, 10, 20, 50], ["2", "5", "10", "20", "50"])
        ax.set_ylim(1.8, 80.0)
        ax.minorticks_off()
        ax.set_xlabel("时间窗上限 / s")
        _letter(ax, f"({'abc'[GASES.index(gas)]}) {GAS_LABEL[gas]}")
    axes[0].set_ylabel("中位绝对相对误差 / %")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    figure.suptitle("留一传感器降级：60 s 窗多数缺失通道可收敛，H₂ 缺失仍不可替代", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig15_degradation")


def figure_budget() -> None:
    data = pd.read_csv(ANALYSIS / "error_budget.csv")
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


def figure_residual_map() -> None:
    with np.load(ANALYSIS / "residual_matrix.npz") as data:
        time = data["time"]
        residual = data["residual"]
        scale = data["scale"]
        files = data["file_id"]
    normalized = residual / scale[:, None, :] * 100.0
    limit = float(np.percentile(np.abs(normalized), 99.0))
    figure, axes = plt.subplots(3, 1, figsize=(7.6, 6.4), sharex=True)
    image = None
    for index, (ax, gas) in enumerate(zip(axes, GASES)):
        image = ax.imshow(
            normalized[:, :, index],
            aspect="auto",
            origin="lower",
            extent=(time[0], time[-1], -0.5, len(files) - 0.5),
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        for boundary in (6.5, 20.5, 34.5, 48.5):
            ax.axhline(boundary, color="#333333", lw=0.5, ls=":")
        ax.set_yticks([3, 13.5, 27.5, 41.5, 55.5], ["10", "20", "30", "40", "50"])
        ax.set_ylabel("NH₃ 档位")
        _letter(ax, f"({'abc'[index]}) {GAS_LABEL[gas]}")
    axes[-1].set_xlabel("时间 / s")
    figure.suptitle("62 条实验的折外残差：误差集中在 NH₃ 高浓度端与 H₂ 边角", fontsize=9.5)
    figure.tight_layout(rect=(0, 0, 0.91, 0.97))
    colorbar = figure.colorbar(image, cax=figure.add_axes((0.93, 0.08, 0.012, 0.82)))
    colorbar.set_label("折外残差 / 训练折尺度 %", fontsize=8)
    _save(figure, "fig17_residual_map")


def figure_pca() -> None:
    modes = pd.read_csv(ANALYSIS / "residual_pca_modes.csv")
    scores = pd.read_csv(ANALYSIS / "residual_pca_scores.csv")
    diagnostics = pd.read_csv(ANALYSIS / "residual_diagnostics.csv")
    correlation = pd.read_csv(ANALYSIS / "residual_correlations.csv")
    variance = modes.drop_duplicates(["gas", "component"]).pivot(
        index="gas", columns="component", values="variance_ratio"
    )

    figure, axes = plt.subplots(2, 2, figsize=(7.4, 5.4))
    ax = axes[0, 0]
    width = 0.26
    for position, gas in enumerate(GASES):
        values = variance.loc[gas, [1, 2, 3]].to_numpy() * 100.0
        ax.bar(np.arange(3) + (position - 1) * width, values, width, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
    ax.set_xticks(range(3), ["PC1", "PC2", "PC3"])
    ax.set_ylabel("方差占比 / %")
    ax.legend(frameon=False)
    _letter(ax, "(a) 低秩结构")

    for ax, component, letter in ((axes[0, 1], 1, "(b) 第一模式"), (axes[1, 0], 2, "(c) 第二模式")):
        for gas in GASES:
            subset = modes[(modes["gas"] == gas) & (modes["component"] == component)]
            ax.plot(subset["time"], subset["shape"], color=GAS_COLORS[gas], lw=1.2, label=GAS_LABEL[gas])
        ax.set_xlabel("时间 / s")
        ax.set_ylabel("模式形状 / a.u.")
        _letter(ax, letter)
    axes[0, 1].legend(frameon=False, fontsize=7)
    axes[0, 1].text(1.5, 0.088, "NH₃ 阶跃来自 file 59–61 的采集跳变", fontsize=6.5, color="#444444")

    ax = axes[1, 1]
    subset = scores[(scores["gas"] == "NO2") & (scores["component"] == 2)]
    subset = subset.merge(diagnostics[diagnostics["gas"] == "NO2"][["file", "amplitude", "tau"]], on="file")
    scatter = ax.scatter(subset["amplitude"], subset["score"], s=14, c=subset["tau"], cmap="viridis")
    ax.set_xlabel("NO₂ 幅度 A")
    ax.set_ylabel("PC2 得分")
    selected = correlation[(correlation["gas"] == "NO2") & (correlation["feature"] == "score_pc2")]
    rho_amplitude = float(selected[selected["variable"] == "amplitude"]["rho"].iloc[0])
    rho_tau = float(selected[selected["variable"] == "tau"]["rho"].iloc[0])
    ax.text(0.52, 0.95, f"rho(A)={rho_amplitude:+.2f}\nrho(tau)={rho_tau:+.2f}", transform=ax.transAxes, va="top", fontsize=7)
    figure.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02).set_label("τ / s")
    _letter(ax, "(d) NO₂ 第二模式与幅度/时间常数")
    figure.tight_layout()
    _save(figure, "fig18_residual_pca")


def figure_shape() -> None:
    data = pd.read_csv(ANALYSIS / "residual_diagnostics.csv")
    correlation = pd.read_csv(ANALYSIS / "residual_correlations.csv")
    figure, axes = plt.subplots(2, 2, figsize=(7.6, 5.4))

    ax = axes[0, 0]
    share = data.groupby("gas")[["gain_share", "shape_share"]].median()
    for position, gas in enumerate(GASES):
        gain = share.loc[gas, "gain_share"] * 100.0
        shape = share.loc[gas, "shape_share"] * 100.0
        ax.bar(position, gain, 0.6, color=GAS_COLORS[gas])
        ax.bar(position, shape, 0.6, bottom=gain, color=GAS_COLORS[gas], alpha=0.35, hatch="//", edgecolor="white")
        ax.text(position, gain / 2, f"{gain:.0f}", ha="center", va="center", fontsize=7.5, color="white" if gas != "NH3" else "#3A2A00")
        ax.text(position, gain + shape / 2, f"{shape:.0f}", ha="center", va="center", fontsize=7.5, color="#333333")
    ax.set_xticks(range(3), [GAS_LABEL[gas] for gas in GASES])
    ax.set_ylabel("残差能量占比 / %（增益 / 形状）")
    _letter(ax, "(a) 增益与形状分解（中位）")

    ax = axes[0, 1]
    for gas in GASES:
        energy = data[data["gas"] == gas].sort_values("energy", ascending=False)["energy"].to_numpy()
        cumulative = np.cumsum(energy) / energy.sum() * 100.0
        ax.plot(np.arange(1, len(energy) + 1), cumulative, color=GAS_COLORS[gas], lw=1.2, label=GAS_LABEL[gas])
    ax.axhline(80.0, color="#777777", lw=0.7, ls="--")
    ax.set_xlabel("实验数（按残差能量降序）")
    ax.set_ylabel("累计能量 / %")
    ax.legend(frameon=False, fontsize=7)
    _letter(ax, "(b) 误差集中在少数实验")

    ax = axes[1, 0]
    for gas in GASES:
        subset = data[data["gas"] == gas]
        ax.scatter(subset["knn1"], subset["nrmse"] * 100.0, s=10, color=GAS_COLORS[gas], alpha=0.75, label=GAS_LABEL[gas])
    ax.set_xlabel("到最近训练点的对数浓度距离")
    ax.set_ylabel("nRMSE / %")
    ax.annotate("file 0", xy=(0.69, 33.0), xytext=(0.58, 30.5), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6, color="#555555"))
    ax.legend(frameon=False, fontsize=7)
    _letter(ax, "(c) 设计稀疏度与误差")

    ax = axes[1, 1]
    subset = data[data["gas"] == "NO2"]
    ax.scatter(subset["amplitude"], subset["shape_rms"] * 100.0, s=12, color=GAS_COLORS["NO2"], alpha=0.75)
    selected = correlation[(correlation["gas"] == "NO2") & (correlation["feature"] == "shape_rms") & (correlation["variable"] == "amplitude")]
    rho = float(selected["rho"].iloc[0])
    ax.text(0.03, 0.95, f"rho={rho:+.2f}", transform=ax.transAxes, va="top", fontsize=7)
    ax.set_xlabel("NO₂ 幅度 A")
    ax.set_ylabel("形状残差 RMS / %")
    _letter(ax, "(d) NO₂ 形状随幅度变化")
    figure.tight_layout()
    _save(figure, "fig19_residual_shape")


def figure_protocols() -> None:
    curves = pd.read_csv(VALIDATION / "curve_summary.csv")
    estimates = pd.read_csv(VALIDATION / "estimate_summary.csv")
    macro = curves[curves["gas"] == "macro"]
    width = 0.26
    figure, axes = plt.subplots(1, 4, figsize=(9.6, 3.3), gridspec_kw={"width_ratios": [1.35, 1, 1, 1]})
    short_labels = ["均值", "岭回归", "两阶段", "TCN", "灰盒", "混合"]

    ax = axes[0]
    handles = []
    for position, protocol in enumerate(PROTOCOL_ORDER):
        subset = macro[macro["protocol"] == protocol].set_index("model").reindex(MODEL_ORDER)
        bars = ax.bar(np.arange(len(MODEL_ORDER)) + (position - 1) * width, subset["nrmse_mean"] * 100.0, width, color=PROTOCOL_COLORS[protocol], label=PROTOCOL_LABELS[protocol])
        handles.append(bars)
    ax.set_yscale("log")
    ax.set_yticks([3, 5, 10, 20, 30], ["3", "5", "10", "20", "30"])
    ax.minorticks_off()
    ax.set_xticks(range(len(MODEL_ORDER)), short_labels, rotation=30, ha="right")
    ax.set_ylabel("宏平均 nRMSE / %")
    _letter(ax, "(a) 曲线误差")

    for index, gas in enumerate(GASES):
        ax = axes[index + 1]
        ten = estimates[(estimates["tmax"] == 10.0) & (estimates["gas"] == gas)]
        for position, protocol in enumerate(PROTOCOL_ORDER):
            subset = ten[ten["protocol"] == protocol].set_index("model").reindex(DOWNSTREAM_MODELS)
            ax.bar(np.arange(len(DOWNSTREAM_MODELS)) + (position - 1) * width, subset["median_abs_rel_error"] * 100.0, width, color=PROTOCOL_COLORS[protocol])
        ax.set_yscale("log")
        ticks = {"H2": [4, 5, 7, 10], "NH3": [3, 4, 5], "NO2": [5, 7, 10]}[gas]
        ax.set_yticks(ticks, [str(tick) for tick in ticks])
        ax.minorticks_off()
        ax.set_xticks(range(len(DOWNSTREAM_MODELS)), ["灰盒", "混合", "岭回归"], rotation=30, ha="right")
        _letter(ax, f"({'bcd'[index]}) {GAS_LABEL[gas]}")
    figure.legend(handles, [PROTOCOL_LABELS[protocol] for protocol in PROTOCOL_ORDER], loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=7.5)
    figure.suptitle("留一档位/采集块协议：10 s 窗的曲线与浓度误差", fontsize=9.5)
    figure.tight_layout(rect=(0, 0.08, 1, 0.93))
    _save(figure, "fig20_protocols")




STRUCTURED_COLORS = {"spline": "#8C8C8C", "structured": "#4C72B0"}
STRUCTURED_LABELS = {"spline": "自由 B 样条", "structured": "结构化基"}
BASIS_LABELS = ["快速上升", "4 s 脉冲", "15 s 鼓包", "30 s 鼓包", "晚期漂移"]


def figure_structured_basis(experiments) -> None:
    basis = pd.read_csv(ANALYSIS / "structured_basis.csv")
    time = experiments[0].time
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.2), sharey=True)
    for ax, name, letter in zip(
        axes, ("spline", "structured"), ("(a) 自由 B 样条（当前）", "(b) 结构化基（P3）")
    ):
        subset = basis[basis["basis"] == name]
        for component, group in subset.groupby("component"):
            label = f"φ{component}" if name == "spline" else BASIS_LABELS[component - 1]
            ax.plot(group["time"], group["value"], lw=1.2, label=label)
        for start, end in ((0, 10), (10, 30), (30, 60)):
            ax.axvspan(start, end, color="#F2F2F2" if start != 10 else "#FAFAFA", zorder=0)
        ax.axvline(10, color="#BBBBBB", lw=0.7, ls=":")
        ax.axvline(30, color="#BBBBBB", lw=0.7, ls=":")
        ax.set_xlim(0, time[-1])
        ax.set_xlabel("时间 / s")
        if ax is axes[0]:
            ax.set_ylabel("基函数值（归一化）")
        ax.legend(frameon=True, framealpha=0.9, edgecolor="none", ncol=2, fontsize=6.3, loc="upper right")
        _letter(ax, letter)
    figure.suptitle("残差时间基：自由 B 样条 vs 按物理时段切分的结构化基", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig22_structured_basis")


def figure_structured_residual(experiments) -> None:
    archive = np.load(VALIDATION / "structured_logo_predictions.npz")
    truth = np.stack([experiment.relative for experiment in experiments])
    total = np.sum(truth**2, axis=(0, 1))
    count = (len(experiments) - 1) * truth.shape[1]
    scale = np.sqrt((total[None, :] - np.sum(truth**2, axis=1)) / count)
    files = np.array([experiment.file_id for experiment in experiments])
    figure, axes = plt.subplots(1, 3, figsize=(9.6, 3.1), sharex=True)
    handles = []
    for gas_index, gas in enumerate(GASES):
        ax = axes[gas_index]
        for index in (6.5, 20.5, 34.5, 48.5):
            ax.axvline(index, color="#BBBBBB", lw=0.6, ls=":")
        lines = []
        for variant in ("spline", "structured"):
            prediction = np.stack(
                [archive[f"{variant}_{file}"] for file in files]
            )
            error = prediction - truth
            nrmse = np.sqrt(np.mean(error**2, axis=1)) / scale * 100.0
            rho, _ = spearmanr(files, nrmse[:, gas_index])
            points = ax.scatter(
                files, nrmse[:, gas_index], s=12, alpha=0.75,
                color=STRUCTURED_COLORS[variant], marker="o" if variant == "spline" else "s",
                label=STRUCTURED_LABELS[variant], linewidths=0,
            )
            if gas_index == 0:
                handles.append(points)
            lines.append(f"{'自由' if variant == 'spline' else '结构'} rho={rho:+.2f}")
        ax.set_yscale("log")
        ax.set_xlabel("文件序号")
        ax.set_xticks([0, 10, 20, 30, 40, 50, 60])
        if gas_index == 0:
            ax.set_ylabel("逐文件 nRMSE / %")
        position = (0.98, 0.95) if gas == "H2" else (0.02, 0.95)
        ax.text(*position, "\n".join(lines), transform=ax.transAxes,
                ha="right" if gas == "H2" else "left", va="top", fontsize=6.8)
        _letter(ax, f"({'abc'[gas_index]}) {GAS_LABEL[gas]}")
    figure.legend(handles, [STRUCTURED_LABELS[variant] for variant in ("spline", "structured")],
                  loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False, fontsize=7)
    figure.suptitle("残差-文件序号相关：结构化基是否减少了漂移吸收", fontsize=9.5)
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    _save(figure, "fig23_structured_residual")


def figure_group_coverage() -> None:
    groups = pd.read_csv(ANALYSIS / "coverage_by_group.csv")
    data = groups[(groups["tmax"] == 10.0) & (groups["group"] == "overall")]
    strategies = (
        ("all", "global", "全局"),
        ("NH3", "group", "NH₃ 分档"),
        ("block", "group", "采集块"),
    )
    colors = ("#8C8C8C", "#4C72B0", "#009E73")
    width = 0.26
    positions = np.arange(len(GASES))
    figure, axes = plt.subplots(1, 2, figsize=(8.8, 3.1))
    ax = axes[0]
    for offset, ((group_type, calibration, label), color) in enumerate(zip(strategies, colors)):
        subset = data[
            (data["group_type"] == group_type) & (data["calibration"] == calibration)
        ]
        values = [
            float(subset[subset["gas"] == gas]["median_width_rel"].iloc[0]) * 100.0
            for gas in GASES
        ]
        bars = ax.bar(positions + (offset - 1) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.2, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(positions, [GAS_LABEL[gas] for gas in GASES])
    ax.set_ylabel("90% 区间宽度（中位）/ %")
    ax.legend(frameon=False, ncol=3, fontsize=6.8, loc="upper left")
    _letter(ax, "(a) 区间宽度")

    ax = axes[1]
    for offset, ((group_type, calibration, label), color) in enumerate(zip(strategies, colors)):
        subset = data[
            (data["group_type"] == group_type) & (data["calibration"] == calibration)
        ]
        values = [
            float(subset[subset["gas"] == gas]["coverage"].iloc[0]) * 100.0
            for gas in GASES
        ]
        bars = ax.bar(positions + (offset - 1) * width, values, width, color=color, label=label)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=6.2)
    ax.axhline(90.0, color="#888888", ls=":", lw=0.9, label="名义 90%")
    ax.set_xticks(positions, [GAS_LABEL[gas] for gas in GASES])
    ax.set_ylim(70.0, 100.0)
    ax.set_ylabel("实际覆盖率 / %")
    ax.legend(frameon=False, ncol=2, fontsize=6.8, loc="lower right")
    _letter(ax, "(b) 覆盖率")
    figure.suptitle("分组共形：覆盖率接近名义值，但整体区间宽度没有下降", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig24_group_coverage")


def figure_bias_estimation() -> None:
    summary = pd.read_csv(ANALYSIS / "bias_summary.csv")
    bias = pd.read_csv(ANALYSIS / "bias_estimates.csv")
    median = summary.pivot(index="gas", columns="weighting", values="median_abs_rel_error")
    coverage = summary.pivot(index="gas", columns="weighting", values="coverage_laplace90")
    labels = {"scalar": "标量 Laplace", "covariance": "偏差协方差"}
    colors = {"scalar": "#8C8C8C", "covariance": "#4C72B0"}
    positions = np.arange(len(GASES))
    width = 0.34
    figure, axes = plt.subplots(1, 3, figsize=(9.8, 3.1))
    ax = axes[0]
    for offset, weighting in enumerate(("scalar", "covariance")):
        values = [float(median.loc[gas, weighting]) * 100.0 for gas in GASES]
        bars = ax.bar(positions + (offset - 0.5) * width, values, width,
                      color=colors[weighting], label=labels[weighting])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.12, f"{value:.2f}",
                    ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(positions, [GAS_LABEL[gas] for gas in GASES])
    ax.set_ylabel("10 s 中位绝对相对误差 / %")
    ax.legend(frameon=False, fontsize=6.8)
    _letter(ax, "(a) 点误差")

    ax = axes[1]
    for offset, weighting in enumerate(("scalar", "covariance")):
        values = [float(coverage.loc[gas, weighting]) * 100.0 for gas in GASES]
        bars = ax.bar(positions + (offset - 0.5) * width, values, width,
                      color=colors[weighting], label=labels[weighting])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.0, f"{value:.1f}",
                    ha="center", va="bottom", fontsize=6.2)
    ax.axhline(90.0, color="#888888", ls=":", lw=0.9, label="名义 90%")
    ax.set_xticks(positions, [GAS_LABEL[gas] for gas in GASES])
    ax.set_ylim(0.0, 110.0)
    ax.set_ylabel("Laplace 90% 覆盖率 / %")
    ax.legend(
        frameon=False, fontsize=6.6, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
    )
    _letter(ax, "(b) 区间覆盖")

    ax = axes[2]
    table = bias.pivot_table(
        index=["file", "gas"], columns="weighting", values="abs_rel_error"
    ).reset_index()
    for gas in GASES:
        subset = table[table["gas"] == gas]
        ax.scatter(subset["scalar"] * 100.0, subset["covariance"] * 100.0,
                   s=10, alpha=0.65, linewidths=0, color=GAS_COLORS[gas],
                   label=GAS_LABEL[gas])
    limits = (0.3, 60.0)
    ax.plot(limits, limits, color="#444444", lw=0.9, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    ax.set_yticks([0.5, 1, 2, 5, 10, 20, 50], ["0.5", "1", "2", "5", "10", "20", "50"])
    ax.minorticks_off()
    improved = int((table["covariance"] < table["scalar"]).sum())
    ax.text(0.04, 0.96, f"{improved}/186 改善", transform=ax.transAxes,
            ha="left", va="top", fontsize=7)
    ax.set_xlabel("标量 Laplace 误差 / %")
    ax.set_ylabel("偏差协方差误差 / %")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    _letter(ax, "(c) 逐文件对比")
    figure.suptitle("偏差协方差 MAP：点误差下降，Laplace 区间由欠覆盖转为偏保守", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig25_bias_estimation")


def figure_stopping_decision() -> None:
    table = pd.read_csv(ANALYSIS / "stopping_decision.csv")
    summary = pd.read_csv(ANALYSIS / "stopping_decision_summary.csv").set_index("gas")
    figure, axes = plt.subplots(1, 2, figsize=(8.6, 3.1))
    ax = axes[0]
    grid = np.array([2.0, 5.0, 10.0, 20.0, 60.0])
    for gas in GASES:
        subset = table[table["gas"] == gas]
        decided = subset["decided"].to_numpy(dtype=bool)
        times = subset["stop_time"].to_numpy(dtype=float)
        fraction = [np.mean(decided & (times <= limit)) * 100.0 for limit in grid]
        ax.plot(grid, fraction, "-o", ms=3.5, color=GAS_COLORS[gas], label=GAS_LABEL[gas])
    ax.axvline(10.0, color="#888888", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 60], ["2", "5", "10", "20", "60"])
    ax.minorticks_off()
    ax.set_ylim(0.0, 100.0)
    ax.set_xlabel("决策时间上限 / s")
    ax.set_ylabel("已决定实验比例 / %")
    ax.legend(frameon=False)
    _letter(ax, "(a) 停止时间")

    ax = axes[1]
    order = [gas for gas in GASES if gas in summary.index]
    positions = np.arange(len(order))
    within_correct = np.array([summary.loc[gas, "correct_by_target"] for gas in order], dtype=float)
    within_false = np.array([summary.loc[gas, "false_by_target"] for gas in order], dtype=float)
    late = np.array(
        [summary.loc[gas, "decided"] - summary.loc[gas, "decided_by_target"] for gas in order],
        dtype=float,
    )
    undecided = np.array([summary.loc[gas, "n"] - summary.loc[gas, "decided"] for gas in order], dtype=float)
    bottom = np.zeros(len(order))
    parts = (
        (within_correct, "10 s 内正确", "#009E73"),
        (within_false, "10 s 内误判", "#C0392B"),
        (late, "10 s 后决定", "#8C8C8C"),
        (undecided, "未决", "#DDDDDD"),
    )
    for values, label, color in parts:
        ax.bar(positions, values, bottom=bottom, color=color, label=label, width=0.6)
        bottom += values
    for position, gas in zip(positions, order):
        ax.text(position, summary.loc[gas, "n"] + 1.5,
                f"{int(summary.loc[gas, 'decided_by_target'])}/{int(summary.loc[gas, 'n'])} 决定",
                ha="center", fontsize=6.5)
    ax.set_xticks(positions, [GAS_LABEL[gas] for gas in order])
    ax.set_ylim(0.0, 75.0)
    ax.set_ylabel("实验数")
    ax.legend(
        frameon=False, fontsize=6.4, ncol=2,
        loc="upper center", bbox_to_anchor=(0.5, -0.16),
    )
    _letter(ax, "(b) 决策结果（10 s 上限）")
    figure.suptitle("决策停止：10 s 内多数实验可判定，H₂ 的误判率最高", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig26_stopping_decision")


def figure_fair_augmentation() -> None:
    comparison = pd.read_csv(AUGMENTATION / "fair_comparison.csv")
    summary = pd.read_csv(AUGMENTATION / "fair_summary.csv")
    arms = ("real", "real+synthetic-equal", "pretrain+finetune-equal")
    labels = {
        "real": "仅真实\n(8/步)",
        "real+synthetic-equal": "等曝光\n(8+8/步)",
        "pretrain+finetune-equal": "等真实预算\n(100+150 epoch)",
    }
    short = {
        "real+synthetic-equal": "等曝光",
        "pretrain+finetune-equal": "等真实预算",
    }
    colors = {
        "real": "#8C8C8C",
        "real+synthetic-equal": "#4C72B0",
        "pretrain+finetune-equal": "#009E73",
    }
    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))
    rng = np.random.default_rng(0)
    ax = axes[0]
    subset = comparison[comparison["tmax"] == 10.0]
    per_seed = subset.groupby(["seed", "arm"])["mape_macro"].mean().unstack()
    positions = np.arange(len(arms))
    means = per_seed[list(arms)].mean().to_numpy()
    deviations = per_seed[list(arms)].std(ddof=1).to_numpy()
    bars = ax.bar(
        positions,
        means,
        yerr=deviations,
        capsize=3,
        color=[colors[arm] for arm in arms],
        width=0.62,
        error_kw={"elinewidth": 0.8, "ecolor": "#444444"},
    )
    for position, arm in zip(positions, arms):
        ax.scatter(
            position + rng.uniform(-0.11, 0.11, size=len(per_seed)),
            per_seed[arm],
            s=12,
            color="white",
            edgecolor="#333333",
            linewidth=0.5,
            zorder=3,
        )
    for bar, value, deviation in zip(bars, means, deviations):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + deviation + 0.10,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
        )
    ax.set_xticks(positions, [labels[arm] for arm in arms], fontsize=6.6)
    ax.set_ylabel("10 s 宏平均 MAPE / %")
    ax.set_ylim(0.0, max(means + deviations) * 1.12 + 0.3)
    _letter(ax, "(a) 宏平均相对误差")

    ax = axes[1]
    gases = ("mape_H2", "mape_NH3", "mape_NO2")
    gas_labels = ("H₂", "NH₃", "NO₂")
    width = 0.34
    for offset, arm in enumerate(arms[1:]):
        for index, metric in enumerate(gases):
            row = summary[
                (summary["tmax"] == 10.0)
                & (summary["arm"] == arm)
                & (summary["metric"] == metric)
            ].iloc[0]
            delta = float(row["delta_mean"])
            low, high = float(row["ci_low"]), float(row["ci_high"])
            position = index + (offset - 0.5) * width
            ax.bar(
                position,
                delta,
                width=width,
                color=colors[arm],
                label=short[arm] if index == 0 else None,
            )
            ax.errorbar(
                position,
                delta,
                yerr=[[delta - low], [high - delta]],
                fmt="none",
                ecolor="#333333",
                elinewidth=0.8,
                capsize=2.5,
            )
    ax.axhline(0.0, color="#444444", lw=0.9, ls="--")
    ax.set_xticks(np.arange(len(gases)), gas_labels)
    ax.set_ylabel("相对真实的 ΔMAPE / 百分点")
    ax.legend(frameon=False, fontsize=6.4, loc="upper right")
    _letter(ax, "(b) 10 s 分气体")

    ax = axes[2]
    subset = summary[(summary["metric"] == "mape_macro") & (summary["arm"] != "real")]
    for offset, tmax in enumerate((5.0, 10.0)):
        for index, arm in enumerate(arms[1:]):
            row = subset[(subset["arm"] == arm) & (subset["tmax"] == tmax)].iloc[0]
            delta = float(row["delta_mean"])
            low, high = float(row["ci_low"]), float(row["ci_high"])
            ax.errorbar(
                index + (offset - 0.5) * 0.3,
                delta,
                yerr=[[delta - low], [high - delta]],
                fmt="o" if tmax == 5.0 else "s",
                ms=4,
                capsize=3,
                lw=1.0,
                color=colors[arm],
                label=f"{int(tmax)} s" if index == 0 else None,
            )
    ax.axhline(0.0, color="#444444", lw=0.9, ls="--")
    ax.set_xticks(np.arange(len(arms) - 1), [short[arm] for arm in arms[1:]], fontsize=6.8)
    ax.set_ylabel("宏平均 ΔMAPE / 百分点")
    ax.legend(frameon=False, fontsize=6.8, title="时间窗", title_fontsize=6.8)
    _letter(ax, "(c) 配对差与 95% CI")
    figure.suptitle("公平 B：宏平均改善，但 H₂ 在等曝光臂变差", fontsize=9.5)
    figure.tight_layout()
    _save(figure, "fig27_fair_augmentation")


def main() -> None:
    _style()
    experiments = load_experiments()
    model = HybridArrayModel.load(MODELS / "hybrid.json")
    estimate = None
    path = ANALYSIS / "estimates.csv"
    if path.exists():
        estimate = pd.read_csv(path)
    figure_hybrid(experiments, model)
    figure_crlb(estimate)
    figure_identifiability_map(experiments)
    if estimate is not None:
        figure_error_map(estimate, experiments)
        figure_early_error(estimate)
        figure_coverage(pd.read_csv(ANALYSIS / "conformal.csv"))
        figure_stopping(pd.read_csv(ANALYSIS / "stopping.csv"))
    if (ANALYSIS / "degradation.csv").exists():
        figure_degradation()
    if (ANALYSIS / "error_budget.csv").exists():
        figure_budget()
    if (ANALYSIS / "residual_matrix.npz").exists():
        figure_residual_map()
    if (ANALYSIS / "residual_pca_modes.csv").exists():
        figure_pca()
    if (ANALYSIS / "residual_diagnostics.csv").exists():
        figure_shape()
    if (VALIDATION / "curve_summary.csv").exists() and (VALIDATION / "estimate_summary.csv").exists():
        figure_protocols()
    if (ANALYSIS / "structured_basis.csv").exists():
        figure_structured_basis(experiments)
    if (VALIDATION / "structured_logo_predictions.npz").exists():
        figure_structured_residual(experiments)
    if (ANALYSIS / "coverage_by_group.csv").exists():
        figure_group_coverage()
    if (ANALYSIS / "bias_summary.csv").exists() and (ANALYSIS / "bias_estimates.csv").exists():
        figure_bias_estimation()
    if (ANALYSIS / "stopping_decision.csv").exists() and (
        ANALYSIS / "stopping_decision_summary.csv"
    ).exists():
        figure_stopping_decision()
    if (AUGMENTATION / "fair_comparison.csv").exists() and (
        AUGMENTATION / "fair_summary.csv"
    ).exists():
        figure_fair_augmentation()


if __name__ == "__main__":
    main()
