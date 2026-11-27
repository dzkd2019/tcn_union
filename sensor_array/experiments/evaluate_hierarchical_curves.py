"""评估带实验级随机效应的正模型：nRMSE 与 62 条曲线对比。

这里做的是“后验重构”：先在全部 62 条上拟合一阶/双指数骨架 + MLP 均值，
再为每条实验拟合 (gain, time-scale, drift) 三个随机效应，用拟合值重构曲线。
它是新模型在已知实验条件下的拟合上界，不是对未见实验的留一误差。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ..core.data import GASES, load_experiments
from ..core.hierarchical import HierarchicalRandomEffectsModel
from ..core.metrics import curve_metrics
from ..core.mlp_per_gas import PerGasMLPDoubleExponentialModel
from ..paths import DATA, EVALUATION, FIGURES_MODEL, MODELS, ensure
from ..plotting.style import GAS_COLORS, GAS_LABEL, _save, _style


def _reconstruct(model, hierarchical, experiments) -> np.ndarray:
    labels = np.stack([item.concentration for item in experiments])
    times = experiments[0].time
    baseline = model.predict_relative(labels, times)
    effects = hierarchical.effects_.reshape(len(experiments), 3, 3)
    prediction = np.empty_like(baseline)
    for gas in range(3):
        gain = np.exp(effects[:, gas, 0])
        scale = np.exp(effects[:, gas, 1])
        drift = effects[:, gas, 2]
        for index in range(len(experiments)):
            prediction[index, :, gas] = (
                gain[index]
                * np.interp(times / scale[index], times, baseline[index, :, gas])
                + drift[index] * times
            )
    return prediction


def _per_experiment(prediction, truth, scale) -> pd.DataFrame:
    rows = []
    for index in range(len(truth)):
        row = {"file": index}
        row.update(curve_metrics(prediction[index], truth[index], scale))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_curves(experiments, truth, previous, current, current_metrics) -> None:
    _style()
    figure, axes = plt.subplots(3, 1, figsize=(7.4, 8.2), sharex=True)
    for gas_index, gas in enumerate(GASES):
        axis = axes[gas_index]
        for index, experiment in enumerate(experiments):
            axis.plot(
                experiment.time,
                truth[index, :, gas_index] * 100,
                color="#7F7F7F",
                lw=0.45,
                alpha=0.28,
            )
        for index, experiment in enumerate(experiments):
            axis.plot(
                experiment.time,
                current[index, :, gas_index] * 100,
                color=GAS_COLORS[gas],
                lw=0.55,
                alpha=0.55,
            )
        axis.set_ylabel(f"{GAS_LABEL[gas]} 相对响应 / %")
        axis.grid(alpha=0.25)
        axis.text(
            0.99,
            0.93,
            f"新模型 nRMSE 中位 {current_metrics[f'current_{gas}'].median() * 100:.2f}%\n"
            f"旧模型 中位 {current_metrics[f'previous_{gas}'].median() * 100:.2f}%",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.8),
        )
    axes[-1].set_xlabel("时间 / s")
    handles = [
        plt.Line2D([], [], color="#7F7F7F", lw=1.0, alpha=0.5, label="实测（62 条）"),
        plt.Line2D([], [], color="#333333", lw=1.0, label="重构（各气体颜色）"),
    ]
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )
    figure.suptitle(
        "实验级随机效应正模型：62 条重构曲线与实测曲线", fontsize=10, y=0.99
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.955))
    _save(figure, "fig12_hierarchical_62_curves")


def _plot_compare(table: pd.DataFrame) -> None:
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.8), sharey=False)
    for axis, gas in zip(axes, GASES):
        previous = table[f"nrmse_{gas}"] * 100
        current = table[f"current_{gas}"] * 100
        axis.scatter(previous, current, s=12, color=GAS_COLORS[gas], alpha=0.75)
        low = min(previous.min(), current.min()) * 0.9
        high = max(previous.max(), current.max()) * 1.05
        axis.plot([low, high], [low, high], color="#555555", lw=0.7, ls="--")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel("旧模型 nRMSE / %")
        axis.set_title(GAS_LABEL[gas])
        axis.grid(alpha=0.25)
        improved = int((current < previous).sum())
        axis.text(
            0.04,
            0.96,
            f"改善 {improved}/62\n中位 {current.median():.2f}%",
            transform=axis.transAxes,
            va="top",
            fontsize=7,
        )
    axes[0].set_ylabel("新模型 nRMSE / %")
    figure.suptitle("逐实验 nRMSE：新模型 vs 旧模型", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    _save(figure, "fig13_hierarchical_nrmse_compare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument(
        "--model", default=str(MODELS / "mlp_per_gas.json"), help="旧正模型路径"
    )
    args = parser.parse_args()
    ensure()
    experiments = load_experiments(Path(args.data))
    model = PerGasMLPDoubleExponentialModel.load(Path(args.model))
    truth = np.stack([item.relative for item in experiments])
    labels = np.stack([item.concentration for item in experiments])
    scale = np.sqrt(np.mean(truth**2, axis=(0, 1)))

    previous = model.predict_relative(labels, experiments[0].time)
    hierarchical = HierarchicalRandomEffectsModel().fit(model, experiments, 10)
    current = _reconstruct(model, hierarchical, experiments)

    previous_table = _per_experiment(previous, truth, scale)
    current_table = _per_experiment(current, truth, scale)
    table = pd.DataFrame({"file": previous_table["file"]})
    for gas in GASES:
        table[f"nrmse_{gas}"] = previous_table[f"nrmse_{gas}"]
        table[f"current_{gas}"] = current_table[f"nrmse_{gas}"]
        table[f"previous_{gas}"] = previous_table[f"nrmse_{gas}"]
    table.to_csv(EVALUATION / "hierarchical_curves.csv", index=False)
    np.savez_compressed(
        EVALUATION / "hierarchical_curve_predictions.npz",
        previous=previous,
        current=current,
        truth=truth,
    )

    print("旧模型 in-sample nRMSE（%）")
    for gas in GASES:
        print(f"  {gas}: {previous_table[f'nrmse_{gas}'].mean() * 100:.3f}")
    print(f"  macro: {previous_table['nrmse_macro'].mean() * 100:.3f}")
    print("新模型（拟合实验级随机效应）in-sample nRMSE（%）")
    for gas in GASES:
        print(f"  {gas}: {current_table[f'nrmse_{gas}'].mean() * 100:.3f}")
    print(f"  macro: {current_table['nrmse_macro'].mean() * 100:.3f}")

    _plot_curves(experiments, truth, previous, current, table)
    _plot_compare(table)
    print("saved", FIGURES_MODEL)


if __name__ == "__main__":
    main()
