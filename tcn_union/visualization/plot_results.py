"""从评估产物绘制逐气体预测与误差对比图。

只读取 evaluate.py 写出的测试预测和汇总报告，不重新拟合模型或计算测试指标；
所有图均按气体拆分，避免不同浓度量级混在同一坐标轴而难以解释。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]

import numpy as np
import pandas as pd

GASES = ["H2", "NH3", "NO2"]
WINDOWS = [5, 10]
TCN_SEEDS = [20260923, 20260924, 20260925]
WINDOW_COLORS = {5: "#0072B2", 10: "#D55E00"}


def _load_rows(path: Path) -> list[dict[str, str]]:
    """读取 CSV 为字典行列表。

    参数：`path` 是评估 CSV 路径。
    返回：每行一个字符串值字典的列表；保持 CSV 原始字段，后续绘图再按需转数值。
    """
    return pd.read_csv(
        path, dtype=str, keep_default_na=False, encoding="utf-8"
    ).to_dict(orient="records")


def _primary_row(
    rows: list[dict[str, str]], seconds: int, model: str, seed: str = ""
) -> dict[str, dict[str, str]]:
    """筛选某个窗口/模型/种子的逐实验预测。

    参数：`rows` 是测试预测 CSV 行；`seconds`、`model`、`seed` 指定要取的
    预测组，其中默认空 seed 用于无随机种子的基线。
    返回：以实验文件名为键的预测行映射。必须恰有 12 条，防止缺样本时生成
    外观正常但不完整的测试图。
    """
    matches = [
        row
        for row in rows
        if int(row["window_seconds"]) == seconds
        and row["model"] == model
        and row["seed"] == seed
    ]
    if len(matches) != 12:
        raise ValueError(
            f"expected 12 test predictions for {seconds}s {model} seed={seed}, found {len(matches)}"
        )
    return {row["file"]: row for row in matches}


def _load_inputs(
    results_dir: Path,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """加载并核对绘图所需的两份评估产物。

    参数：`results_dir` 包含 test_predictions.csv 和 evaluation.json。
    返回：`(预测行, 汇总报告)`；要求完整的 12 个测试实验及预期 144 条记录
    （两窗口下三种基线加三种子 TCN），避免基于部分评估数据绘图。
    """
    predictions = _load_rows(results_dir / "test_predictions.csv")
    report = json.loads(
        (results_dir / "evaluation.json").read_text(encoding="utf-8")
    )
    if len(predictions) != 144 or report.get("test_n") != 12:
        raise ValueError(
            "expected completed evaluation artifacts for 12 test runs"
        )
    return predictions, report


def _style() -> None:
    """设置本模块所有 Matplotlib 图的统一字体和视觉参数。"""
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 180,
        }
    )


def plot_predicted_vs_true(
    predictions: list[dict[str, str]], output: Path
) -> None:
    """绘制主种子 TCN 在两个窗口下的预测值-实测值散点图。

    参数：`predictions` 是逐实验预测行；`output` 是目标 PNG 路径。
    返回：无。H2/NH3/NO2 分面展示，并以 y=x 参考线呈现偏差；相同横纵坐标
    范围及等比例坐标便于识别系统偏差和离散程度。
    """
    primary = {
        seconds: _primary_row(predictions, seconds, "TCN", "20260923")
        for seconds in WINDOWS
    }
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    markers = {5: "o", 10: "s"}
    for axis, gas in zip(axes, GASES, strict=True):
        values = []
        for seconds in WINDOWS:
            rows = list(primary[seconds].values())
            truth = np.array([float(row[f"{gas}_true"]) for row in rows])
            predicted = np.array([float(row[f"{gas}_pred"]) for row in rows])
            values.extend(
                (truth.min(), truth.max(), predicted.min(), predicted.max())
            )
            axis.scatter(
                truth,
                predicted,
                s=43,
                marker=markers[seconds],
                color=WINDOW_COLORS[seconds],
                edgecolor="white",
                linewidth=0.5,
                alpha=0.9,
                label=f"{seconds} s",
                zorder=3,
            )
        low, high = min(values), max(values)
        padding = max((high - low) * 0.06, abs(high) * 0.01, 1e-6)
        limits = (low - padding, high + padding)
        axis.plot(
            limits,
            limits,
            color="#444444",
            linestyle="--",
            linewidth=1,
            label="ideal",
        )
        axis.set_xlim(limits)
        axis.set_ylim(limits)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(gas)
        axis.set_xlabel("实际气体浓度(ppm)")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("预测气体浓度(ppm)")
    axes[0].legend(loc="upper left", frameon=False)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_metric_comparison(
    report: dict[str, Any], output: Path, metric_key: str, title: str
) -> None:
    """按气体绘制所选测试集指标的基线与 TCN 对比。

    参数：`report` 是 evaluation.json 解析出的报告；`output` 是 PNG 路径；
    `metric_key` 只允许 NMAE 或 MAPE_percent；`title` 是整图标题。
    返回：无。柱显示均值/基线和主种子结果，额外 TCN 种子以菱形标记，展示
    随机初始化波动；逐气体分面避免不同浓度尺度互相遮蔽。
    """
    if metric_key not in {"NMAE", "MAPE_percent"}:
        raise ValueError("metric_key must be NMAE or MAPE_percent")
    metric_rows = report["test_metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    labels = [
        "Mean",
        "Ridge\nt=0",
        "Ridge\nearly 5s",
        "Ridge\nearly 10s",
        "TCN\n5s",
        "TCN\n10s",
    ]
    metric_specs = [
        (5, "label_mean", ""),
        (5, "ridge_t0", ""),
        (5, "ridge_early", ""),
        (10, "ridge_early", ""),
        (5, "TCN", "20260923"),
        (10, "TCN", "20260923"),
    ]
    for axis, gas in zip(axes, GASES, strict=True):
        metric_records = [
            next(
                item
                for item in metric_rows
                if item["window_seconds"] == seconds
                and item["model"] == model
                and str(item["seed"]) == seed
            )
            for seconds, model, seed in metric_specs
        ]
        scores = [
            np.nan
            if row[gas][metric_key] is None
            else float(row[gas][metric_key])
            for row in metric_records
        ]
        x = np.arange(len(labels))
        colors = ["#8C8C8C"] * 4 + [WINDOW_COLORS[5], WINDOW_COLORS[10]]
        axis.bar(
            x,
            scores,
            color=colors,
            alpha=0.83,
            width=0.68,
            edgecolor="white",
            linewidth=0.6,
        )
        for position, seconds in ((4, 5), (5, 10)):
            for seed in TCN_SEEDS[1:]:
                seed_row = next(
                    item
                    for item in metric_rows
                    if item["window_seconds"] == seconds
                    and item["model"] == "TCN"
                    and item["seed"] == seed
                )
                value = seed_row[gas][metric_key]
                if value is not None:
                    offset = -0.08 if seed == TCN_SEEDS[1] else 0.08
                    axis.scatter(
                        position + offset,
                        float(value),
                        marker="D",
                        s=27,
                        color=WINDOW_COLORS[seconds],
                        edgecolor="black",
                        linewidth=0.45,
                        zorder=4,
                    )
        axis.set_title(gas)
        axis.set_ylabel(
            "NMAE (MAE / gas range)" if metric_key == "NMAE" else "MAPE (%)"
        )
        axis.set_xticks(x, labels, rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title, y=1.05)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_tcn_metric_comparison(
    report: dict[str, Any], output: Path, metric_key: str, title: str
) -> None:
    """绘制 TCN 在两个预测窗口下的逐气体测试指标，不显示均值或 Ridge 基线。

    参数：`report` 是 evaluation.json 解析出的报告；`output` 是 PNG 路径；
    `metric_key` 只允许 NMAE 或 MAPE_percent；`title` 是整图标题。
    返回：无。主种子 20260923 用柱显示，其余两个 TCN 种子用菱形标记。
    """
    if metric_key not in {"NMAE", "MAPE_percent"}:
        raise ValueError("metric_key must be NMAE or MAPE_percent")
    metric_rows = report["test_metrics"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    labels = [f"TCN\n{seconds}s" for seconds in WINDOWS]
    x = np.arange(len(WINDOWS))
    for axis, gas in zip(axes, GASES, strict=True):
        primary_records = [
            next(
                item
                for item in metric_rows
                if item["window_seconds"] == seconds
                and item["model"] == "TCN"
                and str(item["seed"]) == str(TCN_SEEDS[0])
            )
            for seconds in WINDOWS
        ]
        scores = [
            np.nan
            if row[gas][metric_key] is None
            else float(row[gas][metric_key])
            for row in primary_records
        ]
        axis.bar(
            x,
            scores,
            color=[WINDOW_COLORS[seconds] for seconds in WINDOWS],
            alpha=0.86,
            width=0.62,
            edgecolor="white",
            linewidth=0.6,
            label=f"Primary seed {TCN_SEEDS[0]}",
        )
        for position, seconds in enumerate(WINDOWS):
            for seed in TCN_SEEDS[1:]:
                seed_row = next(
                    item
                    for item in metric_rows
                    if item["window_seconds"] == seconds
                    and item["model"] == "TCN"
                    and item["seed"] == seed
                )
                value = seed_row[gas][metric_key]
                if value is not None:
                    label = (
                        "Other TCN seeds"
                        if gas == GASES[0]
                        and position == 0
                        and seed == TCN_SEEDS[1]
                        else "_nolegend_"
                    )
                    offset = -0.08 if seed == TCN_SEEDS[1] else 0.08
                    axis.scatter(
                        position + offset,
                        float(value),
                        marker="D",
                        s=30,
                        color=WINDOW_COLORS[seconds],
                        edgecolor="black",
                        linewidth=0.45,
                        zorder=4,
                        label=label,
                    )
        axis.set_title(gas)
        axis.set_ylabel(
            "NMAE (MAE / gas range)" if metric_key == "NMAE" else "MAPE (%)"
        )
        axis.set_xticks(x, labels)
        axis.grid(axis="y", alpha=0.25)
        if gas == GASES[0]:
            axis.legend(frameon=False, fontsize=8)
    fig.suptitle(title, y=1.05)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_residuals(
    predictions: list[dict[str, str]], report: dict[str, Any], output: Path
) -> None:
    """绘制主种子 TCN 的逐实验归一化有符号残差热图。

    参数：`predictions` 是逐文件预测；`report` 提供开发集气体极差；
    `output` 是 PNG 保存路径。
    返回：无。颜色表示 `(prediction-truth)/development_range`，按实验和
    气体/窗口展开，保留误差方向并便于发现个别实验或气体上的系统偏差。
    """
    dev_ranges = report["development_ranges"]
    primary = {
        seconds: _primary_row(predictions, seconds, "TCN", "20260923")
        for seconds in WINDOWS
    }
    files = list(primary[5])
    columns = [(gas, seconds) for gas in GASES for seconds in WINDOWS]
    matrix = np.zeros((len(files), len(columns)), dtype=np.float64)
    for row_index, filename in enumerate(files):
        for col_index, (gas, seconds) in enumerate(columns):
            row = primary[seconds][filename]
            matrix[row_index, col_index] = (
                float(row[f"{gas}_pred"]) - float(row[f"{gas}_true"])
            ) / float(dev_ranges[gas])
    limit = max(float(np.max(np.abs(matrix))), 1e-6)
    fig, axis = plt.subplots(figsize=(8.6, 6.2), constrained_layout=True)
    image = axis.imshow(
        matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto"
    )
    axis.set_yticks(np.arange(len(files)), [Path(name).stem for name in files])
    axis.set_xticks(
        np.arange(len(columns)),
        [f"{gas}\n{seconds}s" for gas, seconds in columns],
    )
    axis.set_xlabel("Gas and prediction window")
    axis.set_ylabel("Test experiment file ID")
    axis.set_title(
        "TCN normalized residuals (prediction − truth) / development range"
    )
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            color = (
                "white"
                if abs(matrix[row_index, col_index]) > 0.55 * limit
                else "black"
            )
            axis.text(
                col_index,
                row_index,
                f"{matrix[row_index, col_index]:+.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.86)
    colorbar.set_label("Normalized signed error")
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def make_figures(results_dir: Path, output_dir: Path) -> list[Path]:
    """加载评估结果并生成全部四张图。

    参数：`results_dir` 是评估 CSV/JSON 所在目录；`output_dir` 是图像输出目录。
    返回：按生成顺序排列的四个 PNG 路径。统一先校验输入完整性，再绘制实测-
    预测、逐气体 NMAE、逐气体 MAPE 和归一化残差图。
    """
    predictions, report = _load_inputs(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _style()
    outputs = [
        output_dir / "test_predicted_vs_true.png",
        output_dir / "test_nmae_by_gas.png",
        output_dir / "test_mape_by_gas.png",
        output_dir / "test_normalized_residuals.png",
    ]
    plot_predicted_vs_true(predictions, outputs[0])
    plot_tcn_metric_comparison(
        report,
        outputs[1],
        "NMAE",
        "Locked test set: TCN NMAE by gas (n=12)",
    )
    plot_tcn_metric_comparison(
        report,
        outputs[2],
        "MAPE_percent",
        "Locked test set: TCN MAPE by gas (n=12)",
    )

    plot_normalized_residuals(predictions, report, outputs[3])
    return outputs


def main() -> None:
    """解析绘图 CLI 路径并打印生成的图像文件名。"""
    parser = argparse.ArgumentParser(
        description="Plot locked test-set TCN results"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("artifacts/evaluation")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/evaluation/figures")
    )
    args = parser.parse_args()
    for path in make_figures(args.results_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
