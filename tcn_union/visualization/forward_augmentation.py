"""正模型增强 TCN 的汇总图（fig16/fig17）。"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sensor_array.plotting.style import _save, _style

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/forward_augmentation"
ARMS = ("real", "forward", "pca_vae", "forward+pca_vae")
LABELS = {
    "real": "仅真实",
    "forward": "正模型",
    "pca_vae": "PCA-VAE",
    "forward+pca_vae": "正模型 + PCA-VAE",
}
COLORS = {
    "real": "#7F7F7F",
    "forward": "#4C72B0",
    "pca_vae": "#E69F00",
    "forward+pca_vae": "#009E73",
}
WINDOWS = (5, 10)


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bars(
    axes,
    summary: list[dict],
    oof: list[dict],
    metric: str,
    title: str,
    window: int,
) -> None:
    x = np.arange(len(ARMS))
    width = 0.72
    for index, arm in enumerate(ARMS):
        selected = [row for row in summary if row["arm"] == arm]
        mean = float(selected[0][f"{metric}_mean"])
        sd = float(selected[0][f"{metric}_sd"])
        axes.bar(
            index,
            mean,
            width,
            color=COLORS[arm],
            alpha=0.88,
            yerr=sd,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        seeds = [
            float(row[metric])
            for row in oof
            if row["arm"] == arm and int(row["window"]) == window
        ]
        offsets = np.linspace(-0.17, 0.17, len(seeds))
        axes.scatter(
            index + offsets,
            seeds,
            s=10,
            facecolors="white",
            edgecolors="#222222",
            linewidths=0.55,
            zorder=3,
        )
        axes.text(
            index,
            mean + sd + 0.02 * max(1.0, mean),
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    axes.set_xticks(x, [LABELS[arm] for arm in ARMS], rotation=12, ha="right")
    axes.set_ylabel(title)
    axes.grid(axis="y", color="#DDDDDD", linewidth=0.5, alpha=0.7)
    axes.set_axisbelow(True)


def _paired(axes, folds: list[dict], arm: str, window: int) -> None:
    real = [
        float(row["macro_three_sigma"])
        for row in folds
        if row["arm"] == "real" and int(row["window"]) == window
    ]
    other = [
        float(row["macro_three_sigma"])
        for row in folds
        if row["arm"] == arm and int(row["window"]) == window
    ]
    if len(real) != len(other):
        raise ValueError("real 与增强臂的折数不一致")
    axes.scatter(real, other, s=13, color=COLORS[arm], alpha=0.75)
    low = min(min(real), min(other))
    high = max(max(real), max(other))
    pad = 0.06 * (high - low)
    axes.plot(
        [low - pad, high + pad],
        [low - pad, high + pad],
        color="#555555",
        linewidth=0.8,
        linestyle=(0, (4, 3)),
    )
    axes.set_xlim(low - pad, high + pad)
    axes.set_ylim(low - pad, high + pad)
    axes.set_xlabel("仅真实 宏 3σ / %")
    axes.set_ylabel(f"{LABELS[arm]} 宏 3σ / %")
    axes.grid(color="#E5E5E5", linewidth=0.5, alpha=0.7)
    axes.set_axisbelow(True)
    difference = np.asarray(other) - np.asarray(real)
    better = int(np.sum(difference < 0))
    axes.text(
        0.05,
        0.94,
        f"{better}/{len(difference)} 改善\n中位差 {np.median(difference):+.2f} 百分点",
        transform=axes.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color="#222222",
    )


def figure_metrics() -> None:
    _style()
    summary = _rows(BASE / "summary.csv")
    oof = _rows(BASE / "oof.csv")
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 6.2), layout="constrained")
    for row, window in enumerate(WINDOWS):
        selected = [item for item in summary if int(item["window"]) == window]
        _bars(
            axes[row, 0],
            selected,
            oof,
            "macro_three_sigma",
            "宏 3σ / %",
            window,
        )
        _bars(
            axes[row, 1],
            selected,
            oof,
            "macro_mape",
            "宏 MAPE / %",
            window,
        )
        axes[row, 0].set_title(f"({chr(97 + 2 * row)}) {window} s：宏 3σ")
        axes[row, 1].set_title(f"({chr(98 + 2 * row)}) {window} s：宏 MAPE")
    figure.suptitle("正模型增强 TCN：3 seed × 5 折 OOF，柱为均值、空心点为各 seed")
    _save(figure, "fig16_forward_augmentation")


def figure_paired() -> None:
    _style()
    folds = _rows(BASE / "folds.csv")
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 6.0), layout="constrained")
    for row, window in enumerate(WINDOWS):
        for column, arm in enumerate(ARMS[1:]):
            _paired(axes[row, column], folds, arm, window)
            axes[row, column].set_title(
                f"({chr(97 + 3 * row + column)}) {window} s：{LABELS[arm]}"
            )
    figure.suptitle("逐折配对：横轴仅真实，纵轴增强臂；点在对角线下方为改善")
    _save(figure, "fig17_forward_augmentation_paired")


def main() -> None:
    figure_metrics()
    figure_paired()


if __name__ == "__main__":
    main()
