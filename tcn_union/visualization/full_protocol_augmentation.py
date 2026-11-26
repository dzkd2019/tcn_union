"""完整旧 TCN 协议四臂对照图（fig19）。

读取 artifacts/augmentation/protocols/full/{summary,oof_seed,fold_seed}.csv：

* (a) 10 s 宏 3σ：3 seed 均值、seed 标准差、各 seed 点；
* (b) 10 s 宏 MAPE：同一版式；
* (c) 逐 seed/折的宏 3σ 配对差（候选 − 仅真实，负值更好）。
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

from sensor_array.plotting.style import _save, _style

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/protocols/full"
ARMS = ("real", "vae", "forward", "forward+vae")
CANDIDATES = ("vae", "forward", "forward+vae")
LABELS = {
    "real": "仅真实",
    "vae": "PCA-VAE",
    "forward": "正模型",
    "forward+vae": "正模型 + PCA-VAE",
}
COLORS = {
    "real": "#7F7F7F",
    "vae": "#4C72B0",
    "forward": "#D55E00",
    "forward+vae": "#009E73",
}


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _bars(axes, summary: list[dict], seeds: list[dict], metric: str, title: str) -> None:
    top = max(
        float(row[f"{metric}_mean"]) + float(row[f"{metric}_sd"])
        for row in summary
    )
    axes.set_ylim(0.0, top * 1.25)
    for index, arm in enumerate(ARMS):
        row = next(item for item in summary if item["arm"] == arm)
        mean = float(row[f"{metric}_mean"])
        sd = float(row[f"{metric}_sd"])
        axes.bar(
            index,
            mean,
            width=0.72,
            color=COLORS[arm],
            alpha=0.88,
            yerr=sd,
            capsize=2.5,
            error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
        )
        values = [
            float(item[metric]) for item in seeds if item["arm"] == arm
        ]
        offsets = np.linspace(-0.17, 0.17, len(values))
        axes.scatter(
            index + offsets,
            values,
            s=11,
            facecolors="white",
            edgecolors="#222222",
            linewidths=0.55,
            zorder=3,
        )
        axes.text(
            index,
            mean + sd + 0.02 * top,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )
    axes.set_xticks(range(len(ARMS)))
    axes.set_xticklabels([LABELS[arm] for arm in ARMS], rotation=15)
    axes.set_title(title)
    axes.set_ylabel("宏平均 / %")
    axes.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes.set_axisbelow(True)


def _paired(axes, folds: list[dict]) -> None:
    table = {
        (int(row["seed"]), int(row["fold"]), row["arm"]): float(
            row["macro_three_sigma"]
        )
        for row in folds
    }
    keys = sorted({(int(row["seed"]), int(row["fold"])) for row in folds})
    rng = np.random.default_rng(0)
    data = {}
    for arm in CANDIDATES:
        data[arm] = np.array(
            [
                table[(seed, fold, arm)] - table[(seed, fold, "real")]
                for seed, fold in keys
            ]
        )
    for index, arm in enumerate(CANDIDATES):
        differences = data[arm]
        axes.scatter(
            index + rng.uniform(-0.16, 0.16, len(differences)),
            differences,
            s=15,
            color=COLORS[arm],
            alpha=0.85,
            edgecolors="none",
            zorder=2,
        )
        axes.plot(
            [index - 0.28, index + 0.28],
            [np.median(differences), np.median(differences)],
            color="#222222",
            linewidth=1.0,
            zorder=3,
        )
    axes.axhline(0.0, color="#555555", linewidth=0.8, linestyle="--")
    low = min(float(values.min()) for values in data.values())
    high = max(float(values.max()) for values in data.values())
    span = max(high - low, 1.0)
    axes.set_ylim(low - 0.08 * span, high + 0.30 * span)
    for index, arm in enumerate(CANDIDATES):
        differences = data[arm]
        wins = int((differences < 0).sum())
        p_value = (
            wilcoxon(differences).pvalue
            if np.any(differences)
            else 1.0
        )
        axes.text(
            index,
            high + 0.12 * span,
            f"{wins}/{len(differences)} 改善\n"
            + ("p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"),
            ha="center",
            va="bottom",
            fontsize=6.6,
        )
    axes.set_xticks(range(len(CANDIDATES)))
    axes.set_xticklabels([LABELS[arm] for arm in CANDIDATES], rotation=15)
    axes.set_ylabel("相对仅真实的宏 3σ 差 / 百分点")
    axes.set_title("(c) 逐 seed/折配对差（负值更好）")
    axes.grid(axis="y", color="#E5E5E5", linewidth=0.6)
    axes.set_axisbelow(True)


def main() -> None:
    summary = _rows(BASE / "summary.csv")
    seeds = _rows(BASE / "oof_seed.csv")
    folds = _rows(BASE / "fold_seed.csv")
    _style()
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.5), layout="constrained")
    _bars(axes[0], summary, seeds, "macro_three_sigma", "(a) 10 s 宏 3σ")
    _bars(axes[1], summary, seeds, "macro_mape", "(b) 10 s 宏 MAPE")
    _paired(axes[2], folds)
    handles = [
        plt.Rectangle(
            (0, 0), 1, 1, color=COLORS[arm], alpha=0.88, label=LABELS[arm]
        )
        for arm in ARMS
    ]
    figure.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7.5,
    )
    figure.suptitle(
        "完整旧 TCN 协议：0.1 噪声 + 边缘复制 + 两套低浓度配置，四臂 3 seed × 5 折"
    )
    _save(figure, "fig19_full_protocol_augmentation")


if __name__ == "__main__":
    main()
