"""绘制 PCA 主成分数选择的证据图。

三个面板：
(a) 累计解释方差；
(b) 五折验证重构误差；
(c) 下游 OOF macro-NMAE 随 K 的变化（K=4/6/8 用三个 seed 的均值±标准差）。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Noto Serif CJK SC"
plt.rcParams["font.serif"] = ["Noto Serif CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300

AXES = ("H2", "NH3", "NO2")
COLORS = {"H2": "#0072B2", "NH3": "#E69F00", "NO2": "#D55E00", "macro": "#333333"}

# 单 seed（20260923）的下游 OOF；K=6 来自当前最终模型记录。
SINGLE_SEED = {
    2: 0.02476433999836445,
    4: 0.021917950734496115,
    6: 0.02401529364287853,
    8: 0.02268291786313057,
    10: 0.02312142662703991,
    12: 0.023799529299139977,
}
# K=4/6/8 的三种子 OOF（同一协议）。
THREE_SEED = {
    4: [0.021917950734496115, 0.025430469959974288, 0.024208027124404907],
    6: [0.02401529364287853, 0.02304329015314579, 0.023956895247101782],
    8: [0.02268291786313057, 0.025382630154490472, 0.02595249004662037],
}


def _read_components(path: Path) -> dict[str, dict[int, dict[str, float]]]:
    output: dict[str, dict[int, dict[str, float]]] = {gas: {} for gas in AXES}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            output[row["gas"]][int(row["k"])] = {
                "ev": float(row["explained_variance"]),
                "recon": float(row["reconstruction_percent"]),
            }
    return output


def _macro(records: dict[str, dict[int, dict[str, float]]], key: str) -> list[float]:
    ks = sorted(records[AXES[0]])
    return [float(np.mean([records[gas][k][key] for gas in AXES])) for k in ks]


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 PCA 主成分数证据图")
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _read_components(args.components)
    ks = sorted(records[AXES[0]])

    figure, panels = plt.subplots(1, 3, figsize=(12.6, 3.7), constrained_layout=True)
    for gas in AXES:
        panels[0].plot(
            ks,
            [100.0 * records[gas][k]["ev"] for k in ks],
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            color=COLORS[gas],
            label=gas,
        )
        panels[1].plot(
            ks,
            [records[gas][k]["recon"] for k in ks],
            marker="o",
            markersize=2.5,
            linewidth=1.1,
            color=COLORS[gas],
            label=gas,
        )
    panels[0].plot(
        ks,
        [100.0 * value for value in _macro(records, "ev")],
        color=COLORS["macro"],
        linewidth=1.7,
        linestyle="--",
        label="三气体平均",
    )
    panels[1].plot(
        ks,
        _macro(records, "recon"),
        color=COLORS["macro"],
        linewidth=1.7,
        linestyle="--",
        label="三气体平均",
    )
    single_k = sorted(SINGLE_SEED)
    panels[2].plot(
        single_k,
        [SINGLE_SEED[k] for k in single_k],
        color="#777777",
        marker="o",
        markersize=3,
        linewidth=1.0,
        label="单 seed 试点",
    )
    three_k = sorted(THREE_SEED)
    means = [float(np.mean(THREE_SEED[k])) for k in three_k]
    stds = [float(np.std(THREE_SEED[k], ddof=1)) for k in three_k]
    panels[2].errorbar(
        three_k,
        means,
        yerr=stds,
        color="#2A6F77",
        marker="s",
        markersize=3.5,
        linewidth=1.3,
        capsize=2.5,
        label="三 seed 均值 ± SD",
    )
    panels[2].axvline(6, color="#888888", linestyle=":", linewidth=0.9)
    panels[2].annotate(
        f"K=6: {np.mean(THREE_SEED[6]):.5f}\nSD={np.std(THREE_SEED[6], ddof=1):.5f}",
        xy=(6, np.mean(THREE_SEED[6])),
        xytext=(6.25, 0.0258),
        fontsize=6.5,
        color="#2A6F77",
    )

    for panel in panels:
        panel.axvline(6, color="#888888", linestyle=":", linewidth=0.9)
        panel.grid(alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7)
        panel.set_xlabel("主成分数 K", fontsize=8)
    panels[0].set_ylabel("累计解释方差 / %", fontsize=8)
    panels[1].set_ylabel("验证 RMSE / 量程 %", fontsize=8)
    panels[2].set_ylabel("OOF macro-NMAE", fontsize=8)
    panels[0].set_title("(a) 解释方差", fontsize=9, loc="left")
    panels[1].set_title("(b) 五折重构误差", fontsize=9, loc="left")
    panels[2].set_title("(c) 下游 OOF（越低越好）", fontsize=9, loc="left")
    panels[0].legend(fontsize=6.5, frameon=False)
    panels[2].legend(fontsize=6.5, frameon=False, loc="upper left")

    for suffix in ("png", "pdf"):
        path = args.output_dir / f"fig_pca_k_evidence.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
