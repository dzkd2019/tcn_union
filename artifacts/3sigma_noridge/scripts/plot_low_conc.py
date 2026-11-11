"""低浓度损失加权的原理示意图（PNG+SVG）。

三 panel：
(a) 权重函数 `w=(median/y)^p` 随 `y/median` 的变化（p=0.3/0.5/1.0/2.0，
    裁剪到 0.25–4）；
(b) 50 个开发实验的逐气体浓度档位分布（相对中位数）；
(c) OOF 上 H2/NO2 的 3σ 随加权方案的变化（说明两配置集成的必要性）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.evaluation.evaluate import _manifest

GASES = ("H2", "NH3", "NO2")
COLORS = {"H2": "#1F77B4", "NH3": "#F2A900", "NO2": "#D2691E"}
FIGS = ROOT / "artifacts/3sigma_noridge/figs"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _panel_weights(panel) -> None:
    ratio = np.logspace(np.log10(0.25), np.log10(4.0), 400)
    for power, style, label in (
        (0.3, ":", "p=0.3"),
        (0.5, "--", "p=0.5"),
        (1.0, "-", "p=1.0"),
        (2.0, "-.", "p=2.0"),
    ):
        panel.plot(ratio, np.clip(ratio ** (-power), 0.25, 4.0), style, linewidth=1.5, label=label)
    panel.axhline(1.0, color="0.6", linewidth=0.8)
    panel.axhline(0.25, color="0.75", linewidth=0.6, linestyle="--")
    panel.axhline(4.0, color="0.75", linewidth=0.6, linestyle="--")
    panel.set_xscale("log")
    panel.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0], ["0.25×", "0.5×", "1×", "2×", "4×"])
    panel.minorticks_off()
    panel.set_xlabel("y / median(y)（对数轴）", fontsize=8.5)
    panel.set_ylabel("损失权重 w", fontsize=8.5)
    panel.set_title("(a) 权重函数与裁剪边界", fontsize=9.5, loc="left")
    panel.legend(fontsize=7.5, frameon=False)
    panel.grid(alpha=0.2)
    panel.tick_params(labelsize=7.5)


def _panel_labels(panel) -> None:
    """低浓度半区获得的损失质量占比（p=1.0）。"""
    rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    dev = [row for row in rows if row["split"] == "development"]
    labels = np.array([row["_labels"] for row in dev], dtype=float)
    median = np.median(labels, axis=0)
    x = np.arange(len(GASES))
    width = 0.2
    count_low = []
    mass_low = []
    for index, _gas in enumerate(GASES):
        values = labels[:, index]
        low = values <= median[index]
        weights = np.clip((median[index] / values) ** 1.0, 0.25, 4.0)
        count_low.append(float(low.mean()))
        mass_low.append(float(weights[low].sum() / weights.sum()))
    panel.bar(x - 1.5 * width, count_low, width, color="0.65", label="样本占比（≤中位数）")
    panel.bar(x - 0.5 * width, mass_low, width, color="#1F6F78", label="损失质量占比（≤中位数）")
    panel.bar(x + 0.5 * width, [1 - value for value in count_low], width, color="0.85", label="样本占比（>中位数）")
    panel.bar(x + 1.5 * width, [1 - value for value in mass_low], width, color="#C0504A", label="损失质量占比（>中位数）")
    panel.set_xticks(x, GASES, fontsize=8)
    panel.set_ylim(0, 1.0)
    panel.set_ylabel("占比", fontsize=8.5)
    panel.set_title("(b) p=1.0：低浓度半区获得更多损失质量", fontsize=9.5, loc="left")
    panel.legend(fontsize=6.6, frameon=False, ncol=2)
    panel.grid(axis="y", alpha=0.2)
    panel.tick_params(labelsize=7.5)


def _panel_screening(panel) -> None:
    # OOF 宏 3σ（3 seed + 裁剪，无 Ridge）；数据见 ablation/low_conc_screening.json
    schemes = [
        ("无加权", 9.32, 10.65),
        ("p=0.5", 9.89, 10.55),
        ("p=1.0", 9.95, 8.11),
        ("逐气体", 9.62, 9.01),
        ("两配置平均", 9.31, 8.16),
    ]
    x = np.arange(len(schemes))
    width = 0.36
    panel.bar(x - width / 2, [s[1] for s in schemes], width, color=COLORS["H2"], label="H2")
    panel.bar(x + width / 2, [s[2] for s in schemes], width, color=COLORS["NO2"], label="NO2")
    panel.axhline(10.0, color="0.3", linestyle="--", linewidth=0.9)
    panel.set_xticks(x, [s[0] for s in schemes], rotation=20, fontsize=7)
    panel.set_ylabel("OOF 3σ (%)", fontsize=8.5)
    panel.set_title("(c) H2/NO2 的权衡：两配置平均更稳", fontsize=9.5, loc="left")
    panel.legend(fontsize=7, frameon=False)
    panel.grid(axis="y", alpha=0.2)
    panel.tick_params(labelsize=7.5)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    figure, panels = plt.subplots(1, 3, figsize=(13.0, 3.9), constrained_layout=True)
    _panel_weights(panels[0])
    _panel_labels(panels[1])
    _panel_screening(panels[2])
    figure.suptitle(
        "低浓度损失加权：权重设计、损失质量占比与 OOF 权衡",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_low_conc_weighting.{suffix}"
        figure.savefig(path, dpi=280, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
