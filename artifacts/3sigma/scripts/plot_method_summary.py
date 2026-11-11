"""3σ 方法总结图：堆叠/精简路径、模型数与 OOF、seed 数影响（PNG+SVG）。

数据来自 `artifacts/3sigma/method_ablation.json`（由
`artifacts/3sigma/scripts/ablate_3sigma_methods.py` 生成，仅用 OOF）。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
JSON_PATH = ROOT / "artifacts/3sigma/method_ablation.json"
FIGS = ROOT / "artifacts/3sigma/figs"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ACCENT = "#1F6F78"
MUTED = "#9AA5AD"
HIGHLIGHT = "#C0504A"


def _read() -> dict:
    return json.loads(JSON_PATH.read_text(encoding="utf-8"))


def _panel_stack(ax, data: dict) -> None:
    stack = data["stack"]
    labels = [row["label"] for row in stack]
    values = [row["hybrid_macro"] for row in stack]
    colors = [MUTED] * len(values)
    for index, row in enumerate(stack):
        if "精简方案" in row["label"]:
            colors[index] = ACCENT
        elif "9 配置" in row["label"] or "5 配置" in row["label"]:
            colors[index] = HIGHLIGHT
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    for index, value in enumerate(values):
        ax.text(value + 0.08, index, f"{value:.2f}", va="center", fontsize=7.5)
    ax.axvline(10.0, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("OOF 宏平均 3σ (%)", fontsize=8.5)
    ax.set_title("(a) 方法路径：OOF 宏 3σ（6 seed，越低越好）", fontsize=9.5, loc="left")
    ax.set_xlim(0, max(values) * 1.15)
    ax.grid(axis="x", alpha=0.2)
    ax.tick_params(labelsize=7.5)


def _panel_models(ax, data: dict) -> None:
    lean_3 = data["seed_ablation"][2]["hybrid"]["macro"]
    lean_6 = data["subsets"][1]["hybrid_macro"]
    no_edge_6 = data["subsets"][0]["hybrid_macro"]
    base5 = data["subsets"][4]["hybrid_macro"]
    full9 = data["subsets"][5]["hybrid_macro"]
    points = (
        ("精简·3 seed（推荐）", 3, lean_3, ACCENT, (8, -14)),
        ("精简·6 seed", 6, lean_6, ACCENT, (8, -4)),
        ("精简·无边缘·6 seed", 6, no_edge_6, "#7FA8A0", (-72, 10)),
        ("基础 5 配置（v4）", 30, base5, HIGHLIGHT, (-30, 10)),
        ("完整 9 配置（v5）", 54, full9, HIGHLIGHT, (-88, -12)),
    )
    for label, models, value, color, offset in points:
        ax.scatter(models, value, s=48, color=color, zorder=3)
        ax.annotate(
            f"{label}（{models} 模型）",
            (models, value),
            textcoords="offset points",
            xytext=offset,
            fontsize=6.6,
            color=color,
        )
    ax.axhline(10.0, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_xlabel("部署模型数（TCN checkpoint）", fontsize=8.5)
    ax.set_ylabel("OOF 宏平均 3σ (%)", fontsize=8.5)
    ax.set_title("(b) 模型数 vs OOF：堆料没有收益", fontsize=9.5, loc="left")
    ax.set_ylim(6.8, 10.1)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7.5)


def _panel_seeds(ax, data: dict) -> None:
    for set_name, color, label in (
        ("精简方案", ACCENT, "精简方案（逐气体头+边缘4）"),
        ("精简-无边缘", "#7FA8A0", "精简-无边缘"),
        ("9 配置集成", HIGHLIGHT, "9 配置集成（v5）"),
    ):
        rows = [row for row in data["seed_ablation"] if row["set"] == set_name]
        seeds = [row["seeds"] for row in rows]
        values = [row["hybrid"]["macro"] for row in rows]
        ax.plot(seeds, values, marker="o", color=color, linewidth=1.4, label=label)
        for seed, value in zip(seeds, values):
            ax.annotate(f"{value:.2f}", (seed, value), textcoords="offset points", xytext=(0, 7), fontsize=6.3, ha="center", color=color)
    ax.axhline(10.0, color="0.4", linestyle="--", linewidth=0.9)
    ax.set_xlabel("seed 数", fontsize=8.5)
    ax.set_ylabel("OOF 宏平均 3σ (%)", fontsize=8.5)
    ax.set_title("(c) seed 数：3 个已足够，更多无收益", fontsize=9.5, loc="left")
    ax.set_xticks([1, 2, 3, 6])
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=7.5)


def main() -> None:
    data = _read()
    figure, panels = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    _panel_stack(panels[0], data)
    _panel_models(panels[1], data)
    _panel_seeds(panels[2], data)
    figure.suptitle(
        "3σ 降方差：堆料 vs 精简（OOF 50 个开发实验；虚线 = 10% 验收线）",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_3sigma_method_summary.{suffix}"
        figure.savefig(path, dpi=280, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
