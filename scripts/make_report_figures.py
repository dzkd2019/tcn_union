"""生成数据增强报告使用的示意图和汇总图。"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "report" / "figures"


def _box(ax, x, y, w, h, text, color="#e8eefc", edge="#3557a5"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color="#172033")


def _arrow(ax, start, end, color="#3557a5"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=color, shrinkA=0, shrinkB=0))


def figure_pipeline() -> None:
    fig = plt.figure(figsize=(14.2, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    time = np.linspace(0, 10, 51)
    signal = 0.02 + 0.38 * (1 - np.exp(-time / 2.2)) + 0.012 * time
    ax.plot(time, signal, color="#172033", linewidth=2.0, label="原始 10 秒响应")
    start, end = 8, 16
    ax.plot(time[start:end + 1], signal[start:end + 1], color="#d94841", linewidth=3.2)
    ax.plot([time[start], time[end]], [signal[start], signal[end]], "--", color="#d94841", linewidth=1.8)
    ax.axvspan(time[start], time[end], color="#d94841", alpha=0.10)
    ax.annotate("随机内部区间\n（1–10 个采样点，≤ 2 秒）", xy=(time[start] + 0.4, signal[start] + 0.03),
                xytext=(0.4, 0.78), textcoords="axes fraction", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#d94841", "lw": 1.1})
    ax.annotate("线性桥", xy=(time[end], signal[end]), xytext=(0.58, 0.55),
                textcoords="axes fraction", fontsize=9, color="#d94841",
                arrowprops={"arrowstyle": "->", "color": "#d94841", "lw": 1.1})
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("时间（秒）", fontsize=10)
    ax.set_ylabel("原始响应（a.u.）", fontsize=10)
    ax.set_title("(a) 训练时仅对原始响应做 cutout", fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.04, "t = 0 与 t = 10 秒不修改；标签不变。",
            transform=ax.transAxes, fontsize=8.5, color="#555")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) 多前缀监督流程", fontsize=11, loc="left")
    _box(ax, 0.01, 0.56, 0.15, 0.17, "原始 10 秒\n响应\n（51 × 3）")
    _box(ax, 0.20, 0.56, 0.15, 0.17, "cutout\n（仅训练）", color="#fdeceb", edge="#c4504a")
    _box(ax, 0.39, 0.56, 0.17, 0.17, "raw_delta\n特征\n（6 × 51）")
    _box(ax, 0.60, 0.56, 0.16, 0.17, "因果 TCN\n3 块，16 通道")
    _box(ax, 0.80, 0.71, 0.15, 0.14, "主输出头\n@ 10 秒\nL_main", color="#e9f5ee", edge="#2f7d4f")
    _box(ax, 0.80, 0.38, 0.15, 0.14, "辅助头\n@ 5 秒\n0.5 L_aux", color="#e9f5ee", edge="#2f7d4f")
    for x0, y0, x1, y1 in [
        (0.16, 0.645, 0.20, 0.645),
        (0.35, 0.645, 0.39, 0.645),
        (0.56, 0.645, 0.60, 0.645),
        (0.76, 0.68, 0.80, 0.78),
        (0.76, 0.61, 0.80, 0.45),
    ]:
        _arrow(ax, (x0, y0), (x1, y1))
    ax.text(0.80, 0.28, "总训练损失", fontsize=9.2, weight="bold")
    ax.text(0.80, 0.22, "L_main + 0.5 L_aux", fontsize=10.2, weight="bold", color="#1f4e8c")
    ax.text(0.01, 0.30, "验证 / 测试 / 推理", fontsize=9.5, weight="bold")
    ax.text(0.01, 0.22, "干净 10 秒响应 → TCN → 主输出头 → 三 seed 平均", fontsize=9)
    ax.text(0.01, 0.15, "不做 cutout，不加辅助损失，不读取未来样本。", fontsize=9, color="#555")

    fig.suptitle("10 秒响应加速的数据增强", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_augmentation_pipeline.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def figure_mape() -> None:
    selection = json.loads((ROOT / "artifacts/augmentation_final/selection.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "artifacts/evaluation_final/evaluation.json").read_text(encoding="utf-8"))
    oof = {
        "5 秒原始": selection["selected"]["5s"]["oof_metrics"]["macro_MAPE_percent"],
        "10 秒原始": 4.98251237763299,
        "10 秒 cutout + 5 秒辅助": selection["selected"]["10s"]["oof_metrics"]["macro_MAPE_percent"],
    }
    test_ensemble = {}
    for item in evaluation["metrics"]:
        if item["seed"] == "ensemble":
            key = f'{item["model"]}_{item["window_seconds"]}'
            test_ensemble[key] = item["macro_MAPE_percent"]
    test = {
        "5 秒原始": test_ensemble["baseline_5"],
        "10 秒原始": test_ensemble["baseline_10"],
        "10 秒 cutout + 5 秒辅助": test_ensemble["augmented_10"],
    }
    labels = list(oof)
    colors = ["#9ca3af", "#f59e0b", "#2563eb"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=False)
    for ax, values, title in [
        (axes[0], oof, "(a) 开发集 OOF（50 次）"),
        (axes[1], test, "(b) 锁定测试（12 次，三 seed 集成）"),
    ]:
        bars = ax.bar(labels, [values[k] for k in labels], color=colors, width=0.62)
        for bar, key in zip(bars, labels, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{values[key]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("macro-MAPE（%）")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(values.values()) * 1.22)
        ax.tick_params(axis="x", labelsize=8.6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_mape_summary.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_pipeline()
    figure_mape()
    print(f"已写出图像到 {OUT}")
