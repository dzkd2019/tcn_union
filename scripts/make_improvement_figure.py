"""生成 EMA 改进实验的对比图。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "research" / "to_human" / "ema_improvement.png"


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    fig.subplots_adjust(wspace=0.28)

    ax = axes[0]
    seeds = ["seed 23", "seed 24", "seed 25"]
    clean = [0.027097, 0.028427, 0.029521]
    current = [0.024592, 0.031708, 0.028850]
    ema = [0.024718, 0.028601, 0.029149]
    x = np.arange(3)
    width = 0.26
    for index, (values, label, color) in enumerate(
        zip(
            (clean, current, ema),
            ("原始 10 秒", "当前 10 秒", "EMA 10 秒"),
            ("#9ca3af", "#f59e0b", "#2563eb"),
            strict=True,
        )
    ):
        bars = ax.bar(x + (index - 1) * width, values, width, label=label, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0004,
                    f"{bar.get_height():.5f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x, seeds)
    ax.set_ylabel("OOF macro-NMAE（越低越好）")
    ax.set_title("(a) 三种子 OOF：EMA 降低 seed 24 回退", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, 0.036)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    names = ["5 秒原始", "当前 10 秒", "EMA 10 秒"]
    values = [0.020487, 0.018321, 0.016692]
    colors = ("#9ca3af", "#f59e0b", "#2563eb")
    bars = ax.bar(names, values, color=colors, width=0.6)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                f"{bar.get_height():.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("测试 macro-NMAE（越低越好）")
    ax.set_title("(b) 锁定测试：三 seed 集成", fontsize=11, loc="left")
    ax.set_ylim(0, 0.026)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("EMA 权重平均：降低 OOF 种子方差，并进一步改善测试集", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"已写出 {OUT}")


if __name__ == "__main__":
    main()
