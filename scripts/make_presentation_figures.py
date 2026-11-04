"""生成组会汇报使用的中文标注图。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "report" / "ppt_figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 10

C_GRAY = "#9ca3af"
C_ORANGE = "#f59e0b"
C_BLUE = "#2563eb"
C_RED = "#d94841"
C_GREEN = "#2f7d4f"
C_TEXT = "#172033"


def box(ax, x, y, w, h, text, fc="#e8eefc", ec="#3557a5", fs=10.0, weight="normal"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C_TEXT, weight=weight)


def arrow(ax, start, end, color="#3557a5", style="-|>", lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=12,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=0, shrinkB=0))


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig0_data():
    frame = pd.read_csv(ROOT / "data/0.csv")
    time = frame["time"].to_numpy(dtype=float)
    response = frame[["H2", "NH3", "NO2"]].to_numpy(dtype=float)
    mask = time <= 10.0 + 1e-9
    fig = plt.figure(figsize=(13.2, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.75, 1.0], wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    for idx, name in enumerate(("H2", "NH3", "NO2")):
        y = response[mask, idx]
        ax.plot(time[mask], y / y[0], linewidth=1.8, label=name)
    ax.axvspan(0, 5, color="#dbeafe", alpha=0.55, label="5 秒窗口")
    ax.axvspan(5, 10, color="#fef3c7", alpha=0.55, label="10 秒窗口")
    ax.set_xlim(0, 10)
    ax.set_xlabel("时间 / 秒")
    ax.set_ylabel("响应 / 初值")
    ax.set_title("(a) 一次实验的三路响应（前 10 秒）", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=2)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(b) 实验级划分", fontsize=11, loc="left")
    box(ax, 0.12, 0.55, 0.76, 0.18, "62 次实验", fc="#f3f4f6", ec="#6b7280", fs=11, weight="bold")
    box(ax, 0.05, 0.16, 0.42, 0.18, "50 次\n开发集", fc="#dbeafe", ec="#2563eb", fs=11, weight="bold")
    box(ax, 0.56, 0.16, 0.39, 0.18, "12 次\n锁定测试", fc="#fef3c7", ec="#d97706", fs=10.5, weight="bold")
    arrow(ax, (0.38, 0.55), (0.27, 0.35))
    arrow(ax, (0.62, 0.55), (0.75, 0.35))
    ax.text(0.5, 0.04, "同一实验不拆到训练和测试", ha="center", fontsize=8.5, color="#555")

    ax = fig.add_subplot(gs[0, 2])
    levels = ["10", "20", "30", "40", "50"]
    dev = [6, 11, 11, 11, 11]
    test = [1, 3, 3, 3, 2]
    x = np.arange(len(levels))
    ax.bar(x, dev, color="#93c5fd", label="开发集")
    ax.bar(x, test, bottom=dev, color="#fbbf24", label="测试集")
    for i, (d, t) in enumerate(zip(dev, test, strict=True)):
        ax.text(i, d + t + 0.25, f"{d + t}", ha="center", fontsize=8.5)
    ax.set_xticks(x, levels)
    ax.set_xlabel("NH3 浓度档位")
    ax.set_ylabel("实验数")
    ax.set_title("(c) NH3 档位样本数：低浓度最少", fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylim(0, 16)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("NH3=10 测试仅 1 次", xy=(0, 7), xytext=(1.1, 12.5),
                arrowprops={"arrowstyle": "->", "color": C_RED, "lw": 1.0},
                fontsize=8.5, color=C_RED)

    fig.suptitle("任务与数据：三路响应预测三种浓度，按实验划分", fontsize=13, y=1.02)
    save(fig, "fig0_任务与数据")


def fig1_model():
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.25], wspace=0.18)

    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(a) 输入特征：原始响应 + 相对增量", fontsize=11, loc="left")
    box(ax, 0.03, 0.70, 0.26, 0.13, "H2 原始响应", fs=9.5)
    box(ax, 0.03, 0.52, 0.26, 0.13, "NH3 原始响应", fs=9.5)
    box(ax, 0.03, 0.34, 0.26, 0.13, "NO2 原始响应", fs=9.5)
    box(ax, 0.38, 0.52, 0.26, 0.31, "三路相对增量\n$\\frac{x(t)-x(0)}{|x(0)|}$", fc="#fdeceb", ec=C_RED, fs=9.5)
    box(ax, 0.72, 0.44, 0.24, 0.26, "6 通道\n× 51 点", fc="#e9f5ee", ec=C_GREEN, fs=11, weight="bold")
    for y in (0.765, 0.585, 0.405):
        arrow(ax, (0.29, y), (0.38, 0.675))
    arrow(ax, (0.64, 0.675), (0.72, 0.57))
    ax.text(0.5, 0.14, "原始 3 通道 + 增量 3 通道 = 6 通道", ha="center", fontsize=9, color="#555")
    ax.text(0.5, 0.05, "增量按初值归一化，减小量纲差异，保留动态变化", ha="center", fontsize=8.5, color="#555")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(b) 因果 TCN：只看过去，输出三种浓度", fontsize=11, loc="left")
    box(ax, 0.01, 0.55, 0.13, 0.18, "输入\n6×51", fs=9.5)
    for i, (x, label) in enumerate(zip((0.18, 0.35, 0.52), ("残差块 1\n膨胀率 1", "残差块 2\n膨胀率 2", "残差块 3\n膨胀率 4"))):
        box(ax, x, 0.55, 0.15, 0.18, label, fs=8.8)
        arrow(ax, (x - 0.02 if i else 0.14, 0.64), (x, 0.64))
    box(ax, 0.70, 0.55, 0.14, 0.18, "取最后\n时刻", fc="#fef3c7", ec="#d97706", fs=9.2)
    arrow(ax, (0.67, 0.64), (0.70, 0.64))
    box(ax, 0.87, 0.55, 0.12, 0.18, "全连接\n→ 3", fs=9.2)
    arrow(ax, (0.84, 0.64), (0.87, 0.64))
    ax.text(0.5, 0.40, "每个卷积只在左侧补零：$t$ 时刻只能看到 $0,\\dots,t$", ha="center", fontsize=9, color="#555")
    ax.text(0.5, 0.30, "感受野 57 个采样点，覆盖完整 10 秒输入", ha="center", fontsize=9, color="#555")
    # 因果关系时间轴
    ax.plot([0.08, 0.92], [0.16, 0.16], color="#9ca3af", lw=2)
    ax.plot([0.08], [0.16], "o", color=C_BLUE, ms=7)
    ax.plot([0.92], [0.16], "o", color=C_RED, ms=7)
    ax.text(0.08, 0.08, "t=0", ha="center", fontsize=8.5)
    ax.text(0.92, 0.08, "t=10 s", ha="center", fontsize=8.5)
    ax.annotate("预测点", xy=(0.92, 0.16), xytext=(0.72, 0.25), fontsize=8.5,
                arrowprops={"arrowstyle": "->", "color": C_RED, "lw": 1.0})
    ax.text(0.34, 0.22, "可见范围", color=C_BLUE, fontsize=8.5)

    fig.suptitle("模型：用 10 秒内的因果 TCN 回归三种浓度", fontsize=13, y=1.02)
    save(fig, "fig1_模型结构")


def fig2_training():
    fig, ax = plt.subplots(figsize=(14.0, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("训练方法：五折选轮数，三个种子重训，测试集最后才看", fontsize=13, pad=12)

    # 62 次实验与数据划分
    box(ax, 0.40, 0.84, 0.20, 0.10, "62 次实验", fc="#f3f4f6", ec="#6b7280", fs=11, weight="bold")
    box(ax, 0.02, 0.57, 0.13, 0.12, "50 次\n开发集", fc="#dbeafe", ec="#2563eb", fs=10, weight="bold")
    arrow(ax, (0.44, 0.84), (0.09, 0.69))

    # 开发集分支
    box(ax, 0.18, 0.57, 0.14, 0.12, "五折交叉\n验证", fs=9.0)
    box(ax, 0.35, 0.57, 0.13, 0.12, "取 5 折\nepoch 中位数", fs=8.6)
    box(ax, 0.51, 0.57, 0.13, 0.12, "50 个开发\n实验重训", fs=8.6)
    box(ax, 0.67, 0.57, 0.12, 0.12, "三个固定\nseed", fs=8.8)
    box(ax, 0.82, 0.57, 0.13, 0.12, "三 seed\n集成", fc="#e9f5ee", ec=C_GREEN, fs=9.2, weight="bold")
    for x0, x1 in [(0.15, 0.18), (0.32, 0.35), (0.48, 0.51), (0.64, 0.67), (0.79, 0.82)]:
        arrow(ax, (x0, 0.63), (x1, 0.63))
    ax.text(0.25, 0.53, "每折 40 训练 / 10 验证\n验证集早停，记录最佳 epoch", ha="center", fontsize=7.8, color="#555")
    ax.text(0.575, 0.53, "固定轮数，不用测试集早停", ha="center", fontsize=7.8, color="#555")

    # 测试集分支
    box(ax, 0.70, 0.25, 0.13, 0.12, "12 次\n锁定测试", fc="#fef3c7", ec="#d97706", fs=9.5, weight="bold")
    box(ax, 0.86, 0.25, 0.12, 0.12, "最终评估\n只看一次", fc="#fff7ed", ec="#fb923c", fs=8.6)
    arrow(ax, (0.83, 0.31), (0.86, 0.31), color="#d97706", ls="--")
    ax.text(0.815, 0.19, "不参与调参", fontsize=8.0, color="#d97706", ha="center")
    arrow(ax, (0.76, 0.57), (0.76, 0.37), color="#d97706", ls="--")
    ax.text(0.78, 0.46, "方案冻结\n后才看", fontsize=8.0, color="#d97706")

    ax.text(0.02, 0.88, "数据划分", fontsize=10, weight="bold", color="#3557a5")
    ax.text(0.02, 0.48, "开发集：选轮数与重训", fontsize=10, weight="bold", color="#3557a5")
    ax.text(0.70, 0.40, "测试集：最终评估", fontsize=10, weight="bold", color="#d97706")
    ax.text(0.70, 0.14, "测试集来自同一批 62 次实验，与开发集实验级互斥", fontsize=7.6, color="#555")
    save(fig, "fig2_训练方法")


def fig3_problem():
    fig = plt.figure(figsize=(13.0, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])
    seeds = ["seed 23", "seed 24", "seed 25"]
    five = [0.027605, 0.029835, 0.031231]
    ten = [0.027097, 0.028427, 0.029521]
    x = np.arange(3)
    width = 0.34
    b1 = ax.bar(x - width / 2, five, width, color=C_GRAY, label="5 秒原始")
    b2 = ax.bar(x + width / 2, ten, width, color=C_ORANGE, label="10 秒旧模型")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                    f"{bar.get_height():.5f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(x, seeds)
    ax.set_ylabel("macro-NMAE（越低越好）")
    ax.set_title("(a) 开发集 OOF：10 秒每个 seed 都更好", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 0.036)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("表面结论和 OOF 相反", xy=(0.5, 0.033), ha="center", fontsize=9, color=C_RED)

    ax = fig.add_subplot(gs[0, 1])
    labels = ["file2\nNH3=10", "file7\nNH3=20"]
    old = [29.68, 16.60]
    new = [3.34, 6.27]
    x = np.arange(2)
    width = 0.34
    b1 = ax.bar(x - width / 2, old, width, color=C_ORANGE, label="旧 10 秒")
    b2 = ax.bar(x + width / 2, new, width, color=C_BLUE, label="新 10 秒")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{bar.get_height():.2f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_ylabel("NH3 绝对百分比误差 APE (%)")
    ax.set_title("(b) 测试集两个低浓度样本：旧 10 秒明显高估", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, 36)
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("预测 12.97 / 真值 10", xy=(0, 29.68), xytext=(0.2, 33.5),
                fontsize=8.5, color=C_RED,
                arrowprops={"arrowstyle": "->", "color": C_RED, "lw": 1.0})

    fig.suptitle("问题：不是 10 秒没信息，而是 10 秒模型被末端曲线带偏", fontsize=13, y=1.02)
    save(fig, "fig3_存在的问题")


def fig4_solution():
    fig = plt.figure(figsize=(14.0, 5.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.35], wspace=0.15)

    ax = fig.add_subplot(gs[0, 0])
    time = np.linspace(0, 10, 51)
    signal = 0.04 + 0.38 * (1 - np.exp(-time / 2.2)) + 0.012 * time
    ax.plot(time, signal, color=C_TEXT, lw=2.0, label="原始响应")
    start, end = 8, 16
    ax.plot(time[start:end + 1], signal[start:end + 1], color=C_RED, lw=3.2, label="被挖去的片段")
    ax.plot([time[start], time[end]], [signal[start], signal[end]], "--", color=C_RED, lw=1.8, label="直线连接")
    ax.axvspan(time[start], time[end], color=C_RED, alpha=0.10)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("时间 / 秒")
    ax.set_ylabel("响应")
    ax.set_title("(a) cutout：训练时随机挖掉中间一小段", fontsize=11, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.05, "初值、终点、标签都不变；只在训练时使用", transform=ax.transAxes, fontsize=8.5, color="#555")

    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("(b) 5 秒辅助监督 + 三 seed 集成", fontsize=11, loc="left")
    box(ax, 0.01, 0.56, 0.14, 0.16, "原始 10 秒\n响应", fs=8.8)
    box(ax, 0.19, 0.56, 0.14, 0.16, "cutout\n（仅训练）", fc="#fdeceb", ec=C_RED, fs=8.8)
    box(ax, 0.37, 0.56, 0.15, 0.16, "6 通道\n特征", fs=8.8)
    box(ax, 0.56, 0.56, 0.15, 0.16, "因果 TCN", fs=9.0)
    box(ax, 0.76, 0.72, 0.16, 0.13, "10 秒主输出\n主损失", fc="#e9f5ee", ec=C_GREEN, fs=8.5)
    box(ax, 0.76, 0.40, 0.16, 0.13, "5 秒辅助输出\n0.5 × 辅助损失", fc="#e9f5ee", ec=C_GREEN, fs=8.0)
    for x0, y0, x1, y1 in [
        (0.15, 0.64, 0.19, 0.64),
        (0.33, 0.64, 0.37, 0.64),
        (0.52, 0.64, 0.56, 0.64),
        (0.71, 0.68, 0.76, 0.78),
        (0.71, 0.60, 0.76, 0.46),
    ]:
        arrow(ax, (x0, y0), (x1, y1))
    ax.text(0.56, 0.30, "训练损失 = 主损失 + 0.5 × 辅助损失", fontsize=9.5, weight="bold", color="#1f4e8c")
    box(ax, 0.03, 0.08, 0.18, 0.10, "三个 seed", fs=9.0, fc="#f3f4f6", ec="#6b7280")
    box(ax, 0.28, 0.08, 0.22, 0.10, "取平均 → 最终预测", fs=9.0, fc="#dbeafe", ec=C_BLUE)
    arrow(ax, (0.21, 0.13), (0.28, 0.13))
    ax.text(0.56, 0.13, "推理只用 10 秒主输出", fontsize=8.8, color="#555")

    fig.suptitle("解决：cutout 逼模型别死记局部，5 秒辅助监督防止忘掉低浓度经验", fontsize=12.5, y=1.02)
    save(fig, "fig4_解决方案")


def fig5_results():
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.0))
    fig.subplots_adjust(hspace=0.55, wspace=0.28)
    names = ["5 秒\n原始", "10 秒\n旧模型", "10 秒\n新方案"]
    colors = [C_GRAY, C_ORANGE, C_BLUE]
    macro_nmae = [0.020487, 0.024807, 0.018321]
    macro_mape = [4.0176, 4.6476, 3.3882]

    for ax, values, title, ylabel, fmt in [
        (axes[0, 0], macro_nmae, "(a) 测试集 macro-NMAE", "macro-NMAE（越低越好）", "{:.5f}"),
        (axes[0, 1], macro_mape, "(b) 测试集 macro-MAPE (%)", "macro-MAPE (%)", "{:.3f}"),
    ]:
        bars = ax.bar(names, values, color=colors, width=0.6)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(bar.get_height()), ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(values) * 1.25)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", labelsize=8.5)

    ax = axes[1, 0]
    gases = ["H2", "NH3", "NO2"]
    five = [0.02261, 0.01153, 0.02733]
    old = [0.02790, 0.01715, 0.02937]
    new = [0.02074, 0.00829, 0.02593]
    x = np.arange(3)
    width = 0.25
    for i, (vals, label, color) in enumerate(zip((five, old, new), ("5 秒原始", "10 秒旧模型", "10 秒新方案"), colors, strict=True)):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0004,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.0)
    ax.set_xticks(x, gases)
    ax.set_ylabel("NMAE（越低越好）")
    ax.set_title("(c) 测试集逐气体 NMAE：三个气体都更好", fontsize=11, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(0, 0.036)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 1]
    seeds = ["seed 23", "seed 24", "seed 25", "集成"]
    five = [0.026269, 0.024762, 0.022614, 0.020487]
    new = [0.024165, 0.023790, 0.019876, 0.018321]
    x = np.arange(4)
    width = 0.34
    b1 = ax.bar(x - width / 2, five, width, color=C_GRAY, label="5 秒原始")
    b2 = ax.bar(x + width / 2, new, width, color=C_BLUE, label="10 秒新方案")
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0003,
                    f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_xticks(x, seeds)
    ax.set_ylabel("测试集 macro-NMAE")
    ax.set_title("(d) 逐 seed：10 秒都不高于对应 5 秒", fontsize=11, loc="left")
    ax.legend(fontsize=7.5, frameon=False)
    ax.set_ylim(0, 0.032)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("最终结果：锁定测试集上 10 秒方案在三个气体上都优于 5 秒", fontsize=13, y=0.98)
    save(fig, "fig5_最终结果")


def fig6_next():
    fig, ax = plt.subplots(figsize=(12.0, 4.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("下一步：先把数据补齐，再把结论验证扎实", fontsize=13, pad=14)
    data = [
        (0.02, "补数据", "同浓度重复实验\n低浓度缺失格\n不同日期 / 器件", "#dbeafe", "#2563eb"),
        (0.35, "稳训练", "更多 seed\n更稳定的单种子协议\n按气体调整辅助权重", "#fef3c7", "#d97706"),
        (0.68, "再验证", "独立测试集\n外部盲测\n明确误差容限", "#e9f5ee", "#2f7d4f"),
    ]
    for x, title, body, fc, ec in data:
        box(ax, x, 0.30, 0.28, 0.42, "", fc=fc, ec=ec)
        ax.text(x + 0.14, 0.60, title, ha="center", va="center", fontsize=13, weight="bold", color=ec)
        ax.text(x + 0.14, 0.44, body, ha="center", va="center", fontsize=9.2, color="#333")
    for x0, x1 in ((0.30, 0.35), (0.63, 0.68)):
        arrow(ax, (x0, 0.51), (x1, 0.51), color="#6b7280", lw=1.8)
    ax.text(0.5, 0.12, "当前结论只适用于现有 62 次实验和固定 50/12 划分", ha="center", fontsize=9, color="#555")
    save(fig, "fig6_下一步")




def fig7_initial_correlation():
    manifest = pd.read_csv(ROOT / "artifacts/splits/manifest.csv", dtype=str).to_dict("records")
    dev = [row for row in manifest if row["split"] == "development"]
    inits, labels, ids = [], [], []
    for row in dev:
        frame = pd.read_csv(ROOT / row["file"])
        inits.append(frame[["H2", "NH3", "NO2"]].to_numpy(dtype=float)[0])
        labels.append([float(row["H2"]), float(row["NH3"]), float(row["NO2"])])
        ids.append(int(Path(row["file"]).stem))
    init = np.asarray(inits, dtype=float)
    label = np.asarray(labels, dtype=float)
    run_id = np.asarray(ids, dtype=float)

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    def residual(y, x):
        design = np.column_stack([np.ones(len(y)), x])
        beta = np.linalg.lstsq(design, y, rcond=None)[0]
        return y - design @ beta

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6))
    fig.subplots_adjust(wspace=0.32)
    scatter = axes[0].scatter(init[:, 1], label[:, 1], c=run_id, cmap="viridis", s=42, edgecolor="white", zorder=3)
    axes[0].set_xlabel("t=0 时 NH3 通道响应")
    axes[0].set_ylabel("NH3 浓度标签")
    axes[0].set_title(f"(a) 表面强相关：r = {corr(init[:, 1], label[:, 1]):.3f}", fontsize=11, loc="left")
    axes[0].grid(alpha=0.25)
    fig.colorbar(scatter, ax=axes[0], fraction=0.046, pad=0.03, label="文件编号（采集顺序代理）")

    axes[1].scatter(run_id, init[:, 1], c=run_id, cmap="viridis", s=42, edgecolor="white", zorder=3)
    axes[1].set_xlabel("文件编号（采集顺序代理）")
    axes[1].set_ylabel("t=0 时 NH3 通道响应")
    axes[1].set_title(f"(b) 初值随实验顺序漂移：r = {corr(run_id, init[:, 1]):.3f}", fontsize=11, loc="left")
    axes[1].grid(alpha=0.25)

    ax = axes[2]
    ax.scatter(residual(init[:, 1], run_id), residual(label[:, 1], run_id), s=42, color=C_BLUE, edgecolor="white", zorder=3)
    ax.set_xlabel("去掉文件编号后的初值残差")
    ax.set_ylabel("去掉文件编号后的 NH3 标签残差")
    ax.set_title(f"(c) 控制顺序后：r = {corr(residual(init[:, 1], run_id), residual(label[:, 1], run_id)):.3f}", fontsize=11, loc="left")
    ax.grid(alpha=0.25)

    fig.suptitle("初值相关性诊断：表面很强，但主要来自实验顺序/批次漂移", fontsize=13, y=1.02)
    save(fig, "fig3b_初值相关性诊断")


if __name__ == "__main__":
    fig0_data()
    fig1_model()
    fig2_training()
    fig3_problem()
    fig4_solution()
    fig5_results()
    fig6_next()
    fig7_initial_correlation()
    print(f"Wrote figures to {OUT}")
