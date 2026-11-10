"""绘制 GAN 与 PCA-VAE 数据增强的简明结构框图。

用于口头汇报：每张图左侧讲“怎么生成”，右侧讲“怎么用于 TCN 训练”。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT = "Noto Sans CJK SC"
plt.rcParams.update(
    {
        "font.family": FONT,
        "font.sans-serif": [FONT],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)

COLORS = {
    "gray": "#F2F4F5",
    "blue": "#E3EEF9",
    "orange": "#FCEAD7",
    "teal": "#E0F1ED",
    "edge": "#39434D",
    "arrow": "#39434D",
    "text": "#1F2933",
    "muted": "#5D6A75",
}


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str = "",
    color: str = "gray",
    *,
    dashed: bool = False,
    title_size: float = 8.2,
    body_size: float = 6.8,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=COLORS[color],
            edgecolor=COLORS["edge"],
            linewidth=0.8,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.66,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["text"],
            zorder=3,
        )
        ax.text(
            x + w / 2,
            y + h * 0.28,
            body,
            ha="center",
            va="center",
            fontsize=body_size,
            color=COLORS["muted"],
            linespacing=1.2,
            zorder=3,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=COLORS["text"],
            zorder=3,
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str = "arrow",
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            linestyle="--" if dashed else "-",
            color=COLORS[color],
            connectionstyle=f"arc3,rad={rad}",
            zorder=1,
        )
    )


def _panel(ax: plt.Axes, x: float, y: float, w: float, h: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor="white",
            edgecolor="#B8C2CC",
            linewidth=0.7,
            zorder=0,
        )
    )
    ax.text(x + 0.12, y + h - 0.18, title, ha="left", va="center", fontsize=8.6, fontweight="bold", color=COLORS["text"])


def draw_gan(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.1))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.1)
    ax.set_axis_off()

    _panel(ax, 0.10, 0.10, 5.15, 3.85, "① GAN 怎么生成")
    _panel(ax, 5.40, 0.10, 5.10, 3.85, "② 我们怎么用于训练")

    # 左侧：生成器/判别器对抗
    _box(ax, 0.30, 2.55, 1.10, 0.62, "噪声 z", "随机向量", "blue")
    _box(ax, 0.30, 1.70, 1.10, 0.62, "浓度标签 c", "H2/NH3/NO2", "orange")
    _box(ax, 1.65, 2.15, 1.05, 0.62, "生成器 G", "假曲线 3×51", "blue")
    _box(ax, 2.95, 2.65, 1.02, 0.55, "假曲线", "", "blue")
    _box(ax, 2.95, 1.62, 1.02, 0.55, "真实曲线", "", "gray")
    _box(ax, 4.18, 2.10, 0.88, 0.62, "判别器 D", "真/假分数", "orange")
    _arrow(ax, (1.42, 2.86), (1.63, 2.55))
    _arrow(ax, (1.42, 2.01), (1.63, 2.35))
    _arrow(ax, (2.72, 2.46), (2.93, 2.86))
    _arrow(ax, (3.97, 2.88), (4.16, 2.50))
    _arrow(ax, (3.97, 1.90), (4.16, 2.30))
    ax.text(4.10, 1.35, "对抗：G 骗过 D，D 识别真假", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])
    ax.text(0.30, 0.30, "只在训练折拟合；验证/测试不加入合成样本", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])

    # 右侧：增强流程
    _box(ax, 5.60, 2.55, 1.05, 0.64, "训练折", "40 真曲线", "gray")
    _box(ax, 6.90, 2.55, 1.10, 0.64, "训练 GAN", "300 epoch", "blue")
    _box(ax, 8.25, 2.55, 1.08, 0.64, "采样", "120 条合成", "blue")
    _box(ax, 5.60, 1.35, 1.05, 0.64, "转 raw_delta", "6×51", "gray")
    _box(ax, 6.95, 1.35, 1.05, 0.64, "拼接", "40 真+120 合成", "gray")
    _box(ax, 8.25, 1.35, 1.05, 0.64, "训练 TCN", "三气体预测", "teal")
    _arrow(ax, (6.67, 2.87), (6.88, 2.87))
    _arrow(ax, (8.02, 2.87), (8.23, 2.87))
    _arrow(ax, (8.79, 2.53), (6.15, 2.01), rad=0.10)
    _arrow(ax, (6.67, 1.67), (6.93, 1.67))
    _arrow(ax, (8.02, 1.67), (8.23, 1.67))
    ax.text(5.60, 0.72, "合成曲线 → TCN 特征；标签沿用训练集浓度", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])

    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_pca_vae(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.6, 4.3))
    ax.set_xlim(0, 10.6)
    ax.set_ylim(0, 4.3)
    ax.set_axis_off()

    _panel(ax, 0.10, 0.10, 5.15, 4.05, "① PCA + VAE 怎么生成")
    _panel(ax, 5.40, 0.10, 5.10, 4.05, "② 我们怎么用于训练")

    _box(ax, 0.30, 2.85, 0.95, 0.60, "真实曲线", "3×51", "gray")
    _box(ax, 1.43, 2.85, 0.90, 0.60, "逐气体 PCA", "6 个系数", "teal")
    _box(ax, 2.50, 2.85, 0.95, 0.60, "PCA 系数", "3×6", "teal")
    _box(ax, 3.72, 2.85, 0.72, 0.60, "反 PCA", "", "teal")
    _box(ax, 4.62, 2.85, 0.42, 0.60, "曲线", "", "blue")
    _box(ax, 0.30, 1.75, 0.95, 0.60, "浓度标签", "3 维", "orange")
    _box(ax, 1.43, 1.75, 0.95, 0.60, "条件均值", "标签→系数", "orange")
    _box(ax, 2.55, 1.75, 0.90, 0.60, "VAE 残差", "学个体差异", "blue")
    _box(ax, 3.72, 1.75, 0.72, 0.60, "相加", "均值+残差", "orange")
    _arrow(ax, (1.26, 3.15), (1.41, 3.15))
    _arrow(ax, (2.34, 3.15), (2.48, 3.15))
    _arrow(ax, (3.46, 3.15), (3.70, 3.15))
    _arrow(ax, (4.45, 3.15), (4.60, 3.15))
    _arrow(ax, (1.26, 2.05), (1.41, 2.05))
    _arrow(ax, (2.39, 2.05), (2.53, 2.05))
    _arrow(ax, (3.46, 2.05), (3.70, 2.05))
    _arrow(ax, (4.08, 2.36), (4.08, 2.83))
    ax.text(0.30, 1.15, "VAE 只学“同浓度下的残差”，PCA 保证曲线在真实子空间", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])

    _box(ax, 5.60, 2.85, 1.05, 0.62, "训练折", "40 真曲线", "gray")
    _box(ax, 6.85, 2.85, 1.25, 0.62, "拟合 PCA+VAE", "每折独立", "teal")
    _box(ax, 8.30, 2.85, 1.05, 0.62, "采样", "120 合成", "teal")
    _box(ax, 5.60, 1.55, 1.05, 0.62, "真实噪声", "40 条 0.1×", "orange")
    _box(ax, 6.85, 1.55, 1.10, 0.62, "拼接", "40真+40噪+120合", "gray")
    _box(ax, 8.15, 1.55, 1.05, 0.62, "训练 TCN", "三气体预测", "teal")
    _arrow(ax, (6.67, 3.16), (6.83, 3.16))
    _arrow(ax, (8.12, 3.16), (8.28, 3.16))
    _arrow(ax, (8.82, 2.83), (7.42, 2.19), rad=0.05)
    _arrow(ax, (6.67, 1.86), (6.83, 1.86))
    _arrow(ax, (7.97, 1.86), (8.13, 1.86))
    ax.text(5.60, 0.75, "验证/测试始终为真实曲线；三 seed 预测平均", ha="left", va="center", fontsize=6.8, color=COLORS["muted"])

    fig.savefig(output.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 GAN 与 PCA-VAE 数据增强结构图")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/augmentation_manuscript"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw_gan(args.output_dir / "fig_gan_flow")
    draw_pca_vae(args.output_dir / "fig_pca_vae_flow")
    print(f"Wrote figures to {args.output_dir}")


if __name__ == "__main__":
    main()
