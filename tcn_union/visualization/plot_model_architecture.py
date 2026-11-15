"""绘制 GasTCN 模型架构图（中文，PNG+SVG）。

风格参考 `example/tcn.png`：白底、浅绿/浅黄圆角框、黑色箭头、虚线分组。
主框架给出 GasTCN 的数据流；主框架外单独给出：
- 训练目标与损失（真实浓度标签 → log 变换 → 低浓度加权 SmoothL1）；
- 因果卷积块（Residual block）内部结构；
- PCA-VAE 数据扩充（训练时）作为输入侧的数据来源。
图中不展开多 seed 平均，也不展开低浓度加权的两套配置。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from tcn_union.visualization.plot_style import apply_style

GREEN_FILL = "#D9EAD3"
GREEN_EDGE = "#5C8A3C"
YELLOW_FILL = "#FFF2CC"
YELLOW_EDGE = "#B45F06"
ARROW = dict(arrowstyle="-|>", color="black", lw=1.5, shrinkA=0, shrinkB=0, mutation_scale=14)


def _box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    lines: list[tuple[str, float, bool]],
    line_gap: float = 0.30,
    dashed: bool = False,
) -> None:
    """画一个圆角框，`lines` 为 (文字, 字号, 是否加粗)，从中心向上排列。"""
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
        linestyle="--" if dashed else "-",
        zorder=2,
    )
    ax.add_patch(patch)
    total = (len(lines) - 1) * line_gap
    first_y = y + total / 2
    for index, (text, size, bold) in enumerate(lines):
        ax.text(
            x,
            first_y - index * line_gap,
            text,
            ha="center",
            va="center",
            fontsize=size,
            fontweight="bold" if bold else "normal",
            color="black",
            zorder=3,
        )


def _arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=ARROW, zorder=4)


def _line(ax, xs: list[float], ys: list[float], *, dashed: bool = False) -> None:
    ax.plot(
        xs,
        ys,
        color="black",
        linewidth=1.5,
        linestyle="--" if dashed else "-",
        solid_capstyle="butt",
        zorder=4,
    )


def _draw_main_frame(ax) -> None:
    """主框架：数据 → 特征 → 因果 TCN → 逐气体头 → 输出。"""
    # 输入侧：真实响应 + 训练时的 PCA-VAE 扩充
    _box(
        ax, 4.65, 1.35, 2.7, 0.95,
        face=YELLOW_FILL, edge=YELLOW_EDGE,
        lines=[("真实响应", 10.5, True), ("10 秒前缀（51×3）", 8.8, False)],
    )
    _box(
        ax, 7.75, 1.35, 2.7, 0.95,
        face=GREEN_FILL, edge=GREEN_EDGE,
        lines=[("PCA-VAE 数据扩充", 10.0, True), ("（仅训练时）", 8.6, False)],
        dashed=True,
    )
    _arrow(ax, (4.65, 1.83), (4.65, 2.25))
    _arrow(ax, (7.75, 1.83), (7.75, 2.25))

    _box(
        ax, 6.2, 2.75, 5.9, 0.95,
        face=GREEN_FILL, edge=GREEN_EDGE,
        lines=[("特征构造：raw_delta", 10.5, True), ("6 通道 × 51 个采样点", 8.8, False)],
    )
    _arrow(ax, (6.2, 3.23), (6.2, 3.55))

    # 因果 TCN 编码器（虚线分组）
    group_x, group_w = 2.95, 6.5
    group_y, group_h = 3.55, 4.10
    ax.add_patch(
        Rectangle(
            (group_x, group_y), group_w, group_h,
            fill=False, edgecolor="black", linestyle=(0, (5, 4)), linewidth=1.3, zorder=1,
        )
    )
    ax.text(group_x + 0.2, group_y + group_h - 0.26, "因果 TCN 编码器", fontsize=11.0, fontweight="bold", ha="left", va="center", zorder=3)
    ax.text(group_x + 0.2, group_y + group_h - 0.55, "3 个块 · 16 通道", fontsize=8.6, ha="left", va="center", color="#333333", zorder=3)

    block_centers = (4.25, 5.45, 6.65)
    for center, dilation in zip(block_centers, (4, 2, 1), strict=True):
        _box(
            ax, 6.2, center, 5.5, 0.72,
            face=GREEN_FILL, edge=GREEN_EDGE,
            lines=[(f"因果卷积块（k=5，d={dilation}）", 10.2, True)],
        )
    _arrow(ax, (6.2, 3.55), (6.2, 3.89))
    _arrow(ax, (6.2, 4.61), (6.2, 5.09))
    _arrow(ax, (6.2, 5.81), (6.2, 6.29))
    _arrow(ax, (6.2, 7.01), (6.2, 8.02))

    _box(
        ax, 6.2, 8.42, 5.4, 0.80,
        face=YELLOW_FILL, edge=YELLOW_EDGE,
        lines=[(r"最后时刻隐状态  $h_T \in \mathbb{R}^{16}$", 10.8, True)],
    )

    # 三路逐气体输出头
    head_xs = (4.35, 6.2, 8.05)
    head_y = 9.55
    _arrow(ax, (6.2, 8.82), (6.2, 9.00))
    _line(ax, [head_xs[0], head_xs[-1]], [9.00, 9.00])
    for x, gas in zip(head_xs, ("H2", "NH3", "NO2"), strict=True):
        _arrow(ax, (x, 9.00), (x, 9.08))
        _box(
            ax, x, head_y, 1.58, 0.95,
            face=GREEN_FILL, edge=GREEN_EDGE,
            lines=[(f"{gas} 输出头", 10.0, True), ("16→16→1", 8.4, False)],
            line_gap=0.28,
        )
        _arrow(ax, (x, head_y + 0.48), (x, 10.35))
        ax.text(x, 10.55, rf"$\hat{{y}}_{{\mathrm{{{gas}}}}}$（ppm）", fontsize=11.5, ha="center", va="center")

    # 主框架虚线边框与标题
    ax.add_patch(
        Rectangle(
            (2.45, 0.65), 7.55, 10.85,
            fill=False, edgecolor="black", linestyle=(0, (6, 4)), linewidth=1.5, zorder=1,
        )
    )
    ax.text(2.65, 11.28, "GasTCN 主框架", fontsize=11.5, fontweight="bold", ha="left", va="center")


def _draw_loss_panel(ax) -> None:
    """主框架外：训练目标与损失。"""
    x0, y0, w, h = 10.25, 7.2, 5.6, 4.3
    ax.add_patch(
        Rectangle(
            (x0, y0), w, h,
            fill=False, edgecolor="black", linestyle=(0, (5, 4)), linewidth=1.4, zorder=1,
        )
    )
    ax.text(x0 + 0.2, y0 + h - 0.28, "训练目标与损失（仅训练时）", fontsize=11.0, fontweight="bold", ha="left", va="center")

    _box(
        ax, 11.35, 10.65, 1.95, 0.62,
        face=YELLOW_FILL, edge=YELLOW_EDGE,
        lines=[("真实浓度标签", 9.2, True)],
    )
    _arrow(ax, (12.33, 10.65), (12.75, 10.65))
    _box(
        ax, 14.0, 10.65, 2.35, 0.72,
        face=GREEN_FILL, edge=GREEN_EDGE,
        lines=[("log 变换", 9.4, True), ("H2 / NO2；NH3 原空间", 7.4, False)],
        line_gap=0.24,
    )
    _arrow(ax, (14.0, 10.29), (14.0, 9.80))
    _box(
        ax, 13.45, 9.25, 2.15, 0.85,
        face=GREEN_FILL, edge=GREEN_EDGE,
        lines=[("SmoothL1 损失", 9.2, True), ("低浓度加权", 8.4, False)],
        line_gap=0.22,
    )
    _box(
        ax, 11.2, 9.25, 1.75, 0.62,
        face=YELLOW_FILL, edge=YELLOW_EDGE,
        lines=[("预测浓度", 9.2, True)],
    )
    _arrow(ax, (12.08, 9.25), (12.36, 9.25))
    ax.text(11.2, 8.75, "（来自主框架）", fontsize=7.2, color="#444444", ha="center", va="center")
    ax.text(13.45, 8.35, "权重：clip((median(y)/y)^p, 0.25, 4)", fontsize=7.2, color="#444444", ha="center", va="center")
    ax.text(13.45, 8.00, "逐气体加权；p 为低浓度加权强度", fontsize=7.2, color="#444444", ha="center", va="center")


def _draw_residual_block_panel(ax) -> None:
    """主框架外：因果卷积块（Residual block）内部结构。"""
    x0, y0, w, h = 10.25, 0.6, 5.6, 6.35
    ax.add_patch(
        Rectangle(
            (x0, y0), w, h,
            fill=False, edgecolor="black", linestyle=(0, (5, 4)), linewidth=1.4, zorder=1,
        )
    )
    ax.text(x0 + 0.2, y0 + h - 0.28, "因果卷积块结构", fontsize=11.0, fontweight="bold", ha="left", va="center")
    ax.text(x0 + 0.2, y0 + h - 0.56, "（k=5，d 为膨胀率）", fontsize=8.4, ha="left", va="center", color="#333333")

    main_x = 11.75
    plus_x = 13.85
    _box(
        ax, main_x, 1.05, 2.15, 0.55,
        face=YELLOW_FILL, edge=YELLOW_EDGE,
        lines=[(r"输入 $z^{(i-1)}$", 9.0, True)],
    )
    _arrow(ax, (main_x, 1.33), (main_x, 1.55))
    rows = (
        ("膨胀因果卷积", GREEN_FILL, GREEN_EDGE, 1.80),
        ("ReLU", GREEN_FILL, GREEN_EDGE, 2.38),
        ("Dropout", YELLOW_FILL, YELLOW_EDGE, 2.96),
        ("膨胀因果卷积", GREEN_FILL, GREEN_EDGE, 3.54),
        ("ReLU", GREEN_FILL, GREEN_EDGE, 4.12),
        ("Dropout", YELLOW_FILL, YELLOW_EDGE, 4.70),
    )
    for label, face, edge, y in rows:
        _box(ax, main_x, y, 2.15, 0.42, face=face, edge=edge, lines=[(label, 8.6, False)])
    _arrow(ax, (main_x, 2.01), (main_x, 2.17))
    _arrow(ax, (main_x, 2.59), (main_x, 2.75))
    _arrow(ax, (main_x, 3.17), (main_x, 3.33))
    _arrow(ax, (main_x, 3.75), (main_x, 3.91))
    _arrow(ax, (main_x, 4.33), (main_x, 4.49))

    # 残差支路：1×1 卷积（可选）
    _line(ax, [main_x + 1.08, plus_x], [1.05, 1.05], dashed=True)
    _arrow(ax, (plus_x, 1.05), (plus_x, 1.48))
    _box(
        ax, plus_x, 1.80, 1.55, 0.62,
        face=GREEN_FILL, edge=GREEN_EDGE,
        lines=[("1×1 卷积", 8.2, False), ("（可选）", 7.4, False)],
        line_gap=0.22,
        dashed=True,
    )
    _line(ax, [plus_x, plus_x], [2.11, 4.70], dashed=True)
    _line(ax, [main_x + 1.08, plus_x], [4.70, 4.70])

    # 残差相加后再过一层 ReLU
    circle = plt.Circle((plus_x, 4.90), 0.13, facecolor="white", edgecolor="black", linewidth=1.3, zorder=5)
    ax.add_patch(circle)
    ax.text(plus_x, 4.90, "+", fontsize=10.5, ha="center", va="center", zorder=6)
    _line(ax, [plus_x, plus_x], [4.70, 4.77])
    _arrow(ax, (plus_x, 5.03), (plus_x, 5.05))
    _box(
        ax, plus_x, 5.24, 2.15, 0.38,
        face=GREEN_FILL, edge=GREEN_EDGE,
        lines=[("ReLU", 8.6, False)],
    )
    _arrow(ax, (plus_x, 5.43), (plus_x, 5.62))
    ax.text(plus_x, 5.82, r"输出 $z^{(i)}$", fontsize=9.2, ha="center", va="center")


def draw_architecture() -> plt.Figure:
    apply_style()
    figure, ax = plt.subplots(figsize=(15.2, 11.2))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 11.9)
    ax.axis("off")

    _draw_main_frame(ax)
    _draw_loss_panel(ax)
    _draw_residual_block_panel(ax)
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 GasTCN 模型架构图（中文，PNG+SVG）")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/3sigma_noridge/figs"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure = draw_architecture()
    for suffix in ("png", "svg"):
        path = args.output_dir / f"fig_model_architecture.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
