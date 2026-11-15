"""绘制当前 3σ 方案所用 GasTCN 的模型架构图（PNG+SVG）。

风格参考 `example/tcn.png`：白底、浅绿/浅黄圆角框、黑色箭头、虚线分组。
图中只画模型结构与数据流，不展开多 seed 平均、低浓度加权的两套配置等训练细节。
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
) -> None:
    """画一个圆角框，`lines` 为 (文字, 字号, 是否加粗) 列表，从中心向上排列。"""
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.4,
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


def _line(ax, xs: list[float], ys: list[float]) -> None:
    ax.plot(xs, ys, color="black", linewidth=1.5, solid_capstyle="butt", zorder=4)


def draw_architecture() -> plt.Figure:
    apply_style()
    figure, ax = plt.subplots(figsize=(8.6, 10.6))
    figure.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13.3)
    ax.axis("off")

    input_y = 0.75
    _box(
        ax,
        5.0,
        input_y,
        6.6,
        0.95,
        face=YELLOW_FILL,
        edge=YELLOW_EDGE,
        lines=[
            ("Input: 10 s response prefix", 11.0, True),
            (r"$X \in \mathbb{R}^{3 \times 51}$", 11.5, False),
        ],
        line_gap=0.34,
    )
    _arrow(ax, (5.0, input_y + 0.48), (5.0, 1.95))

    _box(
        ax,
        5.0,
        2.45,
        6.6,
        0.95,
        face=GREEN_FILL,
        edge=GREEN_EDGE,
        lines=[
            ("Feature construction: raw_delta", 11.0, True),
            ("6 channels × 51 samples", 9.6, False),
        ],
        line_gap=0.34,
    )
    _arrow(ax, (5.0, 2.93), (5.0, 3.55))

    # ---- Causal TCN encoder（虚线分组，内部三个残差块自下而上）----
    group_x, group_w = 1.25, 7.5
    group_y, group_h = 3.55, 4.75
    ax.add_patch(
        Rectangle(
            (group_x, group_y),
            group_w,
            group_h,
            fill=False,
            edgecolor="black",
            linestyle=(0, (5, 4)),
            linewidth=1.3,
            zorder=1,
        )
    )
    ax.text(
        group_x + 0.2,
        group_y + group_h - 0.28,
        "Causal TCN encoder",
        fontsize=11.0,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=3,
    )
    ax.text(
        group_x + 0.2,
        group_y + group_h - 0.58,
        "3 residual blocks · 16 channels",
        fontsize=8.6,
        ha="left",
        va="center",
        color="#333333",
        zorder=3,
    )

    block_centers = (4.45, 5.70, 6.95)
    for center, dilation in zip(block_centers, (4, 2, 1), strict=True):
        _box(
            ax,
            5.0,
            center,
            5.1,
            0.80,
            face=GREEN_FILL,
            edge=GREEN_EDGE,
            lines=[(f"Residual block (k=5, d={dilation})", 10.5, True)],
        )
    _arrow(ax, (5.0, 3.55), (5.0, 4.05))
    _arrow(ax, (5.0, 4.85), (5.0, 5.30))
    _arrow(ax, (5.0, 6.10), (5.0, 6.55))
    _arrow(ax, (5.0, 7.35), (5.0, 8.50))

    _box(
        ax,
        5.0,
        8.90,
        6.4,
        0.80,
        face=YELLOW_FILL,
        edge=YELLOW_EDGE,
        lines=[(r"Last timestep hidden state   $h_T \in \mathbb{R}^{16}$", 11.0, True)],
    )

    # ---- 三路逐气体输出头 ----
    head_xs = (2.7, 5.0, 7.3)
    head_y = 10.55
    _arrow(ax, (5.0, 9.30), (5.0, 9.72))
    _line(ax, [head_xs[0], head_xs[-1]], [9.72, 9.72])
    for x, gas, target in zip(head_xs, ("H2", "NH3", "NO2"), ("log target", "linear target", "log target"), strict=True):
        _arrow(ax, (x, 9.72), (x, 10.10))
        _box(
            ax,
            x,
            head_y,
            2.02,
            0.92,
            face=GREEN_FILL,
            edge=GREEN_EDGE,
            lines=[(f"{gas} head", 10.4, True), (f"Linear 16→16→1 · {target}", 8.0, False)],
            line_gap=0.30,
        )
        _arrow(ax, (x, head_y + 0.47), (x, 11.70))
        ax.text(
            x,
            11.95,
            rf"$\hat{{y}}_{{{gas}}}$ (ppm)",
            fontsize=12.5,
            ha="center",
            va="center",
        )

    ax.text(
        5.0,
        12.85,
        r"$\hat{y} = (\hat{y}_{H2},\ \hat{y}_{NH3},\ \hat{y}_{NO2})$",
        fontsize=13,
        ha="center",
        va="center",
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 GasTCN 模型架构图（PNG+SVG）")
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
