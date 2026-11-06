"""绘制参照论文版式的 GasTCN 架构图。

布局参考 example/pic_2.jpg：
(a) 左侧为主流程：数据构造 → GasTCN 特征提取 → 输出头；
(b) 右上为残差块内部；
(c) 右下为因果膨胀卷积与感受野。

所有参数从 GasTCN 实例读取；输出 PNG 和矢量 PDF。
"""
from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from torch import nn

from .model import GasTCN

FONT_FAMILY = "Noto Serif CJK SC"
FONT_FILE: str | None = None

COLORS = {
    "green": "#6FA97A",
    "green_fill": "#F0F8F2",
    "pink": "#C97A8B",
    "pink_fill": "#FDF1F3",
    "purple": "#9A7CB8",
    "purple_fill": "#F8F2FC",
    "blue_fill": "#EDF4FB",
    "yellow_fill": "#FFF8E7",
    "box_edge": "#3F4A54",
    "text": "#1D2730",
    "muted": "#59656F",
    "arrow": "#38434D",
}

W, H = 12.0, 7.5


def _configure_font() -> FontProperties:
    global FONT_FILE
    font = FontProperties(family=FONT_FAMILY)
    FONT_FILE = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY],
            "font.size": 7,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "mathtext.fontset": "stix",
        }
    )
    return FontProperties(family=FONT_FAMILY, fname=FONT_FILE)


def _font(size: float, weight: str = "normal") -> FontProperties:
    if FONT_FILE is None:
        raise RuntimeError("configure font first")
    return FontProperties(family=FONT_FAMILY, fname=FONT_FILE, size=size, weight=weight)


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    face: str,
    edge: str = COLORS["box_edge"],
    *,
    dashed: bool = False,
    title_size: float = 7.0,
    body_size: float = 6.0,
    linewidth: float = 0.8,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            linestyle="--" if dashed else "-",
            zorder=3,
        )
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.68,
            title,
            ha="center",
            va="center",
            fontproperties=_font(title_size, "bold"),
            color=COLORS["text"],
            zorder=4,
        )
        ax.text(
            x + w / 2,
            y + h * 0.30,
            body,
            ha="center",
            va="center",
            fontproperties=_font(body_size),
            color=COLORS["muted"],
            linespacing=1.18,
            zorder=4,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontproperties=_font(title_size, "bold"),
            color=COLORS["text"],
            zorder=4,
        )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    dashed: bool = False,
    color: str = COLORS["arrow"],
    width: float = 0.8,
    head: float = 7.0,
    rad: float = 0.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )


def _group(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    color: str,
    fill: str,
    label: str,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=fill,
            edgecolor=color,
            linewidth=1.1,
            linestyle="--",
            zorder=1,
        )
    )
    ax.text(
        x + 0.12,
        y + h - 0.16,
        label,
        ha="left",
        va="center",
        fontproperties=_font(7.2, "bold"),
        color=color,
        zorder=2,
    )


def _panel_border(ax: plt.Axes, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="white",
            edgecolor=COLORS["box_edge"],
            linewidth=0.9,
            zorder=0,
        )
    )


def _read_specs() -> tuple[list[dict[str, object]], list[nn.Linear], int]:
    model = GasTCN(in_channels=6)
    specs = []
    for index, block in enumerate(model.blocks):
        conv1, conv2 = block.conv1, block.conv2
        if not isinstance(conv1, nn.Conv1d) or not isinstance(conv2, nn.Conv1d):
            raise TypeError("residual block must use Conv1d")
        specs.append(
            {
                "index": index + 1,
                "kernel": int(conv1.kernel_size[0]),
                "dilation": int(block.dilation),
                "input_channels": int(conv1.in_channels),
                "channels": int(conv1.out_channels),
                "dropout": float(block.dropout1.p),
                "skip": "1×1 Conv" if isinstance(block.projection, nn.Conv1d) else "Identity",
            }
        )
    linears = [layer for layer in model.head if isinstance(layer, nn.Linear)]
    if len(linears) != 2:
        raise ValueError("head must contain two linear layers")
    receptive = 1 + 2 * sum(
        (int(item["kernel"]) - 1) * int(item["dilation"]) for item in specs
    )
    return specs, linears, receptive


def draw_architecture(output_path: Path) -> tuple[Path, Path]:
    _configure_font()
    specs, linears, receptive = _read_specs()
    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # ---------- 主面板 (a) ----------
    _panel_border(ax, 0.15, 0.20, 8.00, 7.08)
    _group(ax, 0.32, 1.42, 2.02, 5.55, COLORS["green"], COLORS["green_fill"], "训练数据构造")
    _group(ax, 2.56, 1.42, 2.72, 5.55, COLORS["pink"], COLORS["pink_fill"], "GasTCN 特征提取")
    _group(ax, 5.50, 1.42, 2.42, 5.55, COLORS["purple"], COLORS["purple_fill"], "输出头与预测")

    _box(ax, 0.52, 5.80, 1.62, 0.64, "真实响应", "50 条开发实验\n6×51（10 s）", COLORS["green_fill"], COLORS["green"])
    _box(ax, 0.52, 4.92, 1.62, 0.45, "测量噪声", "0.1× 真实噪声", COLORS["yellow_fill"], COLORS["green"])
    _box(ax, 0.52, 4.14, 1.62, 0.45, "PCA + 条件 VAE", "6 主成分 / 3× 合成", COLORS["green_fill"], COLORS["green"])
    _box(ax, 0.52, 2.92, 1.62, 0.72, "训练样本", "40 真 + 40 噪\n+ 120 合成", COLORS["green_fill"], COLORS["green"])
    _box(ax, 0.52, 1.68, 1.62, 0.55, "验证 / 测试", "10 + 12 条真实曲线", "white", COLORS["green"], dashed=True, body_size=5.8)
    _arrow(ax, (1.33, 5.78), (1.33, 5.39), width=0.7)
    _arrow(ax, (1.33, 4.90), (1.33, 4.60), width=0.7)
    _arrow(ax, (1.33, 4.12), (1.33, 3.66), width=0.7)

    # 特征提取
    _box(ax, 2.78, 5.78, 1.60, 0.50, "输入 x", "raw_delta  6×51", COLORS["blue_fill"], COLORS["pink"], body_size=5.8)
    block_y = [4.95, 4.10, 3.25]
    for y, spec in zip(block_y, specs, strict=True):
        _box(
            ax,
            2.78,
            y,
            1.60,
            0.62,
            f"因果残差块 {spec['index']}",
            f"d={spec['dilation']} · 16 通道\nk={spec['kernel']} · Dropout {spec['dropout']:.1f}",
            COLORS["pink_fill"],
            COLORS["pink"],
            title_size=6.7,
            body_size=5.6,
        )
    _box(ax, 2.78, 2.38, 1.60, 0.48, "隐藏序列 H", "[B, 16, 51]", COLORS["pink_fill"], COLORS["pink"], body_size=5.8)
    _arrow(ax, (3.58, 5.76), (3.58, 5.59), width=0.7)
    for upper, lower in pairwise(block_y):
        _arrow(ax, (3.58, upper), (3.58, lower + 0.64), width=0.7)
    _arrow(ax, (3.58, block_y[-1]), (3.58, 2.88), width=0.7)

    # 从数据组到特征组
    ax.plot([2.15, 2.36, 2.36], [3.28, 3.28, 6.03], color=COLORS["arrow"], linewidth=0.8, zorder=2)
    _arrow(ax, (2.36, 6.03), (2.76, 6.03), width=0.8)

    # 末时刻与输出头
    _box(ax, 4.52, 2.38, 0.90, 0.48, "末时刻", "h[:, :, −1]", COLORS["blue_fill"], COLORS["pink"], title_size=6.0, body_size=5.4)
    _arrow(ax, (4.40, 2.62), (4.50, 2.62), width=0.7)
    _box(
        ax,
        5.72,
        5.55,
        1.62,
        0.62,
        "训练协议",
        "5 折 OOF\n3 seed 预测平均",
        COLORS["purple_fill"],
        COLORS["purple"],
        dashed=True,
        title_size=6.2,
        body_size=5.2,
    )
    _box(
        ax,
        5.72,
        4.50,
        1.62,
        0.62,
        "推理",
        "只用 10 秒主输出\n反 z-score",
        COLORS["purple_fill"],
        COLORS["purple"],
        dashed=True,
        title_size=6.2,
        body_size=5.2,
    )
    _box(ax, 5.72, 2.95, 1.62, 0.70, "主输出头", f"Linear {linears[0].in_features}→{linears[0].out_features}\nReLU → Linear 16→3", COLORS["yellow_fill"], COLORS["purple"], title_size=6.7, body_size=5.5)
    _arrow(ax, (5.44, 2.62), (5.70, 2.62), width=0.7)
    _arrow(ax, (6.53, 2.93), (6.53, 2.35), width=0.7)
    _box(ax, 5.72, 1.70, 1.62, 0.56, "输出浓度", "H2 / NH3 / NO2", COLORS["purple_fill"], COLORS["purple"], title_size=6.7, body_size=5.8)
    ax.text(3.58, 2.18, f"感受野 {receptive} 点", ha="center", va="center", fontproperties=_font(5.8), color=COLORS["pink"])
    ax.text(4.08, 0.42, "(a) GasTCN 总览", ha="center", va="center", fontproperties=_font(8.4, "bold"), color=COLORS["text"])

    # ---------- 面板 (b) 残差块 ----------
    _panel_border(ax, 8.34, 4.02, 3.50, 3.26)
    ax.text(8.48, 7.02, "残差块内部", ha="left", va="center", fontproperties=_font(7.2, "bold"), color=COLORS["pink"])
    mini = [
        (6.52, "输入 x", "[B, C, L]", COLORS["blue_fill"]),
        (6.10, "因果 Conv1d", "k=5, d", COLORS["pink_fill"]),
        (5.68, "ReLU + Dropout", "0.1", COLORS["pink_fill"]),
        (5.26, "因果 Conv1d", "k=5, d", COLORS["pink_fill"]),
        (4.84, "ReLU + Dropout", "0.1", COLORS["pink_fill"]),
        (4.42, "相加 + ReLU", "残差", COLORS["yellow_fill"]),
    ]
    for y, title, body, color in mini:
        _box(ax, 8.52, y, 1.55, 0.30, title, body, color, COLORS["pink"], title_size=5.2, body_size=4.6, linewidth=0.6)
    for (y1, *_), (y2, *_) in pairwise(mini):
        _arrow(ax, (9.30, y1), (9.30, y2 + 0.31), width=0.55, head=5.5)
    ax.plot([10.30, 10.88, 10.88], [6.67, 6.67, 4.57], color=COLORS["pink"], linewidth=0.7, linestyle="--")
    _arrow(ax, (10.88, 4.57), (10.10, 4.57), color=COLORS["pink"], width=0.7, head=5.5)
    ax.text(10.92, 5.62, "跳连\n1×1 Conv / Identity", ha="left", va="center", fontproperties=_font(4.8), color=COLORS["pink"])
    ax.text(8.48, 4.18, "左 padding = (k−1)d，只看过去", ha="left", va="center", fontproperties=_font(5.0), color=COLORS["muted"])
    ax.text(10.09, 3.88, "(b) 残差块", ha="center", va="center", fontproperties=_font(8.0, "bold"), color=COLORS["text"])

    # ---------- 面板 (c) 因果膨胀卷积 ----------
    _panel_border(ax, 8.34, 0.20, 3.50, 3.56)
    ax.text(8.48, 3.52, "因果膨胀卷积", ha="left", va="center", fontproperties=_font(7.2, "bold"), color=COLORS["pink"])
    x_start, x_step = 8.80, 0.21
    rows = [(2.92, 1, 5, "d=1"), (2.12, 2, 9, "d=2"), (1.32, 4, 17, "d=4")]
    for y, dilation, span, label in rows:
        for index in range(9):
            color = "#BFC8D0" if index < 8 - (span - 1) // 2 else COLORS["pink"]
            ax.add_patch(Rectangle((x_start + index * x_step, y), 0.09, 0.09, facecolor=color, edgecolor="none", zorder=2))
        ax.text(x_start - 0.12, y + 0.04, label, ha="right", va="center", fontproperties=_font(5.2), color=COLORS["muted"])
        ax.text(x_start + 9 * x_step + 0.08, y + 0.04, f"{span} 点", ha="left", va="center", fontproperties=_font(4.8), color=COLORS["pink"])
    ax.text(8.48, 3.28, "灰点：左侧补零 / 粉色：当前输出可见范围", ha="left", va="center", fontproperties=_font(4.8), color=COLORS["muted"])
    ax.text(8.48, 0.82, f"RF = 1 + 2(k−1)(1+2+4) = {receptive} 点", ha="left", va="center", fontproperties=_font(5.4), color=COLORS["text"])
    ax.text(8.48, 0.60, "主输出读取 t=10 s；因果卷积保证不访问未来", ha="left", va="center", fontproperties=_font(4.9), color=COLORS["muted"])
    ax.text(10.09, 0.36, "(c) 因果卷积", ha="center", va="center", fontproperties=_font(8.0, "bold"), color=COLORS["text"])

    # 标题
    ax.text(6.0, 7.38, "GasTCN：三气体 10 秒早期响应预测模型", ha="center", va="center", fontproperties=_font(11.0, "bold"), color=COLORS["text"])

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制参照论文版式的 GasTCN 架构图")
    parser.add_argument("--output", type=Path, default=Path("figures/fig_tcn_architecture.png"))
    args = parser.parse_args()
    for path in draw_architecture(args.output):
        print(path)


if __name__ == "__main__":
    main()
