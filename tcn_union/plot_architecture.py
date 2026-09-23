"""绘制 GasTCN 的当前网络结构，并导出 PNG 与矢量 PDF。

图中结构从 `GasTCN` 实例读取卷积核、膨胀率、通道数、Dropout 和线性头，
避免手写参数与实际模型失配。所有中文文字强制使用 Noto Sans CJK SC。
"""
from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.patheffects import Normal, Stroke
from torch import nn

from .model import GasTCN

FONT_FAMILY = "Noto Sans CJK SC"
FONT_FILE: str | None = None
COLORS = {
    "input": "#EAF2F8",
    "block": "#E8F2EE",
    "head": "#FFF2DF",
    "output": "#F0EAF8",
    "detail": "#F7F8FA",
    "edge": "#52616B",
    "text": "#263238",
    "accent": "#2A6F77",
}


def _configure_font() -> FontProperties:
    """验证并配置所需中文字体。

    返回：绑定到 Noto Sans CJK SC 字体文件的 `FontProperties`。字体不存在时
    立即报错，避免 Matplotlib 静默回退后生成缺字或错误字体的图片。
    """
    global FONT_FILE
    font = FontProperties(family=FONT_FAMILY, weight=100)
    FONT_FILE = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.sans-serif": [FONT_FAMILY],
            "font.weight": 100,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
        }
    )
    return FontProperties(family=FONT_FAMILY, fname=FONT_FILE)


def _font_properties(size: float) -> FontProperties:
    """创建绑定到 Noto Sans CJK SC 字体文件的 Matplotlib 字体属性。"""
    if FONT_FILE is None:
        raise RuntimeError("configure the Noto Sans CJK SC font before drawing text")
    return FontProperties(family=FONT_FAMILY, fname=FONT_FILE, size=size, weight=100)


def _add_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    *,
    fontsize: float = 9.2,
    title_size: float = 11.0,
    linewidth: float = 1.2,
) -> None:
    """在架构图指定位置绘制圆角模块框和中英文标签。

    参数：`axis` 为目标坐标轴；`x/y` 是左下角坐标；`width/height` 是框尺寸；
    `title/body` 分别是模块标题和说明；`facecolor` 是填充色；其余关键字参数
    控制正文字号、标题字号及边线宽度。无返回值。
    """
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=facecolor,
        edgecolor=COLORS["edge"],
        linewidth=linewidth,
        zorder=2,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height * 0.76,
        title,
        ha="center",
        va="center",
        fontproperties=_font_properties(title_size),
        color=COLORS["text"],
        path_effects=[Stroke(linewidth=0.34, foreground=COLORS["text"]), Normal()],
        linespacing=1.15,
        zorder=3,
    )
    axis.text(
        x + width / 2,
        y + height * 0.39,
        body,
        ha="center",
        va="center",
        fontproperties=_font_properties(fontsize),
        color=COLORS["text"],
        path_effects=[Stroke(linewidth=0.30, foreground=COLORS["text"]), Normal()],
        linespacing=1.25,
        zorder=3,
    )


def _add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    linewidth: float = 1.55,
    style: str = "-|>",
    connectionstyle: str = "arc3,rad=0",
) -> None:
    """绘制带箭头的数据流连线。

    参数：`axis` 为目标坐标轴；`start/end` 是箭头起止坐标；可选颜色、线宽、
    箭头样式和曲线连接样式。无返回值。
    """
    axis.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=13,
            linewidth=linewidth,
            color=color or COLORS["edge"],
            connectionstyle=connectionstyle,
            zorder=1,
        )
    )


def draw_architecture(output_path: Path) -> tuple[Path, Path]:
    """根据当前 `GasTCN` 实例绘制架构图并保存 PNG/PDF。

    参数：`output_path` 是 PNG 输出路径；同一目录下会写入同名 PDF。
    返回：`(png_path, pdf_path)`。架构框内的层参数从模型对象读取；同时显示
    5/10 秒输入形状、残差块细节、逐点跳连、末时刻读取和浓度反标准化。
    """
    _configure_font()
    model = GasTCN(in_channels=6)
    blocks = list(model.blocks)
    if len(blocks) != 3:
        raise ValueError(f"expected three residual blocks, found {len(blocks)}")

    block_specs: list[dict[str, int | float | str]] = []
    for index, block in enumerate(blocks):
        conv1 = block.conv1
        conv2 = block.conv2
        dropout1 = block.dropout1
        if not isinstance(conv1, nn.Conv1d) or not isinstance(conv2, nn.Conv1d):
            raise TypeError(f"residual block {index + 1} must contain two Conv1d layers")
        if not isinstance(dropout1, nn.Dropout):
            raise TypeError(f"residual block {index + 1} must contain Dropout layers")
        kernel = int(conv1.kernel_size[0])
        dilation_value = block.dilation
        if not isinstance(dilation_value, int):
            raise TypeError(f"residual block {index + 1} dilation must be an integer")
        dilation = dilation_value
        if int(conv2.kernel_size[0]) != kernel or int(conv2.dilation[0]) != dilation:
            raise ValueError(f"residual block {index + 1} uses inconsistent convolutions")
        skip = "1×1 投影" if isinstance(block.projection, nn.Conv1d) else "Identity"
        block_specs.append(
            {
                "index": index + 1,
                "kernel": kernel,
                "dilation": dilation,
                "padding": (kernel - 1) * dilation,
                "input_channels": int(conv1.in_channels),
                "hidden_channels": int(conv1.out_channels),
                "output_channels": int(conv2.out_channels),
                "dropout": float(dropout1.p),
                "skip": skip,
            }
        )

    receptive_field = 1 + 2 * sum(
        (int(spec["kernel"]) - 1) * int(spec["dilation"]) for spec in block_specs
    )
    linear_layers = [layer for layer in model.head if isinstance(layer, nn.Linear)]
    if len(linear_layers) != 2:
        raise ValueError(f"expected two linear head layers, found {len(linear_layers)}")
    first_linear, final_linear = linear_layers

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(20, 9.2))
    axis.set_xlim(0, 19.25)
    axis.set_ylim(0, 9.0)
    axis.set_axis_off()
    figure.suptitle(
        "GasTCN：三气体早期响应预测架构",
        fontproperties=_font_properties(19),
        color=COLORS["text"],
        path_effects=[Stroke(linewidth=0.40, foreground=COLORS["text"]), Normal()],
        y=0.965,
    )
    axis.text(
        9.6,
        8.26,
        f"因果输入前缀  ·  感受野 {receptive_field} 点  ·  末时刻特征回归",
        ha="center",
        va="center",
        fontproperties=_font_properties(11.5),
        color=COLORS["accent"],
        path_effects=[Stroke(linewidth=0.34, foreground=COLORS["accent"]), Normal()],
    )

    top_y, top_h, flow_y = 4.85, 2.45, 6.075
    input_x, input_w = 0.32, 2.0
    block_xs, block_w = [2.60, 5.48, 8.36], 2.72
    last_x, last_w = 11.24, 1.72
    head_x, head_w = 13.24, 2.30
    output_x, output_w = 15.70, 3.18

    _add_box(
        axis,
        input_x,
        top_y,
        input_w,
        top_h,
        "传感器前缀",
        "raw_delta：C=6\ndelta：C=3\nL=26（5 s）/ 51（10 s）",
        COLORS["input"],
    )
    for x, spec in zip(block_xs, block_specs, strict=True):
        channels = (
            f"C→{spec['hidden_channels']}→{spec['output_channels']}"
            if int(spec["index"]) == 1
            else f"16→{spec['hidden_channels']}→{spec['output_channels']}"
        )
        body = (
            f"2× Conv1d(k={spec['kernel']}, d={spec['dilation']})\n"
            f"通道：{channels}\n"
            f"每层左填充 p={spec['padding']}\n"
            f"ReLU + Dropout({spec['dropout']:.1f})\n"
            f"跳连：{spec['skip']}"
        )
        _add_box(
            axis,
            x,
            top_y,
            block_w,
            top_h,
            f"残差块 {spec['index']}（d={spec['dilation']}）",
            body,
            COLORS["block"],
            fontsize=8.5,
            title_size=10.4,
        )
    _add_box(
        axis,
        last_x,
        top_y,
        last_w,
        top_h,
        "末时刻读取",
        "h[:, :, −1]\n[B,16]",
        COLORS["input"],
        fontsize=9.0,
    )
    _add_box(
        axis,
        head_x,
        top_y,
        head_w,
        top_h,
        "回归头",
        f"Linear({first_linear.in_features}→{first_linear.out_features})\n"
        f"ReLU\nLinear({final_linear.in_features}→{final_linear.out_features})",
        COLORS["head"],
        fontsize=9.0,
    )
    _add_box(
        axis,
        output_x,
        top_y,
        output_w,
        top_h,
        "三气体浓度",
        "标准化输出 [B,3]\nH2 · NH3 · NO2\n反 z-score → 原始单位",
        COLORS["output"],
        fontsize=9.0,
    )

    node_edges = [
        (input_x + input_w, block_xs[0]),
        (block_xs[0] + block_w, block_xs[1]),
        (block_xs[1] + block_w, block_xs[2]),
        (block_xs[2] + block_w, last_x),
        (last_x + last_w, head_x),
        (head_x + head_w, output_x),
    ]
    for start_x, end_x in node_edges:
        _add_arrow(axis, (start_x + 0.035, flow_y), (end_x - 0.035, flow_y))

    panel = FancyBboxPatch(
        (0.32, 0.45),
        18.56,
        3.75,
        boxstyle="round,pad=0.04,rounding_size=0.12",
        facecolor=COLORS["detail"],
        edgecolor="#D4DCE2",
        linewidth=1.0,
        zorder=0,
    )
    axis.add_patch(panel)
    axis.text(
        0.62,
        3.91,
        "残差块内部数据流（两层卷积使用相同膨胀率）",
        ha="left",
        va="center",
        fontproperties=_font_properties(11.5),
        color=COLORS["text"],
        path_effects=[Stroke(linewidth=0.34, foreground=COLORS["text"]), Normal()],
    )

    detail_y, detail_h = 2.42, 0.82
    details = [
        (0.72, 1.28, "块输入 x", "[B,C,L]", COLORS["input"]),
        (
            2.32,
            2.85,
            "因果 Conv1d",
            "k=5，膨胀率 d\n左填充 p=(k−1)d",
            COLORS["block"],
        ),
        (5.43, 2.05, "激活与正则化", "ReLU → Dropout(0.1)", COLORS["block"]),
        (
            7.74,
            2.85,
            "因果 Conv1d",
            "k=5，膨胀率 d\n左填充 p=(k−1)d",
            COLORS["block"],
        ),
        (10.85, 2.05, "激活与正则化", "ReLU → Dropout(0.1)", COLORS["block"]),
        (13.22, 2.48, "残差合并", "逐点相加 → ReLU", COLORS["head"]),
        (16.02, 2.28, "块输出", "[B,16,L]", COLORS["output"]),
    ]
    for x, width, title, body, color in details:
        _add_box(
            axis,
            x,
            detail_y,
            width,
            detail_h,
            title,
            body,
            color,
            fontsize=8.2,
            title_size=8.8,
            linewidth=0.9,
        )
    detail_centers = [(x, width) for x, width, *_ in details]
    for (x1, w1), (x2, _) in pairwise(detail_centers):
        _add_arrow(
            axis,
            (x1 + w1 + 0.035, detail_y + detail_h / 2),
            (x2 - 0.035, detail_y + detail_h / 2),
            linewidth=1.15,
        )

    skip_spec = block_specs[0]
    skip_text = (
        "跳连由块输入 x 分出：块 1 使用 1×1 Conv（C∈{3,6}→"
        f"{skip_spec['hidden_channels']}）投影；块 2/3 使用 Identity；结果与主支路逐点相加。"
    )
    _add_box(
        axis,
        2.32,
        1.18,
        10.05,
        0.62,
        "跳连支路",
        skip_text,
        "#FFFFFF",
        fontsize=8.4,
        title_size=8.8,
        linewidth=0.8,
    )
    _add_arrow(axis, (1.36, detail_y - 0.03), (2.50, 1.83), linewidth=1.05)
    _add_arrow(axis, (12.39, 1.49), (14.25, detail_y - 0.03), linewidth=1.05)

    padding_text = ", ".join(f"d={s['dilation']}→p={s['padding']}" for s in block_specs)
    axis.text(
        0.62,
        0.77,
        f"左侧填充（每层）：{padding_text}    |    感受野：1 + 2(k−1)Σd = {receptive_field} 点（覆盖 10 秒的 51 点输入）",
        ha="left",
        va="center",
        fontproperties=_font_properties(9.2),
        color=COLORS["text"],
        path_effects=[Stroke(linewidth=0.30, foreground=COLORS["text"]), Normal()],
    )

    figure.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return png_path, pdf_path


def main() -> None:
    """解析输出路径，绘制网络图并打印 PNG/PDF 文件位置。"""
    parser = argparse.ArgumentParser(description="绘制当前 GasTCN 模型架构")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/tcn_architecture.png"),
        help="PNG 输出路径；同路径名同时生成 PDF",
    )
    args = parser.parse_args()
    for path in draw_architecture(args.output):
        print(path)


if __name__ == "__main__":
    main()
