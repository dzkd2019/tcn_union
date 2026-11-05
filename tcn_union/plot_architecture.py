"""绘制论文版 GasTCN 架构与训练/推理流程图。

图分成四块：
(a) 逐折训练数据构造：真实响应、测量噪声、PCA 条件 VAE；
(b) GasTCN 主结构：输入、三个因果残差块、10 秒主输出；
(c) 单个残差块的内部数据流与因果填充；
(d) 五折 OOF、三 seed 重训与推理协议。

图中的卷积核、膨胀率、通道数、Dropout、线性头维度均从 ``GasTCN`` 实例读取，
避免图和代码不一致。输出 PNG 与矢量 PDF。
"""
from __future__ import annotations

import argparse
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from torch import nn

from .model import GasTCN

FONT_FAMILY = "Noto Serif CJK SC"
FONT_FILE: str | None = None

COLORS = {
    "data": "#EAF2F8",
    "noise": "#FFF3E0",
    "vae": "#E8F2EE",
    "merge": "#F3F0E8",
    "model": "#E8EDF2",
    "detail": "#F7F8FA",
    "head": "#FFF2DF",
    "output": "#F0EAF8",
    "protocol": "#EEF2F0",
    "test": "#F5F5F5",
    "edge": "#3D4A57",
    "text": "#1F2933",
    "muted": "#5B6770",
    "accent": "#2A6F77",
    "aux": "#9A6B2F",
}
W, H = 7.2, 9.9


def _configure_font() -> FontProperties:
    global FONT_FILE
    font = FontProperties(family=FONT_FAMILY)
    FONT_FILE = font_manager.findfont(font, fallback_to_default=False)
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.serif": [FONT_FAMILY],
            "font.size": 8,
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
        raise RuntimeError("configure the font before drawing")
    return FontProperties(
        family=FONT_FAMILY, fname=FONT_FILE, size=size, weight=weight
    )


def _box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    color: str,
    *,
    dashed: bool = False,
    title_size: float = 8.2,
    body_size: float = 7.1,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.09",
            facecolor=color,
            edgecolor=COLORS["edge"],
            linewidth=0.85,
            linestyle="--" if dashed else "-",
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h * 0.70,
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
        linespacing=1.25,
        zorder=4,
    )


def _arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str | None = None,
    dashed: bool = False,
    rad: float = 0.0,
    width: float = 1.0,
    head: float = 8.0,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=head,
            linewidth=width,
            linestyle="--" if dashed else "-",
            color=color or COLORS["edge"],
            connectionstyle=f"arc3,rad={rad}",
            zorder=2,
        )
    )


def _panel_title(ax: plt.Axes, x: float, y: float, text: str) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontproperties=_font(9.6, "bold"),
        color=COLORS["text"],
    )


def _read_model_specs() -> dict[str, object]:
    model = GasTCN(in_channels=6)
    blocks = list(model.blocks)
    specs = []
    for index, block in enumerate(blocks):
        conv1, conv2 = block.conv1, block.conv2
        if not isinstance(conv1, nn.Conv1d) or not isinstance(conv2, nn.Conv1d):
            raise TypeError("residual block must use Conv1d")
        specs.append(
            {
                "index": index + 1,
                "kernel": int(conv1.kernel_size[0]),
                "dilation": int(block.dilation),
                "in_channels": int(conv1.in_channels),
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
    return {
        "specs": specs,
        "linears": linears,
        "receptive": receptive,
    }


def draw_architecture(output_path: Path) -> tuple[Path, Path]:
    _configure_font()
    info = _read_model_specs()
    specs = info["specs"]
    linears = info["linears"]
    receptive = int(info["receptive"])

    png_path = output_path.with_suffix(".png")
    pdf_path = output_path.with_suffix(".pdf")
    png_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------
    # (a) 数据构造
    # ------------------------------------------------------------
    _panel_title(ax, 0.18, 9.55, "(a) 逐折训练数据构造")
    ax.text(
        0.18,
        9.23,
        "每折只用 40 个训练实验拟合噪声统计和 PCA-VAE；验证/测试保持真实曲线",
        fontproperties=_font(6.7),
        color=COLORS["muted"],
    )
    _box(ax, 0.18, 8.28, 1.28, 0.63, "真实响应", "50 条开发实验\n6×51（10 s）", COLORS["data"])
    _box(ax, 1.78, 8.58, 1.62, 0.48, "测量噪声", "0.1× MAD 估计", COLORS["noise"])
    _box(ax, 1.78, 8.00, 1.62, 0.48, "PCA + 条件 VAE", "6 PCs / 3× 合成", COLORS["vae"])
    _box(
        ax,
        3.76,
        8.10,
        1.60,
        0.80,
        "每折训练集",
        "40 真实 + 40 噪声\n+ 120 合成",
        COLORS["merge"],
    )
    _box(
        ax,
        5.78,
        8.10,
        1.22,
        0.80,
        "验证 / 测试",
        "10 + 12 条\n真实曲线",
        COLORS["test"],
        dashed=True,
    )
    _arrow(ax, (1.49, 8.58), (1.73, 8.75), width=0.8)
    _arrow(ax, (1.49, 8.45), (1.73, 8.23), width=0.8)
    _arrow(ax, (3.43, 8.80), (3.72, 8.60), width=0.8)
    _arrow(ax, (3.43, 8.23), (3.72, 8.40), width=0.8)

    # ------------------------------------------------------------
    # (b) GasTCN 主结构
    # ------------------------------------------------------------
    _panel_title(ax, 0.18, 7.42, "(b) GasTCN：因果 TCN 与 10 秒主读出")
    ax.text(
        0.18,
        7.10,
        f"3 个残差块 · 16 通道 · k={int(specs[0]['kernel'])} · d=1/2/4 · 末时刻回归 · 感受野 {receptive} 点",
        fontproperties=_font(6.6),
        color=COLORS["muted"],
    )
    _box(ax, 0.18, 5.95, 1.02, 0.78, "输入 x", "raw_delta\n6×51（10 s）", COLORS["data"])
    block_x = [1.42, 2.72, 4.02]
    for x, spec in zip(block_x, specs, strict=True):
        _box(
            ax,
            x,
            5.88,
            1.08,
            0.92,
            f"残差块 {spec['index']}",
            f"Conv1d k={spec['kernel']}\nd={spec['dilation']} · 16 ch\nDropout {spec['dropout']:.1f}",
            COLORS["model"],
            title_size=7.8,
            body_size=6.4,
        )
    _box(
        ax,
        5.35,
        5.95,
        0.96,
        0.78,
        "主输出头",
        f"Linear {linears[0].in_features}→{linears[0].out_features}\n"
        f"ReLU\nLinear {linears[1].in_features}→{linears[1].out_features}",
        COLORS["head"],
        title_size=7.6,
        body_size=6.2,
    )
    _box(
        ax,
        6.38,
        5.95,
        0.68,
        0.78,
        "三气体",
        "10 s\nH2/NH3/NO2",
        COLORS["output"],
        title_size=7.0,
        body_size=5.8,
    )

    _box(
        ax,
        3.90,
        4.96,
        1.38,
        0.48,
        "5 s 辅助头（可选）",
        "训练时读取 index 25",
        COLORS["test"],
        dashed=True,
        title_size=6.8,
        body_size=5.8,
    )
    _arrow(ax, (4.58, 5.87), (4.58, 5.47), dashed=True, color=COLORS["aux"], width=0.7, head=6.0)
    ax.text(4.68, 5.62, "辅助损失", fontproperties=_font(5.4), color=COLORS["aux"])
    ax.text(5.35, 5.20, "最终方案未启用", fontproperties=_font(5.8), color=COLORS["aux"])

    for x1, x2 in [(1.20, 1.40), (2.50, 2.70), (3.80, 4.00), (5.10, 5.28), (6.28, 6.36)]:
        _arrow(ax, (x1 + 0.01, 6.34), (x2 - 0.01, 6.34), width=0.9)
    ax.text(2.58, 6.94, "隐藏序列 H ∈ R^{16×51}", ha="center", fontproperties=_font(6.5), color=COLORS["accent"])

    # 数据到模型：绕到面板标题下方，避免穿过 (b) 标题
    ax.plot([4.56, 4.56, 0.69], [8.08, 7.76, 7.76], color=COLORS["edge"], linewidth=0.9, zorder=2)
    _arrow(ax, (0.69, 7.76), (0.69, 6.75), width=0.9)
    ax.text(2.42, 7.67, "训练特征", fontproperties=_font(6.2), color=COLORS["muted"])

    # ------------------------------------------------------------
    # (c) 残差块内部
    # ------------------------------------------------------------
    ax.add_patch(
        FancyBboxPatch(
            (0.18, 3.18),
            6.84,
            1.50,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=COLORS["detail"],
            edgecolor="#D5DCE3",
            linewidth=0.7,
            zorder=1,
        )
    )
    _panel_title(ax, 0.30, 4.55, "(c) 单个残差块内部")
    detail = [
        (0.32, 0.72, "x", "[B,C,L]", COLORS["data"]),
        (1.18, 1.00, "因果 Conv1d", "左 padding\n(k−1)d", COLORS["model"]),
        (2.36, 0.82, "ReLU", "Dropout", COLORS["model"]),
        (3.32, 1.00, "因果 Conv1d", "同 k、同 d", COLORS["model"]),
        (4.50, 0.82, "ReLU", "Dropout", COLORS["model"]),
        (5.48, 0.68, "相加", "残差", COLORS["head"]),
        (6.30, 0.60, "输出", "[B,16,L]", COLORS["output"]),
    ]
    for x, w, title, body, color in detail:
        _box(ax, x, 3.72, w, 0.55, title, body, color, title_size=6.6, body_size=5.7)
    for (x1, w1, *_), (x2, *_) in pairwise(detail):
        _arrow(ax, (x1 + w1 + 0.01, 4.00), (x2 - 0.01, 4.00), width=0.75, head=6.5)
    ax.plot([0.68, 0.68, 5.82, 5.82], [3.64, 3.50, 3.50, 3.64], color=COLORS["aux"], linewidth=0.8, linestyle="--")
    ax.text(3.25, 3.39, "跳连：块 1 用 1×1 Conv；块 2/3 用 Identity", ha="center", fontproperties=_font(5.6), color=COLORS["aux"])
    ax.text(
        0.30,
        3.26,
        f"每层左填充 p=(k−1)d，禁止访问未来；RF = 1 + 2(k−1)Σd = {receptive} 点",
        fontproperties=_font(5.4),
        color=COLORS["muted"],
    )

    # ------------------------------------------------------------
    # (d) 训练 / 推理协议
    # ------------------------------------------------------------
    _panel_title(ax, 0.18, 2.82, "(d) 五折 OOF、三 seed 重训与推理")
    protocol = [
        (0.18, 1.12, "5 折 OOF", "40 训 / 10 验", COLORS["protocol"]),
        (1.46, 1.18, "早停", "验证 macro-NMAE", COLORS["protocol"]),
        (2.80, 1.10, "轮数", "5 折最佳 epoch\n取中位数", COLORS["protocol"]),
        (4.06, 1.12, "全开发集重训", "50 条 × 3 seeds", COLORS["protocol"]),
        (5.34, 1.14, "三 seed 集成", "预测平均", COLORS["head"]),
        (6.64, 0.38, "输出", "y", COLORS["output"]),
    ]
    for x, w, title, body, color in protocol:
        _box(ax, x, 1.87, w, 0.66, title, body, color, title_size=6.8, body_size=5.9)
    for (x1, w1, *_), (x2, *_) in pairwise(protocol):
        _arrow(ax, (x1 + w1 + 0.01, 2.20), (x2 - 0.01, 2.20), width=0.75, head=6.5)
    ax.text(0.20, 1.54, "锁定 12 个测试实验：方案冻结后才评估；推理只使用 10 秒主输出，不读取未来采样。", fontproperties=_font(6.4), color=COLORS["muted"])
    ax.text(0.20, 1.29, "增强只作用于训练折；验证/测试始终使用真实曲线。", fontproperties=_font(6.4), color=COLORS["muted"])
    ax.text(0.20, 0.82, "输入 6 通道 = 三路原始响应 + 三路相对增量；输出反 z-score 到原始浓度单位。", fontproperties=_font(6.1), color=COLORS["muted"])
    ax.text(0.20, 0.55, "所有卷积只在左侧补零；主输出读取 t=10 s 的最后一个隐藏位置。", fontproperties=_font(6.1), color=COLORS["muted"])

    figure_title = "GasTCN：三气体 10 秒早期响应预测模型与训练流程"
    ax.text(
        W / 2,
        9.83,
        figure_title,
        ha="center",
        va="center",
        fontproperties=_font(11.2, "bold"),
        color=COLORS["text"],
    )

    fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制论文版 GasTCN 架构与算法图")
    parser.add_argument("--output", type=Path, default=Path("figures/fig_tcn_architecture.png"))
    args = parser.parse_args()
    for path in draw_architecture(args.output):
        print(path)


if __name__ == "__main__":
    main()
