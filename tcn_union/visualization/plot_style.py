"""绘图统一字体与输出设置。

新绘制的 3σ 相关图统一使用该设置：优先思源黑体，回退到本机常见中文字体；
仅输出 PNG 与 SVG（不输出 PDF）。
"""
from __future__ import annotations

import matplotlib.pyplot as plt

CJK_FONTS = ["Source Han Sans CN", "Microsoft YaHei", "SimHei", "DejaVu Sans"]


def apply_style() -> None:
    """设置中文 sans-serif 字体与 SVG 文本可编辑选项。"""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": CJK_FONTS,
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
