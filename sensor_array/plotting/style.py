"""绘图样式、配色与保存（PNG + SVG）。"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import FIGURES_MODEL

GAS_LABEL = {"H2": r"$\mathrm{H}_2$", "NH3": r"$\mathrm{NH}_3$", "NO2": r"$\mathrm{NO}_2$"}
GAS_COLORS = {"H2": "#C0392B", "NH3": "#E69F00", "NO2": "#0072B2"}
MODEL_COLORS = {
    "double": "#4C72B0",
    "mlp_per_gas": "#009E73",
}
MODEL_LABELS = {
    "double": "双指数骨架",
    "mlp_per_gas": "三路独立 MLP + 双指数",
}
MODEL_ORDER = ["double", "mlp_per_gas"]

# Noto Serif 覆盖拉丁字母与 H₂ 等下标，Source Han Sans CN 覆盖中文；
# 单用 SimSun 在多面板图里会丢部分 CJK 字形。
FONT_FAMILY = ["Noto Serif CJK SC"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "axes.unicode_minus": False,
            "mathtext.fontset": "stix",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "savefig.bbox": "tight",
            "figure.dpi": 120,
        }
    )


def _save(figure: plt.Figure, name: str, directory: Path = FIGURES_MODEL) -> None:
    """保存 PNG 300 dpi + SVG；默认写入当前模型的图目录。"""
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=300)
    figure.savefig(directory / f"{name}.svg")
    plt.close(figure)
    print("saved", directory / f"{name}.png")