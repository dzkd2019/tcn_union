"""绘图样式、配色与保存（PNG + SVG）。"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..paths import FIGURES

GAS_LABEL = {"H2": "H$_2$", "NH3": "NH$_3$", "NO2": "NO$_2$"}
GAS_COLORS = {"H2": "#C0392B", "NH3": "#E69F00", "NO2": "#0072B2"}
MODEL_COLORS = {
    "mean_curve": "#8C8C8C",
    "direct_ridge": "#E69F00",
    "two_stage": "#CC79A7",
    "tcn": "#6A4C93",
    "grey_box": "#009E73",
    "hybrid": "#4C72B0",
}
MODEL_LABELS = {
    "mean_curve": "均值曲线",
    "direct_ridge": "直接岭回归",
    "two_stage": "两阶段回归",
    "tcn": "TCN",
    "grey_box": "灰盒",
    "hybrid": "混合（本文）",
}
MODEL_ORDER = ["mean_curve", "direct_ridge", "two_stage", "tcn", "grey_box", "hybrid"]


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Noto Serif CJK SC"],
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


def _save(figure: plt.Figure, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.png", dpi=300)
    figure.savefig(FIGURES / f"{name}.svg")
    plt.close(figure)
    print("saved", FIGURES / f"{name}.png")


def _letter(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", fontsize=9.5)
