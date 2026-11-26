"""10 秒前缀 TCN 的相对误差与均值 95% 置信区间。

点：每个实验的相对误差；黑色菱形/竖线：该组的均值及 95% t 置信区间。
OOF 50 与锁定 12 分开绘制，避免把训练内与一次性测试混在一起。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/3sigma_noridge"
PREDICTIONS = BASE / "all62_predictions.csv"
FIGS = BASE / "figs"

GASES = ("H2", "NH3", "NO2")
COLORS = {"H2": "#1F77B4", "NH3": "#D98E04", "NO2": "#C0392B"}
SCOPES = (("OOF", "OOF 50", "o", 0.45), ("测试", "锁定 12", "^", 0.85))

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def _relative(frame: pd.DataFrame, gas: str) -> np.ndarray:
    truth = frame[f"{gas}_true"].to_numpy(float)
    pred = frame[f"{gas}_pred"].to_numpy(float)
    return 100.0 * (pred - truth) / np.maximum(np.abs(truth), 1e-9)


def _ci95(values: np.ndarray) -> tuple[float, float]:
    n = len(values)
    mean = float(values.mean())
    half = float(stats.t.ppf(0.975, n - 1) * values.std(ddof=1) / np.sqrt(n))
    return mean, half


def main() -> None:
    frame = pd.read_csv(PREDICTIONS)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=False)
    for ax, gas in zip(axes, GASES):
        for position, (source, label, marker, alpha) in enumerate(SCOPES):
            subset = frame[frame["source"] == source]
            values = _relative(subset, gas)
            jitter = rng.uniform(-0.10, 0.10, size=len(values))
            ax.scatter(
                position + jitter,
                values,
                s=22,
                marker=marker,
                color=COLORS[gas],
                alpha=alpha,
                edgecolor="white",
                linewidth=0.3,
                zorder=2,
            )
            mean, half = _ci95(values)
            ax.errorbar(
                position,
                mean,
                yerr=half,
                fmt="D",
                color="black",
                markersize=4.5,
                capsize=5,
                linewidth=1.2,
                zorder=3,
            )
            ax.annotate(
                f"n={len(values)}\n均值 {mean:+.2f}%\n95% CI [{mean - half:+.2f}, {mean + half:+.2f}]",
                xy=(position, 1.0),
                xycoords=("data", "axes fraction"),
                xytext=(0, -8),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=5.8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.82),
                zorder=4,
            )
        ax.axhline(0.0, color="0.25", linewidth=0.8, zorder=1)
        ax.set_xticks([0, 1], [label for _, label, _, _ in SCOPES])
        ax.set_xlim(-0.35, 1.35)
        lower = min(-1.0, float(np.min(_relative(frame, gas)))) - 1.5
        upper = max(1.0, float(np.max(_relative(frame, gas)))) + 2.5
        ax.set_ylim(lower, upper)
        ax.set_title(gas, fontsize=10)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.9", linewidth=0.6)
    axes[0].set_ylabel("相对误差 / %", fontsize=8)
    fig.suptitle(
        "10 秒 TCN 相对误差分布：点=单个实验，黑菱形/竖线=均值及 95% t 置信区间",
        fontsize=10,
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIGS.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_10s_relative_error_ci95.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
