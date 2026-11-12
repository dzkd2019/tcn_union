"""绘制当前方案在不同真实浓度下的相对误差散点。

沿用之前 3σ 误差诊断的版式：
- 横轴为真实浓度，纵轴为相对误差；
- 每个气体按 62 点真实浓度的中位数切成低/高浓度；
- 低浓度蓝色圆点、高浓度橙色方块，虚线为中位数切分点；
- 面板内标出低/高浓度各自的 3σ。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "Noto Serif CJK SC",
        "font.serif": ["Noto Serif CJK SC"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)

GASES = ("H2", "NH3", "NO2")
LOW_COLOR = "#0072B2"
HIGH_COLOR = "#D55E00"


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        f"{gas}_{kind}" for gas in GASES for kind in ("true", "pred")
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    if len(frame) != 62:
        raise ValueError(f"{path} 应为 62 点，实际 {len(frame)} 点")
    if frame["file"].duplicated().any():
        raise ValueError(f"{path} 存在重复实验")
    return frame


def _relative_error(truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)


def plot_current_relative_error(
    predictions_path: Path,
    output_dir: Path,
    title: str = "当前最终方案：相对误差随真实浓度的变化（62 点）",
) -> Path:
    frame = _load_predictions(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, panels = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    records: list[dict[str, float | int | str]] = []
    for panel, gas in zip(panels, GASES, strict=True):
        truth = frame[f"{gas}_true"].to_numpy(dtype=float)
        prediction = frame[f"{gas}_pred"].to_numpy(dtype=float)
        relative = _relative_error(truth, prediction)
        threshold = float(np.median(truth))
        low = truth <= threshold
        span = max(float(truth.max() - truth.min()), 1e-9)
        jitter = np.linspace(-0.012, 0.012, len(truth)) * span

        panel.scatter(
            truth[low] + jitter[low],
            relative[low],
            s=19,
            color=LOW_COLOR,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.25,
            label="低浓度",
            zorder=3,
        )
        panel.scatter(
            truth[~low] + jitter[~low],
            relative[~low],
            s=22,
            marker="s",
            color=HIGH_COLOR,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.25,
            label="高浓度",
            zorder=3,
        )
        panel.axvline(threshold, color="#777777", linestyle=":", linewidth=1.0)
        panel.axhline(0.0, color="#999999", linewidth=0.6)
        panel.set_xlabel(f"真实浓度（{gas}）", fontsize=8.5)
        panel.set_ylabel("相对误差 / %", fontsize=8.5)
        panel.set_title(f"{gas} · 相对误差", fontsize=9.5, loc="left")
        panel.grid(alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.margins(y=0.18)

        low_sigma = 3.0 * relative[low].std(ddof=1)
        high_sigma = 3.0 * relative[~low].std(ddof=1)
        panel.text(
            0.02,
            0.96,
            f"低浓度 3σ={low_sigma:.2f}%",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color=LOW_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        panel.text(
            0.98,
            0.96,
            f"高浓度 3σ={high_sigma:.2f}%",
            transform=panel.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color=HIGH_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        records.append(
            {
                "gas": gas,
                "threshold": threshold,
                "low_n": int(low.sum()),
                "high_n": int((~low).sum()),
                "low_3sigma_percent": low_sigma,
                "high_3sigma_percent": high_sigma,
                "overall_3sigma_percent": 3.0 * relative.std(ddof=1),
                "mean_relative_percent": float(relative.mean()),
            }
        )

    handles, labels = panels[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(title, fontsize=10.5)

    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_current_relative_error_vs_concentration.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)

    stats_path = output_dir / "current_relative_error_low_high.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {stats_path}")
    return stats_path


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制当前方案不同浓度下的相对误差散点")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/3sigma_noridge/all62_predictions.csv"),
        help="62 点预测 CSV，需包含六列 true/pred",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/3sigma_noridge/figs"),
    )
    args = parser.parse_args()
    plot_current_relative_error(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
