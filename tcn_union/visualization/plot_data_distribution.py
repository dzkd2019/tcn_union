"""绘制 62 个实验的按气体浓度分布（开发集 + 锁定测试集）。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GASES = ("H2", "NH3", "NO2")
DEV_COLOR = "#4C72B0"
TEST_COLOR = "#DD8452"


def _load_predictions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"file", "source", *(f"{gas}_true" for gas in GASES)}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} 缺少列: {sorted(missing)}")
    if len(frame) != 62:
        raise ValueError(f"{path} 应为 62 个实验，实际 {len(frame)}")
    if len(frame[["H2_true", "NH3_true", "NO2_true"]].drop_duplicates()) != 62:
        raise ValueError("62 个实验的浓度组合应互不重复")
    return frame


def plot_distribution(predictions_path: Path, output_dir: Path) -> None:
    frame = _load_predictions(predictions_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "Noto Serif CJK SC",
            "font.serif": ["Noto Serif CJK SC"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    figure, panels = plt.subplots(1, 3, figsize=(10.4, 3.5), constrained_layout=True)
    rows: list[dict[str, float | int | str]] = []
    for panel, gas in zip(panels, GASES, strict=True):
        levels = sorted(frame[f"{gas}_true"].unique())
        dev_counts = [
            int(((frame["source"] == "OOF") & (frame[f"{gas}_true"] == level)).sum())
            for level in levels
        ]
        test_counts = [
            int(((frame["source"] == "测试") & (frame[f"{gas}_true"] == level)).sum())
            for level in levels
        ]
        x = np.arange(len(levels))
        panel.bar(
            x,
            dev_counts,
            width=0.68,
            color=DEV_COLOR,
            label="开发集（50）",
            edgecolor="white",
            linewidth=0.6,
        )
        panel.bar(
            x,
            test_counts,
            width=0.68,
            bottom=dev_counts,
            color=TEST_COLOR,
            label="锁定测试（12）",
            edgecolor="white",
            linewidth=0.6,
        )
        for index, level in enumerate(levels):
            total = dev_counts[index] + test_counts[index]
            panel.text(
                index,
                total + 0.5,
                f"{total}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
            rows.append(
                {
                    "gas": gas,
                    "concentration": level,
                    "all_n": total,
                    "development_n": dev_counts[index],
                    "test_n": test_counts[index],
                }
            )
        panel.set_xticks(x, [f"{level:g}" for level in levels])
        panel.set_xlabel(f"{gas} 浓度 / ppm", fontsize=8.5)
        panel.set_ylabel("实验数 / 个", fontsize=8.5)
        panel.set_title(f"{gas}（{len(levels)} 个浓度水平）", fontsize=9.5, loc="left")
        panel.grid(axis="y", alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.8)
        panel.set_ylim(0, max(np.array(dev_counts) + np.array(test_counts)) * 1.18)

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
    figure.suptitle("62 个实验的按气体浓度分布（50 开发 + 12 锁定测试）", fontsize=10.5)
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_concentration_distribution.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)

    stats_path = output_dir / "concentration_distribution.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {stats_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制 62 个实验的浓度分布")
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/3sigma_noridge/all62_predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data_distribution"),
    )
    args = parser.parse_args()
    plot_distribution(args.predictions, args.output_dir)


if __name__ == "__main__":
    main()
