"""绘制当前最终方案的逐气体绝对误差散点。

包含两张图：
1. 按实验序号排列的绝对误差散点，区分 OOF 与锁定测试；
2. 按真实浓度排列的绝对误差散点，按各气体浓度中位数区分低/高浓度。

绝对误差定义为 |pred - true|，单位为 ppm。三个气体量级差异很大，纵轴各自独立。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tcn_union.visualization.plot_concentration_error import (
    GASES,
    HIGH_COLOR,
    LOW_COLOR,
    load_predictions,
)

SOURCE_COLORS = {"OOF": "#4477AA", "测试": "#CC6677"}
SOURCE_MARKERS = {"OOF": "o", "测试": "s"}


def _absolute_error(frame: pd.DataFrame, gas: str) -> np.ndarray:
    truth = frame[f"{gas}_true"].to_numpy(dtype=float)
    prediction = frame[f"{gas}_pred"].to_numpy(dtype=float)
    return np.abs(prediction - truth)


def _format_value(value: float, gas: str) -> str:
    if gas == "H2":
        return f"{value:.0f}"
    if gas == "NH3":
        return f"{value:.2f}"
    return f"{value:.2f}"


def _sort_by_file_id(frame: pd.DataFrame) -> pd.DataFrame:
    order = frame["file"].str.extract(r"(\d+)")[0].astype(int)
    return frame.assign(_order=order).sort_values("_order").reset_index(drop=True)


def plot_absolute_error_by_sample(
    frame: pd.DataFrame,
    output_dir: Path,
    title: str = "当前最终方案：三种气体绝对误差散点（62 点）",
) -> Path:
    ordered = _sort_by_file_id(frame)
    x = np.arange(1, len(ordered) + 1)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, panels = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    for panel, gas in zip(panels, GASES, strict=True):
        errors = _absolute_error(ordered, gas)
        for source in ("OOF", "测试"):
            mask = ordered["source"].to_numpy() == source
            panel.scatter(
                x[mask],
                errors[mask],
                s=22,
                marker=SOURCE_MARKERS[source],
                color=SOURCE_COLORS[source],
                alpha=0.78,
                edgecolor="white",
                linewidth=0.3,
                label=source,
                zorder=3,
            )
        panel.axhline(
            float(errors.mean()),
            color="#777777",
            linestyle="--",
            linewidth=0.9,
            zorder=2,
        )
        index = int(np.argmax(errors))
        panel.annotate(
            f"max {_format_value(errors[index], gas)}\n{ordered.loc[index, 'file']}",
            xy=(x[index], errors[index]),
            xytext=(8, -14),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
            ha="left",
            va="top",
        )
        panel.set_xlabel("实验序号（按 data 编号排序）", fontsize=8.5)
        panel.set_ylabel("绝对误差 / ppm", fontsize=8.5)
        panel.set_title(f"{gas} · 绝对误差", fontsize=9.5, loc="left")
        panel.grid(alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.set_ylim(0.0, float(errors.max()) * 1.18)
        panel.text(
            0.02,
            0.95,
            f"MAE={_format_value(float(errors.mean()), gas)} ppm",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#444444",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
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
        path = output_dir / f"fig_current_absolute_error_by_sample.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)
    return output_dir / "fig_current_absolute_error_by_sample.png"


def plot_absolute_error_vs_concentration(
    frame: pd.DataFrame,
    output_dir: Path,
    title: str = "当前最终方案：绝对误差随真实浓度的变化（62 点）",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure, panels = plt.subplots(1, 3, figsize=(12.6, 3.9), constrained_layout=True)
    records: list[dict[str, float | int | str]] = []

    for panel, gas in zip(panels, GASES, strict=True):
        truth = frame[f"{gas}_true"].to_numpy(dtype=float)
        errors = _absolute_error(frame, gas)
        threshold = float(np.median(truth))
        low = truth <= threshold
        span = max(float(truth.max() - truth.min()), 1e-9)
        jitter = np.linspace(-0.012, 0.012, len(truth)) * span

        panel.scatter(
            truth[low] + jitter[low],
            errors[low],
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
            errors[~low],
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
        panel.set_ylabel("绝对误差 / ppm", fontsize=8.5)
        panel.set_title(f"{gas} · 绝对误差", fontsize=9.5, loc="left")
        panel.grid(alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.set_ylim(0.0, float(errors.max()) * 1.18)

        low_mae = float(errors[low].mean())
        high_mae = float(errors[~low].mean())
        low_sigma = 3.0 * float(errors[low].std(ddof=1))
        high_sigma = 3.0 * float(errors[~low].std(ddof=1))
        panel.text(
            0.02,
            0.96,
            f"低浓度 3σ={_format_value(low_sigma, gas)} ppm",
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
            f"高浓度 3σ={_format_value(high_sigma, gas)} ppm",
            transform=panel.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color=HIGH_COLOR,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        max_index = int(np.argmax(errors))
        records.append(
            {
                "gas": gas,
                "threshold": threshold,
                "low_n": int(low.sum()),
                "high_n": int((~low).sum()),
                "overall_mae": float(errors.mean()),
                "overall_median": float(np.median(errors)),
                "overall_3sigma": 3.0 * float(errors.std(ddof=1)),
                "low_mae": low_mae,
                "high_mae": high_mae,
                "low_3sigma": low_sigma,
                "high_3sigma": high_sigma,
                "max_error": float(errors[max_index]),
                "max_file": frame.loc[max_index, "file"],
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
        path = output_dir / f"fig_current_absolute_error_vs_concentration.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)

    stats_path = output_dir / "current_absolute_error_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {stats_path}")
    return stats_path


def write_point_errors(frame: pd.DataFrame, path: Path) -> None:
    rows = []
    for _, row in frame.iterrows():
        for gas in GASES:
            truth = float(row[f"{gas}_true"])
            prediction = float(row[f"{gas}_pred"])
            rows.append(
                {
                    "file": row["file"],
                    "source": row["source"],
                    "gas": gas,
                    "true": truth,
                    "pred": prediction,
                    "absolute_error": abs(prediction - truth),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制当前方案三种气体的绝对误差散点")
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

    frame = load_predictions(args.predictions)
    plot_absolute_error_by_sample(frame, args.output_dir)
    stats_path = plot_absolute_error_vs_concentration(frame, args.output_dir)
    write_point_errors(frame, stats_path.parent / "current_absolute_error_points.csv")


if __name__ == "__main__":
    main()
