"""绘制当前最终方案的逐气体测试指标与宏平均。

数据来自 ensemble_evaluate 的 evaluation.json；只取 seed="ensemble" 一行。
指标统一换算成百分比，便于在同一张图上比较。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GASES = ("H2", "NH3", "NO2")
LABELS = (*GASES, "宏平均")
METRICS = (
    ("MAPE", "(a) MAPE（平均绝对百分比误差）", "#4477AA"),
    ("NMAE", "(b) NMAE（MAE / 量程）", "#EE6677"),
    ("NRMSE", "(c) NRMSE（RMSE / 量程）", "#228833"),
)
BAR_COLORS = ("#4477AA", "#EE6677", "#228833", "#777777")


def _load(report_path: Path) -> tuple[dict, dict[str, float]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("test_n") != 12:
        raise ValueError("expected 12 locked test runs")
    rows = [
        row for row in report["metrics"] if str(row["seed"]) == "ensemble"
    ]
    if len(rows) != 1:
        raise ValueError("expected exactly one ensemble row")
    return rows[0], report["development_ranges"]


def _percent_rows(row: dict, ranges: dict[str, float]) -> list[dict[str, float | str]]:
    output = []
    for gas in GASES:
        metrics = row[gas]
        output.append(
            {
                "group": gas,
                "MAPE_percent": float(metrics["MAPE_percent"]),
                "NMAE_percent": 100.0 * float(metrics["NMAE"]),
                "NRMSE_percent": 100.0
                * float(metrics["RMSE"])
                / float(ranges[gas]),
            }
        )
    output.append(
        {
            "group": "Macro",
            "MAPE_percent": float(np.mean([row[g]["MAPE_percent"] for g in GASES])),
            "NMAE_percent": 100.0 * float(np.mean([row[g]["NMAE"] for g in GASES])),
            "NRMSE_percent": 100.0
            * float(np.mean([row[g]["RMSE"] / ranges[g] for g in GASES])),
        }
    )
    return output


def _plot(rows: list[dict[str, float | str]], output_dir: Path) -> None:
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
    figure, panels = plt.subplots(1, 3, figsize=(10.2, 3.5), constrained_layout=True)
    x = np.arange(len(LABELS))
    for panel, (key, title, _) in zip(panels, METRICS, strict=True):
        values = [float(row[f"{key}_percent"]) for row in rows]
        bars = panel.bar(
            x,
            values,
            width=0.62,
            color=BAR_COLORS,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in bars:
            panel.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{bar.get_height():.2f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#333333",
            )
        panel.axvline(2.5, color="#999999", linestyle="--", linewidth=0.9)
        panel.set_xticks(x, LABELS, fontsize=8.2)
        panel.set_ylabel(f"{key} / %", fontsize=8.5)
        panel.set_title(title, fontsize=9.0, loc="left")
        panel.grid(axis="y", alpha=0.22, linewidth=0.5)
        panel.tick_params(axis="y", labelsize=7.8)
        panel.set_ylim(0.0, max(values) * 1.24)

    figure.suptitle(
        "当前最终方案逐气体测试指标（12 个锁定测试实验，A+B 六 checkpoint 集成）",
        fontsize=10.2,
    )
    figure.text(
        0.5,
        -0.07,
        "宏平均为三种气体指标的算术平均；NMAE/NRMSE 的分母为开发集浓度量程（H2 24000、NH3 40、NO2 8）。",
        ha="center",
        fontsize=7.2,
        color="#555555",
    )
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_current_per_gas_metrics.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制当前方案逐气体 MAPE/NMAE/NRMSE")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("artifacts/3sigma_noridge/evaluation/evaluation.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/3sigma_noridge/figs"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row, ranges = _load(args.evaluation)
    rows = _percent_rows(row, ranges)
    with (args.output_dir / "current_per_gas_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output_dir)
    for row in rows:
        print(
            f"{row['group']}: MAPE {row['MAPE_percent']:.2f}%, "
            f"NMAE {row['NMAE_percent']:.2f}%, NRMSE {row['NRMSE_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
