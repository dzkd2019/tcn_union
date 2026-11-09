"""绘制最终 noise + PCA-VAE 方案的逐气体测试指标。

三个面板分别展示 MAPE、NMAE 和归一化 RMSE（NRMSE = RMSE / 开发集浓度极差）。
baseline 与 candidate 都使用三 seed 集成预测，避免单 seed 波动。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

GASES = ("H2", "NH3", "NO2")
METRICS = (
    ("MAPE_percent", "MAPE / %", "(a) MAPE（越低越好）"),
    ("NMAE", "NMAE（MAE / 量程）", "(b) NMAE（越低越好）"),
    ("NRMSE", "NRMSE（RMSE / 量程）", "(c) 归一化 RMSE（越低越好）"),
)
COLORS = {"baseline": "#9AA5AD", "candidate": "#2A7F8F"}


def _load(report_path: Path) -> tuple[dict[str, dict], dict[str, float]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("test_n") != 12:
        raise ValueError("expected 12 locked test runs")
    rows = [row for row in report["metrics"] if str(row["seed"]) == "ensemble"]
    by_model = {row["model"]: row for row in rows}
    if set(by_model) != {"baseline", "candidate"}:
        raise ValueError("evaluation must contain baseline and candidate ensemble rows")
    return by_model, report["development_ranges"]


def _rows(by_model: dict[str, dict], ranges: dict[str, float]) -> list[dict]:
    output = []
    for gas in GASES:
        values = {}
        for model in ("baseline", "candidate"):
            metrics = by_model[model][gas]
            values[model] = {
                "MAPE_percent": float(metrics["MAPE_percent"]),
                "NMAE": float(metrics["NMAE"]),
                "RMSE": float(metrics["RMSE"]),
                "NRMSE": float(metrics["RMSE"]) / float(ranges[gas]),
            }
        output.append(
            {
                "gas": gas,
                "range": float(ranges[gas]),
                "baseline_MAPE_percent": values["baseline"]["MAPE_percent"],
                "candidate_MAPE_percent": values["candidate"]["MAPE_percent"],
                "baseline_NMAE": values["baseline"]["NMAE"],
                "candidate_NMAE": values["candidate"]["NMAE"],
                "baseline_RMSE": values["baseline"]["RMSE"],
                "candidate_RMSE": values["candidate"]["RMSE"],
                "baseline_NRMSE": values["baseline"]["NRMSE"],
                "candidate_NRMSE": values["candidate"]["NRMSE"],
            }
        )
    return output


def _plot(rows: list[dict], output_dir: Path) -> None:
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
    figure, panels = plt.subplots(1, 3, figsize=(9.6, 3.5), constrained_layout=True)
    x = np.arange(len(GASES))
    width = 0.34
    handles = []
    for panel, (key, ylabel, title) in zip(panels, METRICS, strict=True):
        base = [row[f"baseline_{key}"] for row in rows]
        cand = [row[f"candidate_{key}"] for row in rows]
        base_bars = panel.bar(
            x - width / 2,
            base,
            width,
            label="clean 基线",
            color=COLORS["baseline"],
            edgecolor="white",
            linewidth=0.6,
        )
        cand_bars = panel.bar(
            x + width / 2,
            cand,
            width,
            label="noise + PCA-VAE",
            color=COLORS["candidate"],
            edgecolor="white",
            linewidth=0.6,
        )
        for bars in (base_bars, cand_bars):
            for bar in bars:
                panel.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{bar.get_height():.4f}" if key in {"NMAE", "NRMSE"} else f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.2,
                    color="#333333",
                )
        panel.set_xticks(x, GASES)
        panel.set_ylabel(ylabel, fontsize=8.5)
        panel.set_title(title, fontsize=9.2, loc="left")
        panel.grid(axis="y", alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=8)
        if not handles:
            handles = [base_bars, cand_bars]
    figure.suptitle(
        "最终方案逐气体测试指标（12 个锁定测试实验，三 seed 集成）",
        fontsize=10.5,
        y=1.04,
    )
    figure.legend(
        handles,
        ["clean 基线", "noise + PCA-VAE"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=7.5,
    )
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_final_gas_metrics.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制最终方案逐气体 MAPE/NMAE/NRMSE")
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    by_model, ranges = _load(args.evaluation)
    rows = _rows(by_model, ranges)
    with (args.output_dir / "final_metric_by_gas.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _plot(rows, args.output_dir)
    for row in rows:
        print(
            f"{row['gas']}: MAPE {row['baseline_MAPE_percent']:.3f}→"
            f"{row['candidate_MAPE_percent']:.3f}%, "
            f"NMAE {row['baseline_NMAE']:.5f}→{row['candidate_NMAE']:.5f}, "
            f"NRMSE {row['baseline_NRMSE']:.5f}→{row['candidate_NRMSE']:.5f}"
        )


if __name__ == "__main__":
    main()
