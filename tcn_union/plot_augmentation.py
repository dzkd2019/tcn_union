"""绘制增强后 10 秒模型在锁定测试集上的报告图。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

AXES = ("H2", "NH3", "NO2")
LABELS = {
    "baseline_5s": "5 秒原始",
    "baseline_10s": "10 秒原始",
    "augmented_10s": "10 秒 cutout + 5 秒辅助",
}
COLORS = {"baseline_5s": "#9ca3af", "baseline_10s": "#f59e0b", "augmented_10s": "#2563eb"}


def _ensemble_rows(frame: pd.DataFrame, model: str, seconds: int) -> pd.DataFrame:
    rows = frame[
        (frame["model"] == model)
        & (frame["window_seconds"] == seconds)
        & (frame["seed"] == "ensemble")
    ]
    if len(rows) != 12:
        raise ValueError(f"expected 12 ensemble rows for {model} {seconds}s, got {len(rows)}")
    return rows.sort_values("file")


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制增强后 10 秒模型的锁定测试结果")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.results_dir / "evaluation.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(args.results_dir / "test_predictions.csv")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    keys = [("baseline", 5, "baseline_5s"), ("baseline", 10, "baseline_10s"), ("augmented", 10, "augmented_10s")]
    summary = {}
    for model, seconds, key in keys:
        metric = next(
            item
            for item in report["metrics"]
            if item["model"] == model and item["window_seconds"] == seconds and item["seed"] == "ensemble"
        )
        summary[key] = metric

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    values = [summary[key]["macro_NMAE"] for key in LABELS]
    bars = ax.bar(list(LABELS.values()), values, color=[COLORS[key] for key in LABELS], width=0.6)
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.0004, f"{value:.5f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("macro-NMAE（越低越好）")
    ax.set_title("锁定 12 个测试实验：三 seed 集成")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(values) * 1.22)
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_macro_nmae_comparison.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    width = 0.26
    x = np.arange(len(AXES))
    for index, key in enumerate(LABELS):
        values = [summary[key][axis]["NMAE"] for axis in AXES]
        ax.bar(x + (index - 1) * width, values, width, label=LABELS[key], color=COLORS[key])
    ax.set_xticks(x, AXES)
    ax.set_ylabel("NMAE（越低越好）")
    ax.set_title("锁定测试集逐气体 NMAE")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_nmae_by_gas.png", dpi=160)
    plt.close(fig)

    rows = _ensemble_rows(frame, "augmented", 10)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    for ax, axis in zip(axes, AXES, strict=True):
        truth = rows[f"{axis}_true"].to_numpy()
        pred = rows[f"{axis}_pred"].to_numpy()
        ax.scatter(truth, pred, s=42, color=COLORS["augmented_10s"], edgecolor="white", zorder=3)
        low = min(truth.min(), pred.min())
        high = max(truth.max(), pred.max())
        ax.plot([low, high], [low, high], "--", color="#6b7280", linewidth=1)
        ax.set_xlabel(f"真实 {axis}")
        ax.set_ylabel(f"预测 {axis}")
        ax.set_title(f"{axis}: NMAE={summary['augmented_10s'][axis]['NMAE']:.4f}")
        ax.grid(alpha=0.25)
    fig.suptitle("10 秒 cutout + 5 秒辅助监督，三 seed 集成")
    fig.tight_layout()
    fig.savefig(args.output_dir / "test_predictions_augmented_10s.png", dpi=160)
    plt.close(fig)
    print(f"已写出图像到 {args.output_dir}")


if __name__ == "__main__":
    main()
