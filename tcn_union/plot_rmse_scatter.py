"""绘制逐气体 RMSE 散点图：真实浓度 vs 预测浓度。

输入是评估脚本写出的 ``test_predictions.csv``。每个气体单独一个面板，
虚线是 y=x；标题和面板内标注该气体的 RMSE、MAE 与样本数。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei",
    "Noto Sans CJK SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

AXES = ("H2", "NH3", "NO2")
COLORS = {"H2": "#0072B2", "NH3": "#009E73", "NO2": "#D55E00"}


def plot_rmse_scatter(
    predictions_path: Path,
    output_dir: Path,
    model: str,
    window: int,
    seed: str,
    prefix: str = "",
) -> None:
    frame = pd.read_csv(predictions_path)
    required = {"model", "window_seconds", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"predictions CSV 缺少列: {sorted(required - set(frame.columns))}")
    selected = frame[
        (frame["model"] == model)
        & (frame["window_seconds"].astype(int) == window)
        & (frame["seed"].astype(str) == str(seed))
    ]
    if selected.empty:
        raise ValueError(
            f"没有匹配的行: model={model}, window={window}, seed={seed}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    figure, panels = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for panel, gas in zip(panels, AXES, strict=True):
        truth = selected[f"{gas}_true"].to_numpy(dtype=float)
        prediction = selected[f"{gas}_pred"].to_numpy(dtype=float)
        error = prediction - truth
        rmse = float(np.sqrt(np.mean(error**2)))
        mae = float(np.mean(np.abs(error)))
        low = min(float(truth.min()), float(prediction.min()))
        high = max(float(truth.max()), float(prediction.max()))
        span = max(high - low, 1e-9)
        low -= 0.05 * span
        high += 0.05 * span

        panel.scatter(
            truth,
            prediction,
            s=42,
            alpha=0.8,
            color=COLORS[gas],
            edgecolors="white",
            linewidths=0.5,
        )
        panel.plot([low, high], [low, high], "k--", linewidth=1.0, label="y = x")
        panel.set_xlim(low, high)
        panel.set_ylim(low, high)
        panel.set_aspect("equal", adjustable="box")
        panel.set_xlabel(f"{gas} 真实浓度")
        panel.set_ylabel(f"{gas} 预测浓度")
        panel.set_title(f"{gas}：RMSE={rmse:.4g}，MAE={mae:.4g}（n={len(truth)}）")
        panel.grid(alpha=0.25)
        panel.legend(loc="upper left", fontsize=8)

    figure.suptitle(
        f"逐气体 RMSE 散点图（{window} 秒，{model}，seed={seed}）",
        fontsize=13,
    )
    stem = f"{prefix}_rmse_scatter_by_gas" if prefix else "rmse_scatter_by_gas"
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制逐气体 RMSE 散点图")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--seed", default="ensemble")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    plot_rmse_scatter(
        args.predictions,
        args.output_dir,
        args.model,
        args.window,
        str(args.seed),
        args.prefix,
    )


if __name__ == "__main__":
    main()
