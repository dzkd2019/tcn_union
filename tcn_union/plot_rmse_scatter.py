"""绘制逐气体评估图。

两种样式：
- ``prediction``：真实浓度 vs 预测浓度，标注 RMSE/MAE；
- ``error-sigma``：参考 example/pic_1.png 的相对误差散点，并画
  mean、±1σ、±2σ、±3σ 水平线，便于看误差分布。
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
COLORS = {"H2": "#0072B2", "NH3": "#E69F00", "NO2": "#D55E00"}
MARKERS = {"H2": "o", "NH3": "^", "NO2": "s"}


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


def _select(
    frame: pd.DataFrame, model: str, window: int, seed: str
) -> pd.DataFrame:
    required = {"model", "window_seconds", "seed"}
    if not required.issubset(frame.columns):
        raise ValueError(f"predictions CSV 缺少列: {sorted(required - set(frame.columns))}")
    selected = frame[
        (frame["model"] == model)
        & (frame["window_seconds"].astype(int) == window)
        & (frame["seed"].astype(str) == str(seed))
    ]
    if selected.empty:
        raise ValueError(f"没有匹配的行: model={model}, window={window}, seed={seed}")
    return selected


def plot_error_sigma(
    predictions_path: Path,
    output_dir: Path,
    model: str,
    window: int,
    seed: str,
    prefix: str = "",
) -> None:
    """参考 example/pic_1.png 的相对误差散点，并叠加 ±1/2/3σ 线。"""
    selected = _select(pd.read_csv(predictions_path), model, window, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, panels = plt.subplots(2, 2, figsize=(9.2, 7.4), constrained_layout=True)
    panel_specs = [("H2", panels[0, 0]), ("NH3", panels[0, 1]), ("NO2", panels[1, 0])]
    for gas, panel in panel_specs:
        truth = selected[f"{gas}_true"].to_numpy(dtype=float)
        prediction = selected[f"{gas}_pred"].to_numpy(dtype=float)
        error = 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)
        mean = float(np.mean(error))
        sigma = float(np.std(error, ddof=1)) if len(error) > 1 else 0.0
        _draw_error_panel(
            panel,
            error,
            mean,
            sigma,
            gas,
            marker=MARKERS[gas],
            color=COLORS[gas],
        )

    # (d) 三气体合并：所有相对误差放在一个面板，便于看整体分布。
    panel = panels[1, 1]
    pooled = []
    pooled_colors = []
    for gas in AXES:
        truth = selected[f"{gas}_true"].to_numpy(dtype=float)
        prediction = selected[f"{gas}_pred"].to_numpy(dtype=float)
        values = 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)
        pooled.extend(values.tolist())
        pooled_colors.extend([COLORS[gas]] * len(values))
    pooled_array = np.asarray(pooled)
    mean = float(np.mean(pooled_array))
    sigma = float(np.std(pooled_array, ddof=1)) if len(pooled_array) > 1 else 0.0
    _draw_error_panel(
        panel,
        pooled_array,
        mean,
        sigma,
        "All gases",
        marker="o",
        color=None,
        point_colors=pooled_colors,
    )

    for panel, label in zip(panels.flat, ("(a)", "(b)", "(c)", "(d)"), strict=True):
        panel.text(
            -0.13,
            1.03,
            label,
            transform=panel.transAxes,
            fontsize=14,
            fontweight="bold",
            va="bottom",
        )
    figure.suptitle(
        f"Relative error distribution ({window} s, {model}, seed={seed})",
        fontsize=13,
    )
    stem = f"{prefix}_error_sigma_by_gas" if prefix else "error_sigma_by_gas"
    for suffix in ("png", "pdf"):
        path = output_dir / f"{stem}.{suffix}"
        figure.savefig(path, dpi=180, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)


def _draw_error_panel(
    panel,
    error: np.ndarray,
    mean: float,
    sigma: float,
    gas: str,
    *,
    marker: str,
    color: str | None,
    point_colors: list[str] | None = None,
) -> None:
    """在单个面板画相对误差点和 mean/±1σ/±2σ/±3σ 线。"""
    points = np.arange(len(error))
    panel.scatter(
        points,
        error,
        s=26,
        marker=marker,
        c=color if color is not None else point_colors,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.4,
        zorder=3,
    )
    panel.axhline(0.0, color="0.55", linewidth=0.8, zorder=1)
    panel.axhline(mean, color="0.1", linewidth=1.2, linestyle="-", zorder=2, label="mean")
    bands = [(1.0, "--", "0.35"), (2.0, ":", "0.45"), (3.0, "-.", "0.55")]
    for k, style, shade in bands:
        for sign in (-1.0, 1.0):
            panel.axhline(
                mean + sign * k * sigma,
                color=shade,
                linewidth=1.0,
                linestyle=style,
                zorder=2,
                label=f"±{int(k)}σ" if sign > 0 else None,
            )
    if sigma > 0.0:
        panel.axhspan(mean - sigma, mean + sigma, color="0.85", alpha=0.25, zorder=0)

    limit = max(3.0 * sigma, float(np.max(np.abs(error))), 1e-9) * 1.15
    panel.set_ylim(-limit, limit)
    panel.set_xlim(-0.5, max(len(error) - 0.5, 0.5))
    panel.set_xlabel("Points", fontsize=11)
    panel.set_ylabel("Relative error (%)", fontsize=11)
    panel.set_title(
        f"{gas}  n={len(error)}, mean={mean:.2f}%, σ={sigma:.2f}%",
        fontsize=10.5,
        loc="right",
    )
    panel.grid(axis="y", alpha=0.2)
    for spine in panel.spines.values():
        spine.set_linewidth(1.0)
    panel.tick_params(labelsize=9)
    panel.legend(loc="lower left", fontsize=7.2, frameon=False, ncol=1, labelspacing=0.25)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制逐气体 RMSE 散点图")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--seed", default="ensemble")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--style", choices=["prediction", "error-sigma"], default="prediction")
    args = parser.parse_args()
    if args.style == "error-sigma":
        plot_error_sigma(
            args.predictions,
            args.output_dir,
            args.model,
            args.window,
            str(args.seed),
            args.prefix,
        )
    else:
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
