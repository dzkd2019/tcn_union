"""对比 cutout+aux 与当前 noise+PCA-VAE 的 62 点误差分布。

重点回答：低浓度还是高浓度导致相对误差标准差更大。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .plot_all62 import combine_predictions
from .plot_rmse_scatter import plot_error_sigma

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
COLORS = {"cutout_aux": "#D55E00", "current": "#1F6F78"}
LABELS = {"cutout_aux": "cutout + aux", "current": "noise + PCA-VAE"}
MARKERS = {"cutout_aux": "o", "current": "s"}
COLS = ["file"] + [f"{gas}_{kind}" for gas in GASES for kind in ("true", "pred")]


def _load_schemes(root: Path) -> dict[str, pd.DataFrame]:
    # cutout+aux：50 个 OOF + 12 个测试，统一组装成 62 点。
    cutout_all = root / "artifacts/error_analysis/cutout_aux_all62_predictions.csv"
    combine_predictions(
        root / "artifacts/augmentation_final/oof/10s_raw_delta.csv",
        root / "artifacts/evaluation_final/test_predictions.csv",
        "augmented",
        10,
        "ensemble",
        cutout_all,
    )
    schemas = {
        "cutout_aux": pd.read_csv(cutout_all),
        "current": pd.read_csv(
            root / "artifacts/final_vae_hybrid/evaluation/candidate_all62_predictions.csv"
        ),
    }
    for frame in schemas.values():
        if len(frame) != 62 or frame["file"].duplicated().any():
            raise ValueError("each scheme must contain 62 unique runs")
    return schemas


def _stats(schemes: dict[str, pd.DataFrame]) -> list[dict]:
    records = []
    for scheme, frame in schemes.items():
        for gas in GASES:
            truth = frame[f"{gas}_true"].to_numpy(dtype=float)
            prediction = frame[f"{gas}_pred"].to_numpy(dtype=float)
            absolute = np.abs(prediction - truth)
            relative = 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)
            median = float(np.median(truth))
            for half, mask in (
                ("low", truth <= median),
                ("high", truth > median),
            ):
                records.append(
                    {
                        "scheme": scheme,
                        "gas": gas,
                        "half": half,
                        "n": int(mask.sum()),
                        "true_min": float(truth[mask].min()),
                        "true_max": float(truth[mask].max()),
                        "mean_abs_error": float(absolute[mask].mean()),
                        "std_abs_error": float(absolute[mask].std(ddof=1)),
                        "mean_relative_percent": float(relative[mask].mean()),
                        "std_relative_percent": float(relative[mask].std(ddof=1)),
                        "three_sigma_percent": float(3.0 * relative[mask].std(ddof=1)),
                    }
                )
    return records


def _plot_all62(schemes: dict[str, pd.DataFrame], output_dir: Path) -> None:
    for scheme, frame in schemes.items():
        path = output_dir / f"{scheme}_all62_predictions.csv"
        frame.to_csv(path, index=False)
        plot_error_sigma(
            path,
            output_dir,
            "all62",
            10,
            "OOF+测试",
            prefix=scheme,
            title=f"{LABELS[scheme]}：全部 62 组相对误差",
        )


def _plot_absolute(schemes: dict[str, pd.DataFrame], output_dir: Path) -> None:
    figure, panels = plt.subplots(1, 3, figsize=(12.6, 4.0), constrained_layout=True)
    for panel, gas in zip(panels, GASES, strict=True):
        low = min(
            frame[f"{gas}_true"].min() for frame in schemes.values()
        )
        high = max(
            frame[f"{gas}_true"].max() for frame in schemes.values()
        )
        span = max(high - low, 1e-9)
        for index, (scheme, frame) in enumerate(schemes.items()):
            truth = frame[f"{gas}_true"].to_numpy(dtype=float)
            error = np.abs(frame[f"{gas}_pred"].to_numpy(dtype=float) - truth)
            jitter = np.linspace(-0.012, 0.012, len(truth)) * span
            panel.scatter(
                truth + jitter,
                error,
                s=18,
                alpha=0.65,
                color=COLORS[scheme],
                marker=MARKERS[scheme],
                edgecolor="white",
                linewidth=0.25,
                label=LABELS[scheme],
                zorder=3,
            )
        panel.axvline(
            float(np.median(np.concatenate([f[f"{gas}_true"] for f in schemes.values()]))),
            color="#777777",
            linestyle=":",
            linewidth=1.0,
        )
        panel.set_xlabel(f"真实浓度（{gas}）", fontsize=9)
        panel.set_ylabel("绝对误差", fontsize=9)
        panel.set_title(gas, fontsize=10, loc="left")
        panel.grid(alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.legend(fontsize=7, frameon=False)
    figure.suptitle("绝对误差随真实浓度的变化（62 点：50 OOF + 12 测试）", fontsize=10.5)
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_abs_error_vs_concentration.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def _plot_std_bars(records: list[dict], output_dir: Path) -> None:
    figure, panels = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    for panel, gas in zip(panels, GASES, strict=True):
        x = np.arange(2)
        width = 0.34
        for offset, scheme in ((-width / 2, "cutout_aux"), (width / 2, "current")):
            values = [
                next(
                    row["std_relative_percent"]
                    for row in records
                    if row["scheme"] == scheme and row["gas"] == gas and row["half"] == half
                )
                for half in ("low", "high")
            ]
            bars = panel.bar(
                x + offset,
                values,
                width,
                label=LABELS[scheme],
                color=COLORS[scheme],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                panel.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#333333",
                )
        panel.set_xticks(x, ["低浓度", "高浓度"])
        panel.set_ylabel("相对误差标准差 / %", fontsize=9)
        panel.set_title(gas, fontsize=10, loc="left")
        panel.grid(axis="y", alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.legend(fontsize=6.8, frameon=False)
    figure.suptitle("低/高浓度下的相对误差离散程度（越低越稳）", fontsize=10.5)
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_relative_error_std_low_high.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)



def _plot_current_abs_rel(schemes: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """当前方案：绝对误差和相对误差随真实浓度的 2×3 散点图。"""
    frame = schemes["current"]
    figure, panels = plt.subplots(2, 3, figsize=(12.6, 7.0), constrained_layout=True)
    for column, gas in enumerate(GASES):
        truth = frame[f"{gas}_true"].to_numpy(dtype=float)
        prediction = frame[f"{gas}_pred"].to_numpy(dtype=float)
        absolute = np.abs(prediction - truth)
        relative = 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)
        threshold = float(np.median(truth))
        low = truth <= threshold
        span = max(float(truth.max() - truth.min()), 1e-9)
        jitter = np.linspace(-0.012, 0.012, len(truth)) * span

        for row, values, ylabel in (
            (0, absolute, "绝对误差"),
            (1, relative, "相对误差 / %"),
        ):
            panel = panels[row, column]
            panel.scatter(
                truth[low] + jitter[low],
                values[low],
                s=19,
                color="#0072B2",
                alpha=0.75,
                edgecolor="white",
                linewidth=0.25,
                label="低浓度",
                zorder=3,
            )
            panel.scatter(
                truth[~low] + jitter[~low],
                values[~low],
                s=22,
                marker="s",
                color="#D55E00",
                alpha=0.72,
                edgecolor="white",
                linewidth=0.25,
                label="高浓度",
                zorder=3,
            )
            panel.axvline(threshold, color="#777777", linestyle=":", linewidth=1.0)
            panel.axhline(0.0, color="#999999", linewidth=0.6)
            panel.set_xlabel(f"真实浓度（{gas}）", fontsize=8.5)
            panel.set_ylabel(ylabel, fontsize=8.5)
            panel.set_title(f"{gas} · {'绝对误差' if row == 0 else '相对误差'}", fontsize=9.5, loc="left")
            panel.grid(alpha=0.2, linewidth=0.5)
            panel.tick_params(labelsize=7.5)

        low_sigma = 3.0 * relative[low].std(ddof=1)
        high_sigma = 3.0 * relative[~low].std(ddof=1)
        bottom = panels[1, column]
        bottom.text(
            0.02,
            0.96,
            f"低浓度 3σ={low_sigma:.1f}%",
            transform=bottom.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#0072B2",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )
        bottom.text(
            0.98,
            0.96,
            f"高浓度 3σ={high_sigma:.1f}%",
            transform=bottom.transAxes,
            ha="right",
            va="top",
            fontsize=6.8,
            color="#D55E00",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
        )

    handles, labels = panels[0, 0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.10),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    figure.suptitle(
        "当前方案：绝对误差与相对误差随真实浓度的变化（62 点）",
        fontsize=10.5,
    )
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_current_abs_rel_vs_concentration.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)

def main() -> None:
    parser = argparse.ArgumentParser(description="cutout+aux 与当前方案的 62 点误差诊断")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/error_analysis"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    schemes = _load_schemes(args.root)
    records = _stats(schemes)
    with (args.output_dir / "low_high_error_stats.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _plot_all62(schemes, args.output_dir)
    _plot_absolute(schemes, args.output_dir)
    _plot_std_bars(records, args.output_dir)
    _plot_current_abs_rel(schemes, args.output_dir)
    for row in records:
        print(
            f"{row['scheme']:10s} {row['gas']:3s} {row['half']:4s} "
            f"n={row['n']:2d} MAE={row['mean_abs_error']:.4g} "
            f"std_rel={row['std_relative_percent']:.2f}% 3σ={row['three_sigma_percent']:.2f}%"
        )


if __name__ == "__main__":
    main()
