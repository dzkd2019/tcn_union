"""Protocol ablation figures for the 10 s TCN augmentation study.

Reads the tables produced by tcn_union.experiments.compare_protocols and
writes three 1x3 figures (one panel per gas): macro 3-sigma, macro MAPE, and
the paired difference of hier/hier+vae against VAE.  Figures are written to
the comparison figs directory as PNG (300 dpi) and SVG; no PDF.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROTOCOLS = ("syn", "noise_syn", "full", "gas")
PROTOCOL_LABELS = {
    "syn": "真实+合成",
    "noise_syn": "真实+噪声+合成",
    "full": "完整协议",
    "gas": "H2 扩充",
}
ARMS = ("hier", "vae", "hier+vae")
ARM_LABELS = {
    "real": "仅真实",
    "hier": "分层随机效应",
    "vae": "PCA-VAE",
    "hier+vae": "分层+VAE",
}
ARM_COLORS = {
    "real": "#7F7F7F",
    "hier": "#D55E00",
    "vae": "#4C72B0",
    "hier+vae": "#009E73",
}
GASES = ("H2", "NH3", "NO2")
GAS_LABELS = {"H2": "H₂", "NH3": "NH₃", "NO2": "NO₂"}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Microsoft YaHei", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)


def _metric_bars(ensemble: pd.DataFrame, seeds: pd.DataFrame, metric: str, ylabel: str, name: str, directory: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), layout="constrained")
    width = 0.25
    offsets = np.linspace(-width, width, len(ARMS))
    for axis, gas in zip(axes, GASES):
        x = np.arange(len(PROTOCOLS))
        for offset, arm in zip(offsets, ARMS):
            values = []
            for protocol in PROTOCOLS:
                row = ensemble[(ensemble["protocol"] == protocol) & (ensemble["arm"] == arm)]
                values.append(float(row[f"{gas}_{metric}"].iloc[0]) if len(row) else np.nan)
            axis.bar(x + offset, values, width=width, color=ARM_COLORS[arm], edgecolor="black", linewidth=0.35, label=ARM_LABELS[arm])
            for position, value, protocol in zip(x + offset, values, PROTOCOLS):
                if not np.isfinite(value):
                    continue
                axis.text(position, value, f"{value:.1f}", ha="center", va="bottom", fontsize=5.6, rotation=90)
                seed_rows = seeds[(seeds["protocol"] == protocol) & (seeds["arm"] == arm)]
                if len(seed_rows):
                    axis.scatter(
                        np.full(len(seed_rows), position),
                        seed_rows[f"{gas}_{metric}"].to_numpy(),
                        s=7,
                        facecolor="white",
                        edgecolor="black",
                        linewidth=0.35,
                        zorder=4,
                    )
        real_row = ensemble[(ensemble["protocol"] == PROTOCOLS[0]) & (ensemble["arm"] == "real")]
        real_value = float(real_row[f"{gas}_{metric}"].iloc[0]) if len(real_row) else np.nan
        arm_values = []
        for protocol in PROTOCOLS:
            for arm in ARMS:
                row = ensemble[(ensemble["protocol"] == protocol) & (ensemble["arm"] == arm)]
                if len(row):
                    arm_values.append(float(row[f"{gas}_{metric}"].iloc[0]))
        top = np.nanmax([value for value in arm_values if np.isfinite(value)])
        axis.plot([], [], color=ARM_COLORS["real"], linestyle="--", linewidth=0.8, label="仅真实")
        if np.isfinite(real_value) and real_value <= top * 1.18:
            axis.axhline(real_value, color=ARM_COLORS["real"], linestyle="--", linewidth=0.8)
        elif np.isfinite(real_value):
            axis.text(0.99, 0.97, f"仅真实 {real_value:.1f}", transform=axis.transAxes, ha="right", va="top", fontsize=6.5, color=ARM_COLORS["real"])
        axis.set_xticks(x)
        axis.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOLS], fontsize=6.8)
        axis.set_ylabel(ylabel)
        axis.set_title(GAS_LABELS[gas], fontsize=9.5)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.7)
        axis.set_axisbelow(True)
        axis.set_ylim(0.0, top * 1.18)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7.2)
    figure.suptitle("训练协议消融：6 checkpoint 集成，空心点为单个 seed", fontsize=10.0)
    _save(figure, directory, name)


def _paired_delta(paired: pd.DataFrame, directory: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), layout="constrained")
    width = 0.34
    for axis, gas in zip(axes, GASES):
        x = np.arange(len(PROTOCOLS))
        for offset, arm in zip((-width / 2, width / 2), ("hier", "hier+vae")):
            means, low, high = [], [], []
            for protocol in PROTOCOLS:
                selected = paired[
                    (paired["protocol"] == protocol)
                    & (paired["metric"] == f"{gas}_three_sigma")
                    & (paired["candidate"] == arm)
                    & (paired["baseline"] == "vae")
                ]
                if len(selected):
                    row = selected.iloc[0]
                    means.append(float(row["mean_diff"]))
                    low.append(float(row["mean_diff"] - row["ci_low"]))
                    high.append(float(row["ci_high"] - row["mean_diff"]))
                else:
                    means.append(np.nan)
                    low.append(np.nan)
                    high.append(np.nan)
            axis.bar(x + offset, means, width=width, color=ARM_COLORS[arm], edgecolor="black", linewidth=0.35, label=f"{ARM_LABELS[arm]} − PCA-VAE")
            axis.errorbar(x + offset, means, yerr=np.array([low, high]), fmt="none", ecolor="black", elinewidth=0.8, capsize=2.0)
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_xticks(x)
        axis.set_xticklabels([PROTOCOL_LABELS[p] for p in PROTOCOLS], fontsize=6.8)
        axis.set_ylabel("宏 3σ 配对差 / 百分点")
        axis.set_title(GAS_LABELS[gas], fontsize=9.5)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.45, alpha=0.7)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=7.2)
    figure.suptitle("相对 PCA-VAE 的配对差：负值表示该协议更好，误差棒为 95% CI", fontsize=10.0)
    _save(figure, directory, "fig_protocol_delta_3sigma")


def run(root: Path, prefix: str, comparison: Path) -> None:
    _style()
    ensemble = pd.read_csv(comparison / "ensemble_metrics.csv")
    seeds = pd.read_csv(comparison / "seed_metrics.csv")
    paired = pd.read_csv(comparison / "paired_within_protocol.csv")
    directory = comparison / "figs"
    _metric_bars(ensemble, seeds, "three_sigma", "宏 3σ / %", "fig_protocol_3sigma", directory)
    _metric_bars(ensemble, seeds, "mape", "宏 MAPE / %", "fig_protocol_mape", directory)
    _paired_delta(paired, directory)
    (comparison / "figure_config.json").write_text(
        json.dumps(
            {
                "root": str(root),
                "prefix": prefix,
                "figures": [
                    "fig_protocol_3sigma",
                    "fig_protocol_mape",
                    "fig_protocol_delta_3sigma",
                ],
                "protocols": list(PROTOCOLS),
                "arms": list(ARMS),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/augmentation/hierarchical"))
    parser.add_argument("--prefix", default="fullrun_")
    parser.add_argument("--comparison", type=Path, default=None)
    args = parser.parse_args()
    comparison = args.comparison or args.root / f"{args.prefix.rstrip('_')}_comparison"
    run(args.root, args.prefix, comparison)


if __name__ == "__main__":
    main()
