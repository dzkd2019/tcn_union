"""对比 GAN / PCA / VAE / PCA-VAE 的生成质量。

生成器只在 50 个开发实验上拟合；测试集不参与。
输出真实/合成曲线对比图和四个质量指标柱状图。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from .device import configure_threads, describe_device, resolve_device
from .direct_vae import DirectVAEConfig, fit_direct_vae
from .gan import GANConfig, fit_gan
from .generative_metrics import AXES, evaluate_generation
from .pca_generator import PCAConfig, fit_pca
from .train import _read_manifest
from .vae import VAEConfig, fit_vae

plt.rcParams["font.family"] = "Noto Serif CJK SC"
plt.rcParams["font.serif"] = ["Noto Serif CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300

METHODS = ("GAN", "PCA", "VAE", "PCA+VAE")
METHOD_COLORS = {
    "GAN": "#D55E00",
    "PCA": "#0072B2",
    "VAE": "#009E73",
    "PCA+VAE": "#CC79A7",
}
REAL_COLOR = "0.72"
TIME = np.arange(51) * 0.2


def _load_dev_prefixes(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    responses, labels = [], []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        mask = time <= 10.0 + 1e-9
        responses.append(response[mask].T)
        labels.append(np.asarray(row["_labels"], dtype=np.float64))
    return np.stack(responses), np.stack(labels)


def _sample_all(
    dev_rows: list[dict], samples: int, epochs: int, seed: int, device: torch.device
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    output: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    gan = fit_gan(
        dev_rows,
        GANConfig(seconds=10.0, epochs=epochs, residual=False, anchor_endpoints=False),
        seed,
        device,
    )
    output["GAN"] = gan.sample(samples, seed + 1)
    pca = fit_pca(dev_rows, PCAConfig(seconds=10.0, components=6), seed)
    output["PCA"] = pca.sample(samples, seed + 1)
    vae = fit_vae(dev_rows, VAEConfig(seconds=10.0, epochs=epochs, components=6), seed, device)
    output["PCA+VAE"] = vae.sample(samples, seed + 1)
    direct = fit_direct_vae(
        dev_rows, DirectVAEConfig(seconds=10.0, epochs=epochs), seed, device
    )
    output["VAE"] = direct.sample(samples, seed + 1)
    return output


def _plot_curves(
    real: np.ndarray,
    generated: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: Path,
) -> None:
    figure, panels = plt.subplots(4, 3, figsize=(7.2, 8.2), constrained_layout=True)
    for row, method in enumerate(METHODS):
        fake, _ = generated[method]
        for col, gas in enumerate(AXES):
            panel = panels[row, col]
            for curve in real[:20, col]:
                panel.plot(TIME, curve, color=REAL_COLOR, linewidth=0.7, alpha=0.55)
            for curve in fake[:40, col]:
                panel.plot(
                    TIME,
                    curve,
                    color=METHOD_COLORS[method],
                    linewidth=0.65,
                    alpha=0.35,
                )
            panel.set_xlim(0.0, 10.0)
            panel.grid(alpha=0.16, linewidth=0.5)
            panel.tick_params(labelsize=7)
            if row == 0:
                panel.set_title(gas, fontsize=9.2, fontweight="bold")
            if col == 0:
                panel.set_ylabel(method, fontsize=8.0)
            if row == 3:
                panel.set_xlabel("时间 / 秒", fontsize=7.6)
    figure.suptitle("真实响应（灰）与生成响应（彩色）对比", fontsize=10.5, fontweight="bold")
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_generated_curves_comparison.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def _plot_metrics(metrics: dict[str, dict], output_dir: Path) -> None:
    specs = (
        ("wasserstein_normalized", "Wasserstein 距离（归一化，越低越好）", False),
        ("acf_distance", "ACF 距离（越低越好）", False),
        ("pairwise_distance_ratio", "成对距离比（越接近 1 越好）", True),
        ("nearest_neighbor_distance", "最近邻距离（归一化）", False),
    )
    figure, panels = plt.subplots(2, 2, figsize=(7.2, 5.6), constrained_layout=True)
    for panel, (key, title, _) in zip(panels.flat, specs, strict=True):
        values = [metrics[method]["macro"][key] for method in METHODS]
        bars = panel.bar(
            METHODS,
            values,
            color=[METHOD_COLORS[method] for method in METHODS],
            width=0.62,
            edgecolor="white",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values, strict=True):
            panel.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3g}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        panel.set_title(title, fontsize=8.6)
        panel.tick_params(labelsize=7)
        panel.grid(axis="y", alpha=0.18, linewidth=0.5)
        panel.set_ylabel("指标值", fontsize=7.5)
    figure.suptitle("四类生成数据的真实性与多样性指标", fontsize=10.5, fontweight="bold")
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_generative_metrics.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 GAN/PCA/VAE/PCA-VAE 生成质量")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    args = parser.parse_args()
    device = resolve_device(args.device)
    configure_threads(device)
    print(f"[device] {describe_device(device)}")
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    real, _ = _load_dev_prefixes(dev_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated = _sample_all(dev_rows, args.samples, args.epochs, args.seed, device)
    metrics = {
        method: evaluate_generation(real, generated[method][0]) for method in METHODS
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "method",
                "wasserstein_normalized",
                "acf_distance",
                "pairwise_distance_ratio",
                "nearest_neighbor_distance",
                "nearest_neighbor_ratio",
                "all_gases_pairwise_ratio",
            ]
        )
        for method in METHODS:
            macro = metrics[method]["macro"]
            writer.writerow(
                [
                    method,
                    f"{macro['wasserstein_normalized']:.6g}",
                    f"{macro['acf_distance']:.6g}",
                    f"{macro['pairwise_distance_ratio']:.6g}",
                    f"{macro['nearest_neighbor_distance']:.6g}",
                    f"{macro['nearest_neighbor_ratio']:.6g}",
                    f"{macro['all_gases_pairwise_distance_ratio']:.6g}",
                ]
            )
    _plot_curves(real, generated, args.output_dir)
    _plot_metrics(metrics, args.output_dir)
    for method in METHODS:
        macro = metrics[method]["macro"]
        print(
            f"{method:8s} WD={macro['wasserstein_normalized']:.4g} "
            f"ACF={macro['acf_distance']:.4g} "
            f"PairRatio={macro['pairwise_distance_ratio']:.3f} "
            f"NN={macro['nearest_neighbor_distance']:.4g}"
        )


if __name__ == "__main__":
    main()
