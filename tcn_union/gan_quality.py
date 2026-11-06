"""GAN 生成质量的快速检查。

这不是严格的物理验证，而是进入下游 TCN 前的“排雷”：
检查平滑度、初值/末端分布、最近邻距离和多样性，并画真实/合成曲线。
只使用开发集；测试集不参与。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import TIME_STEP_SECONDS
from .device import configure_threads, describe_device, resolve_device
from .gan import GANConfig, fit_gan
from .train import _read_manifest
from .vae import VAEConfig, fit_vae

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
AXES = ("H2", "NH3", "NO2")


def _real_prefixes(rows: list[dict[str, Any]]) -> np.ndarray:
    length = round(10.0 / TIME_STEP_SECONDS) + 1
    values = []
    for row in rows:
        time = np.asarray(row["_time"], dtype=np.float64)
        response = np.asarray(row["_response"], dtype=np.float64)
        values.append(response[time <= 10.0 + 1e-9][:length].T)
    return np.stack(values)


def _pairwise_mean(values: np.ndarray) -> float:
    flattened = values.reshape(len(values), -1)
    distances = np.linalg.norm(flattened[:, None, :] - flattened[None, :, :], axis=2)
    return float(distances[np.triu_indices(len(values), k=1)].mean())


def _nearest_ratio(real: np.ndarray, fake: np.ndarray) -> float:
    scale = real.std(axis=(0, 2), keepdims=True) + 1e-8
    real_flat = (real / scale).reshape(len(real), -1)
    fake_flat = (fake / scale).reshape(len(fake), -1)
    distances = np.linalg.norm(fake_flat[:, None, :] - real_flat[None, :, :], axis=2)
    fake_nn = float(distances.min(axis=1).mean())
    real_dist = np.linalg.norm(real_flat[:, None, :] - real_flat[None, :, :], axis=2)
    np.fill_diagonal(real_dist, np.inf)
    real_nn = float(real_dist.min(axis=1).mean())
    return fake_nn / max(real_nn, 1e-8)


def check_quality(
    manifest: Path,
    output_dir: Path,
    samples: int,
    gan_epochs: int,
    seed: int,
    device: torch.device,
    generator: str = "gan",
    vae_components: int = 6,
    vae_beta: float = 0.1,
) -> dict[str, Any]:
    rows = _read_manifest(manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    real = _real_prefixes(dev_rows)
    real_labels = np.stack([row["_labels"] for row in dev_rows]).astype(np.float64)
    if generator == "vae":
        vae = fit_vae(
            dev_rows,
            VAEConfig(
                epochs=gan_epochs,
                components=vae_components,
                beta=vae_beta,
            ),
            seed,
            device,
        )
        fake, labels = vae.sample(samples, seed + 1)
    elif generator == "gan":
        gan = fit_gan(dev_rows, GANConfig(epochs=gan_epochs), seed, device)
        fake, labels = gan.sample(samples, seed + 1)
    else:
        raise ValueError("generator 必须是 gan 或 vae")

    real_smooth = float(np.mean(np.abs(np.diff(real, n=2, axis=2))))
    fake_smooth = float(np.mean(np.abs(np.diff(fake, n=2, axis=2))))
    initial_mae = float(np.mean(np.abs(fake[:, :, 0].mean(axis=0) - real[:, :, 0].mean(axis=0))))
    end_mae = float(np.mean(np.abs(fake[:, :, -1].mean(axis=0) - real[:, :, -1].mean(axis=0))))
    range_span = np.ptp(real, axis=(0, 2))
    initial_ratio = float(np.mean(np.abs(fake[:, :, 0].mean(axis=0) - real[:, :, 0].mean(axis=0)) / range_span))
    end_ratio = float(np.mean(np.abs(fake[:, :, -1].mean(axis=0) - real[:, :, -1].mean(axis=0)) / range_span))

    correlation = {}
    for index, gas in enumerate(AXES):
        correlation[gas] = {
            "initial_real": float(np.corrcoef(real_labels[:, index], real[:, index, 0])[0, 1]),
            "initial_fake": float(np.corrcoef(labels[:, index], fake[:, index, 0])[0, 1]),
            "end_real": float(np.corrcoef(real_labels[:, index], real[:, index, -1])[0, 1]),
            "end_fake": float(np.corrcoef(labels[:, index], fake[:, index, -1])[0, 1]),
        }

    report = {
        "n_real": len(real),
        "n_fake": samples,
        "generator": generator,
        "gan_epochs": gan_epochs,
        "vae_components": vae_components if generator == "vae" else None,
        "vae_beta": vae_beta if generator == "vae" else None,
        "seed": seed,
        "smoothness_ratio": fake_smooth / max(real_smooth, 1e-12),
        "initial_mae": initial_mae,
        "end_mae": end_mae,
        "initial_ratio": initial_ratio,
        "end_ratio": end_ratio,
        "nearest_neighbor_ratio": _nearest_ratio(real, fake),
        "pairwise_diversity_ratio": _pairwise_mean(fake) / max(_pairwise_mean(real), 1e-12),
        "condition_correlation": correlation,
        "note": "生成值已按训练集逐通道范围裁剪；这些指标只能排除明显失败，不能证明物理正确。",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gan_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    time = np.arange(real.shape[2]) * TIME_STEP_SECONDS
    figure, panels = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
    for index, (panel, gas) in enumerate(zip(panels, AXES, strict=True)):
        for curve in real[:, index]:
            panel.plot(time, curve, color="0.75", alpha=0.35, linewidth=0.8)
        for curve in fake[:, index]:
            panel.plot(time, curve, color="#D55E00", alpha=0.35, linewidth=0.8)
        panel.set_title(f"{gas}：真实（灰） vs 合成（橙）")
        panel.set_xlabel("时间 / 秒")
        panel.set_ylabel(f"{gas} 响应")
        panel.grid(alpha=0.2)
    figure.suptitle("GAN 生成质量快速检查：只用于排除明显不合理曲线")
    for suffix in ("png", "pdf"):
        path = output_dir / f"gan_samples.{suffix}"
        figure.savefig(path, dpi=160, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(figure)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="检查条件 GAN 的生成质量")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--gan-epochs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--generator", choices=["gan", "vae"], default="gan")
    parser.add_argument("--vae-components", type=int, default=6)
    parser.add_argument("--vae-beta", type=float, default=0.1)
    args = parser.parse_args()
    device = resolve_device(args.device)
    configure_threads(device)
    print(f"[device] {describe_device(device)}")
    check_quality(
        args.manifest,
        args.output_dir,
        args.samples,
        args.gan_epochs,
        args.seed,
        device,
        args.generator,
        args.vae_components,
        args.vae_beta,
    )


if __name__ == "__main__":
    main()
