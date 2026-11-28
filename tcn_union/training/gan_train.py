"""PCA-VAE 增强的 TCN 训练入口（保留历史文件名）。

保留的训练路径只有一种数据增强：PCA 条件 VAE。训练折内可叠加
0.1× 真实测量噪声视图，并按边缘浓度组合做条件采样；验证折始终是真实曲线。
GAN、直接 VAE、PCA 生成器和辅助头等历史模式已删除。

五折 OOF 负责选 epoch；全部 50 个开发实验重训多个 seed，预测时取平均。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcn_union.augmentation.noise import NoiseConfig, noisy_features
from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.core.data import TIME_STEP_SECONDS, make_features
from tcn_union.core.device import configure_threads, describe_device, resolve_device
from tcn_union.evaluation.evaluate import AXES, _metric_summary
from tcn_union.training.train import (
    BASE_SEED,
    MAX_EPOCHS,
    SEEDS,
    _features,
    _low_conc_powers,
    _macro_nmae,
    _read_manifest,
    _save_checkpoint,
    _target_flags,
    _train_network,
)

MODE = "raw_delta"
MODES = {
    "clean": (False, False),
    "noise": (True, False),
    "syn": (False, True),
    "noise_syn": (True, True),
}


def _edge_case_labels(rows: list[dict[str, Any]]) -> np.ndarray:
    """训练行中任一气体处于最小/最大浓度档位的标签组合（去重）。"""
    labels = np.stack(
        [np.asarray(row["_labels"], dtype=np.float64) for row in rows]
    )
    lower = labels.min(axis=0)
    upper = labels.max(axis=0)
    is_edge = np.zeros(len(labels), dtype=bool)
    for axis in range(labels.shape[1]):
        is_edge |= np.isclose(labels[:, axis], lower[axis]) | np.isclose(
            labels[:, axis], upper[axis]
        )
    edge = labels[is_edge]
    if edge.size == 0:
        return np.empty((0, labels.shape[1]), dtype=np.float64)
    return np.unique(edge, axis=0)


def _vae_features(
    rows: list[dict[str, Any]],
    count: int,
    config: VAEConfig,
    seed: int,
    device: torch.device,
    edge_copies: int = 0,
    feature_mode: str = MODE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """拟合 PCA-VAE，生成基础样本和可选的边缘条件样本。"""
    model = fit_vae(rows, config, seed, device)
    responses, labels = model.sample(count, seed + 1)
    times = (
        np.arange(round(config.seconds / TIME_STEP_SECONDS) + 1)
        * TIME_STEP_SECONDS
    )
    features = np.stack(
        [
            make_features(times, response.T, config.seconds, feature_mode)
            for response in responses
        ]
    )
    edge_count = 0
    if edge_copies > 0:
        edge_labels = _edge_case_labels(rows)
        if edge_labels.size > 0:
            repeated = np.repeat(edge_labels, edge_copies, axis=0)
            edge_responses, edge_labels_out = model.sample(
                len(repeated), seed + 2, condition_labels=repeated
            )
            edge_features = np.stack(
                [
                    make_features(
                        times, response.T, config.seconds, feature_mode
                    )
                    for response in edge_responses
                ]
            )
            features = np.concatenate((features, edge_features), axis=0)
            labels = np.concatenate(
                (labels, edge_labels_out.astype(np.float32)), axis=0
            )
            edge_count = len(repeated)
    return (
        features.astype(np.float32),
        labels.astype(np.float32),
        edge_count,
    )


def _augmented_arrays(
    rows: list[dict[str, Any]],
    seconds: float,
    use_noise: bool,
    use_synthetic: bool,
    synthetic_ratio: float,
    noise_config: NoiseConfig,
    vae_config: VAEConfig,
    seed: int,
    device: torch.device,
    edge_copies: int = 0,
    feature_mode: str = MODE,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """返回 (x, y, noise_count, synthetic_count, edge_count)。"""
    features = _features(rows, seconds, feature_mode)
    x = np.stack([features[row["file"]] for row in rows])
    y = np.stack([row["_labels"] for row in rows]).astype(np.float32)
    noise_count = 0
    synthetic_count = 0
    edge_count = 0
    if use_noise:
        x_noise, y_noise = noisy_features(
            rows, seconds, feature_mode, noise_config, seed + 17
        )
        x = np.concatenate((x, x_noise), axis=0)
        y = np.concatenate((y, y_noise), axis=0)
        noise_count = len(x_noise)
    if use_synthetic:
        count = max(1, round(synthetic_ratio * len(rows)))
        x_syn, y_syn, edge_count = _vae_features(
            rows,
            count,
            vae_config,
            seed,
            device,
            edge_copies,
            feature_mode,
        )
        synthetic_count = len(x_syn)
        x = np.concatenate((x, x_syn), axis=0)
        y = np.concatenate((y, y_syn), axis=0)
    return x, y, noise_count, synthetic_count, edge_count


def _fold_metrics(
    truth: np.ndarray, prediction: np.ndarray, train_y: np.ndarray
) -> dict[str, Any]:
    _, axis_nmae = _macro_nmae(truth, prediction, train_y)
    ranges = np.ptp(train_y, axis=0)
    errors = np.abs(prediction - truth)
    mae = errors.mean(axis=0)
    rmse = np.sqrt((errors**2).mean(axis=0))
    mape = 100.0 * (errors / np.maximum(np.abs(truth), 1e-8)).mean(axis=0)
    return {
        "macro_nmae": float(np.mean(axis_nmae)),
        "H2_nmae": float(axis_nmae[0]),
        "NH3_nmae": float(axis_nmae[1]),
        "NO2_nmae": float(axis_nmae[2]),
        "H2_mae": float(mae[0]),
        "NH3_mae": float(mae[1]),
        "NO2_mae": float(mae[2]),
        "H2_rmse": float(rmse[0]),
        "NH3_rmse": float(rmse[1]),
        "NO2_rmse": float(rmse[2]),
        "H2_nrmse": float(rmse[0] / ranges[0]),
        "NH3_nrmse": float(rmse[1] / ranges[1]),
        "NO2_nrmse": float(rmse[2] / ranges[2]),
        "H2_mape": float(mape[0]),
        "NH3_mape": float(mape[1]),
        "NO2_mape": float(mape[2]),
    }


def cross_validate(
    dev_rows: list[dict[str, Any]],
    mode: str,
    seed: int,
    device: torch.device,
    vae_config: VAEConfig,
    synthetic_ratio: float,
    noise_config: NoiseConfig,
    edge_copies: int = 0,
    max_epochs: int = MAX_EPOCHS,
    validation_metric: str = "nmae",
    target_transform: str | list[str] = "none",
    feature_mode: str = MODE,
    per_gas_head: bool = False,
    low_conc_power: float = 0.0,
    seconds: float = 10.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    use_noise, use_synthetic = MODES[mode]
    fold_scores: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    for fold in range(5):
        train_rows = [row for row in dev_rows if int(row["fold"]) != fold]
        val_rows = [row for row in dev_rows if int(row["fold"]) == fold]
        x_train, y_train, noise_count, synthetic_count, edge_count = (
            _augmented_arrays(
                train_rows,
                seconds,
                use_noise,
                use_synthetic,
                synthetic_ratio,
                noise_config,
                vae_config,
                seed + 1000 * fold,
                device,
                edge_copies,
                feature_mode,
            )
        )
        y_train_real = y_train[: len(train_rows)]
        features = _features(val_rows, seconds, feature_mode)
        x_val = np.stack([features[row["file"]] for row in val_rows])
        y_val = np.stack([row["_labels"] for row in val_rows]).astype(np.float32)

        _, _, best_epoch, best_score, prediction = _train_network(
            x_train,
            y_train,
            x_val,
            y_val,
            y_train_real,
            max_epochs,
            seed,
            device,
            True,
            validation_metric=validation_metric,
            target_transform=target_transform,
            per_gas_head=per_gas_head,
            low_conc_power=low_conc_power,
        )
        if prediction is None or best_score is None:
            raise RuntimeError("五折训练没有产生验证预测")

        row = {
            "fold": fold,
            "best_epoch": best_epoch,
            "noise_count": noise_count,
            "synthetic_count": synthetic_count,
            "edge_count": edge_count,
        }
        row.update(_fold_metrics(y_val, prediction, y_train_real))
        row["macro_nmae"] = float(best_score)
        fold_scores.append(row)
        for item, truth, pred in zip(val_rows, y_val, prediction, strict=True):
            oof.append(
                {
                    "file": item["file"],
                    "fold": fold,
                    "H2_true": float(truth[0]),
                    "NH3_true": float(truth[1]),
                    "NO2_true": float(truth[2]),
                    "H2_pred": float(pred[0]),
                    "NH3_pred": float(pred[1]),
                    "NO2_pred": float(pred[2]),
                }
            )
    return fold_scores, oof


def _summary(fold_scores: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "macro_nmae": float(
            np.mean([row["macro_nmae"] for row in fold_scores])
        ),
        "by_gas_nmae": {
            axis: float(np.mean([row[f"{axis}_nmae"] for row in fold_scores]))
            for axis in AXES
        },
        "by_gas_rmse": {
            axis: float(np.mean([row[f"{axis}_rmse"] for row in fold_scores]))
            for axis in AXES
        },
        "best_epochs": [int(row["best_epoch"]) for row in fold_scores],
        "noise_counts": [int(row["noise_count"]) for row in fold_scores],
        "synthetic_counts": [
            int(row["synthetic_count"]) for row in fold_scores
        ],
        "edge_counts": [int(row["edge_count"]) for row in fold_scores],
    }


def _global_summary(
    oof: list[dict[str, Any]], dev_ranges: np.ndarray
) -> dict[str, Any]:
    truth = np.array([[row[f"{axis}_true"] for axis in AXES] for row in oof])
    pred = np.array([[row[f"{axis}_pred"] for axis in AXES] for row in oof])
    return _metric_summary(truth, pred, dev_ranges)


def train_final(
    dev_rows: list[dict[str, Any]],
    mode: str,
    epochs: int,
    seed: int,
    device: torch.device,
    output_path: Path,
    vae_config: VAEConfig,
    synthetic_ratio: float,
    noise_config: NoiseConfig,
    edge_copies: int = 0,
    target_transform: str | list[str] = "none",
    feature_mode: str = MODE,
    per_gas_head: bool = False,
    low_conc_power: float = 0.0,
    seconds: float = 10.0,
) -> None:
    use_noise, use_synthetic = MODES[mode]
    x_dev, y_dev, noise_count, synthetic_count, edge_count = _augmented_arrays(
        dev_rows,
        seconds,
        use_noise,
        use_synthetic,
        synthetic_ratio,
        noise_config,
        vae_config,
        seed + 5000,
        device,
        edge_copies,
        feature_mode,
    )
    model, stats, _, _, _ = _train_network(
        x_dev,
        y_dev,
        None,
        None,
        y_dev,
        epochs,
        seed,
        device,
        False,
        target_transform=target_transform,
        per_gas_head=per_gas_head,
        low_conc_power=low_conc_power,
    )
    metadata = {
        "mode": mode,
        "noise": use_noise,
        "noise_scale": noise_config.scale,
        "noise_copies": noise_config.copies,
        "synthetic": use_synthetic,
        "vae_components": vae_config.components,
        "vae_latent_dim": vae_config.latent_dim,
        "vae_beta": vae_config.beta,
        "noise_count": noise_count,
        "synthetic_count": synthetic_count,
        "edge_count": edge_count,
        "edge_copies": edge_copies,
    }
    _save_checkpoint(
        output_path,
        model,
        stats,
        feature_mode,
        seconds,
        dev_rows,
        seed,
        epochs,
        device,
        augmentation=metadata,
        target_transform=target_transform,
        per_gas_head=per_gas_head,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="PCA-VAE 增强训练")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--final-seeds", type=int, nargs="+", default=SEEDS
    )
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument(
        "--modes", nargs="+", choices=sorted(MODES), default=list(MODES)
    )
    parser.add_argument(
        "--feature-mode", choices=["raw_delta", "delta"], default=MODE
    )
    parser.add_argument("--per-gas-head", action="store_true")
    parser.add_argument(
        "--low-conc-power",
        type=str,
        default="0.0",
        help="低浓度损失加权强度：单值或 H2,NH3,NO2 三值",
    )
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--noise-copies", type=int, default=1)
    parser.add_argument("--edge-copies", type=int, default=0)
    parser.add_argument("--vae-epochs", type=int, default=300)
    parser.add_argument("--vae-components", type=int, default=6)
    parser.add_argument("--vae-latent-dim", type=int, default=4)
    parser.add_argument("--vae-beta", type=float, default=0.1)
    parser.add_argument(
        "--validation-metric",
        choices=["nmae", "three_sigma"],
        default="nmae",
    )
    parser.add_argument("--target-transform", default="none")
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--oof-only", action="store_true")
    args = parser.parse_args()

    if not 2 <= args.seconds <= 10:
        parser.error("--seconds 必须在 2–10 秒之间")
    if args.synthetic_ratio <= 0.0:
        parser.error("--synthetic-ratio 必须为正")
    if args.noise_scale < 0.0:
        parser.error("--noise-scale 不能为负")
    if args.noise_copies < 1:
        parser.error("--noise-copies 必须为正")
    if args.max_epochs < 1:
        parser.error("--max-epochs 必须为正")
    if args.edge_copies < 0:
        parser.error("--edge-copies 不能为负")
    try:
        _low_conc_powers(args.low_conc_power)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        _target_flags(args.target_transform)
    except ValueError as exc:
        parser.error(str(exc))

    device = resolve_device(args.device)
    configure_threads(device)
    print(f"[device] {describe_device(device)}", flush=True)
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    dev_ranges = np.ptp(
        np.stack([row["_labels"] for row in dev_rows]), axis=0
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vae_config = VAEConfig(
        seconds=float(args.seconds),
        epochs=args.vae_epochs,
        components=args.vae_components,
        latent_dim=args.vae_latent_dim,
        beta=args.vae_beta,
    )
    noise_config = NoiseConfig(scale=args.noise_scale, copies=args.noise_copies)

    selection: dict[str, Any] = {
        "protocol": "PCA-VAE augmentation on training folds; validation real",
        "base_seed": args.seed,
        "seconds": args.seconds,
        "final_seeds": args.final_seeds,
        "mode": args.feature_mode,
        "feature_mode": args.feature_mode,
        "per_gas_head": args.per_gas_head,
        "low_conc_power": args.low_conc_power,
        "synthetic_ratio": args.synthetic_ratio,
        "noise_scale": args.noise_scale,
        "noise_copies": args.noise_copies,
        "vae_epochs": args.vae_epochs,
        "vae_components": args.vae_components,
        "vae_latent_dim": args.vae_latent_dim,
        "vae_beta": args.vae_beta,
        "edge_copies": args.edge_copies,
        "validation_metric": args.validation_metric,
        "target_transform": args.target_transform,
        "selected": {},
    }
    for mode in args.modes:
        print(f"[mode] {mode}", flush=True)
        fold_scores, oof = cross_validate(
            dev_rows,
            mode,
            args.seed,
            device,
            vae_config,
            args.synthetic_ratio,
            noise_config,
            args.edge_copies,
            args.max_epochs,
            validation_metric=args.validation_metric,
            target_transform=args.target_transform,
            feature_mode=args.feature_mode,
            per_gas_head=args.per_gas_head,
            low_conc_power=args.low_conc_power,
            seconds=float(args.seconds),
        )
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        oof_path = mode_dir / "oof.csv"
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
            writer.writeheader()
            writer.writerows(oof)

        epochs = math.ceil(
            float(np.median([row["best_epoch"] for row in fold_scores]))
        )
        print(
            f"  OOF macro-NMAE={_summary(fold_scores)['macro_nmae']:.6f} "
            f"epochs={epochs}",
            flush=True,
        )
        if not args.oof_only:
            for seed in args.final_seeds:
                path = mode_dir / f"{args.seconds}s" / f"seed_{seed}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                train_final(
                    dev_rows,
                    mode,
                    epochs,
                    seed,
                    device,
                    path,
                    vae_config,
                    args.synthetic_ratio,
                    noise_config,
                    args.edge_copies,
                    target_transform=args.target_transform,
                    feature_mode=args.feature_mode,
                    per_gas_head=args.per_gas_head,
                    low_conc_power=args.low_conc_power,
                    seconds=float(args.seconds),
                )
                print(f"  saved {path}", flush=True)

        selection["selected"][mode] = {
            "seconds": float(args.seconds),
            "noise": MODES[mode][0],
            "synthetic": MODES[mode][1],
            "selected_epochs": epochs,
            "fold_scores": fold_scores,
            "fold_summary": _summary(fold_scores),
            "global_oof": _global_summary(oof, dev_ranges),
            "oof_file": str(oof_path),
        }
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {args.output_dir / 'selection.json'}", flush=True)


if __name__ == "__main__":
    main()
