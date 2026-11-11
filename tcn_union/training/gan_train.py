"""最终 10 秒模型的训练入口，同时保留增强消融模式。

最终方案（``--modes noise_gan --generator vae``）：
- 训练折内加 0.1× 真实测量噪声；
- 用 PCA 条件 VAE 生成 3× 合成样本；
- 验证/测试始终是真实曲线，测试集只在最后评估。

五折 OOF 负责选 epoch；全部 50 个开发实验重训 3 个 seed，预测时做平均。
其他 generator/mode 只用于对比实验，不影响默认主路径。
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

from tcn_union.augmentation.direct_vae import DirectVAEConfig, fit_direct_vae
from tcn_union.augmentation.gan import GANConfig, fit_gan
from tcn_union.augmentation.noise import NoiseConfig, noisy_features
from tcn_union.augmentation.pca_generator import PCAConfig, fit_pca
from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.core.data import TIME_STEP_SECONDS, make_features
from tcn_union.core.device import configure_threads, describe_device, resolve_device
from tcn_union.evaluation.evaluate import AXES, _metric_summary
from tcn_union.training.train import (
    BASE_SEED,
    MAX_EPOCHS,
    SEEDS,
    _features,
    _macro_nmae,
    _read_manifest,
    _save_checkpoint,
    _target_flags,
    _train_network,
)

MODE = "raw_delta"
AUX_POSITIONS = (25,)
AUX_WEIGHT = 0.5
MODES = {
    "clean": (False, False, ()),
    "aux": (False, False, AUX_POSITIONS),
    "noise": (True, False, ()),
    "noise_aux": (True, False, AUX_POSITIONS),
    "gan": (False, True, ()),
    "gan_aux": (False, True, AUX_POSITIONS),
    "noise_gan": (True, True, ()),
    "noise_gan_aux": (True, True, AUX_POSITIONS),
}


def _edge_case_labels(rows: list[dict[str, Any]]) -> np.ndarray:
    """返回训练行中“任一气体处于开发集最小/最大浓度档位”的标签组合（去重）。

    边缘档位是相对误差方差的集中来源（低浓度小分母、高浓度端系统性偏差），
    后续用 VAE 按这些标签条件采样，相当于给边缘案例补充重复试验。
    """
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


def _generative_features(
    rows: list[dict[str, Any]],
    count: int,
    config: GANConfig,
    vae_config: VAEConfig,
    pca_config: PCAConfig,
    direct_vae_config: DirectVAEConfig,
    generator: str,
    seed: int,
    device: torch.device,
    edge_copies: int = 0,
    feature_mode: str = MODE,
) -> tuple[np.ndarray, np.ndarray, int]:
    """在给定训练行上拟合指定生成器，并生成 TCN 可用的特征。

    `edge_copies>0` 时（仅 VAE）额外按边缘浓度组合条件采样：每个边缘组合
    复制 `edge_copies` 次，作为“边缘案例的重复试验”拼进合成样本。
    返回 `(features, labels, edge_count)`。
    """
    if generator == "vae":
        model = fit_vae(rows, vae_config, seed, device)
    elif generator == "gan":
        model = fit_gan(rows, config, seed, device)
    elif generator == "pca":
        model = fit_pca(rows, pca_config, seed)
    elif generator == "direct_vae":
        model = fit_direct_vae(rows, direct_vae_config, seed, device)
    else:
        raise ValueError("generator 必须是 gan/vae/pca/direct_vae")
    responses, labels = model.sample(count, seed + 1)
    times = np.arange(round(config.seconds / TIME_STEP_SECONDS) + 1) * TIME_STEP_SECONDS
    features = np.stack(
        [make_features(times, response.T, config.seconds, feature_mode) for response in responses]
    )
    edge_count = 0
    if edge_copies > 0:
        if generator != "vae":
            raise ValueError("edge_copies 目前只支持 --generator vae")
        edge_labels = _edge_case_labels(rows)
        if edge_labels.size > 0:
            repeated = np.repeat(edge_labels, edge_copies, axis=0)
            edge_responses, edge_labels_out = model.sample(
                len(repeated), seed + 2, condition_labels=repeated
            )
            edge_features = np.stack(
                [
                    make_features(times, response.T, config.seconds, feature_mode)
                    for response in edge_responses
                ]
            )
            features = np.concatenate((features, edge_features), axis=0)
            labels = np.concatenate((labels, edge_labels_out.astype(np.float32)), axis=0)
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
    use_gan: bool,
    synthetic_ratio: float,
    noise_config: NoiseConfig,
    gan_config: GANConfig,
    vae_config: VAEConfig,
    pca_config: PCAConfig,
    direct_vae_config: DirectVAEConfig,
    generator: str,
    seed: int,
    device: torch.device,
    edge_copies: int = 0,
    feature_mode: str = MODE,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    """返回真实训练特征，按需拼上噪声视图和 GAN 合成特征。

    返回 `(x, y, noise_count, gan_count, edge_count)`；gan_count 含边缘复制样本。
    """
    features = _features(rows, seconds, feature_mode)
    x = np.stack([features[row["file"]] for row in rows])
    y = np.stack([row["_labels"] for row in rows]).astype(np.float32)
    noise_count = 0
    gan_count = 0
    edge_count = 0
    if use_noise:
        x_noise, y_noise = noisy_features(
            rows, seconds, feature_mode, noise_config, seed + 17
        )
        x = np.concatenate((x, x_noise), axis=0)
        y = np.concatenate((y, y_noise), axis=0)
        noise_count = len(x_noise)
    if use_gan:
        base_gan_count = max(1, round(synthetic_ratio * len(rows)))
        x_gan, y_gan, edge_count = _generative_features(
            rows,
            base_gan_count,
            gan_config,
            vae_config,
            pca_config,
            direct_vae_config,
            generator,
            seed,
            device,
            edge_copies,
            feature_mode,
        )
        gan_count = len(x_gan)
        x = np.concatenate((x, x_gan), axis=0)
        y = np.concatenate((y, y_gan), axis=0)
    return x, y, noise_count, gan_count, edge_count


def _fold_metrics(
    truth: np.ndarray, prediction: np.ndarray, train_y: np.ndarray
) -> dict[str, Any]:
    """计算逐折指标；NMAE 使用该折训练标签范围。"""
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
    gan_config: GANConfig,
    vae_config: VAEConfig,
    pca_config: PCAConfig,
    direct_vae_config: DirectVAEConfig,
    generator: str,
    synthetic_ratio: float,
    noise_config: NoiseConfig,
    ema_decay: float | None,
    edge_copies: int = 0,
    max_epochs: int = MAX_EPOCHS,
    validation_metric: str = "nmae",
    target_transform: str | list[str] = "none",
    feature_mode: str = MODE,
    per_gas_head: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """对一种训练模式跑五折 OOF。

    `target_transform` 透传给 ``_train_network``：单值或逐气体配置，log 气体
    在 log 浓度空间训练和早停，返回的验证预测仍是原始浓度。`edge_copies>0`
    时用 VAE 对边缘浓度组合额外条件采样（见 ``_generative_features``）。
    """
    # MODES 决定这一轮是否加噪声/生成样本/辅助头；最终方案是 noise_gan。
    use_noise, use_gan, aux_positions = MODES[mode]
    fold_scores: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    # 五折是 manifest 中预先固定的实验级划分，不在这里随机切分。
    for fold in range(5):
        train_rows = [row for row in dev_rows if int(row["fold"]) != fold]
        val_rows = [row for row in dev_rows if int(row["fold"]) == fold]

        # 增强只作用于训练折；验证折始终保留真实曲线。
        # 噪声和生成器的随机种子都随 fold 改变，避免不同折共用同一批合成样本。
        x_train, y_train, noise_count, gan_count, edge_count = _augmented_arrays(
            train_rows,
            10.0,
            use_noise,
            use_gan,
            synthetic_ratio,
            noise_config,
            gan_config,
            vae_config,
            pca_config,
            direct_vae_config,
            generator,
            seed + 1000 * fold,
            device,
            edge_copies,
            feature_mode,
        )
        # 前 len(train_rows) 条是真实样本；后面是噪声/合成样本。
        # 早停和 NMAE 的分母仍只用真实训练标签，避免合成标签改变评价尺度。
        y_train_real = y_train[: len(train_rows)]
        features = _features(val_rows, 10.0, feature_mode)
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
            aux_positions=aux_positions,
            aux_weight=AUX_WEIGHT,
            ema_decay=ema_decay,
            validation_metric=validation_metric,
            target_transform=target_transform,
            per_gas_head=per_gas_head,
        )
        if prediction is None or best_score is None:
            raise RuntimeError("五折训练没有产生验证预测")

        row = {
            "fold": fold,
            "best_epoch": best_epoch,
            "noise_count": noise_count,
            "gan_count": gan_count,
            "edge_count": edge_count,
            "synthetic_count": noise_count + gan_count,
        }
        row.update(_fold_metrics(y_val, prediction, y_train_real))
        row["macro_nmae"] = float(best_score)
        fold_scores.append(row)
        # 每个验证实验只在它所属的折里预测一次，最后拼成 50 条 OOF。
        for row, truth, pred in zip(val_rows, y_val, prediction, strict=True):
            oof.append(
                {
                    "file": row["file"],
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
        "macro_nmae": float(np.mean([row["macro_nmae"] for row in fold_scores])),
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
        "gan_counts": [int(row["gan_count"]) for row in fold_scores],
        "edge_counts": [int(row["edge_count"]) for row in fold_scores],
        "synthetic_counts": [int(row["synthetic_count"]) for row in fold_scores],
    }


def _global_summary(oof: list[dict[str, Any]], dev_ranges: np.ndarray) -> dict[str, Any]:
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
    gan_config: GANConfig,
    vae_config: VAEConfig,
    pca_config: PCAConfig,
    direct_vae_config: DirectVAEConfig,
    generator: str,
    synthetic_ratio: float,
    noise_config: NoiseConfig,
    ema_decay: float | None,
    edge_copies: int = 0,
    target_transform: str | list[str] = "none",
    feature_mode: str = MODE,
    per_gas_head: bool = False,
) -> None:
    """在全部 50 个开发实验上训练一个最终模型。

    与 ``cross_validate`` 的区别：这里没有验证集、没有早停，训练轮数由五折
    最佳 epoch 中位数固定；测试集仍然不参与。`target_transform` 同时传给
    训练和 checkpoint（逐气体配置会以长度 3 列表记录），供推理端还原原始浓度。
    """
    use_noise, use_gan, aux_positions = MODES[mode]
    # 最终训练也只在开发集上构造增强样本。
    x_dev, y_dev, noise_count, gan_count, edge_count = _augmented_arrays(
        dev_rows,
        10.0,
        use_noise,
        use_gan,
        synthetic_ratio,
        noise_config,
        gan_config,
        vae_config,
        pca_config,
        direct_vae_config,
        generator,
        seed + 5000,
        device,
        edge_copies,
        feature_mode,
    )
    # early_stop=False：最终重训没有验证集，用固定 epochs 训练。
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
        aux_positions=aux_positions,
        aux_weight=AUX_WEIGHT,
        ema_decay=ema_decay,
        target_transform=target_transform,
        per_gas_head=per_gas_head,
    )
    metadata = {
        "mode": mode,
        "noise": use_noise,
        "noise_scale": noise_config.scale,
        "noise_copies": noise_config.copies,
        "gan": use_gan,
        "generator": generator,
        "vae_components": vae_config.components if generator == "vae" else None,
        "vae_latent_dim": vae_config.latent_dim if generator == "vae" else None,
        "vae_beta": vae_config.beta if generator == "vae" else None,
        "pca_components": pca_config.components if generator == "pca" else None,
        "direct_vae_latent_dim": direct_vae_config.latent_dim if generator == "direct_vae" else None,
        "noise_count": noise_count,
        "gan_count": gan_count,
        "edge_count": edge_count,
        "edge_copies": edge_copies,
        "synthetic_count": noise_count + gan_count,
        "aux_positions": list(aux_positions),
        "aux_weight": AUX_WEIGHT if aux_positions else None,
        "ema_decay": ema_decay,
    }
    _save_checkpoint(
        output_path,
        model,
        stats,
        feature_mode,
        10.0,
        dev_rows,
        seed,
        epochs,
        device,
        augmentation=metadata,
        target_transform=target_transform,
        per_gas_head=per_gas_head,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="GAN 扩充 + 辅助监督消融")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument(
        "--final-seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help="全开发集最终重训使用的种子列表（默认与历史协议一致）",
    )
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=list(MODES))
    parser.add_argument(
        "--feature-mode",
        choices=["raw_delta", "delta"],
        default=MODE,
        help="TCN 输入特征模式（默认 raw_delta；delta 只保留相对增量）",
    )
    parser.add_argument(
        "--per-gas-head",
        action="store_true",
        help="使用逐气体输出头（NO2 专项实验）",
    )
    parser.add_argument("--synthetic-ratio", type=float, default=1.0)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--noise-copies", type=int, default=1)
    parser.add_argument("--generator", choices=["gan", "pca", "vae", "direct_vae"], default="gan")
    parser.add_argument(
        "--edge-copies",
        type=int,
        default=0,
        help="用 VAE 对边缘浓度组合条件采样的重复次数（0 关闭；仅 --generator vae）",
    )
    parser.add_argument("--gan-epochs", type=int, default=300)
    parser.add_argument("--vae-components", type=int, default=6)
    parser.add_argument("--vae-latent-dim", type=int, default=4)
    parser.add_argument("--vae-beta", type=float, default=0.1)
    parser.add_argument("--pca-components", type=int, default=6)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument(
        "--validation-metric",
        choices=["nmae", "three_sigma"],
        default="nmae",
        help="早停指标：macro-NMAE 或相对误差 3σ 宏平均",
    )
    parser.add_argument(
        "--target-transform",
        default="none",
        help=(
            "标签空间：none/log 表示三气体共用，或按 H2,NH3,NO2 逗号分隔，"
            "如 none,log,log；log 气体在 log 浓度空间训练以对齐相对误差"
        ),
    )
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--oof-only", action="store_true")
    args = parser.parse_args()

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
    if args.edge_copies > 0 and args.generator != "vae":
        parser.error("--edge-copies 目前只支持 --generator vae")
    try:
        # 提前校验标签空间配置，避免读到 manifest、开始五折训练后才报错。
        _target_flags(args.target_transform)
    except ValueError as exc:
        parser.error(str(exc))
    device = resolve_device(args.device)
    configure_threads(device)
    print(f"[device] {describe_device(device)}", flush=True)
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    dev_ranges = np.ptp(np.stack([row["_labels"] for row in dev_rows]), axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gan_config = GANConfig(
        seconds=10.0,
        epochs=args.gan_epochs,
        residual=False,
        anchor_endpoints=False,
    )
    vae_config = VAEConfig(
        seconds=10.0,
        epochs=args.gan_epochs,
        components=args.vae_components,
        latent_dim=args.vae_latent_dim,
        beta=args.vae_beta,
    )
    pca_config = PCAConfig(seconds=10.0, components=args.pca_components)
    direct_vae_config = DirectVAEConfig(
        seconds=10.0,
        epochs=args.gan_epochs,
        latent_dim=args.vae_latent_dim,
        beta=args.vae_beta,
    )
    noise_config = NoiseConfig(scale=args.noise_scale, copies=args.noise_copies)

    selection: dict[str, Any] = {
        "protocol": "train-fold-only noise/GAN; clean/aux/noise/gan/combined ablation; test untouched",
        "base_seed": args.seed,
        "final_seeds": args.final_seeds,
        "mode": args.feature_mode,
        "feature_mode": args.feature_mode,
        "per_gas_head": args.per_gas_head,
        "synthetic_ratio": args.synthetic_ratio,
        "noise_scale": args.noise_scale,
        "noise_copies": args.noise_copies,
        "generator": args.generator,
        "gan_epochs": args.gan_epochs,
        "vae_components": args.vae_components,
        "vae_latent_dim": args.vae_latent_dim,
        "vae_beta": args.vae_beta,
        "pca_components": args.pca_components,
        "ema_decay": args.ema_decay,
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
            gan_config,
            vae_config,
            pca_config,
            direct_vae_config,
            args.generator,
            args.synthetic_ratio,
            noise_config,
            args.ema_decay,
            args.edge_copies,
            args.max_epochs,
            validation_metric=args.validation_metric,
            target_transform=args.target_transform,
            feature_mode=args.feature_mode,
            per_gas_head=args.per_gas_head,
        )
        mode_dir = args.output_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        oof_path = mode_dir / "oof.csv"
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
            writer.writeheader()
            writer.writerows(oof)

        # 每个方案用 5 折最佳 epoch 的中位数作为全开发集重训轮数。
        epochs = math.ceil(float(np.median([row["best_epoch"] for row in fold_scores])))
        print(
            f"  OOF macro-NMAE={_summary(fold_scores)['macro_nmae']:.6f} epochs={epochs}",
            flush=True,
        )
        if not args.oof_only:
            for seed in args.final_seeds:
                path = mode_dir / "10s" / f"seed_{seed}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                train_final(
                    dev_rows,
                    mode,
                    epochs,
                    seed,
                    device,
                    path,
                    gan_config,
                    vae_config,
                    pca_config,
                    direct_vae_config,
                    args.generator,
                    args.synthetic_ratio,
                    noise_config,
                    args.ema_decay,
                    args.edge_copies,
                    target_transform=args.target_transform,
                    feature_mode=args.feature_mode,
                    per_gas_head=args.per_gas_head,
                )
                print(f"  saved {path}", flush=True)

        selection["selected"][mode] = {
            "seconds": 10.0,
            "noise": MODES[mode][0],
            "gan": MODES[mode][1],
            "aux_positions": list(MODES[mode][2]),
            "aux_weight": AUX_WEIGHT if MODES[mode][2] else None,
            "selected_epochs": epochs,
            "fold_scores": fold_scores,
            "fold_summary": _summary(fold_scores),
            "global_oof": _global_summary(oof, dev_ranges),
            "oof_file": str(oof_path),
        }
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Wrote {args.output_dir / 'selection.json'}", flush=True)


if __name__ == "__main__":
    main()
