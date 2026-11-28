"""组合使用：PCA-VAE 与传感器模型生成的数据结合使用。

训练时先按完整协议得到 PCA-VAE 基线；随后用传感器模型与 PCA-VAE 的联合样本
微调 H2 头，同时冻结共享卷积骨干和 NH3/NO2 头。NH3/NO2 头因此保持 VAE
基线的预测，H2 头可从传感器模型样本获益。本脚本只评估这一种组合方式。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tcn_union.augmentation.noise import NoiseConfig, noisy_features
from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.training.gan_train import (
    MODE,
    _edge_case_labels,
    _read_manifest,
)
from tcn_union.training.train import _restore_target, _train_network

from tcn_union.experiments.full_protocol_augmentation import (
    DEFAULT_CONFIGS,
    DEFAULT_SEEDS,
    EDGE_COPIES,
    EPOCHS,
    NOISE_SCALE,
    SYNTHETIC_FACTOR,
    _clip,
    _fit_generators,
    _forward_prob_joint_features,
    _initial_for_labels,
    _metric_values,
    _real_features,
    _training_concentrations,
    _vae_features,
    _write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/no2_guard"
HEAD_EPOCHS = 200
PATIENCE = 40
BATCH = 8


def _forward_features(forward, train_rows, labels, initial, seconds, rng):
    return _forward_prob_joint_features(
        forward, train_rows, labels, initial, seconds, rng, components=10
    )


def _h2_weights(count: int) -> np.ndarray:
    return np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))


def _h2_union(
    forward,
    vae,
    train_rows,
    labels,
    seconds,
    seed,
    rng,
    n_vae: int,
    n_forward: int,
):
    """VAE 三气体样本 + 传感器模型三气体样本，传感器模型样本只监督 H2。"""
    count = max(SYNTHETIC_FACTOR * 40, n_forward)
    target = _training_concentrations(labels, count, rng)
    initial = _initial_for_labels(train_rows, target)
    vae_data = _vae_features(vae, n_vae, seed + 1, seconds)
    forward_data = _forward_features(
        forward, train_rows, target[:n_forward], initial[:n_forward], seconds, rng
    )
    return (
        np.concatenate((vae_data[0], forward_data[0]), axis=0),
        np.concatenate((vae_data[1], forward_data[1]), axis=0),
        np.concatenate((np.ones((n_vae, 3)), _h2_weights(n_forward)), axis=0),
    )


def _vae_edge(vae, train_rows, seconds, aug_seed):
    edge_labels = _edge_case_labels(train_rows)
    if edge_labels.size == 0:
        return None
    repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
    return _vae_features(
        vae, len(repeated), aug_seed + 2, seconds, condition_labels=repeated
    )


def _h2_union_edge(
    forward,
    vae,
    train_rows,
    seconds,
    rng,
    aug_seed,
    n_forward: int,
):
    edge_labels = _edge_case_labels(train_rows)
    if edge_labels.size == 0:
        return None
    repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
    n_forward = min(n_forward, len(repeated))
    initial = _initial_for_labels(train_rows, repeated)
    vae_data = _vae_features(
        vae, len(repeated), aug_seed + 2, seconds, condition_labels=repeated
    )
    forward_data = _forward_features(
        forward, train_rows, repeated[:n_forward], initial[:n_forward], seconds, rng
    )
    return (
        np.concatenate((vae_data[0], forward_data[0]), axis=0),
        np.concatenate((vae_data[1], forward_data[1]), axis=0),
        np.concatenate((np.ones((len(repeated), 3)), _h2_weights(n_forward)), axis=0),
    )


def _h2_z(y: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    transformed = np.log(np.maximum(y[:, 0], 1e-6))
    return ((transformed - stats["y_mean"][0]) / stats["y_std"][0]).astype(
        np.float32
    )


def _fit_h2_head(
    model,
    stats: dict[str, np.ndarray],
    x_union: np.ndarray,
    y_union: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    device: torch.device,
) -> tuple[object, int]:
    """冻结共享骨干和 NH3/NO2 头，只训练 H2 头。"""
    for parameter in model.blocks.parameters():
        parameter.requires_grad_(False)
    for index, head in enumerate(model.heads):
        for parameter in head.parameters():
            parameter.requires_grad_(index == 0)
    model.blocks.eval()
    model.heads[1].eval()
    model.heads[2].eval()

    x_z = (
        (x_union - stats["x_mean"][None, :, None])
        / stats["x_std"][None, :, None]
    ).astype(np.float32)
    y_z = _h2_z(y_union, stats)
    x_val_z = (
        (x_val - stats["x_mean"][None, :, None])
        / stats["x_std"][None, :, None]
    ).astype(np.float32)

    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.from_numpy(x_z), torch.from_numpy(y_z)
        ),
        batch_size=BATCH,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.heads[0].parameters(), lr=1e-3, weight_decay=1e-3
    )
    criterion = nn.SmoothL1Loss(beta=1.0)

    def validation_score() -> float:
        model.eval()
        with torch.no_grad():
            prediction = model(torch.from_numpy(x_val_z).to(device)).cpu().numpy()
        prediction = (
            prediction * stats["y_std"][None, :] + stats["y_mean"][None, :]
        )
        prediction = _restore_target(prediction, "log,none,log")
        return _metric_values(prediction, y_val)["macro_three_sigma"]

    # 先评估未微调的 VAE 头，保证组合使用至少不劣于 VAE。
    best_score = validation_score()
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in model.heads[0].state_dict().items()
    }
    best_epoch = 0
    stale = 0
    for epoch in range(1, HEAD_EPOCHS + 1):
        model.heads[0].train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                features = model.blocks(batch_x)[:, :, -1]
            prediction = model.heads[0](features).squeeze(-1)
            loss = criterion(prediction, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite head loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.heads[0].parameters(), 1.0)
            optimizer.step()
        score = validation_score()
        if score < best_score:
            best_score = score
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.heads[0].state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError("H2 head fine-tuning produced no finite checkpoint")
    model.heads[0].load_state_dict(best_state)
    model.eval()
    return model, best_epoch


def run(
    folds: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    seconds: int = 10,
    configs: tuple[str, ...] = DEFAULT_CONFIGS,
    output: Path | str = BASE / "pilot",
) -> None:
    out = Path(output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    configure_threads(device)
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    y_dev = np.stack([row["_labels"] for row in development]).astype(np.float32)
    from sensor_array.core.data import load_experiments

    experiments = {
        experiment.file_id: experiment for experiment in load_experiments()
    }
    fold_rows: list[dict] = []
    predictions: dict[tuple[str, int, str], dict[int, np.ndarray]] = {}
    truths: dict[int, np.ndarray] = {}
    started = time.time()

    for fold in range(folds):
        train_rows = [row for row in development if int(row["fold"]) != fold]
        val_rows = [row for row in development if int(row["fold"]) == fold]
        train_experiments = [
            experiments[int(Path(row["file"]).stem)] for row in train_rows
        ]
        x_real, y_real = _real_features(train_rows, seconds)
        x_val, y_val = _real_features(val_rows, seconds)
        truths[fold] = y_val
        labels = np.stack([row["_labels"] for row in train_rows]).astype(float)
        x_noise, y_noise = noisy_features(
            train_rows,
            seconds,
            MODE,
            NoiseConfig(scale=NOISE_SCALE, copies=1),
            fold + 17,
        )

        for seed in seeds:
            aug_seed = seed + 1000 * fold
            forward, vae = _fit_generators(
                train_rows, train_experiments, aug_seed, device, True, True
            )
            rng = np.random.default_rng(aug_seed)
            vae_data = _vae_features(vae, SYNTHETIC_FACTOR * 40, aug_seed + 1, seconds)
            vae_edge = _vae_edge(vae, train_rows, seconds, aug_seed)
            base_x = np.concatenate(
                (x_real, x_noise, vae_data[0], vae_edge[0]), axis=0
            )
            base_y = np.concatenate(
                (y_real, y_noise, vae_data[1], vae_edge[1]), axis=0
            )
            union = _h2_union(
                forward,
                vae,
                train_rows,
                labels,
                seconds,
                aug_seed,
                rng,
                SYNTHETIC_FACTOR * 40,
                180,
            )
            union_edge = _h2_union_edge(
                forward, vae, train_rows, seconds, rng, aug_seed, 180
            )
            union_x = np.concatenate(
                (x_real, x_noise, union[0], union_edge[0]), axis=0
            )
            union_y = np.concatenate(
                (y_real, y_noise, union[1], union_edge[1]), axis=0
            )

            for config in configs:
                model, stats, base_epochs, _, base_pred = _train_network(
                    base_x,
                    base_y,
                    x_val,
                    y_val,
                    y_real,
                    EPOCHS,
                    seed,
                    device,
                    True,
                    validation_metric="three_sigma",
                    target_transform="log,none,log",
                    per_gas_head=True,
                    low_conc_power=config,
                )
                base_pred = _clip(base_pred, y_dev)
                base_metrics = _metric_values(base_pred, y_val)
                fold_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "config": config,
                        "arm": "vae",
                        "n_train": len(base_y),
                        "epochs": base_epochs,
                        **base_metrics,
                    }
                )
                predictions.setdefault(("vae", seed, config), {})[fold] = base_pred

                model, head_epochs = _fit_h2_head(
                    model, stats, union_x, union_y, x_val, y_val, seed, device
                )
                x_val_z = (
                    (x_val - stats["x_mean"][None, :, None])
                    / stats["x_std"][None, :, None]
                ).astype(np.float32)
                with torch.no_grad():
                    prediction = model(
                        torch.from_numpy(x_val_z).to(device)
                    ).cpu().numpy()
                prediction = (
                    prediction * stats["y_std"][None, :] + stats["y_mean"][None, :]
                )
                prediction = _restore_target(prediction, "log,none,log")
                prediction = _clip(prediction, y_dev)
                metrics = _metric_values(prediction, y_val)
                fold_rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "config": config,
                        "arm": "guard",
                        "n_train": len(union_y),
                        "epochs": head_epochs,
                        **metrics,
                    }
                )
                predictions.setdefault(("guard", seed, config), {})[fold] = prediction
        print(f"fold {fold} done ({time.time() - started:.0f}s)", flush=True)

    _write_csv(out / "folds.csv", fold_rows)
    truth = np.concatenate([truths[fold] for fold in range(folds)], axis=0)
    oof_rows = []
    arrays = {}
    for (arm, seed, config), fold_predictions in predictions.items():
        prediction = np.concatenate(
            [fold_predictions[fold] for fold in range(folds)], axis=0
        )
        arrays[f"{arm}|{seed}|{config}"] = prediction
        oof_rows.append(
            {
                "arm": arm,
                "seed": seed,
                "config": config,
                **_metric_values(prediction, truth),
            }
        )
    _write_csv(out / "oof.csv", oof_rows)
    np.savez(out / "predictions.npz", **arrays)
    (out / "run_config.json").write_text(
        json.dumps(
            {
                "folds": folds,
                "seeds": list(seeds),
                "seconds": seconds,
                "configs": list(configs),
                "head_epochs": HEAD_EPOCHS,
                "patience": PATIENCE,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument(
        "--configs",
        default=None,
        help="分号分隔的配置；默认 DEFAULT_CONFIGS（第二个含逗号）",
    )
    parser.add_argument("--output", default=str(BASE / "pilot"))
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(int(item) for item in args.seeds.split(",")),
        seconds=args.seconds,
        configs=(
            DEFAULT_CONFIGS
            if args.configs is None
            else tuple(item.strip() for item in args.configs.split(";"))
        ),
        output=args.output,
    )


if __name__ == "__main__":
    main()
