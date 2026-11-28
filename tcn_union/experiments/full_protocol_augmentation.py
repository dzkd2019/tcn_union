"""三种保留增强方式的统一训练脚本：仅真实、PCA-VAE、传感器模型生成。

保留三种数据增强方式：
* real：只用训练折真实曲线；
* vae：PCA-VAE 合成曲线；
* forward_prob_joint10：传感器模型均值 + 三气体联合残差 PCA 条件采样。

三者共用旧完整训练协议：10 s 前缀、0.1×MAD 噪声视图、边缘浓度复制、
两套低浓度配置、多个 seed、6 checkpoint 平均、裁剪到开发集浓度范围。
锁定 12 条不参与。

输出目录 artifacts/augmentation/protocols/full/：
folds.csv、fold_seed.csv、oof_seed.csv、oof.csv、summary.csv、
predictions.npz、summary.md、run_config.json。
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from tcn_union.augmentation.noise import NoiseConfig, noisy_features
from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.core.data import make_features
from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.training.gan_train import MODE, _edge_case_labels, _read_manifest
from tcn_union.training.train import _train_network, fit_scalers

from sensor_array.core.data import load_experiments
from sensor_array.core.double_exponential import DoubleExponentialArrayModel
from sensor_array.core.mlp_per_gas import PerGasMLPDoubleExponentialModel

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/protocols/full"
ARMS = ("real", "vae", "forward_prob_joint10")
DEFAULT_SEEDS = (20260923, 20260924, 20260925)
DEFAULT_CONFIGS = ("1.0", "0,0.3,1.0")
EPOCHS = 500
SYNTHETIC_FACTOR = 3
NOISE_SCALE = 0.1
EDGE_COPIES = 4
GASES = ("H2", "NH3", "NO2")


def _times(seconds: int) -> np.ndarray:
    return np.arange(round(seconds / 0.2) + 1) * 0.2


def _training_concentrations(
    labels: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    """从训练标签中有放回抽样，不产生新的浓度组合。"""
    index = rng.integers(0, len(labels), size=count)
    return labels[index]


def _real_features(rows: list[dict], seconds: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.stack(
        [make_features(row["_time"], row["_response"], seconds, MODE) for row in rows]
    )
    y = np.stack([row["_labels"] for row in rows]).astype(np.float32)
    return x.astype(np.float32), y


def _metric_values(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for axis, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, axis] - truth[:, axis]) / np.maximum(
            np.abs(truth[:, axis]), 1e-9
        )
        absolute = np.abs(relative)
        out[f"{gas}_three_sigma"] = float(3.0 * np.std(relative, ddof=1))
        out[f"{gas}_mape"] = float(np.mean(absolute))
        out[f"{gas}_median"] = float(np.median(absolute))
        out[f"{gas}_p90"] = float(np.quantile(absolute, 0.9))
    for key in ("three_sigma", "mape", "median", "p90"):
        out[f"macro_{key}"] = float(np.mean([out[f"{gas}_{key}"] for gas in GASES]))
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prob_design(values: np.ndarray) -> np.ndarray:
    u = np.log(values)
    return np.column_stack(
        (
            np.ones(len(u)),
            u,
            u**2,
            u[:, [0]] * u[:, [1]],
            u[:, [0]] * u[:, [2]],
            u[:, [1]] * u[:, [2]],
        )
    )


def _forward_prob_joint_features(
    model: PerGasMLPDoubleExponentialModel,
    rows: list[dict],
    labels: np.ndarray,
    initial: np.ndarray,
    seconds: int,
    rng: np.random.Generator,
    components: int = 10,
    ridge: float = 1e-2,
    shrink: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """正模型均值 + 三气体联合残差的低秩条件分布采样。"""
    times = _times(seconds)
    train_labels = np.stack([row["_labels"] for row in rows]).astype(np.float64)
    train_response = np.stack(
        [np.asarray(row["_response"])[: len(times)] for row in rows]
    )
    train_initial = train_response[:, 0, :]
    train_relative = train_response / train_initial[:, None, :] - 1.0
    train_prediction = model.predict_relative(train_labels, times)
    residual = (train_relative - train_prediction).reshape(len(rows), -1)

    design = _prob_design(train_labels)
    design_mean = design[:, 1:].mean(axis=0)
    design_scale = design[:, 1:].std(axis=0)
    design_scale[design_scale == 0] = 1.0
    design[:, 1:] = (design[:, 1:] - design_mean) / design_scale
    target_design = _prob_design(labels)
    target_design[:, 1:] = (target_design[:, 1:] - design_mean) / design_scale

    mean = residual.mean(axis=0)
    centered = residual - mean[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    count = min(components, len(rows) - 1, residual.shape[1])
    basis = vt[:count]
    scores = centered @ basis.T

    penalty = ridge * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ scores)
    error = scores - design @ coefficients
    covariance = np.atleast_2d(np.cov(error, rowvar=False))
    covariance = 0.5 * (covariance + covariance.T)
    covariance += shrink * np.trace(covariance) / count * np.eye(count)
    covariance += 1e-10 * np.eye(count)

    predicted = target_design @ coefficients
    noise = rng.multivariate_normal(np.zeros(count), covariance, size=len(labels))
    reconstruction = mean[None, :] + (predicted + noise) @ basis
    reconstruction = reconstruction.reshape(len(labels), len(times), 3)
    low = residual.reshape(len(rows), len(times), 3).min(axis=0)
    high = residual.reshape(len(rows), len(times), 3).max(axis=0)
    sampled = np.clip(reconstruction, low[None, :, :], high[None, :, :])

    relative = model.predict_relative(labels, times) + sampled
    response = initial[:, None, :] * (1.0 + relative)
    features = np.stack(
        [make_features(times, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _vae_features(
    model,
    count: int,
    seed: int,
    seconds: int,
    condition_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    responses, labels = model.sample(
        count, seed, condition_labels=condition_labels
    )
    times = _times(seconds)
    features = np.stack(
        [make_features(times, response.T, seconds, MODE) for response in responses]
    )
    return features.astype(np.float32), labels.astype(np.float32)


def _initial_for_labels(rows: list[dict], labels: np.ndarray) -> np.ndarray:
    """按标签查真实 R0；查不到时从训练折 R0 经验分布抽。"""
    lookup = {
        tuple(np.asarray(row["_labels"], dtype=np.float64)): row["_response"][0]
        for row in rows
    }
    values = np.stack([row["_response"][0] for row in rows])
    rng = np.random.default_rng(0)
    initial = []
    for label in labels:
        key = tuple(np.asarray(label, dtype=np.float64))
        if key in lookup:
            initial.append(lookup[key])
        else:
            initial.append(values[rng.integers(0, len(values))])
    return np.stack(initial)


def _fit_generators(
    train_rows: list[dict],
    train_experiments: list,
    seed: int,
    device,
    need_forward: bool,
    need_vae: bool,
):
    forward = None
    vae = None
    if need_forward:
        skeleton = DoubleExponentialArrayModel().fit(train_experiments)
        forward = PerGasMLPDoubleExponentialModel(
            device=str(device), epochs=600, random_state=seed
        ).fit(train_experiments, skeleton=skeleton)
    if need_vae:
        vae = fit_vae(train_rows, VAEConfig(seconds=10.0), seed, device)
    return forward, vae


def _train(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    y_real: np.ndarray,
    seed: int,
    device,
    epochs: int,
    low_conc_power: str,
    scaler_stats: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, int]:
    _, _, epochs_used, _, prediction = _train_network(
        x_train,
        y_train,
        x_val,
        y_val,
        y_real,
        epochs,
        seed,
        device,
        True,
        validation_metric="three_sigma",
        target_transform="log,none,log",
        per_gas_head=True,
        low_conc_power=low_conc_power,
        scaler_stats=scaler_stats,
    )
    if prediction is None:
        raise RuntimeError("训练没有产生验证预测")
    return prediction, epochs_used


def _clip(prediction: np.ndarray, labels: np.ndarray) -> np.ndarray:
    return np.clip(prediction, labels.min(axis=0), labels.max(axis=0))


def _transformed_labels(labels: np.ndarray) -> np.ndarray:
    transformed = np.asarray(labels, dtype=np.float64).copy()
    transformed[:, 0] = np.log(np.maximum(transformed[:, 0], 1e-6))
    transformed[:, 2] = np.log(np.maximum(transformed[:, 2], 1e-6))
    return transformed


def _summary_markdown(summary_rows: list[dict]) -> str:
    lines = [
        "| arm | seeds | configs | 宏 3σ / % | 宏 MAPE / % | 宏中位相对误差 / % | 宏 p90 / % |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['arm']} | {row['seeds']} | {row['configs']} | "
            f"{row['seed_mean_macro_three_sigma']:.3f} | "
            f"{row['seed_mean_macro_mape']:.3f} | "
            f"{row['seed_mean_macro_median']:.3f} | "
            f"{row['seed_mean_macro_p90']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def run(
    folds: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    seconds: int = 10,
    epochs: int = EPOCHS,
    configs: tuple[str, ...] = DEFAULT_CONFIGS,
    arms: tuple[str, ...] = ARMS,
    output: Path | str = BASE,
    real_scaler: bool = False,
) -> None:
    if folds < 1 or folds > 5:
        raise ValueError("folds 必须在 1–5 之间")
    if seconds < 2 or seconds > 10:
        raise ValueError("seconds 必须在 2–10 之间")
    if not set(arms) <= set(ARMS):
        raise ValueError(f"arms 必须是 {ARMS} 的子集")
    if not configs:
        raise ValueError("至少需要一个低浓度配置")

    out = Path(output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    if len(development) != 50:
        raise ValueError(f"开发集应为 50 条，实际 {len(development)}")
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    y_dev = np.stack([row["_labels"] for row in development]).astype(np.float32)
    synthetic_count = SYNTHETIC_FACTOR * 40
    need_forward = "forward_prob_joint10" in arms
    need_vae = "vae" in arms

    fold_rows: list[dict] = []
    fold_predictions: dict[tuple[str, int, int, int], np.ndarray] = {}
    fold_truth: dict[int, np.ndarray] = {}
    truth_parts: list[np.ndarray] = []
    file_ids: list[int] = []
    started = time.time()

    for fold in range(folds):
        train_rows = [row for row in development if int(row["fold"]) != fold]
        val_rows = [row for row in development if int(row["fold"]) == fold]
        train_experiments = [
            experiments[int(Path(row["file"]).stem)] for row in train_rows
        ]
        x_real, y_real = _real_features(train_rows, seconds)
        x_val, y_val = _real_features(val_rows, seconds)
        scaler_stats = (
            fit_scalers(x_real, _transformed_labels(y_real)) if real_scaler else None
        )
        fold_truth[fold] = y_val
        truth_parts.append(y_val)
        file_ids.extend(int(Path(row["file"]).stem) for row in val_rows)

        for seed in seeds:
            aug_seed = seed + 1000 * fold
            forward, vae = _fit_generators(
                train_rows,
                train_experiments,
                aug_seed,
                device,
                need_forward,
                need_vae,
            )
            labels = np.stack([row["_labels"] for row in train_rows]).astype(
                np.float64
            )
            rng = np.random.default_rng(aug_seed)
            x_noise, y_noise = noisy_features(
                train_rows,
                seconds,
                MODE,
                NoiseConfig(scale=NOISE_SCALE, copies=1),
                aug_seed + 17,
            )

            vae_base = (
                _vae_features(vae, synthetic_count, aug_seed + 1, seconds)
                if vae is not None
                else (np.empty((0, 0)), np.empty((0, 3)))
            )
            joint_labels = _training_concentrations(labels, synthetic_count, rng)
            joint_initial = _initial_for_labels(train_rows, joint_labels)
            joint_base = (
                _forward_prob_joint_features(
                    forward,
                    train_rows,
                    joint_labels,
                    joint_initial,
                    seconds,
                    rng,
                    components=10,
                )
                if forward is not None
                else (np.empty((0, 0)), np.empty((0, 3)))
            )

            edge_labels = _edge_case_labels(train_rows)
            if edge_labels.size:
                repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
                edge_initial = _initial_for_labels(train_rows, repeated)
                edge_vae = (
                    _vae_features(
                        vae,
                        len(repeated),
                        aug_seed + 2,
                        seconds,
                        condition_labels=repeated,
                    )
                    if vae is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
                edge_joint = (
                    _forward_prob_joint_features(
                        forward,
                        train_rows,
                        repeated,
                        edge_initial,
                        seconds,
                        rng,
                        components=10,
                    )
                    if forward is not None
                    else (np.empty((0, 0)), np.empty((0, 3)))
                )
            else:
                edge_vae = (np.empty((0, 0)), np.empty((0, 3)))
                edge_joint = (np.empty((0, 0)), np.empty((0, 3)))

            arm_arrays: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {
                "real": (x_real, y_real, 0, 0)
            }
            if "vae" in arms:
                arm_arrays["vae"] = (
                    np.concatenate((x_real, x_noise, vae_base[0], edge_vae[0]), axis=0),
                    np.concatenate((y_real, y_noise, vae_base[1], edge_vae[1]), axis=0),
                    len(vae_base[1]),
                    len(edge_vae[1]),
                )
            if "forward_prob_joint10" in arms:
                arm_arrays["forward_prob_joint10"] = (
                    np.concatenate(
                        (x_real, x_noise, joint_base[0], edge_joint[0]), axis=0
                    ),
                    np.concatenate(
                        (y_real, y_noise, joint_base[1], edge_joint[1]), axis=0
                    ),
                    len(joint_base[1]),
                    len(edge_joint[1]),
                )

            for arm in arms:
                x_train, y_train, n_syn, n_edge = arm_arrays[arm]
                for config_index, power in enumerate(configs):
                    prediction, epochs_used = _train(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        y_real,
                        seed,
                        device,
                        epochs,
                        power,
                        scaler_stats=scaler_stats,
                    )
                    pred_clip = _clip(prediction, y_dev)
                    metrics = _metric_values(pred_clip, y_val)
                    fold_rows.append(
                        {
                            "window": seconds,
                            "fold": fold,
                            "seed": seed,
                            "config": power,
                            "arm": arm,
                            "n_train": len(x_train),
                            "n_noise": len(y_noise) if arm != "real" else 0,
                            "n_synthetic": n_syn,
                            "n_edge": n_edge,
                            "epochs": epochs_used,
                            **{
                                key: round(value, 6)
                                for key, value in metrics.items()
                            },
                        }
                    )
                    fold_predictions[(arm, seed, fold, config_index)] = prediction
                    print(
                        f"fold={fold} seed={seed} arm={arm:20s} cfg={power:9s} "
                        f"n={len(x_train):3d} ep={epochs_used:3d} "
                        f"3σ={metrics['macro_three_sigma']:.3f} "
                        f"MAPE={metrics['macro_mape']:.3f}",
                        flush=True,
                    )

    fold_seed_rows: list[dict] = []
    for arm in arms:
        for seed in seeds:
            for fold in range(folds):
                preds = [
                    fold_predictions[(arm, seed, fold, index)]
                    for index in range(len(configs))
                ]
                metrics = _metric_values(
                    _clip(np.mean(preds, axis=0), y_dev), fold_truth[fold]
                )
                fold_seed_rows.append(
                    {
                        "window": seconds,
                        "fold": fold,
                        "seed": seed,
                        "arm": arm,
                        "checkpoints": len(configs),
                        **{
                            key: round(value, 6)
                            for key, value in metrics.items()
                        },
                    }
                )

    oof_seed_rows: list[dict] = []
    seed_predictions: dict[tuple[str, int], np.ndarray] = {}
    for arm in arms:
        for seed in seeds:
            parts = []
            for fold in range(folds):
                preds = [
                    fold_predictions[(arm, seed, fold, index)]
                    for index in range(len(configs))
                ]
                parts.append(np.mean(preds, axis=0))
            prediction = _clip(np.concatenate(parts, axis=0), y_dev)
            truth = np.concatenate(truth_parts, axis=0)
            metrics = _metric_values(prediction, truth)
            oof_seed_rows.append(
                {
                    "window": seconds,
                    "seed": seed,
                    "arm": arm,
                    "configs": ";".join(configs),
                    "n": len(truth),
                    **{
                        key: round(value, 6)
                        for key, value in metrics.items()
                    },
                }
            )
            seed_predictions[(arm, seed)] = prediction

    oof_rows: list[dict] = []
    seed_mean_predictions: dict[str, np.ndarray] = {}
    for arm in arms:
        parts = []
        for fold in range(folds):
            preds = [
                fold_predictions[(arm, seed, fold, index)]
                for seed in seeds
                for index in range(len(configs))
            ]
            parts.append(np.mean(preds, axis=0))
        prediction = _clip(np.concatenate(parts, axis=0), y_dev)
        truth = np.concatenate(truth_parts, axis=0)
        metrics = _metric_values(prediction, truth)
        oof_rows.append(
            {
                "window": seconds,
                "arm": arm,
                "checkpoints": len(seeds) * len(configs),
                "n": len(truth),
                **{key: round(value, 6) for key, value in metrics.items()},
            }
        )
        seed_mean_predictions[arm] = prediction

    summary_rows: list[dict] = []
    for arm in arms:
        summary: dict[str, float | int | str] = {
            "window": seconds,
            "arm": arm,
            "seeds": len(seeds),
            "configs": len(configs),
        }
        pooled = next(row for row in oof_rows if row["arm"] == arm)
        summary["seed_mean_macro_three_sigma"] = pooled["macro_three_sigma"]
        summary["seed_mean_macro_mape"] = pooled["macro_mape"]
        summary["seed_mean_macro_median"] = pooled["macro_median"]
        summary["seed_mean_macro_p90"] = pooled["macro_p90"]
        summary_rows.append(summary)

    _write_csv(out / "folds.csv", fold_rows)
    _write_csv(out / "fold_seed.csv", fold_seed_rows)
    _write_csv(out / "oof_seed.csv", oof_seed_rows)
    _write_csv(out / "oof.csv", oof_rows)
    _write_csv(out / "summary.csv", summary_rows)

    truth = np.concatenate(truth_parts, axis=0).astype(np.float32)
    payload = {
        "truth": truth,
        "file_id": np.asarray(file_ids, dtype=np.int32),
    }
    for (arm, seed), prediction in seed_predictions.items():
        payload[f"{arm}_s{seed}"] = prediction.astype(np.float32)
    for arm, prediction in seed_mean_predictions.items():
        payload[f"{arm}_seed_mean"] = prediction.astype(np.float32)
    np.savez_compressed(out / "predictions.npz", **payload)

    (out / "summary.md").write_text(
        _summary_markdown(summary_rows), encoding="utf-8"
    )
    noise_count = next(
        (row["n_noise"] for row in fold_rows if row["arm"] != "real"), 0
    )
    config = {
        "folds": folds,
        "seeds": list(seeds),
        "window": seconds,
        "epochs": epochs,
        "arms": list(arms),
        "low_conc_configs": list(configs),
        "synthetic_count_per_arm": synthetic_count,
        "synthetic_ratio": SYNTHETIC_FACTOR,
        "noise_scale": NOISE_SCALE,
        "noise_count_per_augmented_arm": noise_count,
        "edge_copies": EDGE_COPIES,
        "real_scaler": real_scaler,
        "clip_to_development_range": True,
        "elapsed_seconds": round(time.time() - started, 1),
        "mode": MODE,
        "device": str(device),
    }
    (out / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Wrote {out} in {config['elapsed_seconds']} s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument("--seconds", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_CONFIGS),
        help='低浓度配置，例如 --configs 1.0 "0,0.3,1.0"',
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        choices=ARMS,
        default=list(ARMS),
    )
    parser.add_argument("--output", default=str(BASE))
    parser.add_argument("--real-scaler", action="store_true")
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(args.seeds),
        seconds=args.seconds,
        epochs=args.epochs,
        configs=tuple(args.configs),
        arms=tuple(args.arms),
        output=args.output,
        real_scaler=args.real_scaler,
    )


if __name__ == "__main__":
    main()
