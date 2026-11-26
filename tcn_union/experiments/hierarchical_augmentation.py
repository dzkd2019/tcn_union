"""分层随机效应正模型与 PCA-VAE 的联合增强评估。

协议与 ``full_protocol_augmentation`` 保持一致：10 s 前缀、50 条开发集
5 折、两套低浓度配置、3 seed、6 checkpoint；锁定 12 条不参与。

本脚本只增加两个臂：

* ``hier``：分层随机效应正模型（方案 1）；
* ``hier+vae``：60 条分层正模型 + 60 条 PCA-VAE，边缘复制各半。

同时保留 ``real`` / ``vae`` / ``joint10`` 作为同轮对照，避免跨脚本比较。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from tcn_union.augmentation.noise import NoiseConfig, noisy_features
from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.training.gan_train import MODE, _edge_case_labels, _read_manifest

from sensor_array.core.hierarchical import HierarchicalRandomEffectsModel

from tcn_union.experiments.full_protocol_augmentation import (
    DEFAULT_CONFIGS,
    DEFAULT_SEEDS,
    EPOCHS,
    NOISE_SCALE,
    SYNTHETIC_FACTOR,
    _clip,
    _fit_generators,
    _forward_prob_joint_features,
    _initial_for_labels,
    _metric_values,
    _real_features,
    _train,
    _training_concentrations,
    _vae_features,
    _write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/hierarchical"
EDGE_COPIES = 4


def _hier_features(hier, forward, labels, initial, seconds, rng):
    response = hier.sample(forward, labels, initial, seconds, rng)
    times = np.arange(round(seconds / 0.2) + 1) * 0.2
    from tcn_union.core.data import make_features
    from sensor_array.core.data import GASES  # noqa: F401  (确保包可导入)

    features = np.stack(
        [make_features(times, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), np.asarray(labels, dtype=np.float32)


def _build_synthetic(
    arm,
    forward,
    vae,
    hier,
    train_rows,
    labels,
    seconds,
    seed,
    rng,
):
    count = SYNTHETIC_FACTOR * 40
    if arm == "real":
        return None
    if arm == "vae":
        return _vae_features(vae, count, seed + 1, seconds)
    if arm == "joint10":
        target = _training_concentrations(labels, count, rng)
        initial = _initial_for_labels(train_rows, target)
        return _forward_prob_joint_features(
            forward, train_rows, target, initial, seconds, rng, components=10
        )
    if arm in {"hier", "hier+vae"}:
        target = _training_concentrations(labels, count, rng)
        initial = _initial_for_labels(train_rows, target)
        hier_data = _hier_features(hier, forward, target, initial, seconds, rng)
        if arm == "hier":
            return hier_data
        half = count // 2
        vae_data = _vae_features(vae, half, seed + 1, seconds)
        return (
            np.concatenate((hier_data[0][:half], vae_data[0]), axis=0),
            np.concatenate((hier_data[1][:half], vae_data[1]), axis=0),
        )
    raise ValueError(arm)


def _build_edge(arm, forward, vae, hier, train_rows, seconds, rng, aug_seed):
    edge_labels = _edge_case_labels(train_rows)
    if edge_labels.size == 0:
        return None
    repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
    initial = _initial_for_labels(train_rows, repeated)
    if arm in {"hier", "hier+vae"}:
        hier_data = _hier_features(hier, forward, repeated, initial, seconds, rng)
        if arm == "hier":
            return hier_data
        half = len(repeated) // 2
        vae_data = _vae_features(
            vae, half, aug_seed + 2, seconds, condition_labels=repeated[:half]
        )
        return (
            np.concatenate((hier_data[0][:half], vae_data[0]), axis=0),
            np.concatenate((hier_data[1][:half], vae_data[1]), axis=0),
        )
    if arm == "vae":
        return _vae_features(
            vae, len(repeated), aug_seed + 2, seconds, condition_labels=repeated
        )
    if arm == "joint10":
        return _forward_prob_joint_features(
            forward, train_rows, repeated, initial, seconds, rng, components=10
        )
    return None


def run(
    folds: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    seconds: int = 10,
    configs: tuple[str, ...] = DEFAULT_CONFIGS,
    arms: tuple[str, ...] = ("real", "vae", "hier", "hier+vae", "joint10"),
    output: Path | str = BASE / "pilot",
) -> None:
    out = Path(output)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    from sensor_array.core.data import load_experiments

    experiments = {
        experiment.file_id: experiment for experiment in load_experiments()
    }
    fold_rows: list[dict] = []
    predictions: dict[tuple[str, int, str], dict[int, np.ndarray]] = {}
    truths: dict[int, np.ndarray] = {}
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
        truths[fold] = y_val
        file_ids.extend(int(Path(row["file"]).stem) for row in val_rows)
        labels = np.stack([row["_labels"] for row in train_rows]).astype(float)

        for seed in seeds:
            aug_seed = seed + 1000 * fold
            forward, vae = _fit_generators(
                train_rows,
                train_experiments,
                aug_seed,
                device,
                True,
                "vae" in arms or "hier+vae" in arms,
            )
            hier = HierarchicalRandomEffectsModel().fit(
                forward, train_experiments, seconds
            )
            rng = np.random.default_rng(aug_seed)
            x_noise, y_noise = noisy_features(
                train_rows,
                seconds,
                MODE,
                NoiseConfig(scale=NOISE_SCALE, copies=1),
                aug_seed + 17,
            )
            for arm in arms:
                synthetic = _build_synthetic(
                    arm, forward, vae, hier, train_rows, labels, seconds, aug_seed, rng
                )
                edge = _build_edge(
                    arm, forward, vae, hier, train_rows, seconds, rng, aug_seed
                )
                for config in configs:
                    if synthetic is None:
                        x_train, y_train = x_real, y_real
                    else:
                        parts = [synthetic]
                        if edge is not None:
                            parts.append(edge)
                        x_train = np.concatenate(
                            [x_real, x_noise] + [part[0] for part in parts], axis=0
                        )
                        y_train = np.concatenate(
                            [y_real, y_noise] + [part[1] for part in parts], axis=0
                        )
                    prediction, epochs_used = _train(
                        x_train,
                        y_train,
                        x_val,
                        y_val,
                        y_real,
                        seed,
                        device,
                        EPOCHS,
                        config,
                    )
                    prediction = _clip(prediction, y_val)
                    metrics = _metric_values(prediction, y_val)
                    fold_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "config": config,
                            "arm": arm,
                            "epochs": epochs_used,
                            **metrics,
                        }
                    )
                    predictions.setdefault((arm, seed, config), {})[fold] = prediction

        print(f"fold {fold} done ({time.time() - started:.0f}s)", flush=True)

    _write_csv(out / "folds.csv", fold_rows)

    oof_rows = []
    arrays = {}
    truth = np.concatenate([truths[fold] for fold in range(folds)], axis=0)
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
    arrays["truth"] = truth
    arrays["file_ids"] = np.asarray(file_ids)
    np.savez(out / "predictions.npz", **arrays)

    summary = []
    for arm in arms:
        subset = [row for row in oof_rows if row["arm"] == arm]
        if not subset:
            continue
        for key in ("macro_three_sigma", "macro_mape"):
            values = np.array([row[key] for row in subset])
            summary.append(
                {
                    "arm": arm,
                    "metric": key,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    _write_csv(out / "summary.csv", summary)
    print(f"wrote {out} ({time.time() - started:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument("--arms", nargs="+", default=["real", "vae", "hier", "hier+vae", "joint10"])
    parser.add_argument("--output", type=Path, default=BASE / "pilot")
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(args.seeds),
        configs=tuple(args.configs),
        arms=tuple(args.arms),
        output=args.output,
    )


if __name__ == "__main__":
    main()
