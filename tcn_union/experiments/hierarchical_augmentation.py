"""分层随机效应正模型与 PCA-VAE 的联合增强评估。

协议与 ``full_protocol_augmentation`` 保持一致：10 s 前缀、50 条开发集
5 折、两套低浓度配置、3 seed、6 checkpoint；锁定 12 条不参与。

本脚本只增加两个臂：

* ``hier``：分层随机效应正模型（方案 1）；
* ``hier+vae``：60 条分层正模型 + 60 条 PCA-VAE，边缘复制各半。

同时保留 ``real`` / ``vae`` / ``joint10`` 作为同轮对照，避免跨脚本比较。

训练协议由 ``protocol`` 控制：

* ``syn``：real + synthetic，无噪声、无边缘复制；
* ``noise_syn``：real + noise + synthetic；
* ``full``：real + noise + synthetic + edge（既有完整协议）；
* ``gas``：real + noise + H2 扩充 + H2-only edge；NH3/NO2 的标签只来自
  训练折已有标签，不做边缘复制。
"""
from __future__ import annotations

import argparse
import json
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
    _initial_for_conditional,
    _initial_for_labels,
    _metric_values,
    _real_features,
    _synthetic_concentration_excluding,
    _train,
    _training_concentrations,
    _vae_features,
    _write_csv,
)

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/hierarchical"
EDGE_COPIES = 4
PROTOCOLS = ("syn", "noise_syn", "full", "gas")


def _hier_features(hier, forward, labels, initial, seconds, rng):
    response = hier.sample(forward, labels, initial, seconds, rng)
    times = np.arange(round(seconds / 0.2) + 1) * 0.2
    from tcn_union.core.data import make_features
    from sensor_array.core.data import GASES  # noqa: F401  (确保包可导入)

    features = np.stack(
        [make_features(times, item, seconds, MODE) for item in response]
    )
    return features.astype(np.float32), np.asarray(labels, dtype=np.float32)


def _gas_targets(
    labels: np.ndarray,
    val_labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """H2 沿用完整协议的基础标签分布；NH3/NO2 只用训练折已有标签。"""
    target = _training_concentrations(labels, count, rng)
    h2_source = _synthetic_concentration_excluding(labels, val_labels, count, rng)
    target[:, 0] = h2_source[:, 0]
    return target


def _gas_edge_labels(
    train_rows: list[dict],
    rng: np.random.Generator,
) -> np.ndarray:
    """H2 取最小/最大档位，NH3/NO2 从训练折已有标签中抽样。"""
    labels = np.stack(
        [np.asarray(row["_labels"], dtype=np.float64) for row in train_rows]
    )
    count = len(_edge_case_labels(train_rows))
    if count == 0:
        return np.empty((0, 3), dtype=np.float64)
    half = max(count // 2, 1)
    lower, upper = labels.min(axis=0), labels.max(axis=0)
    parts = []
    for value in (lower[0], upper[0]):
        index = rng.integers(0, len(labels), size=half * 2)
        part = np.column_stack(
            (np.full(len(index), value), labels[index, 1], labels[index, 2])
        )
        parts.append(np.unique(part, axis=0)[:half])
    return np.concatenate(parts, axis=0)


def _build_synthetic(
    arm,
    forward,
    vae,
    hier,
    train_rows,
    labels,
    val_labels,
    protocol,
    seconds,
    seed,
    rng,
):
    count = SYNTHETIC_FACTOR * 40
    gas = protocol == "gas"
    if arm == "real":
        return None
    if arm == "vae":
        if gas:
            target = _gas_targets(labels, val_labels, count, rng)
            return _vae_features(
                vae, count, seed + 1, seconds, condition_labels=target
            )
        return _vae_features(vae, count, seed + 1, seconds)
    if arm == "joint10":
        if gas:
            raise ValueError("joint10 不支持 gas 协议")
        target = _training_concentrations(labels, count, rng)
        initial = _initial_for_labels(train_rows, target)
        return _forward_prob_joint_features(
            forward, train_rows, target, initial, seconds, rng, components=10
        )
    if arm in {"hier", "hier+vae"}:
        if gas:
            target = _gas_targets(labels, val_labels, count, rng)
            initial = _initial_for_conditional(train_rows, target)
        else:
            target = _training_concentrations(labels, count, rng)
            initial = _initial_for_labels(train_rows, target)
        hier_data = _hier_features(hier, forward, target, initial, seconds, rng)
        if arm == "hier":
            return hier_data
        half = count // 2
        vae_data = (
            _vae_features(
                vae,
                half,
                seed + 1,
                seconds,
                condition_labels=target[:half],
            )
            if gas
            else _vae_features(vae, half, seed + 1, seconds)
        )
        return (
            np.concatenate((hier_data[0][:half], vae_data[0]), axis=0),
            np.concatenate((hier_data[1][:half], vae_data[1]), axis=0),
        )
    raise ValueError(arm)


def _h2_weights(count: int) -> np.ndarray:
    return np.tile(np.array([1.0, 0.0, 0.0]), (count, 1))


UNION_COUNTS = {
    "union_h2": (120, 120),
    "union_h2_180": (120, 180),
}


def _h2_union(
    forward,
    vae,
    hier,
    train_rows,
    labels,
    seconds,
    seed,
    rng,
    n_vae: int,
    n_hier: int,
):
    """VAE 全气体样本 + 分层正模型只监督 H2 的额外样本。"""
    count = max(SYNTHETIC_FACTOR * 40, n_hier)
    target = _training_concentrations(labels, count, rng)
    initial = _initial_for_labels(train_rows, target)
    vae_data = _vae_features(vae, n_vae, seed + 1, seconds)
    hier_data = _hier_features(hier, forward, target[:n_hier], initial[:n_hier], seconds, rng)
    return (
        np.concatenate((vae_data[0], hier_data[0]), axis=0),
        np.concatenate((vae_data[1], hier_data[1]), axis=0),
        np.concatenate((np.ones((n_vae, 3)), _h2_weights(n_hier)), axis=0),
    )


def _h2_union_edge(
    forward,
    vae,
    hier,
    train_rows,
    seconds,
    rng,
    aug_seed,
    n_hier: int,
):
    edge_labels = _edge_case_labels(train_rows)
    if edge_labels.size == 0:
        return None
    repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
    n_hier = min(n_hier, len(repeated))
    initial = _initial_for_labels(train_rows, repeated)
    vae_data = _vae_features(
        vae, len(repeated), aug_seed + 2, seconds, condition_labels=repeated
    )
    hier_data = _hier_features(
        hier, forward, repeated[:n_hier], initial[:n_hier], seconds, rng
    )
    return (
        np.concatenate((vae_data[0], hier_data[0]), axis=0),
        np.concatenate((vae_data[1], hier_data[1]), axis=0),
        np.concatenate((np.ones((len(repeated), 3)), _h2_weights(n_hier)), axis=0),
    )


def _build_edge(
    arm,
    forward,
    vae,
    hier,
    train_rows,
    protocol,
    seconds,
    rng,
    aug_seed,
):
    if protocol not in {"full", "gas"}:
        return None
    if protocol == "gas":
        edge_labels = _gas_edge_labels(train_rows, rng)
    else:
        edge_labels = _edge_case_labels(train_rows)
    if edge_labels.size == 0:
        return None
    repeated = np.repeat(edge_labels, EDGE_COPIES, axis=0)
    initial = (
        _initial_for_conditional(train_rows, repeated)
        if protocol == "gas"
        else _initial_for_labels(train_rows, repeated)
    )
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
        if protocol == "gas":
            raise ValueError("joint10 不支持 gas 协议")
        return _forward_prob_joint_features(
            forward, train_rows, repeated, initial, seconds, rng, components=10
        )
    return None


def run(
    folds: int = 5,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    seconds: int = 10,
    configs: tuple[str, ...] = DEFAULT_CONFIGS,
    arms: tuple[str, ...] = ("real", "vae", "hier", "hier+vae"),
    protocol: str = "full",
    output: Path | str = BASE / "pilot",
) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"protocol 必须是 {PROTOCOLS} 之一")
    if protocol == "gas" and "joint10" in arms:
        raise ValueError("gas 协议不支持 joint10")
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
        val_labels = np.stack([row["_labels"] for row in val_rows]).astype(float)

        use_noise = protocol != "syn"
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
            if use_noise:
                x_noise, y_noise = noisy_features(
                    train_rows,
                    seconds,
                    MODE,
                    NoiseConfig(scale=NOISE_SCALE, copies=1),
                    aug_seed + 17,
                )
            else:
                x_noise = np.empty((0, 0))
                y_noise = np.empty((0, 3))
            for arm in arms:
                union_counts = UNION_COUNTS.get(arm)
                if union_counts is not None:
                    n_vae, n_hier = union_counts
                    synthetic = _h2_union(
                        forward,
                        vae,
                        hier,
                        train_rows,
                        labels,
                        seconds,
                        aug_seed,
                        rng,
                        n_vae,
                        n_hier,
                    )
                    edge = _h2_union_edge(
                        forward,
                        vae,
                        hier,
                        train_rows,
                        seconds,
                        rng,
                        aug_seed,
                        min(n_hier, 120),
                    )
                else:
                    synthetic = _build_synthetic(
                        arm,
                        forward,
                        vae,
                        hier,
                        train_rows,
                        labels,
                        val_labels,
                        protocol,
                        seconds,
                        aug_seed,
                        rng,
                    )
                    edge = _build_edge(
                        arm,
                        forward,
                        vae,
                        hier,
                        train_rows,
                        protocol,
                        seconds,
                        rng,
                        aug_seed,
                    )
                for config in configs:
                    if synthetic is None:
                        x_train, y_train = x_real, y_real
                        sample_weight = None
                    else:
                        parts = [synthetic]
                        if edge is not None:
                            parts.append(edge)
                        clean = [x_real]
                        clean_labels = [y_real]
                        if use_noise:
                            clean.append(x_noise)
                            clean_labels.append(y_noise)
                        x_train = np.concatenate(
                            clean + [part[0] for part in parts], axis=0
                        )
                        y_train = np.concatenate(
                            clean_labels + [part[1] for part in parts], axis=0
                        )
                        sample_weight = (
                            np.concatenate(
                                [np.ones((len(y_real), 3))]
                                + ([] if not use_noise else [np.ones((len(y_noise), 3))])
                                + [part[2] for part in parts],
                                axis=0,
                            )
                            if union_counts is not None
                            else None
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
                        sample_weight=sample_weight,
                    )
                    prediction = _clip(prediction, y_val)
                    metrics = _metric_values(prediction, y_val)
                    fold_rows.append(
                        {
                            "fold": fold,
                            "seed": seed,
                            "config": config,
                            "protocol": protocol,
                            "arm": arm,
                            "n_train": len(x_train),
                            "n_noise": len(y_noise) if synthetic is not None else 0,
                            "n_synthetic": 0 if synthetic is None else len(synthetic[1]),
                            "n_edge": 0 if edge is None else len(edge[1]),
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
                "protocol": protocol,
                "arm": arm,
                "seed": seed,
                "config": config,
                **_metric_values(prediction, truth),
            }
        )

    # 与 AGENTS §10 的 final_report 对齐：每 seed 先平均配置，再输出
    # 6 checkpoint 集成，方便后续统一画图与复算。
    for arm in arms:
        for seed in seeds:
            parts = [
                np.mean(
                    [
                        predictions[(arm, seed, config)][fold]
                        for config in configs
                    ],
                    axis=0,
                )
                for fold in range(folds)
            ]
            arrays[f"{arm}_s{seed}"] = _clip(
                np.concatenate(parts, axis=0).astype(np.float32), truth
            )
        parts = [
            np.mean(
                [
                    predictions[(arm, seed, config)][fold]
                    for seed in seeds
                    for config in configs
                ],
                axis=0,
            )
            for fold in range(folds)
        ]
        arrays[f"{arm}_ensemble"] = _clip(
            np.concatenate(parts, axis=0).astype(np.float32), truth
        )

    _write_csv(out / "oof.csv", oof_rows)
    arrays["truth"] = truth.astype(np.float32)
    arrays["file_id"] = np.asarray(file_ids, dtype=np.int32)
    np.savez_compressed(out / "predictions.npz", **arrays)

    summary = []
    for arm in arms:
        subset = [row for row in oof_rows if row["arm"] == arm]
        if not subset:
            continue
        for key in ("macro_three_sigma", "macro_mape"):
            values = np.array([row[key] for row in subset])
            summary.append(
                {
                    "protocol": protocol,
                    "arm": arm,
                    "metric": key,
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "n": len(values),
                }
            )
    _write_csv(out / "summary.csv", summary)

    config = {
        "protocol": protocol,
        "folds": folds,
        "seeds": list(seeds),
        "configs": list(configs),
        "arms": list(arms),
        "n_train": sorted({row["n_train"] for row in fold_rows}),
        "n_noise": sorted({row["n_noise"] for row in fold_rows}),
        "n_synthetic": sorted({row["n_synthetic"] for row in fold_rows}),
        "n_edge": sorted({row["n_edge"] for row in fold_rows}),
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (out / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out} ({time.time() - started:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_CONFIGS))
    parser.add_argument(
        "--arms", nargs="+", default=["real", "vae", "hier", "hier+vae"]
    )
    parser.add_argument(
        "--protocol", choices=PROTOCOLS, default="full", help="训练协议"
    )
    parser.add_argument("--output", type=Path, default=BASE / "pilot")
    args = parser.parse_args()
    run(
        folds=args.folds,
        seeds=tuple(args.seeds),
        configs=tuple(args.configs),
        arms=tuple(args.arms),
        protocol=args.protocol,
        output=args.output,
    )


if __name__ == "__main__":
    main()
