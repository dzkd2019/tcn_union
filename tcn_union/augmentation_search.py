"""仅用于开发集的训练期增强方案搜索。

本命令行工具与锁定评估故意分离：它只在既有开发集五折上运行一个或多个具名
配方，写出 OOF 预测和逐折分数。这里永不读取锁定的 12 个测试实验；配方必须
先由这些 OOF 结果选定并冻结，之后才能运行最终测试评估。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .augment import AugmentConfig, make_context
from .train import (
    BASE_SEED,
    MAX_EPOCHS,
    _features,
    _macro_nmae,
    _read_manifest,
    _train_network,
)

AXES = ("H2", "NH3", "NO2")


def _configs() -> dict[str, AugmentConfig]:
    """开发集搜索使用的具名配方。"""
    return {
        "clean": AugmentConfig("clean", probability=0.0),
        "noise_0.001": AugmentConfig("noise_0.001", noise=0.001),
        "noise_0.003": AugmentConfig("noise_0.003", noise=0.003),
        "noise_0.006": AugmentConfig("noise_0.006", noise=0.006),
        "transient_0.03": AugmentConfig("transient_0.03", transient=0.03),
        "transient_0.06": AugmentConfig("transient_0.06", transient=0.06),
        "transient_0.10": AugmentConfig("transient_0.10", transient=0.10),
        "transient_nh3_0.03": AugmentConfig(
            "transient_nh3_0.03", transient=0.03, transient_weights=(0.0, 1.0, 0.0)
        ),
        "transient_nh3_0.06": AugmentConfig(
            "transient_nh3_0.06", transient=0.06, transient_weights=(0.0, 1.0, 0.0)
        ),
        "transient_no2_0.03": AugmentConfig(
            "transient_no2_0.03", transient=0.03, transient_weights=(0.0, 0.0, 1.0)
        ),
        "transient_h2_0.03": AugmentConfig(
            "transient_h2_0.03", transient=0.03, transient_weights=(1.0, 0.0, 0.0)
        ),
        "cutout_0.5_1s": AugmentConfig(
            "cutout_0.5_1s", cutout_probability=0.5, cutout_max_seconds=1.0
        ),
        "cutout_1.0_2s": AugmentConfig(
            "cutout_1.0_2s", cutout_probability=1.0, cutout_max_seconds=2.0
        ),
        "warp20_mixup02": AugmentConfig(
            "warp20_mixup02", warp=0.20, mixup_alpha=0.2, mixup_probability=0.5
        ),
        "warp20_transient003": AugmentConfig(
            "warp20_transient003", warp=0.20, transient=0.03
        ),
        "warp20_mixup02_transient003": AugmentConfig(
            "warp20_mixup02_transient003",
            warp=0.20,
            mixup_alpha=0.2,
            mixup_probability=0.5,
            transient=0.03,
        ),
        "warp_transient_nh3": AugmentConfig(
            "warp_transient_nh3", warp=0.10, transient=0.03, transient_weights=(0.0, 1.0, 0.0)
        ),
        "warp_prefix": AugmentConfig(
            "warp_prefix", warp=0.10, prefix_probability=0.5
        ),
        "gain_0.01": AugmentConfig("gain_0.01", gain=0.01),
        "gain_0.03": AugmentConfig("gain_0.03", gain=0.03),
        "warp_0.05": AugmentConfig("warp_0.05", warp=0.05),
        "warp_0.10": AugmentConfig("warp_0.10", warp=0.10),
        "warp_0.20": AugmentConfig("warp_0.20", warp=0.20),
        "offset_0.002": AugmentConfig("offset_0.002", offset=0.002),
        "offset_0.005": AugmentConfig("offset_0.005", offset=0.005),
        "mixup_0.2": AugmentConfig("mixup_0.2", mixup_alpha=0.2, mixup_probability=0.5),
        "mixup_0.4": AugmentConfig("mixup_0.4", mixup_alpha=0.4, mixup_probability=0.5),
        "prefix_0.5": AugmentConfig("prefix_0.5", prefix_probability=0.5),
        "prefix_lengths_5to10": AugmentConfig(
            "prefix_lengths_5to10", prefix_lengths=(5.0, 6.0, 7.0, 8.0, 9.0, 10.0)
        ),
        "prefix_lengths_7to10": AugmentConfig(
            "prefix_lengths_7to10", prefix_lengths=(7.0, 8.0, 9.0, 10.0)
        ),
        "prefix_lengths_jitter": AugmentConfig(
            "prefix_lengths_jitter",
            prefix_lengths=(5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
            noise=0.001,
            transient=0.05,
            warp=0.10,
        ),
        "noise_transient_warp": AugmentConfig(
            "noise_transient_warp", noise=0.003, transient=0.06, warp=0.10
        ),
        "noise_transient_warp_mixup": AugmentConfig(
            "noise_transient_warp_mixup",
            noise=0.003,
            transient=0.06,
            warp=0.10,
            mixup_alpha=0.2,
            mixup_probability=0.5,
        ),
        "noise_transient_warp_prefix": AugmentConfig(
            "noise_transient_warp_prefix",
            noise=0.003,
            transient=0.06,
            warp=0.10,
            prefix_probability=0.5,
        ),
    }


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    macro = float(np.mean([r["macro_nmae"] for r in records]))
    by_gas = {
        gas: float(np.mean([r[f"{gas}_nmae"] for r in records])) for gas in AXES
    }
    mape = {
        gas: float(np.mean([r[f"{gas}_mape"] for r in records])) for gas in AXES
    }
    return {
        "macro_nmae": macro,
        "by_gas_nmae": by_gas,
        "by_gas_mape_percent": mape,
        "best_epochs": [int(r["best_epoch"]) for r in records],
    }


def run_recipe(
    rows: list[dict[str, Any]],
    seconds: float,
    mode: str,
    config: AugmentConfig,
    seed: int,
    device: torch.device,
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """对单个配方运行五折，返回（逐折分数，OOF 预测）。"""
    dev_rows = [row for row in rows if row["split"] == "development"]
    features = _features(dev_rows, seconds, mode)
    fold_scores: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    clean = config.probability == 0.0 and not config.prefix_lengths and all(
        getattr(config, key) == 0.0
        for key in (
            "noise",
            "gain",
            "offset",
            "warp",
            "transient",
            "mixup_alpha",
            "mixup_probability",
            "prefix_probability",
            "cutout_probability",
        )
    )
    for fold in range(5):
        train_rows = [row for row in dev_rows if int(row["fold"]) != fold]
        validation_rows = [row for row in dev_rows if int(row["fold"]) == fold]
        x_train = np.stack([features[row["file"]] for row in train_rows])
        y_train = np.stack([row["_labels"] for row in train_rows]).astype(np.float32)
        x_val = np.stack([features[row["file"]] for row in validation_rows])
        y_val = np.stack([row["_labels"] for row in validation_rows]).astype(np.float32)
        context = None
        if not clean:
            context = make_context(train_rows, seconds, mode, config, BASE_SEED)
        _, _, best_epoch, best_score, prediction = _train_network(
            x_train,
            y_train,
            x_val,
            y_val,
            y_train,
            MAX_EPOCHS,
            seed,
            device,
            True,
            augment=context,
            aux_positions=aux_positions,
            aux_weight=aux_weight,
        )
        if prediction is None or best_score is None:
            raise RuntimeError("missing validation prediction")
        _, axis_scores = _macro_nmae(y_val, prediction, y_train)
        mape = 100.0 * np.mean(np.abs(prediction - y_val) / np.maximum(np.abs(y_val), 1e-8), axis=0)
        fold_scores.append(
            {
                "fold": fold,
                "macro_nmae": best_score,
                "H2_nmae": axis_scores[0],
                "NH3_nmae": axis_scores[1],
                "NO2_nmae": axis_scores[2],
                "H2_mape": float(mape[0]),
                "NH3_mape": float(mape[1]),
                "NO2_mape": float(mape[2]),
                "best_epoch": best_epoch,
            }
        )
        for row, truth, pred in zip(validation_rows, y_val, prediction, strict=True):
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


def _global_summary(oof: list[dict[str, Any]], dev_ranges: np.ndarray) -> dict[str, Any]:
    truth = np.array([[row[f"{gas}_true"] for gas in AXES] for row in oof])
    pred = np.array([[row[f"{gas}_pred"] for gas in AXES] for row in oof])
    errors = np.abs(pred - truth)
    mae = errors.mean(axis=0)
    mape = 100.0 * (errors / np.maximum(np.abs(truth), 1e-8)).mean(axis=0)
    nmae = mae / dev_ranges
    return {
        "n": len(oof),
        "by_gas_mae": dict(zip(AXES, mae.tolist(), strict=True)),
        "by_gas_nmae": dict(zip(AXES, nmae.tolist(), strict=True)),
        "by_gas_mape_percent": dict(zip(AXES, mape.tolist(), strict=True)),
        "macro_nmae": float(nmae.mean()),
        "macro_mape_percent": float(mape.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search training-time augmentation recipes")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--seconds", type=float, nargs="+", default=[5.0, 10.0])
    parser.add_argument("--recipes", nargs="+", default=None)
    parser.add_argument("--mode", default="raw_delta")
    parser.add_argument("--aux-positions", type=int, nargs="*", default=[])
    parser.add_argument("--aux-weight", type=float, default=0.5)
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    dev_ranges = np.ptp(np.stack([row["_labels"] for row in dev_rows]), axis=0)
    configs = _configs()
    selected = args.recipes or list(configs)
    unknown = [name for name in selected if name not in configs]
    if unknown:
        raise ValueError(f"unknown recipes: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    for seconds in args.seconds:
        for name in selected:
            config = configs[name]
            print(f"[search] {seconds:g}s {name} seed={args.seed}", flush=True)
            fold_scores, oof = run_recipe(
                rows,
                float(seconds),
                args.mode,
                config,
                args.seed,
                device,
                aux_positions=tuple(args.aux_positions),
                aux_weight=args.aux_weight,
            )
            key = f"{seconds:g}s/{name}"
            result = {
                "seconds": seconds,
                "recipe": name,
                "config": config.to_dict(),
                "seed": args.seed,
                "aux_positions": list(args.aux_positions),
                "aux_weight": args.aux_weight,
                "folds": fold_scores,
                "fold_summary": _summary(fold_scores),
                "global_oof": _global_summary(oof, dev_ranges),
            }
            all_results[key] = result
            recipe_dir = args.output_dir / f"{seconds:g}s" / name
            recipe_dir.mkdir(parents=True, exist_ok=True)
            (recipe_dir / "result.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            with (recipe_dir / "oof.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
                writer.writeheader()
                writer.writerows(oof)
            summary = result["global_oof"]
            print(
                f"  macro-NMAE={summary['macro_nmae']:.6f} "
                f"gas={summary['by_gas_nmae']} MAPE={summary['by_gas_mape_percent']} "
                f"epochs={result['fold_summary']['best_epochs']}",
                flush=True,
            )
            (args.output_dir / "search_results.json").write_text(
                json.dumps(all_results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )


if __name__ == "__main__":
    main()
