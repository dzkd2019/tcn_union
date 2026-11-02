"""Train and freeze augmented 5 s / 10 s TCNs on the locked development split.

The locked 12-run test set is never read by this module.  Recipes are selected
from the development OOF search (``tcn_union.augmentation_search``), then the
same five-fold protocol is rerun here to pick the refit epoch count and to
train the three fixed final seeds on all 50 development experiments.
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

from .augment import AugmentConfig, make_context
from .augmentation_search import _configs, _global_summary, _summary
from .train import (
    BASE_SEED,
    MAX_EPOCHS,
    SEEDS,
    _features,
    _macro_nmae,
    _read_manifest,
    _save_checkpoint,
    _train_network,
)

AXES = ("H2", "NH3", "NO2")


def _is_clean(config: AugmentConfig) -> bool:
    return config.probability == 0.0 and not config.prefix_lengths and all(
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


def cross_validate(
    dev_rows: list[dict[str, Any]],
    seconds: float,
    mode: str,
    config: AugmentConfig,
    seed: int,
    device: torch.device,
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Five-fold OOF for one frozen recipe."""
    features = _features(dev_rows, seconds, mode)
    fold_scores: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    for fold in range(5):
        train_rows = [row for row in dev_rows if int(row["fold"]) != fold]
        validation_rows = [row for row in dev_rows if int(row["fold"]) == fold]
        x_train = np.stack([features[row["file"]] for row in train_rows])
        y_train = np.stack([row["_labels"] for row in train_rows]).astype(np.float32)
        x_val = np.stack([features[row["file"]] for row in validation_rows])
        y_val = np.stack([row["_labels"] for row in validation_rows]).astype(np.float32)
        context = None if _is_clean(config) else make_context(
            train_rows, seconds, mode, config, seed
        )
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
        mape = 100.0 * np.mean(
            np.abs(prediction - y_val) / np.maximum(np.abs(y_val), 1e-8), axis=0
        )
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


def train_final(
    dev_rows: list[dict[str, Any]],
    seconds: float,
    mode: str,
    config: AugmentConfig,
    epochs: int,
    seed: int,
    device: torch.device,
    output_path: Path,
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
) -> None:
    """Refit one seed on all development runs with online augmentation."""
    features = _features(dev_rows, seconds, mode)
    x_dev = np.stack([features[row["file"]] for row in dev_rows])
    y_dev = np.stack([row["_labels"] for row in dev_rows]).astype(np.float32)
    context = None if _is_clean(config) else make_context(
        dev_rows, seconds, mode, config, seed
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
        augment=context,
        aux_positions=aux_positions,
        aux_weight=aux_weight,
    )
    metadata = None
    if not _is_clean(config) or aux_positions:
        metadata = {
            "recipe": config.to_dict(),
            "aux_positions": list(aux_positions),
            "aux_weight": aux_weight,
        }
    _save_checkpoint(
        output_path,
        model,
        stats,
        mode,
        seconds,
        dev_rows,
        seed,
        epochs,
        device,
        augmentation=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train frozen augmented TCN models")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="raw_delta")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--config-5s", default="clean")
    parser.add_argument("--config-10s", default="clean")
    parser.add_argument("--aux-positions-5s", type=int, nargs="*", default=[])
    parser.add_argument("--aux-weight-5s", type=float, default=0.5)
    parser.add_argument("--aux-positions-10s", type=int, nargs="*", default=[])
    parser.add_argument("--aux-weight-10s", type=float, default=0.5)
    parser.add_argument("--recipes-json", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    dev_ranges = np.ptp(np.stack([row["_labels"] for row in dev_rows]), axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recipes = _configs()
    if args.recipes_json is not None:
        payload = json.loads(args.recipes_json.read_text(encoding="utf-8"))
        for name, config in payload.items():
            recipes[name] = AugmentConfig.from_dict(config)
    chosen = {
        "5s": (
            recipes[args.config_5s],
            tuple(args.aux_positions_5s),
            args.aux_weight_5s,
        ),
        "10s": (
            recipes[args.config_10s],
            tuple(args.aux_positions_10s),
            args.aux_weight_10s,
        ),
    }

    output: dict[str, Any] = {
        "protocol": "five-fold development OOF; final refit on 50 development runs; test not read",
        "base_seed": args.seed,
        "final_seeds": SEEDS,
        "mode": args.mode,
        "selected": {},
    }
    for key, (config, aux_positions, aux_weight) in chosen.items():
        seconds = float(key[:-1])
        print(f"[train] {key} recipe={config.name}", flush=True)
        fold_scores, oof = cross_validate(
            dev_rows,
            seconds,
            args.mode,
            config,
            args.seed,
            device,
            aux_positions=aux_positions,
            aux_weight=aux_weight,
        )
        oof_dir = args.output_dir / "oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        oof_path = oof_dir / f"{seconds:g}s_{args.mode}.csv"
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
            writer.writeheader()
            writer.writerows(oof)
        selected_epochs = math.ceil(
            float(np.median([score["best_epoch"] for score in fold_scores]))
        )
        print(
            f"  OOF macro-NMAE={_summary(fold_scores)['macro_nmae']:.6f} "
            f"epochs={selected_epochs}",
            flush=True,
        )
        for seed in SEEDS:
            output_path = args.output_dir / key / f"seed_{seed}.pt"
            train_final(
                dev_rows,
                seconds,
                args.mode,
                config,
                selected_epochs,
                seed,
                device,
                output_path,
                aux_positions=aux_positions,
                aux_weight=aux_weight,
            )
            print(f"  saved {output_path}", flush=True)
        output["selected"][key] = {
            "seconds": seconds,
            "mode": args.mode,
            "recipe": config.to_dict(),
            "aux_positions": list(aux_positions),
            "aux_weight": aux_weight,
            "selected_epochs": selected_epochs,
            "fold_scores": fold_scores,
            "fold_summary": _summary(fold_scores),
            "global_oof": _global_summary(oof, dev_ranges),
            "oof_file": str(oof_path),
            "checkpoints": {
                str(seed): str(args.output_dir / key / f"seed_{seed}.pt")
                for seed in SEEDS
            },
        }
        (args.output_dir / "selection.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Wrote {args.output_dir / 'selection.json'}", flush=True)


if __name__ == "__main__":
    main()
