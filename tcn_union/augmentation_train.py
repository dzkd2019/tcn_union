"""最终 10 秒模型的训练入口。

最终方案只有两个窗口：
- 5 秒 clean：不增强、不辅助监督、不使用 EMA；
- 10 秒最终模型：cutout + 5 秒辅助监督 + EMA 0.99。

测试集只在方案冻结后评估；本模块不读取测试集。
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

from .augment import (
    CUTOUT_MAX_SECONDS,
    CUTOUT_PROBABILITY,
    CutoutConfig,
    make_cutout_context,
)
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
MODE = "raw_delta"
CUTOUT = CutoutConfig(probability=CUTOUT_PROBABILITY, max_seconds=CUTOUT_MAX_SECONDS)
AUX_POSITIONS = (25,)
AUX_WEIGHT = 0.5
EMA_DECAY = 0.99


def summary(fold_scores: list[dict[str, Any]]) -> dict[str, Any]:
    """把逐折分数汇总成 macro 均值。"""
    return {
        "macro_nmae": float(np.mean([row["macro_nmae"] for row in fold_scores])),
        "by_gas_nmae": {
            gas: float(np.mean([row[f"{gas}_nmae"] for row in fold_scores]))
            for gas in AXES
        },
        "by_gas_mape_percent": {
            gas: float(np.mean([row[f"{gas}_mape"] for row in fold_scores]))
            for gas in AXES
        },
        "best_epochs": [int(row["best_epoch"]) for row in fold_scores],
    }


def global_summary(oof: list[dict[str, Any]], dev_ranges: np.ndarray) -> dict[str, Any]:
    """汇总全部 OOF 预测的逐气体误差。"""
    truth = np.array([[row[f"{gas}_true"] for gas in AXES] for row in oof])
    pred = np.array([[row[f"{gas}_pred"] for gas in AXES] for row in oof])
    errors = np.abs(pred - truth)
    mae = errors.mean(axis=0)
    nmae = mae / dev_ranges
    mape = 100.0 * (errors / np.maximum(np.abs(truth), 1e-8)).mean(axis=0)
    return {
        "n": len(oof),
        "by_gas_mae": dict(zip(AXES, mae.tolist(), strict=True)),
        "by_gas_nmae": dict(zip(AXES, nmae.tolist(), strict=True)),
        "by_gas_mape_percent": dict(zip(AXES, mape.tolist(), strict=True)),
        "macro_nmae": float(nmae.mean()),
        "macro_mape_percent": float(mape.mean()),
    }


def cross_validate(
    dev_rows: list[dict[str, Any]],
    seconds: float,
    seed: int,
    device: torch.device,
    *,
    cutout: bool,
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
    ema_decay: float | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """对一个窗口跑五折 OOF，返回逐折分数和 OOF 预测。"""
    features = _features(dev_rows, seconds, MODE)
    fold_scores: list[dict[str, Any]] = []
    oof: list[dict[str, Any]] = []
    for fold in range(5):
        train_rows = [row for row in dev_rows if int(row["fold"]) != fold]
        val_rows = [row for row in dev_rows if int(row["fold"]) == fold]
        x_train = np.stack([features[row["file"]] for row in train_rows])
        y_train = np.stack([row["_labels"] for row in train_rows]).astype(np.float32)
        x_val = np.stack([features[row["file"]] for row in val_rows])
        y_val = np.stack([row["_labels"] for row in val_rows]).astype(np.float32)

        context = (
            make_cutout_context(train_rows, seconds, MODE, CUTOUT, seed)
            if cutout
            else None
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
            cutout=context,
            aux_positions=aux_positions,
            aux_weight=aux_weight,
            ema_decay=ema_decay,
        )
        if prediction is None or best_score is None:
            raise RuntimeError("五折训练没有产生验证预测")

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


def train_final(
    dev_rows: list[dict[str, Any]],
    seconds: float,
    epochs: int,
    seed: int,
    device: torch.device,
    output_path: Path,
    *,
    cutout: bool,
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
    ema_decay: float | None = None,
) -> None:
    """在全部开发实验上重训一个种子并保存 checkpoint。"""
    features = _features(dev_rows, seconds, MODE)
    x_dev = np.stack([features[row["file"]] for row in dev_rows])
    y_dev = np.stack([row["_labels"] for row in dev_rows]).astype(np.float32)
    context = (
        make_cutout_context(dev_rows, seconds, MODE, CUTOUT, seed)
        if cutout
        else None
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
        cutout=context,
        aux_positions=aux_positions,
        aux_weight=aux_weight,
        ema_decay=ema_decay,
    )
    metadata = None
    if cutout or aux_positions:
        metadata = {
            "recipe": "cutout",
            "cutout_probability": CUTOUT.probability,
            "cutout_max_seconds": CUTOUT.max_seconds,
            "aux_positions": list(aux_positions),
            "aux_weight": aux_weight,
            "ema_decay": ema_decay,
        }
    _save_checkpoint(
        output_path,
        model,
        stats,
        MODE,
        seconds,
        dev_rows,
        seed,
        epochs,
        device,
        augmentation=metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="训练最终 5 秒/10 秒 TCN")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--oof-only", action="store_true", help="只跑五折 OOF，不重训最终模型")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    dev_ranges = np.ptp(np.stack([row["_labels"] for row in dev_rows]), axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 5 秒只用 clean；10 秒使用 cutout + 5 秒辅助 + EMA。
    windows = (
        ("5s", 5.0, False, (), None),
        ("10s", 10.0, True, AUX_POSITIONS, EMA_DECAY),
    )
    selection: dict[str, Any] = {
        "protocol": "固定开发集五折 OOF；最终在 50 个开发实验上重训；测试集不参与训练",
        "base_seed": args.seed,
        "final_seeds": SEEDS,
        "mode": MODE,
        "selected": {},
    }
    for key, seconds, use_cutout, aux_positions, ema_decay in windows:
        print(f"[train] {key}", flush=True)
        fold_scores, oof = cross_validate(
            dev_rows,
            seconds,
            args.seed,
            device,
            cutout=use_cutout,
            aux_positions=aux_positions,
            ema_decay=ema_decay,
        )
        oof_dir = args.output_dir / "oof"
        oof_dir.mkdir(parents=True, exist_ok=True)
        oof_path = oof_dir / f"{seconds:g}s_{MODE}.csv"
        with oof_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
            writer.writeheader()
            writer.writerows(oof)

        epochs = math.ceil(float(np.median([row["best_epoch"] for row in fold_scores])))
        print(f"  OOF macro-NMAE={summary(fold_scores)['macro_nmae']:.6f} epochs={epochs}", flush=True)
        if not args.oof_only:
            for seed in SEEDS:
                path = args.output_dir / key / f"seed_{seed}.pt"
                train_final(
                    dev_rows,
                    seconds,
                    epochs,
                    seed,
                    device,
                    path,
                    cutout=use_cutout,
                    aux_positions=aux_positions,
                    aux_weight=AUX_WEIGHT,
                    ema_decay=ema_decay,
                )
                print(f"  saved {path}", flush=True)

        selection["selected"][key] = {
            "seconds": seconds,
            "cutout": use_cutout,
            "aux_positions": list(aux_positions),
            "aux_weight": AUX_WEIGHT if aux_positions else None,
            "ema_decay": ema_decay,
            "selected_epochs": epochs,
            "fold_scores": fold_scores,
            "fold_summary": summary(fold_scores),
            "global_oof": global_summary(oof, dev_ranges),
            "oof_file": str(oof_path),
        }
        (args.output_dir / "selection.json").write_text(
            json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(f"Wrote {args.output_dir / 'selection.json'}", flush=True)


if __name__ == "__main__":
    main()
