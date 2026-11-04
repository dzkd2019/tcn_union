"""最终 10 秒模型的五折 OOF 入口。

这里不再维护增强配方目录；最终方案固定为
cutout 2 秒 + 5 秒辅助监督 0.5 + EMA 0.99。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from .augmentation_train import (
    AUX_POSITIONS,
    AUX_WEIGHT,
    EMA_DECAY,
    cross_validate,
    global_summary,
    summary,
)
from .train import BASE_SEED, _read_manifest

AXES = ("H2", "NH3", "NO2")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行最终 10 秒模型的五折 OOF")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    parser.add_argument("--recipes", default="cutout_1.0_2s", help="兼容旧命令，只支持 cutout")
    parser.add_argument("--aux-positions", type=int, nargs="*", default=list(AUX_POSITIONS))
    parser.add_argument("--aux-weight", type=float, default=AUX_WEIGHT)
    parser.add_argument("--ema-decay", type=float, default=EMA_DECAY)
    args = parser.parse_args()

    if args.seconds != 10.0:
        parser.error("最终 OOF 只支持 10 秒窗口")
    if args.recipes not in {"cutout", "cutout_1.0_2s"}:
        parser.error("最终方案只支持 cutout")
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)

    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    dev_ranges = np.ptp(np.stack([row["_labels"] for row in dev_rows]), axis=0)

    fold_scores, oof = cross_validate(
        dev_rows,
        10.0,
        args.seed,
        device,
        cutout=True,
        aux_positions=tuple(args.aux_positions),
        aux_weight=args.aux_weight,
        ema_decay=args.ema_decay,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    oof_dir = args.output_dir / "10s" / "cutout"
    oof_dir.mkdir(parents=True, exist_ok=True)
    oof_path = oof_dir / "oof.csv"
    with oof_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
        writer.writeheader()
        writer.writerows(oof)

    result = {
        "seconds": 10.0,
        "recipe": "cutout_1.0_2s",
        "seed": args.seed,
        "aux_positions": list(args.aux_positions),
        "aux_weight": args.aux_weight,
        "ema_decay": args.ema_decay,
        "folds": fold_scores,
        "fold_summary": summary(fold_scores),
        "global_oof": global_summary(oof, dev_ranges),
    }
    (args.output_dir / "search_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metrics = result["global_oof"]
    print(
        f"macro-NMAE={metrics['macro_nmae']:.6f} "
        f"MAPE={metrics['macro_mape_percent']:.3f}% "
        f"epochs={result['fold_summary']['best_epochs']}",
        flush=True,
    )
    print(f"Wrote {oof_path} and {args.output_dir / 'search_results.json'}")


if __name__ == "__main__":
    main()
