"""在锁定的 12 个测试实验上一次性评估冻结的增强 TCN。

训练冻结后才读取测试集。报告保留三个固定种子各自的结果，并把种子平均集成
作为次要视图，同时将增强模型与原始 clean checkpoint 做比较。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .evaluate import (
    AXES,
    _bootstrap_difference,
    _load_features,
    _manifest,
    _metric_summary,
    _tcn_predict,
)


def _load_predictions(
    model_dir: Path,
    seconds: int,
    mode: str,
    test_rows: list[dict[str, Any]],
    device: torch.device,
    seeds: list[int],
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    features = _load_features(test_rows, float(seconds), mode)
    x = np.stack([features[row["file"]] for row in test_rows])
    predictions: dict[int, np.ndarray] = {}
    for seed in seeds:
        path = model_dir / f"{seconds}s" / f"seed_{seed}.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        prediction, checkpoint = _tcn_predict(path, x, device)
        if int(checkpoint["seconds"]) != seconds or checkpoint.get("mode") != mode:
            raise ValueError(f"{path}: checkpoint protocol does not match {seconds}s/{mode}")
        predictions[seed] = prediction
    ensemble = np.mean(np.stack([predictions[seed] for seed in seeds]), axis=0)
    return predictions, ensemble


def evaluate(
    manifest_path: Path,
    baseline_dir: Path,
    model_dir: Path,
    output_dir: Path,
    device: torch.device,
    seeds: list[int] | None = None,
    mode: str = "raw_delta",
    windows: list[int] | None = None,
) -> dict[str, Any]:
    dev_rows, test_rows = _manifest(manifest_path)
    if seeds is None:
        seeds = [20260923, 20260924, 20260925]
    if windows is None:
        windows = [5, 10]
    y_test = np.stack([row["_labels"] for row in test_rows])
    y_dev = np.stack([row["_labels"] for row in dev_rows])
    dev_ranges = np.ptp(y_dev, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, Any]] = []
    predictions_table: list[dict[str, Any]] = []
    prediction_arrays: dict[str, dict[str, np.ndarray]] = {}
    for label, directory in (("baseline", baseline_dir), ("candidate", model_dir)):
        for seconds in windows:
            seed_predictions, ensemble = _load_predictions(
                directory, seconds, mode, test_rows, device, seeds
            )
            prediction_arrays[f"{label}_{seconds}s"] = {
                str(seed): seed_predictions[seed] for seed in seeds
            }
            prediction_arrays[f"{label}_{seconds}s"]["ensemble"] = ensemble
            for seed in seeds:
                prediction = seed_predictions[seed]
                summary = _metric_summary(y_test, prediction, dev_ranges)
                metrics.append(
                    {
                        "model": label,
                        "window_seconds": seconds,
                        "seed": seed,
                        "n": len(test_rows),
                        **summary,
                    }
                )
                for row, truth, pred in zip(test_rows, y_test, prediction, strict=True):
                    predictions_table.append(
                        {
                            "model": label,
                            "window_seconds": seconds,
                            "seed": seed,
                            "file": row["file"],
                            **{f"{axis}_true": float(truth[i]) for i, axis in enumerate(AXES)},
                            **{f"{axis}_pred": float(pred[i]) for i, axis in enumerate(AXES)},
                        }
                    )
            summary = _metric_summary(y_test, ensemble, dev_ranges)
            metrics.append(
                {
                    "model": label,
                    "window_seconds": seconds,
                    "seed": "ensemble",
                    "n": len(test_rows),
                    **summary,
                }
            )
            for row, truth, pred in zip(test_rows, y_test, ensemble, strict=True):
                predictions_table.append(
                    {
                        "model": label,
                        "window_seconds": seconds,
                        "seed": "ensemble",
                        "file": row["file"],
                        **{f"{axis}_true": float(truth[i]) for i, axis in enumerate(AXES)},
                        **{f"{axis}_pred": float(pred[i]) for i, axis in enumerate(AXES)},
                    }
                )

    comparisons: dict[str, Any] = {}
    pairs = [
        ("candidate_10s_minus_candidate_5s", "candidate_10s", "candidate_5s"),
        ("candidate_10s_minus_baseline_10s", "candidate_10s", "baseline_10s"),
        ("candidate_5s_minus_baseline_5s", "candidate_5s", "baseline_5s"),
    ]
    for name, left, right in pairs:
        if left not in prediction_arrays or right not in prediction_arrays:
            continue
        comparisons[name] = _bootstrap_difference(
            y_test,
            prediction_arrays[left]["ensemble"],
            prediction_arrays[right]["ensemble"],
            dev_ranges,
        )

    metrics_path = output_dir / "test_metrics.csv"
    flat_rows = []
    for item in metrics:
        row = {
            "model": item["model"],
            "window_seconds": item["window_seconds"],
            "seed": item["seed"],
            "n": item["n"],
            "macro_NMAE": item["macro_NMAE"],
            "macro_MAPE_percent": item["macro_MAPE_percent"],
        }
        for axis in AXES:
            row[f"{axis}_MAE"] = item[axis]["MAE"]
            row[f"{axis}_RMSE"] = item[axis]["RMSE"]
            row[f"{axis}_R2"] = item[axis]["R2"]
            row[f"{axis}_NMAE"] = item[axis]["NMAE"]
            row[f"{axis}_MAPE_percent"] = item[axis]["MAPE_percent"]
        flat_rows.append(row)
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    predictions_path = output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions_table[0]))
        writer.writeheader()
        writer.writerows(predictions_table)

    report = {
        "protocol": "locked 12-run test; candidate frozen before test read",
        "test_n": len(test_rows),
        "development_ranges": dict(zip(AXES, dev_ranges.tolist(), strict=True)),
        "seeds": seeds,
        "mode": mode,
        "metrics": metrics,
        "comparisons": comparisons,
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen candidate TCNs on the locked test set")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="raw_delta")
    parser.add_argument("--seeds", type=int, nargs="+", default=[20260923, 20260924, 20260925])
    parser.add_argument("--windows", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--rmse-plots", action="store_true", help="额外生成逐气体 RMSE 散点图")
    args = parser.parse_args()
    if args.device == "cpu":
        torch.set_num_threads(1)
    device = torch.device(args.device)
    report = evaluate(
        args.manifest,
        args.baseline_dir,
        args.model_dir,
        args.output_dir,
        device,
        seeds=args.seeds,
        mode=args.mode,
        windows=args.windows,
    )
    for item in report["metrics"]:
        if item["seed"] == "ensemble":
            print(
                f"{item['model']} {item['window_seconds']}s ensemble "
                f"macro-NMAE={item['macro_NMAE']:.6f} "
                f"MAPE={item['macro_MAPE_percent']:.4f}"
            )
    if args.rmse_plots:
        from .plot_rmse_scatter import plot_error_sigma, plot_rmse_scatter

        for label in ("baseline", "candidate"):
            for seconds in args.windows:
                prefix = f"{label}_{seconds}s"
                plot_rmse_scatter(
                    args.output_dir / "test_predictions.csv",
                    args.output_dir / "figures",
                    label,
                    seconds,
                    "ensemble",
                    prefix=prefix,
                )
                plot_error_sigma(
                    args.output_dir / "test_predictions.csv",
                    args.output_dir / "figures",
                    label,
                    seconds,
                    "ensemble",
                    prefix=prefix,
                )
    print(f"Wrote {args.output_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
