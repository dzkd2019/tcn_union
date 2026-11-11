"""锁定测试集的 TCN 多 seed / 多目录集成一次性评估（无 Ridge）。

协议：开发集（50 个实验）只用于提供裁剪范围；测试集（12 个锁定实验）只在
本模块最后评估一次。TCN 侧读取多个训练目录的 ``<dir>/10s/seed_<seed>.pt``
（允许每个目录只含部分 seed，兼容按 seed 分目录的训练产物），先对“所有目录
× 可用 seed”的预测取算术平均得到 TCN 集平均，再可选裁剪到开发集逐气体
标签范围：

``pred = clip(tcn_mean)``

不再包含 Ridge 或任何额外模型；产物 CSV 列名保持不变。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tcn_union.core.device import configure_threads, describe_device, resolve_device
from tcn_union.evaluation.evaluate import (
    AXES,
    _load_features,
    _manifest,
    _metric_summary,
    _tcn_predict,
)
from tcn_union.training.train import SEEDS

PREDICTION_FIELDS = [
    "model",
    "window_seconds",
    "seed",
    "file",
    "H2_true",
    "NH3_true",
    "NO2_true",
    "H2_pred",
    "NH3_pred",
    "NO2_pred",
]
METRIC_FIELDS = [
    "model",
    "window_seconds",
    "seed",
    "n",
    "macro_NMAE",
    "macro_MAPE_percent",
    *[
        f"{axis}_{name}"
        for axis in AXES
        for name in ("MAE", "RMSE", "R2", "NMAE", "MAPE_percent")
    ],
]


def _mean_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    """对多组同形预测逐元素取算术平均。"""
    return np.mean(np.stack(predictions), axis=0)

def _clip_to_development_range(
    prediction: np.ndarray, y_dev: np.ndarray
) -> np.ndarray:
    """把三气体预测裁剪到开发集标签的逐气体最小/最大值。

    浓度不可能超出实验设计范围；裁剪是部署端可执行的物理约束，不接触测试标签。
    """
    lower = y_dev.min(axis=0)
    upper = y_dev.max(axis=0)
    return np.clip(prediction, lower[None, :], upper[None, :])

def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """相对误差 3σ（%）：3 * std(100*(pred-true)/max(|true|,1e-9), ddof=1)。

    宏平均为三气体 3σ 的算术平均。
    """
    relative = 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)
    scores = 3.0 * np.std(relative, axis=0, ddof=1)
    result = {axis: float(value) for axis, value in zip(AXES, scores, strict=True)}
    result["macro"] = float(np.mean(scores))
    return result

def _available_seeds(model_dir: Path, seconds: float, seeds: list[int]) -> list[int]:
    """返回目录中实际存在的请求种子。

    训练入口支持 ``--final-seeds`` 每次只重训一个 seed，因此最终 checkpoint
    可能按 seed 分散在多个目录；这里按目录扫描，配合多个 ``--model-dirs``
    仍能按“目录 × 可用 seed”集成。
    """
    target_dir = model_dir / f"{int(seconds)}s"
    return [seed for seed in seeds if (target_dir / f"seed_{seed}.pt").is_file()]

def _tcn_predictions(
    model_dirs: list[Path],
    test_rows: list[dict[str, Any]],
    seconds: float,
    seeds: list[int],
    device: torch.device,
) -> tuple[dict[int, np.ndarray], str]:
    """加载多个目录的 TCN checkpoint，返回逐 seed 的平均测试集预测。

    参数：`model_dirs` 每个目录下含 ``<seconds>s/seed_<seed>.pt``（可只含部分
    种子）；`test_rows` 是锁定测试实验；`seconds` 是预测窗口；`seeds` 是要
    集成的种子。
    返回：`(seed 到预测[runs,3] 的映射, 特征模式描述)`。集成覆盖所有目录中
    实际存在的“目录 × 请求 seed”，同一 seed 在多个目录出现时取算术平均，因此
    CSV 中每个 seed 只有一行；允许不同目录使用不同特征模式（如 raw_delta 与
    delta 混合集成），按各自模式构造输入；窗口不一致时显式报错。
    """
    per_seed: dict[int, list[np.ndarray]] = {seed: [] for seed in seeds}
    modes: set[str] = set()
    features_by_mode: dict[str, np.ndarray] = {}
    for model_dir in model_dirs:
        available = _available_seeds(model_dir, seconds, seeds)
        if not available:
            raise FileNotFoundError(
                f"no TCN checkpoint for seeds {seeds} under {model_dir / f'{int(seconds)}s'}"
            )
        for seed in available:
            path = model_dir / f"{int(seconds)}s" / f"seed_{seed}.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            checkpoint_seconds = float(checkpoint["seconds"])
            if checkpoint_seconds != float(seconds):
                raise ValueError(
                    f"checkpoint window mismatch: {path} has {checkpoint_seconds}s"
                )
            checkpoint_mode = str(checkpoint["mode"])
            modes.add(checkpoint_mode)
            if checkpoint_mode not in features_by_mode:
                feature_map = _load_features(test_rows, float(seconds), checkpoint_mode)
                features_by_mode[checkpoint_mode] = np.stack(
                    [feature_map[row["file"]] for row in test_rows]
                )
            prediction, _ = _tcn_predict(
                path, features_by_mode[checkpoint_mode], device
            )
            per_seed[seed].append(prediction)
    missing = [seed for seed in seeds if not per_seed[seed]]
    if missing:
        raise FileNotFoundError(
            f"no TCN checkpoint found for seeds {missing} in any --model-dirs"
        )
    return {
        seed: _mean_predictions(predictions) for seed, predictions in per_seed.items()
    }, "+".join(sorted(modes))

def _prediction_rows(
    seconds: float,
    model_name: str,
    seed: int | str,
    test_rows: list[dict[str, Any]],
    truth: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    """组装一组 `[runs,3]` 预测的逐实验 CSV 记录。"""
    rows = []
    for row, target, value in zip(test_rows, truth, prediction, strict=True):
        record: dict[str, Any] = {
            "model": model_name,
            "window_seconds": int(seconds),
            "seed": seed,
            "file": row["file"],
        }
        for axis, true_value, predicted_value in zip(
            AXES, target, value, strict=True
        ):
            record[f"{axis}_true"] = float(true_value)
            record[f"{axis}_pred"] = float(predicted_value)
        rows.append(record)
    return rows


def _metric_row(
    seconds: float,
    model_name: str,
    seed: int | str,
    truth: np.ndarray,
    prediction: np.ndarray,
    dev_ranges: np.ndarray,
) -> dict[str, Any]:
    """把 `_metric_summary` 展开成与参考 test_metrics.csv 相同的扁平行。"""
    summary = _metric_summary(truth, prediction, dev_ranges)
    row: dict[str, Any] = {
        "model": model_name,
        "window_seconds": int(seconds),
        "seed": seed,
        "n": len(truth),
        "macro_NMAE": summary["macro_NMAE"],
        "macro_MAPE_percent": summary["macro_MAPE_percent"],
    }
    for axis in AXES:
        for name in ("MAE", "RMSE", "R2", "NMAE", "MAPE_percent"):
            row[f"{axis}_{name}"] = summary[axis][name]
    return row


def _summary_row(
    seconds: float,
    model_name: str,
    seed: int | str,
    truth: np.ndarray,
    prediction: np.ndarray,
    dev_ranges: np.ndarray,
) -> dict[str, Any]:
    """组装 evaluation.json 使用的嵌套指标行（保留 _metric_summary 原结构）。"""
    summary = _metric_summary(truth, prediction, dev_ranges)
    return {
        "model": model_name,
        "window_seconds": int(seconds),
        "seed": seed,
        "n": len(truth),
        **summary,
    }

def evaluate_ensemble(
    manifest_path: Path,
    model_dirs: list[Path],
    output_dir: Path,
    device: torch.device,
    windows: list[int],
    seeds: list[int],
    clip_to_development_range: bool = False,
) -> dict[str, Any]:
    """对锁定测试集做一次 TCN 集平均评估并写出三项产物。

    参数：`manifest_path` 是锁定划分；`model_dirs` 是多个训练目录（各含
    ``10s/seed_<seed>.pt``，允许每个目录只含部分 seed）；`output_dir` 是产物
    目录；`device` 指定推理设备；`windows` 目前只支持 10 秒；`seeds` 是要
    集成的 TCN 种子；`clip_to_development_range` 为真时把集平均预测裁剪到
    开发集逐气体标签范围（物理约束，不使用测试标签）。
    返回：含协议、目录、seed、指标与 3σ 的报告字典；同时写出
    ``test_predictions.csv``、``test_metrics.csv`` 和 ``evaluation.json``。
    """
    if not windows or any(int(window) != 10 for window in windows):
        raise ValueError("only the 10 s prefix is supported")
    if not seeds:
        raise ValueError("at least one seed is required")
    dev_rows, test_rows = _manifest(manifest_path)
    y_dev = np.stack([row["_labels"] for row in dev_rows])
    y_test = np.stack([row["_labels"] for row in test_rows])
    dev_ranges = np.ptp(y_dev, axis=0)
    if np.any(dev_ranges == 0):
        raise ValueError("development label range is zero; normalized error is undefined")

    prediction_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    three_sigma_rows: list[dict[str, Any]] = []
    mode: str | None = None
    for window in windows:
        seconds = float(window)
        seed_predictions, mode = _tcn_predictions(
            model_dirs, test_rows, seconds, seeds, device
        )
        for seed in seeds:
            prediction = seed_predictions[seed]
            prediction_rows.extend(
                _prediction_rows(seconds, "candidate", seed, test_rows, y_test, prediction)
            )
            metric_rows.append(
                _metric_row(seconds, "candidate", seed, y_test, prediction, dev_ranges)
            )
            summary_rows.append(
                _summary_row(seconds, "candidate", seed, y_test, prediction, dev_ranges)
            )
            three_sigma_rows.append(
                {
                    "model": "candidate",
                    "window_seconds": int(seconds),
                    "seed": seed,
                    **_three_sigma(y_test, prediction),
                }
            )
        # TCN 集平均：所有目录 × 所有 seed 的逐 seed 平均再跨 seed 平均。
        ensemble = _mean_predictions([seed_predictions[seed] for seed in seeds])
        if clip_to_development_range:
            ensemble = _clip_to_development_range(ensemble, y_dev)
        prediction_rows.extend(
            _prediction_rows(
                seconds, "candidate", "ensemble", test_rows, y_test, ensemble
            )
        )
        metric_rows.append(
            _metric_row(seconds, "candidate", "ensemble", y_test, ensemble, dev_ranges)
        )
        summary_rows.append(
            _summary_row(seconds, "candidate", "ensemble", y_test, ensemble, dev_ranges)
        )
        three_sigma_rows.append(
            {
                "model": "candidate",
                "window_seconds": int(seconds),
                "seed": "ensemble",
                **_three_sigma(y_test, ensemble),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(prediction_rows)
    metrics_path = output_dir / "test_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(metric_rows)
    report = {
        "protocol": (
            "locked 12-run test; candidate = TCN ensemble mean over model dirs and seeds"
            + ("; predictions clipped to development label range" if clip_to_development_range else "")
            + "; no Ridge; test read once per candidate version"
        ),
        "test_n": len(test_rows),
        "locked_test_runs": [row["file"] for row in test_rows],
        "windows": [int(window) for window in windows],
        "mode": mode,
        "seeds": [int(seed) for seed in seeds],
        "model_dirs": [str(model_dir) for model_dir in model_dirs],
        "clip_to_development_range": bool(clip_to_development_range),
        "development_ranges": dict(zip(AXES, dev_ranges.tolist(), strict=True)),
        "metrics": summary_rows,
        "three_sigma": three_sigma_rows,
        "outputs": {
            "test_predictions": str(predictions_path),
            "test_metrics": str(metrics_path),
        },
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    """解析评估 CLI，检查设备后对锁定测试集做一次性评估。"""
    parser = argparse.ArgumentParser(
        description="Locked-test evaluation: TCN ensemble mean with optional physical-range clipping"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--model-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="训练目录，内部为 <dir>/10s/seed_<seed>.pt；也可传入按 seed 拆分的多个目录",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    parser.add_argument("--windows", type=int, nargs="+", default=[10])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument(
        "--clip-to-development-range",
        action="store_true",
        help="把集平均预测裁剪到开发集逐气体标签范围（物理约束）",
    )
    args = parser.parse_args()
    if any(window != 10 for window in args.windows):
        parser.error("只支持 10 秒前缀评估")
    device = resolve_device(args.device)
    configure_threads(device)
    print(f"[device] {describe_device(device)}")
    report = evaluate_ensemble(
        args.manifest,
        args.model_dirs,
        args.output_dir,
        device,
        args.windows,
        args.seeds,
        clip_to_development_range=args.clip_to_development_range,
    )
    print(
        f"Wrote {args.output_dir / 'evaluation.json'} for {report['test_n']} locked test runs"
    )


if __name__ == "__main__":
    main()
