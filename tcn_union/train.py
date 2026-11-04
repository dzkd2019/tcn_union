"""五折选择预处理方式，并训练最终早期响应 TCN。

主流程只用 50 个开发实验完成 5/10 秒 × raw_delta/delta 的五折验证；按
macro-NMAE 选择每个时限的输入模式，再用折内最佳轮数中位数在全部开发集
重训三个固定随机种子。验证折之外的数据不参与该折的标准化参数拟合，锁定
测试集也不在本模块中打分，防止信息泄漏和测试集驱动的模型选择。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .augment import AugmentContext, AugmentedDataset
from .data import load_run, make_features
from .model import GasTCN

BASE_SEED = 20260923
SEEDS = [20260923, 20260924, 20260925]
SECONDS = [5.0, 10.0]
MODES = ["raw_delta", "delta"]
MAX_EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 8


def set_seed(seed: int) -> None:
    """同时重置 Python、NumPy 和 PyTorch 随机数生成器。

    参数：`seed` 是当前训练运行的固定种子。若 CUDA 可用也重置其随机状态，
    以便交叉验证和最终多种子实验可复现。
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - 按固定种子协议重置 NumPy 全局随机数生成器
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scalers(
    x_train: np.ndarray, y_train: np.ndarray
) -> dict[str, np.ndarray]:
    """仅用训练实验拟合输入通道和三路标签的 z-score 参数。

    参数：`x_train` 形状为 `[runs,channels,time]`；`y_train` 形状为
    `[runs,3]`。
    返回：含 `x_mean/x_std/y_mean/y_std` 的字典。输入均值/标准差按训练实验
    与时间点在每通道上统计；标签统计按训练实验逐气体统计。标准化让不同
    响应幅度和浓度单位可共同优化；只用训练折统计量可避免验证/测试泄漏。
    小于 `1e-8` 的标准差置为 1，避免除零并保留常量通道。
    """
    x_mean = np.mean(x_train, axis=(0, 2), dtype=np.float64).astype(np.float32)
    x_std = np.std(x_train, axis=(0, 2), dtype=np.float64).astype(np.float32)
    y_mean = np.mean(y_train, axis=0, dtype=np.float64).astype(np.float32)
    y_std = np.std(y_train, axis=0, dtype=np.float64).astype(np.float32)
    x_std[x_std < 1e-8] = 1.0
    y_std[y_std < 1e-8] = 1.0
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}


def _scaled(
    x: np.ndarray, y: np.ndarray, stats: dict[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """用已拟合统计量标准化输入特征和回归标签。

    参数：`x`、`y` 分别是批量输入/标签；`stats` 由 `fit_scalers` 从训练集得到。
    返回：形状不变的 `(x_z,y_z)`，标签也按气体独立 z-score，使 H2 等大数值
    单位不会仅凭量级主导三目标的联合损失。
    """
    return (
        (x - stats["x_mean"][None, :, None]) / stats["x_std"][None, :, None],
        (y - stats["y_mean"][None, :]) / stats["y_std"][None, :],
    )


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    """读取并验证锁定的训练 manifest。

    参数：`path` 是划分清单路径。
    返回：包含开发/测试行的清单记录；开发行额外加载完整时序和标签到内部
    字段。验证固定为 50 个开发实验、12 个测试实验和五折各 10 个验证实验，
    并核对开发文件哈希及标签是否与 manifest 一致，避免训练数据版本漂移。
    """
    rows = pd.read_csv(
        path, dtype=str, keep_default_na=False, encoding="utf-8"
    ).to_dict(orient="records")
    required = {"file", "sha256", "split", "fold", "H2", "NH3", "NO2"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain fields {sorted(required)}")
    development = [row for row in rows if row["split"] == "development"]
    if (
        len(rows) != 62
        or len(development) != 50
        or sum(row["split"] == "test" for row in rows) != 12
    ):
        raise ValueError(
            "manifest must contain the locked 50-development/12-test split"
        )
    seen_fold = {int(row["fold"]) for row in development}
    if seen_fold != set(range(5)) or any(
        sum(int(row["fold"]) == f for row in development) != 10
        for f in range(5)
    ):
        raise ValueError(
            "manifest must contain five fixed validation folds of ten runs"
        )
    for row in development:
        path_on_disk = Path(row["file"])
        digest = hashlib.sha256(path_on_disk.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"manifest hash mismatch: {row['file']}")
        time, response, labels = load_run(path_on_disk)
        manifest_labels = np.array(
            [float(row[k]) for k in ("H2", "NH3", "NO2")]
        )
        if not np.array_equal(labels, manifest_labels):
            raise ValueError(f"manifest concentration mismatch: {row['file']}")
        row["_time"] = time
        row["_response"] = response
        row["_labels"] = labels
    return rows


def _macro_nmae(
    truth: np.ndarray, prediction: np.ndarray, train_y: np.ndarray
) -> tuple[float, list[float]]:
    """计算每气体 NMAE 及三气体 macro-NMAE。

    参数：`truth`、`prediction` 均为 `[runs,3]` 原始单位浓度；`train_y` 是本
    折训练标签，用其逐气体极差作归一化分母。
    返回：`(三气体 NMAE 均值, [H2, NH3, NO2] NMAE)`。按开发训练范围归一化
    使不同单位气体可比较；范围为零时显式报错而非给出无意义分数。
    """
    ranges = np.ptp(train_y, axis=0)
    if np.any(ranges == 0):
        axes = ["H2", "NH3", "NO2"][np.flatnonzero(ranges == 0).tolist()[0]]
        raise ValueError(
            f"training fold has zero label range for {axes}; cannot normalize validation error"
        )
    axis_scores = np.mean(np.abs(prediction - truth), axis=0) / ranges
    return float(np.mean(axis_scores)), axis_scores.tolist()


def _train_network(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray | None,
    y_validation: np.ndarray | None,
    train_range_y: np.ndarray,
    epochs: int,
    seed: int,
    device: torch.device,
    early_stop: bool,
    *,
    augment: AugmentContext | None = None,
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
) -> tuple[GasTCN, dict[str, np.ndarray], int, float | None, np.ndarray | None]:
    """训练一个网络，可选执行验证集选 epoch 与早停。

    参数：`x_train/y_train` 是训练批量；`x_validation/y_validation` 可选；
    `train_range_y` 提供 macro-NMAE 的训练标签范围；`epochs` 是最多轮数；
    `seed` 控制初始化与打乱；`device` 指定计算设备；`early_stop` 控制是否按
    验证分数保存最佳权重并在 50 轮无改善后停止。
    返回：`(model, scaler_stats, epochs_used, best_validation_score,
    validation_predictions)`；无验证集的最终训练中，后两项为 `None`。
    `augment` 非空时每轮在线生成增强视图并用干净训练统计量标准化；
    `aux_positions` 非空时在同一隐藏序列的过去位置增加辅助损失，权重为
    `aux_weight`，不改变主输出和推理接口。
    输入与标签均仅用训练数据 z-score，避免响应尺度和 H2 数值量级主导联合回归。
    固定 batch size=8、AdamW(lr=1e-3, weight_decay=1e-3, betas=(0.9,0.999))；
    对标准化三目标使用 mean SmoothL1(beta=1)，相较纯平方误差对小样本异常误差
    更稳健；每步将梯度范数裁剪到 1，并在 loss/gradient 非有限时立即失败。
    CV 最多训练 500 轮，按反标准化预测的验证 macro-NMAE 保存最佳权重，并在
    50 轮无改善时早停；最终全开发集重训使用折最佳轮数中位数，不使用测试集早停。
    """
    set_seed(seed)
    stats = fit_scalers(x_train, y_train)
    x_train_z, y_train_z = _scaled(x_train, y_train, stats)

    if augment is None:
        dataset = TensorDataset(
            torch.from_numpy(x_train_z), torch.from_numpy(y_train_z)
        )
    else:
        dataset = AugmentedDataset(augment)
        if len(dataset) != len(x_train):
            raise ValueError("augmentation context does not match the training runs")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )

    model = GasTCN(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999)
    )
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")

    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: np.ndarray | None = None
    stale_epochs = 0
    val_tensors = None

    if x_validation is not None and y_validation is not None:
        x_val_z = (x_validation - stats["x_mean"][None, :, None]) / stats[
            "x_std"
        ][None, :, None]
        val_tensors = torch.from_numpy(x_val_z).to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        if augment is not None:
            dataset.epoch = epoch
        for batch_x, batch_y in loader:
            if augment is not None:
                batch_x_np, batch_y_np = _scaled(
                    batch_x.numpy(), batch_y.numpy(), stats
                )
                batch_x = torch.from_numpy(batch_x_np)
                batch_y = torch.from_numpy(batch_y_np)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = criterion(prediction, batch_y)
            if aux_positions:
                for auxiliary in model.forward_positions(batch_x, aux_positions):
                    loss = loss + aux_weight * criterion(auxiliary, batch_y)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite training loss at epoch {epoch}"
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if any(
                parameter.grad is not None
                and not torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ):
                raise FloatingPointError(
                    f"non-finite gradient at epoch {epoch}"
                )
            optimizer.step()
        if val_tensors is not None:
            model.eval()
            with torch.no_grad():
                pred_z = model(val_tensors).cpu().numpy()
            pred = pred_z * stats["y_std"][None, :] + stats["y_mean"][None, :]
            if y_validation is None:
                raise RuntimeError("validation labels are missing")
            score, _ = _macro_nmae(y_validation, pred, train_range_y)
            if not np.isfinite(score):
                raise FloatingPointError(
                    f"non-finite validation score at epoch {epoch}"
                )
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_prediction = pred.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
            if early_stop and stale_epochs >= PATIENCE:
                break
    if val_tensors is not None:
        if best_state is None or best_prediction is None:
            raise RuntimeError(
                "validation training did not produce a finite best checkpoint"
            )
        model.load_state_dict(best_state)
        return model, stats, best_epoch, best_score, best_prediction
    return model, stats, epochs, None, None


def _channel_order(mode: str) -> list[str]:
    """返回所选特征模式的通道名称与拼接顺序。

    参数：`mode` 为 `raw_delta` 或 `delta`。
    返回：`raw_delta` 时先列出 H2/NH3/NO2 原始响应、再列出三路增量；
    `delta` 时只列出三路增量。明确顺序供 checkpoint 记录和推理复用。
    """
    raw = ["H2_raw", "NH3_raw", "NO2_raw"]
    delta = ["H2_delta", "NH3_delta", "NO2_delta"]
    return raw + delta if mode == "raw_delta" else delta


def _features(
    rows: list[dict[str, Any]], seconds: float, mode: str
) -> dict[str, np.ndarray]:
    """为开发集中的每个实验构造指定窗口和模式的特征。

    参数：`rows` 为已加载 manifest 记录；`seconds` 是 5/10 秒前缀；
    `mode` 决定使用 delta 或 raw_delta 通道。
    返回：`文件相对路径 -> [channels,samples] 特征` 映射。测试行被排除，且
    特征只依赖响应前缀，使后续折分组保持实验级并确保因果输入。
    """
    result: dict[str, np.ndarray] = {}
    for row in rows:
        if row["split"] != "development":
            continue
        result[row["file"]] = make_features(
            row["_time"], row["_response"], seconds, mode
        )
    return result


def _save_checkpoint(
    output_path: Path,
    model: GasTCN,
    stats: dict[str, np.ndarray],
    mode: str,
    seconds: float,
    train_rows: list[dict[str, Any]],
    seed: int,
    epochs: int,
    device: torch.device,
    *,
    augmentation: dict[str, Any] | None = None,
) -> None:
    """保存可复现推理所需的权重、预处理统计量和训练元数据。

    参数：`output_path` 是 checkpoint 输出路径；`model` 为已训练网络；
    `stats` 为训练集 z-score 参数；`mode/seconds` 定义特征协议；`train_rows`
    提供训练文件哈希；`seed/epochs/device` 记录训练配置。
    返回：无。文件除 state_dict 外还记录架构、通道顺序、标签/输入缩放量、
    数据哈希及软件版本，保证预测端可还原训练时的数据变换。
    """
    payload = {
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        },
        "architecture": {
            "name": "GasTCN",
            "in_channels": len(_channel_order(mode)),
            "channels": 16,
            "dilations": [1, 2, 4],
            "kernel_size": 5,
            "dropout": 0.1,
            "receptive_field_samples": 57,
        },
        "mode": mode,
        "seconds": seconds,
        "channel_order": _channel_order(mode),
        "input_mean": stats["x_mean"].tolist(),
        "input_std": stats["x_std"].tolist(),
        "label_mean": stats["y_mean"].tolist(),
        "label_std": stats["y_std"].tolist(),
        "training_files": {row["file"]: row["sha256"] for row in train_rows},
        "seed": seed,
        "epochs": epochs,
        "optimizer": {
            "name": "AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-3,
            "batch_size": BATCH_SIZE,
        },
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
        "device_used": str(device),
        "augmentation": augmentation,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def train(
    manifest_path: Path, output_dir: Path, device: torch.device
) -> dict[str, Any]:
    """执行五折预处理选择及最终多种子 TCN 训练。

    参数：`manifest_path` 为锁定实验划分；`output_dir` 接收交叉验证、OOF、
    checkpoint 和 selection 结果；`device` 为已通过 CLI 校验的 PyTorch 设备。
    返回：按 5s/10s 汇总所选模式、训练轮数、折分数及随机种子的字典。
    流程先比较两个窗口下 raw_delta 与 delta 的五折 macro-NMAE；同分选 delta，
    epoch 数取所选模式各折最佳轮数的中位数向上取整。然后在 50 个开发实验
    上重新拟合缩放量并训练三个固定种子；不使用 12 个测试实验做选择或评估。
    """
    rows = _read_manifest(manifest_path)
    dev_rows = [row for row in rows if row["split"] == "development"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cv_records: list[dict[str, Any]] = []
    oof_by_combo: dict[tuple[float, str], list[dict[str, Any]]] = {}
    for seconds in SECONDS:
        for mode in MODES:
            combo_features = _features(dev_rows, seconds, mode)
            oof: list[dict[str, Any]] = []
            for fold in range(5):
                train_rows = [
                    row for row in dev_rows if int(row["fold"]) != fold
                ]
                validation_rows = [
                    row for row in dev_rows if int(row["fold"]) == fold
                ]
                x_train = np.stack(
                    [combo_features[row["file"]] for row in train_rows]
                )
                y_train = np.stack(
                    [row["_labels"] for row in train_rows]
                ).astype(np.float32)
                x_val = np.stack(
                    [combo_features[row["file"]] for row in validation_rows]
                )
                y_val = np.stack(
                    [row["_labels"] for row in validation_rows]
                ).astype(np.float32)
                _, _, best_epoch, best_score, prediction = _train_network(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    y_train,
                    MAX_EPOCHS,
                    BASE_SEED,
                    device,
                    early_stop=True,
                )
                if prediction is None or best_score is None:
                    raise RuntimeError("missing out-of-fold predictions")
                _, axis_scores = _macro_nmae(y_val, prediction, y_train)
                cv_records.append(
                    {
                        "seconds": int(seconds),
                        "mode": mode,
                        "fold": fold,
                        "macro_nmae": best_score,
                        "H2_nmae": axis_scores[0],
                        "NH3_nmae": axis_scores[1],
                        "NO2_nmae": axis_scores[2],
                        "best_epoch": best_epoch,
                    }
                )
                for row, truth, pred in zip(
                    validation_rows, y_val, prediction, strict=True
                ):
                    oof.append(
                        {
                            "file": row["file"],
                            "fold": fold,
                            "H2_true": truth[0],
                            "NH3_true": truth[1],
                            "NO2_true": truth[2],
                            "H2_pred": pred[0],
                            "NH3_pred": pred[1],
                            "NO2_pred": pred[2],
                        }
                    )
            oof_by_combo[(seconds, mode)] = oof
            dest = output_dir / "oof" / f"{int(seconds)}s_{mode}.csv"
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(oof[0]))
                writer.writeheader()
                writer.writerows(oof)
            print(f"Completed {int(seconds)}s {mode} five-fold validation")

    scores_path = output_dir / "cv_scores.csv"
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cv_records[0]))
        writer.writeheader()
        writer.writerows(cv_records)

    selections: dict[str, Any] = {}
    for seconds in SECONDS:
        mode_means = {
            mode: float(
                np.mean(
                    [
                        r["macro_nmae"]
                        for r in cv_records
                        if r["seconds"] == int(seconds) and r["mode"] == mode
                    ]
                )
            )
            for mode in MODES
        }
        selected_mode = (
            "delta"
            if mode_means["delta"] <= mode_means["raw_delta"]
            else "raw_delta"
        )
        fold_epochs = [
            r["best_epoch"]
            for r in cv_records
            if r["seconds"] == int(seconds) and r["mode"] == selected_mode
        ]
        selected_epochs = math.ceil(float(np.median(fold_epochs)))
        selections[f"{int(seconds)}s"] = {
            "seconds": seconds,
            "selected_mode": selected_mode,
            "selected_epochs": selected_epochs,
            "mode_mean_macro_nmae": mode_means,
            "fold_best_epochs": fold_epochs,
            "fold_macro_nmae": {
                mode: [
                    r["macro_nmae"]
                    for r in cv_records
                    if r["seconds"] == int(seconds) and r["mode"] == mode
                ]
                for mode in MODES
            },
        }

    for key, selection in selections.items():
        seconds = selection["seconds"]
        mode = selection["selected_mode"]
        combo_features = _features(dev_rows, seconds, mode)
        x_dev = np.stack([combo_features[row["file"]] for row in dev_rows])
        y_dev = np.stack([row["_labels"] for row in dev_rows]).astype(
            np.float32
        )
        for seed in SEEDS:
            model, stats, _, _, _ = _train_network(
                x_dev,
                y_dev,
                None,
                None,
                y_dev,
                selection["selected_epochs"],
                seed,
                device,
                early_stop=False,
            )
            output_path = output_dir / key / f"seed_{seed}.pt"
            _save_checkpoint(
                output_path,
                model,
                stats,
                mode,
                seconds,
                dev_rows,
                seed,
                selection["selected_epochs"],
                device,
            )
            print(f"Saved {output_path}")

    result = {
        "protocol": "locked run-level split; test data not evaluated during training",
        "base_seed": BASE_SEED,
        "final_seeds": SEEDS,
        "cv_scores_file": str(scores_path),
        "selected": selections,
    }
    selection_path = output_dir / "selection.json"
    selection_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    """解析训练 CLI、验证设备并运行交叉验证与最终训练。"""
    parser = argparse.ArgumentParser(
        description="Cross-validate and train selected early-response TCNs"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be cpu, cuda, or cuda:0")
    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)
    result = train(args.manifest, args.output_dir, device)
    for key, selection in result["selected"].items():
        print(
            f"{key}: selected {selection['selected_mode']} at {selection['selected_epochs']} epochs"
        )
    print(f"Wrote {args.output_dir / 'selection.json'}")


if __name__ == "__main__":
    main()
