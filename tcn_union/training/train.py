"""训练核心：数据校验、z-score、TCN 训练循环和 checkpoint 保存。

最终 10 秒模型的完整训练入口见 ``gan_train.py``；本模块只保留可复用的
底层函数，避免把数据划分、增强策略和命令行逻辑混在一起。
"""

from __future__ import annotations

import hashlib
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

from tcn_union.core.data import load_run, make_features
from tcn_union.core.model import GasTCN

BASE_SEED = 20260923
SEEDS = [20260923, 20260924, 20260925]
MAX_EPOCHS = 500
PATIENCE = 50
BATCH_SIZE = 8
TARGET_TRANSFORMS = ("none", "log")
# log 变换的浓度下限：真实标签均为正，该截断只用于吸收增强样本中可能出现的非正值。
LOG_EPS = 1e-6


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
    aux_positions: tuple[int, ...] = (),
    aux_weight: float = 0.5,
    ema_decay: float | None = None,
    sample_weight_power: float = 0.0,
    sample_weight_powers: tuple[float, float, float] | None = None,
    validation_metric: str = "nmae",
    target_transform: str = "none",
) -> tuple[GasTCN, dict[str, np.ndarray], int, float | None, np.ndarray | None]:
    """训练一个网络，可选执行验证集选 epoch 与早停。

    参数：`x_train/y_train` 是训练批量；`x_validation/y_validation` 可选；
    `train_range_y` 提供 macro-NMAE 的训练标签范围；`epochs` 是最多轮数；
    `seed` 控制初始化与打乱；`device` 指定计算设备；`early_stop` 控制是否按
    验证分数保存最佳权重并在 50 轮无改善后停止。
    返回：`(model, scaler_stats, epochs_used, best_validation_score,
    validation_predictions)`；无验证集的最终训练中，后两项为 `None`。
    `aux_positions` 非空时在同一隐藏序列的过去位置增加辅助损失，权重为
    `aux_weight`，不改变主输出和推理接口。`ema_decay` 非空时维护权重的
    指数滑动平均，并用 EMA 权重做验证、早停和最终返回。
    `sample_weight_power>0` 时按 (每气体训练中位数浓度 / 样本浓度)^power 加权，
    让低浓度样本对损失贡献更大；`validation_metric="three_sigma"` 时用验证集
    相对误差 3σ 的宏平均做早停，而不是 macro-NMAE。
    `target_transform="log"` 时先把训练标签变为 `log(max(y,1e-6))`，再在 log
    空间拟合 z-score、计算 SmoothL1 和早停；验证预测会先反标准化再取指数，
    因此 `three_sigma`/NMAE 仍按原始浓度计算。log 空间的绝对误差近似原始空间
    的相对误差，可让训练目标直接对齐低浓度点的相对误差；默认 `none` 保持
    原始浓度空间，行为与之前完全一致。
    输入与标签均仅用训练数据 z-score，避免响应尺度和 H2 数值量级主导联合回归。
    固定 batch size=8、AdamW(lr=1e-3, weight_decay=1e-3, betas=(0.9,0.999))；
    对标准化三目标使用 mean SmoothL1(beta=1)，相较纯平方误差对小样本异常误差
    更稳健；每步将梯度范数裁剪到 1，并在 loss/gradient 非有限时立即失败。
    CV 最多训练 500 轮，按反标准化预测的验证 macro-NMAE 保存最佳权重，并在
    50 轮无改善时早停；最终全开发集重训使用折最佳轮数中位数，不使用测试集早停。
    """
    if target_transform not in TARGET_TRANSFORMS:
        raise ValueError("target_transform 必须是 none 或 log")
    # 固定随机种子：保证同一 seed 下的初始化、打乱顺序和 Dropout 可复现。
    set_seed(seed)

    # 可选标签变换：log 空间中的绝对误差近似原始空间的相对误差，
    # 使训练目标与 three_sigma/MAPE 关注的相对误差对齐。默认 none 不变。
    y_train_fit = y_train
    if target_transform == "log":
        y_train_fit = np.log(np.maximum(y_train, LOG_EPS)).astype(y_train.dtype)
    # z-score 统计量只从当前训练折拟合，验证集复用同一组统计量，避免泄漏。
    stats = fit_scalers(x_train, y_train_fit)
    x_train_z, y_train_z = _scaled(x_train, y_train_fit, stats)

    # 低浓度样本的绝对误差小，但相对误差大；按浓度中位数反比加权，
    # 让损失更关注低浓度尾部。power=0 时退化为原来的等权训练。
    if sample_weight_power < 0.0:
        raise ValueError("sample_weight_power 不能为负")
    if sample_weight_powers is not None:
        if len(sample_weight_powers) != 3 or any(p < 0 for p in sample_weight_powers):
            raise ValueError("sample_weight_powers 必须是三个非负数")
        power_vector = np.asarray(sample_weight_powers, dtype=np.float64)
    else:
        power_vector = np.full(3, float(sample_weight_power))
    safe_y = np.maximum(y_train.astype(np.float64), 1e-9)
    if np.any(power_vector > 0.0):
        median_y = np.median(safe_y, axis=0, keepdims=True)
        sample_weights = np.clip(median_y / safe_y, 0.1, 10.0) ** power_vector[None, :]
        sample_weights = sample_weights / sample_weights.mean(axis=0, keepdims=True)
    else:
        sample_weights = np.ones_like(safe_y, dtype=np.float64)

    # 小样本训练集直接放进内存；每个 epoch 重新打乱一次。
    dataset = TensorDataset(
        torch.from_numpy(x_train_z),
        torch.from_numpy(y_train_z),
        torch.from_numpy(sample_weights.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, generator=generator
    )

    # 模型输入通道数由特征决定：raw_delta 为 6，delta 为 3。
    model = GasTCN(x_train.shape[1]).to(device)
    if ema_decay is not None and not 0.0 < ema_decay < 1.0:
        raise ValueError("ema_decay must be in (0, 1)")
    ema_state = None
    if ema_decay is not None:
        ema_state = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }
    # AdamW 负责参数更新；SmoothL1 对离群误差没有平方损失那么敏感。
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9, 0.999)
    )
    criterion = nn.SmoothL1Loss(beta=1.0, reduction="none")

    best_score = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_prediction: np.ndarray | None = None
    stale_epochs = 0
    val_tensors = None

    if x_validation is not None and y_validation is not None:
        # 验证集必须用训练折的均值和标准差标准化，不能重新拟合。
        x_val_z = (x_validation - stats["x_mean"][None, :, None]) / stats[
            "x_std"
        ][None, :, None]
        val_tensors = torch.from_numpy(x_val_z).to(device)
    for epoch in range(1, epochs + 1):
        # ---- 训练阶段：只更新参数，不做验证 ----
        model.train()
        for batch_x, batch_y, batch_weight in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_weight = batch_weight.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = (criterion(prediction, batch_y) * batch_weight).mean()
            if aux_positions:
                # 辅助头只在训练时加损失；推理仍然只使用最后一个位置的主输出。
                for auxiliary in model.forward_positions(batch_x, aux_positions):
                    aux_loss = (criterion(auxiliary, batch_y) * batch_weight).mean()
                    loss = loss + aux_weight * aux_loss
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
            if ema_state is not None:
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        ema_state[name].mul_(ema_decay).add_(
                            parameter.detach(), alpha=1.0 - ema_decay
                        )
        # ---- 验证阶段：算 macro-NMAE，保存最好一轮 ----
        if val_tensors is not None:
            model.eval()
            training_state = None
            if ema_state is not None:
                training_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }
                model.load_state_dict(ema_state)
            with torch.no_grad():
                pred_z = model(val_tensors).cpu().numpy()
            if training_state is not None:
                model.load_state_dict(training_state)
            pred = pred_z * stats["y_std"][None, :] + stats["y_mean"][None, :]
            if target_transform == "log":
                # log 空间模型输出需先反标准化再取指数，才能与原始浓度标签比较。
                pred = np.exp(pred)
            if y_validation is None:
                raise RuntimeError("validation labels are missing")
            if validation_metric == "three_sigma":
                relative = 100.0 * (pred - y_validation) / np.maximum(
                    np.abs(y_validation), 1e-9
                )
                score = float(3.0 * np.mean(np.std(relative, axis=0, ddof=1)))
            elif validation_metric == "nmae":
                score, _ = _macro_nmae(y_validation, pred, train_range_y)
            else:
                raise ValueError("validation_metric 必须是 nmae 或 three_sigma")
            if not np.isfinite(score):
                raise FloatingPointError(
                    f"non-finite validation score at epoch {epoch}"
                )
            # 只按验证 macro-NMAE 选 epoch；测试集从不参与这一步。
            if score < best_score:
                best_score = score
                best_epoch = epoch
                state_for_best = ema_state if ema_state is not None else model.state_dict()
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in state_for_best.items()
                }
                best_prediction = pred.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
            # 连续 PATIENCE 轮没有改善就提前停止，减少过拟合和训练时间。
            if early_stop and stale_epochs >= PATIENCE:
                break
    # 验证训练结束后，恢复到验证集最好那一轮的权重（不是最后一轮）。
    if val_tensors is not None:
        if best_state is None or best_prediction is None:
            raise RuntimeError(
                "validation training did not produce a finite best checkpoint"
            )
        model.load_state_dict(best_state)
        return model, stats, best_epoch, best_score, best_prediction
    if ema_state is not None:
        model.load_state_dict(ema_state)
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
    target_transform: str = "none",
) -> None:
    """保存可复现推理所需的权重、预处理统计量和训练元数据。

    参数：`output_path` 是 checkpoint 输出路径；`model` 为已训练网络；
    `stats` 为训练集 z-score 参数；`mode/seconds` 定义特征协议；`train_rows`
    提供训练文件哈希；`seed/epochs/device` 记录训练配置；`target_transform`
    记录标签空间（`none` 为原始浓度，`log` 时推理端需对还原结果取指数）。
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
        "target_transform": target_transform,
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



