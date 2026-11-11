"""用已训练 checkpoint 从 CSV 的 5/10 秒因果前缀预测三气体浓度。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from tcn_union.core.data import load_prediction_prefix, make_features
from tcn_union.core.device import configure_threads, describe_device, resolve_device
from tcn_union.core.model import GasTCN
from tcn_union.training.train import _restore_target


def predict(checkpoint_path: Path, csv_path: Path, device: torch.device) -> dict[str, float | str]:
    """加载 checkpoint 并预测一次实验的 H2、NH3、NO2 浓度。

    参数：`checkpoint_path` 指向训练产物，包含网络权重及输入/标签缩放量；
    `csv_path` 是含 time/H2/NH3/NO2 的文件，可不含浓度标签；`device` 是
    `torch.device`，由 CLI 明确选择 CPU 或可用 CUDA。
    返回：JSON 可序列化的原始浓度预测字典，含窗口秒数和原始单位标记。
    先按 checkpoint 的 mode 构造因果特征、复用训练统计量标准化，再反标准化
    网络输出；TCN 宽度由 `architecture.channels` 决定（缺键时按 16 处理，
    兼容旧 checkpoint）。checkpoint 的 `target_transform` 为单值或逐气体配置
    时，由 `_restore_target` 逐气体取指数还原到原始浓度（旧 checkpoint 缺该
    键时按 `none` 处理）。不截断负预测或超训练范围结果，以免掩盖模型误差。
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    seconds = float(checkpoint["seconds"])
    mode = str(checkpoint["mode"])
    time, responses = load_prediction_prefix(csv_path, seconds)
    features = make_features(time, responses, seconds, mode)
    input_mean = np.asarray(checkpoint["input_mean"], dtype=np.float32)
    input_std = np.asarray(checkpoint["input_std"], dtype=np.float32)
    if features.shape[0] != len(input_mean) or input_mean.shape != input_std.shape:
        raise ValueError("checkpoint input statistics do not match feature channels")
    features = (features - input_mean[:, None]) / input_std[:, None]
    architecture = checkpoint["architecture"]
    model = GasTCN(
        int(architecture["in_channels"]),
        channels=int(architecture.get("channels", 16)),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        prediction_z = model(torch.from_numpy(features).unsqueeze(0).to(device)).cpu().numpy()[0]
    prediction = prediction_z * np.asarray(checkpoint["label_std"]) + np.asarray(checkpoint["label_mean"])
    prediction = _restore_target(prediction, checkpoint.get("target_transform"))
    if not np.isfinite(prediction).all():
        raise FloatingPointError("model produced a non-finite concentration prediction")
    return {
        "seconds": seconds,
        "H2": float(prediction[0]),
        "NH3": float(prediction[1]),
        "NO2": float(prediction[2]),
        "unit": "original",
    }


def main() -> None:
    """解析预测 CLI 参数、检查设备并将预测结果输出为 JSON。"""
    parser = argparse.ArgumentParser(description="Predict gas concentrations from a response prefix")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/cuda:0")
    args = parser.parse_args()
    device = resolve_device(args.device)
    configure_threads(device)
    print(f"[device] {describe_device(device)}", file=sys.stderr)
    result = predict(args.checkpoint, args.csv, device)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
