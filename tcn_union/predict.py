"""用已训练 checkpoint 从 CSV 的 5/10 秒因果前缀预测三气体浓度。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import load_prediction_prefix, make_features
from .model import GasTCN


def predict(checkpoint_path: Path, csv_path: Path, device: torch.device) -> dict[str, float | str]:
    """加载 checkpoint 并预测一次实验的 H2、NH3、NO2 浓度。

    参数：`checkpoint_path` 指向训练产物，包含网络权重及输入/标签缩放量；
    `csv_path` 是含 time/H2/NH3/NO2 的文件，可不含浓度标签；`device` 是
    `torch.device`，由 CLI 明确选择 CPU 或可用 CUDA。
    返回：JSON 可序列化的原始浓度预测字典，含窗口秒数和原始单位标记。
    先按 checkpoint 的 mode 构造因果特征、复用训练统计量标准化，再反标准化
    网络输出；不截断负预测或超训练范围结果，以免掩盖模型误差。
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
    model = GasTCN(int(checkpoint["architecture"]["in_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    with torch.no_grad():
        prediction_z = model(torch.from_numpy(features).unsqueeze(0).to(device)).cpu().numpy()[0]
    prediction = prediction_z * np.asarray(checkpoint["label_std"]) + np.asarray(checkpoint["label_mean"])
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if args.device not in {"cpu", "cuda", "cuda:0"}:
        raise ValueError("device must be cpu, cuda, or cuda:0")
    result = predict(args.checkpoint, args.csv, torch.device(args.device))
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
