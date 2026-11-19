"""曲线误差指标。"""
from __future__ import annotations

import numpy as np


def curve_metrics(prediction: np.ndarray, truth: np.ndarray, scale: np.ndarray) -> dict[str, float]:
    """单条实验、三路通道的曲线指标。

    `prediction`/`truth` 形状 (T,3)，`scale` 形状 (3,) 为训练集各通道相对
    响应的均方根。nRMSE 用训练集尺度归一化，避免小幅度通道（H2）的误差被
    绝对数值掩盖。
    """
    error = prediction - truth
    rmse = np.sqrt(np.mean(error**2, axis=0))
    total = np.sum((truth - truth.mean(axis=0)) ** 2, axis=0)
    r2 = 1.0 - np.sum(error**2, axis=0) / np.maximum(total, 1e-12)
    end_error = (prediction[-1] - truth[-1]) / np.maximum(np.abs(truth[-1]), 1e-9)
    return {
        "nrmse_H2": rmse[0] / scale[0],
        "nrmse_NH3": rmse[1] / scale[1],
        "nrmse_NO2": rmse[2] / scale[2],
        "r2_H2": r2[0],
        "r2_NH3": r2[1],
        "r2_NO2": r2[2],
        "end_err_H2": end_error[0],
        "end_err_NH3": end_error[1],
        "end_err_NO2": end_error[2],
        "nrmse_macro": float(np.mean(rmse / scale)),
    }
