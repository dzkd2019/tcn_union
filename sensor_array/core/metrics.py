"""留一实验交叉验证与曲线误差指标。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import GASES, Experiment


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


def logo(
    experiments: list[Experiment],
    fit_predict,
    name: str,
) -> pd.DataFrame:
    """留一实验交叉验证。

    `fit_predict(train_experiments, test_experiment)` 返回测试实验的 (T,3)
    相对响应预测。每条曲线是一个样本，因此按实验留一，而不是随机拆时间点。
    """
    rows = []
    predictions = {}
    for index, test in enumerate(experiments):
        train = [item for i, item in enumerate(experiments) if i != index]
        prediction = fit_predict(train, test)
        predictions[test.file_id] = prediction
        scale = np.sqrt(np.mean(np.stack([item.relative for item in train]) ** 2, axis=(0, 1)))
        row = {"file": test.file_id, "model": name}
        row.update(curve_metrics(prediction, test.relative, scale))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["predictions"] = predictions
    return frame


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """按模型与气体汇总 nRMSE：均值、标准差、中位数与 90 分位。"""
    records = []
    for model, group in results.groupby("model", sort=False):
        for gas in GASES:
            values = group[f"nrmse_{gas}"]
            records.append(
                {
                    "model": model,
                    "gas": gas,
                    "nrmse_mean": values.mean(),
                    "nrmse_std": values.std(),
                    "nrmse_median": values.median(),
                    "nrmse_p90": values.quantile(0.9),
                    "r2_median": group[f"r2_{gas}"].median(),
                }
            )
        records.append(
            {
                "model": model,
                "gas": "macro",
                "nrmse_mean": group["nrmse_macro"].mean(),
                "nrmse_std": group["nrmse_macro"].std(),
                "nrmse_median": group["nrmse_macro"].median(),
                "nrmse_p90": group["nrmse_macro"].quantile(0.9),
                "r2_median": np.nan,
            }
        )
    return pd.DataFrame(records)
