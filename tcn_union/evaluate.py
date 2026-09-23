"""在锁定测试集评估 TCN，并与简单预测基线作逐实验比较。

验证折只用于 Ridge 正则强度选择；最终 Ridge 在开发集上重拟合。该模块读取
训练阶段预先选定的 TCN 模式，不通过测试结果调整模型。按实验汇总 MAE、
RMSE、NMAE、MAPE 等指标，避免把同一实验中的多个采样点误当独立样本。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .data import load_run, make_features
from .model import GasTCN

ALPHAS = [0.01, 0.1, 1.0, 10.0, 100.0]
AXES = ["H2", "NH3", "NO2"]


def _manifest(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """读取并校验锁定的数据划分及其原始数据版本。

    参数：`path` 为 manifest CSV。
    返回：`(development_rows,test_rows)`。校验 50/12 划分、每个文件哈希以及
    manifest 浓度标签与 CSV 标签一致，保证评估用的测试文件没有被替换。
    """
    rows = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8").to_dict(orient="records")
    if len(rows) != 62 or sum(row["split"] == "development" for row in rows) != 50 or sum(row["split"] == "test" for row in rows) != 12:
        raise ValueError("evaluation requires the locked 50-development/12-test manifest")
    for row in rows:
        file_path = Path(row["file"])
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"manifest hash mismatch: {row['file']}")
        _, _, labels = load_run(file_path)
        expected = np.array([float(row[a]) for a in AXES])
        if not np.array_equal(labels, expected):
            raise ValueError(f"manifest concentration mismatch: {row['file']}")
        row["_labels"] = labels
    return [r for r in rows if r["split"] == "development"], [r for r in rows if r["split"] == "test"]


def _load_features(rows: list[dict[str, Any]], seconds: float, mode: str) -> dict[str, np.ndarray]:
    """为给定实验列表按训练时协议生成模型输入特征。

    参数：`rows` 是带文件路径的 manifest 记录；`seconds` 是前缀窗口；
    `mode` 是训练时选择的 raw_delta 或 delta。
    返回：文件路径到 `[channels,samples]` 特征的映射；测试曲线也只截取指定
    早期前缀，保证与训练输入定义相同。
    """
    features: dict[str, np.ndarray] = {}
    for row in rows:
        time, response, _ = load_run(Path(row["file"]))
        features[row["file"]] = make_features(time, response, seconds, mode)
    return features


def _ridge_fit(x_train: np.ndarray, y_train: np.ndarray, alpha: float) -> dict[str, np.ndarray]:
    """拟合带截距的多输出 Ridge 回归。

    参数：`x_train` 为 `[runs,features]`；`y_train` 为 `[runs,3]`；
    `alpha` 是 L2 惩罚强度。
    返回：输入/标签均值与标准差及回归系数。先按训练数据逐列标准化，避免
    特征尺度影响惩罚；截距不惩罚，让回归可以表示总体偏移。
    """
    x_mean = np.mean(x_train, axis=0)
    x_std = np.std(x_train, axis=0)
    x_std[x_std < 1e-8] = 1.0
    y_mean = np.mean(y_train, axis=0)
    y_std = np.std(y_train, axis=0)
    y_std[y_std < 1e-8] = 1.0
    xz = (x_train - x_mean) / x_std
    yz = (y_train - y_mean) / y_std
    design = np.column_stack((np.ones(len(xz)), xz))
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ yz)
    return {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std, "beta": beta}


def _ridge_predict(model: dict[str, np.ndarray], x: np.ndarray) -> np.ndarray:
    """用拟合好的 Ridge 参数预测并还原标签原始单位。

    参数：`model` 是 `_ridge_fit` 返回的参数字典；`x` 是一个或多个特征行。
    返回：`[runs,3]` 的原始单位浓度预测。
    """
    xz = (x - model["x_mean"]) / model["x_std"]
    design = np.column_stack((np.ones(len(xz)), xz))
    return (design @ model["beta"]) * model["y_std"] + model["y_mean"]


def _macro_nmae(truth: np.ndarray, prediction: np.ndarray, ranges: np.ndarray) -> float:
    """计算各气体 MAE 按开发集标签范围归一化后的宏平均。

    参数：`truth/prediction` 为 `[runs,3]`；`ranges` 是开发集逐气体浓度极差。
    返回：标量 macro-NMAE。极差为零时指标未定义，因此抛出错误。
    """
    if np.any(ranges == 0):
        raise ValueError("development label range is zero; normalized error is undefined")
    return float(np.mean(np.mean(np.abs(prediction - truth), axis=0) / ranges))


def _select_ridge_alpha(rows: list[dict[str, Any]], features: dict[str, np.ndarray]) -> tuple[float, dict[float, list[float]]]:
    """用固定五折选择 Ridge 正则系数。

    参数：`rows` 是开发集 manifest 行（含 fold 和标签）；`features` 将文件路径
    映射到特征向量。
    返回：`(最佳 alpha, 每个候选 alpha 的逐折 macro-NMAE)`。所有候选只在开发
    折内比较，同分时选择更大的 alpha，作为较强正则化的确定性平局规则。
    """
    scores: dict[float, list[float]] = {alpha: [] for alpha in ALPHAS}
    for fold in range(5):
        train_rows = [row for row in rows if int(row["fold"]) != fold]
        val_rows = [row for row in rows if int(row["fold"]) == fold]
        train_x = np.stack([features[row["file"]] for row in train_rows])
        val_x = np.stack([features[row["file"]] for row in val_rows])
        train_y = np.stack([row["_labels"] for row in train_rows])
        val_y = np.stack([row["_labels"] for row in val_rows])
        ranges = np.ptp(train_y, axis=0)
        if np.any(ranges == 0):
            raise ValueError(f"fold {fold} has zero training label range for Ridge selection")
        for alpha in ALPHAS:
            fit = _ridge_fit(train_x, train_y, alpha)
            pred = _ridge_predict(fit, val_x)
            scores[alpha].append(_macro_nmae(val_y, pred, ranges))
    means = {alpha: float(np.mean(value)) for alpha, value in scores.items()}
    chosen = min(ALPHAS, key=lambda alpha: (means[alpha], -alpha))
    return chosen, scores


def _early_features(time: np.ndarray, response: np.ndarray, seconds: float) -> np.ndarray:
    """提取早期 Ridge 基线的 9 个手工特征。

    参数：`time` 是采样时间；`response[N,3]` 是三路响应；`seconds` 是 5/10 秒
    终点。
    返回：float64 向量，依次包含三路 t=0 初值、终点相对初值增量、以及每路
    前缀对时间最小二乘斜率。初值只量化静态信息，增量和斜率给出简单动态
    信息，用来判断 TCN 是否超出这些简单基线的能力。
    """
    mask = time <= seconds + 1e-6
    t = time[mask]
    values = response[mask]
    if t.size != int(seconds / 0.2) + 1 or abs(t[-1] - seconds) > 1e-6:
        raise ValueError(f"{seconds} s prefix unavailable for Ridge feature extraction")
    baseline = values[0]
    centered_t = t - np.mean(t)
    slopes = centered_t @ (values - np.mean(values, axis=0)) / np.sum(centered_t**2)
    return np.concatenate((baseline, values[-1] - baseline, slopes)).astype(np.float64)


def _metric_summary(truth: np.ndarray, prediction: np.ndarray, dev_ranges: np.ndarray) -> dict[str, Any]:
    """汇总三个气体的误差、归一化误差及百分比误差。

    参数：`truth/prediction` 为原始单位的 `[runs,3]` 矩阵；`dev_ranges` 是开发
    集逐气体浓度极差。
    返回：含逐气体 MAE、RMSE、R²、NMAE、NRMSE、MAPE 及 macro 均值的字典。
    MAPE 按每个真实值归一化，低浓度误差相对权重较大；任何真实值为 0 时该
    气体 MAPE 返回 null 和原因，避免除零或掩盖未定义情形。
    """
    output: dict[str, Any] = {}
    axis_nmae: list[float] = []
    axis_mape: list[float | None] = []
    for index, axis in enumerate(AXES):
        error = prediction[:, index] - truth[:, index]
        mae = float(np.mean(np.abs(error)))
        rmse = float(np.sqrt(np.mean(error**2)))
        variance = float(np.sum((truth[:, index] - np.mean(truth[:, index])) ** 2))
        if variance == 0.0:
            r2: float | None = None
            r2_reason = "test target variance is zero"
        else:
            r2 = float(1.0 - np.sum(error**2) / variance)
            r2_reason = None
        denominator = np.abs(truth[:, index])
        if np.any(denominator == 0):
            mape: float | None = None
            mape_reason = "MAPE is undefined because at least one target is zero"
        else:
            mape = float(100.0 * np.mean(np.abs(error) / denominator))
            mape_reason = None
        nmae = mae / dev_ranges[index]
        axis_nmae.append(nmae)
        axis_mape.append(mape)
        output[axis] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "R2_null_reason": r2_reason,
            "NMAE": nmae,
            "NRMSE": rmse / dev_ranges[index],
            "MAPE_percent": mape,
            "MAPE_null_reason": mape_reason,
        }
    valid_mape = [value for value in axis_mape if value is not None]
    if len(valid_mape) == len(AXES):
        macro_mape: float | None = float(np.mean(valid_mape))
        macro_mape_reason = None
    else:
        macro_mape = None
        macro_mape_reason = "macro MAPE is undefined because at least one gas has a zero target"
    output["macro_NMAE"] = float(np.mean(axis_nmae))
    output["macro_MAPE_percent"] = macro_mape
    output["macro_MAPE_null_reason"] = macro_mape_reason
    return output


def _bootstrap_difference(
    truth: np.ndarray, model_prediction: np.ndarray, baseline_prediction: np.ndarray, dev_ranges: np.ndarray
) -> dict[str, Any]:
    """对同一批实验的模型和基线误差差值做配对 bootstrap。

    参数：`truth`、两组预测均为 `[runs,3]`；`dev_ranges` 为开发集气体极差。
    返回：macro-NMAE 差（模型减基线）的均值和 2000 次重采样 95% 百分位区间。
    以实验为单位成对重采样，保留模型/基线在每个实验上的对应关系；不把
    时间采样点当独立样本。
    """
    per_run = np.mean(np.abs(model_prediction - truth) / dev_ranges[None, :], axis=1) - np.mean(
        np.abs(baseline_prediction - truth) / dev_ranges[None, :], axis=1
    )
    generator = np.random.default_rng(20260923)
    indices = generator.integers(0, len(truth), size=(2000, len(truth)))
    differences = per_run[indices].mean(axis=1)
    return {
        "mean_difference_model_minus_baseline": float(per_run.mean()),
        "ci95_percentile": [float(np.percentile(differences, 2.5)), float(np.percentile(differences, 97.5))],
        "bootstrap_samples": 2000,
        "resampling_unit": "experiment",
        "seed": 20260923,
    }


def _tcn_predict(checkpoint_path: Path, features: np.ndarray, device: torch.device) -> tuple[np.ndarray, dict[str, Any]]:
    """按 checkpoint 统计量运行 TCN 前向推理并还原浓度单位。

    参数：`checkpoint_path` 是最终模型文件；`features` 为
    `[runs,channels,time]` 特征；`device` 是计算设备。
    返回：`(预测[runs,3], checkpoint内容)`。输入使用训练保存的均值/标准差，
    输出使用标签统计量反标准化；eval/no_grad 确保推理时 dropout 关闭且不建梯度。
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    x = (features - np.asarray(checkpoint["input_mean"], dtype=np.float32)[None, :, None]) / np.asarray(
        checkpoint["input_std"], dtype=np.float32
    )[None, :, None]
    model = GasTCN(int(checkpoint["architecture"]["in_channels"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    label_std = np.asarray(checkpoint["label_std"])
    label_mean = np.asarray(checkpoint["label_mean"])
    with torch.no_grad():
        output = np.stack(
            [
                model(torch.from_numpy(sample).unsqueeze(0).to(device)).cpu().numpy()[0] * label_std + label_mean
                for sample in x
            ]
        )
    return output, checkpoint


def evaluate(manifest_path: Path, model_dir: Path, output_dir: Path, device: torch.device) -> dict[str, Any]:
    """读取训练结果，评估 OOF、基线和锁定测试集并写出报告。

    参数：`manifest_path` 是实验划分；`model_dir` 含 selection、OOF 与 TCN
    checkpoint；`output_dir` 是评估产物目录；`device` 指定推理设备。
    返回：含锁定测试指标、逐实验预测、OOF 模式比较、Ridge 选择结果和配对
    bootstrap 区间的报告字典。开发集只用于基线拟合/正则选择及误差归一化；
    测试集用于最终评价，不反馈到训练或模式选择。
    """
    selection_path = model_dir / "selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"training selection not found: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    dev_rows, test_rows = _manifest(manifest_path)
    y_dev = np.stack([row["_labels"] for row in dev_rows])
    y_test = np.stack([row["_labels"] for row in test_rows])
    dev_ranges = np.ptp(y_dev, axis=0)
    if np.any(dev_ranges == 0):
        raise ValueError("development concentration range is zero")
    output_dir.mkdir(parents=True, exist_ok=True)

    oof_comparison: list[dict[str, Any]] = []
    for seconds in (5, 10):
        for mode in ("raw_delta", "delta"):
            oof_path = model_dir / "oof" / f"{seconds}s_{mode}.csv"
            oof_rows = pd.read_csv(oof_path, dtype=str, keep_default_na=False, encoding="utf-8").to_dict(
                orient="records"
            )
            if len(oof_rows) != len(dev_rows):
                raise ValueError(f"OOF file {oof_path} does not contain 50 unique development runs")
            truth = np.array([[float(row[f"{axis}_true"]) for axis in AXES] for row in oof_rows])
            prediction = np.array([[float(row[f"{axis}_pred"]) for axis in AXES] for row in oof_rows])
            metrics = _metric_summary(truth, prediction, dev_ranges)
            oof_comparison.append({"seconds": seconds, "mode": mode, "n": len(oof_rows), **metrics})

    baseline_predictions: dict[str, dict[int, np.ndarray]] = {"mean": {}, "t0_ridge": {}, "early_ridge": {}}
    ridge_selection: dict[str, Any] = {}
    mean_prediction = np.mean(y_dev, axis=0)
    for seconds in (5, 10):
        # Ridge input based on only t=0 sensor responses.
        t0_features: dict[str, np.ndarray] = {}
        early_features: dict[str, np.ndarray] = {}
        for row in dev_rows + test_rows:
            t, response, _ = load_run(Path(row["file"]))
            t0_features[row["file"]] = response[0].astype(np.float64)
            early_features[row["file"]] = _early_features(t, response, float(seconds))
        t0_alpha, t0_scores = _select_ridge_alpha(dev_rows, t0_features)
        early_alpha, early_scores = _select_ridge_alpha(dev_rows, early_features)
        t0_fit = _ridge_fit(
            np.stack([t0_features[row["file"]] for row in dev_rows]), y_dev, t0_alpha
        )
        early_fit = _ridge_fit(
            np.stack([early_features[row["file"]] for row in dev_rows]), y_dev, early_alpha
        )
        test_t0 = np.stack([t0_features[row["file"]] for row in test_rows])
        test_early = np.stack([early_features[row["file"]] for row in test_rows])
        baseline_predictions["mean"][seconds] = np.repeat(mean_prediction[None, :], len(test_rows), axis=0)
        baseline_predictions["t0_ridge"][seconds] = _ridge_predict(t0_fit, test_t0)
        baseline_predictions["early_ridge"][seconds] = _ridge_predict(early_fit, test_early)
        ridge_selection[str(seconds)] = {
            "t0_alpha": t0_alpha,
            "t0_fold_macro_nmae": {str(a): values for a, values in t0_scores.items()},
            "early_alpha": early_alpha,
            "early_fold_macro_nmae": {str(a): values for a, values in early_scores.items()},
        }

    predictions: dict[tuple[int, str], np.ndarray] = {}
    prediction_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    baseline_names = {"mean": "label_mean", "t0_ridge": "ridge_t0", "early_ridge": "ridge_early"}
    for seconds in (5, 10):
        for baseline, by_seconds in baseline_predictions.items():
            pred = by_seconds[seconds]
            predictions[(seconds, baseline)] = pred
            summary_rows.append(
                {
                    "window_seconds": seconds,
                    "model": baseline_names[baseline],
                    "seed": "",
                    "n": len(test_rows),
                    **_metric_summary(y_test, pred, dev_ranges),
                }
            )
            for row, truth, value in zip(test_rows, y_test, pred, strict=True):
                prediction_rows.append(_prediction_record(seconds, baseline_names[baseline], "", row, truth, value))

    ci_results: dict[str, Any] = {}
    for seconds in (5, 10):
        key = f"{seconds}s"
        chosen_mode = selection["selected"][key]["selected_mode"]
        feature_map = _load_features(test_rows, float(seconds), chosen_mode)
        test_x = np.stack([feature_map[row["file"]] for row in test_rows])
        tcn_predictions: dict[int, np.ndarray] = {}
        for seed in (20260923, 20260924, 20260925):
            checkpoint_path = model_dir / key / f"seed_{seed}.pt"
            pred, checkpoint = _tcn_predict(checkpoint_path, test_x, device)
            if checkpoint["mode"] != chosen_mode or float(checkpoint["seconds"]) != float(seconds):
                raise ValueError(f"checkpoint protocol mismatch: {checkpoint_path}")
            tcn_predictions[seed] = pred
            summary_rows.append(
                {
                    "window_seconds": seconds,
                    "model": "TCN",
                    "seed": seed,
                    "n": len(test_rows),
                    **_metric_summary(y_test, pred, dev_ranges),
                }
            )
            for row, truth, value in zip(test_rows, y_test, pred, strict=True):
                prediction_rows.append(_prediction_record(seconds, "TCN", seed, row, truth, value))
        predictions[(seconds, "tcn_primary")] = tcn_predictions[20260923]
        ci_results[key] = {
            baseline_names[baseline]: _bootstrap_difference(
                y_test, tcn_predictions[20260923], baseline_predictions[baseline][seconds], dev_ranges
            )
            for baseline in ("mean", "t0_ridge", "early_ridge")
        }

    summary_path = output_dir / "test_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fields = ["window_seconds", "model", "seed", "n", "H2", "NH3", "NO2", "macro_NMAE", "macro_MAPE_percent"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({key: json.dumps(row[key], ensure_ascii=False) if key in AXES else row[key] for key in fields})
    predictions_path = output_dir / "test_predictions.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    oof_path = output_dir / "oof_mode_comparison.json"
    oof_path.write_text(json.dumps(oof_comparison, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report = {
        "locked_test_runs": [row["file"] for row in test_rows],
        "test_n": len(test_rows),
        "development_ranges": dict(zip(AXES, dev_ranges.tolist(), strict=True)),
        "ridge_selection": ridge_selection,
        "oof_mode_comparison": oof_comparison,
        "test_metrics": summary_rows,
        "paired_bootstrap": ci_results,
        "collection_window_reduction_percent": {"5s": 100 * (1 - 5 / 59.6), "10s": 100 * (1 - 10 / 59.6)},
        "caveat": "12 experiment-level test runs; bootstrap intervals may be wide, especially the single H2=32000 run.",
    }
    report_path = output_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def _prediction_record(
    seconds: int, model_name: str, seed: int | str, row: dict[str, Any], truth: np.ndarray, prediction: np.ndarray
) -> dict[str, Any]:
    """组装一条实验的一项预测记录及其逐气体误差。

    参数：`seconds` 是预测窗口；`model_name/seed` 标识模型；`row` 是实验清单
    行；`truth/prediction` 是该实验的三气体标签与预测。
    返回：适合 CSV 输出的字典，含原值、误差和逐气体绝对百分比误差；真实
    浓度为零时 APE 写为 null，因为分母为零而不定义。
    """
    result: dict[str, Any] = {
        "window_seconds": seconds,
        "model": model_name,
        "seed": seed,
        "file": row["file"],
    }
    for index, axis in enumerate(AXES):
        error = float(prediction[index] - truth[index])
        result[f"{axis}_true"] = float(truth[index])
        result[f"{axis}_pred"] = float(prediction[index])
        result[f"{axis}_error"] = error
        result[f"{axis}_APE_percent"] = None if truth[index] == 0 else 100.0 * abs(error) / abs(float(truth[index]))
    return result


def main() -> None:
    """解析评估 CLI，检查设备后运行锁定测试集评估。"""
    parser = argparse.ArgumentParser(description="Evaluate locked test runs and simple baselines")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
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
    report = evaluate(args.manifest, args.model_dir, args.output_dir, device)
    print(f"Wrote {args.output_dir / 'evaluation.json'} for {report['test_n']} locked test runs")


if __name__ == "__main__":
    main()
