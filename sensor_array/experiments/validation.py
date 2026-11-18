"""更严格的验证协议：留一浓度水平、留一采集块与下游浓度误差。

曲线协议对全部六个模型运行；留一实验（LOGO）结果直接读取
artifacts/logo/logo_metrics.csv，不重复计算 TCN。下游协议只对灰盒、
混合与直接岭回归三个可估模型运行：每折拟合一次，在 5/10/20 s 窗上做 MAP
浓度估计，记录中位绝对相对误差、p90 与 Laplace 90% 覆盖率。

用法：
    .venv/bin/python -m sensor_array.experiments.validation
    .venv/bin/python -m sensor_array.experiments.validation --part curve
    .venv/bin/python -m sensor_array.experiments.validation --part downstream
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from ..analysis.estimation import _start_points, concentration_bounds, map_estimate
from ..analysis.identifiability import estimate_noise
from ..core.data import GASES, load_experiments
from ..core.grey_box import fit_experiment_shapes
from ..core.metrics import block_folds, cross_validate, level_folds, summarize
from ..paths import LOGO, VALIDATION, ensure
from .logo import _shape_by_file, _subset_init, fit_model

CURVE_MODELS = ["mean_curve", "two_stage", "grey_box", "hybrid", "direct_ridge", "tcn"]
DOWNSTREAM_MODELS = ["grey_box", "hybrid", "direct_ridge"]
WINDOWS = (5.0, 10.0, 20.0)
Z90 = 1.6448536269514722


def _protocol_folds(experiments):
    """三种协议的折定义：单条留出、整档留出、连续块留出。"""
    return {
        "logo": [(str(experiment.file_id), np.array([index])) for index, experiment in enumerate(experiments)],
        "level": level_folds(experiments),
        "block": block_folds(experiments),
    }


def _curve_fit_predict(name, experiments, shapes):
    """曲线预测闭包；同一训练折只拟合一次。"""
    shape_by_file = _shape_by_file(experiments, shapes)
    cache = {"key": None, "model": None}

    def fit_predict(train, test):
        key = tuple(experiment.file_id for experiment in train)
        if key != cache["key"]:
            cache["key"] = key
            cache["model"] = fit_model(name, train, _subset_init(train, shape_by_file))
        return cache["model"].predict_relative(test.concentration[None], test.time)[0]

    return fit_predict


def _fold_fit(name, experiments, shapes):
    """返回按训练折缓存的 (model, sigma_model)，供下游估计使用。"""
    shape_by_file = _shape_by_file(experiments, shapes)
    cache = {"key": None, "model": None, "sigma": None}

    def fit(train):
        key = tuple(experiment.file_id for experiment in train)
        if key != cache["key"]:
            model = fit_model(name, train, _subset_init(train, shape_by_file))
            relative = np.stack([experiment.relative for experiment in train])
            concentration = np.stack([experiment.concentration for experiment in train])
            residual = relative - model.predict_relative(concentration, train[0].time)
            cache.update(
                key=key,
                model=model,
                sigma=np.sqrt(np.mean(residual**2, axis=(0, 1))),
            )
        return cache["model"], cache["sigma"]

    return fit


def curve_protocols(experiments, shapes):
    """LOGO 读已有产物，level/block 现场跑；返回三协议合并的逐文件指标。"""
    logo_metrics = pd.read_csv(LOGO / "logo_metrics.csv")
    logo_metrics.insert(0, "protocol", "logo")
    level_frames = []
    block_frames = []
    for name in CURVE_MODELS:
        fit_predict = _curve_fit_predict(name, experiments, shapes)
        level_frames.append(
            cross_validate(experiments, fit_predict, name, level_folds(experiments))
        )
        block_frames.append(
            cross_validate(experiments, fit_predict, name, block_folds(experiments))
        )
        print(f"曲线协议完成：{name}", flush=True)
    level = pd.concat(level_frames, ignore_index=True)
    block = pd.concat(block_frames, ignore_index=True)
    level.insert(0, "protocol", "level")
    block.insert(0, "protocol", "block")
    return logo_metrics, level, block


def summarize_curve(*frames):
    """逐协议汇总，并保留 summarize 的 macro 行。"""
    parts = []
    for frame in frames:
        table = summarize(frame)
        table.insert(0, "protocol", frame["protocol"].iloc[0])
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def downstream_estimates(experiments, shapes, noise):
    """在三协议上估计浓度，返回逐文件 × 窗口 × 气体的误差表。"""
    low, high = concentration_bounds(experiments)
    starts = _start_points(low, high)
    rows = []
    for protocol, folds in _protocol_folds(experiments).items():
        for name in DOWNSTREAM_MODELS:
            fit = _fold_fit(name, experiments, shapes)
            for label, test_indices in folds:
                test_set = {int(index) for index in test_indices}
                train = [experiment for index, experiment in enumerate(experiments) if index not in test_set]
                model, sigma_model = fit(train)
                sigma_eff = np.sqrt(noise**2 + sigma_model**2)
                for index in test_indices:
                    test = experiments[int(index)]
                    theta_true = np.log(test.concentration)
                    for tmax in WINDOWS:
                        mask = test.time <= tmax
                        result = map_estimate(
                            model,
                            test.relative[mask],
                            test.time[mask],
                            sigma_eff,
                            low,
                            high,
                            starts=starts,
                        )
                        for gas_index, gas in enumerate(GASES):
                            error = float(result["theta"][gas_index] - theta_true[gas_index])
                            rows.append(
                                {
                                    "protocol": protocol,
                                    "fold": label,
                                    "model": name,
                                    "file": test.file_id,
                                    "tmax": tmax,
                                    "gas": gas,
                                    "theta_hat": float(result["theta"][gas_index]),
                                    "theta_true": float(theta_true[gas_index]),
                                    "se": float(result["se"][gas_index]),
                                    "abs_rel_error": float(abs(np.exp(error) - 1.0)),
                                    "covered": bool(abs(error) <= Z90 * float(result["se"][gas_index])),
                                    "success": bool(result["success"]),
                                }
                            )
            print(f"下游协议完成：{protocol} / {name}", flush=True)
    return pd.DataFrame(rows)


def summarize_estimates(frame):
    """按协议/模型/窗口/气体汇总下游指标。"""
    return (
        frame.groupby(["protocol", "model", "tmax", "gas"], sort=False)
        .agg(
            n=("file", "nunique"),
            median_abs_rel_error=("abs_rel_error", "median"),
            p90_abs_rel_error=("abs_rel_error", lambda values: values.quantile(0.9)),
            coverage_laplace90=("covered", "mean"),
        )
        .reset_index()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="留一档位/采集块与下游浓度验证")
    parser.add_argument("--part", choices=("all", "curve", "downstream"), default="all")
    arguments = parser.parse_args()

    ensure()
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)

    if arguments.part in ("all", "curve"):
        logo, level, block = curve_protocols(experiments, shapes)
        level.to_csv(VALIDATION / "level_out.csv", index=False, encoding="utf-8")
        block.to_csv(VALIDATION / "block_out.csv", index=False, encoding="utf-8")
        summary = summarize_curve(logo, level, block)
        summary.to_csv(VALIDATION / "curve_summary.csv", index=False, encoding="utf-8")
        macro = summary[summary["gas"] == "macro"]
        print("\n宏平均 nRMSE（均值）")
        print(macro.pivot(index="model", columns="protocol", values="nrmse_mean").round(4).to_string())

    if arguments.part in ("all", "downstream"):
        noise = estimate_noise(experiments)
        estimates = downstream_estimates(experiments, shapes, noise)
        estimates.to_csv(VALIDATION / "estimates.csv", index=False, encoding="utf-8")
        summary = summarize_estimates(estimates)
        summary.to_csv(VALIDATION / "estimate_summary.csv", index=False, encoding="utf-8")
        ten = summary[(summary["tmax"] == 10.0) & (summary["gas"] != "macro")]
        print("\n10 s 中位绝对相对误差")
        print(ten.pivot_table(index="model", columns=["gas", "protocol"], values="median_abs_rel_error").round(4).to_string())


if __name__ == "__main__":
    main()
