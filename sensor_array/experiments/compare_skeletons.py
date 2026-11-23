"""四种响应骨架的 62 折留一对比。

骨架只决定正模型的形状参数化；每种骨架都接同一个三路独立 MLP 残差，
因此曲线误差差异来自骨架，而不是网络结构。默认骨架是对照的现有双指数，
另外三个是计划中的一阶、临界阻尼二阶、两个二阶系统并联。

输出 artifacts/comparison/ 下的逐折指标、汇总、预测数组和自动 Markdown 摘要
（comparison_auto.md，不覆盖人工整理的 comparison_summary.md）；
folds=62 时另把每个骨架的全量模型存到 artifacts/models/comparison/。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ..core.data import GASES, Experiment, load_experiments
from ..core.double_exponential import DoubleExponentialArrayModel
from ..core.metrics import curve_metrics
from ..core.residual import PerGasMLPResidualModel
from ..core.shapes import fit_experiment_shapes
from ..core.skeletons import (
    DoubleSecondOrderArrayModel,
    FirstOrderArrayModel,
    SecondOrderArrayModel,
)
from ..paths import COMPARISON, COMPARISON_MODELS, ensure

SKELETONS = {
    "double": (DoubleExponentialArrayModel, 69, "双指数（当前）"),
    "first": (FirstOrderArrayModel, 45, "一阶"),
    "second": (SecondOrderArrayModel, 45, "二阶"),
    "second2": (DoubleSecondOrderArrayModel, 69, "两个二阶并联"),
}
DEFAULT_MODELS = ",".join(SKELETONS)


def _parse_list(value: str, cast=str) -> list:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("列表不能为空")
    return [cast(item) for item in items]


def _skeleton(key: str, max_nfev: int):
    model_class = SKELETONS[key][0]
    return model_class(max_nfev=max_nfev)


def _curve_row(
    test: Experiment,
    model: str,
    seed: int,
    prediction: np.ndarray,
    scale: np.ndarray,
) -> dict:
    row = {"model": model, "seed": seed, "file": test.file_id}
    row.update(curve_metrics(prediction, test.relative, scale))
    return row


def _summarize(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, seed), group in curve.groupby(["model", "seed"], sort=False):
        for gas in (*GASES, "macro"):
            column = "nrmse_macro" if gas == "macro" else f"nrmse_{gas}"
            values = group[column]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "gas": gas,
                    "nrmse_mean": values.mean(),
                    "nrmse_std": values.std(),
                    "nrmse_median": values.median(),
                    "nrmse_p90": values.quantile(0.9),
                }
            )
    return pd.DataFrame(rows)


def _markdown(
    summary: pd.DataFrame,
    models: list[str],
    seeds: list[int],
    folds: int,
    elapsed: float,
) -> str:
    macro = summary[summary.gas == "macro"]
    macro_mean = macro.groupby("model")["nrmse_mean"].mean()
    macro_std = macro.groupby("model")["nrmse_mean"].std().fillna(0.0)
    pivot = (
        summary.groupby(["model", "gas"])["nrmse_mean"].mean().unstack()
    )
    ranking = macro_mean.sort_values()
    lines = [
        "# 响应骨架对比（62 折留一）",
        "",
        f"- 数据：62 条实验，本次评估 {folds} 折；每折用其余 61 条训练。",
        f"- 种子：{", ".join(str(seed) for seed in seeds)}；每个种子只改变三路 MLP 的初始化。",
        "- 网络：三路独立 MLP，输入 7 维（三个对数浓度、归一化时间、骨架三路预测），结构 7-64-64-1，每路独立。",
        "- 指标：曲线 nRMSE，分母为训练折各通道相对响应的均方根；表中已乘 100 显示为百分比。",
        "",
        "| 模型 | 参数数 | 宏 nRMSE 均值/% | 种子标准差/% | H2/% | NH3/% | NO2/% |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ranking.index:
        label = SKELETONS[model][2]
        lines.append(
            f"| {label} | {SKELETONS[model][1]} | {macro_mean[model] * 100:.3f} | "
            f"{macro_std[model] * 100:.3f} | {pivot.loc[model, 'H2'] * 100:.3f} | "
            f"{pivot.loc[model, 'NH3'] * 100:.3f} | {pivot.loc[model, 'NO2'] * 100:.3f} |"
        )
    lines.extend(
        [
            "",
            f"按宏 nRMSE 均值排序，前两名是 {SKELETONS[ranking.index[0]][2]} "
            f"和 {SKELETONS[ranking.index[1]][2]}。",
            "",
            f"运行耗时：{elapsed:.1f} s。",
            "",
            "说明：这是 1 个或多个 MLP 种子的筛选结果，不是锁定测试的独立盲测；"
            "锁定 12 条没有参与本轮调参。人工整理的配对统计与取舍见 comparison_summary.md。",
        ]
    )
    return "\n".join(lines) + "\n"


def run(
    folds: int = 62,
    models: list[str] | None = None,
    seeds: list[int] | None = None,
    max_iter: int = 600,
    batch_size: int = 0,
    max_nfev: int = 600,
) -> None:
    ensure()
    models = models or list(SKELETONS)
    seeds = seeds or [0]
    for model in models:
        if model not in SKELETONS:
            raise ValueError(f"未知骨架：{model}")
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    curve_rows: list[dict] = []
    predictions: dict[str, dict[str, np.ndarray]] = {model: {} for model in models}
    started = time.time()

    for index, test in enumerate(experiments[:folds]):
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        shape_init = tuple(shapes[k][train_indices] for k in range(3))
        scale = np.sqrt(
            np.mean(np.stack([item.relative for item in train]) ** 2, axis=(0, 1))
        )
        for model in models:
            skeleton = _skeleton(model, max_nfev).fit(train, shape_init=shape_init)
            for seed in seeds:
                combined = PerGasMLPResidualModel(
                    epochs=max_iter,
                    batch_size=batch_size,
                    random_state=seed,
                ).fit(train, skeleton=skeleton)
                prediction = combined.predict_relative(
                    test.concentration[None, :], test.time
                )[0]
                curve_rows.append(
                    _curve_row(test, model, seed, prediction, scale)
                )
                predictions[model][f"s{seed}_file{test.file_id}"] = prediction
        elapsed = time.time() - started
        print(
            f"fold {index + 1}/{min(folds, len(experiments))} file {test.file_id} "
            f"({elapsed:.1f} s)",
            flush=True,
        )

    curve = pd.DataFrame(curve_rows)
    summary = _summarize(curve)
    curve.to_csv(COMPARISON / "skeleton_metrics.csv", index=False)
    summary.to_csv(COMPARISON / "skeleton_summary.csv", index=False)
    for model, arrays in predictions.items():
        np.savez_compressed(COMPARISON / f"predictions_{model}.npz", **arrays)
    elapsed = time.time() - started
    (COMPARISON / "comparison_auto.md").write_text(
        _markdown(summary, models, seeds, folds, elapsed), encoding="utf-8"
    )

    if folds >= len(experiments):
        for model in models:
            skeleton = _skeleton(model, max_nfev).fit(experiments, shape_init=shapes)
            for seed in seeds:
                combined = PerGasMLPResidualModel(
                    epochs=max_iter,
                    batch_size=batch_size,
                    random_state=seed,
                ).fit(experiments, skeleton=skeleton)
                combined.save(COMPARISON_MODELS / f"{model}_s{seed}.json")

    print("ranking by macro nRMSE (fraction):")
    print(
        summary[summary.gas == "macro"]
        .groupby("model")["nrmse_mean"]
        .mean()
        .sort_values()
        .to_string()
    )
    print("saved", COMPARISON, "and", COMPARISON_MODELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=62, help="留一折数；调试可用 3")
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=f"逗号分隔的骨架：{DEFAULT_MODELS}",
    )
    parser.add_argument(
        "--seeds",
        default="0",
        help="逗号分隔的 MLP 随机种子，例如 0,1,2",
    )
    parser.add_argument("--max-iter", type=int, default=600, help="三路独立 MLP 的训练轮数")
    parser.add_argument("--batch-size", type=int, default=0, help="0 表示全批")
    parser.add_argument("--max-nfev", type=int, default=600, help="骨架最小二乘的最大函数评估数")
    args = parser.parse_args()
    run(
        folds=args.folds,
        models=_parse_list(args.models),
        seeds=_parse_list(args.seeds, int),
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        max_nfev=args.max_nfev,
    )


if __name__ == "__main__":
    main()
