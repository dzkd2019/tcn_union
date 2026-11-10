"""用全部 62 组实验绘制相对误差散点图。

62 组 = 50 个开发集 OOF 预测 + 12 个锁定测试预测。
- 开发集使用折外预测，避免把训练集自身预测当成独立测试；
- 测试集使用三 seed 集成预测；
- 只用于展示误差分布，不重新计算“独立测试”指标。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tcn_union.visualization.plot_rmse_scatter import plot_error_sigma

PREDICTION_COLUMNS = [
    "H2_true",
    "NH3_true",
    "NO2_true",
    "H2_pred",
    "NH3_pred",
    "NO2_pred",
]


def combine_predictions(
    oof_path: Path,
    test_predictions_path: Path,
    model: str,
    window: int,
    seed: str,
    output_path: Path,
) -> pd.DataFrame:
    oof = pd.read_csv(oof_path)
    missing = set(PREDICTION_COLUMNS) - set(oof.columns)
    if missing:
        raise ValueError(f"OOF CSV 缺少列: {sorted(missing)}")
    oof = oof[["file", *PREDICTION_COLUMNS]].copy()
    oof["source"] = "OOF"

    test = pd.read_csv(test_predictions_path)
    required = {"model", "window_seconds", "seed", *PREDICTION_COLUMNS}
    missing = required - set(test.columns)
    if missing:
        raise ValueError(f"测试预测 CSV 缺少列: {sorted(missing)}")
    test = test[
        (test["model"] == model)
        & (test["window_seconds"].astype(int) == window)
        & (test["seed"].astype(str) == str(seed))
    ][["file", *PREDICTION_COLUMNS]].copy()
    test["source"] = "test"
    if test.empty:
        raise ValueError(f"没有匹配的测试预测: model={model}, window={window}, seed={seed}")

    combined = pd.concat((oof, test), ignore_index=True)
    if len(combined) != 62:
        raise ValueError(f"合并后应为 62 组，实际为 {len(combined)} 组")
    if combined["file"].duplicated().any():
        raise ValueError("合并后存在重复实验文件")
    combined["model"] = "all62"
    combined["window_seconds"] = int(window)
    combined["seed"] = "OOF+测试"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="用全部 62 组数据绘制相对误差散点图")
    parser.add_argument("--oof", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="candidate")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--seed", default="ensemble")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    stem = f"{args.prefix}_predictions.csv" if args.prefix else "all62_predictions.csv"
    combined_path = args.output_dir / stem
    combine_predictions(
        args.oof,
        args.test_predictions,
        args.model,
        args.window,
        str(args.seed),
        combined_path,
    )
    plot_error_sigma(
        combined_path,
        args.output_dir / "figures",
        "all62",
        args.window,
        "OOF+测试",
        prefix=args.prefix,
        title=f"全部 62 组相对误差分布（50 个开发集 OOF + 12 个锁定测试，{args.window} 秒）",
    )


if __name__ == "__main__":
    main()
