"""汇总 OOF 与锁定测试集的逐气体相对误差 3σ，用于方差优化实验对比。

口径与 ``docs/误差方差优化.md`` 一致：对每个气体，
``rel = 100 * (pred - true) / max(|true|, 1e-9)``，``3σ = 3 * std(rel, ddof=1)``，
宏平均为三气体 3σ 的算术平均。

OOF 可以传入同一方案多个 seed 的 ``oof.csv``，脚本先按实验平均预测再算 3σ，
使得 OOF 口径与部署端多 seed 集成一致。测试集默认取 ``candidate`` 的
``ensemble`` 行。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

GASES = ("H2", "NH3", "NO2")
PREDICTION_COLUMNS = [f"{gas}_pred" for gas in GASES]


def _relative(frame: pd.DataFrame, gas: str) -> np.ndarray:
    """返回单个气体的相对误差百分数。"""
    truth = frame[f"{gas}_true"].to_numpy(dtype=float)
    prediction = frame[f"{gas}_pred"].to_numpy(dtype=float)
    return 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)


def summarize(frame: pd.DataFrame) -> dict[str, object]:
    """计算逐气体 3σ、MAPE、平均偏差及宏平均。"""
    result: dict[str, object] = {"n": int(len(frame))}
    sigma: dict[str, float] = {}
    mape: dict[str, float] = {}
    mean_relative: dict[str, float] = {}
    for gas in GASES:
        relative = _relative(frame, gas)
        sigma[gas] = float(3.0 * np.std(relative, ddof=1))
        mape[gas] = float(np.mean(np.abs(relative)))
        mean_relative[gas] = float(np.mean(relative))
    result["three_sigma"] = sigma
    result["three_sigma_macro"] = float(np.mean(list(sigma.values())))
    result["mape"] = mape
    result["mape_macro"] = float(np.mean(list(mape.values())))
    result["mean_relative"] = mean_relative
    return result


def load_oof(paths: list[Path]) -> tuple[pd.DataFrame, int]:
    """读取一个或多个 seed 的 OOF 预测；多个时按实验平均预测。"""
    frames = [pd.read_csv(path).set_index("file") for path in paths]
    base = frames[0]
    for frame in frames[1:]:
        if set(frame.index) != set(base.index):
            raise ValueError("OOF 文件包含不同的实验集合")
    if len(frames) == 1:
        return base.copy(), 1
    prediction = np.mean(
        [frame.loc[base.index, PREDICTION_COLUMNS].to_numpy(dtype=float) for frame in frames],
        axis=0,
    )
    merged = base.copy()
    merged.loc[:, PREDICTION_COLUMNS] = prediction
    return merged, len(frames)


def load_test(
    path: Path, model: str = "candidate", window: int = 10, seed: str = "ensemble"
) -> pd.DataFrame:
    """读取测试集预测中的指定模型/窗口/种子行。"""
    frame = pd.read_csv(path)
    filtered = frame[
        (frame["model"] == model)
        & (frame["window_seconds"].astype(int) == window)
        & (frame["seed"].astype(str) == seed)
    ]
    if filtered.empty:
        raise ValueError(f"测试预测中没有 model={model} window={window} seed={seed} 的行")
    return filtered.reset_index(drop=True)


def _prediction_details(frame: pd.DataFrame) -> list[dict[str, object]]:
    """生成逐实验明细，便于定位 3σ 尾部。"""
    rows: list[dict[str, object]] = []
    for file_name, row in frame.iterrows():
        record: dict[str, object] = {"file": file_name}
        for gas in GASES:
            truth = float(row[f"{gas}_true"])
            prediction = float(row[f"{gas}_pred"])
            record[f"{gas}_true"] = truth
            record[f"{gas}_pred"] = prediction
            record[f"{gas}_relative"] = 100.0 * (prediction - truth) / max(abs(truth), 1e-9)
        rows.append(record)
    return rows


def run(
    label: str,
    oof_paths: list[Path],
    test_path: Path | None,
    output_dir: Path,
    test_model: str,
    test_window: int,
    test_seed: str,
) -> list[dict[str, object]]:
    """计算一个方案的 OOF/测试 3σ，返回扁平汇总行。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_frame, seed_count = load_oof(oof_paths)
    scopes = [("oof", oof_frame)]
    if test_path is not None:
        scopes.append(("test", load_test(test_path, test_model, test_window, test_seed)))

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {"label": label, "seeds": seed_count, "scopes": {}}
    for scope, frame in scopes:
        summary = summarize(frame)
        details["scopes"][scope] = summary
        for gas in (*GASES, "macro"):
            if gas == "macro":
                three_sigma = summary["three_sigma_macro"]
                mape = summary["mape_macro"]
                mean_relative = float(np.mean(list(summary["mean_relative"].values())))
            else:
                three_sigma = summary["three_sigma"][gas]
                mape = summary["mape"][gas]
                mean_relative = summary["mean_relative"][gas]
            rows.append(
                {
                    "label": label,
                    "scope": scope,
                    "gas": gas,
                    "n": summary["n"],
                    "seeds": seed_count,
                    "three_sigma": three_sigma,
                    "mape": mape,
                    "mean_relative": mean_relative,
                }
            )
        with (output_dir / f"{label}_{scope}_predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(_prediction_details(frame)[0]))
            writer.writeheader()
            writer.writerows(_prediction_details(frame))
    (output_dir / f"{label}_summary.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 OOF / 锁定测试集的相对误差 3σ")
    parser.add_argument("--label", required=True, help="方案名，用于输出文件名")
    parser.add_argument(
        "--oof",
        type=Path,
        nargs="+",
        required=True,
        help="一个或多个 seed 的 oof.csv；多个时按实验平均",
    )
    parser.add_argument("--test", type=Path, help="test_predictions.csv；缺省时只汇总 OOF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--test-model", default="candidate")
    parser.add_argument("--test-window", type=int, default=10)
    parser.add_argument("--test-seed", default="ensemble")
    args = parser.parse_args()

    rows = run(
        args.label,
        args.oof,
        args.test,
        args.output_dir,
        args.test_model,
        args.test_window,
        args.test_seed,
    )
    for row in rows:
        if row["gas"] == "macro":
            print(
                f"{row['label']:<24} {row['scope']:<4} macro n={row['n']:<3} "
                f"3σ={row['three_sigma']:.2f}% MAPE={row['mape']:.2f}%"
            )
    csv_path = args.output_dir / "three_sigma_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
