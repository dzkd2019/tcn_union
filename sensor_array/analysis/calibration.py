"""P4 区间校准：在 62 折折外浓度估计上校准 90% 区间。

输入是 `artifacts/evaluation/downstream.csv`（三路独立模型的 5 s / 10 s 折外估计），
输出原始 Laplace、全局共形、按 NH3 档位 / 采集块分组的共形区间，以及覆盖率与宽度。
校准分位数只用 50 条开发集；12 条锁定测试只作描述性复核。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..core.data import GASES, Experiment, load_experiments
from ..paths import CALIBRATION, EVALUATION, ensure

MODEL_NAME = "mlp_per_gas"
LOCKED_FILES = frozenset({2, 7, 15, 20, 23, 27, 31, 38, 41, 46, 55, 60})
ALPHA = 0.1
Z = {0.80: 1.2815515655446004, 0.90: 1.6448536269514722, 0.95: 1.959963984540054}


def _group_table(experiments: list[Experiment]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file": [experiment.file_id for experiment in experiments],
            "NH3": [float(experiment.concentration[1]) for experiment in experiments],
            "block": [(experiment.file_id // 8) * 8 for experiment in experiments],
        }
    )


def _load() -> pd.DataFrame:
    frame = pd.read_csv(EVALUATION / "downstream.csv")
    frame = frame[frame["model"] == MODEL_NAME].copy()
    if frame.empty:
        raise ValueError(f"downstream.csv 中没有模型 {MODEL_NAME}")
    frame["se_log"] = frame["se"] / frame["theta_hat"]
    frame["abs_log_error"] = np.abs(np.log(frame["theta_hat"] / frame["theta_true"]))
    frame["score"] = frame["abs_log_error"] / frame["se_log"].clip(lower=1e-12)
    frame["split"] = np.where(frame["file"].isin(LOCKED_FILES), "locked", "development")
    return frame.merge(_group_table(load_experiments()), on="file", how="left")


def _quantile(scores: np.ndarray, alpha: float = ALPHA) -> float:
    scores = np.asarray(scores, dtype=float)
    if scores.size == 0:
        return float("nan")
    level = min(1.0, np.ceil((scores.size + 1) * (1.0 - alpha)) / scores.size)
    return float(np.quantile(scores, level, method="higher"))


def laplace_intervals(frame: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for level, z in Z.items():
        out = frame.copy()
        out["method"] = f"laplace{int(round(100 * level))}"
        out["group_type"] = "all"
        out["q"] = z
        out["half_log"] = z * out["se_log"]
        out["lower"] = out["theta_hat"] * np.exp(-out["half_log"])
        out["upper"] = out["theta_hat"] * np.exp(out["half_log"])
        out["covered"] = (out["theta_true"] >= out["lower"]) & (out["theta_true"] <= out["upper"])
        out["width_rel"] = np.exp(out["half_log"]) - 1.0
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True)


def conformal_intervals(frame: pd.DataFrame, group_type: str) -> pd.DataFrame:
    """开发集留一文件校准；锁定文件使用全部开发集分数。"""
    rows = []
    for (tmax, gas), subset in frame.groupby(["tmax", "gas"], sort=False):
        development = subset[subset["split"] == "development"]
        for _, row in subset.iterrows():
            if row["split"] == "development":
                pool = development[development["file"] != row["file"]]
            else:
                pool = development
            global_q = _quantile(pool["score"].to_numpy())
            if group_type == "all":
                q = global_q
                group = "overall"
            else:
                same = pool[pool[group_type] == row[group_type]]
                q = _quantile(same["score"].to_numpy()) if same.shape[0] >= 2 else global_q
                group = row[group_type]
            record = row.to_dict()
            record.update({"method": f"conformal_{group_type}", "group_type": group_type,
                           "group": group, "q": q})
            rows.append(record)
    out = pd.DataFrame(rows)
    out["half_log"] = out["q"] * out["se_log"]
    out["lower"] = out["theta_hat"] * np.exp(-out["half_log"])
    out["upper"] = out["theta_hat"] * np.exp(out["half_log"])
    out["covered"] = (out["theta_true"] >= out["lower"]) & (out["theta_true"] <= out["upper"])
    out["width_rel"] = np.exp(out["half_log"]) - 1.0
    return out


def _summarize(intervals: pd.DataFrame) -> pd.DataFrame:
    return (
        intervals.groupby(["method", "split", "tmax", "gas"], sort=False)
        .agg(
            n=("file", "size"),
            coverage=("covered", "mean"),
            median_width_rel=("width_rel", "median"),
            p90_width_rel=("width_rel", lambda values: values.quantile(0.9)),
            median_q=("q", "median"),
        )
        .reset_index()
    )


def _by_group(intervals: pd.DataFrame) -> pd.DataFrame:
    subset = intervals[intervals["group_type"].isin(("NH3", "block"))]
    return (
        subset.groupby(["method", "group_type", "group", "split", "tmax", "gas"], sort=False)
        .agg(
            n=("file", "size"),
            coverage=("covered", "mean"),
            median_width_rel=("width_rel", "median"),
        )
        .reset_index()
    )


def run() -> None:
    ensure()
    frame = _load()
    intervals = pd.concat(
        [
            laplace_intervals(frame),
            conformal_intervals(frame, "all"),
            conformal_intervals(frame, "NH3"),
            conformal_intervals(frame, "block"),
        ],
        ignore_index=True,
    )
    keep = [
        "file", "split", "tmax", "gas", "method", "group_type", "group",
        "theta_hat", "theta_true", "se_log", "q", "lower", "upper",
        "covered", "width_rel",
    ]
    intervals = intervals[keep]
    summary = _summarize(intervals)
    by_group = _by_group(intervals)
    intervals.to_csv(CALIBRATION / "intervals.csv", index=False)
    summary.to_csv(CALIBRATION / "summary.csv", index=False)
    by_group.to_csv(CALIBRATION / "by_group.csv", index=False)

    print("P4 区间校准（10 s，开发集覆盖 / 中位宽度 %）")
    ten = summary[(summary["tmax"] == 10.0) & (summary["split"] == "development")]
    print(
        (ten.pivot(index="method", columns="gas", values="coverage") * 100)
        .round(1)
        .to_string()
    )
    print("\n10 s 开发集中位宽度 / %")
    print(
        (ten.pivot(index="method", columns="gas", values="median_width_rel") * 100)
        .round(1)
        .to_string()
    )
    print("\n10 s 锁定集覆盖 / %")
    locked = summary[(summary["tmax"] == 10.0) & (summary["split"] == "locked")]
    print(
        (locked.pivot(index="method", columns="gas", values="coverage") * 100)
        .round(1)
        .to_string()
    )
    print("saved", CALIBRATION)


if __name__ == "__main__":
    run()
