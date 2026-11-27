"""汇总组合增强 run 的 6 checkpoint 集成指标与逐 seed×fold 配对统计。

输入是 hierarchical_augmentation.py 的输出目录：folds.csv 与 predictions.npz。
配对单位是 seed×fold，同一 seed×fold 的多个配置先平均；
bootstrap 用 100,000 次百分位区间，p 值为双侧 Wilcoxon。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tcn_union.experiments.forward_augmentation import _metric_values
from tcn_union.experiments.old_protocol_augmentation import _write_csv

METRICS = (
    "macro_three_sigma",
    "macro_mape",
    "macro_median",
    "macro_p90",
    "H2_three_sigma",
    "NH3_three_sigma",
    "NO2_three_sigma",
    "H2_mape",
    "NH3_mape",
    "NO2_mape",
)


def _bootstrap(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(diff) < 2:
        return float("nan"), float("nan")
    index = rng.integers(0, len(diff), size=(100_000, len(diff)))
    means = diff[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired(
    table: pd.DataFrame,
    candidates: tuple[str, ...],
    baselines: tuple[str, ...],
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for metric in METRICS:
        if metric not in table:
            continue
        pivot = table.pivot_table(index=["seed", "fold"], columns="arm", values=metric)
        for candidate in candidates:
            if candidate not in pivot:
                continue
            for baseline in baselines:
                if candidate == baseline or baseline not in pivot:
                    continue
                diff = (pivot[candidate] - pivot[baseline]).dropna().to_numpy()
                if len(diff) < 2:
                    continue
                low, high = _bootstrap(diff, rng)
                rows.append(
                    {
                        "metric": metric,
                        "candidate": candidate,
                        "baseline": baseline,
                        "n": len(diff),
                        "mean_diff": float(diff.mean()),
                        "median_diff": float(np.median(diff)),
                        "improved": int((diff < 0).sum()),
                        "ci_low": low,
                        "ci_high": high,
                        "p": float(wilcoxon(diff).pvalue)
                        if len(diff) >= 6
                        else float("nan"),
                    }
                )
    return rows


def run(
    source: Path,
    folds_path: Path,
    output: Path,
    candidates: tuple[str, ...],
    baselines: tuple[str, ...],
) -> None:
    folds = pd.read_csv(folds_path)
    columns = [metric for metric in METRICS if metric in folds]
    table = folds.groupby(["seed", "fold", "arm"], as_index=False)[columns].mean()

    payload = np.load(source)
    truth = payload["truth"].astype(np.float64)
    ensemble_rows = []
    for arm in sorted(folds["arm"].unique()):
        key = f"{arm}_ensemble"
        if key not in payload.files:
            continue
        ensemble_rows.append(
            {
                "arm": arm,
                **{
                    name: round(float(value), 6)
                    for name, value in _metric_values(payload[key], truth).items()
                },
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "ensemble_metrics.csv", ensemble_rows)
    _write_csv(
        output / "paired.csv",
        _paired(table, candidates, baselines, np.random.default_rng(20261126)),
    )
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates", nargs="+", required=True)
    parser.add_argument("--baselines", nargs="+", default=["real", "vae", "hier"])
    args = parser.parse_args()
    run(
        args.source,
        args.folds,
        args.output,
        tuple(args.candidates),
        tuple(args.baselines),
    )


if __name__ == "__main__":
    main()
