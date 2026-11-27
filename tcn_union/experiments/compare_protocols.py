"""比较不同训练协议下 real / VAE / hier / hier+VAE 的 10 s 结果。

输入是 hierarchical_augmentation.py 产出的目录（每个协议一个目录），
使用 folds.csv 的 seed×fold 指标与 predictions.npz 的 6 checkpoint 集成。
配对统计以 seed×fold 为单位（同协议内比较），跨协议比较也按 seed×fold 对齐。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tcn_union.experiments.forward_augmentation import _metric_values
from tcn_union.experiments.old_protocol_augmentation import _write_csv

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation/hierarchical"
PROTOCOLS = ("syn", "noise_syn", "full", "gas")
ARMS = ("real", "vae", "hier", "hier+vae")
METRICS = (
    "macro_three_sigma",
    "macro_mape",
    "H2_three_sigma",
    "NH3_three_sigma",
    "NO2_three_sigma",
    "H2_mape",
    "NH3_mape",
    "NO2_mape",
)


def _bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    if len(diff) < 2:
        return float("nan"), float("nan")
    index = rng.integers(0, len(diff), size=(100_000, len(diff)))
    means = diff[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _paired_rows(
    table: pd.DataFrame,
    candidates: tuple[str, ...],
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for metric in METRICS:
        pivot = table.pivot_table(
            index=["seed", "fold"], columns="arm", values=metric
        )
        for candidate in candidates:
            for baseline in ("real", "vae"):
                if candidate == baseline or candidate not in pivot or baseline not in pivot:
                    continue
                diff = (pivot[candidate] - pivot[baseline]).dropna().to_numpy()
                if len(diff) < 2:
                    continue
                pvalue = (
                    float(wilcoxon(diff).pvalue) if len(diff) >= 6 else float("nan")
                )
                low, high = _bootstrap_ci(diff, rng)
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
                        "p": pvalue,
                    }
                )
    return rows


def _protocol_effect(
    tables: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> list[dict]:
    rows = []
    for arm in ARMS:
        for metric in METRICS:
            for left in PROTOCOLS:
                for right in PROTOCOLS:
                    if left >= right:
                        continue
                    if left not in tables or right not in tables:
                        continue
                    left_table = tables[left]
                    right_table = tables[right]
                    left_values = left_table[left_table.arm == arm]
                    right_values = right_table[right_table.arm == arm]
                    if left_values.empty or right_values.empty:
                        continue
                    left_series = left_values.set_index(["seed", "fold"])[metric]
                    right_series = right_values.set_index(["seed", "fold"])[metric]
                    common = left_series.index.intersection(right_series.index)
                    if len(common) < 2:
                        continue
                    diff = (right_series.loc[common] - left_series.loc[common]).to_numpy()
                    pvalue = (
                        float(wilcoxon(diff).pvalue)
                        if len(diff) >= 6
                        else float("nan")
                    )
                    low, high = _bootstrap_ci(diff, rng)
                    rows.append(
                        {
                            "arm": arm,
                            "metric": metric,
                            "protocol_a": left,
                            "protocol_b": right,
                            "n": len(diff),
                            "mean_diff_b_minus_a": float(diff.mean()),
                            "median_diff_b_minus_a": float(np.median(diff)),
                            "improved_b": int((diff < 0).sum()),
                            "ci_low": low,
                            "ci_high": high,
                            "p": pvalue,
                        }
                    )
    return rows


def run(
    root: Path,
    output: Path,
    protocols: tuple[str, ...],
    prefix: str = "",
) -> None:
    rng = np.random.default_rng(20261126)
    ensemble_rows = []
    seed_rows = []
    tables = {}
    for protocol in protocols:
        directory = root / f"{prefix}{protocol}"
        folds_path = directory / "folds.csv"
        predictions_path = directory / "predictions.npz"
        if not folds_path.exists() or not predictions_path.exists():
            print(f"skip {protocol}: missing {directory}")
            continue
        folds = pd.read_csv(folds_path)
        if "protocol" not in folds:
            folds["protocol"] = protocol
        folds = folds[folds["protocol"] == protocol].copy()
        # 同一 seed×fold 的多个 config 先平均，配对单位是 seed×fold。
        table = (
            folds.groupby(["protocol", "seed", "fold", "arm"], as_index=False)[
                list(METRICS)
            ]
            .mean()
        )
        tables[protocol] = table

        data = np.load(predictions_path)
        truth = data["truth"]
        for arm in ARMS:
            key = f"{arm}_ensemble"
            if key in data:
                prediction = data[key]
                ensemble_rows.append(
                    {
                        "protocol": protocol,
                        "arm": arm,
                        "n_checkpoints": int(
                            len([k for k in data.files if k.startswith(f"{arm}|")])
                        ),
                        **_metric_values(prediction, truth),
                    }
                )
            for seed in sorted({int(value) for value in folds["seed"].unique()}):
                key = f"{arm}_s{seed}"
                if key not in data:
                    continue
                seed_rows.append(
                    {
                        "protocol": protocol,
                        "arm": arm,
                        "seed": seed,
                        **_metric_values(data[key], truth),
                    }
                )
    output.mkdir(parents=True, exist_ok=True)
    _write_csv(output / "ensemble_metrics.csv", ensemble_rows)
    _write_csv(output / "seed_metrics.csv", seed_rows)

    summary_rows = []
    if seed_rows:
        seed_frame = pd.DataFrame(seed_rows)
        for (protocol, arm), subset in seed_frame.groupby(["protocol", "arm"]):
            for metric in METRICS:
                values = subset[metric].to_numpy()
                summary_rows.append(
                    {
                        "protocol": protocol,
                        "arm": arm,
                        "metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                        "n": len(values),
                    }
                )
    _write_csv(output / "seed_summary.csv", summary_rows)

    paired_rows = []
    for protocol, table in tables.items():
        for row in _paired_rows(table, ("hier", "hier+vae"), rng):
            paired_rows.append({"protocol": protocol, **row})
    _write_csv(output / "paired_within_protocol.csv", paired_rows)

    protocol_rows = _protocol_effect(tables, rng)
    if protocol_rows:
        _write_csv(output / "protocol_effect.csv", protocol_rows)
    config = {
        "protocols": list(protocols),
        "prefix": prefix,
        "protocols_found": sorted(tables),
        "arms": list(ARMS),
        "metrics": list(METRICS),
        "paired_unit": "seed x fold, config averaged",
        "bootstrap": 100_000,
    }
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=BASE)
    parser.add_argument("--output", type=Path, default=BASE / "comparison")
    parser.add_argument("--protocols", nargs="+", default=list(PROTOCOLS))
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()
    root = args.root if args.root.is_absolute() else ROOT / args.root
    output = args.output if args.output.is_absolute() else ROOT / args.output
    run(root, output, tuple(args.protocols), args.prefix)


if __name__ == "__main__":
    main()
