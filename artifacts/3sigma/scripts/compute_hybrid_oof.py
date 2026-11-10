"""复现 OOF 上的 TCN 集 + Ridge 混合 3σ（不读锁定测试集）。

只使用开发集：TCN 侧读取多个 seed/多配置的 ``oof.csv``，先在预测层平均；
Ridge 侧按 manifest 的 5 折划分逐折拟合（log 浓度目标，alpha=0.03，57 维手工
特征与 ``tcn_union.evaluation.hybrid_evaluate`` 一致），最后按
``pred = (1 - w) * tcn_mean + w * ridge`` 融合。

用法（项目根目录）：
    .venv/bin/python artifacts/3sigma/scripts/compute_hybrid_oof.py

产物：artifacts/3sigma/hybrid_oof.csv、artifacts/3sigma/hybrid_oof_summary.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.evaluation.evaluate import AXES, _manifest
from tcn_union.evaluation.hybrid_evaluate import (
    _fit_ridge,
    _predict_ridge,
    _ridge_features_for_rows,
)

GASES = AXES
# 每个配置的 OOF 文件（同配置不同 seed 先平均，再在配置间平均）。
TCN_OOF = {
    "log_p0": [ROOT / "artifacts/3sigma/log_p0" / f"seed_{seed}" / "noise_gan/oof.csv"
               for seed in (20260923, 20260924, 20260925)],
    "log_n02": [ROOT / "artifacts/3sigma/log_p0_noise02" / f"seed_{seed}" / "noise_gan/oof.csv"
                for seed in (20260923, 20260924, 20260925)],
    "none_pg422": [
        ROOT / "artifacts/variance_search/pg422_final/noise_gan/oof.csv",
        ROOT / "artifacts/3sigma/baseline_pg422/seed_20260924/noise_gan/oof.csv",
        ROOT / "artifacts/3sigma/baseline_pg422/seed_20260925/noise_gan/oof.csv",
    ],
    "pergas_p0": [
        ROOT / "artifacts/3sigma/pergas_p0" / f"seed_{seed}" / "noise_gan/oof.csv"
        for seed in (20260923, 20260924, 20260925)
    ],
}
RIDGE_WEIGHT = 0.5
OUTPUT_DIR = ROOT / "artifacts/3sigma"


def _load_frame(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """读取单个 oof.csv，返回 file -> (真值, 预测) 的映射。"""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["file"]: (
            np.array([float(row[f"{gas}_true"]) for gas in GASES]),
            np.array([float(row[f"{gas}_pred"]) for gas in GASES]),
        )
        for row in rows
    }


def _config_prediction(paths: list[Path], reference_files: list[str]) -> np.ndarray:
    """把一个配置内多个 seed 的预测按实验平均（按 reference_files 顺序对齐）。"""
    predictions = []
    for path in paths:
        frame = _load_frame(path)
        missing = set(reference_files) - set(frame)
        if missing:
            raise ValueError(f"OOF 缺少实验：{sorted(missing)}（{path}）")
        predictions.append(np.stack([frame[name][1] for name in reference_files]))
    return np.mean(predictions, axis=0)


def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """逐气体 3σ 与宏平均、MAPE、平均相对偏差。"""
    sigma = {}
    mape = {}
    bias = {}
    for index, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, index] - truth[:, index]) / np.maximum(
            np.abs(truth[:, index]), 1e-9
        )
        sigma[gas] = float(3.0 * np.std(relative, ddof=1))
        mape[gas] = float(np.mean(np.abs(relative)))
        bias[gas] = float(np.mean(relative))
    return {
        "three_sigma": sigma,
        "three_sigma_macro": float(np.mean(list(sigma.values()))),
        "mape": mape,
        "mape_macro": float(np.mean(list(mape.values()))),
        "mean_relative": bias,
    }


def _macro_three_sigma(truth: np.ndarray, prediction: np.ndarray) -> float:
    """返回三气体 3σ 的宏平均。"""
    return float(_three_sigma(truth, prediction)["three_sigma_macro"])


def main() -> None:
    dev_rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    files = [row["file"] for row in dev_rows]
    folds = np.array([int(row["fold"]) for row in dev_rows])
    truth = np.array([row["_labels"] for row in dev_rows], dtype=float)

    # Ridge OOF：每折只用其余四折拟合，避免同点自拟合。
    x_all = _ridge_features_for_rows(dev_rows)
    ridge_oof = np.zeros_like(truth)
    for fold in range(5):
        train = folds != fold
        validation = folds == fold
        model = _fit_ridge(x_all[train], truth[train])
        ridge_oof[validation] = _predict_ridge(model, x_all[validation])

    tcn_by_config = {
        name: _config_prediction(paths, files) for name, paths in TCN_OOF.items()
    }
    tcn_mean = np.mean(list(tcn_by_config.values()), axis=0)
    hybrid = (1.0 - RIDGE_WEIGHT) * tcn_mean + RIDGE_WEIGHT * ridge_oof

    # 实验级配对 bootstrap：评估混合相对“旧协议基线/纯 TCN 集”的宏 3σ 改善区间。
    baseline = tcn_by_config["none_pg422"]
    rng = np.random.default_rng(0)
    delta_baseline: list[float] = []
    delta_tcn: list[float] = []
    for _ in range(4000):
        pick = rng.integers(0, len(files), len(files))
        hybrid_score = _macro_three_sigma(truth[pick], hybrid[pick])
        delta_baseline.append(hybrid_score - _macro_three_sigma(truth[pick], baseline[pick]))
        delta_tcn.append(hybrid_score - _macro_three_sigma(truth[pick], tcn_mean[pick]))

    def _interval(values: list[float]) -> dict[str, object]:
        array = np.asarray(values)
        return {
            "mean": float(array.mean()),
            "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
            "p_improve": float(np.mean(array < 0.0)),
        }

    result = {
        "protocol": "OOF 50 development runs; ridge fitted per fold; locked test untouched",
        "tcn_configs": {name: [str(path) for path in paths] for name, paths in TCN_OOF.items()},
        "ridge_weight": RIDGE_WEIGHT,
        "baseline_none_pg422": _three_sigma(truth, baseline),
        "tcn_mean": _three_sigma(truth, tcn_mean),
        "ridge": _three_sigma(truth, ridge_oof),
        "hybrid": _three_sigma(truth, hybrid),
        "per_config": {
            name: _three_sigma(truth, prediction)
            for name, prediction in tcn_by_config.items()
        },
        "bootstrap_delta_macro3sigma": {
            "hybrid_minus_baseline": _interval(delta_baseline),
            "hybrid_minus_tcn_mean": _interval(delta_tcn),
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "hybrid_oof_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "hybrid_oof.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["file", "fold", *[f"{gas}_true" for gas in GASES], *[f"{gas}_pred" for gas in GASES]]
        )
        for index, name in enumerate(files):
            writer.writerow(
                [
                    name,
                    int(folds[index]),
                    *truth[index].tolist(),
                    *hybrid[index].tolist(),
                ]
            )
    for name, summary in (
        ("baseline", result["baseline_none_pg422"]),
        ("tcn_mean", result["tcn_mean"]),
        ("ridge", result["ridge"]),
        ("hybrid", result["hybrid"]),
    ):
        per_gas = "  ".join(
            f"{gas}={summary['three_sigma'][gas]:.2f}" for gas in GASES
        )
        print(
            f"{name:<10} {per_gas}  macro={summary['three_sigma_macro']:.2f}  "
            f"MAPE={summary['mape_macro']:.2f}"
        )
    print(f"Wrote {OUTPUT_DIR / 'hybrid_oof_summary.json'}")


if __name__ == "__main__":
    main()
