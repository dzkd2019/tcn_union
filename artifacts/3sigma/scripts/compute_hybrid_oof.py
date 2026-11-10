"""复现 OOF 上的 TCN 集 + Ridge 混合 3σ（不读锁定测试集）。

两条协议同时输出：
- v1：每个配置 3 个 seed（20260923–25）先按配置平均、再跨配置平均，Ridge 权重 0.5；
- v2：在 v1 基础上追加 seed 20260926–28（6 seed），Ridge 权重 0.4，并把融合预测
  裁剪到开发集逐气体标签范围。v2 的超参只在 OOF 上确定。

Ridge 侧按 manifest 的 5 折划分逐折拟合（log 浓度目标，alpha=0.03，57 维手工
特征与 ``tcn_union.evaluation.hybrid_evaluate`` 一致），预测后与 TCN 集平均融合。

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
    _clip_to_development_range,
    _fit_ridge,
    _predict_ridge,
    _ridge_features_for_rows,
)

GASES = AXES
SEEDS_V1 = (20260923, 20260924, 20260925)
SEEDS_V2 = (20260923, 20260924, 20260925, 20260926, 20260927, 20260928)
RIDGE_WEIGHT_V1 = 0.5
RIDGE_WEIGHT_V2 = 0.4
OUTPUT_DIR = ROOT / "artifacts/3sigma"


def _config_paths(name: str, seeds: tuple[int, ...]) -> list[Path]:
    """返回一个 TCN 配置在给定 seed 集合下的 OOF 文件路径。"""
    base = {
        "log_p0": ("noise_gan", "log_p0"),
        "log_n02": ("noise_gan", "log_p0_noise02"),
        "pergas_p0": ("noise_gan", "pergas_p0"),
        "none_pg422": ("noise_gan", "baseline_pg422"),
    }[name]
    mode, directory = base
    paths = []
    for seed in seeds:
        if name == "none_pg422" and seed == 20260923:
            # 历史基线 seed23 的 OOF 在 variance_search 下。
            paths.append(ROOT / "artifacts/variance_search/pg422_final/noise_gan/oof.csv")
        else:
            paths.append(
                ROOT / "artifacts/3sigma" / directory / f"seed_{seed}" / mode / "oof.csv"
            )
    return paths


TCN_CONFIGS = ("log_p0", "log_n02", "pergas_p0", "none_pg422")


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


def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
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


def _interval(values: list[float]) -> dict[str, object]:
    array = np.asarray(values)
    return {
        "mean": float(array.mean()),
        "ci95": [float(np.percentile(array, 2.5)), float(np.percentile(array, 97.5))],
        "p_improve": float(np.mean(array < 0.0)),
    }


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

    def tcn_mean(seeds: tuple[int, ...]) -> dict[str, np.ndarray]:
        by_config = {
            name: _config_prediction(_config_paths(name, seeds), files)
            for name in TCN_CONFIGS
        }
        return {"mean": np.mean(list(by_config.values()), axis=0), "configs": by_config}

    tcn_v1 = tcn_mean(SEEDS_V1)
    tcn_v2 = tcn_mean(SEEDS_V2)
    hybrid_v1 = (1.0 - RIDGE_WEIGHT_V1) * tcn_v1["mean"] + RIDGE_WEIGHT_V1 * ridge_oof
    hybrid_v2 = (1.0 - RIDGE_WEIGHT_V2) * tcn_v2["mean"] + RIDGE_WEIGHT_V2 * ridge_oof
    hybrid_v2 = _clip_to_development_range(hybrid_v2, truth)

    baseline = tcn_v2["configs"]["none_pg422"]
    rng = np.random.default_rng(0)
    delta_baseline: list[float] = []
    delta_tcn: list[float] = []
    for _ in range(4000):
        pick = rng.integers(0, len(files), len(files))
        hybrid_score = _macro_three_sigma(truth[pick], hybrid_v2[pick])
        delta_baseline.append(hybrid_score - _macro_three_sigma(truth[pick], baseline[pick]))
        delta_tcn.append(
            hybrid_score - _macro_three_sigma(truth[pick], tcn_v2["mean"][pick])
        )

    result = {
        "protocol": "OOF 50 development runs; ridge fitted per fold; locked test untouched",
        "v1": {
            "seeds": list(SEEDS_V1),
            "ridge_weight": RIDGE_WEIGHT_V1,
            "tcn_mean": _three_sigma(truth, tcn_v1["mean"]),
            "hybrid": _three_sigma(truth, hybrid_v1),
        },
        "v2": {
            "seeds": list(SEEDS_V2),
            "ridge_weight": RIDGE_WEIGHT_V2,
            "clip_to_development_range": True,
            "tcn_mean": _three_sigma(truth, tcn_v2["mean"]),
            "ridge": _three_sigma(truth, ridge_oof),
            "hybrid": _three_sigma(truth, hybrid_v2),
            "baseline_none_pg422": _three_sigma(truth, baseline),
            "per_config": {
                name: _three_sigma(truth, prediction)
                for name, prediction in tcn_v2["configs"].items()
            },
            "bootstrap_delta_macro3sigma": {
                "hybrid_minus_baseline": _interval(delta_baseline),
                "hybrid_minus_tcn_mean": _interval(delta_tcn),
            },
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
                [name, int(folds[index]), *truth[index].tolist(), *hybrid_v2[index].tolist()]
            )

    for version, summary in (("v1", result["v1"]), ("v2", result["v2"])):
        for name in ("tcn_mean", "ridge", "hybrid"):
            if name not in summary:
                continue
            per_gas = "  ".join(
                f"{gas}={summary[name]['three_sigma'][gas]:.2f}" for gas in GASES
            )
            print(
                f"{version} {name:<9} {per_gas}  macro={summary[name]['three_sigma_macro']:.2f}  "
                f"MAPE={summary[name]['mape_macro']:.2f}"
            )
    print("bootstrap:", json.dumps(result["v2"]["bootstrap_delta_macro3sigma"], ensure_ascii=False))
    print(f"Wrote {OUTPUT_DIR / 'hybrid_oof_summary.json'}")


if __name__ == "__main__":
    main()
