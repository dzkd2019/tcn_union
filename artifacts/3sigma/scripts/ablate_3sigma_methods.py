"""3σ 降方差手段的堆叠消融与精简方案对比（仅 OOF，不读锁定测试集）。

输入：`artifacts/3sigma/` 下已训练好的 OOF 预测（每个配置 6 个 seed）。
输出：
- `artifacts/3sigma/method_ablation.json`：各方法/组合的逐气体 3σ；
- 终端表格：①单配置效果 ②堆叠增量 ③seed 数影响 ④精简组合的模型数
  与 OOF 指标 ⑤Ridge/裁剪开关 ⑥Ridge 权重敏感性。

口径与正式方案一致：TCN 预测先按 seed 平均，再算逐实验相对误差的 3σ
（ddof=1）；Ridge 为 57 维手工特征、alpha=0.01、五折交叉拟合；混合权重
w=0.4；可选裁剪到开发集浓度范围。只用开发集标签，不触碰测试集。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.evaluation.evaluate import _manifest
from tcn_union.evaluation.hybrid_evaluate import (
    _clip_to_development_range,
    _fit_ridge,
    _predict_ridge,
    _ridge_features_for_rows,
)

GASES = ("H2", "NH3", "NO2")
SEEDS = (20260923, 20260924, 20260925, 20260926, 20260927, 20260928)
RIDGE_WEIGHT = 0.4
RIDGE_ALPHA = 0.01
OUTPUT_JSON = ROOT / "artifacts/3sigma/method_ablation.json"

CONFIG_PATHS = {
    "log_p0": [ROOT / f"artifacts/3sigma/log_p0/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "log_n02": [ROOT / f"artifacts/3sigma/log_p0_noise02/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "pergas_p0": [ROOT / f"artifacts/3sigma/pergas_p0/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "pergas_gw112": [ROOT / f"artifacts/3sigma/pergas_gw112/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "none_pg422": [ROOT / "artifacts/variance_search/pg422_final/noise_gan/oof.csv"]
    + [ROOT / f"artifacts/3sigma/baseline_pg422/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS[1:]],
    "pergas_pghead": [ROOT / f"artifacts/3sigma/pergas_pghead/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "pergas_edge4": [ROOT / f"artifacts/3sigma/pergas_edge4/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "pergas_edge8": [ROOT / f"artifacts/3sigma/pergas_edge8/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "pghead_log": [ROOT / f"artifacts/3sigma/pghead_log/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    "pghead_pergas_edge4": [ROOT / f"artifacts/3sigma/pghead_pergas_edge4/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
}

BASE5 = ("log_p0", "log_n02", "none_pg422", "pergas_p0", "pergas_gw112")
V5 = BASE5 + ("pergas_edge4", "pergas_edge8", "pghead_log", "pghead_pergas_edge4")
LEAN = ("pghead_pergas_edge4",)
LEAN_NO_EDGE = ("pergas_pghead",)


def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    values = {}
    for index, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, index] - truth[:, index]) / np.maximum(
            np.abs(truth[:, index]), 1e-9
        )
        values[gas] = float(3.0 * np.std(relative, ddof=1))
    values["macro"] = float(np.mean([values[gas] for gas in GASES]))
    values["mape"] = float(
        np.mean(np.abs(100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)))
    )
    return values


def _seed_mean(paths: list[Path]) -> np.ndarray:
    frames = [pd.read_csv(path).set_index("file") for path in paths]
    return np.stack(
        [
            np.mean(
                [frame.loc[FILES, f"{gas}_pred"].to_numpy(float) for frame in frames],
                axis=0,
            )
            for gas in GASES
        ],
        axis=1,
    )


def _ensemble(configs: tuple[str, ...], seed_count: int) -> np.ndarray:
    return np.mean([_seed_mean(CONFIG_PATHS[name][:seed_count]) for name in configs], axis=0)


def _hybrid(tcn: np.ndarray, ridge: np.ndarray, clip: bool, weight: float = RIDGE_WEIGHT) -> np.ndarray:
    fused = (1.0 - weight) * tcn + weight * ridge
    return _clip_to_development_range(fused, TRUTH) if clip else fused


def main() -> None:
    global FILES, TRUTH
    dev_rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    FILES = [row["file"] for row in dev_rows]
    folds = np.array([int(row["fold"]) for row in dev_rows])
    TRUTH = np.array([row["_labels"] for row in dev_rows], dtype=float)

    features = _ridge_features_for_rows(dev_rows)
    ridge = np.zeros_like(TRUTH)
    for fold in range(5):
        train = folds != fold
        validation = folds == fold
        model = _fit_ridge(features[train], TRUTH[train], alpha=RIDGE_ALPHA)
        ridge[validation] = _predict_ridge(model, features[validation])

    result: dict[str, object] = {
        "protocol": "OOF 50 development runs only; locked test untouched",
        "ridge_weight": RIDGE_WEIGHT,
        "ridge_alpha": RIDGE_ALPHA,
        "ridge_feature_dim": int(features.shape[1]),
        "configs": {name: list(map(str, paths)) for name, paths in CONFIG_PATHS.items()},
    }

    # ① 单配置：TCN 与 单配置+Ridge+裁剪
    print("\n=== ① 单配置效果（6 seed） ===")
    print(f"{'config':<24}{'TCN':>7}{'+Ridge+clip':>13}   H2/NH3/NO2 (混合)")
    solo: dict[str, dict[str, object]] = {}
    for name in CONFIG_PATHS:
        tcn = _ensemble((name,), 6)
        fused = _hybrid(tcn, ridge, clip=True)
        tcn_sigma = _three_sigma(TRUTH, tcn)
        hybrid_sigma = _three_sigma(TRUTH, fused)
        solo[name] = {"tcn": tcn_sigma, "hybrid": hybrid_sigma}
        print(
            f"{name:<24}{tcn_sigma['macro']:>7.2f}{hybrid_sigma['macro']:>13.2f}"
            f"   {hybrid_sigma['H2']:.2f}/{hybrid_sigma['NH3']:.2f}/{hybrid_sigma['NO2']:.2f}"
        )
    result["solo"] = solo

    # ② 堆叠与精简路径
    print("\n=== ② 堆叠 / 精简路径（OOF 宏 3σ） ===")
    stack_sequence = [
        ("旧协议基线（none+加权）", ("none_pg422",)),
        ("+ log 目标", ("log_p0",)),
        ("逐气体目标", ("pergas_p0",)),
        ("+ 逐气体头（无边缘扩充）", LEAN_NO_EDGE),
        ("+ 边缘扩充（精简方案）", LEAN),
        ("基础 5 配置集成", BASE5),
        ("完整 9 配置集成（v5）", V5),
    ]
    stack: list[dict[str, object]] = []
    for label, configs in stack_sequence:
        tcn = _ensemble(configs, 6)
        fused = _hybrid(tcn, ridge, clip=True)
        sigma = _three_sigma(TRUTH, fused)
        tcn_sigma = _three_sigma(TRUTH, tcn)
        stack.append(
            {
                "label": label,
                "configs": list(configs),
                "models": len(configs) * 6,
                "tcn_macro": tcn_sigma["macro"],
                "hybrid_macro": sigma["macro"],
                "hybrid": sigma,
            }
        )
        print(
            f"{label:<28} 模型数={len(configs) * 6:<3} TCN={tcn_sigma['macro']:>6.2f}"
            f"  混合={sigma['macro']:>6.2f}  ({sigma['H2']:.2f}/{sigma['NH3']:.2f}/{sigma['NO2']:.2f})"
        )
    result["stack"] = stack

    # ③ seed 数影响（精简方案与 9 配置）
    print("\n=== ③ seed 数影响 ===")
    seed_ablation: list[dict[str, object]] = []
    for label, configs in (
        ("精简方案", LEAN),
        ("精简-无边缘", LEAN_NO_EDGE),
        ("9 配置集成", V5),
    ):
        for count in (1, 2, 3, 6):
            tcn = _ensemble(configs, count)
            tcn_sigma = _three_sigma(TRUTH, tcn)
            fused = _hybrid(tcn, ridge, clip=True)
            hybrid_sigma = _three_sigma(TRUTH, fused)
            seed_ablation.append(
                {
                    "set": label,
                    "seeds": count,
                    "models": len(configs) * count,
                    "tcn": tcn_sigma,
                    "hybrid": hybrid_sigma,
                }
            )
            print(
                f"{label:<12} {count} seed: 模型数={len(configs) * count:<3}"
                f" TCN={tcn_sigma['macro']:.2f}  混合={hybrid_sigma['macro']:.2f}"
            )
    result["seed_ablation"] = seed_ablation

    # ④ 精简组合：配置数/模型数 vs OOF
    print("\n=== ④ 精简组合（6 seed，Ridge+裁剪） ===")
    subsets = [
        ("单配置：逐气体头（无边缘）", LEAN_NO_EDGE),
        ("单配置：逐气体头 + 边缘4（推荐）", LEAN),
        ("双配置：精简 + log", ("log_p0", "pghead_pergas_edge4")),
        ("三配置：精简 + log + pergas", ("log_p0", "pergas_p0", "pghead_pergas_edge4")),
        ("基础 5 配置（v4 组成）", BASE5),
        ("完整 9 配置（v5）", V5),
    ]
    subset_results: list[dict[str, object]] = []
    for label, configs in subsets:
        tcn = _ensemble(configs, 6)
        fused = _hybrid(tcn, ridge, clip=True)
        sigma = _three_sigma(TRUTH, fused)
        sigma_tcn = _three_sigma(TRUTH, tcn)
        subset_results.append(
            {
                "label": label,
                "configs": list(configs),
                "models": len(configs) * 6,
                "tcn_macro": sigma_tcn["macro"],
                "hybrid_macro": sigma["macro"],
                "hybrid": sigma,
            }
        )
        print(
            f"{label:<34} 模型数={len(configs) * 6:<3} TCN={sigma_tcn['macro']:>6.2f}"
            f"  混合={sigma['macro']:>6.2f}  ({sigma['H2']:.2f}/{sigma['NH3']:.2f}/{sigma['NO2']:.2f})"
        )
    result["subsets"] = subset_results

    # ⑤ Ridge / 裁剪开关（精简方案与 9 配置）
    print("\n=== ⑤ Ridge 与裁剪开关 ===")
    toggles: dict[str, dict[str, object]] = {}
    for label, configs in (("精简方案", LEAN), ("9 配置集成", V5)):
        tcn = _ensemble(configs, 6)
        entries = {
            "仅 TCN": tcn,
            "TCN + Ridge（不裁剪）": _hybrid(tcn, ridge, clip=False),
            "TCN + Ridge + 裁剪": _hybrid(tcn, ridge, clip=True),
        }
        toggles[label] = {}
        for entry_label, prediction in entries.items():
            sigma = _three_sigma(TRUTH, prediction)
            toggles[label][entry_label] = sigma
            print(
                f"{label:<10}{entry_label:<22} 宏={sigma['macro']:>6.2f}"
                f"  ({sigma['H2']:.2f}/{sigma['NH3']:.2f}/{sigma['NO2']:.2f})"
            )
    result["toggles"] = toggles

    # ⑥ Ridge 混合权重敏感性（精简方案）
    print("\n=== ⑥ Ridge 权重敏感性（精简方案，6 seed） ===")
    tcn = _ensemble(LEAN, 6)
    weights: dict[str, dict[str, float]] = {}
    for weight in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6):
        sigma = _three_sigma(TRUTH, _hybrid(tcn, ridge, clip=True, weight=weight))
        weights[f"{weight:.1f}"] = sigma
        print(f"w={weight:.1f}  宏={sigma['macro']:.2f}  ({sigma['H2']:.2f}/{sigma['NH3']:.2f}/{sigma['NO2']:.2f})")
    result["ridge_weight_sweep"] = weights

    result["test_history"] = {
        "baseline": {"H2": 10.75, "NH3": 6.65, "NO2": 16.14, "macro": 11.18},
        "v4": {"H2": 11.00, "NH3": 3.71, "NO2": 10.04, "macro": 8.25},
        "v5": {"H2": 10.60, "NH3": 3.80, "NO2": 10.10, "macro": 8.17},
        "note": "v5 对应 9 配置×6 seed；精简方案未在现有 12 点测试集上评估（测试集已读 5 次）",
    }

    OUTPUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
