"""在 OOF 上做 TCN 配置集成的贪心前向选择（仅开发集，不读锁定测试）。

基础集是 v4 的 5 个配置（各 6 seed）；候选池是本分支新增的配置（3 seed 或
6 seed）与 per-gas head / edge-case VAE / 宽 TCN / 新特征模式。融合方式固定为
OOF 已确定的 ``pred = 0.6 * TCN 集平均 + 0.4 * Ridge(alpha=0.01, 57 维特征)``，
再裁剪到开发集标签范围；只按 OOF 宏平均 3σ 选择是否加入一个配置。

用法（项目根目录，需先训练相应 OOF）：
    .venv/bin/python artifacts/3sigma/scripts/search_ensemble_oof.py

产物：artifacts/3sigma/ensemble_search.json
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
RIDGE_WEIGHT = 0.4
RIDGE_ALPHA = 0.01
OUTPUT = ROOT / "artifacts/3sigma/ensemble_search.json"

S6 = (20260923, 20260924, 20260925, 20260926, 20260927, 20260928)
S3 = S6[:3]


def _paths(config: str, seeds: tuple[int, ...], root: Path) -> list[Path]:
    if config == "log":
        return [root / f"artifacts/3sigma/log_p0/seed_{s}/noise_gan/oof.csv" for s in seeds]
    if config == "n02":
        return [root / f"artifacts/3sigma/log_p0_noise02/seed_{s}/noise_gan/oof.csv" for s in seeds]
    if config == "pergas":
        return [root / f"artifacts/3sigma/pergas_p0/seed_{s}/noise_gan/oof.csv" for s in seeds]
    if config == "gw112":
        return [root / f"artifacts/3sigma/pergas_gw112/seed_{s}/noise_gan/oof.csv" for s in seeds]
    if config == "none":
        return [root / "artifacts/variance_search/pg422_final/noise_gan/oof.csv"] + [
            root / f"artifacts/3sigma/baseline_pg422/seed_{s}/noise_gan/oof.csv" for s in seeds[1:]
        ]
    if config in {"w32", "rel16", "rel32", "log_edge4", "pergas_edge4", "pergas_edge8",
                  "pghead_log", "pghead_log_edge4", "pghead_pergas_edge4"}:
        return [root / f"artifacts/3sigma/{config}/seed_{s}/noise_gan/oof.csv" for s in seeds]
    raise KeyError(config)


BASE = ("log", "n02", "none", "pergas", "gw112")
CANDIDATES = (
    "w32",
    "rel16",
    "rel32",
    "log_edge4",
    "pergas_edge4",
    "pergas_edge8",
    "pghead_log",
    "pghead_log_edge4",
    "pghead_pergas_edge4",
)


def _load_config(config: str) -> np.ndarray | None:
    seeds = S6 if config in BASE else S3
    paths = _paths(config, seeds, ROOT)
    if not all(path.is_file() for path in paths):
        return None
    frames = [pd.read_csv(path).set_index("file") for path in paths]
    reference = _manifest(ROOT / "artifacts/splits/manifest.csv")[0]
    files = [row["file"] for row in reference]
    return np.stack(
        [
            np.mean(
                [frame.loc[files, f"{gas}_pred"].to_numpy(float) for frame in frames],
                axis=0,
            )
            for gas in GASES
        ],
        axis=1,
    )


def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> dict[str, object]:
    sigma = {}
    for index, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, index] - truth[:, index]) / np.maximum(
            np.abs(truth[:, index]), 1e-9
        )
        sigma[gas] = float(3.0 * np.std(relative, ddof=1))
    sigma["macro"] = float(np.mean([sigma[gas] for gas in GASES]))
    return sigma


def main() -> None:
    dev_rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    files = [row["file"] for row in dev_rows]
    folds = np.array([int(row["fold"]) for row in dev_rows])
    truth = np.array([row["_labels"] for row in dev_rows], dtype=float)

    x_all = _ridge_features_for_rows(dev_rows)
    ridge = np.zeros_like(truth)
    for fold in range(5):
        train = folds != fold
        validation = folds == fold
        model = _fit_ridge(x_all[train], truth[train], alpha=RIDGE_ALPHA)
        ridge[validation] = _predict_ridge(model, x_all[validation])

    predictions = {config: _load_config(config) for config in (*BASE, *CANDIDATES)}

    def hybrid(configs: tuple[str, ...]) -> np.ndarray:
        tcn = np.mean([predictions[name] for name in configs if predictions[name] is not None], axis=0)
        fused = (1.0 - RIDGE_WEIGHT) * tcn + RIDGE_WEIGHT * ridge
        return _clip_to_development_range(fused, truth)

    selected = list(BASE)
    current = _three_sigma(truth, hybrid(tuple(selected)))["macro"]
    history = [{"added": None, "configs": selected.copy(), "macro": current}]
    print(f"base5: macro={current:.2f}")
    while True:
        best = None
        for candidate in CANDIDATES:
            if candidate in selected or predictions[candidate] is None:
                continue
            score = _three_sigma(truth, hybrid(tuple(selected + [candidate])))["macro"]
            if best is None or score < best[1]:
                best = (candidate, score)
        if best is None or best[1] >= current - 1e-6:
            break
        selected.append(best[0])
        current = best[1]
        history.append({"added": best[0], "configs": selected.copy(), "macro": current})
        print(f"+ {best[0]}: macro={current:.2f}")

    final = _three_sigma(truth, hybrid(tuple(selected)))
    result = {
        "protocol": "OOF 50 development runs only; locked test untouched",
        "ridge_weight": RIDGE_WEIGHT,
        "ridge_alpha": RIDGE_ALPHA,
        "base_configs": list(BASE),
        "selected_configs": selected,
        "history": history,
        "final_three_sigma": final,
        "available_candidates": {
            name: predictions[name] is not None for name in CANDIDATES
        },
    }
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"final ({len(selected)} configs): "
        + "  ".join(f"{gas}={final[gas]:.2f}" for gas in GASES)
        + f"  macro={final['macro']:.2f}"
    )
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
