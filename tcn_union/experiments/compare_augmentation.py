"""汇总四种增强方式的 OOF 预测，并生成统一评估图。

注意：四个臂来自不同轮次，合成量/基线并不完全相同。本脚本只做
50 条开发集、10 s、ensemble 口径的并列表述；同轮配对见各自的
summary/paired 文件。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tcn_union.evaluation.final_report import run as report_run

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/augmentation"
OUTPUT = BASE / "comparison_20261127"


def _config_mean(payload, arm: str) -> dict[str, np.ndarray]:
    keys = [key for key in payload.files if key.startswith(f"{arm}|")]
    seeds = sorted({key.split("|")[1] for key in keys})
    result = {}
    for seed in seeds:
        seed_keys = [key for key in keys if key.split("|")[1] == seed]
        result[f"{arm}_s{seed}"] = np.mean([payload[key] for key in seed_keys], axis=0)
    result[f"{arm}_ensemble"] = np.mean(
        [payload[key] for key in keys], axis=0
    )
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pure = np.load(BASE / "pure_forward/joint10/predictions.npz")
    truth = pure["truth"].astype(np.float64)
    file_ids = pure["file_id"].astype(int)
    combined = np.load(BASE / "combined_full180/predictions.npz")
    guard = np.load(BASE / "no2_guard/full/predictions.npz")

    payload = {"truth": truth, "file_id": file_ids}
    payload.update(_config_mean(combined, "vae"))
    payload.update(_config_mean(combined, "union_h2_180"))
    payload.update(_config_mean(guard, "guard"))
    payload.update(_config_mean(guard, "vae"))
    # 保留 guard 轮 VAE，避免覆盖 combined 轮 VAE。
    for key in list(payload):
        if key.startswith("vae_s") and key.endswith(tuple(str(s) for s in ())):
            pass
    for key in [k for k in list(payload) if k.startswith("vae_")]:
        if key != "vae_ensemble" and not key.startswith("vae_s"):
            continue
    # 重新命名：combined vae -> vae；guard vae -> vae_guard
    for key in [k for k in payload if k.startswith("vae_s")]:
        payload[key.replace("vae_s", "vae_guard_s")] = payload.pop(key)
    payload["vae_guard_ensemble"] = payload.pop("vae_ensemble")
    # 重新加入 combined vae（上面被改名了）
    combined_vae = _config_mean(combined, "vae")
    payload.update(combined_vae)

    pure_ensemble = {
        arm: pure[f"{arm}_ensemble"].astype(np.float64)
        for arm in ("vae", "forward_prob_joint", "forward_prob_joint10")
    }
    for arm in ("vae", "forward_prob_joint", "forward_prob_joint10"):
        for key in pure.files:
            if key.startswith(f"{arm}_s"):
                payload[key.replace(arm, f"{arm}_pure", 1)] = pure[key].astype(np.float64)
        payload[f"{arm}_pure_ensemble"] = pure_ensemble[arm]

    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    (OUTPUT / "run_config.json").write_text(
        json.dumps(
            {
                "n": int(len(truth)),
                "window_s": 10,
                "range_source": "development_50",
                "sources": {
                    "vae": "combined_full180",
                    "vae_pure": "pure_forward/joint10",
                    "forward_prob_joint10": "pure_forward/joint10",
                    "union_h2_180": "combined_full180",
                    "guard": "no2_guard/full",
                    "vae_guard": "no2_guard/full",
                },
                "note": "不同轮次，绝对数不可直接横比；以同轮 VAE 为基线。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    arms = {
        "vae": ("vae", "PCA-VAE（combined 轮）"),
        "forward": ("forward_prob_joint10_pure", "仅正模型 joint10"),
        "union": ("union_h2_180", "PCA-VAE+正模型 union"),
        "guard": ("guard", "NO2 guard"),
        "vae_pure": ("vae_pure", "PCA-VAE（pure 轮）"),
        "vae_guard": ("vae_guard", "PCA-VAE（guard 轮）"),
    }
    for name, (prefix, label) in arms.items():
        report_run(
            OUTPUT / "predictions.npz",
            OUTPUT / name,
            prefix,
            10.0,
            f"comparison_20261127_{name}",
            label,
        )

    rows = []
    for name in arms:
        table = pd.read_csv(OUTPUT / name / "metrics.csv")
        row = table[table["prediction"] == "ensemble"].iloc[0].to_dict()
        row["arm"] = name
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    columns = [
        "arm",
        "macro_MAPE_percent",
        "macro_three_sigma_percent",
        "macro_NRMSE",
        "H2_MAPE_percent",
        "H2_three_sigma_percent",
        "NH3_MAPE_percent",
        "NH3_three_sigma_percent",
        "NO2_MAPE_percent",
        "NO2_three_sigma_percent",
    ]
    print(summary[columns].to_string(index=False))
    print("wrote", OUTPUT)


if __name__ == "__main__":
    main()
