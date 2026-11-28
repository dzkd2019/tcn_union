"""并列展示三种保留增强方式：PCA-VAE、传感器模型生成、组合使用。

三种方式都使用完整训练协议（10 s、多个 seed、两套低浓度配置、6 checkpoint）。
三者都在同一开发集上给出 OOF 预测，按绝对指标直接比较。
组合使用的 H₂ 来自组合分支，NH₃/NO₂ 采用 PCA-VAE 的多个 seed 平均预测。
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


def _seed_mean(payload, arm: str) -> dict[str, np.ndarray]:
    """把逐 seed/配置的预测取平均，返回 seed 级和总体平均。"""
    keys = [key for key in payload.files if key.startswith(f"{arm}|")]
    seeds = sorted({key.split("|")[1] for key in keys})
    result: dict[str, np.ndarray] = {}
    for seed in seeds:
        seed_keys = [key for key in keys if key.split("|")[1] == seed]
        result[f"{seed}"] = np.mean([payload[key] for key in seed_keys], axis=0)
    result["seed_mean"] = np.mean([payload[key] for key in keys], axis=0)
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    vae = np.load(BASE / "protocols/full/predictions.npz")
    joint = np.load(BASE / "pure_forward/joint10/predictions.npz")
    guard = np.load(BASE / "no2_guard/full/predictions.npz")
    truth = vae["truth"].astype(np.float64)
    file_ids = vae["file_id"].astype(int)

    payload: dict[str, np.ndarray] = {
        "truth": truth,
        "file_id": file_ids,
    }
    for seed in [key.split("_s")[-1] for key in vae.files if key.startswith("vae_s")]:
        payload[f"vae_s{seed}"] = vae[f"vae_s{seed}"].astype(np.float64)
    payload["vae_seed_mean"] = vae["vae_seed_mean"].astype(np.float64)
    for seed in [
        key.split("_s")[-1]
        for key in joint.files
        if key.startswith("forward_prob_joint10_s")
    ]:
        payload[f"forward_s{seed}"] = joint[
            f"forward_prob_joint10_s{seed}"
        ].astype(np.float64)
    payload["forward_seed_mean"] = joint[
        "forward_prob_joint10_seed_mean"
    ].astype(np.float64)

    guard_data = _seed_mean(guard, "guard")
    vae_seed_mean = payload["vae_seed_mean"]
    for seed, prediction in guard_data.items():
        calibrated = prediction.copy()
        calibrated[:, 1:] = vae_seed_mean[:, 1:]
        payload[
            "guard_seed_mean" if seed == "seed_mean" else f"guard_s{seed}"
        ] = calibrated.astype(np.float64)

    np.savez_compressed(OUTPUT / "predictions.npz", **payload)
    (OUTPUT / "run_config.json").write_text(
        json.dumps(
            {
                "n": int(len(truth)),
                "window_s": 10,
                "range_source": "development_50",
                "sources": {
                    "vae": "protocols/full",
                    "forward": "pure_forward/joint10",
                    "guard": "no2_guard/full",
                },
                "note": "三种方法均在同一开发集上给出 OOF 预测，按绝对指标直接比较。",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    arms = {
        "vae": ("vae", "PCA-VAE"),
        "forward": ("forward", "传感器模型生成"),
        "guard": ("guard", "组合使用"),
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
        row = table[table["prediction"] == "seed_mean"].iloc[0].to_dict()
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
