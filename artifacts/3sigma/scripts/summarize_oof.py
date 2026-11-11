"""汇总 exp/3sigma 的 OOF 实验，输出 CSV 与宏平均 3σ 对比图（PNG+SVG）。

只用开发集 OOF 预测；读取当前分支 artifacts/3sigma 下的各配置 oof.csv，按配置
内多 seed 预测平均后计算逐气体 3σ，再与其他配置做横向对比。混合模型结果读取
``compute_hybrid_oof.py`` 写出的 hybrid_oof.csv。

用法（项目根目录）：
    .venv/bin/python artifacts/3sigma/scripts/summarize_oof.py

产物：
    artifacts/3sigma/experiments_summary.csv
    artifacts/3sigma/figs/fig_3sigma_oof_configs.png / .svg
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": ["Source Han Sans CN", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
    }
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.evaluation.evaluate import AXES

GASES = AXES
SEEDS = (20260923, 20260924, 20260925)
SEEDS6 = (20260923, 20260924, 20260925, 20260926, 20260927, 20260928)
OUTPUT_DIR = ROOT / "artifacts/3sigma"
FIGS = OUTPUT_DIR / "figs"

# 名称 -> (模式子目录, 每 seed 的 oof.csv 相对路径列表)
CONFIGS: list[tuple[str, str, list[str]]] = [
    (
        "noise_gan 基线（旧，单 seed）",
        "noise_gan",
        ["artifacts/variance_search/pg422_final/noise_gan/oof.csv"],
    ),
    (
        "none+低浓度加权（3 seed）",
        "noise_gan",
        [
            "artifacts/variance_search/pg422_final/noise_gan/oof.csv",
            "artifacts/3sigma/baseline_pg422/seed_20260924/noise_gan/oof.csv",
            "artifacts/3sigma/baseline_pg422/seed_20260925/noise_gan/oof.csv",
        ],
    ),
    (
        "log 目标（3 seed）",
        "noise_gan",
        [f"artifacts/3sigma/log_p0/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    ),
    (
        "log + noise0.2（3 seed）",
        "noise_gan",
        [f"artifacts/3sigma/log_p0_noise02/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    ),
    (
        "log + 加权 4,2,2（3 seed）",
        "noise_gan",
        [f"artifacts/3sigma/log_pg422/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    ),
    (
        "逐气体目标 log,none,log（3 seed）",
        "noise_gan",
        [f"artifacts/3sigma/pergas_p0/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    ),
    (
        "log + EMA0.99（3 seed）",
        "noise_gan",
        [f"artifacts/3sigma/log_p0_ema99/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    ),
    (
        "log + synth×5（3 seed）",
        "noise_gan",
        [f"artifacts/3sigma/log_p0_synth5/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS],
    ),
    (
        "仅噪声增强（3 seed）",
        "noise",
        [f"artifacts/3sigma/noiseonly3_log/seed_{seed}/noise/oof.csv" for seed in SEEDS],
    ),
]


def _load_oof(path: Path) -> dict[str, np.ndarray]:
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "file": [row["file"] for row in rows],
        "truth": np.array([[float(row[f"{gas}_true"]) for gas in GASES] for row in rows]),
        "pred": np.array([[float(row[f"{gas}_pred"]) for gas in GASES] for row in rows]),
    }


def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> tuple[dict[str, float], float]:
    sigma = {}
    for index, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, index] - truth[:, index]) / np.maximum(
            np.abs(truth[:, index]), 1e-9
        )
        sigma[gas] = float(3.0 * np.std(relative, ddof=1))
    return sigma, float(np.mean(list(sigma.values())))


def _ensemble(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    reference = _load_oof(paths[0])
    predictions = []
    for path in paths:
        frame = _load_oof(path)
        if frame["file"] != reference["file"]:
            raise ValueError(f"OOF 实验顺序不一致：{path}")
        predictions.append(frame["pred"])
    return reference["truth"], np.mean(predictions, axis=0)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for label, _mode, paths in CONFIGS:
        truth, prediction = _ensemble([ROOT / path for path in paths])
        sigma, macro = _three_sigma(truth, prediction)
        rows.append({"config": label, **{gas: sigma[gas] for gas in GASES}, "macro": macro})

    # TCN 四配置等权集成（对应最终部署候选的 TCN 侧，6 seed/配置）。
    tcn_sources = [
        [f"artifacts/3sigma/log_p0/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS6],
        [f"artifacts/3sigma/log_p0_noise02/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS6],
        [f"artifacts/3sigma/pergas_p0/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS6],
        [f"artifacts/3sigma/pergas_gw112/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS6],
        ["artifacts/variance_search/pg422_final/noise_gan/oof.csv"]
        + [f"artifacts/3sigma/baseline_pg422/seed_{seed}/noise_gan/oof.csv" for seed in SEEDS6[1:]],
    ]
    tcn_predictions = []
    reference_truth = None
    for paths in tcn_sources:
        truth, prediction = _ensemble([ROOT / path for path in paths])
        reference_truth = truth
        tcn_predictions.append(prediction)
    tcn_mean = np.mean(tcn_predictions, axis=0)
    sigma, macro = _three_sigma(reference_truth, tcn_mean)
    rows.append({"config": "TCN 五配置集成（6 seed×5）", **{gas: sigma[gas] for gas in GASES}, "macro": macro})

    # 混合模型：TCN 四配置平均 + Ridge（w=0.5）。
    with (OUTPUT_DIR / "hybrid_oof.csv").open(encoding="utf-8") as handle:
        hybrid_rows = list(csv.DictReader(handle))
    hybrid_truth = np.array([[float(row[f"{gas}_true"]) for gas in GASES] for row in hybrid_rows])
    hybrid_pred = np.array([[float(row[f"{gas}_pred"]) for gas in GASES] for row in hybrid_rows])
    sigma, macro = _three_sigma(hybrid_truth, hybrid_pred)
    rows.append({"config": "TCN 五配置 + Ridge（6 seed, w=0.4, clip）", **{gas: sigma[gas] for gas in GASES}, "macro": macro})

    summary_path = OUTPUT_DIR / "experiments_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 横向条形图：宏平均 3σ（越左越好），并标注 5% / 10% 目标线。
    ordered = sorted(rows, key=lambda row: float(row["macro"]))
    labels = [str(row["config"]) for row in ordered]
    values = [float(row["macro"]) for row in ordered]
    colors = ["#C0504D" if value > 10.0 else "#1F6F78" for value in values]
    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    bars = axis.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
    for bar, value in zip(bars, values):
        axis.text(value + 0.12, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontsize=7)
    for target, style, name in ((10.0, "--", "上限 10%"), (5.0, ":", "目标 5%")):
        axis.axvline(target, color="#C0504D", linestyle=style, linewidth=1.0)
        axis.text(target, len(labels) - 0.4, name, color="#C0504D", fontsize=7, ha="center")
    axis.set_xlabel("OOF 宏平均相对误差 3σ / %", fontsize=9)
    axis.set_title("10 s 模型 OOF（50 实验）宏平均 3σ：配置与集成对比", fontsize=10)
    axis.tick_params(labelsize=7.5)
    axis.grid(axis="x", alpha=0.2, linewidth=0.5)
    axis.set_xlim(0, max(values) * 1.15)
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_3sigma_oof_configs.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)
    print(f"Wrote {summary_path}")
    for row in rows:
        print(f"{row['config']:<30} macro={float(row['macro']):.2f}")


if __name__ == "__main__":
    main()
