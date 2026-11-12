"""绘制逐气体输出头 vs 共享头的受控消融结果。

只读取本轮消融的 selection/evaluation JSON，不重新训练；输出 CSV 和 PNG/SVG。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GASES = ("H2", "NH3", "NO2")
COLORS = {"shared": "#8A9BA8", "per_gas": "#167D8D"}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _oof_macro(selection: dict) -> float:
    folds = next(iter(selection["selected"].values()))["fold_scores"]
    return float(np.mean([fold["macro_nmae"] for fold in folds]))


def _test_ensemble(evaluation: dict) -> dict:
    metrics = next(m for m in evaluation["metrics"] if m.get("seed") == "ensemble")
    sigma = next(s for s in evaluation["three_sigma"] if s.get("seed") == "ensemble")
    return {
        "macro_nmae": metrics["macro_NMAE"],
        "macro_mape": metrics["macro_MAPE_percent"],
        "macro_sigma": sigma["macro"],
        "sigma": {gas: sigma[gas] for gas in GASES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", type=Path, default=Path("artifacts/per_gas_head_ablation/shared"))
    parser.add_argument("--per-gas-dir", type=Path, default=Path("artifacts/3sigma_noridge/final/p10"))
    parser.add_argument("--shared-eval", type=Path, default=Path("artifacts/per_gas_head_ablation/shared_eval/evaluation.json"))
    parser.add_argument("--per-gas-eval", type=Path, default=Path("artifacts/per_gas_head_ablation/per_gas_eval/evaluation.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/per_gas_head_ablation"))
    args = parser.parse_args()

    shared = {"oof": _oof_macro(_load(args.shared_dir / "selection.json")), **_test_ensemble(_load(args.shared_eval))}
    per_gas = {"oof": _oof_macro(_load(args.per_gas_dir / "selection.json")), **_test_ensemble(_load(args.per_gas_eval))}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "ablation_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["head", "oof_macro_sigma", "test_macro_sigma", "test_macro_nmae", "test_macro_mape", *[f"{g}_test_sigma" for g in GASES]])
        for name, item in [("shared", shared), ("per_gas", per_gas)]:
            writer.writerow([
                name,
                f"{item['oof']:.4f}",
                f"{item['macro_sigma']:.4f}",
                f"{item['macro_nmae']:.6f}",
                f"{item['macro_mape']:.4f}",
                *[f"{item['sigma'][g]:.4f}" for g in GASES],
            ])

    plt.rcParams.update({
        "font.family": "Noto Serif CJK SC",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
    })
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.4, 3.05), constrained_layout=True)

    x = np.arange(2)
    width = 0.33
    oof_values = [shared["oof"], per_gas["oof"]]
    test_values = [shared["macro_sigma"], per_gas["macro_sigma"]]
    bars_shared = left.bar(x - width / 2, [oof_values[0], test_values[0]], width, color=COLORS["shared"], label="共享头")
    bars_per_gas = left.bar(x + width / 2, [oof_values[1], test_values[1]], width, color=COLORS["per_gas"], label="逐气体头")
    left.set_xticks(x, ["OOF（50 个开发实验）", "锁定测试（12 个实验）"])
    left.set_ylabel("宏平均 3σ / %")
    left.set_title("(a) 整体：宏平均 3σ")
    left.set_ylim(0, 14.5)
    left.grid(axis="y", alpha=0.25)
    left.legend(loc="upper right", frameon=False)
    for bars in (bars_shared, bars_per_gas):
        for bar in bars:
            left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7)

    x = np.arange(len(GASES))
    shared_gas = [shared["sigma"][g] for g in GASES]
    per_gas_gas = [per_gas["sigma"][g] for g in GASES]
    bars_shared = right.bar(x - width / 2, shared_gas, width, color=COLORS["shared"], label="共享头")
    bars_per_gas = right.bar(x + width / 2, per_gas_gas, width, color=COLORS["per_gas"], label="逐气体头")
    right.set_xticks(x, list(GASES))
    right.set_ylabel("测试 3σ / %")
    right.set_title("(b) 锁定测试：逐气体 3σ")
    right.set_ylim(0, 15.5)
    right.grid(axis="y", alpha=0.25)
    for bars in (bars_shared, bars_per_gas):
        for bar in bars:
            right.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.25, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=7.3)

    figure.suptitle("逐气体输出头 vs 共享头：同一配置 A 的受控消融", fontsize=9.5)
    for suffix in ("png", "svg"):
        figure.savefig(args.output_dir / f"fig_per_gas_head_ablation.{suffix}", dpi=300, bbox_inches="tight")
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'fig_per_gas_head_ablation.png'}")


if __name__ == "__main__":
    main()
