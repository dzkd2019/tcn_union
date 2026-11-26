"""绘制 log/低浓度加权/边缘 VAE/逐气体头的留一消融。

以配置 A 为完整模型，每次去掉一个组件；只读取已有 selection/evaluation，
不重新训练。输出指标 CSV、PNG 和 SVG。
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

from tcn_union.visualization.plot_style import apply_style

GASES = ("H2", "NH3", "NO2")
GAS_COLORS = {"H2": "#2B6F8A", "NH3": "#D5A03A", "NO2": "#C06043"}


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _oof(selection: dict) -> float:
    folds = next(iter(selection["selected"].values()))["fold_scores"]
    return float(np.mean([fold["macro_nmae"] for fold in folds]))


def _test(evaluation: dict) -> dict:
    metrics = next(m for m in evaluation["metrics"] if m.get("seed") == "ensemble")
    sigma = next(s for s in evaluation["three_sigma"] if s.get("seed") == "ensemble")
    return {
        "macro": sigma["macro"],
        "nmae": metrics["macro_NMAE"] * 100,
        "mape": metrics["macro_MAPE_percent"],
        "sigma": {gas: sigma[gas] for gas in GASES},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/ablations/component"))
    args = parser.parse_args()

    root = args.root
    specs = [
        ("完整 A", "inverse/eval_10s/final/p10/selection.json", "analysis/ablations/per_gas_head/per_gas_eval/evaluation.json", None),
        ("去掉 log", "analysis/ablations/component/no_log/selection.json", "analysis/ablations/component/no_log_eval/evaluation.json", "log"),
        ("去掉低浓度加权", "analysis/ablations/component/no_weight/selection.json", "analysis/ablations/component/no_weight_eval/evaluation.json", "weight"),
        ("去掉边缘 VAE", "analysis/ablations/component/no_edge/selection.json", "analysis/ablations/component/no_edge_eval/evaluation.json", "edge"),
        ("去掉逐气体头", "analysis/ablations/per_gas_head/shared/selection.json", "analysis/ablations/per_gas_head/shared_eval/evaluation.json", "head"),
    ]
    rows = []
    for label, selection, evaluation, key in specs:
        item = {"label": label, "key": key, "oof": _oof(_load(root / selection)), **_test(_load(root / evaluation))}
        rows.append(item)
    full = rows[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "component_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "variant", "oof_macro_sigma", "test_macro_sigma", "test_macro_nmae", "test_macro_mape",
            *[f"{gas}_test_sigma" for gas in GASES], "delta_oof", "delta_test",
        ])
        for item in rows:
            writer.writerow([
                item["label"],
                f"{item['oof']:.4f}",
                f"{item['macro']:.4f}",
                f"{item['nmae']:.4f}",
                f"{item['mape']:.4f}",
                *[f"{item['sigma'][gas]:.4f}" for gas in GASES],
                f"{item['oof'] - full['oof']:+.4f}",
                f"{item['macro'] - full['macro']:+.4f}",
            ])

    apply_style()
    figure, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.25), constrained_layout=True)

    labels = [item["label"] for item in rows]
    x = np.arange(len(rows))
    width = 0.36
    oof = [item["oof"] for item in rows]
    test = [item["macro"] for item in rows]
    bars_oof = left.bar(x - width / 2, oof, width, color="#8A9BA8", label="OOF 逐折验证 3σ")
    bars_test = left.bar(x + width / 2, test, width, color="#167D8D", label="测试宏 3σ")
    for bars in (bars_oof, bars_test):
        for bar in bars:
            left.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.22, f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=6.6)
    left.set_xticks(x, labels, rotation=20, ha="right")
    left.set_ylabel("宏平均 3σ / %")
    left.set_title("(a) 完整 A 与留一结果")
    left.set_ylim(0, 16.2)
    left.grid(axis="y", alpha=0.25)
    left.legend(loc="upper right", frameon=False)

    removed = rows[1:]
    x = np.arange(len(removed))
    width = 0.25
    for index, gas in enumerate(GASES):
        deltas = [item["sigma"][gas] - full["sigma"][gas] for item in removed]
        bars = right.bar(x + (index - 1) * width, deltas, width, color=GAS_COLORS[gas], label=gas)
        for bar, value in zip(bars, deltas, strict=True):
            right.text(bar.get_x() + bar.get_width() / 2, value + (0.12 if value >= 0 else -0.12), f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=6.3)
    right.axhline(0, color="#444444", linewidth=0.8)
    right.set_xticks(x, [item["label"] for item in removed], rotation=20, ha="right")
    right.set_ylabel("相对完整 A 的 Δ3σ / pp")
    right.set_title("(b) 逐气体：正值表示去掉后变差")
    right.set_ylim(-3.2, 7.8)
    right.grid(axis="y", alpha=0.25)
    right.legend(loc="upper left", frameon=False, ncol=3)

    figure.suptitle("组件留一消融：从配置 A 中分别去掉一项（3 seed，锁定测试 12 例）", fontsize=9.3)
    for suffix in ("png", "svg"):
        figure.savefig(args.output_dir / f"fig_component_ablation.{suffix}", dpi=300, bbox_inches="tight")
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'fig_component_ablation.png'}")


if __name__ == "__main__":
    main()
