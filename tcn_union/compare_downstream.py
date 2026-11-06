"""汇总四种生成式增强的下游 TCN 结果并绘图。"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Noto Serif CJK SC"
plt.rcParams["font.serif"] = ["Noto Serif CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300

# 目录名与显示名分开：最终 PCA+VAE 在 downstream_comparison/vae
METHOD_DIRS = {"GAN": "gan", "PCA": "pca", "VAE": "direct_vae", "PCA+VAE": "vae"}
COLORS = {
    "Clean": "#8C8C8C",
    "GAN": "#D55E00",
    "PCA": "#0072B2",
    "VAE": "#009E73",
    "PCA+VAE": "#CC79A7",
}
AXES = ("H2", "NH3", "NO2")


def _baseline_metrics(evaluation: dict) -> dict:
    for item in evaluation["metrics"]:
        if item["model"] == "baseline" and item["seed"] == "ensemble":
            return item
    raise ValueError("evaluation 中没有 baseline ensemble")


def _candidate_metrics(evaluation: dict) -> dict:
    for item in evaluation["metrics"]:
        if item["model"] == "candidate" and item["seed"] == "ensemble":
            return item
    raise ValueError("evaluation 中没有 candidate ensemble")


def _oof_macro(path: Path) -> float:
    selection = json.loads((path / "selection.json").read_text())
    selected = selection["selected"]["gan"]
    return float(selected["fold_summary"]["macro_nmae"])


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总生成式增强下游结果")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    baseline = None
    for display, directory in METHOD_DIRS.items():
        method_root = args.root / directory
        evaluation = json.loads((method_root / "evaluation" / "evaluation.json").read_text())
        if baseline is None:
            baseline = _baseline_metrics(evaluation)
        candidate = _candidate_metrics(evaluation)
        rows.append(
            {
                "method": display,
                "oof_macro_nmae": _oof_macro(method_root),
                "test_macro_nmae": candidate["macro_NMAE"],
                "test_macro_mape_percent": candidate["macro_MAPE_percent"],
                **{f"{gas}_rmse": candidate[gas]["RMSE"] for gas in AXES},
                **{f"{gas}_nmae": candidate[gas]["NMAE"] for gas in AXES},
            }
        )
    baseline_row = {
        "method": "Clean",
        "oof_macro_nmae": 0.027097,
        "test_macro_nmae": baseline["macro_NMAE"],
        "test_macro_mape_percent": baseline["macro_MAPE_percent"],
        **{f"{gas}_rmse": baseline[gas]["RMSE"] for gas in AXES},
        **{f"{gas}_nmae": baseline[gas]["NMAE"] for gas in AXES},
    }

    with (args.output_dir / "downstream_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_row))
        writer.writeheader()
        writer.writerow(baseline_row)
        writer.writerows(rows)

    labels = ["Clean"] + [row["method"] for row in rows]
    colors = [COLORS[label] for label in labels]
    all_rows = [baseline_row, *rows]

    figure, panels = plt.subplots(2, 3, figsize=(9.2, 6.0), constrained_layout=True)
    panel = panels[0, 0]
    values = [row["oof_macro_nmae"] for row in all_rows]
    bars = panel.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=6.5)
    panel.set_title("(a) 开发集 OOF macro-NMAE", fontsize=8.8)
    panel.tick_params(axis="x", rotation=25, labelsize=7)
    panel.set_ylabel("越低越好", fontsize=7.5)

    panel = panels[0, 1]
    values = [row["test_macro_nmae"] for row in all_rows]
    bars = panel.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=6.5)
    panel.set_title("(b) 测试集 macro-NMAE", fontsize=8.8)
    panel.tick_params(axis="x", rotation=25, labelsize=7)
    panel.set_ylabel("越低越好", fontsize=7.5)

    panel = panels[0, 2]
    values = [row["test_macro_mape_percent"] for row in all_rows]
    bars = panel.bar(labels, values, color=colors, width=0.62)
    for bar, value in zip(bars, values, strict=True):
        panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}%", ha="center", va="bottom", fontsize=6.5)
    panel.set_title("(c) 测试集 macro-MAPE", fontsize=8.8)
    panel.tick_params(axis="x", rotation=25, labelsize=7)
    panel.set_ylabel("百分比 / %", fontsize=7.5)

    for col, gas in enumerate(AXES):
        panel = panels[1, col]
        values = [row[f"{gas}_rmse"] for row in all_rows]
        bars = panel.bar(labels, values, color=colors, width=0.62)
        for bar, value in zip(bars, values, strict=True):
            panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3g}", ha="center", va="bottom", fontsize=6.2)
        panel.set_title(f"({chr(100 + col)}) {gas} 测试 RMSE", fontsize=8.8)
        panel.tick_params(axis="x", rotation=25, labelsize=7)
        panel.set_ylabel("原始浓度单位", fontsize=7.2)

    figure.suptitle("不同生成式增强方案的下游 TCN 表现", fontsize=11, fontweight="bold")
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"fig_downstream_comparison.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


if __name__ == "__main__":
    main()
