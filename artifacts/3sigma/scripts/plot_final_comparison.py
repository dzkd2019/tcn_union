"""生成 OOF 与锁定测试集上三种方法的逐气体 3σ 对比图（PNG+SVG）。

三种方法：
- 旧协议基线：none 目标 + 低浓度加权，3 seed 均值集成；
- TCN 四配置集成：log / log+noise0.2 / 逐气体目标 / none 基线，各 3 seed；
- TCN 四配置 + Ridge 混合（w=0.5）：本分支最终候选。

OOF 数值来自 artifacts/3sigma/experiments_summary.csv，测试集数值来自
artifacts/3sigma/hybrid_eval/ 与旧的 artifacts/variance_search/pg422_eval/。
"""
from __future__ import annotations

import csv
import json
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
GASES = ("H2", "NH3", "NO2")
METHODS = ("旧协议基线", "TCN 四配置集成", "TCN+Ridge 混合")
COLORS = ("#9AA5AD", "#1F6F78", "#C0504A")


def _sigma(frame: list[dict[str, str]]) -> dict[str, float]:
    """从含 _true/_pred 的预测行计算逐气体 3σ 与宏平均。"""
    output = {}
    for gas in GASES:
        truth = np.array([float(row[f"{gas}_true"]) for row in frame])
        prediction = np.array([float(row[f"{gas}_pred"]) for row in frame])
        relative = 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)
        output[gas] = float(3.0 * np.std(relative, ddof=1))
    output["macro"] = float(np.mean([output[gas] for gas in GASES]))
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _oof_values() -> dict[str, dict[str, float]]:
    with (ROOT / "artifacts/3sigma/experiments_summary.csv").open(encoding="utf-8") as handle:
        rows = {row["config"]: row for row in csv.DictReader(handle)}
    mapping = {
        "旧协议基线": "none+低浓度加权（3 seed）",
        "TCN 四配置集成": "TCN 四配置集成（3 seed×4）",
        "TCN+Ridge 混合": "TCN 四配置 + Ridge（w=0.5）",
    }
    result = {}
    for method, config in mapping.items():
        row = rows[config]
        result[method] = {**{gas: float(row[gas]) for gas in GASES}, "macro": float(row["macro"])}
    return result


def _test_values() -> dict[str, dict[str, float]]:
    old = _read_csv(ROOT / "artifacts/variance_search/pg422_eval/test_predictions.csv")
    baseline = [
        row
        for row in old
        if row["model"] == "candidate"
        and str(row["seed"]) == "ensemble"
        and int(row["window_seconds"]) == 10
    ]
    report = json.loads(
        (ROOT / "artifacts/3sigma/hybrid_eval/evaluation.json").read_text(encoding="utf-8")
    )
    hybrid_rows = [
        item
        for item in report["three_sigma"]
        if item["model"] == "candidate" and item["seed"] == "ensemble"
    ][0]
    test_rows = _read_csv(ROOT / "artifacts/3sigma/hybrid_eval/test_predictions.csv")
    tcn_seed_rows = [
        row
        for row in test_rows
        if row["model"] == "candidate" and str(row["seed"]) != "ensemble"
    ]
    files = sorted({row["file"] for row in tcn_seed_rows})
    tcn_mean_rows = []
    for name in files:
        subset = [row for row in tcn_seed_rows if row["file"] == name]
        row = {"file": name}
        for gas in GASES:
            row[f"{gas}_true"] = float(subset[0][f"{gas}_true"])
            row[f"{gas}_pred"] = float(np.mean([float(item[f"{gas}_pred"]) for item in subset]))
        tcn_mean_rows.append(row)
    return {
        "旧协议基线": _sigma(baseline),
        "TCN 四配置集成": _sigma(tcn_mean_rows),
        "TCN+Ridge 混合": {**{gas: float(hybrid_rows[gas]) for gas in GASES}, "macro": float(hybrid_rows["macro"])},
    }


def _plot(oof: dict[str, dict[str, float]], test: dict[str, dict[str, float]], output: Path) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(10.6, 3.9), constrained_layout=True)
    categories = (*GASES, "macro")
    for panel, title, values in zip(
        panels, ("OOF（50 个开发实验）", "锁定测试集（12 个实验，一次性评估）"), (oof, test), strict=True
    ):
        x = np.arange(len(categories))
        width = 0.26
        for offset, method, color in zip(
            (-width, 0.0, width), METHODS, COLORS, strict=True
        ):
            heights = [values[method][category] for category in categories]
            bars = panel.bar(x + offset, heights, width, label=method, color=color, edgecolor="white", linewidth=0.4)
            for bar, height in zip(bars, heights):
                panel.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 0.15,
                    f"{height:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=5.9,
                )
        for target, style in ((10.0, "--"), (5.0, ":")):
            panel.axhline(target, color="#C0504D", linestyle=style, linewidth=0.9)
        panel.set_xticks(x, categories)
        panel.set_ylabel("相对误差 3σ / %", fontsize=8.5)
        panel.set_title(title, fontsize=9.5, loc="left")
        panel.set_ylim(0, 19)
        panel.grid(axis="y", alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.legend(fontsize=6.8, frameon=False, loc="upper right")
    figure.suptitle("3σ 优化结果：OOF 与锁定测试集（虚线 10% 上限，点线 5% 目标）", fontsize=10.5)
    for suffix in ("png", "svg"):
        path = output / f"fig_3sigma_final_comparison.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    output = ROOT / "artifacts/3sigma/figs"
    output.mkdir(parents=True, exist_ok=True)
    oof = _oof_values()
    test = _test_values()
    for scope, values in (("OOF", oof), ("test", test)):
        for method in METHODS:
            summary = values[method]
            print(
                f"{scope:<4} {method:<14} "
                + "  ".join(f"{gas}={summary[gas]:.2f}" for gas in GASES)
                + f"  macro={summary['macro']:.2f}"
            )
    _plot(oof, test, output)


if __name__ == "__main__":
    main()
