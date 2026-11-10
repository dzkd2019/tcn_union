"""分析 PCA 主成分数 K 的解释方差与重构误差。

五折协议：只用 40 个训练实验拟合 PCA，在 10 个验证实验上计算重构误差。
结果用于说明为什么最终选用 K=6，而不是仅凭经验。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from tcn_union.training.train import _read_manifest

plt.rcParams["font.family"] = "Noto Serif CJK SC"
plt.rcParams["font.serif"] = ["Noto Serif CJK SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["savefig.dpi"] = 300

AXES = ("H2", "NH3", "NO2")
COLORS = {"H2": "#0072B2", "NH3": "#E69F00", "NO2": "#D55E00", "macro": "#333333"}


def analyze(rows: list[dict], max_components: int) -> list[dict]:
    responses = np.stack(
        [np.asarray(row["_response"], dtype=np.float64)[:51].T for row in rows]
    )
    records: list[dict] = []
    for gas_index, gas in enumerate(AXES):
        for k in range(1, max_components + 1):
            explained = []
            reconstruction = []
            for fold in range(5):
                train = responses[
                    [int(row["fold"]) != fold for row in rows], gas_index, :
                ]
                validation = responses[
                    [int(row["fold"]) == fold for row in rows], gas_index, :
                ]
                pca = PCA(n_components=k, svd_solver="full").fit(train)
                reconstructed = pca.inverse_transform(pca.transform(validation))
                rmse = np.sqrt(np.mean((reconstructed - validation) ** 2))
                explained.append(float(pca.explained_variance_ratio_.sum()))
                reconstruction.append(float(100.0 * rmse / np.ptp(train)))
            records.append(
                {
                    "gas": gas,
                    "k": k,
                    "explained_variance": float(np.mean(explained)),
                    "reconstruction_percent": float(np.mean(reconstruction)),
                }
            )
    return records


def _plot(records: list[dict], output_dir: Path) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    for gas in AXES:
        subset = [row for row in records if row["gas"] == gas]
        ks = [row["k"] for row in subset]
        panels[0].plot(
            ks,
            [100.0 * row["explained_variance"] for row in subset],
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=COLORS[gas],
            label=gas,
        )
        panels[1].plot(
            ks,
            [row["reconstruction_percent"] for row in subset],
            marker="o",
            markersize=3,
            linewidth=1.3,
            color=COLORS[gas],
            label=gas,
        )
    macro = []
    for k in sorted({row["k"] for row in records}):
        subset = [row for row in records if row["k"] == k]
        macro.append(
            {
                "k": k,
                "explained_variance": float(
                    np.mean([row["explained_variance"] for row in subset])
                ),
                "reconstruction_percent": float(
                    np.mean([row["reconstruction_percent"] for row in subset])
                ),
            }
        )
    panels[0].plot(
        [row["k"] for row in macro],
        [100.0 * row["explained_variance"] for row in macro],
        color=COLORS["macro"],
        linewidth=1.8,
        linestyle="--",
        label="三气体平均",
    )
    panels[1].plot(
        [row["k"] for row in macro],
        [row["reconstruction_percent"] for row in macro],
        color=COLORS["macro"],
        linewidth=1.8,
        linestyle="--",
        label="三气体平均",
    )
    for panel in panels:
        panel.axvline(6, color="#888888", linestyle=":", linewidth=1.0)
        panel.set_xlabel("主成分数 K", fontsize=8.5)
        panel.grid(alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.legend(fontsize=7, frameon=False)
    panels[0].set_ylabel("累计解释方差 / %", fontsize=8.5)
    panels[1].set_ylabel("验证集重构 RMSE / 量程 %", fontsize=8.5)
    panels[0].set_title("(a) 累计解释方差", fontsize=9.2, loc="left")
    panels[1].set_title("(b) 五折验证重构误差", fontsize=9.2, loc="left")
    panels[0].annotate(
        "K=6",
        xy=(6, 100.0 * macro[5]["explained_variance"]),
        xytext=(6.4, 99.85),
        fontsize=7,
        color="#555555",
    )
    panels[1].annotate(
        "K=6",
        xy=(6, macro[5]["reconstruction_percent"]),
        xytext=(6.4, macro[5]["reconstruction_percent"] * 1.15),
        fontsize=7,
        color="#555555",
    )
    for suffix in ("png", "pdf"):
        path = output_dir / f"fig_pca_component_selection.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 PCA 主成分数选择")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-components", type=int, default=15)
    args = parser.parse_args()
    rows = _read_manifest(args.manifest)
    dev_rows = [row for row in rows if row["split"] == "development"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = analyze(dev_rows, args.max_components)
    with (args.output_dir / "pca_components.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _plot(records, args.output_dir)
    for gas in AXES:
        subset = [row for row in records if row["gas"] == gas]
        for k in (4, 6, 8):
            item = next(row for row in subset if row["k"] == k)
            print(
                f"{gas} K={k}: EV={100*item['explained_variance']:.4f}% "
                f"recon={item['reconstruction_percent']:.4f}%"
            )


if __name__ == "__main__":
    main()
