"""对比低浓度加权前后的逐气体 3σ。"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "Noto Serif CJK SC",
        "font.serif": ["Noto Serif CJK SC"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "savefig.dpi": 300,
    }
)

GASES = ("H2", "NH3", "NO2")
COLORS = {"baseline": "#9AA5AD", "weighted": "#1F6F78"}


def _metrics(oof_path: Path, test_path: Path, model: str) -> dict[str, dict[str, float]]:
    oof = pd.read_csv(oof_path)
    test = pd.read_csv(test_path)
    test = test[
        (test["model"] == model)
        & (test["window_seconds"].astype(int) == 10)
        & (test["seed"].astype(str) == "ensemble")
    ]
    output = {}
    for gas in GASES:
        for scope, frame in (("OOF", oof), ("test", test)):
            truth = frame[f"{gas}_true"].to_numpy(dtype=float)
            pred = frame[f"{gas}_pred"].to_numpy(dtype=float)
            rel = 100.0 * (pred - truth) / np.maximum(np.abs(truth), 1e-9)
            output[f"{scope}_{gas}"] = {
                "three_sigma": float(3.0 * np.std(rel, ddof=1)),
                "mape": float(np.mean(np.abs(rel))),
            }
    return output


def _plot(rows: list[dict], output: Path) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(10.4, 3.8), constrained_layout=True)
    for panel, scope in zip(panels, ("OOF", "test"), strict=True):
        x = np.arange(len(GASES))
        width = 0.34
        for offset, method in ((-width / 2, "baseline"), (width / 2, "weighted")):
            values = [
                next(row["three_sigma"] for row in rows if row["scope"] == scope and row["gas"] == gas and row["method"] == method)
                for gas in GASES
            ]
            bars = panel.bar(
                x + offset,
                values,
                width,
                label="原始" if method == "baseline" else "低浓度加权",
                color=COLORS[method],
                edgecolor="white",
                linewidth=0.5,
            )
            for bar, value in zip(bars, values):
                panel.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                )
        for target in (5.0, 10.0):
            panel.axhline(target, color="#C0504D", linestyle="--" if target == 5 else ":", linewidth=1.0)
        panel.set_xticks(x, GASES)
        panel.set_ylabel("相对误差 3σ / %", fontsize=8.5)
        panel.set_title(f"{scope}（{'50' if scope == 'OOF' else '12'} 个实验）", fontsize=9.5, loc="left")
        panel.set_ylim(0, 23)
        panel.grid(axis="y", alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.legend(fontsize=6.8, frameon=False, loc="upper right")
    figure.suptitle("低浓度加权前后的 3σ 对比（目标 5%，上限 10%）", fontsize=10.5)
    for suffix in ("png", "svg"):
        path = output / f"fig_variance_weighting.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="对比低浓度加权的 3σ")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/variance_search"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline = _metrics(
        Path("artifacts/final_vae_hybrid/noise_gan/oof.csv"),
        Path("artifacts/final_vae_hybrid/evaluation/test_predictions.csv"),
        "candidate",
    )
    weighted = _metrics(
        Path("artifacts/variance_search/pg422_final/noise_gan/oof.csv"),
        Path("artifacts/variance_search/pg422_eval/test_predictions.csv"),
        "candidate",
    )
    rows = []
    for scope in ("OOF", "test"):
        for gas in GASES:
            rows.append({"scope": scope, "gas": gas, "method": "baseline", **baseline[f"{scope}_{gas}"]})
            rows.append({"scope": scope, "gas": gas, "method": "weighted", **weighted[f"{scope}_{gas}"]})
    with (args.output_dir / "variance_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for scope in ("OOF", "test"):
        for method in ("baseline", "weighted"):
            subset = [row for row in rows if row["scope"] == scope and row["method"] == method]
            print(scope, method, "macro3", round(float(np.mean([row["three_sigma"] for row in subset])), 2))
    _plot(rows, args.output_dir)


if __name__ == "__main__":
    main()
