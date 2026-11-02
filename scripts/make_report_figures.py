"""Generate the schematic and summary figures used in the augmentation report."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "report" / "figures"


def _box(ax, x, y, w, h, text, color="#e8eefc", edge="#3557a5"):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.2, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9.2, color="#172033")


def _arrow(ax, start, end, color="#3557a5"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=12,
                                 linewidth=1.2, color=color, shrinkA=0, shrinkB=0))


def figure_pipeline() -> None:
    fig = plt.figure(figsize=(14.2, 6.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.45], wspace=0.12)
    ax = fig.add_subplot(gs[0, 0])
    time = np.linspace(0, 10, 51)
    signal = 0.02 + 0.38 * (1 - np.exp(-time / 2.2)) + 0.012 * time
    ax.plot(time, signal, color="#172033", linewidth=2.0, label="raw 10 s response")
    start, end = 8, 16
    ax.plot(time[start:end + 1], signal[start:end + 1], color="#d94841", linewidth=3.2)
    ax.plot([time[start], time[end]], [signal[start], signal[end]], "--", color="#d94841", linewidth=1.8)
    ax.axvspan(time[start], time[end], color="#d94841", alpha=0.10)
    ax.annotate("random internal segment\n(1-10 samples, ≤ 2 s)", xy=(time[start] + 0.4, signal[start] + 0.03),
                xytext=(0.4, 0.78), textcoords="axes fraction", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": "#d94841", "lw": 1.1})
    ax.annotate("linear bridge", xy=(time[end], signal[end]), xytext=(0.58, 0.55),
                textcoords="axes fraction", fontsize=9, color="#d94841",
                arrowprops={"arrowstyle": "->", "color": "#d94841", "lw": 1.1})
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("time (s)", fontsize=10)
    ax.set_ylabel("raw response (a.u.)", fontsize=10)
    ax.set_title("(a) Training-only cutout on the raw response", fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.04, "t = 0 and t = 10 s are never modified; label is unchanged.",
            transform=ax.transAxes, fontsize=8.5, color="#555")

    ax = fig.add_subplot(gs[0, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("(b) Multi-prefix supervision pipeline", fontsize=11, loc="left")
    _box(ax, 0.01, 0.56, 0.15, 0.17, "raw 10 s\nresponse\n(51 × 3)")
    _box(ax, 0.20, 0.56, 0.15, 0.17, "cutout\n(training only)", color="#fdeceb", edge="#c4504a")
    _box(ax, 0.39, 0.56, 0.17, 0.17, "raw_delta\nfeatures\n(6 × 51)")
    _box(ax, 0.60, 0.56, 0.16, 0.17, "causal TCN\n3 blocks, 16 ch")
    _box(ax, 0.80, 0.71, 0.15, 0.14, "main head\n@ 10 s\nL_main", color="#e9f5ee", edge="#2f7d4f")
    _box(ax, 0.80, 0.38, 0.15, 0.14, "aux head\n@ 5 s\n0.5 L_aux", color="#e9f5ee", edge="#2f7d4f")
    for x0, y0, x1, y1 in [
        (0.16, 0.645, 0.20, 0.645),
        (0.35, 0.645, 0.39, 0.645),
        (0.56, 0.645, 0.60, 0.645),
        (0.76, 0.68, 0.80, 0.78),
        (0.76, 0.61, 0.80, 0.45),
    ]:
        _arrow(ax, (x0, y0), (x1, y1))
    ax.text(0.80, 0.28, "total training loss", fontsize=9.2, weight="bold")
    ax.text(0.80, 0.22, "L_main + 0.5 L_aux", fontsize=10.2, weight="bold", color="#1f4e8c")
    ax.text(0.01, 0.30, "Validation / test / inference", fontsize=9.5, weight="bold")
    ax.text(0.01, 0.22, "clean 10 s response → TCN → main head → 3-seed average", fontsize=9)
    ax.text(0.01, 0.15, "No cutout, no auxiliary loss, no future samples.", fontsize=9, color="#555")

    fig.suptitle("Data augmentation for 10 s response acceleration", fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig1_augmentation_pipeline.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


def figure_mape() -> None:
    selection = json.loads((ROOT / "artifacts/augmentation_final/selection.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "artifacts/evaluation_final/evaluation.json").read_text(encoding="utf-8"))
    oof = {
        "5 s clean": selection["selected"]["5s"]["oof_metrics"]["macro_MAPE_percent"],
        "10 s clean": 4.98251237763299,
        "10 s cutout + 5 s aux": selection["selected"]["10s"]["oof_metrics"]["macro_MAPE_percent"],
    }
    test_ensemble = {}
    for item in evaluation["metrics"]:
        if item["seed"] == "ensemble":
            key = f'{item["model"]}_{item["window_seconds"]}'
            test_ensemble[key] = item["macro_MAPE_percent"]
    test = {
        "5 s clean": test_ensemble["baseline_5"],
        "10 s clean": test_ensemble["baseline_10"],
        "10 s cutout + 5 s aux": test_ensemble["augmented_10"],
    }
    labels = list(oof)
    colors = ["#9ca3af", "#f59e0b", "#2563eb"]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=False)
    for ax, values, title in [
        (axes[0], oof, "(a) Development OOF (50 runs)"),
        (axes[1], test, "(b) Locked test (12 runs, 3-seed ensemble)"),
    ]:
        bars = ax.bar(labels, [values[k] for k in labels], color=colors, width=0.62)
        for bar, key in zip(bars, labels, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
                    f"{values[key]:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(title, fontsize=10.5)
        ax.set_ylabel("macro-MAPE (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, max(values.values()) * 1.22)
        ax.tick_params(axis="x", labelsize=8.6)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"fig2_mape_summary.{suffix}", dpi=220 if suffix == "png" else None)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    figure_pipeline()
    figure_mape()
    print(f"Wrote figures to {OUT}")
