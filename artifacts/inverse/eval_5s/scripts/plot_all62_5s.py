"""5 秒前缀 TCN 的 62 点误差图（50 OOF + 12 锁定测试）。

产物：
- figs/fig_5s_relative_error_all62.{png,svg}
- figs/fig_5s_absolute_error_all62.{png,svg}
- figs/fig_5s_vs_10s_metrics.{png,svg}
- all62_predictions.csv, all62_error_stats.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.evaluation.evaluate import _manifest
from tcn_union.evaluation.ensemble_evaluate import _clip_to_development_range

GASES = ("H2", "NH3", "NO2")
COLORS = {"H2": "#1F77B4", "NH3": "#D98E04", "NO2": "#C0392B"}
MARKERS = {"H2": "o", "NH3": "^", "NO2": "s"}
SEEDS = (20260923, 20260924, 20260925)
BASE = ROOT / "artifacts/3sigma_5s"
BASELINE = ROOT / "artifacts/3sigma_noridge"
FIGS = BASE / "figs"
SECONDS = 5
GAS_RANGES = {"H2": 24000.0, "NH3": 40.0, "NO2": 8.0}

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def _three_sigma(values: np.ndarray) -> float:
    return float(3.0 * np.std(values, ddof=1))


def _stats(frame: pd.DataFrame, gas: str) -> np.ndarray:
    return 100.0 * (frame[f"{gas}_pred"].to_numpy(float) - frame[f"{gas}_true"].to_numpy(float)) / np.maximum(
        np.abs(frame[f"{gas}_true"].to_numpy(float)), 1e-9
    )


def _oof_frame() -> pd.DataFrame:
    dev_rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    files = [row["file"] for row in dev_rows]
    folds = np.array([int(row["fold"]) for row in dev_rows])
    truth = np.array([row["_labels"] for row in dev_rows], dtype=float)
    frames = [pd.read_csv(BASE / f"oof/seed_{seed}/noise_gan/oof.csv").set_index("file") for seed in SEEDS]
    tcn = np.stack(
        [
            np.mean([frame.loc[files, f"{gas}_pred"].to_numpy(float) for frame in frames], axis=0)
            for gas in GASES
        ],
        axis=1,
    )
    prediction = _clip_to_development_range(tcn, truth)
    return pd.DataFrame(
        {
            "file": files,
            "source": "OOF",
            "fold": folds,
            **{f"{gas}_true": truth[:, i] for i, gas in enumerate(GASES)},
            **{f"{gas}_pred": prediction[:, i] for i, gas in enumerate(GASES)},
        }
    )


def _test_frame() -> pd.DataFrame:
    frame = pd.read_csv(BASE / "evaluation/test_predictions.csv")
    frame = frame[(frame["model"] == "candidate") & (frame["seed"].astype(str) == "ensemble")].copy()
    if frame.empty:
        raise ValueError("test_predictions.csv 中没有 candidate/ensemble 行")
    frame["source"] = "锁定测试"
    return frame


def _combined(oof: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    columns = ["file", "source", *(f"{gas}_true" for gas in GASES), *(f"{gas}_pred" for gas in GASES)]
    frame = pd.concat([oof[columns], test[columns]], ignore_index=True)
    frame["run_id"] = frame["file"].str.extract(r"data/(\d+)\.csv").astype(int)
    frame["source_order"] = (frame["source"] == "锁定测试").astype(int)
    return frame.sort_values(["source_order", "run_id"]).reset_index(drop=True)


def _draw_relative(combined: pd.DataFrame, test_start: int) -> None:
    figure, panels = plt.subplots(2, 2, figsize=(10.4, 7.6), constrained_layout=True)
    pooled = np.concatenate([_stats(combined, gas) for gas in GASES])
    layout = [
        ("H2", _stats(combined, "H2"), MARKERS["H2"], COLORS["H2"]),
        ("NH3", _stats(combined, "NH3"), MARKERS["NH3"], COLORS["NH3"]),
        ("NO2", _stats(combined, "NO2"), MARKERS["NO2"], COLORS["NO2"]),
        ("三气体合并", pooled, "o", None),
    ]
    for index, (panel, (label, error, marker, color)) in enumerate(zip(panels.ravel(), layout)):
        if color is None:
            colors = [COLORS[gas] for gas in GASES for _ in range(len(combined))]
            panel.scatter(np.arange(len(pooled)), pooled, s=18, c=colors, alpha=0.85, edgecolors="white", linewidths=0.3)
        else:
            panel.scatter(np.arange(len(error)), error, s=20, marker=marker, c=color, alpha=0.85, edgecolors="white", linewidths=0.35)
        mean, sigma = float(np.mean(error)), float(np.std(error, ddof=1))
        panel.axhline(0.0, color="0.55", linewidth=0.7)
        panel.axhline(mean, color="0.1", linewidth=1.0)
        for k, style, shade in ((1.0, "--", "0.35"), (2.0, ":", "0.45"), (3.0, "-.", "0.55")):
            for sign in (-1.0, 1.0):
                panel.axhline(mean + sign * k * sigma, color=shade, linewidth=0.9, linestyle=style)
        panel.axhspan(mean - sigma, mean + sigma, color="0.85", alpha=0.2)
        start = test_start * 3 if color is None else test_start
        panel.axvspan(start - 0.5, len(error) - 0.5, color="#F2D9A6", alpha=0.35)
        test_sigma = _three_sigma(error[start:])
        panel.text(
            0.01, 0.98,
            f"n={len(error)}  均值={mean:+.2f}%  σ={sigma:.2f}%\n全部 3σ={3 * sigma:.2f}% | 锁定 3σ={test_sigma:.2f}%",
            transform=panel.transAxes, fontsize=6.6, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.85),
        )
        panel.set_xlim(-0.5, len(error) - 0.5)
        panel.grid(axis="y", alpha=0.2)
        panel.tick_params(labelsize=7.5)
        panel.set_title(f"({chr(97 + index)}) {label}", fontsize=9.5, loc="left")
    panels[1, 0].set_xlabel("实验序号（0–49 OOF，50–61 锁定测试）", fontsize=8.5)
    panels[1, 1].set_xlabel("实验序号（0–149 OOF，150–185 锁定测试）", fontsize=8.5)
    panels[0, 0].set_ylabel("相对误差 (%)", fontsize=8.5)
    panels[1, 0].set_ylabel("相对误差 (%)", fontsize=8.5)
    figure.suptitle("5 秒前缀 TCN（两配置 × 3 seed + 裁剪）：全部 62 组相对误差分布", fontsize=11)
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_5s_relative_error_all62.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def _draw_absolute(combined: pd.DataFrame, test_start: int) -> None:
    figure, panels = plt.subplots(1, 3, figsize=(11.0, 3.6), constrained_layout=True)
    for panel, gas in zip(panels, GASES):
        error = combined[f"{gas}_pred"].to_numpy(float) - combined[f"{gas}_true"].to_numpy(float)
        panel.scatter(np.arange(len(error)), error, s=20, marker=MARKERS[gas], c=COLORS[gas], alpha=0.85, edgecolors="white", linewidths=0.35)
        panel.axhline(0.0, color="0.55", linewidth=0.7)
        panel.axvspan(test_start - 0.5, len(error) - 0.5, color="#F2D9A6", alpha=0.35)
        panel.text(
            0.01, 0.98,
            f"n={len(error)}  中位={np.median(np.abs(error)):.2f}\n"
            f"p90={np.percentile(np.abs(error), 90):.2f}" + ("\n（原始单位）" if gas == "H2" else ""),
            transform=panel.transAxes, fontsize=6.6, va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.85),
        )
        panel.set_xlabel("实验序号（0–49 OOF，50–61 锁定测试）", fontsize=8)
        panel.set_title(gas, fontsize=9.5, loc="left")
        panel.grid(axis="y", alpha=0.2)
        panel.tick_params(labelsize=7.5)
        panel.set_ylabel("绝对误差（原始单位）", fontsize=8)
    figure.suptitle("5 秒前缀 TCN：全部 62 组绝对误差分布（底色 = 12 个锁定测试点）", fontsize=11)
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_5s_absolute_error_all62.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def _draw_metrics(combined: pd.DataFrame) -> None:
    ten = pd.read_csv(BASELINE / "all62_error_stats.csv")
    ten_oof = ten[ten["scope"] == "oof50"].set_index("gas")
    rows = []
    for gas in GASES:
        error = _stats(combined[combined["source"] == "OOF"], gas)
        rows.append(
            {
                "gas": gas,
                "prefix": "5 s",
                "oof_three_sigma": _three_sigma(error),
                "oof_mape": float(np.mean(np.abs(error))),
            }
        )
        rows.append(
            {
                "gas": gas,
                "prefix": "10 s",
                "oof_three_sigma": float(ten_oof.loc[gas, "three_sigma"]),
                "oof_mape": float(ten_oof.loc[gas, "mape"]),
            }
        )
    stats = pd.DataFrame(rows)
    figure, panels = plt.subplots(1, 2, figsize=(9.6, 3.8), constrained_layout=True)
    x = np.arange(len(GASES))
    width = 0.35
    for offset, prefix in ((-width / 2, "5 s"), (width / 2, "10 s")):
        panel_data = stats[stats["prefix"] == prefix].set_index("gas").loc[list(GASES)]
        panels[0].bar(x + offset, panel_data["oof_three_sigma"], width, label=prefix)
        panels[1].bar(x + offset, panel_data["oof_mape"], width, label=prefix)
    panels[0].set_ylabel("OOF 3σ (%)", fontsize=8.5)
    panels[1].set_ylabel("OOF MAPE (%)", fontsize=8.5)
    for panel in panels:
        panel.set_xticks(x, GASES)
        panel.grid(axis="y", alpha=0.2)
        panel.legend(frameon=False, fontsize=8)
        panel.tick_params(labelsize=8)
    panels[0].set_title("(a) 相对误差 3σ", fontsize=9.5, loc="left")
    panels[1].set_title("(b) 平均绝对相对误差", fontsize=9.5, loc="left")
    figure.suptitle("5 秒 vs 10 秒前缀：OOF 50 条开发实验", fontsize=11)
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_5s_vs_10s_metrics.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)
    stats.to_csv(BASE / "prefix_metrics.csv", index=False)


def _write_stats(combined: pd.DataFrame) -> None:
    rows = []
    for gas in GASES:
        error = _stats(combined, gas)
        for scope, part in (("all62", error), ("oof50", error[:50]), ("test12", error[50:])):
            rows.append(
                {
                    "gas": gas,
                    "scope": scope,
                    "n": len(part),
                    "mean_relative": float(np.mean(part)),
                    "std_relative": float(np.std(part, ddof=1)),
                    "three_sigma": _three_sigma(part),
                    "mape": float(np.mean(np.abs(part))),
                }
            )
    pd.DataFrame(rows).to_csv(BASE / "all62_error_stats.csv", index=False)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    oof = _oof_frame()
    test = _test_frame()
    combined = _combined(oof, test)
    combined.to_csv(BASE / "all62_predictions.csv", index=False)
    _write_stats(combined)
    _draw_relative(combined, test_start=len(oof))
    _draw_absolute(combined, test_start=len(oof))
    _draw_metrics(combined)


if __name__ == "__main__":
    main()
