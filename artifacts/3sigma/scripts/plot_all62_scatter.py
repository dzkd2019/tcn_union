"""全部 62 组实验的相对误差散点图（参考 candidate_all62_error_sigma_by_gas.png）。

62 组 = 50 个开发集 OOF 预测 + 12 个锁定测试预测。画两种图：
1. ``fig_v4_all62_error_sigma_by_gas``：v4 最终候选的 2×2 参考样式散点图；
2. ``fig_all62_error_sigma_baseline_vs_v4``：旧方案与 v4 的 2×4 对比图，
   同一列共享纵轴，并标出 12 个锁定测试点区域。

坐标口径：相对误差 ``100*(pred-true)/max(|true|,1e-9)``；每个面板画均值、
±1σ/±2σ/±3σ 线，σ 用样本标准差（ddof=1）。
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

from tcn_union.visualization.plot_all62 import combine_predictions
from tcn_union.visualization.plot_rmse_scatter import (
    COLORS,
    MARKERS,
    plot_error_sigma,
)

GASES = ("H2", "NH3", "NO2")
OUTPUT_DIR = ROOT / "artifacts/3sigma"
FIGS = OUTPUT_DIR / "figs"
BASELINE_OOF = ROOT / "artifacts/final_vae_hybrid/noise_gan/oof.csv"
BASELINE_TEST = ROOT / "artifacts/final_vae_hybrid/evaluation/test_predictions.csv"
V4_OOF = OUTPUT_DIR / "hybrid_oof.csv"
V4_TEST = OUTPUT_DIR / "hybrid_eval_v4/test_predictions.csv"
V5_OOF = OUTPUT_DIR / "hybrid_oof_v5.csv"
V5_TEST = OUTPUT_DIR / "hybrid_eval_v5/test_predictions.csv"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "Source Han Sans CN",
    "Microsoft YaHei",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


def _build_all62(name: str, oof: Path, test: Path) -> pd.DataFrame:
    """合并 OOF 与锁定测试预测为 62 行，并写 CSV。"""
    path = OUTPUT_DIR / f"all62_{name}_predictions.csv"
    combine_predictions(oof, test, "candidate", 10, "ensemble", path)
    frame = pd.read_csv(path)
    frame["run_id"] = frame["file"].str.extract(r"data/(\d+)\.csv").astype(int)
    frame["source_order"] = (frame["source"] == "test").astype(int)
    return frame.sort_values(["source_order", "run_id"]).reset_index(drop=True)


def _relative(frame: pd.DataFrame, gas: str) -> np.ndarray:
    truth = frame[f"{gas}_true"].to_numpy(float)
    prediction = frame[f"{gas}_pred"].to_numpy(float)
    return 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)


def _draw_panel(
    panel,
    error: np.ndarray,
    test_start: int,
    marker: str,
    color: str,
    *,
    filled: bool,
    combined: bool = False,
    pooled_colors: list[str] | None = None,
) -> dict[str, float]:
    """画一个相对误差散点面板，返回统计量。"""
    x = np.arange(len(error))
    if combined:
        panel.scatter(
            x,
            error,
            s=20,
            marker="o",
            c=color if not filled else pooled_colors,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
            zorder=3,
        )
    elif filled:
        panel.scatter(
            x,
            error,
            s=22,
            marker=marker,
            c=color,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
    else:
        panel.scatter(
            x,
            error,
            s=22,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=0.7,
            alpha=0.9,
            zorder=3,
        )
    mean = float(np.mean(error))
    sigma = float(np.std(error, ddof=1))
    panel.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
    panel.axhline(mean, color="0.1", linewidth=1.0, zorder=2)
    for k, style, shade in ((1.0, "--", "0.35"), (2.0, ":", "0.45"), (3.0, "-.", "0.55")):
        for sign in (-1.0, 1.0):
            panel.axhline(
                mean + sign * k * sigma,
                color=shade,
                linewidth=0.9,
                linestyle=style,
                zorder=2,
            )
    if sigma > 0.0:
        panel.axhspan(mean - sigma, mean + sigma, color="0.85", alpha=0.2, zorder=0)
    panel.axvspan(
        test_start - 0.5,
        len(error) - 0.5,
        color="#F2D9A6",
        alpha=0.35,
        zorder=0,
    )
    test_error = error[test_start:]
    test_sigma = float(np.std(test_error, ddof=1)) if len(test_error) > 1 else 0.0
    panel.text(
        0.01,
        0.98,
        f"n={len(error)}  均值={mean:.2f}%  σ={sigma:.2f}%\n"
        f"全部 3σ={3.0 * sigma:.2f}% | 测试 12 点 3σ={3.0 * test_sigma:.2f}%",
        transform=panel.transAxes,
        fontsize=6.6,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.85),
    )
    panel.set_xlim(-0.5, len(error) - 0.5)
    panel.grid(axis="y", alpha=0.2)
    panel.tick_params(labelsize=7.5)
    return {"mean": mean, "sigma": sigma, "test_sigma": test_sigma}


def _comparison_figure(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    candidate_label: str = "v4 最终候选",
    stem: str = "fig_all62_error_sigma_baseline_vs_v4",
) -> None:
    """旧方案 vs 候选版本的 2×4 散点对比图。"""
    figure, panels = plt.subplots(2, 4, figsize=(13.4, 5.9), constrained_layout=True)
    methods = (
        ("旧方案（PCA-VAE 10s，3 seed 集成）", baseline, False),
        (candidate_label, candidate, True),
    )
    pooled_colors = [COLORS[gas] for gas in GASES for _ in range(62)]
    limits: list[float] = [0.0, 0.0, 0.0, 0.0]
    for row, (method, frame, filled) in enumerate(methods):
        errors = {gas: _relative(frame, gas) for gas in GASES}
        pooled = np.concatenate([errors[gas] for gas in GASES])
        columns = [
            ("H2", errors["H2"], MARKERS["H2"], COLORS["H2"], False),
            ("NH3", errors["NH3"], MARKERS["NH3"], COLORS["NH3"], False),
            ("NO2", errors["NO2"], MARKERS["NO2"], COLORS["NO2"], False),
            ("三气体合并", pooled, "o", "#1F6F78", True),
        ]
        for column, (label, error, marker, color, combined) in enumerate(columns):
            panel = panels[row, column]
            stats = _draw_panel(
                panel,
                error,
                50,
                marker,
                color,
                filled=filled,
                combined=combined,
                pooled_colors=pooled_colors if combined else None,
            )
            panel.set_title(
                f"{label}" if row == 0 else f"{label} · 候选",
                fontsize=9.5,
                loc="left",
            )
            if row == 1:
                panel.set_xlabel("实验序号（0–49 OOF，50–61 锁定测试）", fontsize=8)
            if column == 0:
                panel.set_ylabel(f"{method}\n相对误差 (%)", fontsize=8.5)
            limit = max(
                3.0 * stats["sigma"],
                float(np.max(np.abs(error))),
                1e-9,
            ) * 1.15
            limits[column] = max(limits[column], limit)
    for column, limit in enumerate(limits):
        for row in range(2):
            panels[row, column].set_ylim(-limit, limit)
    figure.suptitle(
        f"旧方案 vs {candidate_label}：全部 62 组相对误差分布（10 秒；底色 = 12 个锁定测试点）",
        fontsize=11.5,
    )
    for suffix in ("png", "svg"):
        path = FIGS / f"{stem}.{suffix}"
        figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    baseline = _build_all62("baseline", BASELINE_OOF, BASELINE_TEST)
    v4 = _build_all62("v4", V4_OOF, V4_TEST)
    v5 = _build_all62("v5", V5_OOF, V5_TEST)

    # 图 1：参考 candidate_all62_error_sigma_by_gas.png 的 v4 版本。
    plot_error_sigma(
        OUTPUT_DIR / "all62_v4_predictions.csv",
        FIGS,
        "all62",
        10,
        "OOF+测试",
        title="全部 62 组相对误差分布（v4 最终候选：50 个开发集 OOF + 12 个锁定测试，10 秒）",
        stem="fig_v4_all62_error_sigma_by_gas",
    )

    # 图 2：v5 参考样式 + 旧方案与 v5 的对比。
    plot_error_sigma(
        OUTPUT_DIR / "all62_v5_predictions.csv",
        FIGS,
        "all62",
        10,
        "OOF+测试",
        title="全部 62 组相对误差分布（v5 最终候选：50 个开发集 OOF + 12 个锁定测试，10 秒）",
        stem="fig_v5_all62_error_sigma_by_gas",
    )
    _comparison_figure(
        baseline,
        v5,
        candidate_label="v5 最终候选",
        stem="fig_all62_error_sigma_baseline_vs_v5",
    )

    stats_rows: list[dict[str, object]] = []
    for label, frame in (
        ("旧方案", baseline),
        ("v4 最终候选", v4),
        ("v5 最终候选", v5),
    ):
        print(f"\n=== {label} ===")
        for gas in GASES:
            error = _relative(frame, gas)
            print(
                f"{gas}: n={len(error)} 均值={np.mean(error):+.2f}% "
                f"σ={np.std(error, ddof=1):.2f}% 3σ={3 * np.std(error, ddof=1):.2f}%"
            )
            stats_rows.append(
                {
                    "method": label,
                    "scope": "all62",
                    "gas": gas,
                    "n": len(error),
                    "mean_relative": float(np.mean(error)),
                    "std_relative": float(np.std(error, ddof=1)),
                    "three_sigma": float(3.0 * np.std(error, ddof=1)),
                    "mape": float(np.mean(np.abs(error))),
                }
            )
            test_error = error[frame["source_order"].to_numpy() == 1]
            stats_rows.append(
                {
                    "method": label,
                    "scope": "test12",
                    "gas": gas,
                    "n": len(test_error),
                    "mean_relative": float(np.mean(test_error)),
                    "std_relative": float(np.std(test_error, ddof=1)),
                    "three_sigma": float(3.0 * np.std(test_error, ddof=1)),
                    "mape": float(np.mean(np.abs(test_error))),
                }
            )
        pooled = np.concatenate([_relative(frame, gas) for gas in GASES])
        print(
            f"合并: n={len(pooled)} 均值={np.mean(pooled):+.2f}% "
            f"σ={np.std(pooled, ddof=1):.2f}% 3σ={3 * np.std(pooled, ddof=1):.2f}%"
        )
        stats_rows.append(
            {
                "method": label,
                "scope": "all62",
                "gas": "macro",
                "n": len(pooled),
                "mean_relative": float(np.mean(pooled)),
                "std_relative": float(np.std(pooled, ddof=1)),
                "three_sigma": float(3.0 * np.std(pooled, ddof=1)),
                "mape": float(np.mean(np.abs(pooled))),
            }
        )
        pooled_test = np.concatenate(
            [
                _relative(frame, gas)[frame["source_order"].to_numpy() == 1]
                for gas in GASES
            ]
        )
        stats_rows.append(
            {
                "method": label,
                "scope": "test12",
                "gas": "macro",
                "n": len(pooled_test),
                "mean_relative": float(np.mean(pooled_test)),
                "std_relative": float(np.std(pooled_test, ddof=1)),
                "three_sigma": float(3.0 * np.std(pooled_test, ddof=1)),
                "mape": float(np.mean(np.abs(pooled_test))),
            }
        )
    pd.DataFrame(stats_rows).to_csv(OUTPUT_DIR / "all62_error_stats.csv", index=False)
    print(f"\nWrote {OUTPUT_DIR / 'all62_error_stats.csv'}")
    print(f"\nWrote {FIGS / 'fig_v5_all62_error_sigma_by_gas.png'}（PNG+SVG）")
    print(f"Wrote {FIGS / 'fig_all62_error_sigma_baseline_vs_v5.png'}（PNG+SVG）")


if __name__ == "__main__":
    main()
