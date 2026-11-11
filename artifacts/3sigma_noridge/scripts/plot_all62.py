"""无 Ridge 方案的 62 点误差散点图：50 个 OOF + 12 个锁定测试。

无 Ridge 流水线 = 单配置（逐气体目标 `log,none,log` + 逐气体头 + 边缘扩充 4×，
噪声 0.1× + VAE 合成 3× + 低浓度损失加权）× 3 seed → 逐 seed 预测平均 →
裁剪到开发集浓度范围（不使用 Ridge）。

输出（PNG+SVG）：
- `artifacts/3sigma_noridge/figs/fig_noridge_all62_error_sigma_by_gas.{png,svg}`
- `artifacts/3sigma_noridge/figs/fig_noridge_vs_baseline_all62.{png,svg}`
以及 `all62_predictions.csv`、`all62_error_stats.csv`。
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
COLORS = {"H2": "#1F77B4", "NH3": "#F2A900", "NO2": "#D2691E"}
MARKERS = {"H2": "o", "NH3": "^", "NO2": "s"}
SEEDS = (20260923, 20260924, 20260925)

NORIDGE = ROOT / "artifacts/3sigma_noridge"
NORIDGE_OOF = NORIDGE / "oof"
NORIDGE_TEST = NORIDGE / "evaluation/test_predictions.csv"
BASELINE_OOF = ROOT / "artifacts/final_vae_hybrid/noise_gan/oof.csv"
BASELINE_TEST = ROOT / "artifacts/final_vae_hybrid/evaluation/test_predictions.csv"
FIGS = NORIDGE / "figs"

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Source Han Sans CN", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _oof_frame() -> pd.DataFrame:
    """无 Ridge：3 seed TCN 预测平均 + 裁剪，得到 50 行 OOF。"""
    dev_rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    files = [row["file"] for row in dev_rows]
    folds = np.array([int(row["fold"]) for row in dev_rows])
    truth = np.array([row["_labels"] for row in dev_rows], dtype=float)

    frames = [
        pd.read_csv(NORIDGE_OOF / f"seed_{seed}/noise_gan/oof.csv").set_index("file")
        for seed in SEEDS
    ]
    tcn = np.stack(
        [
            np.mean(
                [frame.loc[files, f"{gas}_pred"].to_numpy(float) for frame in frames],
                axis=0,
            )
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
            **{f"{gas}_true": truth[:, index] for index, gas in enumerate(GASES)},
            **{f"{gas}_pred": prediction[:, index] for index, gas in enumerate(GASES)},
        }
    )


def _rows_from_test(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame[(frame["model"] == model) & (frame["seed"].astype(str) == "ensemble")]
    if selected.empty:
        raise ValueError(f"{path} 中找不到 model={model}, seed=ensemble 的预测行")
    selected = selected.copy()
    selected["source"] = "测试"
    return selected


def _combined(oof: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    columns = ["file", "source", *(f"{gas}_true" for gas in GASES), *(f"{gas}_pred" for gas in GASES)]
    frame = pd.concat([oof[columns], test[columns]], ignore_index=True)
    frame["run_id"] = frame["file"].str.extract(r"data/(\d+)\.csv").astype(int)
    frame["source_order"] = (frame["source"] == "测试").astype(int)
    return frame.sort_values(["source_order", "run_id"]).reset_index(drop=True)


def _relative(frame: pd.DataFrame, gas: str) -> np.ndarray:
    truth = frame[f"{gas}_true"].to_numpy(float)
    prediction = frame[f"{gas}_pred"].to_numpy(float)
    return 100.0 * (prediction - truth) / np.maximum(np.abs(truth), 1e-9)


def _three_sigma(values: np.ndarray) -> float:
    return float(3.0 * np.std(values, ddof=1))


def _draw_panel(panel, error: np.ndarray, test_start: int, marker: str, color: str) -> dict[str, float]:
    x = np.arange(len(error))
    panel.scatter(x, error, s=22, marker=marker, c=color, alpha=0.85, edgecolors="white", linewidths=0.35, zorder=3)
    mean = float(np.mean(error))
    sigma = float(np.std(error, ddof=1))
    panel.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
    panel.axhline(mean, color="0.1", linewidth=1.0, zorder=2)
    for k, style, shade in ((1.0, "--", "0.35"), (2.0, ":", "0.45"), (3.0, "-.", "0.55")):
        for sign in (-1.0, 1.0):
            panel.axhline(mean + sign * k * sigma, color=shade, linewidth=0.9, linestyle=style, zorder=2)
    panel.axhspan(mean - sigma, mean + sigma, color="0.85", alpha=0.2, zorder=0)
    panel.axvspan(test_start - 0.5, len(error) - 0.5, color="#F2D9A6", alpha=0.35, zorder=0)
    test_sigma = _three_sigma(error[test_start:])
    panel.text(
        0.01,
        0.98,
        f"n={len(error)}  均值={mean:+.2f}%  σ={sigma:.2f}%\n"
        f"全部 3σ={3.0 * sigma:.2f}% | 测试 12 点 3σ={test_sigma:.2f}%",
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


def _noridge_figure(combined: pd.DataFrame, test_start: int) -> dict[str, dict[str, float]]:
    figure, panels = plt.subplots(2, 2, figsize=(10.6, 8.4), constrained_layout=True)
    pooled = np.concatenate([_relative(combined, gas) for gas in GASES])
    stats: dict[str, dict[str, float]] = {}
    layout = [
        ("H2", _relative(combined, "H2"), MARKERS["H2"], COLORS["H2"]),
        ("NH3", _relative(combined, "NH3"), MARKERS["NH3"], COLORS["NH3"]),
        ("NO2", _relative(combined, "NO2"), MARKERS["NO2"], COLORS["NO2"]),
        ("三气体合并", pooled, "o", None),
    ]
    for panel_index, (panel, (label, error, marker, color)) in enumerate(
        zip(panels.ravel(), layout)
    ):
        if color is None:
            pooled_colors = [COLORS[gas] for gas in GASES for _ in range(len(combined))]
            panel.scatter(np.arange(len(pooled)), pooled, s=20, marker="o", c=pooled_colors, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3)
            mean = float(np.mean(pooled))
            sigma = float(np.std(pooled, ddof=1))
            panel.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
            panel.axhline(mean, color="0.1", linewidth=1.0, zorder=2)
            for k, style, shade in ((1.0, "--", "0.35"), (2.0, ":", "0.45"), (3.0, "-.", "0.55")):
                for sign in (-1.0, 1.0):
                    panel.axhline(mean + sign * k * sigma, color=shade, linewidth=0.9, linestyle=style, zorder=2)
            panel.axhspan(mean - sigma, mean + sigma, color="0.85", alpha=0.2, zorder=0)
            panel.axvspan(test_start * 3 - 0.5, len(pooled) - 0.5, color="#F2D9A6", alpha=0.35, zorder=0)
            test_sigma = _three_sigma(pooled[test_start * 3:])
            panel.text(
                0.01,
                0.98,
                f"n={len(pooled)}  均值={mean:+.2f}%  σ={sigma:.2f}%\n"
                f"全部 3σ={3.0 * sigma:.2f}% | 测试 36 值 3σ={test_sigma:.2f}%",
                transform=panel.transAxes,
                fontsize=6.6,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.85),
            )
            panel.set_xlim(-0.5, len(pooled) - 0.5)
            panel.grid(axis="y", alpha=0.2)
            panel.tick_params(labelsize=7.5)
            stats[label] = {"mean": mean, "sigma": sigma, "test_sigma": test_sigma}
        else:
            stats[label] = _draw_panel(panel, error, test_start, marker, color)
        panel.set_title(
            f"({chr(97 + panel_index)}) {label}",
            fontsize=9.5,
            loc="left",
        )
    panels[1, 0].set_xlabel("实验序号（0–49 OOF，50–61 锁定测试）", fontsize=8.5)
    panels[1, 1].set_xlabel("实验序号（0–149 OOF，150–185 锁定测试）", fontsize=8.5)
    panels[0, 0].set_ylabel("相对误差 (%)", fontsize=8.5)
    panels[1, 0].set_ylabel("相对误差 (%)", fontsize=8.5)
    figure.suptitle(
        "无 Ridge 方案（TCN × 3 seed + 裁剪）：全部 62 组相对误差分布（10 秒；50 OOF + 12 锁定测试）",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_noridge_all62_error_sigma_by_gas.{suffix}"
        figure.savefig(path, dpi=260, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    return stats


def _comparison_figure(baseline: pd.DataFrame, lean: pd.DataFrame, test_start: int) -> None:
    figure, panels = plt.subplots(2, 4, figsize=(13.2, 5.8), constrained_layout=True)
    methods = (
        ("旧方案（PCA-VAE 10s，3 seed 集成）", baseline, False),
        ("无 Ridge 方案（单配置 TCN × 3 seed + 裁剪）", lean, True),
    )
    limits = [0.0, 0.0, 0.0, 0.0]
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
            x = np.arange(len(error))
            if combined:
                pooled_colors = [COLORS[gas] for gas in GASES for _ in range(len(error) // 3)]
                panel.scatter(x, error, s=18, marker="o", c=pooled_colors, alpha=0.85, edgecolors="white", linewidths=0.3, zorder=3)
            elif filled:
                panel.scatter(x, error, s=20, marker=marker, c=color, alpha=0.85, edgecolors="white", linewidths=0.35, zorder=3)
            else:
                panel.scatter(x, error, s=20, marker=marker, facecolors="none", edgecolors=color, linewidths=0.7, alpha=0.9, zorder=3)
            mean = float(np.mean(error))
            sigma = float(np.std(error, ddof=1))
            panel.axhline(0.0, color="0.55", linewidth=0.7, zorder=1)
            panel.axhline(mean, color="0.1", linewidth=1.0, zorder=2)
            for k, style, shade in ((1.0, "--", "0.35"), (2.0, ":", "0.45"), (3.0, "-.", "0.55")):
                for sign in (-1.0, 1.0):
                    panel.axhline(mean + sign * k * sigma, color=shade, linewidth=0.85, linestyle=style, zorder=2)
            test_index = test_start * 3 if combined else test_start
            panel.axvspan(test_index - 0.5, len(error) - 0.5, color="#F2D9A6", alpha=0.35, zorder=0)
            test_sigma = _three_sigma(error[test_index:])
            panel.text(
                0.01,
                0.98,
                f"n={len(error)}  均值={mean:+.2f}%  σ={sigma:.2f}%\n"
                f"全部 3σ={3.0 * sigma:.2f}% | 测试 3σ={test_sigma:.2f}%",
                transform=panel.transAxes,
                fontsize=6.4,
                va="top",
                ha="left",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.8", alpha=0.85),
            )
            panel.set_xlim(-0.5, len(error) - 0.5)
            panel.set_title(f"{label}" if row == 0 else f"{label} · 精简方案", fontsize=9, loc="left")
            panel.grid(axis="y", alpha=0.2)
            panel.tick_params(labelsize=7)
            limit = max(3.0 * sigma, float(np.max(np.abs(error))), 1e-9) * 1.15
            limits[column] = max(limits[column], limit)
    for column, limit in enumerate(limits):
        for row in range(2):
            panels[row, column].set_ylim(-limit, limit)
    panels[1, 0].set_xlabel("实验序号（0–49 OOF，50–61 锁定测试）", fontsize=8)
    figure.suptitle(
        "旧方案 vs 无 Ridge 方案：全部 62 组相对误差分布（底色 = 12 个锁定测试点）",
        fontsize=11,
    )
    for suffix in ("png", "svg"):
        path = FIGS / f"fig_noridge_vs_baseline_all62.{suffix}"
        figure.savefig(path, dpi=250, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def _stats_rows(frame: pd.DataFrame, test_start: int) -> list[dict[str, object]]:
    rows = []
    for gas in GASES:
        error = _relative(frame, gas)
        rows.append(
            {
                "gas": gas,
                "scope": "all62",
                "n": len(error),
                "mean_relative": float(np.mean(error)),
                "std_relative": float(np.std(error, ddof=1)),
                "three_sigma": _three_sigma(error),
                "mape": float(np.mean(np.abs(error))),
            }
        )
        rows.append(
            {
                "gas": gas,
                "scope": "oof50",
                "n": test_start,
                "mean_relative": float(np.mean(error[:test_start])),
                "std_relative": float(np.std(error[:test_start], ddof=1)),
                "three_sigma": _three_sigma(error[:test_start]),
                "mape": float(np.mean(np.abs(error[:test_start]))),
            }
        )
        rows.append(
            {
                "gas": gas,
                "scope": "test12",
                "n": len(error) - test_start,
                "mean_relative": float(np.mean(error[test_start:])),
                "std_relative": float(np.std(error[test_start:], ddof=1)),
                "three_sigma": _three_sigma(error[test_start:]),
                "mape": float(np.mean(np.abs(error[test_start:]))),
            }
        )
    return rows


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    if not NORIDGE_TEST.is_file():
        raise SystemExit(f"缺少无 Ridge 测试预测：{NORIDGE_TEST}，请先运行 tcn_union.evaluation.ensemble_evaluate")

    noridge_oof = _oof_frame()
    lean = _combined(noridge_oof, _rows_from_test(NORIDGE_TEST, "candidate"))
    baseline_oof = pd.read_csv(BASELINE_OOF).assign(source="OOF")
    baseline = _combined(baseline_oof, _rows_from_test(BASELINE_TEST, "candidate"))
    test_start = int((lean["source"] == "OOF").sum())
    if test_start != 50 or len(lean) != 62:
        raise SystemExit(f"预期 50+12 行，实际 {test_start}+{len(lean) - test_start}")

    stats = _noridge_figure(lean, test_start)
    _comparison_figure(baseline, lean, test_start)

    lean[["file", "source", *[f"{gas}_true" for gas in GASES], *[f"{gas}_pred" for gas in GASES]]].to_csv(
        NORIDGE / "all62_predictions.csv", index=False
    )
    stats_rows = _stats_rows(lean, test_start)
    pd.DataFrame(stats_rows).to_csv(NORIDGE / "all62_error_stats.csv", index=False)

    print("\n=== 无 Ridge 方案 62 点统计 ===")
    print(f"{'gas':<6}{'scope':<8}{'n':>4}{'mean':>9}{'sigma':>9}{'3σ':>9}{'MAPE':>9}")
    for row in stats_rows:
        print(
            f"{row['gas']:<6}{row['scope']:<8}{row['n']:>4}{row['mean_relative']:>9.2f}"
            f"{row['std_relative']:>9.2f}{row['three_sigma']:>9.2f}{row['mape']:>9.2f}"
        )
    for gas, item in stats.items():
        print(
            f"{gas}: 均值={item['mean']:+.2f}%  σ={item['sigma']:.2f}%  "
            f"全部3σ={3 * item['sigma']:.2f}%  测试12点3σ={item['test_sigma']:.2f}%"
        )
    print(f"\nWrote {NORIDGE / 'all62_predictions.csv'}")
    print(f"Wrote {NORIDGE / 'all62_error_stats.csv'}")


if __name__ == "__main__":
    main()
