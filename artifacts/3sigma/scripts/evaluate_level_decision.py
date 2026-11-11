"""评估“最近离散浓度档位”决策规则在 OOF 与锁定测试集上的效果。

本项目的浓度标签只取固定档位（H2 4 档、NH3 5 档、NO2 5 档）。若部署端浓度
就是这些标定档位，可在连续回归输出后取最近档位。脚本读取已经产生的预测
（OOF：artifacts/3sigma/hybrid_oof.csv；测试：artifacts/3sigma/hybrid_eval_v4/），
对比吸附前后的 3σ 与错档数，并输出容差规则的敏感性表。

用法（项目根目录）：
    .venv/bin/python artifacts/3sigma/scripts/evaluate_level_decision.py

产物：artifacts/3sigma/level_decision_summary.json/.csv、
      artifacts/3sigma/figs/fig_level_decision.png/.svg
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
import pandas as pd

plt.rcParams.update(
    {
        "font.family": ["Source Han Sans CN", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
    }
)

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.evaluation.evaluate import AXES, _manifest
from tcn_union.evaluation.level_decision import (
    level_matches,
    levels_from_labels,
    snap_to_levels,
)

GASES = AXES
OUTPUT_DIR = ROOT / "artifacts/3sigma"
FIGS = OUTPUT_DIR / "figs"
TOLERANCES = (None, 0.05, 0.10, 0.15, 0.20)


def _three_sigma(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    """逐气体 3σ 与宏平均。"""
    sigma = {}
    for index, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, index] - truth[:, index]) / np.maximum(
            np.abs(truth[:, index]), 1e-9
        )
        sigma[gas] = float(3.0 * np.std(relative, ddof=1))
    sigma["macro"] = float(np.mean([sigma[gas] for gas in GASES]))
    return sigma


def _frame_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    truth = np.column_stack([frame[f"{gas}_true"].to_numpy(float) for gas in GASES])
    prediction = np.column_stack([frame[f"{gas}_pred"].to_numpy(float) for gas in GASES])
    return truth, prediction


def _test_tcn_mean(raw: pd.DataFrame) -> pd.DataFrame:
    """把测试集各 TCN seed 行按实验平均，得到纯 TCN 预测。"""
    rows = []
    for name in sorted(raw["file"].unique()):
        subset = raw[raw["file"] == name]
        row = {"file": name}
        for gas in GASES:
            row[f"{gas}_true"] = float(subset[f"{gas}_true"].iloc[0])
            row[f"{gas}_pred"] = float(subset[f"{gas}_pred"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluate(
    truth: np.ndarray,
    prediction: np.ndarray,
    levels: tuple[np.ndarray, ...],
    tolerance: float | None,
) -> dict[str, float]:
    snapped = snap_to_levels(prediction, levels, tolerance)
    raw = _three_sigma(truth, prediction)
    mapped = _three_sigma(truth, snapped)
    matches = level_matches(truth, prediction, levels)
    return {
        "raw_macro": raw["macro"],
        "snapped_macro": mapped["macro"],
        **{f"raw_{gas}": raw[gas] for gas in GASES},
        **{f"snapped_{gas}": mapped[gas] for gas in GASES},
        "mis_snapped": int((~matches).sum()),
        "n_values": int(matches.size),
    }


def main() -> None:
    dev_rows, _ = _manifest(ROOT / "artifacts/splits/manifest.csv")
    y_dev = np.stack([row["_labels"] for row in dev_rows])
    levels = levels_from_labels(y_dev)

    oof = pd.read_csv(OUTPUT_DIR / "hybrid_oof.csv")
    truth_oof, prediction_oof = _frame_arrays(oof)

    test_rows = pd.read_csv(OUTPUT_DIR / "hybrid_eval_v4/test_predictions.csv")
    ensemble = test_rows[
        (test_rows["model"] == "candidate") & (test_rows["seed"].astype(str) == "ensemble")
    ].sort_values("file").reset_index(drop=True)
    truth_test, prediction_test = _frame_arrays(ensemble)
    tcn_only = test_rows[
        (test_rows["model"] == "candidate") & (test_rows["seed"].astype(str) != "ensemble")
    ]
    tcn_frame = _test_tcn_mean(tcn_only)
    truth_tcn, prediction_tcn = _frame_arrays(tcn_frame)

    summary: dict[str, object] = {
        "assumption": (
            "部署端浓度为实验设计的离散档位；决策规则为取最近档位，"
            "对应档位集合直接从开发集标签提取、不使用测试标签"
        ),
        "levels": {gas: levels[index].tolist() for index, gas in enumerate(GASES)},
        "tolerance_sweep": [],
        "details": {},
    }
    for label, truth, prediction in (
        ("OOF hybrid v4", truth_oof, prediction_oof),
        ("test hybrid v4", truth_test, prediction_test),
        ("test TCN-only", truth_tcn, prediction_tcn),
    ):
        summary["details"][label] = _evaluate(truth, prediction, levels, None)
    for tolerance in TOLERANCES:
        sweep = {"tolerance": tolerance}
        sweep["OOF hybrid v4"] = _evaluate(truth_oof, prediction_oof, levels, tolerance)
        sweep["test hybrid v4"] = _evaluate(truth_test, prediction_test, levels, tolerance)
        summary["tolerance_sweep"].append(sweep)

    rows = []
    for label in ("OOF hybrid v4", "test hybrid v4", "test TCN-only"):
        detail = summary["details"][label]
        for gas in (*GASES, "macro"):
            rows.append(
                {
                    "scope": label,
                    "gas": gas,
                    "raw_three_sigma": detail[f"raw_{gas}"],
                    "snapped_three_sigma": detail[f"snapped_{gas}"],
                    "mis_snapped": detail["mis_snapped"],
                    "n_values": detail["n_values"],
                }
            )
    (OUTPUT_DIR / "level_decision_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (OUTPUT_DIR / "level_decision_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # 图：吸附前后逐气体 3σ（OOF / 测试），并标出错档数。
    figure, panels = plt.subplots(1, 2, figsize=(9.8, 3.8), constrained_layout=True)
    bars = np.arange(len(GASES) + 1)
    width = 0.36
    for panel, (label, title) in zip(
        panels,
        (("OOF hybrid v4", "OOF（50 个开发实验）"), ("test hybrid v4", "锁定测试集（12 个实验）")),
        strict=True,
    ):
        detail = summary["details"][label]
        raw = [detail[f"raw_{gas}"] for gas in (*GASES, "macro")]
        snapped = [detail[f"snapped_{gas}"] for gas in (*GASES, "macro")]
        panel.bar(bars - width / 2, raw, width, label="连续回归输出", color="#9AA5AD")
        panel.bar(bars + width / 2, snapped, width, label="最近档位决策", color="#C0504A")
        for x, value in zip(bars - width / 2, raw):
            panel.text(x, value + 0.1, f"{value:.1f}", ha="center", fontsize=6, va="bottom")
        for x, value in zip(bars + width / 2, snapped):
            panel.text(x, value + 0.1, f"{value:.1f}", ha="center", fontsize=6, va="bottom")
        panel.axhline(10.0, color="#C0504D", linestyle="--", linewidth=0.8)
        panel.axhline(5.0, color="#C0504D", linestyle=":", linewidth=0.8)
        panel.set_xticks(bars, (*GASES, "macro"))
        panel.set_ylabel("相对误差 3σ / %", fontsize=8.5)
        panel.set_title(
            f"{title}；错档 {detail['mis_snapped']}/{detail['n_values']}",
            fontsize=9,
            loc="left",
        )
        panel.set_ylim(0, 13.5)
        panel.grid(axis="y", alpha=0.2, linewidth=0.5)
        panel.tick_params(labelsize=7.5)
        panel.legend(fontsize=6.8, frameon=False, loc="upper right")
    figure.suptitle("离散档位决策规则：吸附前后 3σ（OOF 与锁定测试集）", fontsize=10.5)
    for suffix in ("png", "svg"):
        figure.savefig(
            FIGS / f"fig_level_decision.{suffix}",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"Wrote {FIGS / f'fig_level_decision.{suffix}'}")
    plt.close(figure)

    for label in ("OOF hybrid v4", "test hybrid v4", "test TCN-only"):
        detail = summary["details"][label]
        print(
            f"{label:<16} raw macro={detail['raw_macro']:.2f}  "
            f"snapped macro={detail['snapped_macro']:.2f}  "
            f"错档={detail['mis_snapped']}/{detail['n_values']}"
        )
    print(f"Wrote {OUTPUT_DIR / 'level_decision_summary.json'}")


if __name__ == "__main__":
    main()
