"""10 s 三气体浓度模型：相对误差 3σ 的只读误差结构诊断。

口径（与 docs/误差方差优化.md 一致）：
    rel = 100 * (pred - true) / max(|true|, 1e-9)
    3σ  = 3 * std(rel, ddof=1)
    宏平均 = 三气体 3σ 的算术平均

只读分析：不训练模型、不跑 GPU、不修改 tcn_union/ 与 tests/。
OOF（50 个开发实验，5 折交叉预测）用于结论与后处理选型；
锁定测试集（12 个实验，3 个 seed + ensemble）只做描述性统计。

用法：
    .venv/bin/python artifacts/3sigma_diagnosis/scripts/diagnose_3sigma.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")  # ~/.config/matplotlib 在沙箱内不可写

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "3sigma_diagnosis"
FIG = OUT / "figs"

GASES = ("H2", "NH3", "NO2")
SECONDS = 10.0
STEPS_PER_SECOND = 5  # 0.2 s 采样

CURRENT_SCHEME = "weighted_4_2_2"
CURRENT_OOF = ROOT / "artifacts/variance_search/pg422_final/noise_gan/oof.csv"
CURRENT_TEST = ROOT / "artifacts/variance_search/pg422_eval/test_predictions.csv"

# 同批开发实验上的其他训练方案 / seed（用于判断离群点是否跨方案、跨 seed 复现）
COMPARISON_SCHEMES = {
    "unweighted": ROOT / "artifacts/final_vae_hybrid/noise_gan/oof.csv",
    "p1_nmae": ROOT / "artifacts/variance_search/p1.0_nmae/noise_gan/oof.csv",
    "p1_3sig": ROOT / "artifacts/variance_search/p1.0_three_sigma/noise_gan/oof.csv",
    "p2_3sig": ROOT / "artifacts/variance_search/p2.0_three_sigma/noise_gan/oof.csv",
    "p3_3sig": ROOT / "artifacts/variance_search/p3.0_three_sigma/noise_gan/oof.csv",
    "p4_3sig": ROOT / "artifacts/variance_search/p4.0_three_sigma/noise_gan/oof.csv",
    "k4_seed23": ROOT / "artifacts/pca_analysis/k4/noise_gan/oof.csv",
    "k4_seed24": ROOT / "artifacts/pca_analysis/k4_seed20260924/noise_gan/oof.csv",
    "k4_seed25": ROOT / "artifacts/pca_analysis/k4_seed20260925/noise_gan/oof.csv",
    "k8_seed23": ROOT / "artifacts/pca_analysis/k8/noise_gan/oof.csv",
    "k8_seed24": ROOT / "artifacts/pca_analysis/k8_seed20260924/noise_gan/oof.csv",
    "k8_seed25": ROOT / "artifacts/pca_analysis/k8_seed20260925/noise_gan/oof.csv",
}

# 同一配置的 3 个 seed OOF（仅 k4/k8 这两组有；当前加权方案只有单 seed OOF）
SEED_PROXY = {
    "k4": [ROOT / "artifacts/pca_analysis/k4/noise_gan/oof.csv",
           ROOT / "artifacts/pca_analysis/k4_seed20260924/noise_gan/oof.csv",
           ROOT / "artifacts/pca_analysis/k4_seed20260925/noise_gan/oof.csv"],
    "k8": [ROOT / "artifacts/pca_analysis/k8/noise_gan/oof.csv",
           ROOT / "artifacts/pca_analysis/k8_seed20260924/noise_gan/oof.csv",
           ROOT / "artifacts/pca_analysis/k8_seed20260925/noise_gan/oof.csv"],
}

SEED_NAMES = ("20260923", "20260924", "20260925")
BOOTSTRAP_REPLICATES = 4000
RNG = np.random.default_rng(20261110)


# ---------------------------------------------------------------------------
# 基础指标
# ---------------------------------------------------------------------------
def rel_percent(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐点相对误差百分数。"""
    return 100.0 * (np.asarray(pred, float) - np.asarray(true, float)) / np.maximum(
        np.abs(np.asarray(true, float)), 1e-9
    )


def sigma3(rel: np.ndarray) -> float:
    """3σ = 3 * std(rel, ddof=1)。"""
    return float(3.0 * np.std(rel, ddof=1))


def metric_row(true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rel = rel_percent(pred, true)
    return {
        "three_sigma": sigma3(rel),
        "mape": float(np.mean(np.abs(rel))),
        "mean_relative": float(np.mean(rel)),
        "rmse_relative": float(np.sqrt(np.mean(rel**2))),
        "std_relative": float(np.std(rel, ddof=1)),
    }


def summarize(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    """逐气体 + 宏平均的指标汇总。"""
    result = {gas: metric_row(frame[f"{gas}_true"].to_numpy(float), frame[f"{gas}_pred"].to_numpy(float)) for gas in GASES}
    result["macro"] = {"three_sigma": float(np.mean([result[g]["three_sigma"] for g in GASES]))}
    return result


# ---------------------------------------------------------------------------
# 原始曲线特征（10 s 前缀）
# ---------------------------------------------------------------------------
def load_dev_features(manifest: pd.DataFrame) -> pd.DataFrame:
    """从 data/*.csv 提取 10 s 前缀上的响应幅度与噪声代理。"""
    rows: list[dict[str, object]] = []
    n_samples = round(SECONDS * STEPS_PER_SECOND) + 1
    n_noise = round(1.0 * STEPS_PER_SECOND) + 1  # 0–1 s 共 6 点
    tail_start = round(5.0 * STEPS_PER_SECOND)  # 5 s 起
    for item in manifest.itertuples():
        if item.split != "development":
            continue
        frame = pd.read_csv(ROOT / item.file, encoding="utf-8-sig").iloc[:n_samples]
        response = frame[["H2", "NH3", "NO2"]].to_numpy(float)
        baseline = response[0]
        amplitude = response[-1] - baseline
        delta = amplitude / np.maximum(np.abs(baseline), 1e-8)
        delta_tail = (response[tail_start:].mean(axis=0) - baseline) / np.maximum(np.abs(baseline), 1e-8)
        noise = response[:n_noise].std(axis=0, ddof=1)
        record: dict[str, object] = {
            "file": item.file,
            "fold": int(item.fold),
            "experiment_index": int(Path(item.file).stem),
        }
        for i, gas in enumerate(GASES):
            record[f"{gas}_true"] = getattr(item, gas)
            record[f"amp_{gas}"] = float(amplitude[i])
            record[f"delta_{gas}"] = float(delta[i])
            record[f"delta_tail_{gas}"] = float(delta_tail[i])
            record[f"noise_{gas}"] = float(noise[i])
            record[f"baseline_{gas}"] = float(baseline[i])
            # 相对噪声代理：0–1 s 噪声 σ / |10 s 响应变化量|（= 1/SNR）
            record[f"relnoise_{gas}"] = float(noise[i] / max(abs(amplitude[i]), 1e-12))
        rows.append(record)
    features = pd.DataFrame(rows)
    for gas in GASES:
        features[f"sensitivity_{gas}"] = features[f"amp_{gas}"].abs() / features[f"{gas}_true"]
    return features


# ---------------------------------------------------------------------------
# 1. 尾部：top5、方差贡献、裁剪 top-k、跨方案稳定性
# ---------------------------------------------------------------------------
def analyse_tail(oof: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    detail = oof[["file", "fold"]].copy()
    detail = detail.merge(
        features.drop(columns=[f"{g}_true" for g in GASES] + ["fold"]), on="file", how="left"
    )
    tail_sensitivity: list[dict[str, object]] = []
    variance_share: list[dict[str, object]] = []
    for gas in GASES:
        true = oof[f"{gas}_true"].to_numpy(float)
        pred = oof[f"{gas}_pred"].to_numpy(float)
        rel = rel_percent(pred, true)
        detail[f"{gas}_true"] = true
        detail[f"{gas}_pred"] = pred
        detail[f"{gas}_rel"] = rel
        detail[f"{gas}_absrel"] = np.abs(rel)
        detail[f"{gas}_z"] = (rel - np.mean(rel)) / np.std(rel, ddof=1)
        share = (rel - np.mean(rel)) ** 2 / np.sum((rel - np.mean(rel)) ** 2)
        detail[f"{gas}_variance_share"] = share
        order = np.argsort(-np.abs(rel))
        detail[f"{gas}_absrel_rank"] = 0
        detail.loc[order, f"{gas}_absrel_rank"] = np.arange(1, len(order) + 1)
        for rank, index in enumerate(order[:5], start=1):
            row = detail.loc[index]
            variance_share.append(
                {
                    "gas": gas,
                    "rank": rank,
                    "file": row["file"],
                    "fold": int(row["fold"]),
                    "true": float(true[index]),
                    "pred": float(pred[index]),
                    "relative_error": float(rel[index]),
                    "z": float(row[f"{gas}_z"]),
                    "variance_share": float(share[index]),
                }
            )
        base = sigma3(rel)
        tail_sensitivity.append({"gas": gas, "trim_k": 0, "three_sigma": base, "relative_to_base": 1.0})
        for k in (1, 2, 3):
            keep = np.ones(len(rel), bool)
            keep[order[:k]] = False
            value = sigma3(rel[keep])
            tail_sensitivity.append(
                {"gas": gas, "trim_k": k, "three_sigma": value, "relative_to_base": value / base}
            )
    # 宏平均的 top-k 裁剪（每气体各自裁 top-k）
    tail_frame = pd.DataFrame(tail_sensitivity)
    for k in (0, 1, 2, 3):
        subset = tail_frame[tail_frame["trim_k"] == k]
        tail_sensitivity.append(
            {
                "gas": "macro",
                "trim_k": k,
                "three_sigma": float(subset["three_sigma"].mean()),
                "relative_to_base": float(subset["relative_to_base"].mean()),
            }
        )
    tail_frame = pd.DataFrame(tail_sensitivity)

    # 按“跨气体平均标准化 |rel|”裁剪实验（同一实验三气体一起去掉）
    score = np.mean([np.abs(detail[f"{gas}_rel"]) / np.std(detail[f"{gas}_rel"].to_numpy(), ddof=1) for gas in GASES], axis=0)
    order_exp = np.argsort(-score)
    union_rows: list[dict[str, object]] = []
    for k in (0, 1, 2, 3):
        keep = np.ones(len(score), bool)
        keep[order_exp[:k]] = False
        row = {"trim_k": k, "removed_files": ",".join(detail.loc[order_exp[:k], "file"].tolist())}
        sigmas = []
        for gas in GASES:
            rel = detail[f"{gas}_rel"].to_numpy()[keep]
            row[f"{gas}_three_sigma"] = sigma3(rel)
            sigmas.append(sigma3(rel))
        row["macro_three_sigma"] = float(np.mean(sigmas))
        union_rows.append(row)
    union_frame = pd.DataFrame(union_rows)

    detail.to_csv(OUT / "oof_error_detail.csv", index=False)
    pd.DataFrame(variance_share).to_csv(OUT / "top_outliers.csv", index=False)
    tail_frame.to_csv(OUT / "tail_sensitivity.csv", index=False)
    union_frame.to_csv(OUT / "tail_sensitivity_union.csv", index=False)
    return {
        "detail": detail,
        "top5": pd.DataFrame(variance_share),
        "tail_sensitivity": tail_frame,
        "union": union_frame,
        "variance_share": pd.DataFrame(variance_share),
    }


def analyse_stability(oof: pd.DataFrame) -> dict[str, object]:
    """同一批开发实验上，|rel| 排名在 12 个方案/seed 之间是否复现。"""
    schemes = {CURRENT_SCHEME: oof}
    for name, path in COMPARISON_SCHEMES.items():
        schemes[name] = pd.read_csv(path)
    absrel = {gas: {} for gas in GASES}
    for name, frame in schemes.items():
        frame = frame.set_index("file")
        for gas in GASES:
            absrel[gas][name] = pd.Series(
                np.abs(rel_percent(frame[f"{gas}_pred"].to_numpy(float), frame[f"{gas}_true"].to_numpy(float))),
                index=frame.index,
            )
    ranking_rows: list[dict[str, object]] = []
    for gas in GASES:
        matrix = pd.DataFrame(absrel[gas])
        ranks = matrix.rank(ascending=False, method="min")
        top5 = matrix[CURRENT_SCHEME].nlargest(5).index
        for file in top5:
            row: dict[str, object] = {
                "gas": gas,
                "file": file,
                "current_absrel": float(matrix.loc[file, CURRENT_SCHEME]),
                "current_rank": int(ranks.loc[file, CURRENT_SCHEME]),
                "median_rank_all": float(ranks.loc[file].median()),
                "best_rank_all": int(ranks.loc[file].min()),
                "worst_rank_all": int(ranks.loc[file].max()),
                "n_top5_all": int((ranks.loc[file] <= 5).sum()),
                "n_top10_all": int((ranks.loc[file] <= 10).sum()),
                "mean_absrel_all": float(matrix.loc[file].mean()),
                "std_absrel_all": float(matrix.loc[file].std(ddof=1)),
            }
            for name in schemes:
                row[f"rank_{name}"] = int(ranks.loc[file, name])
            ranking_rows.append(row)
    stability = pd.DataFrame(ranking_rows)
    stability.to_csv(OUT / "outlier_stability.csv", index=False)

    # 跨 run 的 |rel| 排名相关性（衡量“实验级误差有多少是随机的”）
    pairwise_rows: list[dict[str, object]] = []
    groups = dict(COMPARISON_SCHEMES)
    groups.pop("k4_seed23", None)
    groups.pop("k4_seed24", None)
    groups.pop("k4_seed25", None)
    groups.pop("k8_seed23", None)
    groups.pop("k8_seed24", None)
    groups.pop("k8_seed25", None)
    groups[CURRENT_SCHEME] = CURRENT_OOF
    for gas in GASES:
        absrel_frame = pd.DataFrame(
            {name: np.abs(rel_percent(pd.read_csv(path).set_index("file")[f"{gas}_pred"], pd.read_csv(path).set_index("file")[f"{gas}_true"]))
             for name, path in groups.items()}
        )
        correlations = [
            stats.spearmanr(absrel_frame[a], absrel_frame[b]).statistic
            for a, b in _pairs(absrel_frame.columns)
        ]
        pairwise_rows.append(
            {
                "gas": gas,
                "group": "weighting_variants",
                "n_schemes": len(absrel_frame.columns),
                "mean_pairwise_spearman": float(np.mean(correlations)),
                "min_pairwise_spearman": float(np.min(correlations)),
                "max_pairwise_spearman": float(np.max(correlations)),
            }
        )
    for label, paths in SEED_PROXY.items():
        for gas in GASES:
            absrel_frame = pd.DataFrame(
                {f"seed_{i}": np.abs(rel_percent(pd.read_csv(path)[f"{gas}_pred"], pd.read_csv(path)[f"{gas}_true"]))
                 for i, path in enumerate(paths)}
            )
            correlations = [
                stats.spearmanr(absrel_frame[a], absrel_frame[b]).statistic
                for a, b in _pairs(absrel_frame.columns)
            ]
            pairwise_rows.append(
                {
                    "gas": gas,
                    "group": f"{label}_seeds",
                    "n_schemes": len(absrel_frame.columns),
                    "mean_pairwise_spearman": float(np.mean(correlations)),
                    "min_pairwise_spearman": float(np.min(correlations)),
                    "max_pairwise_spearman": float(np.max(correlations)),
                }
            )
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise.to_csv(OUT / "rank_consistency.csv", index=False)
    return {"stability": stability, "pairwise": pairwise}


def _pairs(items) -> list[tuple[object, object]]:
    items = list(items)
    return [(items[i], items[j]) for i in range(len(items)) for j in range(i + 1, len(items))]


# ---------------------------------------------------------------------------
# 2. 因子分析
# ---------------------------------------------------------------------------
FACTORS = (
    "true_conc",
    "log_true_conc",
    "delta10",
    "abs_delta10",
    "delta_tail_5_10s",
    "noise_0_1s",
    "relnoise_0_1s_over_amp",
    "baseline",
    "experiment_index",
    "fold",
)


def _factor_value(detail: pd.DataFrame, gas: str, factor: str) -> tuple[np.ndarray, str]:
    if factor == "true_conc":
        return detail[f"{gas}_true"].to_numpy(float), f"{gas} 真实浓度"
    if factor == "log_true_conc":
        return np.log(detail[f"{gas}_true"].to_numpy(float)), f"log({gas} 真实浓度)"
    if factor == "delta10":
        return detail[f"delta_{gas}"].to_numpy(float), "10s 响应相对增量"
    if factor == "abs_delta10":
        return detail[f"delta_{gas}"].abs().to_numpy(float), "|10s 响应相对增量|"
    if factor == "delta_tail_5_10s":
        return detail[f"delta_tail_{gas}"].to_numpy(float), "5–10s 均值相对增量"
    if factor == "noise_0_1s":
        return detail[f"noise_{gas}"].to_numpy(float), "0–1s 噪声 σ"
    if factor == "relnoise_0_1s_over_amp":
        return detail[f"relnoise_{gas}"].to_numpy(float), "噪声 σ / |响应幅度|"
    if factor == "baseline":
        return detail[f"baseline_{gas}"].to_numpy(float), "t=0 基线响应"
    if factor == "experiment_index":
        return detail["experiment_index"].to_numpy(float), "实验编号（采集顺序）"
    if factor == "fold":
        return detail["fold"].to_numpy(float), "fold 编号"
    raise KeyError(factor)


def _partial_spearman(x: np.ndarray, y: np.ndarray, control: np.ndarray) -> float:
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rc = stats.rankdata(control)
    design = np.vstack([rc, np.ones_like(rc)]).T
    res_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    res_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    return float(np.corrcoef(res_x, res_y)[0, 1])


def analyse_factors(detail: pd.DataFrame) -> dict[str, object]:
    correlation_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    tail_rows: list[dict[str, object]] = []
    abs_error_rows: list[dict[str, object]] = []
    for gas in GASES:
        absrel = detail[f"{gas}_absrel"].to_numpy(float)
        log_true = np.log(detail[f"{gas}_true"].to_numpy(float))
        for factor in FACTORS:
            value, _ = _factor_value(detail, gas, factor)
            pearson = stats.pearsonr(absrel, value)
            spearman = stats.spearmanr(absrel, value)
            partial = _partial_spearman(absrel, value, log_true) if factor not in ("true_conc", "log_true_conc") else float("nan")
            correlation_rows.append(
                {
                    "gas": gas,
                    "factor": factor,
                    "n": len(absrel),
                    "pearson": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                    "partial_spearman_control_conc": partial,
                }
            )
        # 分组均值：浓度档位 / fold / SNR 四分位
        for level, group in detail.groupby(f"{gas}_true"):
            group_rows.append(
                {
                    "gas": gas,
                    "group_type": "concentration_level",
                    "group": f"{gas}={level:g}",
                    "n": len(group),
                    "mean_absrel": float(group[f"{gas}_absrel"].mean()),
                    "median_absrel": float(group[f"{gas}_absrel"].median()),
                    "mean_rel": float(group[f"{gas}_rel"].mean()),
                }
            )
        for fold, group in detail.groupby("fold"):
            group_rows.append(
                {
                    "gas": gas,
                    "group_type": "fold",
                    "group": f"fold={int(fold)}",
                    "n": len(group),
                    "mean_absrel": float(group[f"{gas}_absrel"].mean()),
                    "median_absrel": float(group[f"{gas}_absrel"].median()),
                    "mean_rel": float(group[f"{gas}_rel"].mean()),
                }
            )
        snr = detail[f"delta_{gas}"].abs().to_numpy(float) / np.maximum(detail[f"noise_{gas}"].to_numpy(float), 1e-12)
        quartiles = pd.qcut(snr, 4, labels=False, duplicates="drop")
        for label, group in detail.groupby(quartiles):
            group_rows.append(
                {
                    "gas": gas,
                    "group_type": "snr_quartile",
                    "group": f"Q{int(label) + 1}",
                    "n": len(group),
                    "mean_absrel": float(group[f"{gas}_absrel"].mean()),
                    "median_absrel": float(group[f"{gas}_absrel"].median()),
                    "mean_rel": float(group[f"{gas}_rel"].mean()),
                }
            )
        # 绝对误差结构：|pred-true| 与真实浓度的关系
        abs_error = np.abs(detail[f"{gas}_pred"].to_numpy(float) - detail[f"{gas}_true"].to_numpy(float))
        abs_error_rows.append(
            {
                "gas": gas,
                "corr_abs_error_true_pearson": float(stats.pearsonr(abs_error, detail[f"{gas}_true"]).statistic),
                "corr_abs_error_true_spearman": float(stats.spearmanr(abs_error, detail[f"{gas}_true"]).statistic),
                "mean_abs_error": float(abs_error.mean()),
                "std_abs_error": float(abs_error.std(ddof=1)),
            }
        )
        # 尾部 vs 其余（top5 |rel|）
        tail_index = detail[f"{gas}_absrel"].nlargest(5).index
        for factor in FACTORS:
            value, _ = _factor_value(detail, gas, factor)
            value = np.asarray(value, float)
            tail_values = value[tail_index]
            rest_values = np.delete(value, tail_index)
            mwu = stats.mannwhitneyu(tail_values, rest_values, alternative="two-sided")
            auc = (
                (tail_values[:, None] > rest_values[None, :]).sum()
                + 0.5 * (tail_values[:, None] == rest_values[None, :]).sum()
            ) / (len(tail_values) * len(rest_values))
            tail_rows.append(
                {
                    "gas": gas,
                    "factor": factor,
                    "tail_median": float(np.median(tail_values)),
                    "rest_median": float(np.median(rest_values)),
                    "auc": float(auc),
                    "mwu_p": float(mwu.pvalue),
                }
            )
    for gas in GASES:
        for level, group in detail.groupby(f"{gas}_true"):
            abs_error = (detail.loc[group.index, f"{gas}_pred"] - detail.loc[group.index, f"{gas}_true"]).abs()
            abs_error_rows.append(
                {
                    "gas": gas,
                    "concentration_level": float(level),
                    "n": int(len(group)),
                    "mean_abs_error": float(abs_error.mean()),
                    "std_abs_error": float(abs_error.std(ddof=1)),
                    "mean_absrel": float(group[f"{gas}_absrel"].mean()),
                    "mean_rel": float(group[f"{gas}_rel"].mean()),
                }
            )
    correlation_frame = pd.DataFrame(correlation_rows)
    group_frame = pd.DataFrame(group_rows)
    tail_frame = pd.DataFrame(tail_rows)
    abs_frame = pd.DataFrame(abs_error_rows)
    correlation_frame.to_csv(OUT / "factor_correlations.csv", index=False)
    group_frame.to_csv(OUT / "factor_group_means.csv", index=False)
    tail_frame.to_csv(OUT / "tail_vs_rest.csv", index=False)
    abs_frame.to_csv(OUT / "abs_error_by_level.csv", index=False)
    return {
        "correlations": correlation_frame,
        "groups": group_frame,
        "tail_vs_rest": tail_frame,
        "abs_error": abs_frame,
    }


# ---------------------------------------------------------------------------
# 3. 简单后处理（按 fold 交叉拟合，OOF 上评估）
# ---------------------------------------------------------------------------
def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cumulative, 0.5)])


def fit_scale_ls(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sum(pred * true) / np.sum(pred * pred))


def fit_scale_mape(true: np.ndarray, pred: np.ndarray) -> float:
    # 最小化 mean|a*pred/true - 1| = mean|a - true/pred| * pred/true
    ratio = pred / true
    return _weighted_median(1.0 / ratio, ratio)


def fit_bias_ls(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(true - pred))


def fit_bias_mape(true: np.ndarray, pred: np.ndarray) -> float:
    # 最小化 mean|pred + b - true| / true
    return _weighted_median(true - pred, 1.0 / true)


def fit_bias_sigma(true: np.ndarray, pred: np.ndarray) -> float:
    # 最小化 std(rel + 100*b/true)
    x = (pred - true) / true
    y = 1.0 / true
    k = -np.cov(x, y, ddof=1)[0, 1] / np.var(y, ddof=1)
    return float(k / 100.0)


def fit_affine_ls(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    design = np.vstack([pred, np.ones_like(pred)]).T
    a, b = np.linalg.lstsq(design, true, rcond=None)[0]
    return float(a), float(b)


def fit_loglinear(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    a, b = np.polyfit(np.log(pred), np.log(true), 1)
    return float(a), float(b)


def fit_clip_range(true: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    return float(true.min()), float(true.max())


TRANSFORMS = {
    "scale_ls": (fit_scale_ls, lambda p, par: par * p, "乘性缩放（最小二乘）"),
    "scale_mape": (fit_scale_mape, lambda p, par: par * p, "乘性缩放（最小化 MAPE）"),
    "bias_ls": (fit_bias_ls, lambda p, par: p + par, "加性偏置（最小二乘）"),
    "bias_mape": (fit_bias_mape, lambda p, par: p + par, "加性偏置（最小化 MAPE）"),
    "bias_sigma": (fit_bias_sigma, lambda p, par: p + par, "加性偏置（最小化 3σ）"),
    "affine_ls": (fit_affine_ls, lambda p, par: par[0] * p + par[1], "仿射 a*pred+b（最小二乘）"),
    "loglinear": (fit_loglinear, lambda p, par: np.exp(par[1]) * p ** par[0], "log 空间线性 log(pred)=a*log(true)+b"),
    "clip_range": (fit_clip_range, lambda p, par: np.clip(p, par[0], par[1]), "裁剪到训练浓度范围"),
}


def _params_to_dict(name: str, params: object) -> dict[str, float]:
    if name in ("affine_ls", "loglinear"):
        return {"a": float(params[0]), "b": float(params[1])}
    if name == "clip_range":
        return {"lo": float(params[0]), "hi": float(params[1])}
    return {"a": float(params)}


def crossfit_predict(name: str, true: np.ndarray, pred: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """按 fold 交叉拟合：目标折上的参数只用其余折的 OOF 点拟合。"""
    fit, apply, _ = TRANSFORMS[name]
    output = np.empty_like(pred, dtype=float)
    for current in np.unique(fold):
        train = fold != current
        evaluate = fold == current
        params = fit(true[train], pred[train])
        output[evaluate] = apply(pred[evaluate], params)
    return output


def insample_predict(name: str, true: np.ndarray, pred: np.ndarray) -> np.ndarray:
    fit, apply, _ = TRANSFORMS[name]
    return apply(pred, fit(true, pred))


def evaluate_postprocessing(oof: pd.DataFrame) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    for gas in GASES:
        true = oof[f"{gas}_true"].to_numpy(float)
        pred = oof[f"{gas}_pred"].to_numpy(float)
        fold = oof["fold"].to_numpy(int)
        before = metric_row(true, pred)
        for name in ("baseline", *TRANSFORMS):
            if name == "baseline":
                after = before
                changed = 0
            else:
                transformed = crossfit_predict(name, true, pred, fold)
                after = metric_row(true, transformed)
                changed = int(np.sum(np.abs(transformed - pred) > 1e-9))
                for current in np.unique(fold):
                    train = fold != current
                    params = TRANSFORMS[name][0](true[train], pred[train])
                    fold_rows.append({"gas": gas, "transform": name, "fold": int(current), **_params_to_dict(name, params)})
            rows.append(
                {
                    "gas": gas,
                    "transform": name,
                    "fit": "crossfit",
                    "three_sigma": after["three_sigma"],
                    "three_sigma_before": before["three_sigma"],
                    "delta_percent": 100.0 * (after["three_sigma"] / before["three_sigma"] - 1.0),
                    "mape": after["mape"],
                    "mape_before": before["mape"],
                    "mape_delta_percent": 100.0 * (after["mape"] / before["mape"] - 1.0),
                    "mean_relative": after["mean_relative"],
                    "rmse_relative": after["rmse_relative"],
                    "changed_points": changed,
                }
            )
            # 同样的变换若在全部 50 点上拟合（自欺上界），量化交叉拟合的必要性
            if name != "baseline":
                transformed_in = insample_predict(name, true, pred)
                after_in = metric_row(true, transformed_in)
                rows.append(
                    {
                        "gas": gas,
                        "transform": name,
                        "fit": "insample",
                        "three_sigma": after_in["three_sigma"],
                        "three_sigma_before": before["three_sigma"],
                        "delta_percent": 100.0 * (after_in["three_sigma"] / before["three_sigma"] - 1.0),
                        "mape": after_in["mape"],
                        "mape_before": before["mape"],
                        "mape_delta_percent": 100.0 * (after_in["mape"] / before["mape"] - 1.0),
                        "mean_relative": after_in["mean_relative"],
                        "rmse_relative": after_in["rmse_relative"],
                        "changed_points": changed,
                    }
                )
    frame = pd.DataFrame(rows)
    # 宏平均行
    macro_rows: list[dict[str, object]] = []
    for name in ("baseline", *TRANSFORMS):
        for fit_type in ("crossfit", "insample"):
            if name == "baseline" and fit_type == "insample":
                continue
            subset = frame[(frame["transform"] == name) & (frame["fit"] == fit_type)]
            if subset.empty:
                continue
            macro_rows.append(
                {
                    "gas": "macro",
                    "transform": name,
                    "fit": fit_type,
                    "three_sigma": float(subset["three_sigma"].mean()),
                    "three_sigma_before": float(subset["three_sigma_before"].mean()),
                    "delta_percent": float(subset["delta_percent"].mean()),
                    "mape": float(subset["mape"].mean()),
                    "mape_before": float(subset["mape_before"].mean()),
                    "mape_delta_percent": float(subset["mape_delta_percent"].mean()),
                    "mean_relative": float(subset["mean_relative"].mean()),
                    "rmse_relative": float(subset["rmse_relative"].mean()),
                    "changed_points": int(subset["changed_points"].sum()),
                }
            )
    frame = pd.concat([frame, pd.DataFrame(macro_rows)], ignore_index=True)
    frame.to_csv(OUT / "postprocess_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "postprocess_fold_params.csv", index=False)

    # 3σ 的尺度退化：a*pred 时 3σ(a) = a*3σ(1)，偏差随 a 线性移动
    degeneracy: list[dict[str, object]] = []
    for a in (0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2):
        row: dict[str, object] = {"a": a}
        sigmas = []
        for gas in GASES:
            true = oof[f"{gas}_true"].to_numpy(float)
            pred = oof[f"{gas}_pred"].to_numpy(float)
            metrics = metric_row(true, a * pred)
            row[f"{gas}_three_sigma"] = metrics["three_sigma"]
            row[f"{gas}_mape"] = metrics["mape"]
            row[f"{gas}_mean_relative"] = metrics["mean_relative"]
            sigmas.append(metrics["three_sigma"])
        row["macro_three_sigma"] = float(np.mean(sigmas))
        degeneracy.append(row)
    pd.DataFrame(degeneracy).to_csv(OUT / "scale_degeneracy.csv", index=False)

    return {"metrics": frame, "degeneracy": pd.DataFrame(degeneracy)}


def _protocol_predict(
    name: str,
    true: np.ndarray,
    pred: np.ndarray,
    fold: np.ndarray,
    fixed_bounds: tuple[float, float] | None = None,
) -> np.ndarray:
    """按后处理协议变换预测；clip 使用固定的开发集浓度范围（部署端常量）。"""
    if name == "clip_range" and fixed_bounds is not None:
        return np.clip(pred, fixed_bounds[0], fixed_bounds[1])
    return crossfit_predict(name, true, pred, fold)


def bootstrap_postprocessing(
    oof: pd.DataFrame,
    transforms: tuple[str, ...] = ("clip_range", "loglinear", "affine_ls", "scale_ls", "bias_ls", "scale_mape"),
) -> pd.DataFrame:
    """实验级 paired bootstrap：同一组实验重采样同时用于三气体，给出 Δ3σ 的 95% CI。

    clip 的浓度范围是部署常量（50 个开发实验标签的 min/max），bootstrap 中不重新
    估计；其余变换在每个重采样内重新按 fold 交叉拟合，以同时反映评估点抽样与
    参数估计的不确定性。``delta`` 是在真实 OOF 上的点估计，``bootstrap_mean_delta``
    是重采样分布均值（两者接近说明估计稳定）。
    """
    n = len(oof)
    gases = list(GASES)
    fixed_bounds = {gas: (float(oof[f"{gas}_true"].min()), float(oof[f"{gas}_true"].max())) for gas in gases}
    data = {gas: (oof[f"{gas}_true"].to_numpy(float), oof[f"{gas}_pred"].to_numpy(float), oof["fold"].to_numpy(int)) for gas in gases}

    targets = ("baseline", *transforms)
    baseline_samples = {name: np.empty((BOOTSTRAP_REPLICATES, len(gases))) for name in targets}
    delta_samples = {name: np.empty((BOOTSTRAP_REPLICATES, len(gases))) for name in targets}
    for replicate in range(BOOTSTRAP_REPLICATES):
        idx = RNG.integers(0, n, n)
        for gas_index, gas in enumerate(gases):
            true, pred, fold = data[gas]
            true_b = true[idx]
            pred_b = pred[idx]
            fold_b = fold[idx]
            base_sample = sigma3(rel_percent(pred_b, true_b))
            baseline_samples["baseline"][replicate, gas_index] = base_sample
            delta_samples["baseline"][replicate, gas_index] = 0.0
            for name in transforms:
                transformed = _protocol_predict(name, true_b, pred_b, fold_b, fixed_bounds[gas])
                delta_samples[name][replicate, gas_index] = sigma3(rel_percent(transformed, true_b)) - base_sample

    # 真实 OOF 上的点估计
    point_delta: dict[str, list[float]] = {}
    point_sigma: dict[str, list[float]] = {}
    for name in targets:
        point_delta[name] = []
        point_sigma[name] = []
        for gas in gases:
            true, pred, fold = data[gas]
            base = sigma3(rel_percent(pred, true))
            point_sigma[name].append(base)
            if name == "baseline":
                point_delta[name].append(0.0)
            else:
                transformed = _protocol_predict(name, true, pred, fold, fixed_bounds[gas])
                point_delta[name].append(sigma3(rel_percent(transformed, true)) - base)

    rows: list[dict[str, object]] = []
    for name in targets:
        for gas_index, gas in enumerate((*GASES, "macro")):
            if gas == "macro":
                observed_sigma = float(np.mean(point_sigma[name]))
                baseline_draw = baseline_samples[name].mean(axis=1)
                delta_draw = delta_samples[name].mean(axis=1)
                point = float(np.mean(point_delta[name]))
            else:
                true, pred, _ = data[gas]
                observed_sigma = sigma3(rel_percent(pred, true))
                baseline_draw = baseline_samples[name][:, gas_index]
                delta_draw = delta_samples[name][:, gas_index]
                point = float(point_delta[name][gas_index])
            if name == "baseline":
                rows.append(
                    {
                        "transform": name,
                        "gas": gas,
                        "observed_three_sigma": observed_sigma,
                        "delta": 0.0,
                        "bootstrap_mean_delta": 0.0,
                        "delta_ci_low": float(np.percentile(baseline_draw, 2.5)),
                        "delta_ci_high": float(np.percentile(baseline_draw, 97.5)),
                        "p_delta_negative": float("nan"),
                        "note": "CI 为 3σ 自身（不是 Δ）",
                    }
                )
            else:
                rows.append(
                    {
                        "transform": name,
                        "gas": gas,
                        "observed_three_sigma": observed_sigma,
                        "delta": point,
                        "bootstrap_mean_delta": float(delta_draw.mean()),
                        "delta_ci_low": float(np.percentile(delta_draw, 2.5)),
                        "delta_ci_high": float(np.percentile(delta_draw, 97.5)),
                        "p_delta_negative": float(np.mean(delta_draw < 0)),
                        "note": "Δ3σ = 后处理 - 基线（负值表示改善）",
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "bootstrap_ci.csv", index=False)
    return frame


def evaluate_fold_consistency(oof: pd.DataFrame) -> pd.DataFrame:
    """逐 fold 的配对检查：后处理在 5 折中各有几折让该气体 3σ 下降。

    每折只有 10 个点，单折 3σ 噪声很大，这里只用于判断“改善方向是否一致”，
    不单独作为显著性证据。
    """
    rows: list[dict[str, object]] = []
    for gas in GASES:
        true = oof[f"{gas}_true"].to_numpy(float)
        pred = oof[f"{gas}_pred"].to_numpy(float)
        fold = oof["fold"].to_numpy(int)
        bounds = (float(true.min()), float(true.max()))
        before_all = sigma3(rel_percent(pred, true))
        for name in ("baseline", *TRANSFORMS):
            transformed = pred if name == "baseline" else _protocol_predict(name, true, pred, fold, bounds)
            for current in range(5):
                mask = fold == current
                before = sigma3(rel_percent(pred[mask], true[mask]))
                after = sigma3(rel_percent(transformed[mask], true[mask]))
                rows.append(
                    {
                        "gas": gas,
                        "transform": name,
                        "fold": current,
                        "three_sigma_before": before,
                        "three_sigma_after": after,
                        "delta": after - before,
                        "improved": bool(after < before),
                    }
                )
            subset = [r for r in rows if r["gas"] == gas and r["transform"] == name and r["fold"] != "macro"]
            improved = sum(1 for r in subset if r["improved"])
            rows.append(
                {
                    "gas": gas,
                    "transform": name,
                    "fold": "macro",
                    "three_sigma_before": before_all,
                    "three_sigma_after": sigma3(rel_percent(transformed, true)),
                    "delta": sigma3(rel_percent(transformed, true)) - before_all,
                    "improved": improved >= 3,
                    "n_folds_improved": improved,
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "postprocess_fold_consistency.csv", index=False)
    counts = (
        frame[frame["fold"] == "macro"][["gas", "transform", "n_folds_improved", "delta"]]
        .reset_index(drop=True)
    )
    counts.to_csv(OUT / "postprocess_fold_consistency_summary.csv", index=False)
    return counts


# ---------------------------------------------------------------------------
# 4. seed 方差与跨 seed 聚合（OOF 用 k4/k8 代理；测试集只做描述）
# ---------------------------------------------------------------------------
def analyse_seed_variance() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for label, paths in SEED_PROXY.items():
        frames = [pd.read_csv(path) for path in paths]
        for i, frame in enumerate(frames):
            for gas in GASES:
                metrics = metric_row(frame[f"{gas}_true"].to_numpy(float), frame[f"{gas}_pred"].to_numpy(float))
                rows.append({"config": label, "aggregation": f"seed_{i}", "gas": gas, **metrics})
        for aggregation in ("mean", "median"):
            merged = frames[0].copy()
            for gas in GASES:
                stack = np.stack([frame[f"{gas}_pred"].to_numpy(float) for frame in frames])
                merged[f"{gas}_pred"] = stack.mean(axis=0) if aggregation == "mean" else np.median(stack, axis=0)
            for gas in GASES:
                metrics = metric_row(merged[f"{gas}_true"].to_numpy(float), merged[f"{gas}_pred"].to_numpy(float))
                rows.append({"config": label, "aggregation": f"{aggregation}_ensemble", "gas": gas, **metrics})
    frame = pd.DataFrame(rows)
    # 宏平均
    macro_rows: list[dict[str, object]] = []
    for (config, aggregation), subset in frame.groupby(["config", "aggregation"]):
        macro_rows.append(
            {
                "config": config,
                "aggregation": aggregation,
                "gas": "macro",
                "three_sigma": float(subset["three_sigma"].mean()),
                "mape": float(subset["mape"].mean()),
                "mean_relative": float(subset["mean_relative"].mean()),
                "rmse_relative": float(subset["rmse_relative"].mean()),
                "std_relative": float(subset["std_relative"].mean()),
            }
        )
    frame = pd.concat([frame, pd.DataFrame(macro_rows)], ignore_index=True)
    frame.to_csv(OUT / "seed_variance_proxy.csv", index=False)
    return {"proxy": frame}


def analyse_test_descriptive(manifest: pd.DataFrame, features: pd.DataFrame) -> dict[str, object]:
    """锁定测试集 12 点：只做描述性统计，不用于任何方法选择。"""
    test = pd.read_csv(CURRENT_TEST)
    candidate = test[test["model"] == "candidate"]
    frame_rows: list[dict[str, object]] = []
    per_seed: dict[str, pd.DataFrame] = {}
    for seed in SEED_NAMES:
        frame = candidate[candidate["seed"].astype(str) == seed].set_index("file")
        per_seed[seed] = frame
        for gas in GASES:
            metrics = metric_row(frame[f"{gas}_true"].to_numpy(float), frame[f"{gas}_pred"].to_numpy(float))
            frame_rows.append({"scope": f"seed_{seed}", "gas": gas, **metrics})
    ensemble = candidate[candidate["seed"].astype(str) == "ensemble"].set_index("file")
    for gas in GASES:
        metrics = metric_row(ensemble[f"{gas}_true"].to_numpy(float), ensemble[f"{gas}_pred"].to_numpy(float))
        frame_rows.append({"scope": "mean_ensemble", "gas": gas, **metrics})
    median_ensemble = ensemble.copy()
    for gas in GASES:
        stack = np.stack([per_seed[seed][f"{gas}_pred"].to_numpy(float) for seed in SEED_NAMES])
        median_ensemble[f"{gas}_pred"] = np.median(stack, axis=0)
        metrics = metric_row(median_ensemble[f"{gas}_true"].to_numpy(float), median_ensemble[f"{gas}_pred"].to_numpy(float))
        frame_rows.append({"scope": "median_ensemble", "gas": gas, **metrics})
    frame = pd.DataFrame(frame_rows)
    macro_rows: list[dict[str, object]] = []
    for scope, subset in frame.groupby("scope"):
        macro_rows.append(
            {
                "scope": scope,
                "gas": "macro",
                "three_sigma": float(subset["three_sigma"].mean()),
                "mape": float(subset["mape"].mean()),
                "mean_relative": float(subset["mean_relative"].mean()),
                "rmse_relative": float(subset["rmse_relative"].mean()),
                "std_relative": float(subset["std_relative"].mean()),
            }
        )
    frame = pd.concat([frame, pd.DataFrame(macro_rows)], ignore_index=True)
    frame.to_csv(OUT / "test_descriptive.csv", index=False)

    # 测试集逐点明细 + 尾部（描述性）
    detail_rows: list[dict[str, object]] = []
    for gas in GASES:
        rel_ensemble = rel_percent(ensemble[f"{gas}_pred"].to_numpy(float), ensemble[f"{gas}_true"].to_numpy(float))
        for position, file in enumerate(ensemble.index):
            record = {
                "gas": gas,
                "file": file,
                "true": float(ensemble[f"{gas}_true"].iloc[position]),
                "pred_ensemble": float(ensemble[f"{gas}_pred"].iloc[position]),
                "rel_ensemble": float(rel_ensemble[position]),
            }
            for seed in SEED_NAMES:
                record[f"rel_seed_{seed}"] = float(
                    rel_percent(per_seed[seed].loc[file, f"{gas}_pred"], per_seed[seed].loc[file, f"{gas}_true"])
                )
            detail_rows.append(record)
    detail = pd.DataFrame(detail_rows)
    detail.to_csv(OUT / "test_oof_like_detail.csv", index=False)

    # 描述性：若对 ensemble 预测做训练范围裁剪，测试 3σ 会怎样（方法来自 OOF，不在测试集上比较）
    dev_ranges = {gas: (float(manifest.loc[manifest["split"] == "development", gas].min()),
                        float(manifest.loc[manifest["split"] == "development", gas].max())) for gas in GASES}
    clip_rows: list[dict[str, object]] = []
    for scope, frame_sc in (("ensemble", ensemble), ("median_ensemble", median_ensemble)):
        for gas in GASES:
            true = frame_sc[f"{gas}_true"].to_numpy(float)
            pred = frame_sc[f"{gas}_pred"].to_numpy(float)
            before = metric_row(true, pred)
            lo, hi = dev_ranges[gas]
            after = metric_row(true, np.clip(pred, lo, hi))
            clip_rows.append(
                {
                    "scope": scope,
                    "gas": gas,
                    "three_sigma_before": before["three_sigma"],
                    "three_sigma_after_clip": after["three_sigma"],
                    "mape_before": before["mape"],
                    "mape_after_clip": after["mape"],
                    "clip_range": f"[{lo:g}, {hi:g}]",
                    "changed_points": int(np.sum(np.abs(np.clip(pred, lo, hi) - pred) > 1e-9)),
                }
            )
    clip_frame = pd.DataFrame(clip_rows)
    clip_frame.to_csv(OUT / "test_clip_descriptive.csv", index=False)
    return {"descriptive": frame, "detail": detail, "clip": clip_frame, "dev_ranges": dev_ranges}


def analyse_scheme_comparison(oof: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """同一批开发实验上，各训练方案 / seed 的 3σ、MAPE 与浓度档位平均偏差。

    这里只做描述，不用于选择方案；它用于说明“单 seed OOF 3σ 的跨方案摆动”
    与“H2 高浓度段系统性低估”是否在所有方案上都存在。
    """
    schemes = {CURRENT_SCHEME: oof, **{name: pd.read_csv(path) for name, path in COMPARISON_SCHEMES.items()}}
    summary_rows: list[dict[str, object]] = []
    level_rows: list[dict[str, object]] = []
    for name, frame in schemes.items():
        for gas in GASES:
            true = frame[f"{gas}_true"].to_numpy(float)
            pred = frame[f"{gas}_pred"].to_numpy(float)
            metrics = metric_row(true, pred)
            slope, intercept = np.polyfit(true, pred, 1)
            log_slope, log_intercept = np.polyfit(np.log(true), np.log(pred), 1)
            summary_rows.append(
                {
                    "scheme": name,
                    "gas": gas,
                    "three_sigma": metrics["three_sigma"],
                    "mape": metrics["mape"],
                    "mean_relative": metrics["mean_relative"],
                    "linear_slope_pred_vs_true": float(slope),
                    "loglog_slope": float(log_slope),
                }
            )
            for level, group in pd.DataFrame({"true": true, "rel": rel_percent(pred, true)}).groupby("true"):
                level_rows.append(
                    {
                        "scheme": name,
                        "gas": gas,
                        "concentration_level": float(level),
                        "n": len(group),
                        "mean_relative": float(group["rel"].mean()),
                        "mean_absrel": float(group["rel"].abs().mean()),
                    }
                )
    summary = pd.DataFrame(summary_rows)
    levels = pd.DataFrame(level_rows)
    summary.to_csv(OUT / "scheme_comparison.csv", index=False)
    levels.to_csv(OUT / "scheme_level_bias.csv", index=False)
    return {"summary": summary, "levels": levels}


# ---------------------------------------------------------------------------
# 图
# ---------------------------------------------------------------------------
def setup_matplotlib() -> str:
    available = {item.name for item in font_manager.fontManager.ttflist}
    for candidate in (
        "Noto Serif CJK SC",
        "Source Han Serif CN",
        "Microsoft YaHei",
        "Source Han Sans CN",
        "LXGW WenKai",
        "SimSun",
        "DejaVu Sans",
    ):
        if candidate in available:
            family = candidate
            break
    else:  # pragma: no cover
        family = "DejaVu Sans"
    plt.rcParams.update(
        {
            "font.family": family,
            "font.serif": [family],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "savefig.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )
    return family


def save_figure(figure: plt.Figure, name: str) -> None:
    for suffix in ("png", "svg"):
        path = FIG / f"{name}.{suffix}"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        print(f"Wrote {path}")
    plt.close(figure)


def plot_tail(detail: pd.DataFrame, tail: dict[str, object], output_name: str = "fig_tail_outliers") -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), constrained_layout=True)
    for panel, gas in zip(axes, GASES):
        rel = detail[f"{gas}_rel"].to_numpy(float)
        order = np.argsort(-np.abs(rel))
        ordered = rel[order]
        colors = ["#B5443A" if rank < 5 else "#7E9AA6" for rank in range(len(ordered))]
        panel.bar(np.arange(len(ordered)), ordered, color=colors, width=0.9, linewidth=0)
        sigma = sigma3(rel)
        mean = float(np.mean(rel))
        std = sigma / 3.0
        panel.axhline(mean, color="#333333", linewidth=0.9, linestyle=":", label="均值")
        panel.axhline(mean + std, color="#B5443A", linewidth=0.8, linestyle="--", alpha=0.75, label="均值 ± 1σ")
        panel.axhline(mean - std, color="#B5443A", linewidth=0.8, linestyle="--", alpha=0.75)
        top_files = "  ".join(
            f"{detail.loc[index, 'file'].replace('data/', '').replace('.csv', '')}"
            f"({rel[index]:+.1f}%)"
            for index in order[:5]
        )
        trimmed = {k: sigma3(np.delete(rel, order[:k])) for k in (1, 2, 3)}
        panel.text(
            0.98,
            0.03,
            "top5（file：rel）：\n" + top_files + "\n"
            f"去 top1/2/3 后 3σ：{trimmed[1]:.2f} / {trimmed[2]:.2f} / {trimmed[3]:.2f}%",
            transform=panel.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.8,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "linewidth": 0.6, "alpha": 0.92},
        )
        panel.set_title(f"{gas}：3σ={sigma:.2f}%，MAPE={np.mean(np.abs(rel)):.2f}%", loc="left")
        panel.set_xlabel("实验（按 |rel| 从大到小排序）")
        panel.set_ylabel("相对误差 rel / %")
        panel.legend(loc="upper right", frameon=False, fontsize=7)
        panel.grid(axis="y")
        panel.set_xlim(-1, len(ordered))
    figure.suptitle("图 1  当前 10 s 加权方案 OOF 的逐实验相对误差：3σ 由每气体 2–3 个实验主导（红 = top5）", fontsize=11)
    save_figure(figure, output_name)


def plot_factor_relations(detail: pd.DataFrame, output_name: str = "fig_factor_relations") -> None:
    figure, axes = plt.subplots(3, 4, figsize=(15.5, 9.0), constrained_layout=True)
    fold_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 5))
    for row, gas in enumerate(GASES):
        absrel = detail[f"{gas}_absrel"].to_numpy(float)
        # 1) 真实浓度
        panel = axes[row, 0]
        concentration = detail[f"{gas}_true"].to_numpy(float)
        jitter = (RNG.random(len(concentration)) - 0.5) * 0.12
        panel.scatter(np.log10(concentration) + jitter, absrel, s=16, c=[fold_colors[f] for f in detail["fold"]], alpha=0.85, linewidth=0)
        rho = stats.spearmanr(absrel, concentration).statistic
        panel.set_xlabel(f"log10({gas} 真实浓度 / ppm)")
        panel.set_ylabel("|rel| / %")
        panel.set_title(f"{gas}: |rel| vs 真实浓度  Spearman ρ={rho:+.2f}", loc="left", fontsize=8.5)
        # 2) 响应幅度
        panel = axes[row, 1]
        amplitude = detail[f"delta_{gas}"].abs().to_numpy(float)
        panel.scatter(amplitude, absrel, s=16, c=[fold_colors[f] for f in detail["fold"]], alpha=0.85, linewidth=0)
        panel.set_xscale("log")
        rho = stats.spearmanr(absrel, amplitude).statistic
        panel.set_xlabel("|10 s 响应相对增量|（log）")
        panel.set_title(f"|rel| vs 响应幅度  ρ={rho:+.2f}", loc="left", fontsize=8.5)
        # 3) 相对噪声
        panel = axes[row, 2]
        relnoise = detail[f"relnoise_{gas}"].to_numpy(float)
        panel.scatter(relnoise, absrel, s=16, c=[fold_colors[f] for f in detail["fold"]], alpha=0.85, linewidth=0)
        panel.set_xscale("log")
        rho = stats.spearmanr(absrel, relnoise).statistic
        panel.set_xlabel("噪声 σ(0–1 s) / |响应幅度|（log）")
        panel.set_title(f"|rel| vs 相对噪声  ρ={rho:+.2f}", loc="left", fontsize=8.5)
        # 4) fold 分组
        panel = axes[row, 3]
        groups = [detail.loc[detail["fold"] == f, f"{gas}_absrel"] for f in range(5)]
        box = panel.boxplot(groups, positions=range(5), widths=0.55, patch_artist=True, showfliers=True)
        for patch, color in zip(box["boxes"], fold_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        for f in range(5):
            panel.scatter([f] * len(groups[f]), groups[f], s=9, color="#333333", alpha=0.55, linewidth=0)
        panel.set_xlabel("fold")
        panel.set_title(f"|rel| 按 fold 分组（均值见下）", loc="left", fontsize=8.5)
        panel.text(
            0.03,
            0.97,
            " / ".join(f"f{f}:{detail.loc[detail['fold'] == f, f'{gas}_absrel'].mean():.1f}" for f in range(5)),
            transform=panel.transAxes,
            va="top",
            fontsize=6.5,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "linewidth": 0.5, "alpha": 0.85},
        )
    for panel in axes.ravel():
        panel.grid(axis="y")
        panel.set_ylim(bottom=0)
    figure.suptitle("图 2  |rel| 与候选因子的关系（颜色 = fold；逐气体）", fontsize=11)
    save_figure(figure, output_name)


def plot_calibration(detail: pd.DataFrame, output_name: str = "fig_calibration") -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    fold_colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, 5))
    for panel, gas in zip(axes, GASES):
        true = detail[f"{gas}_true"].to_numpy(float)
        pred = detail[f"{gas}_pred"].to_numpy(float)
        panel.scatter(true, pred, s=18, c=[fold_colors[f] for f in detail["fold"]], alpha=0.9, linewidth=0)
        low = min(true.min(), pred.min()) * 0.9
        high = max(true.max(), pred.max()) * 1.05
        line = np.array([low, high])
        panel.plot(line, line, color="#333333", linestyle="--", linewidth=0.9, label="pred = true")
        slope, intercept = np.polyfit(true, pred, 1)
        panel.plot(line, slope * line + intercept, color="#B5443A", linewidth=1.2, label=f"OLS: pred={slope:.3f}·true+{intercept:.0f}")
        log_slope, log_intercept = np.polyfit(np.log(true), np.log(pred), 1)
        panel.set_xscale("log")
        panel.set_yscale("log")
        panel.set_xlabel(f"{gas} 真实浓度 / ppm")
        panel.set_ylabel(f"{gas} 预测浓度 / ppm")
        panel.set_title(f"{gas}：log-log 斜率 {log_slope:.3f}", loc="left")
        panel.legend(loc="upper left", frameon=False, fontsize=7)
        panel.grid(True, which="both")
    figure.suptitle("图 3  预测 vs 真实：H2 在高浓度端系统性偏低（斜率 < 1）", fontsize=11)
    save_figure(figure, output_name)


def plot_postprocessing(metrics: pd.DataFrame, degeneracy: pd.DataFrame, output_name: str = "fig_postprocess") -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 4.6), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1.0]})
    macro = metrics[(metrics["gas"] == "macro") & (metrics["fit"] == "crossfit")].set_index("transform")
    order = ["baseline", "clip_range", "bias_ls", "scale_mape", "scale_ls", "bias_mape", "loglinear", "affine_ls"]
    panel = axes[0]
    values = [macro.loc[name, "three_sigma"] for name in order]
    colors = ["#4A4A4A" if name == "baseline" else ("#2A7F8F" if macro.loc[name, "delta_percent"] < 0 else "#B5443A") for name in order]
    bars = panel.bar(range(len(order)), values, color=colors, width=0.65)
    for bar, value, name in zip(bars, values, order):
        delta = macro.loc[name, "delta_percent"]
        label = f"{value:.2f}%" if name == "baseline" else f"{value:.2f}%\n({delta:+.1f}%)"
        panel.text(bar.get_x() + bar.get_width() / 2, value + 0.25, label, ha="center", va="bottom", fontsize=6.5)
    panel.axhline(14.46, color="#666666", linestyle=":", linewidth=0.9)
    panel.axhline(10.0, color="#C0504D", linestyle="--", linewidth=0.9)
    panel.axhline(5.0, color="#C0504D", linestyle=":", linewidth=0.9)
    panel.text(
        0.985,
        10.15,
        "10% 上限",
        ha="right",
        fontsize=6.8,
        color="#C0504D",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
    )
    panel.text(
        0.985,
        5.15,
        "5% 目标",
        ha="right",
        fontsize=6.8,
        color="#C0504D",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
    )
    panel.set_xticks(range(len(order)), ["基线", "裁剪范围", "加性LS", "缩放MAPE", "缩放LS", "加性MAPE", "log校准", "仿射LS"], fontsize=7.5)
    panel.set_ylabel("宏平均相对误差 3σ / %")
    panel.set_title("(a) 交叉拟合后处理的宏平均 3σ（括号为相对基线的变化）", loc="left", fontsize=9)
    panel.grid(axis="y")
    panel.set_ylim(0, max(18.0, max(values) + 2.5))

    panel = axes[1]
    gases = list(GASES)
    width = 0.26
    baseline = metrics[(metrics["fit"] == "crossfit") & (metrics["transform"] == "baseline")].set_index("gas")
    clip = metrics[(metrics["fit"] == "crossfit") & (metrics["transform"] == "clip_range")].set_index("gas")
    loglin = metrics[(metrics["fit"] == "crossfit") & (metrics["transform"] == "loglinear")].set_index("gas")
    for offset, frame, label, color in (
        (-width, baseline, "基线", "#4A4A4A"),
        (0.0, clip, "裁剪到训练范围", "#2A7F8F"),
        (width, loglin, "log 空间校准", "#B5443A"),
    ):
        positions = np.arange(len(gases)) + offset
        bars = panel.bar(positions, [frame.loc[gas, "three_sigma"] for gas in gases], width, label=label, color=color)
        for bar, gas in zip(bars, gases):
            panel.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15, f"{frame.loc[gas, 'three_sigma']:.2f}", ha="center", fontsize=6.5)
    panel.set_xticks(np.arange(len(gases)), gases)
    panel.set_ylabel("相对误差 3σ / %")
    panel.set_title("(b) 逐气体 3σ：只有裁剪范围稳定改善", loc="left", fontsize=9)
    panel.legend(frameon=False, fontsize=7)
    panel.grid(axis="y")

    figure.suptitle("图 4  OOF 交叉拟合后处理：常规校准不能降 3σ，裁剪训练范围可降约 6%（注意 3σ 的尺度退化）", fontsize=10.5)
    save_figure(figure, output_name)


def plot_seed_variance(proxy: pd.DataFrame, test_descriptive: pd.DataFrame, output_name: str = "fig_seed_variance") -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 4.2), constrained_layout=True)
    panel = axes[0]
    macro = proxy[(proxy["gas"] == "macro")]
    configs = ["k4", "k8"]
    x_base = np.arange(len(configs))
    seed_colors = {"seed_0": "#9AA5AD", "seed_1": "#C4B08A", "seed_2": "#8A9BB8"}
    seed_labels = {"seed_0": "seed 23", "seed_1": "seed 24", "seed_2": "seed 25"}
    seed_offsets = {"seed_0": -0.08, "seed_1": 0.0, "seed_2": 0.08}
    for aggregation in ("seed_0", "seed_1", "seed_2"):
        values = [
            float(macro[(macro["config"] == config) & (macro["aggregation"] == aggregation)]["three_sigma"].iloc[0])
            for config in configs
        ]
        panel.scatter(
            x_base + seed_offsets[aggregation],
            values,
            s=42,
            color=seed_colors[aggregation],
            zorder=3,
            edgecolor="white",
            linewidth=0.6,
        )
    for offset, aggregation, color, label in (
        (0.26, "mean_ensemble", "#1F6F78", "3-seed 均值"),
        (0.42, "median_ensemble", "#B5443A", "3-seed 中位数"),
    ):
        values = [
            float(macro[(macro["config"] == config) & (macro["aggregation"] == aggregation)]["three_sigma"].iloc[0])
            for config in configs
        ]
        bars = panel.bar(x_base + offset, values, width=0.13, color=color, label=label)
        for bar, value in zip(bars, values):
            panel.text(bar.get_x() + bar.get_width() / 2, value + 0.25, f"{value:.1f}", ha="center", fontsize=6.8)
    spreads = {}
    for config in configs:
        seeds = [
            float(macro[(macro["config"] == config) & (macro["aggregation"] == aggregation)]["three_sigma"].iloc[0])
            for aggregation in ("seed_0", "seed_1", "seed_2")
        ]
        spreads[config] = max(seeds) - min(seeds)
    panel.text(
        0.03,
        0.97,
        "圆点 = 单 seed（23/24/25）；柱 = 3-seed 集成\n"
        f"k4 seed 极差 {spreads['k4']:.1f} pp；k8 seed 极差 {spreads['k8']:.1f} pp",
        transform=panel.transAxes,
        fontsize=7,
        ha="left",
        va="top",
        color="#555555",
        bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "linewidth": 0.5, "alpha": 0.9},
    )
    panel.set_xticks(x_base, ["k4（5 折 OOF 代理）", "k8（5 折 OOF 代理）"])
    panel.set_xlim(-0.55, 1.95)
    panel.set_ylim(0, max(macro["three_sigma"]) + 2.5)
    panel.set_ylabel("宏平均相对误差 3σ / %")
    panel.set_title("(a) 同一配置不同 seed 的 OOF 宏 3σ：seed 间摆动 2.7–5.9 个百分点", loc="left", fontsize=9)
    panel.legend(loc="upper right", frameon=False, fontsize=7)
    panel.grid(axis="y")

    panel = axes[1]
    macro_test = test_descriptive[test_descriptive["gas"] == "macro"].set_index("scope")
    for position, (scope, color, label) in enumerate(
        (
            ("seed_20260923", "#9AA5AD", "seed 23"),
            ("seed_20260924", "#C4B08A", "seed 24"),
            ("seed_20260925", "#8A9BB8", "seed 25"),
            ("mean_ensemble", "#1F6F78", "均值集成"),
            ("median_ensemble", "#B5443A", "中位数集成"),
        )
    ):
        value = float(macro_test.loc[scope, "three_sigma"])
        panel.bar(position, value, color=color, width=0.6)
        panel.text(position, value + 0.2, f"{value:.2f}", ha="center", fontsize=7.5)
    panel.set_xticks(range(5), ["seed 23", "seed 24", "seed 25", "均值集成", "中位数集成"], fontsize=8)
    panel.set_ylabel("宏平均相对误差 3σ / %")
    panel.set_title("(b) 锁定测试集 12 点（仅描述）：seed 集成把宏 3σ 从 10.6–17.3% 收到 10.8–11.2%", loc="left", fontsize=9)
    panel.grid(axis="y")
    panel.set_ylim(0, max(20.0, macro_test["three_sigma"].max() + 2.0))
    figure.suptitle("图 5  seed 方差：单 seed 的 3σ 抖动与“集成”收益", fontsize=11)
    save_figure(figure, output_name)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    family = setup_matplotlib()
    print(f"[font] {family}")

    manifest = pd.read_csv(ROOT / "artifacts/splits/manifest.csv")
    oof = pd.read_csv(CURRENT_OOF)
    assert len(oof) == 50, len(oof)
    features = load_dev_features(manifest)
    assert len(features) == 50

    current = summarize(oof)
    baseline_rows = []
    for gas in (*GASES, "macro"):
        metrics = current[gas]
        baseline_rows.append({"scope": "oof", "gas": gas, **metrics})
    baseline_frame = pd.DataFrame(baseline_rows)
    baseline_frame.to_csv(OUT / "baseline_summary.csv", index=False)

    tail = analyse_tail(oof, features)
    stability = analyse_stability(oof)
    factors = analyse_factors(tail["detail"])
    postprocessing = evaluate_postprocessing(oof)
    bootstrap = bootstrap_postprocessing(oof)
    fold_consistency = evaluate_fold_consistency(oof)
    seed_variance = analyse_seed_variance()
    scheme_comparison = analyse_scheme_comparison(oof)
    test_descriptive = analyse_test_descriptive(manifest, features)

    plot_tail(tail["detail"], tail)
    plot_factor_relations(tail["detail"])
    plot_calibration(tail["detail"])
    plot_postprocessing(postprocessing["metrics"], postprocessing["degeneracy"])
    plot_seed_variance(seed_variance["proxy"], test_descriptive["descriptive"])

    # 汇总 JSON（供报告引用）
    summary: dict[str, object] = {
        "font": family,
        "metric": "rel=100*(pred-true)/max(|true|,1e-9); 3sigma=3*std(rel,ddof=1); macro=mean over gases",
        "baseline_oof": current,
        "tail_sensitivity": tail["tail_sensitivity"].to_dict(orient="records"),
        "tail_sensitivity_union": tail["union"].to_dict(orient="records"),
        "top_outliers": tail["top5"].to_dict(orient="records"),
        "rank_consistency": stability["pairwise"].to_dict(orient="records"),
        "postprocess_crossfit": postprocessing["metrics"][(postprocessing["metrics"]["fit"] == "crossfit") & (postprocessing["metrics"]["gas"] == "macro")].to_dict(orient="records"),
        "postprocess_insample": postprocessing["metrics"][(postprocessing["metrics"]["fit"] == "insample") & (postprocessing["metrics"]["gas"] == "macro")].to_dict(orient="records"),
        "scale_degeneracy": postprocessing["degeneracy"].to_dict(orient="records"),
        "bootstrap": bootstrap.to_dict(orient="records"),
        "fold_consistency": fold_consistency.to_dict(orient="records"),
        "test_descriptive_macro": test_descriptive["descriptive"][test_descriptive["descriptive"]["gas"] == "macro"].to_dict(orient="records"),
        "dev_ranges": test_descriptive["dev_ranges"],
        "scheme_comparison": scheme_comparison["summary"].to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
    print("Wrote", OUT / "summary.json")
    print("\n=== OOF baseline ===")
    print(baseline_frame.to_string(index=False))
    print("\n=== 宏平均 3σ：交叉拟合后处理 ===")
    print(
        postprocessing["metrics"][(postprocessing["metrics"]["gas"] == "macro") & (postprocessing["metrics"]["fit"] == "crossfit")][
            ["transform", "three_sigma", "delta_percent", "mape", "mape_delta_percent"]
        ].to_string(index=False)
    )
    print("\n=== 测试集描述性宏平均 ===")
    print(test_descriptive["descriptive"][test_descriptive["descriptive"]["gas"] == "macro"].to_string(index=False))


if __name__ == "__main__":
    main()
