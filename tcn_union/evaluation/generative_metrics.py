"""生成数据的真实性/多样性指标。

定义（所有曲线形状均为 [N, 3, 51]）：

- Wasserstein Distance：逐气体、逐时间点计算一维经验 W1 距离，再对时间平均；
- ACF Distance：逐曲线计算 lag=1..10 的自相关，比较真实/合成的平均 ACF；
- Pairwise Distance Ratio：合成样本平均两两距离 / 真实样本平均两两距离；
- Nearest Neighbor Distance：合成样本到最近真实样本的平均距离（按真实标准差归一化），
  同时报告它相对真实样本最近邻距离的比值。

除 Wasserstein 同时给出原始单位外，其余距离都按真实数据逐气体标准差归一化。
"""
from __future__ import annotations

from typing import Any

import numpy as np

AXES = ("H2", "NH3", "NO2")
ACF_LAGS = tuple(range(1, 11))


def _wasserstein_1d(real: np.ndarray, fake: np.ndarray) -> float:
    """一维经验 W1；样本数不同时在共同分位点上比较。"""
    if len(real) == len(fake):
        return float(np.mean(np.abs(np.sort(real) - np.sort(fake))))
    quantiles = np.linspace(0.0, 1.0, max(len(real), len(fake)))
    return float(
        np.mean(
            np.abs(
                np.quantile(real, quantiles, method="linear")
                - np.quantile(fake, quantiles, method="linear")
            )
        )
    )


def _acf_curve(curve: np.ndarray) -> np.ndarray:
    centered = curve - curve.mean()
    denominator = float(np.sum(centered**2))
    if denominator < 1e-12:
        return np.zeros(len(ACF_LAGS))
    return np.array(
        [float(np.sum(centered[:-lag] * centered[lag:]) / denominator) for lag in ACF_LAGS]
    )


def _mean_pairwise_distance(flat: np.ndarray) -> float:
    if len(flat) < 2:
        return 0.0
    distances = np.linalg.norm(flat[:, None, :] - flat[None, :, :], axis=2)
    return float(distances[np.triu_indices(len(flat), k=1)].mean())


def _nearest_distances(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    real_distance = np.linalg.norm(real[:, None, :] - real[None, :, :], axis=2)
    np.fill_diagonal(real_distance, np.inf)
    real_nn = float(real_distance.min(axis=1).mean())
    fake_distance = np.linalg.norm(fake[:, None, :] - real[None, :, :], axis=2)
    fake_nn = float(fake_distance.min(axis=1).mean())
    return fake_nn, fake_nn / max(real_nn, 1e-12)


def evaluate_generation(real: np.ndarray, fake: np.ndarray) -> dict[str, Any]:
    """返回逐气体指标和 macro 指标。"""
    if real.ndim != 3 or fake.ndim != 3 or real.shape[1:] != fake.shape[1:]:
        raise ValueError("real/fake 必须是 [N,3,L]，且后两维一致")
    if len(real) < 2 or len(fake) < 2:
        raise ValueError("至少需要两个真实/合成样本")

    scale = real.std(axis=(0, 2))
    scale[scale < 1e-8] = 1.0
    real_norm = real / scale[None, :, None]
    fake_norm = fake / scale[None, :, None]

    by_gas: dict[str, Any] = {}
    for index, gas in enumerate(AXES):
        wd = float(
            np.mean(
                [
                    _wasserstein_1d(real[:, index, t], fake[:, index, t])
                    for t in range(real.shape[2])
                ]
            )
        )
        real_acf = np.mean([_acf_curve(curve) for curve in real[:, index]], axis=0)
        fake_acf = np.mean([_acf_curve(curve) for curve in fake[:, index]], axis=0)
        acf_distance = float(np.mean(np.abs(real_acf - fake_acf)))
        real_flat = real_norm[:, index].reshape(len(real), -1)
        fake_flat = fake_norm[:, index].reshape(len(fake), -1)
        pairwise_ratio = _mean_pairwise_distance(fake_flat) / max(
            _mean_pairwise_distance(real_flat), 1e-12
        )
        nn_distance, nn_ratio = _nearest_distances(real_flat, fake_flat)
        by_gas[gas] = {
            "wasserstein_raw": wd,
            "wasserstein_normalized": wd / scale[index],
            "acf_distance": acf_distance,
            "pairwise_distance_ratio": float(pairwise_ratio),
            "nearest_neighbor_distance": nn_distance,
            "nearest_neighbor_ratio": nn_ratio,
        }

    # 三气体合并后的距离/多样性，作为辅助视图。
    real_all = real_norm.reshape(len(real), -1)
    fake_all = fake_norm.reshape(len(fake), -1)
    all_pairwise = _mean_pairwise_distance(fake_all) / max(
        _mean_pairwise_distance(real_all), 1e-12
    )
    all_nn, all_nn_ratio = _nearest_distances(real_all, fake_all)

    macro = {
        "wasserstein_normalized": float(
            np.mean([by_gas[gas]["wasserstein_normalized"] for gas in AXES])
        ),
        "acf_distance": float(np.mean([by_gas[gas]["acf_distance"] for gas in AXES])),
        "pairwise_distance_ratio": float(
            np.mean([by_gas[gas]["pairwise_distance_ratio"] for gas in AXES])
        ),
        "nearest_neighbor_distance": float(
            np.mean([by_gas[gas]["nearest_neighbor_distance"] for gas in AXES])
        ),
        "nearest_neighbor_ratio": float(
            np.mean([by_gas[gas]["nearest_neighbor_ratio"] for gas in AXES])
        ),
        "all_gases_pairwise_distance_ratio": float(all_pairwise),
        "all_gases_nearest_neighbor_distance": float(all_nn),
        "all_gases_nearest_neighbor_ratio": float(all_nn_ratio),
    }
    return {"per_gas": by_gas, "macro": macro, "n_real": len(real), "n_fake": len(fake)}
