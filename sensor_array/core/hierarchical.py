"""分层随机效应正模型：把实验级增益、时间尺度和漂移显式建模。

方案 1 的生成式形式为

    R_ij(t) = R_0ij [1 + s_j exp(g_ij) h_j(t / exp(l_ij); c_i) + d_ij t]

其中 ``h_j`` 是已拟合的双指数 + MLP 正模型的相对响应，``g_ij``、``l_ij``、
``d_ij`` 是第 i 次实验第 j 路传感器的随机效应。随机效应对 log 浓度做二次
条件均值，并用收缩协方差描述实验间波动。这样基线 R0 仍作为观测条件，
增益、时间常数和缓慢漂移由模型内部采样，而不是在生成后再叠 PCA 残差。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .data import Experiment


def _design(labels: np.ndarray) -> np.ndarray:
    u = np.log(np.asarray(labels, dtype=float))
    return np.column_stack(
        (
            np.ones(len(u)),
            u,
            u**2,
            u[:, [0]] * u[:, [1]],
            u[:, [0]] * u[:, [2]],
            u[:, [1]] * u[:, [2]],
        )
    )


@dataclass
class HierarchicalRandomEffectsModel:
    """由训练折实验拟合的分层随机效应生成器。"""

    ridge: float = 1e-2
    shrink: float = 0.3
    gain_bound: float = 0.7
    time_bound: float = 0.7
    drift_bound: float = 2e-4

    def fit(
        self,
        model,
        experiments: list[Experiment],
        seconds: int = 10,
    ) -> "HierarchicalRandomEffectsModel":
        times = experiments[0].time
        relative = np.stack([experiment.relative for experiment in experiments])
        labels = np.stack([experiment.concentration for experiment in experiments])
        prediction = model.predict_relative(labels, times)
        count = len(experiments)

        effects = np.empty((count, 3, 3), dtype=float)
        for index in range(count):
            for gas in range(3):
                effects[index, gas] = self._fit_one(
                    times,
                    relative[index, :, gas],
                    prediction[index, :, gas],
                )

        self.labels_ = labels
        self.effects_ = effects.reshape(count, 9)
        self.low_ = self.effects_.min(axis=0)
        self.high_ = self.effects_.max(axis=0)
        design = _design(labels)
        self.design_mean_ = design[:, 1:].mean(axis=0)
        self.design_scale_ = design[:, 1:].std(axis=0)
        self.design_scale_[self.design_scale_ == 0.0] = 1.0
        design[:, 1:] = (design[:, 1:] - self.design_mean_) / self.design_scale_

        penalty = self.ridge * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self.coefficients_ = np.linalg.solve(
            design.T @ design + penalty, design.T @ self.effects_
        )
        residual = self.effects_ - design @ self.coefficients_
        covariance = np.cov(residual, rowvar=False)
        covariance = 0.5 * (covariance + covariance.T)
        diagonal = np.diag(np.diag(covariance))
        covariance = (1.0 - self.shrink) * covariance + self.shrink * diagonal
        covariance += 1e-8 * np.eye(covariance.shape[0])
        self.covariance_ = covariance
        return self

    def _fit_one(
        self,
        times: np.ndarray,
        observed: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        scale = max(float(np.sqrt(np.mean(observed**2))), 1e-3)
        bounds = (
            [-self.gain_bound, -self.time_bound, -self.drift_bound],
            [self.gain_bound, self.time_bound, self.drift_bound],
        )

        def residual(parameters: np.ndarray) -> np.ndarray:
            gain, time, drift = parameters
            scaled = np.interp(
                times / np.exp(time),
                times,
                baseline,
            )
            prediction = np.exp(gain) * scaled + drift * times
            return (prediction - observed) / scale

        result = least_squares(
            residual,
            x0=np.zeros(3),
            bounds=bounds,
            max_nfev=120,
            loss="soft_l1",
            f_scale=0.05,
        )
        return result.x

    def sample(
        self,
        model,
        labels: np.ndarray,
        initial: np.ndarray,
        seconds: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        labels = np.asarray(labels, dtype=float)
        initial = np.asarray(initial, dtype=float)
        if initial.ndim == 1:
            initial = initial[None, :]
        design = _design(labels)
        design[:, 1:] = (design[:, 1:] - self.design_mean_) / self.design_scale_
        mean = design @ self.coefficients_
        effects = rng.multivariate_normal(
            np.zeros(self.covariance_.shape[0]),
            self.covariance_,
            size=len(labels),
        )
        effects = np.clip(effects + mean, self.low_, self.high_)
        effects = effects.reshape(len(labels), 3, 3)

        times = np.arange(round(seconds / 0.2) + 1) * 0.2
        baseline = model.predict_relative(labels, times)
        relative = np.empty_like(baseline)
        for gas in range(3):
            gain = np.exp(effects[:, gas, 0])
            time_scale = np.exp(effects[:, gas, 1])
            drift = effects[:, gas, 2]
            for index in range(len(labels)):
                scaled = np.interp(
                    times / time_scale[index],
                    times,
                    baseline[index, :, gas],
                )
                relative[index, :, gas] = gain[index] * scaled + drift[index] * times
        return initial[:, None, :] * (1.0 + relative)
