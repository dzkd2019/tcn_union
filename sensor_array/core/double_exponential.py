"""三路传感器阵列的双指数响应骨架。

每路相对响应写成两个指数分量按浓度相关权重混合：

    y_j(t) = s_j A_j(c) [ f_j(c) (1 - exp(-max(t-d_j,0)/tau_fast_j(c)))
                        + (1-f_j(c)) (1 - exp(-max(t-d_j,0)/tau_slow_j(c))) ]

其中 A_j>0、tau_fast_j>0、tau_slow_j>tau_fast_j，f_j 用 logistic 参数化。
A_j 是 log 浓度的二次多项式，tau_fast_j、log(tau_slow_j-tau_fast_j) 和
logit(f_j) 是一次多项式。模型由逐实验形状拟合给出初值，再用加权最小二乘
与岭惩罚联合拟合。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit

from .data import REFERENCE_CONCENTRATION, RESPONSE_SIGNS, Experiment
from .shapes import (
    DELAY_BOUNDS,
    fit_experiment_shapes,
    monomial_features,
    ridge_least_squares,
)


class DoubleExponentialArrayModel:
    """双指数响应骨架。"""

    def __init__(
        self,
        reference: np.ndarray = REFERENCE_CONCENTRATION,
        signs: np.ndarray = RESPONSE_SIGNS,
        gain_degree: int = 2,
        tau_degree: int = 1,
        ridge: float = 1e-3,
        max_nfev: int = 600,
    ) -> None:
        self.reference = np.asarray(reference, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.gain_degree = gain_degree
        self.tau_degree = tau_degree
        self.ridge = ridge
        self.max_nfev = max_nfev
        self.gain_coef: np.ndarray | None = None
        self.tau_fast_coef: np.ndarray | None = None
        self.delta_coef: np.ndarray | None = None
        self.mix_coef: np.ndarray | None = None
        self.delay: np.ndarray | None = None
        self.nfev_: int | None = None
        self.cost_: float | None = None

    def _features(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.log(np.asarray(concentration, dtype=float) / self.reference)
        return monomial_features(u, self.gain_degree), monomial_features(u, self.tau_degree)

    def _predict_with(
        self,
        gain_coef: np.ndarray,
        tau_fast_coef: np.ndarray,
        delta_coef: np.ndarray,
        mix_coef: np.ndarray,
        delay: np.ndarray,
        concentration: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        gain_features, tau_features = self._features(concentration)
        amplitude = np.exp(gain_features @ gain_coef.T)
        tau_fast = np.exp(tau_features @ tau_fast_coef.T)
        tau_slow = tau_fast + np.exp(tau_features @ delta_coef.T)
        mix = expit(tau_features @ mix_coef.T)
        t = np.clip(time[None, :, None] - delay[None, None, :], 0.0, None)
        fast = 1.0 - np.exp(-t / tau_fast[:, None, :])
        slow = 1.0 - np.exp(-t / tau_slow[:, None, :])
        shape = mix[:, None, :] * fast + (1.0 - mix[:, None, :]) * slow
        return self.signs[None, None, :] * amplitude[:, None, :] * shape

    def fit(
        self,
        experiments: list[Experiment],
        shape_init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> DoubleExponentialArrayModel:
        """由逐实验形状初值联合拟合双指数骨架。"""
        time = experiments[0].time
        relative = np.stack([experiment.relative for experiment in experiments])
        concentration = np.stack([experiment.concentration for experiment in experiments])
        if shape_init is None:
            shape_init = fit_experiment_shapes(experiments)
        amplitude, tau0, delay0 = shape_init
        gain_features, tau_features = self._features(concentration)

        gain_coef = np.stack(
            [
                ridge_least_squares(
                    gain_features,
                    np.log(np.maximum(amplitude[:, j], 1e-8)),
                    self.ridge,
                )
                for j in range(3)
            ]
        )
        tau_fast_coef = np.stack(
            [
                ridge_least_squares(
                    tau_features,
                    np.log(np.maximum(tau0[:, j], 1e-8) / 2.0),
                    self.ridge,
                )
                for j in range(3)
            ]
        )
        delta_coef = np.stack(
            [
                ridge_least_squares(
                    tau_features,
                    np.log(np.maximum(1.5 * tau0[:, j], 1e-8)),
                    self.ridge,
                )
                for j in range(3)
            ]
        )
        mix_coef = np.zeros((3, tau_features.shape[1]))
        delay = np.median(delay0, axis=0)

        n_gain = 3 * gain_coef.shape[1]
        n_tau = 3 * tau_fast_coef.shape[1]
        n_delta = 3 * delta_coef.shape[1]
        n_mix = 3 * mix_coef.shape[1]
        delay_slice = slice(n_gain + n_tau + n_delta + n_mix, None)
        initial = np.r_[
            gain_coef.ravel(),
            tau_fast_coef.ravel(),
            delta_coef.ravel(),
            mix_coef.ravel(),
            delay,
        ]
        lower = np.full(len(initial), -np.inf)
        upper = np.full(len(initial), np.inf)
        lower[delay_slice] = DELAY_BOUNDS[0]
        upper[delay_slice] = DELAY_BOUNDS[1]

        penalised = np.ones(len(initial))
        for start, width in (
            (0, gain_coef.shape[1]),
            (n_gain, tau_fast_coef.shape[1]),
            (n_gain + n_tau, delta_coef.shape[1]),
            (n_gain + n_tau + n_delta, mix_coef.shape[1]),
        ):
            for j in range(3):
                penalised[start + j * width] = 0.0
        penalised[delay_slice] = 0.0

        scales = np.mean(relative**2, axis=(0, 1))
        weights = 1.0 / scales
        weights /= weights.mean()
        sqrt_weights = np.sqrt(weights)

        def predict(params: np.ndarray) -> np.ndarray:
            return self._predict_with(
                params[:n_gain].reshape(3, -1),
                params[n_gain : n_gain + n_tau].reshape(3, -1),
                params[n_gain + n_tau : n_gain + n_tau + n_delta].reshape(3, -1),
                params[n_gain + n_tau + n_delta : n_gain + n_tau + n_delta + n_mix].reshape(3, -1),
                params[delay_slice],
                concentration,
                time,
            )

        def residual(params: np.ndarray) -> np.ndarray:
            error = sqrt_weights[None, None, :] * (predict(params) - relative)
            return np.r_[error.ravel(), np.sqrt(self.ridge) * penalised * params]

        result = least_squares(
            residual,
            x0=initial,
            bounds=(lower, upper),
            max_nfev=self.max_nfev,
            x_scale="jac",
        )
        self.gain_coef = result.x[:n_gain].reshape(3, -1)
        self.tau_fast_coef = result.x[n_gain : n_gain + n_tau].reshape(3, -1)
        self.delta_coef = result.x[n_gain + n_tau : n_gain + n_tau + n_delta].reshape(3, -1)
        self.mix_coef = result.x[
            n_gain + n_tau + n_delta : n_gain + n_tau + n_delta + n_mix
        ].reshape(3, -1)
        self.delay = result.x[delay_slice]
        self.nfev_ = int(result.nfev)
        self.cost_ = float(result.cost)
        return self

    def _predict(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        if any(
            part is None
            for part in (
                self.gain_coef,
                self.tau_fast_coef,
                self.delta_coef,
                self.mix_coef,
                self.delay,
            )
        ):
            raise RuntimeError("模型尚未拟合")
        return self._predict_with(
            self.gain_coef,
            self.tau_fast_coef,
            self.delta_coef,
            self.mix_coef,
            self.delay,
            concentration,
            time,
        )

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        """预测相对响应，形状 (N,T,3)；NO2 通道为负。"""
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        return self._predict(concentration, np.asarray(time, dtype=float))

    def predict_response(
        self,
        concentration: np.ndarray,
        time: np.ndarray,
        initial: np.ndarray,
    ) -> np.ndarray:
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        initial = np.asarray(initial, dtype=float)
        if initial.ndim == 1:
            initial = initial[None, :]
        return initial[:, None, :] * (1.0 + self._predict(concentration, np.asarray(time, dtype=float)))

    def gain(self, concentration: np.ndarray) -> np.ndarray:
        if self.gain_coef is None:
            raise RuntimeError("模型尚未拟合")
        features, _ = self._features(np.atleast_2d(concentration))
        return np.exp(features @ self.gain_coef.T)

    def time_constants(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 (tau_fast, tau_slow, mix)，形状均为 (N,3)。"""
        if any(part is None for part in (self.tau_fast_coef, self.delta_coef, self.mix_coef)):
            raise RuntimeError("模型尚未拟合")
        _, features = self._features(np.atleast_2d(concentration))
        tau_fast = np.exp(features @ self.tau_fast_coef.T)
        tau_slow = tau_fast + np.exp(features @ self.delta_coef.T)
        mix = expit(features @ self.mix_coef.T)
        return tau_fast, tau_slow, mix

    def to_dict(self) -> dict:
        if any(
            part is None
            for part in (
                self.gain_coef,
                self.tau_fast_coef,
                self.delta_coef,
                self.mix_coef,
                self.delay,
            )
        ):
            raise RuntimeError("模型尚未拟合")
        return {
            "kind": "double_exponential",
            "reference": self.reference.tolist(),
            "signs": self.signs.tolist(),
            "gain_degree": self.gain_degree,
            "tau_degree": self.tau_degree,
            "ridge": self.ridge,
            "gain_coef": self.gain_coef.tolist(),
            "tau_fast_coef": self.tau_fast_coef.tolist(),
            "delta_coef": self.delta_coef.tolist(),
            "mix_coef": self.mix_coef.tolist(),
            "delay": self.delay.tolist(),
            "nfev": self.nfev_,
            "cost": self.cost_,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict) -> DoubleExponentialArrayModel:
        if payload.get("kind") != "double_exponential":
            raise ValueError("不是双指数骨架的 JSON")
        model = cls(
            reference=np.array(payload["reference"]),
            signs=np.array(payload["signs"]),
            gain_degree=payload["gain_degree"],
            tau_degree=payload["tau_degree"],
            ridge=payload["ridge"],
        )
        model.gain_coef = np.array(payload["gain_coef"])
        model.tau_fast_coef = np.array(payload["tau_fast_coef"])
        model.delta_coef = np.array(payload["delta_coef"])
        model.mix_coef = np.array(payload["mix_coef"])
        model.delay = np.array(payload["delay"])
        model.nfev_ = payload.get("nfev")
        model.cost_ = payload.get("cost")
        return model

    @classmethod
    def load(cls, path: Path | str) -> DoubleExponentialArrayModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
