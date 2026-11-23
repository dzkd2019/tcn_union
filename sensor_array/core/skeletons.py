"""一阶、二阶与两个二阶并联的响应骨架。

三个骨架共用同一套浓度参数化与联合拟合流程：

    y_j(t) = s_j A_j(c) g_j(t, c)

其中 A_j 为正幅值，g_j 为归一化阶跃形状，g_j(0)=0、g_j 随时间趋于 1。
一阶与二阶各有一个时间常数；两个二阶是两个二阶系统的并联，用混合权重
f 组合快慢两个形状。延迟 d_j 为三路各自的参数，范围 [0,2] s。

这些类只用于骨架对比与正模型数据生成，不替代当前主模型
DoubleExponentialArrayModel。
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


def _ridge_columns(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """对 y 的每一列独立做岭回归，返回 (列数, 特征数)。"""
    return np.stack(
        [ridge_least_squares(x, y[:, j], ridge) for j in range(y.shape[1])]
    )


class _SkeletonArrayModel:
    """三个骨架的公共拟合、预测与序列化。"""

    kind = ""
    name = ""
    time_block_names: tuple[str, ...] = ()

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
        self.time_coefs: list[np.ndarray] = []
        self.delay: np.ndarray | None = None
        self.nfev_: int | None = None
        self.cost_: float | None = None

    def _features(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.log(np.asarray(concentration, dtype=float) / self.reference)
        return monomial_features(u, self.gain_degree), monomial_features(u, self.tau_degree)

    def _initial_time_coefs(
        self, tau: np.ndarray, features: np.ndarray
    ) -> list[np.ndarray]:
        """由逐实验一阶时间常数给出各时间块的岭回归初值。"""
        raise NotImplementedError

    def _shape(self, t: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
        """t 形状 (1,T,3)，parameters 每个形状 (N,3)，返回 (N,T,3)。"""
        raise NotImplementedError

    def _predict_with(
        self,
        gain_coef: np.ndarray,
        time_coefs: list[np.ndarray],
        delay: np.ndarray,
        concentration: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        gain_features, tau_features = self._features(concentration)
        amplitude = np.exp(gain_features @ gain_coef.T)
        parameters = [tau_features @ block.T for block in time_coefs]
        t = np.clip(time[None, :, None] - delay[None, None, :], 0.0, None)
        shape = self._shape(t, parameters)
        return self.signs[None, None, :] * amplitude[:, None, :] * shape

    def fit(
        self,
        experiments: list[Experiment],
        shape_init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> _SkeletonArrayModel:
        """联合拟合一个骨架；shape_init 为逐实验一阶形状初值。"""
        time = experiments[0].time
        relative = np.stack([experiment.relative for experiment in experiments])
        concentration = np.stack([experiment.concentration for experiment in experiments])
        if shape_init is None:
            shape_init = fit_experiment_shapes(experiments)
        amplitude, tau0, delay0 = shape_init
        gain_features, tau_features = self._features(concentration)

        gain_coef = _ridge_columns(
            gain_features,
            np.log(np.maximum(amplitude, 1e-8)),
            self.ridge,
        )
        time_coefs = self._initial_time_coefs(tau0, tau_features)
        delay = np.median(delay0, axis=0)

        n_gain = 3 * gain_features.shape[1]
        n_time = 3 * tau_features.shape[1]
        initial = np.concatenate(
            [gain_coef.ravel(), *[block.ravel() for block in time_coefs], delay]
        )
        delay_slice = slice(len(initial) - 3, None)
        lower = np.full(len(initial), -np.inf)
        upper = np.full(len(initial), np.inf)
        lower[delay_slice] = DELAY_BOUNDS[0]
        upper[delay_slice] = DELAY_BOUNDS[1]

        penalised = np.ones(len(initial))
        start = 0
        for block in (gain_coef, *time_coefs):
            for j in range(3):
                penalised[start + j * block.shape[1]] = 0.0
            start += 3 * block.shape[1]
        penalised[delay_slice] = 0.0

        scales = np.mean(relative**2, axis=(0, 1))
        weights = 1.0 / scales
        weights /= weights.mean()
        sqrt_weights = np.sqrt(weights)

        def predict(params: np.ndarray) -> np.ndarray:
            gain = params[:n_gain].reshape(3, -1)
            offset = n_gain
            blocks = []
            for _ in time_coefs:
                blocks.append(params[offset : offset + n_time].reshape(3, -1))
                offset += n_time
            return self._predict_with(
                gain,
                blocks,
                params[delay_slice],
                concentration,
                time,
            )

        def residual(params: np.ndarray) -> np.ndarray:
            error = sqrt_weights[None, None, :] * (predict(params) - relative)
            return np.concatenate(
                [error.ravel(), np.sqrt(self.ridge) * penalised * params]
            )

        result = least_squares(
            residual,
            x0=initial,
            bounds=(lower, upper),
            max_nfev=self.max_nfev,
            x_scale="jac",
        )
        self.gain_coef = result.x[:n_gain].reshape(3, -1)
        offset = n_gain
        self.time_coefs = []
        for _ in time_coefs:
            self.time_coefs.append(result.x[offset : offset + n_time].reshape(3, -1))
            offset += n_time
        self.delay = result.x[delay_slice]
        self.nfev_ = int(result.nfev)
        self.cost_ = float(result.cost)
        return self

    def _predict(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        if self.gain_coef is None or not self.time_coefs or self.delay is None:
            raise RuntimeError("模型尚未拟合")
        return self._predict_with(
            self.gain_coef,
            self.time_coefs,
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
        return initial[:, None, :] * (
            1.0 + self._predict(concentration, np.asarray(time, dtype=float))
        )

    def gain(self, concentration: np.ndarray) -> np.ndarray:
        if self.gain_coef is None:
            raise RuntimeError("模型尚未拟合")
        features, _ = self._features(np.atleast_2d(concentration))
        return np.exp(features @ self.gain_coef.T)

    def to_dict(self) -> dict:
        if self.gain_coef is None or not self.time_coefs or self.delay is None:
            raise RuntimeError("模型尚未拟合")
        return {
            "kind": self.kind,
            "reference": self.reference.tolist(),
            "signs": self.signs.tolist(),
            "gain_degree": self.gain_degree,
            "tau_degree": self.tau_degree,
            "ridge": self.ridge,
            "gain_coef": self.gain_coef.tolist(),
            "time_coefs": [block.tolist() for block in self.time_coefs],
            "delay": self.delay.tolist(),
            "nfev": self.nfev_,
            "cost": self.cost_,
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict) -> _SkeletonArrayModel:
        if payload.get("kind") != cls.kind:
            raise ValueError(f"不是 {cls.kind} 骨架的 JSON")
        model = cls(
            reference=np.array(payload["reference"]),
            signs=np.array(payload["signs"]),
            gain_degree=payload["gain_degree"],
            tau_degree=payload["tau_degree"],
            ridge=payload["ridge"],
        )
        model.gain_coef = np.array(payload["gain_coef"])
        model.time_coefs = [np.array(block) for block in payload["time_coefs"]]
        model.delay = np.array(payload["delay"])
        model.nfev_ = payload.get("nfev")
        model.cost_ = payload.get("cost")
        return model

    @classmethod
    def load(cls, path: Path | str) -> _SkeletonArrayModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class FirstOrderArrayModel(_SkeletonArrayModel):
    """一阶阶跃响应：g = 1 - exp(-(t-d)_+/tau)。"""

    kind = "first_order"
    name = "first"
    time_block_names = ("tau",)

    def _initial_time_coefs(
        self, tau: np.ndarray, features: np.ndarray
    ) -> list[np.ndarray]:
        return [
            _ridge_columns(features, np.log(np.maximum(tau, 1e-8)), self.ridge)
        ]

    def _shape(self, t: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
        tau = np.exp(np.clip(parameters[0][:, None, :], -20.0, 10.0))
        return 1.0 - np.exp(-t / tau)


class SecondOrderArrayModel(_SkeletonArrayModel):
    """临界阻尼二阶阶跃响应：g = 1 - (1+x) exp(-x)，x=(t-d)_+/tau。"""

    kind = "second_order"
    name = "second"
    time_block_names = ("tau",)

    def _initial_time_coefs(
        self, tau: np.ndarray, features: np.ndarray
    ) -> list[np.ndarray]:
        return [
            _ridge_columns(features, np.log(np.maximum(tau, 1e-8)), self.ridge)
        ]

    def _shape(self, t: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
        tau = np.exp(np.clip(parameters[0][:, None, :], -20.0, 10.0))
        x = t / tau
        return 1.0 - (1.0 + x) * np.exp(-x)


class DoubleSecondOrderArrayModel(_SkeletonArrayModel):
    """两个二阶系统并联：快 tau1、慢 tau2>tau1、混合权重 f。"""

    kind = "double_second_order"
    name = "second2"
    time_block_names = ("tau_fast", "delta", "mix")

    def _initial_time_coefs(
        self, tau: np.ndarray, features: np.ndarray
    ) -> list[np.ndarray]:
        tau_fast = np.maximum(0.5 * tau, 1e-8)
        tau_slow = np.maximum(2.0 * tau, 1e-8)
        delta = np.maximum(tau_slow - tau_fast, 1e-8)
        return [
            _ridge_columns(features, np.log(tau_fast), self.ridge),
            _ridge_columns(features, np.log(delta), self.ridge),
            np.zeros((3, features.shape[1])),
        ]

    def _shape(self, t: np.ndarray, parameters: list[np.ndarray]) -> np.ndarray:
        tau_fast = np.exp(np.clip(parameters[0][:, None, :], -20.0, 10.0))
        delta = np.exp(np.clip(parameters[1][:, None, :], -20.0, 10.0))
        tau_slow = tau_fast + delta
        mix = expit(parameters[2][:, None, :])
        fast = 1.0 - (1.0 + t / tau_fast) * np.exp(-t / tau_fast)
        slow = 1.0 - (1.0 + t / tau_slow) * np.exp(-t / tau_slow)
        return mix * fast + (1.0 - mix) * slow


_SKELETON_CLASSES = {
    "first_order": FirstOrderArrayModel,
    "second_order": SecondOrderArrayModel,
    "double_second_order": DoubleSecondOrderArrayModel,
}


def skeleton_from_dict(payload: dict) -> _SkeletonArrayModel:
    """按 JSON 的 kind 还原任意骨架，包括现有双指数骨架。"""
    kind = payload.get("kind")
    if kind == "double_exponential":
        from .double_exponential import DoubleExponentialArrayModel

        return DoubleExponentialArrayModel.from_dict(payload)
    model_class = _SKELETON_CLASSES.get(kind)
    if model_class is None:
        raise ValueError(f"未知骨架类型：{kind}")
    return model_class.from_dict(payload)
