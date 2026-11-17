"""三路传感器阵列的一阶惯性灰盒响应模型。

对每路传感器 j，把恒定浓度下的阶跃响应写成

    y_j(t) = s_j * A_j(c) * (1 - exp(-max(t - d_j, 0) / tau_j(c)))

其中 y_j(t)=(R_j(t)-R_j(0))/R_j(0) 是相对响应，s_j 是由数据固定的符号
（H2/NH3 为正、NO2 为负），A_j(c)>0 是稳态相对幅度，tau_j(c)>0 是时间常数，
d_j 是纯延迟。A_j 与 tau_j 用对数浓度的多项式参数化：

    log A_j(c) = 关于 u=log(c/c_ref) 的 gain_degree 次多项式
    log tau_j(c) = 关于 u 的 tau_degree 次多项式

选择"一阶惯性 + 纯延迟"的依据见 README §3：二阶模型拟合更差，stretched 指数只把
逐实验拟合误差从 1.31% 降到 1.10%，不足以抵消参数与可解释性的代价。
"""
from __future__ import annotations

import json
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from .data import (
    REFERENCE_CONCENTRATION,
    RESPONSE_SIGNS,
    Experiment,
)

_DELAY_BOUNDS = (0.0, 2.0)
_TAU_BOUNDS = (0.1, 120.0)
_SHAPE_STARTS = (0.0, 0.3)


def _monomial_features(u: np.ndarray, degree: int) -> np.ndarray:
    """构造 [1, 一次项, ..., degree 次项] 的多项式特征，u 形状为 (N, k)。"""
    columns = [np.ones(len(u))]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(u.shape[1]), order):
            columns.append(np.prod(u[:, indices], axis=1))
    return np.column_stack(columns)


def _ridge_least_squares(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    """带岭惩罚的最小二乘；截距与常数项不惩罚。"""
    penalty = np.sqrt(ridge) * np.eye(x.shape[1])
    penalty[0, 0] = 0.0
    return np.linalg.lstsq(
        np.vstack([x, penalty]),
        np.r_[y, np.zeros(x.shape[1])],
        rcond=None,
    )[0]


def _shape_curve(time: np.ndarray, amplitude: float, tau: float, delay: float) -> np.ndarray:
    return amplitude * (1.0 - np.exp(-np.clip(time - delay, 0.0, None) / tau))


def _fit_one_shape(time: np.ndarray, target: np.ndarray) -> tuple[float, float, float]:
    """拟合单条有符号相对响应，返回 (A, tau, delay)。"""
    best = None
    for start_delay in _SHAPE_STARTS:
        result = least_squares(
            lambda p: _shape_curve(time, p[0], p[1], p[2]) - target,
            x0=[max(float(target[-1]), 1e-4), 10.0, start_delay],
            bounds=([0.0, _TAU_BOUNDS[0], _DELAY_BOUNDS[0]], [10.0, _TAU_BOUNDS[1], _DELAY_BOUNDS[1]]),
            max_nfev=2000,
        )
        if best is None or result.cost < best.cost:
            best = result
    assert best is not None
    return float(best.x[0]), float(best.x[1]), float(best.x[2])


def fit_experiment_shapes(experiments: list[Experiment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """逐实验拟合 (A, tau, delay)，用作联合拟合的初值，也用于增益/时间常数评估。

    返回三个 (N,3) 数组。逐实验拟合彼此独立，因此在留一验证中可以只对训练
    实验预先计算一次，再按折取子集。
    """
    time = experiments[0].time
    shapes = np.empty((len(experiments), 3, 3))
    for i, experiment in enumerate(experiments):
        relative = experiment.relative
        for j in range(3):
            target = RESPONSE_SIGNS[j] * relative[:, j]
            shapes[i, j] = _fit_one_shape(time, target)
    return shapes[:, :, 0], shapes[:, :, 1], shapes[:, :, 2]


class FirstOrderArrayModel:
    """三路传感器的一阶惯性阵列模型。"""

    def __init__(
        self,
        reference: np.ndarray = REFERENCE_CONCENTRATION,
        signs: np.ndarray = RESPONSE_SIGNS,
        gain_degree: int = 2,
        tau_degree: int = 1,
        ridge: float = 1e-3,
    ) -> None:
        self.reference = np.asarray(reference, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.gain_degree = gain_degree
        self.tau_degree = tau_degree
        self.ridge = ridge
        self.gain_coef: np.ndarray | None = None
        self.tau_coef: np.ndarray | None = None
        self.delay: np.ndarray | None = None

    @property
    def n_gases(self) -> int:
        return len(self.reference)

    def _features(self, concentration: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u = np.log(np.asarray(concentration, dtype=float) / self.reference)
        return _monomial_features(u, self.gain_degree), _monomial_features(u, self.tau_degree)

    def _predict_with(
        self,
        gain_coef: np.ndarray,
        tau_coef: np.ndarray,
        delay: np.ndarray,
        concentration: np.ndarray,
        time: np.ndarray,
    ) -> np.ndarray:
        gain_features, tau_features = self._features(concentration)
        amplitude = np.exp(gain_features @ gain_coef.T)  # (N,3)
        tau = np.exp(tau_features @ tau_coef.T)  # (N,3)
        shape = 1.0 - np.exp(
            -np.clip(time[None, :, None] - delay[None, None, :], 0.0, None) / tau[:, None, :]
        )
        return self.signs[None, None, :] * amplitude[:, None, :] * shape

    def _predict(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        if self.gain_coef is None or self.tau_coef is None or self.delay is None:
            raise RuntimeError("模型尚未拟合")
        return self._predict_with(self.gain_coef, self.tau_coef, self.delay, concentration, time)

    def fit(
        self,
        experiments: list[Experiment],
        shape_init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        max_nfev: int = 400,
    ) -> FirstOrderArrayModel:
        """联合拟合全部实验。

        `shape_init` 为逐实验 (A, tau, delay)，仅用于初始化；留一验证时可传入
        预先算好的训练子集，避免每折重复拟合。目标函数按传感器归一化，避免
        幅度很小的 H2 通道被 NH3 通道淹没。
        """
        time = experiments[0].time
        relative = np.stack([experiment.relative for experiment in experiments])
        concentration = np.stack([experiment.concentration for experiment in experiments])
        n_experiments, n_time, n_sensors = relative.shape

        scales = np.mean(relative**2, axis=(0, 1))
        weights = 1.0 / scales
        weights /= weights.mean()

        if shape_init is None:
            shape_init = fit_experiment_shapes(experiments)
        amplitude, tau0, delay0 = shape_init
        gain_features, tau_features = self._features(concentration)
        gain_coef = np.stack(
            [
                _ridge_least_squares(gain_features, np.log(np.maximum(amplitude[:, j], 1e-8)), self.ridge)
                for j in range(n_sensors)
            ]
        )
        tau_coef = np.stack(
            [
                _ridge_least_squares(tau_features, np.log(tau0[:, j]), self.ridge)
                for j in range(n_sensors)
            ]
        )
        delay = np.median(delay0, axis=0)

        initial = np.r_[gain_coef.ravel(), tau_coef.ravel(), delay]
        n_gain = n_sensors * gain_coef.shape[1]
        n_tau = n_sensors * tau_coef.shape[1]
        lower = np.r_[np.full(n_gain + n_tau, -np.inf), np.full(n_sensors, _DELAY_BOUNDS[0])]
        upper = np.r_[np.full(n_gain + n_tau, np.inf), np.full(n_sensors, _DELAY_BOUNDS[1])]
        penalised = np.ones_like(initial)
        penalised[:n_gain] = np.tile(
            np.r_[0.0, np.ones(gain_coef.shape[1] - 1)], n_sensors
        )
        penalised[n_gain : n_gain + n_tau] = np.tile(
            np.r_[0.0, np.ones(tau_coef.shape[1] - 1)], n_sensors
        )
        penalised[n_gain + n_tau :] = 0.0

        def predict(params: np.ndarray) -> np.ndarray:
            return self._predict_with(
                params[:n_gain].reshape(n_sensors, -1),
                params[n_gain : n_gain + n_tau].reshape(n_sensors, -1),
                params[n_gain + n_tau :],
                concentration,
                time,
            )

        def residual(params: np.ndarray) -> np.ndarray:
            error = np.sqrt(weights)[None, None, :] * (predict(params) - relative)
            return np.r_[error.ravel(), np.sqrt(self.ridge) * penalised * params]

        result = least_squares(
            residual,
            x0=initial,
            bounds=(lower, upper),
            max_nfev=max_nfev,
            x_scale="jac",
        )
        self.gain_coef = result.x[:n_gain].reshape(n_sensors, -1)
        self.tau_coef = result.x[n_gain : n_gain + n_tau].reshape(n_sensors, -1)
        self.delay = result.x[n_gain + n_tau :]
        self.fit_cost_ = float(result.cost)
        return self

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        """预测未加符号的相对响应，形状 (N, T, 3)；NO2 通道为负。"""
        concentration = np.atleast_2d(concentration)
        return self._predict(concentration, np.asarray(time, dtype=float))

    def predict_response(
        self,
        concentration: np.ndarray,
        time: np.ndarray,
        initial: np.ndarray,
    ) -> np.ndarray:
        """由基线 R(0) 还原绝对响应 R(t)=R(0)*(1+y(t))。

        `initial` 形状为 (3,) 或 (N,3)。基线依赖实验条件与漂移，不能由浓度
        唯一确定，因此作为显式输入。
        """
        concentration = np.atleast_2d(concentration)
        initial = np.asarray(initial, dtype=float)
        if initial.ndim == 1:
            initial = initial[None, :]
        relative = self._predict(concentration, np.asarray(time, dtype=float))
        return initial[:, None, :] * (1.0 + relative)

    def gain(self, concentration: np.ndarray) -> np.ndarray:
        """稳态相对幅度 A_j(c)>0，形状 (N,3)。"""
        if self.gain_coef is None:
            raise RuntimeError("模型尚未拟合")
        features, _ = self._features(np.atleast_2d(concentration))
        return np.exp(features @ self.gain_coef.T)

    def tau(self, concentration: np.ndarray) -> np.ndarray:
        """时间常数 tau_j(c)>0，形状 (N,3)。"""
        if self.tau_coef is None:
            raise RuntimeError("模型尚未拟合")
        _, features = self._features(np.atleast_2d(concentration))
        return np.exp(features @ self.tau_coef.T)

    def to_dict(self) -> dict:
        if self.gain_coef is None:
            raise RuntimeError("模型尚未拟合")
        return {
            "reference": self.reference.tolist(),
            "signs": self.signs.tolist(),
            "gain_degree": self.gain_degree,
            "tau_degree": self.tau_degree,
            "ridge": self.ridge,
            "gain_coef": self.gain_coef.tolist(),
            "tau_coef": self.tau_coef.tolist(),
            "delay": self.delay.tolist(),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict) -> FirstOrderArrayModel:
        model = cls(
            reference=np.array(payload["reference"]),
            signs=np.array(payload["signs"]),
            gain_degree=payload["gain_degree"],
            tau_degree=payload["tau_degree"],
            ridge=payload["ridge"],
        )
        model.gain_coef = np.array(payload["gain_coef"])
        model.tau_coef = np.array(payload["tau_coef"])
        model.delay = np.array(payload["delay"])
        return model

    @classmethod
    def load(cls, path: Path | str) -> FirstOrderArrayModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
