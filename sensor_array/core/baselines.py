"""与灰盒模型对比的基线。

- MeanCurveModel：忽略浓度，预测训练集平均曲线。
- TwoStageModel：先用逐实验一阶拟合得到 (A,tau)，再回归浓度依赖；与联合
  拟合的灰盒模型共用同一函数形式，用来检验"联合拟合"是否值得。
- DirectRidgeModel：逐时间点的浓度二次响应面：每个 (t, 传感器) 输出都用浓度
  二次特征做岭回归，是不利用动态结构的黑盒基线。
- TCNModel：条件时间卷积网络，输入为常数浓度与时间特征，输出三路响应。
  由于一次实验内浓度恒定，时间卷积没有真实的时序输入，它实际学习的是
  (c,t) 的柔性黑盒函数；用它检验神经网络在本数据规模下是否优于灰盒。
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from .data import REFERENCE_CONCENTRATION, Experiment
from .grey_box import FirstOrderArrayModel, _monomial_features, _ridge_least_squares, fit_experiment_shapes


class MeanCurveModel:
    """忽略浓度，输出训练集平均相对响应。"""

    def fit(self, experiments: list[Experiment], shape_init=None) -> MeanCurveModel:
        self.mean_curve_ = np.mean([item.relative for item in experiments], axis=0)
        return self

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        return np.tile(self.mean_curve_[None, :, :], (np.atleast_2d(concentration).shape[0], 1, 1))


class TwoStageModel:
    """逐实验拟合形状参数后，再回归浓度依赖的两阶段模型。"""

    def __init__(self, gain_degree: int = 2, tau_degree: int = 1, ridge: float = 1e-3) -> None:
        self.model = FirstOrderArrayModel(gain_degree=gain_degree, tau_degree=tau_degree, ridge=ridge)

    def fit(self, experiments: list[Experiment], shape_init=None) -> TwoStageModel:
        if shape_init is None:
            shape_init = fit_experiment_shapes(experiments)
        amplitude, tau, delay = shape_init
        concentration = np.stack([item.concentration for item in experiments])
        u = np.log(concentration / self.model.reference)
        gain_features = _monomial_features(u, self.model.gain_degree)
        tau_features = _monomial_features(u, self.model.tau_degree)
        n_sensors = amplitude.shape[1]
        self.model.gain_coef = np.stack(
            [
                _ridge_least_squares(gain_features, np.log(np.maximum(amplitude[:, j], 1e-8)), self.model.ridge)
                for j in range(n_sensors)
            ]
        )
        self.model.tau_coef = np.stack(
            [
                _ridge_least_squares(tau_features, np.log(tau[:, j]), self.model.ridge)
                for j in range(n_sensors)
            ]
        )
        self.model.delay = np.median(delay, axis=0)
        return self

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        return self.model.predict_relative(concentration, time)


class DirectRidgeModel:
    """逐时间点的浓度二次响应面（不用动态结构）。"""

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha

    def fit(self, experiments: list[Experiment], shape_init=None) -> DirectRidgeModel:
        concentration = np.stack([item.concentration for item in experiments])
        u = np.log(concentration / REFERENCE_CONCENTRATION)
        features = _monomial_features(u, 2)
        targets = np.stack([item.relative for item in experiments]).reshape(len(experiments), -1)
        self.regression_ = Ridge(alpha=self.alpha).fit(features, targets)
        return self

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        u = np.log(np.atleast_2d(concentration) / REFERENCE_CONCENTRATION)
        features = _monomial_features(u, 2)
        prediction = self.regression_.predict(features)
        return prediction.reshape(len(features), len(time), 3)


class _CausalBlock(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int) -> None:
        super().__init__()
        padding = dilation * (kernel - 1)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, padding=padding, dilation=dilation)
        self.activation = nn.GELU()
        self.residual = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        h = self.activation(self.conv1(x)[..., :length])
        h = self.activation(self.conv2(h)[..., :length])
        return self.activation(h + self.residual(x))


class TCNModel:
    """条件 TCN：输入三路 u、t/t_end 与 (t/t_end)^2，输出三路相对响应。"""

    def __init__(
        self,
        hidden: int = 24,
        levels: int = 3,
        kernel: int = 3,
        epochs: int = 120,
        learning_rate: float = 1e-3,
        seed: int = 0,
    ) -> None:
        self.hidden = hidden
        self.levels = levels
        self.kernel = kernel
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.seed = seed

    @staticmethod
    def _context(u: np.ndarray, time: np.ndarray) -> np.ndarray:
        scaled = time / time[-1]
        channels = [np.repeat(u[:, :, None], len(time), axis=2), np.broadcast_to(scaled, (len(u), 1, len(time))), np.broadcast_to(scaled**2, (len(u), 1, len(time)))]
        return np.concatenate(channels, axis=1)

    def _build(self) -> nn.Module:
        layers = [nn.Conv1d(5, self.hidden, 1)]
        for level in range(self.levels):
            layers.append(_CausalBlock(self.hidden, self.kernel, dilation=2**level))
        layers.append(nn.Conv1d(self.hidden, 3, 1))
        return nn.Sequential(*layers)

    def fit(self, experiments: list[Experiment], shape_init=None) -> TCNModel:
        torch.manual_seed(self.seed)
        self.network_ = self._build()
        time = experiments[0].time
        concentration = np.stack([item.concentration for item in experiments])
        u = np.log(concentration / REFERENCE_CONCENTRATION)
        context = self._context(u, time)
        inputs = torch.tensor(context, dtype=torch.float32)
        raw_targets = np.stack([item.relative for item in experiments]).transpose(0, 2, 1)
        self.scale_ = np.sqrt(np.mean(raw_targets**2, axis=(0, 2)))
        targets = torch.tensor(raw_targets / self.scale_[:, None], dtype=torch.float32)

        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        generator = torch.Generator().manual_seed(self.seed)
        batch_size = 8
        for _ in range(self.epochs):
            order = torch.randperm(len(inputs), generator=generator)
            for start in range(0, len(inputs), batch_size):
                index = order[start : start + batch_size]
                prediction = self.network_(inputs[index])
                loss = ((prediction - targets[index]) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return self

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        concentration = np.atleast_2d(concentration)
        u = np.log(concentration / REFERENCE_CONCENTRATION)
        context = self._context(u, time)
        with torch.no_grad():
            prediction = self.network_(torch.tensor(context, dtype=torch.float32))
        return (prediction.numpy() * self.scale_[None, :, None]).transpose(0, 2, 1)
