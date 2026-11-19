"""双指数骨架 + 三路独立 MLP 残差。

每个气体通道各有一个独立的 `7 -> 64 -> 64 -> 1` 残差网络，三路之间不共享参数。
训练用 PyTorch，默认在 CUDA/ROCm GPU 上做全批 Adam；训练结束后把权重拷回 NumPy，
推理和下游 MAP 估计都在 CPU 上完成，避免几百次小规模前向的 GPU 传输开销。

输入仍是三个对数浓度、归一化时间以及骨架在当前点的三路预测，共 7 维。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import REFERENCE_CONCENTRATION, RESPONSE_SIGNS, Experiment
from .double_exponential import DoubleExponentialArrayModel

HIDDEN_LAYERS = (64, 64)


class PerGasMLPDoubleExponentialModel:
    """双指数骨架加三路相互独立的 MLP 残差。"""

    def __init__(
        self,
        reference: np.ndarray = REFERENCE_CONCENTRATION,
        signs: np.ndarray = RESPONSE_SIGNS,
        hidden_layers: tuple[int, ...] = HIDDEN_LAYERS,
        learning_rate_init: float = 2e-3,
        weight_decay: float = 1e-4,
        epochs: int = 600,
        batch_size: int = 0,
        device: str | None = None,
        random_state: int = 0,
    ) -> None:
        self.reference = np.asarray(reference, dtype=float)
        self.signs = np.asarray(signs, dtype=float)
        self.hidden_layers = hidden_layers
        self.learning_rate_init = learning_rate_init
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.name = "mlp_per_gas"
        self.skeleton: DoubleExponentialArrayModel | None = None
        self.time_scale: float | None = None
        self.x_mean: np.ndarray | None = None
        self.x_scale: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_scale: np.ndarray | None = None
        self.weights: list[list[np.ndarray]] = []
        self.biases: list[list[np.ndarray]] = []
        self.n_iter_: int | None = None
        self.device_: str | None = None

    def _inputs(
        self,
        concentration: np.ndarray,
        time: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        if self.time_scale is None:
            raise RuntimeError("模型尚未拟合")
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        log_c = np.log(concentration / self.reference)
        n, t = len(concentration), len(time)
        return np.column_stack(
            [
                np.repeat(log_c[:, 0], t),
                np.repeat(log_c[:, 1], t),
                np.repeat(log_c[:, 2], t),
                np.tile(np.asarray(time, dtype=float) / self.time_scale, n),
                np.asarray(baseline, dtype=float).reshape(n * t, 3),
            ]
        )

    def _network(self, x: np.ndarray, gas: int) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("模型尚未拟合")
        hidden = (x - self.x_mean) / self.x_scale
        weights = self.weights[gas]
        biases = self.biases[gas]
        for matrix, bias in zip(weights[:-1], biases[:-1]):
            hidden = np.tanh(hidden @ matrix + bias)
        return hidden @ weights[-1] + biases[-1]

    def fit(
        self,
        experiments: list[Experiment],
        skeleton: DoubleExponentialArrayModel | None = None,
        shape_init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> PerGasMLPDoubleExponentialModel:
        """拟合骨架与三路独立残差网络；可传入已拟合的骨架。"""
        import torch
        from torch import nn

        # 网络很小，GPU 才是主要算力；限制 CPU 线程可避免 OpenMP 自旋把 CPU 占满。
        torch.set_num_threads(1)
        if skeleton is None:
            skeleton = DoubleExponentialArrayModel(
                reference=self.reference,
                signs=self.signs,
            ).fit(experiments, shape_init=shape_init)
        self.skeleton = skeleton
        time = experiments[0].time
        concentration = np.stack([experiment.concentration for experiment in experiments])
        relative = np.stack([experiment.relative for experiment in experiments])
        baseline = skeleton.predict_relative(concentration, time)
        residual = relative - baseline
        self.time_scale = float(np.asarray(time, dtype=float)[-1])

        x = self._inputs(concentration, time, baseline)
        y = residual.reshape(-1, 3)
        self.x_mean = x.mean(axis=0)
        self.x_scale = x.std(axis=0) + 1e-8
        self.y_mean = y.mean(axis=0)
        self.y_scale = y.std(axis=0) + 1e-8
        x_scaled = ((x - self.x_mean) / self.x_scale).astype(np.float32)
        y_scaled = ((y - self.y_mean) / self.y_scale).astype(np.float32)

        if self.device is None:
            self.device_ = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device_ = self.device
        if self.device_ == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求了 CUDA/ROCm 设备，但 torch.cuda.is_available() 为 False")
        device = torch.device(self.device_)

        torch.manual_seed(self.random_state)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)
        layers = []
        for _ in range(3):
            modules: list[nn.Module] = []
            width = x_scaled.shape[1]
            for hidden in self.hidden_layers:
                modules.extend([nn.Linear(width, hidden), nn.Tanh()])
                width = hidden
            modules.append(nn.Linear(width, 1))
            layers.append(nn.Sequential(*modules).to(device))
        optimiser = torch.optim.Adam(
            [parameter for layer in layers for parameter in layer.parameters()],
            lr=self.learning_rate_init,
            weight_decay=self.weight_decay,
        )

        x_tensor = torch.as_tensor(x_scaled, device=device)
        y_tensor = torch.as_tensor(y_scaled, device=device)
        n = len(x_scaled)
        batch = n if self.batch_size <= 0 else min(self.batch_size, n)
        for _ in range(self.epochs):
            order = torch.randperm(n, device=device)
            for start in range(0, n, batch):
                index = order[start : start + batch]
                prediction = torch.cat([layer(x_tensor[index]) for layer in layers], dim=1)
                loss = ((prediction - y_tensor[index]) ** 2).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

        self.weights = []
        self.biases = []
        for layer in layers:
            weights = []
            biases = []
            for module in layer:
                if isinstance(module, nn.Linear):
                    weights.append(module.weight.detach().cpu().numpy().T.copy())
                    biases.append(module.bias.detach().cpu().numpy().copy())
            self.weights.append(weights)
            self.biases.append(biases)
        self.n_iter_ = self.epochs
        if device.type == "cuda":
            del x_tensor, y_tensor, layers, optimiser
            torch.cuda.empty_cache()
        return self

    def _residual(
        self,
        concentration: np.ndarray,
        time: np.ndarray,
        baseline: np.ndarray,
    ) -> np.ndarray:
        x = self._inputs(concentration, time, baseline)
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        n, t = len(concentration), len(time)
        residual = np.column_stack(
            [self._network(x, gas) for gas in range(3)]
        )
        return (residual * self.y_scale + self.y_mean).reshape(n, t, 3)

    def predict_residual(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        if self.skeleton is None:
            raise RuntimeError("模型尚未拟合")
        time = np.asarray(time, dtype=float)
        baseline = self.skeleton.predict_relative(concentration, time)
        return self._residual(concentration, time, baseline)

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        if self.skeleton is None:
            raise RuntimeError("模型尚未拟合")
        time = np.asarray(time, dtype=float)
        baseline = self.skeleton.predict_relative(concentration, time)
        return baseline + self._residual(concentration, time, baseline)

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
        return initial[:, None, :] * (1.0 + self.predict_relative(concentration, time))

    def to_dict(self) -> dict:
        if self.skeleton is None or not self.weights:
            raise RuntimeError("模型尚未拟合")
        return {
            "kind": "mlp_per_gas_double_exponential",
            "reference": self.reference.tolist(),
            "signs": self.signs.tolist(),
            "hidden_layers": list(self.hidden_layers),
            "learning_rate_init": self.learning_rate_init,
            "weight_decay": self.weight_decay,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "device": self.device_,
            "random_state": self.random_state,
            "time_scale": self.time_scale,
            "x_mean": self.x_mean.tolist(),
            "x_scale": self.x_scale.tolist(),
            "y_mean": self.y_mean.tolist(),
            "y_scale": self.y_scale.tolist(),
            "weights": [[matrix.tolist() for matrix in gas] for gas in self.weights],
            "biases": [[bias.tolist() for bias in gas] for gas in self.biases],
            "n_iter": self.n_iter_,
            "skeleton": self.skeleton.to_dict(),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_dict(cls, payload: dict) -> PerGasMLPDoubleExponentialModel:
        if payload.get("kind") != "mlp_per_gas_double_exponential":
            raise ValueError("不是三路独立 MLP 残差的 JSON")
        model = cls(
            reference=np.array(payload["reference"]),
            signs=np.array(payload["signs"]),
            hidden_layers=tuple(payload["hidden_layers"]),
            learning_rate_init=payload["learning_rate_init"],
            weight_decay=payload["weight_decay"],
            epochs=payload["epochs"],
            batch_size=payload["batch_size"],
            device=payload["device"],
            random_state=payload["random_state"],
        )
        model.skeleton = DoubleExponentialArrayModel.from_dict(payload["skeleton"])
        model.time_scale = float(payload["time_scale"])
        model.x_mean = np.array(payload["x_mean"])
        model.x_scale = np.array(payload["x_scale"])
        model.y_mean = np.array(payload["y_mean"])
        model.y_scale = np.array(payload["y_scale"])
        model.weights = [[np.array(matrix) for matrix in gas] for gas in payload["weights"]]
        model.biases = [[np.array(bias) for bias in gas] for gas in payload["biases"]]
        model.n_iter_ = payload.get("n_iter")
        model.device_ = payload.get("device")
        return model

    @classmethod
    def load(cls, path: Path | str) -> PerGasMLPDoubleExponentialModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))