"""传感器阵列动态响应建模（MLP 残差 + 双指数骨架）。"""
from .core.data import GASES, REFERENCE_CONCENTRATION, RESPONSE_SIGNS, Experiment, load_experiments
from .core.double_exponential import DoubleExponentialArrayModel
from .core.mlp_per_gas import PerGasMLPDoubleExponentialModel
from .core.shapes import fit_experiment_shapes

__all__ = [
    "GASES",
    "REFERENCE_CONCENTRATION",
    "RESPONSE_SIGNS",
    "Experiment",
    "DoubleExponentialArrayModel",
    "PerGasMLPDoubleExponentialModel",
    "fit_experiment_shapes",
    "load_experiments",
]