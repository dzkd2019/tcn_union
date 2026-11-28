"""传感器阵列动态响应建模（MLP 残差 + 双指数骨架）。"""
from .core.data import GASES, REFERENCE_CONCENTRATION, RESPONSE_SIGNS, Experiment, load_experiments
from .core.double_exponential import DoubleExponentialArrayModel
from .core.mlp_per_gas import PerGasMLPDoubleExponentialModel
from .core.residual import PerGasMLPResidualModel
from .core.shapes import fit_experiment_shapes
from .core.skeletons import (
    DoubleSecondOrderArrayModel,
    FirstOrderArrayModel,
    SecondOrderArrayModel,
    skeleton_from_dict,
)

__all__ = [
    "GASES",
    "REFERENCE_CONCENTRATION",
    "RESPONSE_SIGNS",
    "Experiment",
    "DoubleExponentialArrayModel",
    "DoubleSecondOrderArrayModel",
    "FirstOrderArrayModel",
    "PerGasMLPDoubleExponentialModel",
    "PerGasMLPResidualModel",
    "SecondOrderArrayModel",
    "fit_experiment_shapes",
    "load_experiments",
    "skeleton_from_dict",
]