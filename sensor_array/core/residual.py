"""任意骨架 + 三路独立 MLP 残差。

这是 PerGasMLPDoubleExponentialModel 的骨架无关版本：网络结构、输入特征、
训练协议都沿用现有实现，只要求调用方传入一个已经拟合好的骨架。
"""
from __future__ import annotations

from .mlp_per_gas import PerGasMLPDoubleExponentialModel


class PerGasMLPResidualModel(PerGasMLPDoubleExponentialModel):
    """任意响应骨架加三路相互独立的 MLP 残差。"""

    kind = "per_gas_mlp_residual"
    name = "per_gas_residual"

    def fit(self, experiments, skeleton=None, shape_init=None):
        if skeleton is None:
            raise ValueError("PerGasMLPResidualModel 需要传入已拟合的骨架")
        return super().fit(experiments, skeleton=skeleton, shape_init=shape_init)
