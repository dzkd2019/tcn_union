"""因果卷积 TCN，用有限早期响应序列回归三个气体浓度。

所有卷积只在左侧补零，因此时间位置不会访问未来采样；模型在指定前缀的
最后一个时间点读取隐藏表示，输出 H2、NH3、NO2 的标准化浓度。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

# 固定隐藏通道宽度：与历史 16 宽 TCN 逐位兼容。
CHANNELS = 16


class CausalResidualBlock(nn.Module):
    """单个两层因果卷积残差块，隐藏通道宽度固定为 16。"""
    def __init__(self, in_channels: int, dilation: int) -> None:
        """初始化该块。

        参数：`in_channels` 是输入通道数；`dilation` 是两层卷积共同使用的
        膨胀率。每层核长为 5，左补零保持序列长度与因果性；残差支路在输入
        通道与 `channels` 不同时通过 1×1 卷积对齐维度。
        """
        super().__init__()
        self.kernel_size = 5
        self.dilation = dilation
        self.conv1 = nn.Conv1d(in_channels, CHANNELS, self.kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(CHANNELS, CHANNELS, self.kernel_size, dilation=dilation)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.projection = nn.Conv1d(in_channels, CHANNELS, kernel_size=1) if in_channels != CHANNELS else nn.Identity()

    def _causal_conv(self, x: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
        """对输入执行左侧补零的单向卷积。

        参数：`x` 形状为 `[batch,channels,time]`；`layer` 是配置好膨胀率的
        Conv1d。返回与输入时间长度相同的卷积结果。仅在左侧补
        `(kernel_size-1)*dilation` 个零，禁止使用未来时刻。
        """
        padding = (self.kernel_size - 1) * self.dilation
        return layer(F.pad(x, (padding, 0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """计算残差分支与两层卷积分支并相加。

        参数：`x` 为 `[batch,in_channels,time]` 的时序张量。
        返回：`[batch,16,time]` 特征。ReLU 和 Dropout 提供非线性及正则化，
        残差连接有助于优化较深的因果卷积堆叠。
        """
        residual = self.projection(x)
        hidden = self.dropout1(F.relu(self._causal_conv(x, self.conv1)))
        hidden = self.dropout2(F.relu(self._causal_conv(hidden, self.conv2)))
        return F.relu(residual + hidden)


class GasTCN(nn.Module):
    """三块因果 TCN，将固定传感器前缀映射到三气体浓度。

    隐藏通道宽度固定为 16，卷积核长 5，膨胀率依次为 1、2、4；每块含两层
    卷积。总感受野为 57 个采样点，覆盖最长 10 秒、51 点输入。输出头只读取
    序列最后一个位置，因而预测只依赖该位置及其过去；输出头可配置为三气体
    共享的 `head`（默认）或三路独立的逐气体 `heads`。
    """

    def __init__(self, in_channels: int, per_gas_head: bool = False) -> None:
        """建立模型结构。

        参数：`in_channels` 为特征通道数；`per_gas_head=True` 时用三个独立的
        16 宽小 MLP 头分别输出 H2/NH3/NO2（共享卷积主干），默认 False 保留原
        共享头。两种模式下模块互斥：False 只有 `head`，True 只有 `heads`。
        """
        super().__init__()
        self.in_channels = in_channels
        self.per_gas_head = per_gas_head
        self.blocks = nn.Sequential(
            CausalResidualBlock(in_channels, dilation=1),
            CausalResidualBlock(CHANNELS, dilation=2),
            CausalResidualBlock(CHANNELS, dilation=4),
        )
        if per_gas_head:
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(CHANNELS, 16), nn.ReLU(), nn.Linear(16, 1)
                    )
                    for _ in range(3)
                ]
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(CHANNELS, 16), nn.ReLU(), nn.Linear(16, 3)
            )

    def _decode(self, hidden: torch.Tensor) -> torch.Tensor:
        """把 `[batch,channels]` 隐藏表示解码成 `[batch,3]` 三气体输出。

        逐气体头分别输出一路气体后按 H2/NH3/NO2 顺序拼接；共享头一次输出
        三路。两个分支都只依赖传入位置的隐藏表示。
        """
        if self.per_gas_head:
            return torch.cat([head(hidden) for head in self.heads], dim=1)
        return self.head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """将一个或多个早期响应前缀回归成三气体浓度。

        参数：`x` 必须是 `[batch,in_channels,time]` 的标准化特征张量。
        返回：`[batch,3]` 的标准化 H2/NH3/NO2 预测；标签反标准化由训练器/
        推理器使用 checkpoint 中保存的标签均值和标准差完成。
        """
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected [batch,{self.in_channels},time] input")
        hidden = self.blocks(x)
        return self._decode(hidden[:, :, -1])

    def forward_positions(
        self, x: torch.Tensor, positions: tuple[int, ...]
    ) -> list[torch.Tensor]:
        """在同一隐藏序列的多个过去位置解码。

        因果卷积保证每个请求位置只看到自己的过去采样。该接口用于训练时的
        多时间尺度监督；推理仍然只读取最后一个位置。
        """
        if x.ndim != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected [batch,{self.in_channels},time] input")
        if not positions:
            raise ValueError("positions must not be empty")
        hidden = self.blocks(x)
        outputs = []
        for position in positions:
            index = int(position)
            if index < 0 or index >= x.shape[2]:
                raise ValueError(f"position {index} outside sequence length {x.shape[2]}")
            outputs.append(self._decode(hidden[:, :, index]))
        return outputs
