"""传感器阵列动态响应实验的读取与相对响应构造。

每个 CSV 对应一次恒定浓度下的阶跃响应实验：前三列是三路传感器响应，
后三列是 H2/NH3/NO2 浓度标签。模型使用相对响应

    y_j(t) = (R_j(t) - R_j(0)) / R_j(0)

作为建模目标。这样做的原因是不同实验的初始基线 R_j(0) 相差很大（NH3 通道
可达 9 倍），而绝对响应幅度近似正比于基线；相对化后可以去掉这一乘性漂移。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..paths import DATA

GASES = ("H2", "NH3", "NO2")
# 三路响应的固定符号：全部实验中 H2、NH3 通道上升、NO2 通道下降。符号由数据
# 固定，不作为待拟合参数；这样幅度 A_j 为正，可以在对数空间参数化。
RESPONSE_SIGNS = np.array([1.0, 1.0, -1.0])
# 对数浓度特征的参考点，取各气体实验区间的几何中心附近。
REFERENCE_CONCENTRATION = np.array([16000.0, 30.0, 6.0])
DEFAULT_DATA_DIR = DATA
EXPECTED_HEADER = ["time", "H2", "NH3", "NO2", "conc1", "conc2", "conc3"]


@dataclass(frozen=True)
class Experiment:
    """一次阶跃响应实验。"""

    file_id: int
    time: np.ndarray  # (T,)
    response: np.ndarray  # (T, 3)
    concentration: np.ndarray  # (3,)

    @property
    def initial(self) -> np.ndarray:
        """实验开始时的三路基线读数 R(0)。"""
        return self.response[0]

    @property
    def relative(self) -> np.ndarray:
        """未加符号的相对响应 (R(t)-R(0))/R(0)，NO2 通道为负。"""
        return (self.response - self.initial) / self.initial


def load_experiments(data_dir: Path | str = DEFAULT_DATA_DIR) -> list[Experiment]:
    """按文件编号读取全部实验，并检查表头、时间轴和恒定浓度。"""
    data_dir = Path(data_dir)
    paths = sorted(data_dir.glob("*.csv"), key=lambda path: int(path.stem))
    if not paths:
        raise FileNotFoundError(f"{data_dir} 中没有 CSV 实验文件")

    experiments: list[Experiment] = []
    reference_time: np.ndarray | None = None
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        if frame.columns.tolist() != EXPECTED_HEADER:
            raise ValueError(f"{path}: 表头应为 {EXPECTED_HEADER}，实际为 {frame.columns.tolist()}")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{path}: 含缺失或非有限数值")

        time = values[:, 0]
        response = values[:, 1:4]
        concentration = values[0, 4:7]
        if np.ptp(values[:, 4:7], axis=0).max() > 1e-9:
            raise ValueError(f"{path}: 浓度标签在一次实验内不恒定")
        if np.any(np.diff(time) <= 0):
            raise ValueError(f"{path}: 时间轴不是严格递增")
        if np.any(response <= 0):
            raise ValueError(f"{path}: 传感器响应应为正电阻值")
        if reference_time is None:
            reference_time = time
        elif not np.allclose(time, reference_time, atol=1e-9):
            raise ValueError(f"{path}: 时间轴与首个实验不一致")

        experiments.append(
            Experiment(
                file_id=int(path.stem),
                time=time,
                response=response,
                concentration=concentration,
            )
        )
    return experiments


def stack(experiments: list[Experiment]) -> dict[str, np.ndarray]:
    """把实验列表堆叠成训练/评估用的数组。

    返回字典包含：time (T,)、concentration (N,3)、response (N,T,3)、
    initial (N,3)、relative (N,T,3)。
    """
    return {
        "time": experiments[0].time,
        "concentration": np.stack([item.concentration for item in experiments]),
        "response": np.stack([item.response for item in experiments]),
        "initial": np.stack([item.initial for item in experiments]),
        "relative": np.stack([item.relative for item in experiments]),
    }


def design_table(experiments: list[Experiment]) -> pd.DataFrame:
    """返回实验设计表：文件编号、三气体浓度和初始基线。"""
    return pd.DataFrame(
        {
            "file": [item.file_id for item in experiments],
            "H2": [item.concentration[0] for item in experiments],
            "NH3": [item.concentration[1] for item in experiments],
            "NO2": [item.concentration[2] for item in experiments],
            "R0_H2": [item.initial[0] for item in experiments],
            "R0_NH3": [item.initial[1] for item in experiments],
            "R0_NO2": [item.initial[2] for item in experiments],
        }
    )
