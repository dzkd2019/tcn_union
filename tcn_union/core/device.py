"""跨 NVIDIA CUDA / AMD ROCm / CPU 的设备选择。

PyTorch 在 ROCm 上沿用 ``cuda`` 这套 API：``torch.cuda.is_available()`` 为
True，``torch.version.hip`` 有版本号，但设备字符串仍然是 ``cuda``。
因此项目不硬编码 CUDA，而是统一从 ``resolve_device`` 取得设备。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch


def _has_rocm_driver() -> bool:
    """检查系统是否存在 AMD ROCm 驱动/工具。"""
    return Path("/dev/kfd").exists() or shutil.which("rocm-smi") is not None


def resolve_device(requested: str = "auto") -> torch.device:
    """解析 ``auto/cpu/cuda/cuda:0``，并在 AMD 机器上给出明确错误。

    - ``auto``：有可用 GPU 就用 GPU，否则 CPU；
    - ROCm 版 PyTorch 也使用 ``cuda`` 设备名；
    - 如果系统有 ROCm 驱动但 PyTorch 是 CUDA 构建，报错提示运行安装脚本，
      避免静默退回 CPU 导致训练极慢。
    """
    value = (requested or "auto").strip().lower()
    if value not in {"auto", "cpu", "cuda", "cuda:0"}:
        raise ValueError("device 必须是 auto、cpu、cuda 或 cuda:0")

    if value == "auto":
        if _has_rocm_driver() and torch.version.hip is None:
            raise RuntimeError(
                "检测到 AMD ROCm 驱动，但当前 PyTorch 不是 ROCm 构建；"
                "请先在项目根目录运行 scripts/setup_rocm.sh"
            )
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(value)
    if device.type == "cuda":
        if _has_rocm_driver() and torch.version.hip is None:
            raise RuntimeError(
                "检测到 AMD ROCm 驱动，但当前 PyTorch 不是 ROCm 构建；"
                "请先运行 scripts/setup_rocm.sh，或显式使用 --device cpu"
            )
        if not torch.cuda.is_available():
            raise RuntimeError(
                "请求了 cuda 设备，但当前 PyTorch 检测不到可用 GPU；"
                "请检查驱动，或使用 --device auto/cpu"
            )
    return device


def describe_device(device: torch.device) -> str:
    """返回便于日志确认的 CUDA/ROCm/CPU 描述。"""
    if device.type != "cuda":
        return "CPU"
    if torch.version.hip is not None:
        name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "AMD GPU"
        return f"ROCm {torch.version.hip} · {name}"
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NVIDIA GPU"
    return f"CUDA {torch.version.cuda} · {name}"


def configure_threads(device: torch.device) -> None:
    """CPU 训练时限制为单线程，避免小模型线程调度开销。"""
    if device.type == "cpu":
        torch.set_num_threads(1)
