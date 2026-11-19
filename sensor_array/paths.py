"""统一路径常量。

所有模块都从这里取数据、产物和图目录，避免在各自文件里重复拼路径。
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / "data"

ARTIFACTS = ROOT / "artifacts"
MODELS = ARTIFACTS / "models"
EVALUATION = ARTIFACTS / "evaluation"
CALIBRATION = ARTIFACTS / "calibration"
FIGURES = ROOT / "figures"
FIGURES_MODEL = FIGURES / "model"


def ensure() -> None:
    """创建当前模型的输出目录；写文件前调用。"""
    for path in (MODELS, EVALUATION, CALIBRATION, FIGURES_MODEL):
        path.mkdir(parents=True, exist_ok=True)
