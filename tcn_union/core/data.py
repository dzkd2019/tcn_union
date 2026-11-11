"""原始传感器 CSV 的校验、读取与早期响应特征构造。

CSV 前三路传感器响应与后三列监督浓度标签分开处理；预测入口只读时间和
传感器列，避免推理依赖真实浓度。所有模型特征仅使用从 t=0 到指定时限的
前缀，增量按初始响应幅度归一化，以保留相对变化并减弱量纲差异。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_HEADER = ["time", "H2", "NH3", "NO2", "conc1", "conc2", "conc3"]
TIME_STEP_SECONDS = 0.2
TIME_TOLERANCE = 1e-6


def _read_rows(path: Path, expected_header: list[str]) -> np.ndarray:
    """读取并验证数值 CSV。

    参数：`path` 为 CSV 路径；`expected_header` 是必须完全匹配的列名顺序。
    返回：行×列的 float64 数组。遇到表头、列数、数值或有限性错误时抛出
    `ValueError`，以免静默丢弃或修补传感器采样。
    """
    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
            index_col=False,
            on_bad_lines="error",
        )
    except OSError as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"{path}: malformed CSV: {exc}") from exc
    if frame.columns.tolist() != expected_header:
        raise ValueError(f"{path}: expected header {expected_header}, got {frame.columns.tolist()}")
    raw = frame.to_numpy(dtype=object)
    try:
        values = raw.astype(np.float64)
    except (TypeError, ValueError) as exc:
        for line_number, row in enumerate(raw, start=2):
            try:
                np.asarray(row, dtype=np.float64)
            except (TypeError, ValueError) as row_exc:
                raise ValueError(f"{path}:{line_number}: non-numeric value") from row_exc
        raise ValueError(f"{path}: contains non-numeric values") from exc
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] != len(expected_header):
        raise ValueError(f"{path}: expected at least two rows and {len(expected_header)} columns")
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: contains missing or non-finite numeric values")
    return values


def load_run(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取一次完整实验，返回时间、三路响应和固定标签。

    参数：`path` 为列顺序固定的实验 CSV。
    返回：`(time[N], responses[N,3], labels[3])`，响应列依次为 H2、NH3、
    NO2，标签来自 conc1–conc3。校验时间从 0 开始、按 0.2 秒严格递增，
    标签非负且在一次实验内不变；按文件返回整段是为了后续按实验划分，
    避免同一实验的时间点泄漏到不同数据集。
    """
    values = _read_rows(Path(path), EXPECTED_HEADER)
    time = values[:, 0]
    responses = values[:, 1:4]
    labels = values[:, 4:7]
    if abs(time[0]) > TIME_TOLERANCE:
        raise ValueError(f"{path}: time must start at 0, got {time[0]}")
    intervals = np.diff(time)
    if np.any(intervals <= 0):
        raise ValueError(f"{path}: time must be strictly increasing")
    if not np.allclose(intervals, TIME_STEP_SECONDS, rtol=0.0, atol=TIME_TOLERANCE):
        raise ValueError(f"{path}: time step must be {TIME_STEP_SECONDS} seconds")
    if np.any(labels < 0):
        raise ValueError(f"{path}: concentration labels must be non-negative")
    if not np.all(labels == labels[0]):
        raise ValueError(f"{path}: concentration labels change within the run")
    return time, responses, labels[0].copy()


def load_prediction_prefix(path: Path, seconds: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """只读取用于推理的传感器前缀，不要求或使用浓度标签。

    参数：`path` 为含 time/H2/NH3/NO2 的 CSV；`seconds` 是可选的读取上限，
    提供时在目标时刻后停止读取，以减少无关数据访问并保证预测不看未来。
    返回：`(time[N], responses[N,3])`。验证固定采样间隔、起点、有限值和
    至少两个样本；输入列名若不符合预测协议则报错。
    """
    expected = ["time", "H2", "NH3", "NO2"]
    row_limit = None
    if seconds is not None and np.isfinite(seconds):
        # 时间采样间隔固定为 0.2 秒；只读到可能出现的第一个越界样本，
        # 避免将未来响应或标签列用于预测。
        row_limit = max(1, int(np.ceil(seconds / TIME_STEP_SECONDS)) + 1)
    try:
        frame = pd.read_csv(
            Path(path),
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
            index_col=False,
            on_bad_lines="error",
            nrows=row_limit,
        )
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"{path}: first four columns must be {expected}, got None") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"{path}: malformed CSV prefix: {exc}") from exc
    header = frame.columns.tolist()
    if header[:4] != expected:
        raise ValueError(f"{path}: first four columns must be {expected}, got {header}")
    rows: list[list[float]] = []
    for line_number, row in enumerate(frame.itertuples(index=False, name=None), start=2):
        if len(row) < 4:
            raise ValueError(f"{path}:{line_number}: expected at least four columns")
        try:
            row_time = float(row[0])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid time value") from exc
        if seconds is not None and row_time > seconds + TIME_TOLERANCE:
            break
        try:
            rows.append([row_time, float(row[1]), float(row[2]), float(row[3])])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}:{line_number}: invalid time/response value") from exc
        if seconds is not None and row_time >= seconds - TIME_TOLERANCE:
            break
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ValueError(f"{path}: prefix needs at least two finite time/response rows")
    time, responses = values[:, 0], values[:, 1:4]
    if abs(time[0]) > TIME_TOLERANCE or np.any(np.diff(time) <= 0):
        raise ValueError(f"{path}: time must start at 0 and strictly increase")
    if not np.allclose(np.diff(time), TIME_STEP_SECONDS, rtol=0.0, atol=TIME_TOLERANCE):
        raise ValueError(f"{path}: time step must be {TIME_STEP_SECONDS} seconds")
    return time, responses


def make_features(
    time: np.ndarray, responses: np.ndarray, seconds: float, mode: str
) -> np.ndarray:
    """构造固定时长、严格因果的 TCN 输入特征。

    参数：`time[N]` 和 `responses[N,3]` 描述时序；`seconds` 仅允许 5 或
    10 秒；`mode` 为 `delta` 或 `raw_delta`。
    返回：float32 数组 `[channels,samples]`，终点包含在前缀中，因此 5/10
    秒输入分别有 26/51 个采样点。相对增量使用
    `(response(t)-response(0))/max(|response(0)|,1e-8)`；`raw_delta` 还拼接
    三路原始响应，`delta` 只保留三路相对增量。相对增量减弱传感器幅度尺度
    影响，保留原始通道则允许模型使用初始响应中可能有用的静态信息；两种
    模式分别训练以做消融比较。
    """
    if mode not in {"raw_delta", "delta"}:
        raise ValueError("mode must be 'raw_delta' or 'delta'")
    if seconds not in {5.0, 10.0}:
        raise ValueError("seconds must be 5.0 or 10.0")
    time = np.asarray(time, dtype=np.float64)
    responses = np.asarray(responses, dtype=np.float64)
    if time.ndim != 1 or responses.shape != (time.size, 3):
        raise ValueError("expected time[N] and responses[N,3]")
    prefix_times: list[float] = []
    prefix_responses: list[np.ndarray] = []
    for sample_time, sample_response in zip(time, responses, strict=True):
        if not np.isfinite(sample_time):
            raise ValueError("time must be finite through the requested prefix")
        if sample_time > seconds + TIME_TOLERANCE:
            break
        if not np.isfinite(sample_response).all():
            raise ValueError("responses must be finite through the requested prefix")
        if not prefix_times:
            if abs(sample_time) > TIME_TOLERANCE:
                raise ValueError("time must start at 0")
        elif not np.isclose(sample_time - prefix_times[-1], TIME_STEP_SECONDS, rtol=0.0, atol=TIME_TOLERANCE):
            raise ValueError(f"time step must be {TIME_STEP_SECONDS} seconds through the requested prefix")
        prefix_times.append(float(sample_time))
        prefix_responses.append(sample_response)
    prefix_time = np.asarray(prefix_times, dtype=np.float64)
    prefix = np.asarray(prefix_responses, dtype=np.float64)
    expected_samples = round(seconds / TIME_STEP_SECONDS) + 1
    if prefix_time.size < expected_samples:
        end_time = prefix_time[-1] if prefix_time.size else 0.0
        raise ValueError(f"recording ends at {end_time} s; {seconds} s prefix is unavailable")
    if prefix_time.size != expected_samples or abs(prefix_time[-1] - seconds) > TIME_TOLERANCE:
        raise ValueError(f"{seconds} s prefix does not have the expected 0.2 s sampling")
    baseline = prefix[0]
    relative_delta = (prefix - baseline) / np.maximum(np.abs(baseline), 1e-8)
    channels = (
        np.concatenate((prefix.T, relative_delta.T), axis=0)
        if mode == "raw_delta"
        else relative_delta.T
    )
    return np.asarray(channels, dtype=np.float32)


def load_features(path: Path, seconds: float, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """加载单个带标签实验并生成训练特征。

    参数：`path` 为完整实验 CSV；`seconds` 与 `mode` 传给
    `make_features`，决定因果前缀长度和通道类型。
    返回：`(features[channels,samples], labels[3])`。此便捷入口适用于训练和
    评估；无标签推理应使用 `load_prediction_prefix`。
    """
    time, responses, labels = load_run(path)
    return make_features(time, responses, seconds, mode), labels
