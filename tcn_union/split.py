"""验证固定的实验级划分，并生成数据清单。

测试集和五个验证折按实验文件预先锁定；划分时只考虑各气体浓度档位覆盖，
不依据响应曲线或模型分数重新抽样。将整个实验作为不可拆分单元，可避免
同一条曲线相邻时间点同时出现在训练与验证/测试数据中。
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import TypedDict

import numpy as np

from .data import load_run


class RunRecord(TypedDict):
    """一次实验在清单中的已校验记录结构。

    字段包含文件编号与路径、原始字节 SHA-256、完整时间数组、三气体标签，
    以及用于 inventory 的行列数和时间范围/步长。
    """
    id: int
    file: str
    sha256: str
    time: np.ndarray
    labels: np.ndarray
    rows: int
    columns: int
    start_time: float
    end_time: float
    step_seconds: float


# 固定测试实验 ID；运行时校验这些 ID，不在观察模型表现后重新抽样。
TEST_IDS = {2, 7, 15, 20, 23, 27, 31, 38, 41, 46, 55, 60}
FOLDS = [
    {5, 6, 16, 19, 22, 24, 42, 45, 50, 59},
    {4, 9, 12, 14, 26, 34, 35, 43, 54, 58},
    {0, 13, 18, 21, 32, 33, 40, 47, 52, 53},
    {3, 10, 11, 28, 30, 36, 37, 39, 49, 51},
    {1, 8, 17, 25, 29, 44, 48, 56, 57, 61},
]
EXPECTED_IDS = set(range(62))
MANIFEST_FIELDS = ["file", "sha256", "split", "fold", "H2", "NH3", "NO2"]
INVENTORY_FIELDS = [
    "file", "sha256", "rows", "columns", "start_time", "end_time", "step_seconds", "H2", "NH3", "NO2"
]


def _sha256(path: Path) -> str:
    """按块读取文件并计算 SHA-256。

    参数：`path` 是 CSV 文件路径。
    返回：文件原始字节的十六进制 SHA-256，用于后续识别数据是否被替换或修改。
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_layout(data_dir: Path) -> list[RunRecord]:
    """检查输入目录是否仍是预期的 62 次完整实验数据。

    参数：`data_dir` 包含编号为 0–61 的 CSV 文件。
    返回：按文件 ID 排序的 `RunRecord` 列表；列表内含经 `load_run` 校验的
    时间/标签，以及哈希和 inventory 元数据。
    发现缺失/额外文件、无效采样、重复响应或重复浓度组合时汇总后报错，
    而不是静默跳过，以保证锁定清单仍对应同一批数据。
    """
    files = sorted(data_dir.glob("*.csv"), key=lambda p: p.name)
    filenames = {path.name for path in files}
    expected_filenames = {f"{run_id}.csv" for run_id in EXPECTED_IDS}
    if filenames != expected_filenames:
        missing = sorted(expected_filenames - filenames)
        extra = sorted(filenames - expected_filenames)
        raise ValueError(f"dataset file set changed: missing files={missing}, extra files={extra}")

    records: list[RunRecord] = []
    signatures: dict[bytes, str] = {}
    labels_seen: dict[tuple[float, float, float], str] = {}
    errors: list[str] = []
    project_root = Path.cwd().resolve()
    for run_id in range(62):
        path = data_dir / f"{run_id}.csv"
        try:
            relative_file = path.resolve().relative_to(project_root).as_posix()
        except ValueError as exc:
            raise ValueError(f"{path}: CSV must be under project root {project_root}") from exc
        try:
            time, response, labels = load_run(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        signature = np.concatenate((time[:, None], response), axis=1).tobytes()
        prior = signatures.get(signature)
        if prior is not None:
            errors.append(f"{path}: numerically duplicates response in {prior}")
        else:
            signatures[signature] = str(path)
        key = (float(labels[0]), float(labels[1]), float(labels[2]))
        if key in labels_seen:
            errors.append(f"{path}: concentration triple duplicates {labels_seen[key]}")
        else:
            labels_seen[key] = str(path)
        records.append(
            {
                "id": run_id,
                "file": relative_file,
                "sha256": _sha256(path),
                "time": time,
                "labels": labels,
                "rows": len(time),
                "columns": 7,
                "start_time": float(time[0]),
                "end_time": float(time[-1]),
                "step_seconds": float(np.median(np.diff(time))),
            }
        )
    if errors:
        raise ValueError("dataset validation failed:\n" + "\n".join(errors))
    return records


def _validate_split(records: list[RunRecord]) -> dict[int, int]:
    """验证固定开发集五折构成无重叠的完整分区。

    参数：`records` 是已通过数据布局检查的实验记录。
    返回：`实验 ID -> 验证折编号` 映射。还检查每折训练集保留三个气体轴的
    全部开发集浓度档位、验证集每轴至少有观测，避免折内归一化误差无法定义。
    """
    development = EXPECTED_IDS - TEST_IDS
    fold_for: dict[int, int] = {}
    for fold_index, validation_ids in enumerate(FOLDS):
        if len(validation_ids) != 10 or validation_ids & TEST_IDS or validation_ids & set(fold_for):
            raise ValueError(f"invalid validation fold {fold_index}")
        fold_for.update({run_id: fold_index for run_id in validation_ids})
    if set(fold_for) != development or len(development) != 50:
        raise ValueError("development folds do not partition the fixed 50-run development set")
    by_id = {record["id"]: record for record in records}
    for fold_index, validation_ids in enumerate(FOLDS):
        train_ids = development - validation_ids
        for axis in range(3):
            all_levels = {float(by_id[i]["labels"][axis]) for i in development}
            train_levels = {float(by_id[i]["labels"][axis]) for i in train_ids}
            val_levels = {float(by_id[i]["labels"][axis]) for i in validation_ids}
            if all_levels != train_levels or not val_levels:
                raise ValueError(f"fold {fold_index} does not cover concentration axis {axis}")
    return fold_for


def generate(data_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    """生成 inventory.csv 与 manifest.csv，不移动或复制原始数据。

    参数：`data_dir` 为原始实验目录；`output_dir` 为清单输出目录。
    返回：`(inventory_path, manifest_path)`。清单记录相对路径、文件哈希、
    锁定的 development/test 身份、开发集折号和标签；inventory 另记采样布局。
    只有输入集合及固定折分通过校验后才写出文件。
    """
    records = _validate_layout(data_dir)
    fold_for = _validate_split(records)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    inventory_path = output_dir / "inventory.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            run_id = record["id"]
            labels = record["labels"]
            writer.writerow(
                {
                    "file": record["file"],
                    "sha256": record["sha256"],
                    "split": "test" if run_id in TEST_IDS else "development",
                    "fold": "" if run_id in TEST_IDS else fold_for[run_id],
                    "H2": labels[0],
                    "NH3": labels[1],
                    "NO2": labels[2],
                }
            )
    with inventory_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INVENTORY_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "file": record["file"],
                    "sha256": record["sha256"],
                    "rows": record["rows"],
                    "columns": record["columns"],
                    "start_time": record["start_time"],
                    "end_time": record["end_time"],
                    "step_seconds": record["step_seconds"],
                    "H2": record["labels"][0],
                    "NH3": record["labels"][1],
                    "NO2": record["labels"][2],
                }
            )
    return inventory_path, manifest_path


def main() -> None:
    """解析命令行路径、执行数据校验并写出两份清单。"""
    parser = argparse.ArgumentParser(description="Generate the locked gas-sensor run split")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory, manifest = generate(args.data_dir, args.output_dir)
    print(f"Wrote {inventory} and {manifest} ({len(EXPECTED_IDS)} runs)")


if __name__ == "__main__":
    main()
