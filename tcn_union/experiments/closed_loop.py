"""当前正模型（双指数 + 三路 MLP 残差）回到逆模型的闭环对照。

对每个窗口（5/10 s）和每个 OOF 折：
  1. 只用该折 40 条训练实验拟合当前正模型；
  2. 用正模型生成合成曲线（训练浓度 + 均匀随机浓度），用真实 R0 重建绝对响应；
  3. 训练逆 TCN：仅真实 vs 真实+合成，比较 OOF 50 的 3σ / MAPE。

锁定 12 条不参与本实验。输出 ``artifacts/analysis/closed_loop/closed_loop.csv``。
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn_union.core.data import TIME_STEP_SECONDS, make_features
from tcn_union.core.device import configure_threads, resolve_device
from tcn_union.training.gan_train import MODE, _read_manifest
from tcn_union.training.train import _train_network, fit_scalers

from sensor_array.core.data import load_experiments
from sensor_array.core.double_exponential import DoubleExponentialArrayModel
from sensor_array.core.mlp_per_gas import PerGasMLPDoubleExponentialModel

GASES = ("H2", "NH3", "NO2")
WINDOWS = (5, 10)
EPOCHS = 500
SEED = 20260923
BASE = ROOT / "artifacts/analysis/closed_loop"


def _three_sigma(values: np.ndarray) -> float:
    return float(3.0 * np.std(values, ddof=1))


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for axis, gas in enumerate(GASES):
        relative = 100.0 * (prediction[:, axis] - truth[:, axis]) / np.maximum(
            np.abs(truth[:, axis]), 1e-9
        )
        out[f"{gas}_three_sigma"] = _three_sigma(relative)
        out[f"{gas}_mape"] = float(np.mean(np.abs(relative)))
    out["macro_three_sigma"] = float(
        np.mean([out[f"{gas}_three_sigma"] for gas in GASES])
    )
    out["macro_mape"] = float(np.mean([out[f"{gas}_mape"] for gas in GASES]))
    return out


def _real_scaler(x_real: np.ndarray, y_real: np.ndarray) -> dict[str, np.ndarray]:
    """用真实训练折拟合 z-score 统计量，供合成预训练/微调共用。"""
    y_fit = y_real.copy()
    y_fit[:, 0] = np.log(np.maximum(y_fit[:, 0], 1e-6))
    y_fit[:, 2] = np.log(np.maximum(y_fit[:, 2], 1e-6))
    return fit_scalers(x_real, y_fit)


def _forward_curves(
    train_experiments: list,
    time: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """拟合当前正模型，生成合成浓度与 60 s 绝对响应（N,T,3）。"""
    skeleton = DoubleExponentialArrayModel().fit(train_experiments)
    forward = PerGasMLPDoubleExponentialModel(device="cuda", epochs=600).fit(
        train_experiments, skeleton=skeleton
    )
    train_concentration = np.stack([experiment.concentration for experiment in train_experiments])
    low = train_concentration.min(axis=0)
    high = train_concentration.max(axis=0)
    # 训练浓度本身 + 设计域内均匀随机浓度；随机部分补充覆盖。
    n_random = len(train_experiments)
    random_concentration = rng.uniform(low, high, size=(n_random, 3))
    concentration = np.concatenate((train_concentration, random_concentration), axis=0)
    relative = forward.predict_relative(concentration, time)
    initial = np.stack([experiment.initial for experiment in train_experiments])
    # 前一半与训练实验一一对应（同浓度、同 R0），后一半随机配 R0 补充基线多样性。
    chosen_initial = np.concatenate(
        (
            initial,
            initial[rng.integers(0, len(initial), size=n_random)],
        ),
        axis=0,
    )
    response = chosen_initial[:, None, :] * (1.0 + relative)
    return concentration, response


def main() -> None:
    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(ROOT / "artifacts/data/splits/manifest.csv")
    development = [row for row in rows if row["split"] == "development"]
    experiments = {experiment.file_id: experiment for experiment in load_experiments()}
    time = experiments[0].time
    rng = np.random.default_rng(SEED)
    BASE.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, float | str | int]] = []
    for fold in range(5):
        train_rows = [row for row in development if int(row["fold"]) != fold]
        val_rows = [row for row in development if int(row["fold"]) == fold]
        train_experiments = [
            experiments[int(Path(row["file"]).stem)] for row in train_rows
        ]
        concentration, synthetic_response = _forward_curves(train_experiments, time, rng)
        truth_train = np.stack([row["_labels"] for row in train_rows]).astype(np.float32)
        truth_val = np.stack([row["_labels"] for row in val_rows]).astype(np.float32)
        synthetic_labels = concentration.astype(np.float32)
        print(f"fold {fold}: synthetic={len(synthetic_labels)}", flush=True)

        for seconds in WINDOWS:
            x_real = np.stack(
                [
                    make_features(row["_time"], row["_response"], seconds, MODE)
                    for row in train_rows
                ]
            )
            x_val = np.stack(
                [
                    make_features(row["_time"], row["_response"], seconds, MODE)
                    for row in val_rows
                ]
            )
            x_synthetic = np.stack(
                [
                    make_features(time, response, seconds, MODE)
                    for response in synthetic_response
                ]
            )
            stats = _real_scaler(x_real, truth_train)
            arms = [
                ("real", x_real, truth_train),
                (
                    "real+synthetic(1:1)",
                    np.concatenate((x_real, x_synthetic[: len(x_real)]), axis=0),
                    np.concatenate((truth_train, synthetic_labels[: len(x_real)]), axis=0),
                ),
            ]
            for arm, x_train, y_train in arms:
                model, _, epochs_used, _, prediction = _train_network(
                    x_train,
                    y_train,
                    x_val,
                    truth_val,
                    truth_train,
                    EPOCHS,
                    SEED,
                    device,
                    True,
                    validation_metric="three_sigma",
                    target_transform="log,none,log",
                    per_gas_head=True,
                    low_conc_power="1.0",
                    scaler_stats=stats,
                )
                if prediction is None:
                    raise RuntimeError("validation prediction is None")
                record = {
                    "window": seconds,
                    "fold": fold,
                    "arm": arm,
                    "epochs": epochs_used,
                    "n_train": len(x_train),
                    **_metrics(prediction, truth_val),
                }
                records.append(record)
                print(
                    f"  {seconds}s {arm}: macro3σ={record['macro_three_sigma']:.2f}",
                    flush=True,
                )

            # 合成预训练 + 真实微调；预训练和微调共用真实训练折的 scaler。
            pretrained, _, pretrain_epochs, _, _ = _train_network(
                x_synthetic[: len(x_real)],
                synthetic_labels[: len(x_real)],
                None,
                None,
                truth_train,
                200,
                SEED,
                device,
                False,
                validation_metric="three_sigma",
                target_transform="log,none,log",
                per_gas_head=True,
                low_conc_power="1.0",
                scaler_stats=stats,
            )
            model, _, epochs_used, _, prediction = _train_network(
                x_real,
                truth_train,
                x_val,
                truth_val,
                truth_train,
                100,
                SEED,
                device,
                True,
                validation_metric="three_sigma",
                target_transform="log,none,log",
                per_gas_head=True,
                low_conc_power="1.0",
                initial_state=pretrained.state_dict(),
                scaler_stats=stats,
            )
            if prediction is None:
                raise RuntimeError("validation prediction is None")
            record = {
                "window": seconds,
                "fold": fold,
                "arm": "pretrain+finetune",
                "epochs": pretrain_epochs + epochs_used,
                "n_train": len(x_real),
                **_metrics(prediction, truth_val),
            }
            records.append(record)
            print(
                f"  {seconds}s pretrain+finetune: macro3σ={record['macro_three_sigma']:.2f}",
                flush=True,
            )

    path = BASE / "closed_loop.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
