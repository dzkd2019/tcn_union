"""62 折留一实验评估：双指数骨架与三路独立 MLP 残差。

输出曲线误差、预测数组和全量训练模型；不再包含 MAP 反演或区间评估。
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from ..core.data import GASES, Experiment, load_experiments
from ..core.double_exponential import DoubleExponentialArrayModel
from ..core.metrics import curve_metrics
from ..core.mlp_per_gas import PerGasMLPDoubleExponentialModel
from ..core.shapes import fit_experiment_shapes
from ..paths import EVALUATION, MODELS, ensure


def _curve_rows(test: Experiment, model, prediction: np.ndarray, scale: np.ndarray) -> dict:
    row = {"file": test.file_id, "model": model.name}
    row.update(curve_metrics(prediction, test.relative, scale))
    return row


def _summarize_curve(curve: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, group in curve.groupby("model", sort=False):
        for gas in (*GASES, "macro"):
            column = "nrmse_macro" if gas == "macro" else f"nrmse_{gas}"
            values = group[column]
            rows.append(
                {
                    "model": model,
                    "gas": gas,
                    "nrmse_mean": values.mean(),
                    "nrmse_std": values.std(),
                    "nrmse_median": values.median(),
                    "nrmse_p90": values.quantile(0.9),
                }
            )
    return pd.DataFrame(rows)


def run(folds: int = 62, max_iter: int = 600, batch_size: int = 0) -> None:
    ensure()
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    curve_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}
    started = time.time()
    for index, test in enumerate(experiments[:folds]):
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        shape_init = tuple(shapes[k][train_indices] for k in range(3))
        double = DoubleExponentialArrayModel().fit(train, shape_init=shape_init)
        double.name = "double"
        combined = PerGasMLPDoubleExponentialModel(
            epochs=max_iter, batch_size=batch_size
        ).fit(train, skeleton=double)
        combined.name = "mlp_per_gas"
        scale = np.sqrt(np.mean(np.stack([item.relative for item in train]) ** 2, axis=(0, 1)))
        for model in (double, combined):
            prediction = model.predict_relative(
                test.concentration[None, :], test.time
            )[0]
            curve_rows.append(_curve_rows(test, model, prediction, scale))
            predictions[f"{test.file_id}_{model.name}"] = prediction
        elapsed = time.time() - started
        print(
            f"fold {index + 1}/{min(folds, len(experiments))} file {test.file_id} "
            f"({elapsed:.1f} s, double nfev={double.nfev_}, "
            f"mlp iter={combined.n_iter_}, device={combined.device_}, "
            f"batch={combined.batch_size}",
            flush=True,
        )

    curve = pd.DataFrame(curve_rows)
    curve.to_csv(EVALUATION / "curve_metrics.csv", index=False)
    _summarize_curve(curve).to_csv(EVALUATION / "curve_summary.csv", index=False)
    np.savez_compressed(EVALUATION / "predictions.npz", **predictions)

    if folds >= len(experiments):
        full_double = DoubleExponentialArrayModel().fit(experiments, shape_init=shapes)
        full_combined = PerGasMLPDoubleExponentialModel(
            epochs=max_iter, batch_size=batch_size
        ).fit(experiments, skeleton=full_double)
        full_combined.save(MODELS / "mlp_per_gas.json")
        full_double.save(MODELS / "double.json")
    print("saved", EVALUATION, "and", MODELS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=62, help="留一折数；调试可用 3")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=600,
        help="三路独立 MLP 的固定训练轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="三路独立 MLP 的小批量大小；0 表示全批",
    )
    args = parser.parse_args()
    run(folds=args.folds, max_iter=args.max_iter, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
