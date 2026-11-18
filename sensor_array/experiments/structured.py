"""P3：比较自由 B 样条残差与结构化残差（早期瞬态 / 中期形状 / 缓慢漂移）。

同一个训练折只拟合一阶骨架一次，再分别拟合两种残差基，保证比较公平。
输出逐折曲线指标、汇总、LOGO 折外预测与残差-设计变量相关。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..core.data import GASES, Experiment, load_experiments
from ..core.grey_box import FirstOrderArrayModel, fit_experiment_shapes
from ..core.hybrid import HybridArrayModel, bspline_basis, structured_basis
from ..core.metrics import block_folds, curve_metrics, level_folds
from ..paths import ANALYSIS, MODELS, VALIDATION, ensure

VARIANTS = {
    "spline": {},
    "structured": {"basis": "structured"},
}


def _logo_folds(experiments: list[Experiment]):
    return [(str(experiment.file_id), np.array([index])) for index, experiment in enumerate(experiments)]


def _subset_shapes(shapes, indices: np.ndarray):
    return tuple(part[indices] for part in shapes)


def _run_folds(experiments: list[Experiment], shapes) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    folds = {
        "logo": _logo_folds(experiments),
        "level": level_folds(experiments),
        "block": block_folds(experiments),
    }
    rows = []
    predictions: dict[str, np.ndarray] = {}
    for protocol, items in folds.items():
        for label, test_indices in items:
            train_indices = np.setdiff1d(np.arange(len(experiments)), test_indices)
            train = [experiments[index] for index in train_indices]
            grey = FirstOrderArrayModel().fit(
                train, shape_init=_subset_shapes(shapes, train_indices)
            )
            scale = np.sqrt(
                np.mean(np.stack([experiment.relative for experiment in train]) ** 2, axis=(0, 1))
            )
            for variant, options in VARIANTS.items():
                model = HybridArrayModel(**options).fit(train, grey=grey)
                for index in test_indices:
                    experiment = experiments[index]
                    prediction = model.predict_relative(
                        experiment.concentration[None], experiment.time
                    )[0]
                    if protocol == "logo":
                        predictions[f"{variant}_{experiment.file_id}"] = prediction
                    metrics = curve_metrics(prediction, experiment.relative, scale)
                    rows.append(
                        {
                            "protocol": protocol,
                            "fold": label,
                            "file": experiment.file_id,
                            "variant": variant,
                            **metrics,
                        }
                    )
        print(f"完成 {protocol}", flush=True)
    return pd.DataFrame(rows), predictions


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["nrmse_H2", "nrmse_NH3", "nrmse_NO2", "nrmse_macro"]
    rows = []
    for (protocol, variant), group in frame.groupby(["protocol", "variant"]):
        row = {"protocol": protocol, "variant": variant, "n": len(group)}
        for column in columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_median"] = float(group[column].median())
            row[f"{column}_p90"] = float(np.percentile(group[column], 90))
        rows.append(row)
    return pd.DataFrame(rows)


def _correlations(experiments: list[Experiment], predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    concentration = np.stack([experiment.concentration for experiment in experiments])
    total = np.sum(np.stack([experiment.relative for experiment in experiments]) ** 2, axis=(0, 1))
    count = (len(experiments) - 1) * experiments[0].relative.shape[0]
    scale = np.sqrt((total[None, :] - np.sum(
        np.stack([experiment.relative for experiment in experiments]) ** 2, axis=1
    )) / count)
    rows = []
    for variant in VARIANTS:
        error = np.stack(
            [
                predictions[f"{variant}_{experiment.file_id}"] - experiment.relative
                for experiment in experiments
            ]
        )
        nrmse = np.sqrt(np.mean(error**2, axis=1)) / scale
        for gas_index, gas in enumerate(GASES):
            values = {
                "file": np.array([experiment.file_id for experiment in experiments], float),
                "h2": concentration[:, 0],
                "nh3": concentration[:, 1],
                "no2": concentration[:, 2],
            }
            for variable, column in values.items():
                rho, p = spearmanr(column, nrmse[:, gas_index])
                rows.append(
                    {
                        "variant": variant,
                        "gas": gas,
                        "variable": variable,
                        "rho": float(rho),
                        "p": float(p),
                        "n": len(experiments),
                    }
                )
    return pd.DataFrame(rows)


def _basis_table(experiments: list[Experiment]) -> pd.DataFrame:
    time = experiments[0].time
    t_max = float(time[-1])
    free = bspline_basis(time, 5, 2, t_max)
    fixed = structured_basis(time, t_max)
    rows = []
    for name, matrix in (("spline", free), ("structured", fixed)):
        for index in range(matrix.shape[1]):
            for t, value in zip(time, matrix[:, index]):
                rows.append({"basis": name, "component": index + 1, "time": t, "value": value})
    return pd.DataFrame(rows)


def main() -> None:
    ensure()
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    frame, predictions = _run_folds(experiments, shapes)
    summary = _summary(frame)
    correlations = _correlations(experiments, predictions)
    basis = _basis_table(experiments)

    frame.to_csv(VALIDATION / "structured_curves.csv", index=False, encoding="utf-8")
    summary.to_csv(VALIDATION / "structured_summary.csv", index=False, encoding="utf-8")
    correlations.to_csv(VALIDATION / "structured_correlations.csv", index=False, encoding="utf-8")
    basis.to_csv(ANALYSIS / "structured_basis.csv", index=False, encoding="utf-8")
    np.savez(VALIDATION / "structured_logo_predictions.npz", **predictions)

    grey = FirstOrderArrayModel().fit(experiments, shape_init=shapes)
    final = HybridArrayModel(basis="structured").fit(experiments, grey=grey)
    final.save(MODELS / "hybrid_structured.json")

    print((summary.set_index(["protocol", "variant"])[
        ["nrmse_macro_mean", "nrmse_macro_median", "nrmse_macro_p90"]
    ] * 100).round(3).to_string())
    print(correlations[correlations.variable == "file"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
