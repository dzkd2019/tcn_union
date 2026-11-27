"""绘制 62 折留一的分层正模型曲线，并输出逐实验 nRMSE。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.data import GASES, load_experiments
from ..paths import EVALUATION, FIGURES_MODEL
from ..plotting.style import _save, _style

OBSERVED_COLOR = "#2F4B7C"
GENERATED_COLOR = "#E07B39"


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    gas_scale = np.sqrt(np.mean(truth**2, axis=(0, 1)))
    rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
    nrmse = rmse / gas_scale[None, :]
    frame = pd.DataFrame(nrmse, columns=list(GASES))
    frame["macro"] = frame.mean(axis=1)
    return frame


def main() -> None:
    _style()
    experiments = load_experiments()
    payload = np.load(EVALUATION / "predictions.npz")
    generated = np.stack(
        [payload[f"{experiment.file_id}_hierarchical"] for experiment in experiments]
    )
    base = np.stack(
        [payload[f"{experiment.file_id}_mlp_per_gas"] for experiment in experiments]
    )
    truth = np.stack([experiment.relative for experiment in experiments])
    time = experiments[0].time

    metrics = _metrics(generated, truth)
    metrics.insert(0, "file", [experiment.file_id for experiment in experiments])
    metrics.to_csv(EVALUATION / "hierarchical_lofo_metrics.csv", index=False)
    base_metrics = _metrics(base, truth)
    print("hierarchical LOFO nRMSE mean:")
    print(metrics[list(GASES) + ["macro"]].mean().round(4).to_string())
    print("base MLP LOFO nRMSE mean:")
    print(base_metrics[list(GASES) + ["macro"]].mean().round(4).to_string())

    figure, axes = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=True)
    for gas_index, (gas, axis) in enumerate(zip(GASES, axes)):
        for experiment in experiments:
            axis.plot(
                time,
                experiment.relative[:, gas_index],
                color=OBSERVED_COLOR,
                linewidth=1.7,
                alpha=0.72,
            )
        for curve in generated:
            axis.plot(
                time,
                curve[:, gas_index],
                color=GENERATED_COLOR,
                linewidth=1.15,
                alpha=0.9,
            )
        axis.set_ylabel(f"{gas} 相对响应 / %")
        axis.set_title({"H2": "H$_2$", "NH3": "NH$_3$", "NO2": "NO$_2$"}[gas])
        axis.grid(alpha=0.18, linewidth=0.6)
        axis.set_xlim(time[0], time[-1])
    axes[-1].set_xlabel("时间 / s")
    handles = [
        plt.Line2D([], [], color=OBSERVED_COLOR, lw=1.8, label="实测（62 条）"),
        plt.Line2D([], [], color=GENERATED_COLOR, lw=1.3, label="生成（62 折留一）"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.005))
    figure.suptitle("62 折留一：生成曲线与实测曲线", fontsize=10.5, y=0.995)
    figure.tight_layout(rect=(0, 0.03, 1, 0.97))
    _save(figure, "fig20_hierarchical_lofo_62_curves", FIGURES_MODEL)


if __name__ == "__main__":
    main()
