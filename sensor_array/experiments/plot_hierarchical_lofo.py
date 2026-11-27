"""绘制 62 折留一的分层正模型曲线（每气体一张），并输出逐实验 nRMSE。"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..core.data import GASES, load_experiments
from ..paths import EVALUATION, FIGURES_MODEL
from ..plotting.style import GAS_COLORS, _save, _style

OBSERVED_COLOR = "#7F7F7F"
GAS_LABELS = {"H2": "H$_2$", "NH3": "NH$_3$", "NO2": "NO$_2$"}


def _metrics(prediction: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    gas_scale = np.sqrt(np.mean(truth**2, axis=(0, 1)))
    rmse = np.sqrt(np.mean((prediction - truth) ** 2, axis=1))
    nrmse = rmse / gas_scale[None, :]
    frame = pd.DataFrame(nrmse, columns=list(GASES))
    frame["macro"] = frame.mean(axis=1)
    return frame


def _figure(
    time: np.ndarray,
    observed: np.ndarray,
    generated: np.ndarray,
    gas: str,
    index: int,
    legend_loc: str,
    filename: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.2, 4.3))
    for experiment in observed:
        axis.plot(
            time,
            experiment[:, index],
            color=OBSERVED_COLOR,
            linewidth=0.95,
            alpha=0.80,
        )
    for curve in generated:
        axis.plot(
            time,
            curve[:, index],
            color=GAS_COLORS[gas],
            linewidth=1.15,
            alpha=0.75,
        )
    axis.set_title(
        f"{GAS_LABELS[gas]}：62 条实测与留一生成",
        fontsize=11.5,
        pad=8,
    )
    axis.set_xlabel("时间 / s", fontsize=10.5)
    axis.set_ylabel("相对响应 / %", fontsize=10.5)
    axis.tick_params(labelsize=10)
    axis.set_xlim(time[0], time[-1])
    axis.grid(alpha=0.18, linewidth=0.6)
    for spine in axis.spines.values():
        spine.set_linewidth(1.0)
    handles = [
        plt.Line2D([], [], color=OBSERVED_COLOR, lw=1.1, label="实测"),
        plt.Line2D([], [], color=GAS_COLORS[gas], lw=1.3, label="生成"),
    ]
    axis.legend(
        handles=handles,
        loc=legend_loc,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        edgecolor="none",
        fontsize=10,
        handlelength=1.8,
    )
    figure.tight_layout()
    _save(figure, filename, FIGURES_MODEL)


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
    print("hierarchical LOFO nRMSE mean:")
    print(metrics[list(GASES) + ["macro"]].mean().round(4).to_string())
    print("base MLP LOFO nRMSE mean:")
    print(_metrics(base, truth)[list(GASES) + ["macro"]].mean().round(4).to_string())

    names = {"H2": "H2", "NH3": "NH3", "NO2": "NO2"}
    legend = {"H2": "upper left", "NH3": "upper left", "NO2": "upper right"}
    for index, gas in enumerate(GASES):
        _figure(
            time,
            truth,
            generated,
            gas,
            index,
            legend[gas],
            f"fig21_hierarchical_lofo_{names[gas]}" if index == 0 else
            f"fig2{index + 1}_hierarchical_lofo_{names[gas]}",
        )


if __name__ == "__main__":
    main()
