"""按 AGENTS.md §10 生成 TCN 逆问题最终评估的 metrics.csv 与图件。

输入是一次运行保存的 predictions.npz（truth、file_id 与 <arm>_s<seed> 键）。
本脚本一次处理一个 arm，输出到独立目录：

    artifacts/<run_name>/<arm>/
        predictions.csv
        metrics.csv
        run_config.json
        figs/fig_abs_error.png/.svg
        figs/fig_rel_error.png/.svg
        figs/fig_ci95.png/.svg

NRMSE 的浓度量程统一取开发集 50 条实验的逐气体极差，所有 seed 共用。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from sensor_array.plotting.style import GAS_COLORS, _style

GASES = ("H2", "NH3", "NO2")
GAS_UNICODE = {"H2": "H₂", "NH3": "NH₃", "NO2": "NO₂"}
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "artifacts/full_protocol_augmentation/predictions.npz"


def _relative(prediction: np.ndarray, truth: np.ndarray) -> np.ndarray:
    return 100.0 * (prediction - truth) / truth


def _metrics(
    prediction: np.ndarray, truth: np.ndarray, ranges: np.ndarray
) -> dict[str, float]:
    relative = _relative(prediction, truth)
    result: dict[str, float] = {}
    for index, gas in enumerate(GASES):
        values = relative[:, index]
        n = len(values)
        mean = float(values.mean())
        sd = float(values.std(ddof=1))
        half = float(t.ppf(0.975, n - 1) * sd / np.sqrt(n))
        result[f"{gas}_n"] = n
        result[f"{gas}_MAPE_percent"] = float(np.mean(np.abs(values)))
        result[f"{gas}_NRMSE"] = float(
            np.sqrt(np.mean((prediction[:, index] - truth[:, index]) ** 2))
            / ranges[index]
        )
        result[f"{gas}_three_sigma_percent"] = 3.0 * sd
        result[f"{gas}_mean_relative_percent"] = mean
        result[f"{gas}_ci95_low_percent"] = mean - half
        result[f"{gas}_ci95_high_percent"] = mean + half
    result["macro_MAPE_percent"] = float(
        np.mean([result[f"{gas}_MAPE_percent"] for gas in GASES])
    )
    result["macro_NRMSE"] = float(
        np.mean([result[f"{gas}_NRMSE"] for gas in GASES])
    )
    result["macro_three_sigma_percent"] = float(
        np.mean([result[f"{gas}_three_sigma_percent"] for gas in GASES])
    )
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row[key]) for key in keys) + "\n")


def _save(figure: plt.Figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=300)
    figure.savefig(directory / f"{name}.svg")
    plt.close(figure)


def _scatter_figure(
    prediction: np.ndarray,
    truth: np.ndarray,
    file_ids: np.ndarray,
    directory: Path,
    kind: str,
    label: str,
) -> None:
    if kind == "absolute":
        values = prediction - truth
        ylabel = "有符号绝对误差 / 原始单位"
        title = "有符号绝对误差"
        name = "fig_abs_error"
    else:
        values = _relative(prediction, truth)
        ylabel = "有符号相对误差 / %"
        title = "有符号相对误差"
        name = "fig_rel_error"
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), layout="constrained")
    for index, gas in enumerate(GASES):
        axis = axes[index]
        series = values[:, index]
        mean = float(np.mean(series))
        sd = float(np.std(series, ddof=1))
        axis.axhline(0.0, color="black", linewidth=1.0)
        axis.scatter(
            file_ids,
            series,
            s=16,
            color=GAS_COLORS[gas],
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )
        for multiple, style in ((1, ":"), (2, "--"), (3, "-.")):
            axis.axhline(
                mean + multiple * sd,
                color="0.35",
                linewidth=0.7,
                linestyle=style,
            )
            axis.axhline(
                mean - multiple * sd,
                color="0.35",
                linewidth=0.7,
                linestyle=style,
            )
        axis.set_title(GAS_UNICODE[gas])
        axis.set_xlabel("实验序号")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.text(
            0.02,
            0.98,
            f"mean={mean:.3f}\n±1σ={sd:.3f}\n±2σ={2*sd:.3f}\n±3σ={3*sd:.3f}",
            transform=axis.transAxes,
            va="top",
            fontsize=7.0,
        )
    figure.suptitle(f"{title}（{label}）")
    _save(figure, directory, name)


def _ci_figure(
    entries: list[tuple[str, np.ndarray]],
    truth: np.ndarray,
    directory: Path,
    label: str,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.2, 3.2), layout="constrained")
    x = np.arange(len(entries))
    for index, gas in enumerate(GASES):
        axis = axes[index]
        values = []
        for _, prediction in entries:
            relative = _relative(prediction, truth)[:, index]
            n = len(relative)
            mean = float(relative.mean())
            half = float(
                t.ppf(0.975, n - 1) * relative.std(ddof=1) / np.sqrt(n)
            )
            values.append((mean, half, n))
        axis.errorbar(
            x,
            [item[0] for item in values],
            yerr=[item[1] for item in values],
            fmt="o",
            color=GAS_COLORS[gas],
            capsize=3,
            markersize=5,
            linewidth=1.0,
        )
        axis.axhline(0.0, color="black", linewidth=0.9, linestyle="--")
        axis.set_xticks(x, [name for name, _ in entries], rotation=20)
        axis.set_ylabel("相对误差均值 / %")
        axis.set_title(GAS_UNICODE[gas])
        axis.grid(alpha=0.2, axis="y")
        for position, (mean, half, n) in enumerate(values):
            axis.text(
                position,
                mean + half,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )
    figure.suptitle(f"均值与 95% CI（{label}）")
    _save(figure, directory, "fig_ci95")


def run(
    source: Path,
    output: Path,
    arm: str,
    window: float,
    run_name: str,
) -> None:
    payload = np.load(source)
    truth = payload["truth"].astype(np.float64)
    file_ids = payload["file_id"].astype(int)
    ranges = np.ptp(truth, axis=0)
    if np.any(ranges <= 0):
        raise ValueError("开发集浓度量程必须为正")

    seeds = []
    for key in payload.files:
        prefix = f"{arm}_s"
        if key.startswith(prefix):
            seeds.append((key[len(prefix):], payload[key].astype(np.float64)))
    if not seeds:
        raise ValueError(f"没有找到 arm={arm} 的逐 seed 预测")
    seeds.sort(key=lambda item: item[0])
    ensemble_key = f"{arm}_ensemble"
    entries = list(seeds)
    if ensemble_key in payload.files:
        entries.append(("ensemble", payload[ensemble_key].astype(np.float64)))

    output.mkdir(parents=True, exist_ok=True)
    prediction_rows: list[dict] = []
    metric_rows: list[dict] = []
    for name, prediction in entries:
        if prediction.shape != truth.shape:
            raise ValueError(f"{name} 预测形状 {prediction.shape} 与真值不符")
        metrics = _metrics(prediction, truth, ranges)
        metric_rows.append(
            {
                "arm": arm,
                "prediction": name,
                "window_s": window,
                "range_source": "development_50",
                **{key: round(value, 6) for key, value in metrics.items()},
            }
        )
        relative = _relative(prediction, truth)
        for row_index, file_id in enumerate(file_ids):
            for gas_index, gas in enumerate(GASES):
                prediction_rows.append(
                    {
                        "arm": arm,
                        "prediction": name,
                        "window_s": window,
                        "file_id": int(file_id),
                        "gas": gas,
                        "truth": float(truth[row_index, gas_index]),
                        "prediction": float(prediction[row_index, gas_index]),
                        "abs_error": float(
                            prediction[row_index, gas_index]
                            - truth[row_index, gas_index]
                        ),
                        "rel_error_percent": float(relative[row_index, gas_index]),
                    }
                )

    _write_csv(output / "metrics.csv", metric_rows)
    _write_csv(output / "predictions.csv", prediction_rows)
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "arm": arm,
                "window_s": window,
                "source": str(source),
                "range_source": "development_50",
                "n": int(len(truth)),
                "seeds": [name for name, _ in seeds],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    _style()
    figure_dir = output / "figs"
    final_prediction = next(
        (prediction for name, prediction in entries if name == "ensemble"),
        entries[0][1],
    )
    _scatter_figure(
        final_prediction, truth, file_ids, figure_dir, "absolute", arm
    )
    _scatter_figure(
        final_prediction, truth, file_ids, figure_dir, "relative", arm
    )
    _ci_figure(entries, truth, figure_dir, arm)
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    run(
        Path(args.source),
        Path(args.output),
        args.arm,
        args.window,
        args.run_name,
    )


if __name__ == "__main__":
    main()
