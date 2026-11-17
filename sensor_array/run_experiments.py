"""训练最终灰盒模型，并在留一实验协议下比较各模型。

模型可以分组运行（长任务便于并行与断点续跑）：

    .venv/bin/python -m sensor_array.run_experiments mean_curve two_stage grey_box direct_ridge
    .venv/bin/python -m sensor_array.run_experiments tcn
    .venv/bin/python -m sensor_array.run_experiments --combine

每个模型把自己的 LOGO 指标与预测写到 ``artifacts/logo_<name>.csv`` 与
``artifacts/pred_<name>.npz``；``--combine`` 汇总全部模型、拟合最终模型并写出
``model.json``。不传参数时顺序跑完全部模型并汇总。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .baselines import DirectRidgeModel, MeanCurveModel, TCNModel, TwoStageModel
from .data import design_table, load_experiments
from .evaluate import logo, summarize
from .model import FirstOrderArrayModel, fit_experiment_shapes

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
MODEL_NAMES = ["mean_curve", "two_stage", "grey_box", "direct_ridge", "tcn"]


def _shape_by_file(experiments, shapes):
    amplitude, tau, delay = shapes
    return {
        experiment.file_id: (amplitude[i], tau[i], delay[i])
        for i, experiment in enumerate(experiments)
    }


def _subset_init(experiments, shape_by_file):
    return tuple(
        np.stack([shape_by_file[experiment.file_id][part] for experiment in experiments])
        for part in range(3)
    )


def run_one(name: str, experiments, shapes) -> pd.DataFrame:
    shape_by_file = _shape_by_file(experiments, shapes)

    def fit(train, test):
        if name == "mean_curve":
            model = MeanCurveModel().fit(train)
        elif name == "two_stage":
            model = TwoStageModel().fit(train, shape_init=_subset_init(train, shape_by_file))
        elif name == "grey_box":
            model = FirstOrderArrayModel().fit(train, shape_init=_subset_init(train, shape_by_file))
        elif name == "direct_ridge":
            model = DirectRidgeModel().fit(train)
        elif name == "tcn":
            model = TCNModel().fit(train)
        else:
            raise ValueError(f"未知模型：{name}")
        return model.predict_relative(test.concentration[None], test.time)[0]

    print(f"running LOGO: {name}", flush=True)
    frame = logo(experiments, fit, name)
    predictions = frame.attrs["predictions"]
    np.savez(ARTIFACTS / f"pred_{name}.npz", **{str(key): value for key, value in predictions.items()})
    frame.to_csv(ARTIFACTS / f"logo_{name}.csv", index=False)
    print(f"finished LOGO: {name}", flush=True)
    return frame


def combine() -> None:
    frames = [pd.read_csv(ARTIFACTS / f"logo_{name}.csv") for name in MODEL_NAMES]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(ARTIFACTS / "logo_metrics.csv", index=False)
    table = summarize(metrics)
    table.to_csv(ARTIFACTS / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))

    predictions = {}
    for name in MODEL_NAMES:
        with np.load(ARTIFACTS / f"pred_{name}.npz") as archive:
            predictions.update({f"{name}_{key}": archive[key] for key in archive.files})
    np.savez(ARTIFACTS / "logo_predictions.npz", **predictions)

    experiments = load_experiments()
    design_table(experiments).to_csv(ARTIFACTS / "design_table.csv", index=False)
    shapes = fit_experiment_shapes(experiments)
    final = FirstOrderArrayModel().fit(experiments, shape_init=shapes)
    final.save(ARTIFACTS / "model.json")
    relative = np.stack([experiment.relative for experiment in experiments])
    prediction = final.predict_relative(np.stack([e.concentration for e in experiments]), experiments[0].time)
    np.savez(
        ARTIFACTS / "final_predictions.npz",
        time=experiments[0].time,
        concentration=np.stack([e.concentration for e in experiments]),
        initial=np.stack([e.initial for e in experiments]),
        relative=relative,
        prediction=prediction,
        delay=final.delay,
    )
    print("delay (s):", np.round(final.delay, 3))
    print("final model saved to", ARTIFACTS / "model.json")


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    arguments = sys.argv[1:]
    if arguments == ["--combine"]:
        combine()
        return
    names = arguments or MODEL_NAMES
    experiments = load_experiments()
    design_table(experiments).to_csv(ARTIFACTS / "design_table.csv", index=False)
    shapes = fit_experiment_shapes(experiments)
    for name in names:
        run_one(name, experiments, shapes)
    if not arguments:
        combine()


if __name__ == "__main__":
    main()
