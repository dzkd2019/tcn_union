"""训练最终灰盒模型，并在留一实验协议下比较各模型。

模型可以分组运行（长任务便于并行与断点续跑）：

    .venv/bin/python -m sensor_array.experiments.logo mean_curve two_stage grey_box direct_ridge
    .venv/bin/python -m sensor_array.experiments.logo tcn
    .venv/bin/python -m sensor_array.experiments.logo --combine

每个模型把自己的 LOGO 指标与预测写到 ``artifacts/logo/logo_<name>.csv`` 与
``artifacts/logo/pred_<name>.npz``；``--combine`` 汇总全部模型、拟合最终模型并写出
``artifacts/models/model.json`` 与 ``artifacts/models/hybrid.json``。不传参数时顺序跑完全部模型并汇总。
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from ..core.baselines import DirectRidgeModel, MeanCurveModel, TCNModel, TwoStageModel
from ..core.data import design_table, load_experiments
from ..core.metrics import logo, summarize
from ..core.hybrid import HybridArrayModel
from ..core.grey_box import FirstOrderArrayModel, fit_experiment_shapes
from ..paths import LOGO, MODELS, ensure

MODEL_NAMES = ["mean_curve", "two_stage", "grey_box", "hybrid", "direct_ridge", "tcn"]


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


def fit_model(name: str, train, shape_init=None):
    """按名称构造并拟合一个模型，供 LOGO 与分组验证协议共用。"""
    if name == "mean_curve":
        return MeanCurveModel().fit(train)
    if name == "two_stage":
        return TwoStageModel().fit(train, shape_init=shape_init)
    if name == "grey_box":
        return FirstOrderArrayModel().fit(train, shape_init=shape_init)
    if name == "hybrid":
        return HybridArrayModel().fit(train, shape_init=shape_init)
    if name == "direct_ridge":
        return DirectRidgeModel().fit(train)
    if name == "tcn":
        return TCNModel().fit(train)
    raise ValueError(f"未知模型：{name}")


def run_one(name: str, experiments, shapes) -> pd.DataFrame:
    shape_by_file = _shape_by_file(experiments, shapes)

    def fit(train, test):
        model = fit_model(name, train, _subset_init(train, shape_by_file))
        return model.predict_relative(test.concentration[None], test.time)[0]

    print(f"running LOGO: {name}", flush=True)
    frame = logo(experiments, fit, name)
    predictions = frame.attrs["predictions"]
    np.savez(LOGO / f"pred_{name}.npz", **{str(key): value for key, value in predictions.items()})
    frame.to_csv(LOGO / f"logo_{name}.csv", index=False)
    print(f"finished LOGO: {name}", flush=True)
    return frame


def combine() -> None:
    frames = [pd.read_csv(LOGO / f"logo_{name}.csv") for name in MODEL_NAMES]
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(LOGO / "logo_metrics.csv", index=False)
    table = summarize(metrics)
    table.to_csv(LOGO / "summary.csv", index=False)
    print(table.round(4).to_string(index=False))

    predictions = {}
    for name in MODEL_NAMES:
        with np.load(LOGO / f"pred_{name}.npz") as archive:
            predictions.update({f"{name}_{key}": archive[key] for key in archive.files})
    np.savez(LOGO / "logo_predictions.npz", **predictions)

    experiments = load_experiments()
    design_table(experiments).to_csv(LOGO / "design_table.csv", index=False)
    shapes = fit_experiment_shapes(experiments)
    final = FirstOrderArrayModel().fit(experiments, shape_init=shapes)
    final.save(MODELS / "model.json")
    hybrid = HybridArrayModel().fit(experiments, grey=final)
    hybrid.save(MODELS / "hybrid.json")
    relative = np.stack([experiment.relative for experiment in experiments])
    prediction = final.predict_relative(np.stack([e.concentration for e in experiments]), experiments[0].time)
    np.savez(
        LOGO / "final_predictions.npz",
        time=experiments[0].time,
        concentration=np.stack([e.concentration for e in experiments]),
        initial=np.stack([e.initial for e in experiments]),
        relative=relative,
        prediction=prediction,
        hybrid_prediction=hybrid.predict_relative(
            np.stack([e.concentration for e in experiments]), experiments[0].time
        ),
        delay=final.delay,
    )
    print("delay (s):", np.round(final.delay, 3))
    print("final model saved to", MODELS / "model.json")


def main() -> None:
    ensure()
    arguments = sys.argv[1:]
    if arguments == ["--combine"]:
        combine()
        return
    names = arguments or MODEL_NAMES
    experiments = load_experiments()
    design_table(experiments).to_csv(LOGO / "design_table.csv", index=False)
    shapes = fit_experiment_shapes(experiments)
    for name in names:
        run_one(name, experiments, shapes)
    if not arguments:
        combine()


if __name__ == "__main__":
    main()
