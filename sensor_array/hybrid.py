"""混合灰盒模型：一阶惯性骨架 + 低秩时空残差。

对每路传感器 j，把相对响应写成

    y_j(t,c) = y_j^grey(t,c) + r_j(t,c)
    r_j(t,c) = sum_{k=1}^{K} phi_k(t) * (X(c) @ beta_{jk})

其中 y^grey 是 ``FirstOrderArrayModel`` 的一阶惯性预测，phi_k 是固定的
B 样条时间基，X(c) 是 log 浓度的二次多项式特征。时间基固定后，模型对
beta 是线性的，可以用一次岭回归解出，不需要交替优化。

这样做的动机：一阶模型已经把 62 条曲线拟合到约 3.5%，残差主要是 20 s
以后的缓慢漂移和少量起始段形状偏差（见 README §5.2）。低秩残差用少量
时间基描述这些系统偏差，同时保留 A/tau 的可解释性。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import BSpline

from .data import REFERENCE_CONCENTRATION, Experiment, load_experiments
from .evaluate import curve_metrics
from .model import FirstOrderArrayModel, _monomial_features, fit_experiment_shapes

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def bspline_basis(
    time: np.ndarray,
    n_basis: int,
    degree: int,
    t_max: float,
) -> np.ndarray:
    """构造在 t=0 处为零的钳位 B 样条基，返回 (T, n_basis)。

    先构造 n_basis+1 个标准钳位基，再丢掉第一个。由于钳位基满足
    B_0(0)=1、B_k(0)=0 (k>=1)，丢掉 B_0 后所有剩余基在 t=0 都为零，
    因此残差在起点严格为零，不需要额外的锚定项。
    """
    total = n_basis + 1
    degree = min(degree, total - 1)
    n_interior = total - degree - 1
    interior = np.linspace(0.0, t_max, n_interior + 2)[1:-1]
    knots = np.r_[
        np.zeros(degree + 1),
        interior,
        np.full(degree + 1, t_max),
    ]
    clipped = np.clip(np.asarray(time, dtype=float), 0.0, t_max)
    return BSpline.design_matrix(clipped, knots, degree).toarray()[:, 1:]


def _ridge_solve(features: np.ndarray, target: np.ndarray, ridge: float) -> np.ndarray:
    """解带岭惩罚的最小二乘；所有列都惩罚（时间基已约束了形状）。"""
    penalty = np.sqrt(ridge) * np.eye(features.shape[1])
    return np.linalg.lstsq(
        np.vstack([features, penalty]),
        np.r_[target, np.zeros(features.shape[1])],
        rcond=None,
    )[0]


class HybridArrayModel:
    """一阶灰盒 + B 样条时间基 × 浓度多项式的低秩残差。"""

    def __init__(
        self,
        n_basis: int = 5,
        concentration_degree: int = 2,
        ridge: float = 1e-2,
    ) -> None:
        self.n_basis = n_basis
        self.concentration_degree = concentration_degree
        self.ridge = ridge
        self.spline_degree = min(2, n_basis)
        self.reference = REFERENCE_CONCENTRATION
        self.t_max: float | None = None
        self.beta: np.ndarray | None = None
        self.grey: FirstOrderArrayModel | None = None

    def _basis(self, time: np.ndarray) -> np.ndarray:
        if self.t_max is None:
            raise RuntimeError("模型尚未拟合")
        return bspline_basis(time, self.n_basis, self.spline_degree, self.t_max)

    def _features(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        """返回 (N*T, K*M) 的时空特征矩阵，行顺序与 (N,T) 展平一致。"""
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        u = np.log(concentration / self.reference)
        concentration_features = _monomial_features(u, self.concentration_degree)
        phi = self._basis(np.asarray(time, dtype=float))
        return np.einsum("tk,nm->ntkm", phi, concentration_features).reshape(
            concentration.shape[0] * len(time), -1
        )

    def fit(
        self,
        experiments: list[Experiment],
        shape_init: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
        grey: FirstOrderArrayModel | None = None,
    ) -> HybridArrayModel:
        """先拟合一阶骨架，再在训练残差上做一次岭回归。"""
        time = experiments[0].time
        relative = np.stack([experiment.relative for experiment in experiments])
        concentration = np.stack([experiment.concentration for experiment in experiments])
        self.t_max = float(time[-1])

        if grey is None:
            grey = FirstOrderArrayModel().fit(experiments, shape_init=shape_init)
        self.grey = grey
        residual = relative - grey.predict_relative(concentration, time)
        features = self._features(concentration, time)

        beta = np.empty((features.shape[1], 3))
        for j in range(3):
            weight = 1.0 / max(float(np.mean(relative[:, :, j] ** 2)), 1e-12)
            beta[:, j] = _ridge_solve(
                features * np.sqrt(weight),
                residual[:, :, j].ravel() * np.sqrt(weight),
                self.ridge,
            )
        self.beta = beta
        return self

    def predict_relative(self, concentration: np.ndarray, time: np.ndarray) -> np.ndarray:
        """预测相对响应，形状 (N,T,3)；NO2 通道为负。"""
        if self.grey is None or self.beta is None:
            raise RuntimeError("模型尚未拟合")
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        time = np.asarray(time, dtype=float)
        grey = self.grey.predict_relative(concentration, time)
        features = self._features(concentration, time)
        residual = (features @ self.beta).reshape(concentration.shape[0], len(time), 3)
        return grey + residual

    def predict_response(
        self,
        concentration: np.ndarray,
        time: np.ndarray,
        initial: np.ndarray,
    ) -> np.ndarray:
        concentration = np.atleast_2d(np.asarray(concentration, dtype=float))
        initial = np.asarray(initial, dtype=float)
        if initial.ndim == 1:
            initial = initial[None, :]
        return initial[:, None, :] * (1.0 + self.predict_relative(concentration, time))

    def gain(self, concentration: np.ndarray) -> np.ndarray:
        if self.grey is None:
            raise RuntimeError("模型尚未拟合")
        return self.grey.gain(concentration)

    def tau(self, concentration: np.ndarray) -> np.ndarray:
        if self.grey is None:
            raise RuntimeError("模型尚未拟合")
        return self.grey.tau(concentration)

    def to_dict(self) -> dict:
        if self.grey is None or self.beta is None:
            raise RuntimeError("模型尚未拟合")
        return {
            "n_basis": self.n_basis,
            "concentration_degree": self.concentration_degree,
            "ridge": self.ridge,
            "t_max": self.t_max,
            "beta": self.beta.tolist(),
            "grey": self.grey.to_dict(),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: dict) -> HybridArrayModel:
        model = cls(
            n_basis=payload["n_basis"],
            concentration_degree=payload["concentration_degree"],
            ridge=payload["ridge"],
        )
        model.t_max = payload["t_max"]
        model.beta = np.array(payload["beta"])
        model.grey = FirstOrderArrayModel.from_dict(payload["grey"])
        return model

    @classmethod
    def load(cls, path: Path | str) -> HybridArrayModel:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _subset_shapes(experiments: list[Experiment], shapes, indices: list[int]):
    amplitude, tau, delay = shapes
    return tuple(part[indices] for part in (amplitude, tau, delay))


def sensitivity_table(experiments: list[Experiment]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """用同一批留一折比较不同 K；每折的一阶骨架只拟合一次。"""
    shapes = fit_experiment_shapes(experiments)
    basis_sizes = (1, 2, 3, 4, 5, 6)
    rows = []
    for index, test in enumerate(experiments):
        train_indices = [i for i in range(len(experiments)) if i != index]
        train = [experiments[i] for i in train_indices]
        grey = FirstOrderArrayModel().fit(
            train, shape_init=_subset_shapes(experiments, shapes, train_indices)
        )
        scale = np.sqrt(np.mean(np.stack([e.relative for e in train]) ** 2, axis=(0, 1)))
        for n_basis in basis_sizes:
            model = HybridArrayModel(n_basis=n_basis).fit(train, grey=grey)
            prediction = model.predict_relative(test.concentration[None], test.time)[0]
            metrics = curve_metrics(prediction, test.relative, scale)
            rows.append({"file": test.file_id, "n_basis": n_basis, **metrics})
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby("n_basis")[["nrmse_H2", "nrmse_NH3", "nrmse_NO2", "nrmse_macro"]]
        .agg(["mean", "median"])
        .reset_index()
    )
    summary.columns = ["n_basis"] + [
        f"{column[0]}_{column[1]}" for column in summary.columns[1:]
    ]
    return frame, summary


def main() -> None:
    experiments = load_experiments()
    shapes = fit_experiment_shapes(experiments)
    grey = FirstOrderArrayModel().fit(experiments, shape_init=shapes)
    model = HybridArrayModel().fit(experiments, grey=grey)
    ARTIFACTS.mkdir(exist_ok=True)
    model.save(ARTIFACTS / "hybrid.json")

    concentration = np.stack([e.concentration for e in experiments])
    prediction = model.predict_relative(concentration, experiments[0].time)
    relative = np.stack([e.relative for e in experiments])
    scale = np.sqrt(np.mean(relative**2, axis=(0, 1)))
    metrics = [curve_metrics(prediction[i], relative[i], scale) for i in range(len(experiments))]
    in_sample = {
        key: float(np.mean([row[key] for row in metrics]))
        for key in ("nrmse_H2", "nrmse_NH3", "nrmse_NO2", "nrmse_macro")
    }
    print("in-sample:", {key: round(value, 5) for key, value in in_sample.items()})

    frame, summary = sensitivity_table(experiments)
    frame.to_csv(ARTIFACTS / "hybrid_sensitivity.csv", index=False, encoding="utf-8")
    print(summary.round(4).to_string(index=False))
    print(f"写入 {ARTIFACTS / 'hybrid.json'} 与 {ARTIFACTS / 'hybrid_sensitivity.csv'}")


if __name__ == "__main__":
    main()
