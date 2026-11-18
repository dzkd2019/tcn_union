"""B 实验：用正向混合模型生成合成曲线 / 循环一致性增强逆模型。

只在 50 条开发实验上做五折；12 条锁定测试实验不参与。
"""
from __future__ import annotations


import os

import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(2)

from tcn_union.cli.split import FOLDS, TEST_IDS
from tcn_union.core.model import GasTCN

from ..core.data import GASES, load_experiments
from ..core.hybrid import HybridArrayModel
from ..core.grey_box import fit_experiment_shapes
from ..paths import AUGMENTATION, ensure

TIME = np.arange(0.0, 10.0 + 0.2, 0.2)
SYNTHETIC = 800
EPOCHS = 150
STEPS_PER_EPOCH = 10
BATCH = 8
SEED = int(os.environ.get("B_SEED", "0"))
CYCLE_WEIGHT = 0.1


def _features(relative: np.ndarray) -> np.ndarray:
    return np.asarray(relative, dtype=np.float32).T


class HybridForward(nn.Module):
    """把 HybridArrayModel 的 numpy 参数冻结成一个可微的正向模块。"""

    def __init__(self, model: HybridArrayModel, time: np.ndarray) -> None:
        super().__init__()
        self.register_buffer("phi", torch.tensor(model._basis(time), dtype=torch.float32))
        self.register_buffer(
            "beta", torch.tensor(model.beta.reshape(model.n_basis, -1, 3), dtype=torch.float32)
        )
        self.register_buffer("gain", torch.tensor(model.grey.gain_coef, dtype=torch.float32))
        self.register_buffer("tau_coef", torch.tensor(model.grey.tau_coef, dtype=torch.float32))
        self.register_buffer("delay", torch.tensor(model.grey.delay, dtype=torch.float32))
        self.register_buffer("signs", torch.tensor(model.grey.signs, dtype=torch.float32))
        self.register_buffer("reference", torch.tensor(model.reference, dtype=torch.float32))
        self.register_buffer("time", torch.tensor(time, dtype=torch.float32))

    def _features(self, log_c: torch.Tensor) -> torch.Tensor:
        u = torch.clamp(log_c - torch.log(self.reference), -4.0, 4.0)
        u0, u1, u2 = u[:, 0], u[:, 1], u[:, 2]
        return torch.stack(
            [torch.ones_like(u0), u0, u1, u2, u0 * u0, u0 * u1, u0 * u2, u1 * u1, u1 * u2, u2 * u2],
            dim=1,
        )

    def forward(self, log_c: torch.Tensor) -> torch.Tensor:
        features = self._features(log_c)
        amplitude = torch.exp(features @ self.gain.T)
        tau = torch.exp(features[:, :4] @ self.tau_coef.T)
        shape = 1.0 - torch.exp(
            -torch.clamp(self.time[None, :, None] - self.delay[None, None, :], min=0.0)
            / tau[:, None, :]
        )
        grey = self.signs[None, None, :] * amplitude[:, None, :] * shape
        residual = torch.einsum("tk,kmj,bm->btj", self.phi, self.beta, features)
        return grey + residual


def _train(
    train_real: np.ndarray,
    train_targets: np.ndarray,
    synthetic: np.ndarray | None,
    synthetic_targets: np.ndarray | None,
    seed: int,
    epochs: int = EPOCHS,
    lr: float = 3e-3,
    batch_size: int = BATCH,
    model: GasTCN | None = None,
    cycle: HybridForward | None = None,
    cycle_weight: float = 0.0,
    target_mean: np.ndarray | None = None,
    target_scale: np.ndarray | None = None,
    curve_scale: np.ndarray | None = None,
    cycle_truth: np.ndarray | None = None,
) -> GasTCN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GasTCN(in_channels=3, per_gas_head=True) if model is None else model
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    real_count = len(train_real)
    synthetic_count = 0 if synthetic is None else len(synthetic)
    mean_tensor = None if target_mean is None else torch.tensor(target_mean, dtype=torch.float32)
    scale_tensor = None if target_scale is None else torch.tensor(target_scale, dtype=torch.float32)
    curve_tensor = None if curve_scale is None else torch.tensor(curve_scale, dtype=torch.float32)
    for _ in range(epochs):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            if synthetic is None:
                real_indices = rng.integers(0, real_count, batch_size)
                batch_x = torch.from_numpy(train_real[real_indices])
                batch_y = torch.from_numpy(train_targets[real_indices])
            else:
                half = batch_size // 2
                real_indices = rng.integers(0, real_count, half)
                synthetic_indices = rng.choice(synthetic_count, half, replace=False)
                batch_x = torch.from_numpy(np.concatenate([train_real[real_indices], synthetic[synthetic_indices]]))
                batch_y = torch.from_numpy(np.concatenate([train_targets[real_indices], synthetic_targets[synthetic_indices]]))
            optimizer.zero_grad()
            prediction = model(batch_x)
            loss = loss_function(prediction, batch_y)
            if cycle is not None:
                real_batch = torch.from_numpy(train_real[real_indices])
                real_prediction = prediction[: len(real_indices)]
                log_c_hat = real_prediction * scale_tensor + mean_tensor
                rendered = cycle(log_c_hat)
                # 渲染结果是 raw 相对响应，不能与标准化后的输入直接相减。
                truth = torch.from_numpy(cycle_truth[real_indices]).transpose(1, 2)
                cycle_loss = (((rendered - truth) / curve_tensor) ** 2).mean()
                loss = loss + cycle_weight * cycle_loss
            loss.backward()
            optimizer.step()
    return model


def main() -> None:
    experiments = load_experiments()
    by_id = {experiment.file_id: experiment for experiment in experiments}
    shapes = fit_experiment_shapes(experiments)
    rows = []
    dev_ids = sorted(set(range(62)) - TEST_IDS)
    for fold, held_out in enumerate(FOLDS):
        train_ids = sorted(set(dev_ids) - held_out)
        test_ids = sorted(held_out)
        train = [by_id[identifier] for identifier in train_ids]
        shape_indices = [index for index, experiment in enumerate(experiments) if experiment.file_id in train_ids]
        shape_init = tuple(part[shape_indices] for part in shapes)
        model = HybridArrayModel().fit(train, shape_init=shape_init)
        concentration = np.stack([experiment.concentration for experiment in train])
        low, high = concentration.min(axis=0), concentration.max(axis=0)
        rng = np.random.default_rng(SEED + fold)
        synthetic_concentration = np.exp(
            rng.uniform(np.log(low), np.log(high), size=(SYNTHETIC, 3))
        )
        synthetic = model.predict_relative(synthetic_concentration, TIME)
        residual = np.stack([experiment.relative[: len(TIME)] for experiment in train]) - model.predict_relative(
            concentration, TIME
        )
        synthetic = synthetic + residual[rng.integers(0, len(residual), SYNTHETIC)]

        mean = np.log(concentration).mean(axis=0)
        scale = np.log(concentration).std(axis=0)
        real_x = np.stack([_features(experiment.relative[: len(TIME)]) for experiment in train])
        # TCN 输入做标准化；循环损失必须在 raw 尺度上比较，故保留副本。
        real_x_raw = real_x.copy()
        real_y = (np.log(concentration) - mean) / scale
        synthetic_x = np.stack([_features(curve) for curve in synthetic])
        synthetic_y = (np.log(synthetic_concentration) - mean) / scale
        x_mean = real_x.mean(axis=(0, 2), keepdims=True)
        x_std = real_x.std(axis=(0, 2), keepdims=True) + 1e-8
        real_x = (real_x - x_mean) / x_std
        synthetic_x = (synthetic_x - x_mean) / x_std
        curve_scale = np.sqrt(np.mean(real_x_raw.transpose(0, 2, 1) ** 2, axis=(0, 1)))
        curve_scale = np.maximum(curve_scale, 1e-6)

        truth = np.stack([by_id[identifier].concentration for identifier in test_ids])
        test_x = np.stack([_features(by_id[identifier].relative[: len(TIME)]) for identifier in test_ids])
        test_x = (test_x - x_mean) / x_std
        cycle = HybridForward(model, TIME)
        for parameter in cycle.parameters():
            parameter.requires_grad_(False)

        arms = {
            "real": _train(real_x, real_y, None, None, SEED + fold),
            "real+synthetic": _train(real_x, real_y, synthetic_x, synthetic_y, SEED + fold),
            "real+cycle": _train(
                real_x,
                real_y,
                None,
                None,
                SEED + fold,
                cycle=cycle,
                cycle_weight=CYCLE_WEIGHT,
                target_mean=mean,
                target_scale=scale,
                curve_scale=curve_scale,
                cycle_truth=real_x_raw,
            ),
            "real+synthetic+cycle": _train(
                real_x,
                real_y,
                synthetic_x,
                synthetic_y,
                SEED + fold,
                cycle=cycle,
                cycle_weight=CYCLE_WEIGHT,
                target_mean=mean,
                target_scale=scale,
                curve_scale=curve_scale,
                cycle_truth=real_x_raw,
            ),
            "pretrain+finetune": _train(synthetic_x, synthetic_y, None, None, SEED + fold, epochs=100),
        }
        model_tcn = arms["pretrain+finetune"]
        optimizer = torch.optim.Adam(model_tcn.parameters(), lr=1e-3, weight_decay=1e-4)
        rng = np.random.default_rng(SEED + fold)
        for _ in range(60):
            model_tcn.train()
            indices = rng.integers(0, len(real_x), BATCH)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model_tcn(torch.from_numpy(real_x[indices])), torch.from_numpy(real_y[indices]))
            loss.backward()
            optimizer.step()

        for arm, model_tcn in arms.items():
            model_tcn.eval()
            with torch.no_grad():
                prediction = model_tcn(torch.from_numpy(test_x)).numpy()
            prediction = np.exp(prediction * scale + mean)
            relative = np.abs(prediction / truth - 1.0)
            absolute = np.abs(prediction - truth)
            label_range = np.ptp(concentration, axis=0)
            rows.append(
                {
                    "seed": SEED,
                    "fold": fold,
                    "arm": arm,
                    "mape_macro": float(relative.mean() * 100.0),
                    "nmae_macro": float((absolute / label_range).mean() * 100.0),
                    **{f"mape_{gas}": float(relative[:, index].mean() * 100.0) for index, gas in enumerate(GASES)},
                    **{f"nmae_{gas}": float((absolute[:, index] / label_range[index]).mean() * 100.0) for index, gas in enumerate(GASES)},
                }
            )
        print(f"fold {fold} done", flush=True)
    frame = pd.DataFrame(rows)
    ensure()
    frame.to_csv(AUGMENTATION / f"inverse_augmentation_seed{SEED}.csv", index=False, encoding="utf-8")
    if SEED == 0:
        frame.to_csv(AUGMENTATION / "inverse_augmentation.csv", index=False, encoding="utf-8")
    columns = ["mape_macro", "nmae_macro", "mape_H2", "mape_NH3", "mape_NO2"]
    print(frame.groupby("arm")[columns].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
