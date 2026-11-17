"""B 实验：用正向混合模型生成合成曲线，检验能否增强逆模型。

只在 50 条开发实验上做五折；12 条锁定测试实验不参与。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

torch.set_num_threads(2)

from tcn_union.cli.split import FOLDS, TEST_IDS
from tcn_union.core.model import GasTCN

from .data import GASES, load_experiments
from .hybrid import HybridArrayModel
from .model import fit_experiment_shapes

ARTIFACTS = __import__("pathlib").Path(__file__).resolve().parent / "artifacts"
TIME = np.arange(0.0, 10.0 + 0.2, 0.2)
SYNTHETIC = 800
EPOCHS = 150
STEPS_PER_EPOCH = 10
BATCH = 8
SEED = 0


def _features(relative: np.ndarray) -> np.ndarray:
    return np.asarray(relative, dtype=np.float32).T


def _train(
    train_real: np.ndarray,
    train_targets: np.ndarray,
    synthetic: np.ndarray | None,
    synthetic_targets: np.ndarray | None,
    seed: int,
    epochs: int = EPOCHS,
    lr: float = 3e-3,
) -> GasTCN:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GasTCN(in_channels=3, per_gas_head=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_function = nn.MSELoss()
    real_count = len(train_real)
    synthetic_count = 0 if synthetic is None else len(synthetic)
    for _ in range(epochs):
        model.train()
        for _ in range(STEPS_PER_EPOCH):
            if synthetic is None:
                indices = rng.integers(0, real_count, BATCH)
                batch_x = torch.from_numpy(train_real[indices])
                batch_y = torch.from_numpy(train_targets[indices])
            else:
                half = BATCH // 2
                real_indices = rng.integers(0, real_count, half)
                synthetic_indices = rng.choice(synthetic_count, half, replace=False)
                batch_x = torch.from_numpy(np.concatenate([train_real[real_indices], synthetic[synthetic_indices]]))
                batch_y = torch.from_numpy(np.concatenate([train_targets[real_indices], synthetic_targets[synthetic_indices]]))
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
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
        real_y = (np.log(concentration) - mean) / scale
        synthetic_x = np.stack([_features(curve) for curve in synthetic])
        synthetic_y = (np.log(synthetic_concentration) - mean) / scale

        truth = np.stack([by_id[identifier].concentration for identifier in test_ids])
        test_x = np.stack([_features(by_id[identifier].relative[: len(TIME)]) for identifier in test_ids])
        arms = {
            "real": _train(real_x, real_y, None, None, SEED + fold),
            "real+synthetic": _train(real_x, real_y, synthetic_x, synthetic_y, SEED + fold),
            "pretrain+finetune": _train(synthetic_x, synthetic_y, None, None, SEED + fold, epochs=100),
        }
        if "pretrain+finetune" in arms:
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
            error = np.abs(prediction / truth - 1.0)
            rows.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "nmae_macro": float(error.mean() * 100.0),
                    **{f"nmae_{gas}": float(error[:, index].mean() * 100.0) for index, gas in enumerate(GASES)},
                }
            )
        print(f"fold {fold} done", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(ARTIFACTS / "inverse_augmentation.csv", index=False, encoding="utf-8")
    print(frame.groupby("arm")[["nmae_macro", "nmae_H2", "nmae_NH3", "nmae_NO2"]].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
