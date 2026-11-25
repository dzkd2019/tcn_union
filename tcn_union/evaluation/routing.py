"""按气体路由正模型增强臂与 PCA-VAE 臂，生成可复现的最终预测。

规则只依赖开发集 OOF：H2 用 forward+vae，NH3/NO2 用 vae。
锁定集不参与任何选择。

    .venv/bin/python -m tcn_union.evaluation.routing \
        --source artifacts/full_protocol_augmentation/predictions.npz \
        --output artifacts/routing_augmentation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tcn_union.evaluation.ensemble_evaluate import _clip_to_development_range
from tcn_union.experiments.full_protocol_augmentation import _metric_values

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "artifacts/full_protocol_augmentation/predictions.npz"
OUTPUT = ROOT / "artifacts/routing_augmentation"
ROUTE = ("forward+vae", "vae", "vae")


def run(source: Path, output: Path) -> None:
    payload = np.load(source)
    truth = payload["truth"].astype(np.float64)
    file_id = payload["file_id"].astype(int)
    seeds = sorted(
        key[len("vae_s"):] for key in payload.files if key.startswith("vae_s")
    )
    if not seeds:
        raise ValueError("没有找到逐 seed 预测")

    routed = {}
    for seed in seeds:
        prediction = np.empty_like(truth)
        for gas_index, arm in enumerate(ROUTE):
            prediction[:, gas_index] = payload[f"{arm}_s{seed}"][:, gas_index]
        routed[f"routing_s{seed}"] = _clip_to_development_range(prediction, truth)

    average = {
        arm: np.mean(
            [payload[f"{arm}_s{seed}"] for seed in seeds], axis=0
        )
        for arm in set(ROUTE)
    }
    ensemble = np.empty_like(truth)
    for gas_index, arm in enumerate(ROUTE):
        ensemble[:, gas_index] = average[arm][:, gas_index]
    routed["routing_ensemble"] = _clip_to_development_range(ensemble, truth)

    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "predictions.npz", truth=truth, file_id=file_id, **routed)
    config = {
        "source": str(source),
        "route": {f"gas_{index}": arm for index, arm in enumerate(ROUTE)},
        "seeds": seeds,
        "clip": "development_50",
        "n": int(len(truth)),
    }
    (output / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for name, prediction in routed.items():
        metrics = _metric_values(prediction, truth)
        print(
            name,
            "3sigma",
            round(metrics["macro_three_sigma"], 3),
            "MAPE",
            round(metrics["macro_mape"], 3),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=str(SOURCE))
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args()
    run(Path(args.source), Path(args.output))


if __name__ == "__main__":
    main()
