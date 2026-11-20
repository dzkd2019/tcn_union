"""用已有 OOF selection 的 epochs，重训 5 秒前缀最终模型（3 seeds × 2 配置）。

不重复跑五折 OOF；epochs 取自 ``oof_raw/<config>/seed_20260923/selection.json``。
输出：``final/<config>/noise_gan/5s/seed_<seed>.pt``。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tcn_union.augmentation.direct_vae import DirectVAEConfig
from tcn_union.augmentation.gan import GANConfig
from tcn_union.augmentation.noise import NoiseConfig
from tcn_union.augmentation.pca_generator import PCAConfig
from tcn_union.augmentation.vae import VAEConfig
from tcn_union.training.gan_train import (
    BASE_SEED,
    MODE,
    _read_manifest,
    configure_threads,
    resolve_device,
    train_final,
)

BASE = ROOT / "artifacts/3sigma_5s"
SEEDS = (20260923, 20260924, 20260925)
CONFIGS = {
    "p10": {"low_conc_power": "1.0"},
    "q": {"low_conc_power": "0,0.3,1.0"},
}
SECONDS = 5.0


def main() -> None:
    device = resolve_device("cuda")
    configure_threads(device)
    rows = _read_manifest(ROOT / "artifacts/splits/manifest.csv")
    dev_rows = [row for row in rows if row["split"] == "development"]
    for config_name, config in CONFIGS.items():
        selection_path = BASE / f"oof_raw/{config_name}/seed_{BASE_SEED}/selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        epochs = int(selection["selected"]["noise_gan"]["selected_epochs"])
        print(f"[{config_name}] epochs={epochs}", flush=True)
        gan_config = GANConfig(seconds=SECONDS, epochs=300, residual=False, anchor_endpoints=False)
        vae_config = VAEConfig(seconds=SECONDS, epochs=300, components=6, latent_dim=4, beta=0.1)
        pca_config = PCAConfig(seconds=SECONDS, components=6)
        direct_vae_config = DirectVAEConfig(seconds=SECONDS, epochs=300, latent_dim=4, beta=0.1)
        noise_config = NoiseConfig(scale=0.1, copies=1)
        for seed in SEEDS:
            path = BASE / f"final/{config_name}/noise_gan/5s/seed_{seed}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            train_final(
                dev_rows,
                "noise_gan",
                epochs,
                seed,
                device,
                path,
                gan_config,
                vae_config,
                pca_config,
                direct_vae_config,
                "vae",
                3.0,
                noise_config,
                None,
                4,
                target_transform="log,none,log",
                feature_mode=MODE,
                per_gas_head=True,
                low_conc_power=config["low_conc_power"],
                seconds=SECONDS,
            )
            print(f"saved {path}", flush=True)


if __name__ == "__main__":
    main()
