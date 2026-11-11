"""把无 Ridge 方案两套配置的 OOF 预测合成为逐 seed 的 OOF 文件。

输入（由 §7 复现命令生成）：
  artifacts/3sigma_noridge/oof_raw/p10/seed_<seed>/noise_gan/oof.csv
  artifacts/3sigma_noridge/oof_raw/q/seed_<seed>/noise_gan/oof.csv
输出：
  artifacts/3sigma_noridge/oof/seed_<seed>/noise_gan/oof.csv
（每行 = 两配置预测平均；后续 3 seed 平均 + 裁剪由评估/绘图脚本完成。）
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "artifacts/3sigma_noridge"
GASES = ("H2", "NH3", "NO2")
SEEDS = (20260923, 20260924, 20260925)


def main() -> None:
    for seed in SEEDS:
        p10 = pd.read_csv(
            BASE / f"oof_raw/p10/seed_{seed}/noise_gan/oof.csv"
        ).set_index("file")
        q = pd.read_csv(
            BASE / f"oof_raw/q/seed_{seed}/noise_gan/oof.csv"
        ).set_index("file")
        frame = p10.copy()
        for gas in GASES:
            frame[f"{gas}_pred"] = (
                p10[f"{gas}_pred"].to_numpy(float) + q[f"{gas}_pred"].to_numpy(float)
            ) / 2.0
        path = BASE / f"oof/seed_{seed}/noise_gan/oof.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.reset_index().to_csv(path, index=False)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
