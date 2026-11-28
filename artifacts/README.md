# artifacts 目录说明

本目录保存当前三种数据增强方式的**可复现产物**。

## 目录结构

```text
artifacts/
├── data/
│   ├── splits/                  # 50 条开发集 / 12 条锁定集划分与清单
│   └── concentration/           # 浓度组合覆盖与分布
├── augmentation/
│   ├── protocols/full/          # PCA-VAE 与仅真实（real/vae）
│   ├── pure_forward/joint10/    # 传感器模型生成（real/vae/forward_prob_joint10）
│   ├── no2_guard/full/          # 组合使用
│   └── comparison_20261127/     # 三种方法统一汇总与图件
└── figures/
    ├── architecture/            # TCN 结构图
    └── pca_vae/                 # PCA-VAE 方法讲解图
```

正问题模型与评估产物在 `sensor_array/artifacts/`，不在本目录。

## 当前入口

| 用途 | 路径 |
|---|---|
| 固定划分 | `data/splits/manifest.csv` |
| 浓度覆盖 | `data/concentration/` |
| PCA-VAE / 仅真实 OOF | `augmentation/protocols/full/summary.csv` |
| 传感器模型生成 OOF | `augmentation/pure_forward/joint10/summary.csv` |
| 组合使用 OOF | `augmentation/no2_guard/full/seed_mean.csv` |
| 三种方法统一对比 | `augmentation/comparison_20261127/summary.csv` |
| PCA-VAE 讲解图 | `figures/pca_vae/01–06` |
| TCN 结构图 | `figures/architecture/tcn_architecture.png` |

每次增强运行保存：

- `folds.csv`：逐折、逐 seed、逐配置指标；
- `oof.csv`：折外预测的逐实验指标；
- `predictions.npz`：`truth`、`file_id`、`<arm>_s<seed>`、`<arm>_seed_mean`；
- `summary.csv` / `summary.md`：使用多个 seed 训练取平均后的指标（full / joint10）；
- `seed_mean.csv`：组合使用按多个 seed 训练取平均后的指标；
- `run_config.json`：折数、seed、窗口、增强参数与运行时间。

`comparison_20261127/` 只做汇总，不重新训练：从上述三个目录读取预测，按
AGENTS.md §10 生成 PCA-VAE、传感器模型生成、组合使用三套 `metrics.csv`、预测与图件。

## 口径

- 三种增强方式：PCA-VAE、传感器模型生成、组合使用。
- 组合使用 = 传感器模型生成与 PCA-VAE 结合，各自对三种气体生成数据；训练时冻结
  共享骨干与 NH₃/NO₂ 输出头，只更新 H₂ 输出头。
- 不保存旧协议、旧 GAN、分层随机效应、路由组合等已被替代的产物。
- 图件只保留 PNG + SVG，不输出 PDF。