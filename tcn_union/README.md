# tcn_union：浓度反演与数据增强

本子项目用因果 TCN 从 0–10 s 早期响应反演 H₂、NH₃、NO₂ 三种气体浓度，并比较三种
保留的数据增强方式：**PCA-VAE**、**传感器模型生成**、**组合使用**。

传感器正问题（响应曲线建模）在 `sensor_array/`，论文口径与必报指标见根目录
`AGENTS.md` §10–§11。

## 数据与划分

- 输入：`data/0.csv`–`data/61.csv`，共 62 次实验，传感器已损坏，不能补测。
- 开发集 50 条用于 5 折实验级 OOF；锁定集 12 条只在最终候选上评估一次。
- 每个样本只取 0–10 s 前缀；同一实验的多个时间点不当作独立样本。
- 固定划分与清单：`artifacts/data/splits/`。

## 模型

`tcn_union/core/model.py` 是因果卷积 TCN：所有卷积只在左侧补零，读取前缀最后
时间点的隐藏表示，输出三气体标准化浓度。隐藏通道固定 16，膨胀率 1/2/4/8，
两层核长 5，残差块外接 dropout 与输出头。

`tcn_union/training/train.py` 提供数据校验、z-score、训练循环与 checkpoint 保存；
`tcn_union/training/gan_train.py` 是当前完整训练入口（文件名保留历史名称，内部只
保留 PCA-VAE，不再包含 GAN）。

## 三种增强方式

### PCA-VAE

对每气体 10 s 响应拟合 K=6 的 PCA 基，用标签线性回归给出条件均值，再训练条件 VAE
学习 PCA 系数残差；采样后反变换为曲线并裁剪到训练折范围。方法细节与公式见
[`docs/PCA-VAE.md`](../docs/PCA-VAE.md)。

### 传感器模型生成

用传感器正问题模型生成三气体响应曲线，取前 10 个主成分作为合成样本，与真实样本
一起训练 TCN。它不混入 PCA-VAE，是单独用传感器模型生成数据的路线。

这里的正模型在项目中统一称为 **hierarchical 正模型**：实现为双指数骨架 + 三路独立
MLP 残差模型，合成时在该模型均值上叠加三气体联合低秩残差采样（`forward_prob_joint10`）。
它不是早期 `HierarchicalRandomEffectsModel`（实验级增益 / 时间尺度 / 漂移）版本。

### 组合使用

把 PCA-VAE 与传感器模型生成的数据结合使用，两者各自对 H₂、NH₃、NO₂ 三种气体
生成数据。训练时先得到 VAE 基线，再冻结共享骨干与 NH₃/NO₂ 输出头，只更新 H₂
输出头，用传感器模型样本改善 H₂，同时保持 NH₃/NO₂ 不变。它不是“只增强 NO₂”，
而是保护 NO₂ 不再变差的训练策略。

三种方式都使用完整协议：多个 seed、两套低浓度配置、每臂 120 条合成曲线、
0.1×MAD 噪声视图、边缘复制 ×4、6 checkpoint 平均。多个 seed 训练后取平均预测，
不把不同 seed 当作重复实验。

## 复现

所有命令从仓库根目录运行，使用 `.venv/bin/python`：

```bash
# PCA-VAE 与仅真实基线
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae \
  --output artifacts/augmentation/protocols/full

# 传感器模型生成
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae forward_prob_joint10 \
  --output artifacts/augmentation/pure_forward/joint10

# 组合使用
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.no2_guard \
  --folds 5 --seeds 20260923,20260924,20260925 \
  --configs '1.0;0,0.3,1.0' \
  --output artifacts/augmentation/no2_guard/full

# 三方法统一汇总、指标与 fig_abs_error / fig_rel_error / fig_ci95
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.compare_augmentation
```

单次运行耗时约 1 小时（CUDA）。`compare_augmentation` 只读三个预测文件，不重新
训练；输出在 `artifacts/augmentation/comparison_20261127/`。

## 评估与图件

`tcn_union/evaluation/final_report.py` 按 AGENTS.md §10 生成：

- `fig_abs_error.{png,svg}`：1×3 有符号绝对误差散点；
- `fig_rel_error.{png,svg}`：1×3 有符号相对误差散点；
- `fig_ci95.{png,svg}`：单张图，三气体均值与 95% CI，不区分 seed。

指标包括逐气体 MAPE、NRMSE（注明量程来源）、3σ、95% CI 与 `n`，以及三气体宏平均。
锁定 12 条只做描述性统计，不用于选择方法、seed 或阈值。

单独生成某次运行的图：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.final_report \
  --source artifacts/augmentation/comparison_20261127/predictions.npz \
  --arm guard --output artifacts/augmentation/comparison_20261127/guard/report \
  --run-name comparison --label "组合使用" --window 10
```

## 目录

```text
tcn_union/
├── core/            # 数据读取、TCN 模型、设备选择
├── augmentation/    # PCA-VAE 与测量噪声
├── training/        # 训练循环与完整训练入口
├── evaluation/      # 锁定测试评估、PCA 证据、三张必报图
├── experiments/     # 三种增强、组合使用分析、统一对比
├── visualization/   # 结构图、PCA K 选择、PCA-VAE 讲解图
└── cli/             # 数据划分与单条预测
```

## 当前产物

| 内容 | 路径 |
|---|---|
| PCA-VAE / 仅真实 | `artifacts/augmentation/protocols/full/` |
| 传感器模型生成 | `artifacts/augmentation/pure_forward/joint10/` |
| 组合使用 | `artifacts/augmentation/no2_guard/full/` |
| 三方法对比与图件 | `artifacts/augmentation/comparison_20261127/` |

每个运行目录包含 `folds.csv`、`oof.csv`、`predictions.npz` 与 `run_config.json`；
full 与传感器模型生成目录另含 `summary.csv`、`summary.md`；组合使用由
`tcn_union.experiments.analyze_no2_guard` 生成 `seed_mean.csv`，其中 H₂ 为组合分支、
NH₃/NO₂ 对齐 PCA-VAE 的多个 seed 平均预测。
目录约定见 [`artifacts/README.md`](../artifacts/README.md)。

## 限制

- 50 条开发实验、每个浓度组合 1 条；所有结论只适用于当前 62 条数据与 10 s 前缀。
- 增强只能覆盖训练分布附近，不能外推到未采样的浓度组合。
- 锁定集样本少，区间与 3σ 只作描述，不应当作泛化保证。
- 不补测、不重新采集数据；任何方案都只能基于现有 62 条实验。
