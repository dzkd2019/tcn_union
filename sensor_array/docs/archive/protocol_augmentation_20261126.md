# 训练协议消融：噪声、边缘复制与按气体扩充（2026-11-26）

> **已归档（2026-11-26）**：本文记录四种训练协议（syn / noise_syn / full / gas）
> 的消融结果。后续不再沿训练协议方向扩展；当前主线是
> `sensor_array/docs/combined_augmentation.md` 与
> `sensor_array/docs/forward_augmentation_20261125.md`。原始产物保存在
> `artifacts/archive/protocol_augmentation_20261126/`。

> 目的：固定“双指数骨架 + 三路独立 MLP”正模型与 TCN 反演网络，
> 只改变训练协议，比较 real / PCA-VAE / 分层随机效应 / 分层+VAE 四条臂。
> 结论以 3 seed × 2 低浓度配置、5 折 OOF、6 checkpoint 集成为准；
> 1 seed 筛选只用于决定是否值得跑全量，不作为最终结论。

## 1. 四种协议

| 协议 | 真实 | 0.1×MAD 噪声 | 合成 | 边缘复制 | 合成标签 |
|---|---:|---:|---:|---:|---|
| syn | 40 | 无 | 120 | 无 | 训练折标签 + 档位网格新组合 |
| noise_syn | 40 | 40 | 120 | 无 | 同上 |
| full | 40 | 40 | 120 | 约 125 | 同上，另加 min/max 边缘标签 |
| gas | 40 | 40 | 120 | 约 118 | H2 用新组合；NH3/NO2 只用训练折已有标签；H2 保留 min/max 边缘 |

- real 臂始终只含 40 条真实训练曲线；噪声、合成、边缘只加在增强臂。
- 合成量固定为 3× 真实（120 条）；hier+vae 为 60 条分层 + 60 条 VAE。
- 边缘复制按旧协议：每个唯一边缘标签复制 4 次；gas 协议只对 H2 做 min/max，
  NH3/NO2 从训练折已有标签抽样，不生成新的边缘组合。
- gas 是“H2 扩充、NH3/NO2 不扩充”的一种解释；它不等于“NH3/NO2 完全冻结”：
  NH3/NO2 仍会与 H2 的新组合一起出现，只是不产生新的 NH3/NO2 档位。

## 2. 实现与统计

- 代码：tcn_union/experiments/hierarchical_augmentation.py 增加 --protocol，
  输出 folds.csv、oof.csv、summary.csv、predictions.npz、run_config.json。
- predictions.npz 同时保存每个 seed 的 {arm}_s{seed} 与 6 checkpoint 的
  {arm}_ensemble，并保存 truth、file_id，便于统一复算。
- 对比脚本：tcn_union/experiments/compare_protocols.py。配对单位是
  seed×fold；同一 seed×fold 的两个低浓度配置先平均，再算配对差。
  区间为 100,000 次 bootstrap 百分位 95% CI，p 值为双侧 Wilcoxon。
- 指标：宏 3σ、宏 MAPE 与逐气体 3σ/MAPE；3σ 是实验间离散度，不是置信区间。
- 锁定 12 条未参与训练、筛选或调参；本文件所有数字均为 50 条开发集 OOF。

## 3. 1 seed 筛选（仅用于选型）

seed=20260923，config=1.0，5 折；表中为 5 折合并 ensemble 的宏 3σ / 宏 MAPE（%）。
real 在同一轮四协议中相同，因此只列一次。

| 协议 | real | hier | VAE | hier+VAE | n_train |
|---|---:|---:|---:|---:|---:|
| syn | 16.80 / 3.81 | 10.85 / 2.33 | 12.73 / 3.10 | 11.53 / 2.76 | 160 |
| noise_syn | 16.80 / 3.81 | 10.23 / 2.41 | 10.16 / 2.39 | **8.79 / 2.03** | 200 |
| full | 16.80 / 3.81 | 11.27 / 2.50 | **9.71 / 2.32** | 10.48 / 2.59 | 325 |
| gas | 16.80 / 3.81 | 15.90 / 2.80 | 11.04 / 2.44 | 11.82 / 2.51 | 318 |

1 seed 的方向：noise_syn 的 hier+VAE 最好；full 的 VAE 最好；gas 回退到接近
real；syn 介于中间。这只说明“噪声 + 合成”值得全量，不能据此下最终结论。

## 4. 全量结果（3 seed × 2 配置，6 checkpoint 集成）

### 4.1 宏平均

下表为 50 条开发集 OOF 的 6 checkpoint 集成；real 在四种协议中完全相同。

| 协议 | real 宏 3σ / MAPE | hier 宏 3σ / MAPE | PCA-VAE 宏 3σ / MAPE | hier+VAE 宏 3σ / MAPE | n_train |
|---|---:|---:|---:|---:|---:|
| syn | 11.23 / 2.94 | 9.46 / 2.38 | 10.38 / 2.71 | 9.11 / 2.28 | 160 |
| noise_syn | 11.23 / 2.94 | 8.29 / 2.11 | 9.00 / 2.33 | 8.11 / 2.05 | 200 |
| full | 11.23 / 2.94 | 9.11 / 2.21 | **7.59 / 1.89** | 7.94 / 2.08 | 约 325 |
| gas | 11.23 / 2.94 | 12.73 / 2.51 | 7.88 / 2.00 | 10.59 / 2.26 | 约 318 |

- 最低宏 3σ 是 **full + PCA-VAE 的 7.59%**，其次 gas + PCA-VAE 7.88%、
  full + hier+VAE 7.94%、noise_syn + hier+VAE 8.11%。
- full 的 VAE 比 noise_syn 的 VAE 低 1.41 百分点（配对 p=0.012），
  比 syn 的 VAE 低 2.79 百分点（p=0.0001）；full 与 gas 的 VAE 差异不显著
  （7.59 对 7.88，p=0.76）。
- 3 seed 的 checkpoint 均值（非集成）也呈同一方向：full VAE 9.89 ± 0.31，
  gas VAE 10.24 ± 1.27，noise_syn VAE 11.54 ± 1.09，syn VAE 13.38 ± 0.81。
- 集成相对单 checkpoint 的下降主要来自 seed/配置平均与裁剪；不能把集成值
  当作单模型可复现的精度。

### 4.2 分气体

| 协议 | 臂 | H2 3σ | NH3 3σ | NO2 3σ |
|---|---|---:|---:|---:|
| syn | real | 11.39 | 6.99 | 15.31 |
| syn | hier | 8.58 | 6.24 | 13.55 |
| syn | VAE | 10.11 | 6.62 | 14.41 |
| syn | hier+VAE | 8.33 | 5.81 | 13.19 |
| noise_syn | hier | 7.23 | 5.78 | 11.85 |
| noise_syn | VAE | 9.57 | 5.30 | 12.13 |
| noise_syn | hier+VAE | 7.81 | 5.53 | 10.99 |
| full | hier | 7.95 | 7.10 | 12.27 |
| full | VAE | 9.51 | 5.10 | **8.15** |
| full | hier+VAE | **7.18** | 5.31 | 11.32 |
| gas | hier | 7.76 | 13.44 | 16.98 |
| gas | VAE | 10.01 | **3.89** | 9.73 |
| gas | hier+VAE | 8.50 | 10.63 | 12.63 |

- H₂：hier 与 hier+VAE 在四种协议下都优于 VAE，最好的是 full + hier+VAE 7.18%；
  但 full 下 hier+VAE 相对 VAE 的 H₂ 配对差 −1.16 百分点并不显著（p=0.33），
  noise_syn 下 hier+VAE 的 −1.71 百分点显著（p=0.041）。
- NH₃：gas + VAE 的 3.89% 最低，但 gas 协议把 hier/hier+VAE 推到 13.44/10.63%；
  full 下三种增强臂都在 5.1–7.1%，VAE 最好。
- NO₂：full + VAE 的 8.15% 明显低于其余组合；这正是 full 协议宏平均最低的主要来源。
  full 下 hier 与 hier+VAE 分别为 12.27/11.32%，相对 VAE 显著更差
  （配对 p=0.0001/0.0084）。

### 4.3 相对 PCA-VAE 的配对差

配对单位为 seed×fold（15 对，负值表示该臂更好）。

| 协议 | hier − VAE | p | hier+VAE − VAE | p |
|---|---:|---:|---:|---:|
| syn | −1.54 | 0.0084 | −1.45 | 0.0015 |
| noise_syn | −0.71 | 0.055 | −0.87 | 0.010 |
| full | +1.33 | 0.12 | +0.71 | 0.30 |
| gas | +3.94 | 0.010 | +2.62 | 0.095 |

- 无边缘复制时（syn/noise_syn），hier 与 hier+VAE 的宏平均优于 VAE；但 syn 的 VAE
  本身只有 10.38%，这个优势不能外推到 full。
- 加边缘后（full），VAE 反而成为最好的臂；hier/hier+VAE 在 NO₂ 上被显著拉开。
- gas 协议对 hier/hier+VAE 明显不利（NH₃ 配对 +7.21，p=0.0026；NO₂ +6.42，p=0.018），
  不应把 gas 与分层生成器组合使用。

## 5. 解释

1. **边缘复制是 VAE 的 NO₂ 增益来源。** full 的 VAE 在 NO₂ 上为 8.15%，去掉边缘
   （noise_syn）后为 12.13%，去掉噪声和边缘（syn）后为 14.41%。NO₂ 的浓度范围窄
   （2–10 原始单位），但响应形状随浓度和实验漂移变化；min/max 边缘标签提供了
   VAE 在分布两端采样所需的条件，单靠 0.1×MAD 噪声不能替代。
2. **噪声对 VAE 与 hier 都有帮助，但不是充分条件。** 从 syn 到 noise_syn，VAE 宏
   3σ 从 10.38% 降到 9.00%，hier 从 9.46% 降到 8.29%；但 noise_syn 仍明显高于
   full 的 VAE 7.59%。噪声让模型见过扰动后的前缀，边缘让模型见过浓度端点，
   两者作用不同。
3. **分层随机效应主要补 H₂，不能替代 VAE 的 NO₂ 形状先验。** hier 用条件 R₀ 与
   实验间形状分布生成合成曲线，H₂ 上稳定优于 VAE；但在 full 协议下，NH₃/NO₂ 的
   边缘条件由确定性条件 R₀ 生成，合成分布偏离真实端点，混入训练后反而伤害 NO₂。
   这不是“分层模型无效”，而是“在当前生成器与边缘组合下，它的 NO₂ 合成不够真”。
4. **gas 协议保住了 NH₃ 的真实标签分布，但丢掉了 NO₂ 的边缘覆盖。** gas + VAE 的
   NH₃ 3.89% 是全部组合中最低；但 NO₂ 9.73% 仍高于 full + VAE 的 8.15%。
   对 hier/hier+VAE，H₂-only 合成与边缘使 NH₃/NO₂ 的监督信号相对不足，NH₃/NO₂
   明显退化，因此 gas 不是通用的“按气体扩充”方案。
5. **没有一种协议在三种气体上同时最好。** full + VAE 赢在宏平均与 NO₂；
   full + hier+VAE 赢在 H₂；gas + VAE 赢在 NH₃。选择取决于部署更看重哪种气体，
   不能只报一个宏平均就声称全面占优。

## 6. 建议

- **单一模型默认：full + PCA-VAE。** 宏 3σ 7.59%、MAPE 1.89%、中位 1.37%、
  p90 4.12%；分气体 9.51/5.10/8.15%。这是本轮 16 个协议×臂组合中最低的宏平均，
  也是 NO₂ 最好的组合。
- **H₂ 优先：full + hier+VAE。** H₂ 7.18%，宏 7.94%；代价是 NO₂ 11.32%。
  若只关心 H₂ 且能接受 NO₂ 退化，可作专用方案。
- **NH₃ 优先：gas + VAE。** NH₃ 3.89%，宏 7.88%；代价是 NO₂ 9.73% 与 H₂ 10.01%。
  不建议与 hier/hier+VAE 组合。
- **不建议把 gas + hier/hier+VAE 作为候选**：NH₃ 和 NO₂ 都显著差于同协议 VAE。
- 以上选择均基于 50 条开发集 OOF；锁定 12 条尚未评估，部署前应按 AGENTS.md §10
  做一次锁定集描述性评估，不得用锁定集回头调协议。

## 7. 限制

1. 15 对（3 seed × 5 折）不是独立样本；同一折的曲线重叠，bootstrap CI 与 Wilcoxon
   p 值只能作描述性不确定性，不是独立重复实验的推断。
2. 只跑 3 seed × 2 低浓度配置，未做超参数搜索；不同 seed 的排序可能变化。
3. real 臂在四种协议中完全相同（同一预测），因此 real 的配对差恒为 0；
   协议效应只在增强臂上可识别。
4. 6 checkpoint 集成先平均后裁剪；集成值高于单 checkpoint 均值，不能当作单模型
   部署精度。
5. 未做按气体路由；第 6 节的“H₂ 优先/NO₂ 优先”是描述性建议，不是已经验证的路由器。
6. 未使用锁定 12 条；所有数字都是开发集 OOF，存在开发集选择偏差。
7. 相对误差的 3σ 对尾部实验敏感；NO₂ 的结论主要由少数高浓度/边缘实验驱动。

## 8. 产物与复现

- 全量运行目录：
  - `artifacts/archive/protocol_augmentation_20261126/fullrun_syn/`
  - `artifacts/archive/protocol_augmentation_20261126/fullrun_noise_syn/`
  - `artifacts/archive/protocol_augmentation_20261126/fullrun_full/`
  - `artifacts/archive/protocol_augmentation_20261126/fullrun_gas/`
- 四协议对比：`artifacts/archive/protocol_augmentation_20261126/fullrun_comparison/`
  （ensemble_metrics.csv、seed_metrics.csv、seed_summary.csv、
  paired_within_protocol.csv、protocol_effect.csv、figs/）。
- 最终候选 full + VAE 的标准评估：
  `artifacts/archive/protocol_augmentation_20261126/fullrun_full/report/vae/`
  （metrics.csv、predictions.csv、run_config.json、figs/fig_abs_error、
  fig_rel_error、fig_ci95，各 PNG+SVG）。
- 图：
  - `fullrun_comparison/figs/fig_protocol_3sigma.png/.svg`
  - `fullrun_comparison/figs/fig_protocol_mape.png/.svg`
  - `fullrun_comparison/figs/fig_protocol_delta_3sigma.png/.svg`

```bash
# 每个协议单独跑；示例为 full，其他协议替换 --protocol 与 --output
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python \
  -m tcn_union.experiments.hierarchical_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 \
  --configs 1.0 0,0.3,1.0 \
  --arms real vae hier hier+vae --protocol full \
  --output artifacts/archive/protocol_augmentation_20261126/fullrun_full

# 四协议合并与图
.venv/bin/python -m tcn_union.experiments.compare_protocols \
  --root artifacts/augmentation/hierarchical --prefix fullrun_ \
  --output artifacts/archive/protocol_augmentation_20261126/fullrun_comparison
MPLCONFIGDIR=/tmp/mpl .venv/bin/python \
  -m tcn_union.visualization.protocol_augmentation \
  --comparison artifacts/archive/protocol_augmentation_20261126/fullrun_comparison

# 最终候选标准图
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.final_report \
  --source artifacts/archive/protocol_augmentation_20261126/fullrun_full/predictions.npz \
  --output artifacts/archive/protocol_augmentation_20261126/fullrun_full/report/vae \
  --arm vae --run-name fullrun_full_vae
```

## 9. 一句话

四种协议里，**full + PCA-VAE 仍是宏平均与 NO₂ 最好的组合（7.59%）**；
分层随机效应只在 H₂ 上有稳定优势，gas 协议只在 NH₃ 上局部占优；
当前数据不支持“用分层生成器全面替代 VAE”。

