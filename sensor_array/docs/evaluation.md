# 62 折留一评估：双指数骨架与三路独立 MLP 残差

## 1. 结论

- 62 折 LOGO 下，曲线宏 nRMSE：双指数骨架 **3.30%**，三路独立 MLP 残差组合 **1.48%**。
- 逐气体（H₂ / NH₃ / NO₂）：骨架 5.17% / 1.69% / 3.04%，组合 2.02% / 1.01% / 1.40%。
- 三路独立组合在 **59/62** 折改善宏 nRMSE，逐文件配对中位改善 **−1.80 百分点**。
- 当前没有独立锁定测试；MAP 浓度反演与区间校准已从本分支删除。
- 正模型增强 TCN 首轮（3 seed × 5 折、简化协议）没有稳定优于仅真实：10 s
  `real` 宏 3σ/MAPE 15.88%/4.11%，`pca_vae` 16.67%/4.41%，正模型与混合臂
  更差；在该简化协议内不采用合成增强（§8）。
- 补做的**旧训练协议增强来源对照**（等 120 条合成量、不含噪声/边缘）显示：
  仅真实 15.47%、PCA-VAE only 14.22%（p=0.804）、正模型 only 16.93%（p=0.107）、
  正模型 + PCA-VAE **11.40%**（14/15，p=0.00018）。单一来源不足，组合才稳定；
  完整旧协议（噪声/边缘/两配置/6 checkpoint/裁剪）已在 §8.8 复刻。
- 进一步补做**完整旧协议**（0.1×噪声 + 边缘 ×4 + 两套低浓度配置 + 6 checkpoint + 裁剪）：
  仅真实 10.79%、PCA-VAE only **7.47%**（14/15，p=0.00012，MAPE 1.79%）、
  正模型 only 10.92%（p=0.524）、正模型 + PCA-VAE 9.10%（14/15，p=0.0084）。
  VAE-only 与旧最终 7.47%/1.79% 对齐；正模型没有增益，混合反而稀释 VAE（§8.8）。

## 2. 协议

- 每次留出 1 条完整实验，用其余 61 条训练；共 62 折。
- 每折先拟合双指数骨架，再用**同一训练折**拟合三路独立 MLP；两模型共享训练数据。
- 骨架用逐实验形状初值 + 加权最小二乘联合拟合；MLP 用骨架输出作为输入，不接触测试折。
- 曲线误差：
  \[
  \mathrm{nRMSE}_{j}=\frac{\sqrt{\frac1T\sum_t(\hat y_j(t)-y_j(t))^2}}
  {\sqrt{\frac1{NT}\sum_{i,t}y_{ij}(t)^2}},
  \]
  分母是训练折该通道相对响应的 RMS；宏平均为三路算术平均。
- 预测数组、逐文件指标和汇总表分别写入：
  `artifacts/evaluation/predictions.npz`、`curve_metrics.csv`、`curve_summary.csv`。

## 3. 曲线结果

| 模型 | H₂ nRMSE | NH₃ nRMSE | NO₂ nRMSE | 宏 nRMSE |
|---|---:|---:|---:|---:|
| 双指数骨架 | 5.17% | 1.69% | 3.04% | 3.30% |
| 三路独立 MLP + 双指数 | 2.02% | 1.01% | 1.40% | 1.48% |

逐文件分布：

| 模型 | 中位 | p90 | 标准差 |
|---|---:|---:|---:|
| 骨架宏 nRMSE | 3.20% | 4.59% | 1.54% |
| 组合宏 nRMSE | 1.17% | 2.30% | 1.13% |

三路独立相对骨架的逐折差值：中位 −1.80 百分点，59/62 折改善。

### 3.1 逐文件误差的离散度

图 15 与 <code>artifacts/evaluation/curve_error_stats.csv</code> 把每个文件的曲线误差
按“62 条实验合并的逐气体尺度”归一化，用来看 62 条实验之间的离散度；
它与上表的均值相差约 0.01 百分点，因为上表分母是每折训练折尺度。
两种尺度不能混用；这里只作描述性统计。

| 气体 | 均值 | 中位 | 标准差 | p90 | 最小 | 最大 |
|---|---:|---:|---:|---:|---:|---:|
| H₂ | 2.01% | 1.08% | 2.51% | 3.67% | 0.34% | 15.25% |
| NH₃ | 1.01% | 0.63% | 1.04% | 2.14% | 0.15% | 5.06% |
| NO₂ | 1.40% | 1.24% | 0.76% | 2.22% | 0.38% | 3.85% |

H₂ 的最大值来自 file 0（H₂=32000 的边角），它同时拉高了 H₂ 的均值与标准差；
中位数只有 1.08%，说明大多数文件的 H₂ 曲线并不差。

### 3.2 响应骨架对比

按 `plan_20261123.md` 的统一定义，比较四种归一化阶跃形状：一阶、临界阻尼二阶、
两个二阶并联、当前双指数。四种骨架接同一个三路独立 MLP 残差，使用完全相同的
62 折训练/留出划分；详细协议、逐折文件和取舍见
[`artifacts/comparison/comparison_summary.md`](../artifacts/comparison/comparison_summary.md)。

第一轮用 seed 0 筛选四个骨架：

| 骨架 | 参数数 | 宏 nRMSE/% | H₂/% | NH₃/% | NO₂/% |
|---|---:|---:|---:|---:|---:|
| 双指数（当前） | 69 | 1.478 | 2.018 | 1.012 | 1.405 |
| 一阶 | 45 | 1.481 | 2.124 | 1.011 | 1.307 |
| 两个二阶并联 | 69 | 1.634 | 2.052 | 1.260 | 1.590 |
| 二阶（临界阻尼） | 45 | 1.766 | 2.106 | 1.419 | 1.771 |

前两名双指数与一阶再用 seed 20260923/24/25 复跑：

| 骨架 | 宏 nRMSE/%（均值 ± seed 标准差） | H₂/% | NH₃/% | NO₂/% |
|---|---:|---:|---:|---:|
| 双指数（当前） | **1.5209 ± 0.0154** | 2.0647 | 1.1574 | 1.3407 |
| 一阶 | 1.5261 ± 0.0233 | 2.1024 | 1.1504 | 1.3254 |

一阶减双指数的逐文件配对差：宏中位 +0.0135 百分点、均值 +0.0052 百分点，
bootstrap 95% CI [−0.0483, +0.0500]，Wilcoxon p=0.273；H₂ 差 +0.038 百分点
（p=0.009），NH₃/NO₂ 差异不显著。也就是说，双指数与一阶在整体上几乎等价，
双指数只在 H₂ 上略好；两个二阶并联与二阶更差。

结论：**骨架形式不是当前误差的主要瓶颈**。保留双指数作为数据生成骨架，
因为它的 3 seed 宏均值最低且是当前默认；一阶作为 45 参数的等价备选保留。
筛选 seed 0 的 1.478/1.481% 偏乐观，3 seed 均值为 1.521/1.526%；单 seed
结果不能当作骨架差异的证据。

## 4. 解释与诊断

- H₂ 是最难通道：相对幅度只有约 1%，骨架的二次增益和双时间常数仍不能完全描述
  低浓度/高 NO₂ 边角；MLP 把 H₂ 宏误差从 5.17% 降到 2.02%，但仍是三路中最大。
- NH₃ 最容易：响应幅度大、形状接近单指数；MLP 从 1.69% 降到 1.01%。
- NO₂ 的负响应和中等时间常数使其对基线/漂移更敏感；MLP 从 3.04% 降到 1.40%。
- 三路独立同时改变了容量与训练方式，不能把全部收益单独归因于“三路独立”；
  它只是当前实现选择。

## 5. 限制

- 62 条实验、每个浓度组合 1 条、无重复；NH₃ 与采集顺序高度混淆，
  H₂=32000 只有 4 条且都在 NO₂=2。
- 单 seed、固定 600 轮、无早停；没有独立锁定测试，绝对误差偏乐观。
- 骨架对比的 1 seed 筛选偏乐观；前两名 3 seed 的差异仍与 seed 方差同量级，
  不能据此声称双指数显著优于一阶。
- 曲线 nRMSE 只衡量形状，不是浓度误差；本分支不再做 MAP 反演。
- 模型只在恒浓度阶跃下验证；时变浓度未验证。
- 正模型增强 TCN 的 `real` 臂与旧 TCN 参考不是同一协议；旧 plan 的 10 s
  7.47% 不能直接当作本轮基线，见第 8 节。
- 增强 TCN 只覆盖开发 50 的 5 折和 3 个 base seed；`real_dup` 同时增加真实曝光
  和更新数，不能把“更多样本”与“更多更新”完全分开。

## 6. 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

# 四骨架 62 折筛选（约 40 分钟）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first,second,second2 --seeds 0 --max-iter 600

# 前两名 3 seed 复跑（约 44 分钟）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first --seeds 20260923,20260924,20260925 --max-iter 600

MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
```

## 7. 图

- [fig01_design_and_data](../figures/model/fig01_design_and_data.png)：设计与示例曲线。
- [fig03_curve_error](../figures/model/fig03_curve_error.png)：逐文件 nRMSE 与骨架/组合对比。
- [fig07_model_architecture](../figures/model/fig07_model_architecture.png)：模型架构。
- [fig09_concentration_design](../figures/model/fig09_concentration_design.png)：浓度组合覆盖。
- [fig10_nh3_initial_drift](../figures/model/fig10_nh3_initial_drift.png)：NH₃ 初值漂移。
- [fig12_all_curves_H2](../figures/model/fig12_all_curves_H2.png)：H₂ 的 62 条实测/留一预测总图。
- [fig13_all_curves_NH3](../figures/model/fig13_all_curves_NH3.png)：NH₃ 总图。
- [fig14_all_curves_NO2](../figures/model/fig14_all_curves_NO2.png)：NO₂ 总图。
- [fig15_curve_error_sigma](../figures/model/fig15_curve_error_sigma.png)：逐文件误差散点与均值/σ 带。
- [fig16_forward_augmentation](../figures/model/fig16_forward_augmentation.png)：正模型增强 TCN 的 3σ/MAPE 汇总。
- [fig17_forward_augmentation_paired](../figures/model/fig17_forward_augmentation_paired.png)：逐折/seed 配对散点。
- [fig18_old_protocol_augmentation](../figures/model/fig18_old_protocol_augmentation.png)：旧训练协议下的增强来源对照。
- [fig19_full_protocol_augmentation](../figures/model/fig19_full_protocol_augmentation.png)：完整旧协议（噪声、边缘、两套低浓度配置、6 checkpoint）的四臂对照。

## 8. 正模型增强逆 TCN（实验分支）

### 8.1 协议

- 数据只来自 50 条开发实验；5 折，每折 40 条训练、10 条验证；锁定 12 条不参与
  OOF、不参与早停或选型。
- 四臂：<code>real</code>（40 条真实）、<code>forward</code>（40 条真实 + 40 条正模型合成）、
  <code>pca_vae</code>（40 + 40 条 PCA-VAE 合成）、<code>forward+pca_vae</code>（40 + 40 + 40）。
- 正模型每折只用该折 40 条训练：先拟合双指数骨架，再拟合三路独立 MLP；
  合成浓度一半复制训练标签、一半在训练档位网格内随机组合；R0 一半取同浓度真实
  初值、一半从训练折经验分布重采样。合成曲线只用于训练，不进入验证/锁定。
- PCA-VAE 只复用了旧 `VAEConfig`：6 个主成分、4 维潜变量、隐藏层 64、300 epoch、beta=0.1；**没有复用旧最终方案的增强协议**（0.1× 噪声、3× 合成、边缘条件 ×4、两套低浓度配置、6 checkpoint 平均、裁剪）。
- 三臂增强与 <code>real</code> 共用同一训练折 scaler、同一 TCN、同一
  <code>validation_metric="three_sigma"</code>；训练轮数上限 500，早停耐心 50。
- 3 个 base seed：20260923/24/25；每个折的训练随机种子为
  <code>seed + 1000 * fold</code>，所以“3 seed”是 3 个 base seed × 5 折，不是同一个种子
  在 5 折上重复。OOF 指标按每个 seed 的 50 条合并后计算，再对 3 个 seed 取
  mean/sd；配对统计用同 (seed, fold) 的 15 对。
- 指标为逐气体相对误差的 3σ（<code>3*std(ddof=1)</code>）、MAPE、中位绝对相对误差、p90，
  再对三气体取宏平均；单位均为 %。

### 8.2 OOF 结果

| 窗口 | 臂 | 宏 3σ | 宏 MAPE | 宏中位 | 宏 p90 |
|---|---|---:|---:|---:|---:|
| 5 s | real | 17.01 ± 1.62 | 4.49 ± 0.50 | 3.80 ± 0.66 | 8.91 ± 0.87 |
| 5 s | forward | 20.49 ± 0.45 | 5.22 ± 0.08 | 4.06 ± 0.30 | 10.82 ± 0.74 |
| 5 s | pca_vae | 17.02 ± 1.41 | 4.35 ± 0.21 | 3.48 ± 0.22 | 8.71 ± 1.03 |
| 5 s | forward+pca_vae | 20.35 ± 3.34 | 5.21 ± 0.79 | 3.91 ± 0.58 | 11.44 ± 1.94 |
| 10 s | real | 15.88 ± 1.33 | 4.11 ± 0.41 | 3.33 ± 0.37 | 8.56 ± 0.91 |
| 10 s | forward | 18.03 ± 2.73 | 4.44 ± 0.58 | 3.27 ± 0.61 | 9.48 ± 0.16 |
| 10 s | pca_vae | 16.67 ± 0.25 | 4.41 ± 0.12 | 3.53 ± 0.14 | 9.03 ± 0.68 |
| 10 s | forward+pca_vae | 19.85 ± 1.47 | 5.03 ± 0.23 | 3.83 ± 0.31 | 10.13 ± 0.87 |

读法：

- **5 s**：PCA-VAE 的 MAPE（4.35 对 4.49）、中位（3.48 对 3.80）和 p90
  （8.71 对 8.91）略低，但宏 3σ 几乎不变（17.02 对 17.01）；3 个 seed 的
  3σ 差为 −0.44/+0.70/−0.22 百分点，方向不一致。正模型与混合臂的 3σ、
  MAPE、p90 都更差。
- **10 s**：PCA-VAE 的宏 3σ 16.67 对 15.88、MAPE 4.41 对 4.11、中位
  3.53 对 3.33、p90 9.03 对 8.56，四项都更差；正模型与混合臂也整体更差。
  上表是**逐 seed 的 OOF 指标再取均值**，不是 3 seed 预测平均；把三个 seed 的
  预测先平均，10 s 宏 3σ 为 `real` 12.78、`pca_vae` 12.81，裁剪后 12.30 / 12.35。
  旧方案的 7.47% 使用的是 6 checkpoint 集成，不能与上表的逐 seed 数字直接比。
- 正模型臂在 10 s 的中位差 −0.06 百分点，但 3σ +2.15、MAPE +0.32、
  p90 +0.92，不能作为改善证据。

补充一个“双倍真实样本”对照（<code>real_dup</code>）：把真实训练折重复一次得到
n=80，不做任何合成，训练轮数上限仍为 500。它不是四臂之一，只用来判断
“更多更新/更多真实曝光”能否解释增强臂的收益。按逐折指标（15 个 fold/seed）
平均：

| 窗口 | real 宏 3σ / MAPE | real_dup 宏 3σ / MAPE | pca_vae 宏 3σ / MAPE |
|---|---:|---:|---:|
| 5 s | 15.99 / 4.49 | **14.24 / 4.07** | 15.42 / 4.35 |
| 10 s | 14.31 / 4.11 | **13.25 / 3.64** | 14.82 / 4.41 |

<code>real_dup</code> 在两个窗口都优于 <code>real</code> 和 <code>pca_vae</code>，说明
“更多真实样本/更多更新”本身就能降低验证误差；PCA-VAE 的 5 s 小幅优势并不能
单独归因于合成数据。由于 <code>real_dup</code> 也把真实曝光翻倍，它不是严格的等真实
曝光对照；本轮只能说增强臂没有显示出超过该对照的独特收益。

逐 (seed, fold) 配对（候选减 <code>real</code>，负值更好；15 对）：

| 窗口 | 臂 | 宏 3σ 中位差 | 改善折数 | 95% CI | Wilcoxon p |
|---|---|---|---:|---|---:|
| 5 s | forward | +3.91 | 5/15 | [+0.35, +4.15] | 0.048 |
| 5 s | pca_vae | −0.98 | 10/15 | [−1.36, +0.27] | 0.151 |
| 5 s | forward+pca_vae | +2.11 | 5/15 | [+0.52, +5.45] | 0.083 |
| 10 s | forward | +0.49 | 7/15 | [−0.96, +4.50] | 0.389 |
| 10 s | pca_vae | +0.53 | 6/15 | [−0.64, +1.74] | 0.421 |
| 10 s | forward+pca_vae | +3.09 | 3/15 | [+1.68, +6.58] | 0.005 |

PCA-VAE 在 5 s 的中位差为负但置信区间跨 0（p=0.151），在 10 s 为正且不显著；
其余增强臂的宏 3σ 中位差为正（更差），10 s 混合臂还显著。**没有一条增强臂在 3σ 上稳定优于仅真实**，
因此没有进入锁定 12 的候选评估；锁定集保持未参与，避免事后比较。

### 8.3 与旧 TCN 参考的差异

plan 中的 5 s 12.99%/2.98% 与 10 s 7.47%/1.79% 来自旧 TCN 特征协议，
不是本轮 <code>real</code> 臂。本轮 <code>real</code> 的 10 s OOF 为 15.88% 宏 3σ / 4.11% 宏 MAPE，
与旧 <code>closed_loop</code> 的 10 s real 逐折均值（14.50%/4.27%）同量级，说明本轮训练
协议本身偏保守；所以本节的结论只适用于“同一 TCN、同一特征、同一早停口径下的
四臂比较”，不能写成“TCN 的绝对性能不如旧方案”。四臂共享同一协议，内部比较
仍然公平。

更重要的是，旧 7.47%/1.79% 不是 `real` 臂，而是旧最终方案的 **6 个 checkpoint
（2 套低浓度加权 × 3 seed）预测平均后再裁剪**；它包含 0.1× 噪声、PCA-VAE 3×、
边缘条件采样 ×4，而本轮 `pca_vae` 臂只有 1× PCA-VAE、无噪声、无边缘、单配置、
不做 checkpoint 平均与裁剪。两者不是同一个系统。

### 8.4 结论

正模型合成、PCA-VAE、两者混合都没有在本轮 3 seed × 5 折 OOF 上给出稳定的
3σ/MAPE 改善；PCA-VAE 只在 5 s 的 MAPE/中位/p90 上有小幅、不显著的优势，
10 s 反而更差，且其逐折平均仍不如双倍真实样本对照 <code>real_dup</code>。**在当前简化协议内**，
本分支不采用合成增强作为默认训练方案，保留 <code>real</code> 作为 TCN 基线。
但 §8.6 已说明该简化臂没有复现旧最终方案，不能据此否定旧 PCA-VAE 结论；
§8.7 的旧训练协议对照进一步显示，单独 PCA-VAE 不显著、正模型单独更差，
而“正模型 + PCA-VAE”混合臂稳定改善（14/15，p=0.00018）。因此更准确的结论是：
**在当前简化协议内不采用合成增强；在旧训练协议、等合成量下，混合来源才有效**。
如果后续要重试，应先解决正模型在 40 条训练折上的分布偏移，并用更多
seed 或独立的等曝光对照把“更多更新”与“合成数据”分开。

### 8.5 产物

- 代码：<code>tcn_union/experiments/forward_augmentation.py</code>（四臂 OOF）、
  <code>tcn_union/experiments/real_control.py</code>（双倍真实样本对照）、
  <code>tcn_union/experiments/locked_forward.py</code>（锁定 12 的一次性评估，本轮未用）。
- 结果：<code>artifacts/forward_augmentation/{folds,oof,summary}.csv</code>、
  <code>real_dup.csv</code>、<code>real_dup_summary.csv</code>、<code>predictions.npz</code>、
  <code>run_config.json</code>。
- 图：[fig16_forward_augmentation](../figures/model/fig16_forward_augmentation.png)、
  [fig17_forward_augmentation_paired](../figures/model/fig17_forward_augmentation_paired.png)；
  旧协议增强来源图见 §8.7 的 [fig18_old_protocol_augmentation](../figures/model/fig18_old_protocol_augmentation.png)。

### 8.6 与旧 PCA-VAE 结论的协议复核（2026-11-23）

本轮 §8 的负结果**不能推翻**旧结论，因为两者不是同一实验：

| 项目 | 旧最终方案 | 本轮 pca_vae 臂 |
|---|---|---|
| 噪声视图 | 0.1× MAD，+40/折 | 无 |
| PCA-VAE 合成 | 3×，+120/折 | 1×，+40/折 |
| 边缘条件采样 | ×4，+120–132/折 | 无 |
| 低浓度配置 | p=1.0 与 0,0.3,1.0 两套 | 只有 p=1.0 |
| 预测聚合 | 2 配置 × 3 seed，共 6 checkpoint 平均 | 每个 seed 单独算 OOF |
| 裁剪 | 裁剪到开发集浓度范围 | 无 |
| 10 s OOF 宏 3σ | 7.47% | 单 seed 16.65%（3 seed 平均 16.67） |

用旧 `gan_train.py` 在本轮同一 manifest、同一 40 条/折上，只跑 seed 20260923 的
旧 p10 协议（noise_gan、3×、边缘 4），10 s OOF 为 **10.22% 宏 3σ / 2.64% MAPE**；
本轮同 seed 的 `pca_vae` 为 16.65% / 4.47%，`real` 为 15.16% / 3.88%。
可见差距主要来自旧增强协议，而不是“PCA-VAE 忽然失效”。旧最终 7.47% 还叠加了
3 seed × 2 配置平均与裁剪，因此不能拿它和本轮单 seed、单配置的数字直接比较。

此外，旧文档自己已经写明：**纯 PCA+VAE 3×（不加噪声/aux）的 OOF macro-NMAE
0.029248 差于 clean 0.027097**，只有测试集更好（MAPE 3.559% 对 4.648%）。
旧结论的准确表述应是“noise 0.1 + PCA-VAE 3× + 边缘扩充 + 多配置多种子平均”
这套组合在 12 条锁定集上略好，而不是“单独 PCA-VAE 稳定提升 OOF”。

因此本轮补做的是**旧训练协议下的增强来源对照**（§8.7），而不是把 §8 的简化臂
当作最终结论。§8.7 仍不是完整旧 final：它去掉了噪声、边缘采样和两套低浓度
配置，只保留旧训练协议与等合成量，用来隔离“增强来源”。§8 的“不采用合成增强”
只适用于当前简化协议。

### 8.7 旧训练协议下的增强来源对照（2026-11-23）

为避免“1× 合成 + 新特征协议”把结论带偏，本轮补做了旧 TCN 训练协议下的
**增强来源**对照。协议固定为：50 条开发集、5 折、每折 40 条训练；旧 p10 的
`three_sigma` 早停、`log,none,log` 标签空间、逐气体输出头、`low_conc_power=1.0`、
最多 500 epoch；每个臂各自拟合 scaler。四臂都**不加** 0.1× 噪声、边缘复制或
两套低浓度配置，以隔离“增强来源”本身；增强臂的合成总数都固定为 120 条，
混合臂为 60 条正模型 + 60 条 PCA-VAE。锁定 12 条仍不参与。

| 10 s，3 seed × 5 折 | 宏 3σ（seed 均值±sd） | 宏 MAPE | 3 seed 集成 3σ | 配对 3σ 改善折数 | 中位配对差 | Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| 仅真实 | 15.47 ± 1.73 | 3.95 | 11.70 | — | — | — |
| PCA-VAE only | 14.22 ± 1.12 | 3.75 | 10.98 | 9/15 | −0.05 | 0.804 |
| 正模型 only | 16.93 ± 2.03 | 4.22 | 13.70 | 5/15 | +1.61 | 0.107 |
| 正模型 + PCA-VAE | **11.40 ± 1.32** | **2.81** | **8.27** | **14/15** | **−2.42** | **0.00018** |

混合臂的配对差均值为 −3.11 百分点，bootstrap 95% CI [−4.42, −1.86]；
分气体 3σ（seed 平均 OOF）为 10.83 / 8.71 / 14.67，仅真实为
16.23 / 10.79 / 19.38（H2/NH3/NO2）。混合臂的 MAPE 配对差中位 −0.87、12/15 改善，p=0.00085；
p90 中位 −2.09、13/15 改善，p=0.00116。

结论：

1. **PCA-VAE 单独**方向上有改善（3σ 15.47→14.22，集成 11.70→10.98），但
   15 对配对只有 9/15 改善、p=0.804，不能称为稳定提升；这和旧文档“单独
   生成式增强收益有限”一致。
2. **正模型单独**不改善（16.93，5/15，p=0.107），说明仅靠正模型补设计域
   不够。
3. **正模型 + PCA-VAE** 在 14/15 对上改善 3σ，p=0.00018，3 seed 集成
   8.27%，是唯一稳定的组合；这与旧结论“组合方案才有效”一致，也解释了
   为什么单纯去掉噪声/边缘后的 VAE-only 达不到旧最终 7.47%。
4. 旧最终 7.47% 仍是 **6 checkpoint + 裁剪**的结果，本表的 8.27% 是
   单配置 3 seed 集成，两者不能直接等同；本表也没有复现旧 final 的噪声、
   边缘采样和两套低浓度配置。

产物：`artifacts/old_protocol_augmentation/{folds,oof,summary}.csv`、
`predictions.npz`、`run_config.json`；图
[fig18_old_protocol_augmentation](../figures/model/fig18_old_protocol_augmentation.png)。
复现命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.old_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500
```

### 8.8 完整旧协议复刻（2026-11-23）

§8.7 去掉了旧最终方案的噪声、边缘复制、两套低浓度配置与 6 checkpoint
平均，只能隔离“增强来源”。本轮把四项全部加回，四臂都按同一套完整旧协议
训练；real 臂保持干净（40 条真实、无噪声/边缘/合成），三个增强臂为
40 真实 + 40 噪声 + 120 合成 + 120–132 边缘。详细协议、分气体表、配对
统计与限制见 [full_protocol_augmentation.md](full_protocol_augmentation.md)。

| 10 s，完整旧协议 | 3 seed 宏 3σ（均值±sd） | 6 checkpoint 宏 3σ | 6 checkpoint 宏 MAPE | 6 checkpoint 中位 | 6 checkpoint p90 |
|---|---:|---:|---:|---:|---:|
| 仅真实 | 13.48 ± 1.23 | 10.79 | 2.77 | 2.30 | 6.12 |
| PCA-VAE only | **9.02 ± 0.29** | **7.47** | **1.79** | **1.29** | **4.11** |
| 正模型 only | 12.88 ± 3.35 | 10.92 | 2.61 | 2.08 | 5.79 |
| 正模型 + PCA-VAE | 11.00 ± 1.32 | 9.10 | 2.11 | 1.47 | 4.22 |

单位 %。相对仅真实，PCA-VAE only 的 15 对宏 3σ 配对为 14/15 改善、
中位 −3.03、均值 −3.63、95% CI [−4.98, −2.46]、p=0.00012；正模型 only
8/15、中位 −0.15、p=0.524；混合臂 14/15、中位 −2.74、p=0.0084。
PCA-VAE only 的 6 checkpoint 宏 3σ 7.47%、MAPE 1.79% 与旧最终
7.47% / 1.79% 对齐，说明协议级复刻成立；正模型单独没有增益，混合臂
反而把 VAE 拉到 9.10%。分气体上混合臂 H2 更低（8.00% 对 9.31%），
但 NO2 明显更差（13.83% 对 8.16%），宏平均仍是 VAE-only 最好。
因此完整旧协议下不采用正模型合成；旧最终方案的主因更可能是 VAE、
噪声、边缘条件与多配置，而不是正模型。锁定 12 条未参与本轮。

### 8.9 正模型条件 R0 与按气体路由（2026-11-25）

条件 R0 修正了正模型合成里“随机基线”与浓度标签不一致的问题，但仍只在 H2
产生稳健增益。固定按气体路由（H2 用 mixed、NH3/NO2 用 VAE）的 50 条 OOF
宏 3σ 为 7.14%，看似优于 VAE 的 7.47%；嵌套选臂后约 7.47%，两者打平。

H2-only 单模型（正模型样本只监督 H2、VAE 样本监督三气体）已完成 3 seed ×
2 配置复验：6 checkpoint 集成宏 3σ 8.04%、VAE 7.99%、仅真实 10.59%；
分气体 7.51/5.12/11.50% 对仅真实 11.00/6.17/14.62%，三条都改善（15/15 对
宏改善，p=6.1×10⁻⁵）。相对 VAE 只有 H2 显著更好（p=0.0034），NH3/NO2
更差，宏平均配对 p=0.934，统计打平。1 seed 曾显示 9.75% 低于 VAE，3 seed
未复现。因此它是“相对仅真实三气体都受益、与 VAE 可比”的单模型备选，
但没有超过 VAE；总体最优仍是 VAE-only，而“三气体都相对仅真实显著改善”
这一目标由 H2-only 达到。

预训练臂（正模型合成预训练 + 真实微调）的 3 seed × 2 配置复验为负：
6 checkpoint 集成宏 3σ 12.98%，分气体 9.52/17.20/12.22%；H2 好于仅真实，
NH3 反而从 6.17% 恶化到 17.20%，部分折 1 epoch 早停。因此未采用按气体
拼接 pretrain 的路由。

详细协议、条件 R0、H2-only、残差 bootstrap 与选择偏差检查见
[forward_augmentation_20261125.md](forward_augmentation_20261125.md)。
### 8.10 纯正模型 joint10：未超过 VAE（2026-11-25）

为回答“不混入 VAE、只靠正模型能否三气体都超过 VAE”，新增纯正模型臂
forward_prob_joint10：正模型均值 + 三气体联合残差的 10 个主成分条件
采样；完整旧协议、5 折 × 3 seed × 2 配置、6 checkpoint 集成。

6 checkpoint 集成宏 3σ/MAPE：仅真实 10.595%/2.745%，PCA-VAE
7.992%/1.980%，joint10 7.866%/1.746%。分气体 3σ：仅真实
11.000/6.165/14.619%，PCA-VAE 10.066/4.334/9.575%，joint10
8.082/4.346/11.169%。joint10 的 H2 比 VAE 低 1.98 百分点，NH3 持平，
NO2 高 1.59 百分点。

配对（15 个 seed×折）：joint10−仅真实宏 12/15 改善、中位 −2.928、
p=0.0151，H2 p=0.0554、NH3 p=0.00263、NO2 p=0.0730；joint10−VAE
宏 8/15、中位 −0.019、p=0.7615，H2 p=0.0946、NH3 p=0.8469、
NO2 p=0.3591。因此 joint10 相对仅真实三气体方向都改善，但没有一种
气体显著超过 VAE；严格目标“纯正模型三气体都明显超过 VAE”未达成。
bootstrap 残差采样全量 8.826%/2.078%，更差；单 seed 的 NO2 优势未复现。

解释：H2 的残差是低频增益/时间常数偏移，低秩联合残差可覆盖；NH3 形状
一致，均值足够；NO2 需要实验特异的快速衰减形状，10 个全局主成分无法
提供，VAE 潜变量可以。详细协议、joint5、bootstrap 负结果与限制见
[pure_forward_augmentation.md](pure_forward_augmentation.md)。
