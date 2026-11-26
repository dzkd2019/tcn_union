# 纯正模型增强：joint10 与 PCA-VAE 的对照（2026-11-25）

## 1. 目的与结论

本分支只研究**正模型合成**，不混入 PCA-VAE 样本：用双指数骨架 +
三路独立 MLP 正模型生成合成曲线，再按完整旧 TCN 协议训练逆模型，
判断能否让 H2/NH3/NO2 三种气体都相对“仅真实”显著改善，并与
PCA-VAE 对照。

结论：

1. 最好的纯正模型方案是 **joint10**：正模型均值 + 三气体联合残差的
   10 主成分条件采样。6 checkpoint 集成宏 3σ 为 **7.866%**，略低于
   PCA-VAE 的 7.992%，但分气体并不全面占优。
2. joint10 相对仅真实：宏 3σ 12/15 折改善，中位 −2.928 百分点，
   p=0.015；H2 p=0.055、NH3 p=0.0026、NO2 p=0.073。方向都改善，
   但只有 NH3 达到常规显著性。
3. joint10 相对 PCA-VAE：宏 3σ 8/15 折改善，中位 −0.019 百分点，
   p=0.762；H2 11/15、中位 −4.364、p=0.095，NH3 p=0.847，
   NO2 6/15、中位 +1.857、p=0.359。**没有一种气体显著超过 VAE，
   NO2 反而更差**。
4. 因此严格目标“纯正模型三气体都明显超过 VAE”**未达成**。当前证据
   支持：纯正模型可以把三气体都拉到优于仅真实，但不稳定超过 VAE；
   NO2 的形状多样性仍需要 VAE 的潜变量或额外实验形状先验。

## 2. 协议

与完整旧 TCN 协议一致，只改变合成来源：

| 项目 | 设置 |
|---|---|
| 数据 | data/0.csv–data/61.csv，开发 50 条 OOF |
| 折 | 5 折，每折训练 40、验证 10 |
| seed | 20260923 / 20260924 / 20260925 |
| 低浓度配置 | 1.0；0,0.3,1.0 |
| 每臂合成量 | 基础 3×40=120 条，加边缘复制 |
| 噪声 | 所有增强臂加 0.1×MAD 噪声视图；仅真实臂干净 |
| TCN | 6→16 因果 TCN，3 残差块，三气体独立头 |
| 目标变换 | H2 log，NH3 原空间，NO2 log |
| 早停 | 验证宏 3σ，patience 50，最多 500 epoch |
| 集成 | 2 配置 × 3 seed = 6 checkpoint，先平均再裁剪 |
| 裁剪 | 开发集逐气体 min/max |
| 锁定 12 | 未参与本轮任何选择或评估 |

3σ 指 50 条 OOF 相对误差的标准差乘 3；宏平均为三气体算术平均。
所有单位均为原始浓度单位。

## 3. 分阶段结果

### 3.1 joint5（5 主成分）

3 seed × 2 配置的 6 checkpoint 集成：

| 方案 | 宏 3σ / % | 宏 MAPE / % | 中位 / % | p90 / % |
|---|---:|---:|---:|---:|
| 仅真实 | 10.595 | 2.745 | 2.361 | 6.178 |
| PCA-VAE | 7.992 | 1.980 | 1.659 | 4.273 |
| joint5 | 7.956 | 1.896 | 1.391 | 4.203 |

分气体 3σ（H2/NH3/NO2）：仅真实 11.000/6.165/14.619；
PCA-VAE 10.066/4.334/9.575；joint5 7.666/4.492/11.711。

joint5 的宏平均略好于 VAE，但 NO2 仍高 2.14 百分点；H2 明显更好。

### 3.2 joint10（10 主成分，主结果）

| 方案 | 宏 3σ / % | 宏 MAPE / % | 中位 / % | p90 / % |
|---|---:|---:|---:|---:|
| 仅真实 | 10.595 | 2.745 | 2.361 | 6.178 |
| PCA-VAE | 7.992 | 1.980 | 1.659 | 4.273 |
| joint10 | **7.866** | **1.746** | **1.269** | 4.376 |

3 seed 自身均值±sd（不是 6 checkpoint 集成）：仅真实
13.411±1.148；PCA-VAE 9.812±0.406；joint10 9.684±0.908。
joint10 的 seed 方差大于 VAE。

分气体 3σ：仅真实 11.000/6.165/14.619；PCA-VAE
10.066/4.334/9.575；joint10 **8.082/4.346/11.169**。
joint10 在 H2 上比 VAE 低 1.98 百分点，NH3 基本持平（高 0.01），
NO2 高 1.59 百分点。

配对检验（15 个 seed×折，候选减对照，负值更好）：

| 对照 | 指标 | 改善折数 | 中位差 | 均值差 | 95% CI | Wilcoxon p |
|---|---|---:|---:|---:|---|---:|
| 仅真实 | 宏 3σ | 12/15 | −2.928 | −3.285 | [−5.160,−1.453] | 0.0151 |
| 仅真实 | H2 3σ | 11/15 | −2.219 | −3.246 | [−5.856,−0.589] | 0.0554 |
| 仅真实 | NH3 3σ | 12/15 | −2.047 | −2.764 | [−4.295,−1.351] | 0.0026 |
| 仅真实 | NO2 3σ | 9/15 | −3.400 | −3.844 | [−6.806,−0.956] | 0.0730 |
| PCA-VAE | 宏 3σ | 8/15 | −0.019 | −0.164 | [−1.514,1.200] | 0.7615 |
| PCA-VAE | H2 3σ | 11/15 | −4.364 | −2.216 | [−4.740,0.700] | 0.0946 |
| PCA-VAE | NH3 3σ | 8/15 | −1.007 | −0.023 | [−1.038,1.042] | 0.8469 |
| PCA-VAE | NO2 3σ | 6/15 | +1.857 | +1.746 | [−0.671,4.327] | 0.3591 |

H2 的 bootstrap CI 不含 0、但 Wilcoxon p=0.095；两个检验结论不一致，
只能写成“方向性更好”，不能写成显著优于 VAE。

### 3.3 负结果：bootstrap 与逐气体采样

为增加形状多样性，试过把联合 PCA 的高斯采样换成训练折经验残差
bootstrap（从最近 5 个训练标签中抽一个残差向量）。3 seed × 2 配置
全量结果：

| 方案 | 宏 3σ / % | 宏 MAPE / % | H2 | NH3 | NO2 |
|---|---:|---:|---:|---:|---:|
| joint10 高斯 | 7.866 | 1.746 | 8.082 | 4.346 | 11.169 |
| joint10 bootstrap | 8.826 | 2.078 | 8.560 | 5.678 | 12.240 |

bootstrap 在 seed 20260923 单 seed 筛选时看起来更好（NO2 8.03），
但 3 seed 全量没有复现；NH3/NO2 都变差，弃用。另试过 16 主成分、
增大协方差收缩、H2/NO2 用 bootstrap 而 NH3 用高斯的逐气体拼接，
5 折单 seed 均未稳定超过 joint10，未进入 3 seed 全量。

## 4. 解释

- 正模型提供了正确的均值骨架和浓度趋势，因此 H2 改善最大；H2 的
  残差主要是增益/时间常数的小偏移，低秩联合残差能覆盖。
- NH3 的曲线形状高度一致，正模型均值已足够；joint10 与 VAE 在 NH3
  上打平。
- NO2 的难点是实验特异的快速衰减形状。双指数骨架 + 10 个全局残差
  主成分只能覆盖共享的低频模式，不能生成足够多样的局部形状；VAE
  的潜变量采样提供了这种多样性。bootstrap 试图补局部形状，但把
  训练折的残差直接移植到新浓度，噪声大于收益。
- 因此“纯正模型”当前的瓶颈不是骨架，也不是残差模型容量，而是
  **实验间形状先验**。只靠 62 条曲线拟合出的确定性正模型，无法替
  代 VAE 在 NO2 上的分布建模。

## 5. 限制

1. 只评估 10 s 前缀、开发 50 条 OOF；锁定 12 条未评估。
2. 3 seed × 2 配置，15 个配对不独立；早停 epoch 在 18–400 间波动，
   同一折的 3σ 可差 1–2 百分点，宏平均差异小于 seed 方差。
3. 单 seed 筛选多次给出错误方向（joint5 7.61、bootstrap NO2 8.03），
   本文件只以 3 seed × 2 配置全量为准。
4. 正模型只用训练折 40 条拟合；合成标签来自训练折浓度分布，不含
   验证折信息；但 estimate_noise 与浓度边界仍用全 62 条，不是严格
   嵌套。
5. 未调 TCN 结构、学习率、早停耐心；未做多种子模型平均之外的
   推理集成。
6. 未做锁定测试的一次性最终评估，因此不能声称泛化到锁定 12 条。

## 6. 产物与复现

产物：

- artifacts/augmentation/pure_forward/joint10/：joint10 全量 folds.csv、fold_seed.csv、
  oof.csv、summary.csv、paired.csv、predictions.npz、run_config.json；
- artifacts/augmentation/pure_forward/joint10/report/{real,vae,forward_prob_joint,forward_prob_joint10}/：
  按 AGENTS.md §10 生成的 metrics.csv、predictions.csv 与
  figs/fig_abs_error、fig_rel_error、fig_ci95（PNG+SVG）；
- artifacts/augmentation/pure_forward/base/：joint5 全量与报告；
- artifacts/augmentation/pure_forward/joint_boot/：bootstrap 负结果。

复现 joint10：

\`\`\`bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
MPLCONFIGDIR=/tmp/mpl .venv/bin/python \\
  -m tcn_union.experiments.full_protocol_augmentation \\
  --folds 5 --seeds 20260923 20260924 20260925 \\
  --configs 1.0 0,0.3,1.0 \\
  --arms real vae forward_prob_joint forward_prob_joint10 \\
  --real-scaler --output artifacts/augmentation/pure_forward/joint10
\`\`\`

生成报告：

\`\`\`bash
for arm in real vae forward_prob_joint forward_prob_joint10; do
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python \\
    -m tcn_union.evaluation.final_report \\
    --source artifacts/augmentation/pure_forward/joint10/predictions.npz \\
    --output artifacts/augmentation/pure_forward/joint10/report/$arm \\
    --arm $arm --window 10 --run-name joint10_full
done
\`\`\`

bootstrap 负结果把环境变量设为 FORWARD_JOINT_SAMPLE=bootstrap，
其余命令相同。

## 7. 下一步

1. 若必须坚持纯正模型：需要显式的实验形状先验，例如按浓度条件建模
   时间常数/延迟的随机效应，或从真实曲线学一个只生成形状、不生成
   标签的生成器；当前 joint10 已证明仅靠低秩残差不够。
2. 若允许混合：H2 用 joint10、NH3/NO2 用 VAE 的按气体路由在
   2026-11-25 的 forward_cond_h2 中已接近可用，但属于混合方案，
   不是本分支的“纯正模型”。
3. 任何最终候选都应在锁定 12 条上只评估一次，不得用锁定结果回调
   超参。
