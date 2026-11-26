# Handoff：完整旧 TCN 协议下的增强来源对照（2026-11-23）

- 分支：`experiment/forward-augmented-tcn`
- 基线 HEAD：`20538ff refactor(sensor-array): remove MAP evaluation and add next-step plan`
- 本次交付：完整旧协议四臂对照、结果文档、fig19、全量文档同步、快照更新
- 提交：见 `git log -1`；已按用户要求 commit + push 到 `origin/experiment/forward-augmented-tcn`
- 之前的交接：`handoff_20261123_map_removal_and_plan.md`、`handoff_20261120_tcn_5s_and_closed_loop.md`

## 0. 一句话

完整旧协议（0.1×MAD 噪声 + 边缘复制 ×4 + 两套低浓度配置 + 6 checkpoint + 裁剪）下，
**PCA-VAE only 6 checkpoint 宏 3σ/MAPE 为 7.47%/1.79%，与旧最终 7.47%/1.79% 对齐，
是四臂中最好的；正模型 only 10.92% 与仅真实 10.79% 持平；正模型 + PCA-VAE 9.10%，
反而把 VAE 结果拉差。** “正模型合成能补 VAE”在简化协议里成立，在完整旧协议下不成立。

## 1. 本轮任务与边界

用户要求：保留已有的 `artifacts/augmentation/protocols/old/`（旧训练协议、等 120 条合成、
无噪声/边缘的来源对照），再补一整轮**完整旧协议**实验，加入 0.1 噪声、边缘复制、
两套低浓度配置，重新比较 real / PCA-VAE / 正模型 / 正模型 + PCA-VAE，写结果文档，
最后写 handoff 并 commit + push。

硬约束：

- 只有 `data/0.csv`–`data/61.csv` 共 62 条实验；不补测、不补采。
- 锁定测试 `{2,7,15,20,23,27,31,38,41,46,55,60}` 不参与训练、调参或本轮评估。
- 本轮只用 50 条开发集的 5 折 OOF；锁定 12 条未评估。
- 浓度单位未注明，写“原始单位”，不写 ppm。
- 图件只出 PNG 300 dpi + SVG，不出 PDF。
- 不修改旧 `artifacts/augmentation/protocols/old/`，新结果写新目录。

## 2. 结果

### 2.1 10 s 宏指标

| 臂 | 3 seed 宏 3σ（均值±sd） | 3 seed 宏 MAPE | 6 checkpoint 宏 3σ | 6 checkpoint 宏 MAPE | 6 checkpoint 中位 | 6 checkpoint p90 |
|---|---:|---:|---:|---:|---:|---:|
| real | 13.48 ± 1.23 | 3.34 | 10.79 | 2.77 | 2.30 | 6.12 |
| vae | **9.02 ± 0.29** | **2.17** | **7.47** | **1.79** | **1.29** | **4.11** |
| forward | 12.88 ± 3.35 | 3.02 | 10.92 | 2.61 | 2.08 | 5.79 |
| forward+vae | 11.00 ± 1.32 | 2.53 | 9.10 | 2.11 | 1.47 | 4.22 |

单位均为 %。3 seed 列先对每个 seed 的两套低浓度配置取平均，再算 3 个 seed 的
均值与样本标准差；6 checkpoint 列是 2 配置 × 3 seed 先平均后裁剪的 OOF。

### 2.2 分气体 6 checkpoint 宏 3σ

| 臂 | H2 | NH3 | NO2 | 宏 |
|---|---:|---:|---:|---:|
| real | 11.22 | 6.15 | 15.01 | 10.79 |
| vae | 9.31 | 4.95 | **8.16** | **7.47** |
| forward | 10.06 | 8.39 | 14.33 | 10.92 |
| forward+vae | **8.00** | 5.47 | 13.83 | 9.10 |

混合臂的 H2 是四臂最低，但 NO2 比 VAE-only 高 5.67 百分点；VAE-only 的
NH3、NO2 都是最低，宏平均因此最好。

### 2.3 配对统计（seed × fold = 15 对）

候选 − real，负值更好；每对先对该 seed 的两套配置取平均。bootstrap 100,000 次
百分位 95% CI，双侧 Wilcoxon。

| 候选 | 指标 | 改善/15 | 中位差 | 均值差 | 95% CI | p |
|---|---|---:|---:|---:|---:|---:|
| vae | 宏 3σ | **14** | **−3.03** | **−3.63** | [−4.98, −2.46] | **0.00012** |
| vae | 宏 MAPE | **14** | **−1.20** | **−1.17** | [−1.53, −0.84] | **0.00012** |
| forward | 宏 3σ | 8 | −0.15 | −0.45 | [−2.89, +2.37] | 0.524 |
| forward | 宏 MAPE | 8 | −0.35 | −0.33 | [−1.00, +0.41] | 0.303 |
| forward+vae | 宏 3σ | **14** | **−2.74** | **−2.39** | [−3.95, −0.50] | **0.0084** |
| forward+vae | 宏 MAPE | **14** | **−0.60** | **−0.81** | [−1.33, −0.22] | **0.0054** |

VAE 与混合的直接配对：宏 3σ 9/15、中位 −0.55、均值 −1.24、
CI [−3.18, +0.34]、p=0.359；方向偏向 VAE，但 15 对不足以显著。
VAE 对 forward：12/15、中位 −2.20、p=0.0043；混合对 forward：11/15、
中位 −2.27、p=0.0151。

分气体（VAE − real）：H2 9/15、中位 −1.60、p=0.277；NH3 12/15、
中位 −1.85、p=0.0054；NO2 14/15、中位 −7.04、p=0.00018。VAE 的主要收益
来自 NH3 与 NO2，H2 不显著。

## 3. 协议要点

- 数据：manifest 的 50 条开发集，固定 5 折，每折 40 训练 / 10 验证；
  验证与 OOF 始终只用真实曲线。
- 网络：16→64 因果 TCN、3 个残差块、kernel 5、dilation 1/2/4、逐气体输出头；
  与旧 `train.py`/`gan_train.py` 的 p10 设置一致。
- 目标：H2/NO2 在 log 空间、NH3 在原空间（`log,none,log`）；SmoothL1；
  `per_gas_head=True`；最多 500 epoch、patience 50，验证指标为宏 3σ。
- 低浓度权重：配置 A `1.0`，配置 B `0,0.3,1.0`；每个配置 × 3 seed
  = 6 checkpoint。预测先对 6 个原始预测平均，再裁剪到开发集逐气体 [min,max]。
- 噪声：每折对 40 条真实训练曲线各生成 1 条 0.1×MAD 噪声视图（+40 条），
  标签不变；real 臂保持干净，不加噪声/边缘/合成。
- 合成：每折基础 120 条（3×40）。VAE 用 `VAEConfig(seconds=10, components=6,
  latent_dim=4, beta=0.1)`，只用该折 40 条训练行拟合；正模型为双指数骨架 +
  三路独立 MLP，同样只用该折 40 条训练行拟合。
- 边缘：取训练折中任一气体处于该气体 min/max 的唯一浓度组合（30–33 个），
  每个组合复制 4 次；混合臂的边缘由 VAE 与正模型各分一半。
- 每折每臂训练行数：real 40；三个增强臂 320–332（40 真实 + 40 噪声 +
  120 合成 + 120–132 边缘）。

与旧实现的关键差异：旧 `old_protocol_augmentation.py` 在正模型合成里执行
`relative[:, 0, :] = 0.0`，把 H2 合成曲线压成平线；本轮
`full_protocol_augmentation._forward_features` 不再置零，三路都用完整相对响应。
所以 forward 臂与旧来源对照的差异不能只归因于噪声/边缘/两配置，也包含这个修正。

## 4. 产物与复现

代码：

- `tcn_union/experiments/full_protocol_augmentation.py`：四臂、噪声/边缘/
  两配置/6 checkpoint、OOF 与配对统计。
- `tcn_union/visualization/full_protocol_augmentation.py`：fig19。
- `sensor_array/docs/full_protocol_augmentation.md`：完整结果文档（9 节）。

数据：

- `artifacts/augmentation/protocols/full/folds.csv`：逐 fold/seed/config/arm；
- `fold_seed.csv`：每 seed/折先平均两配置后的指标（配对用）；
- `oof_seed.csv`：每 seed 的两配置平均 OOF；
- `oof.csv`：6 checkpoint 平均后的 OOF 与分气体指标；
- `summary.csv`、`paired.csv`、`predictions.npz`、`summary.md`、`run_config.json`。

图：`sensor_array/figures/model/fig19_full_protocol_augmentation.png/.svg`，
三面板为 10 s 宏 3σ、宏 MAPE、逐 seed/折配对差。

复现命令（约 56 分钟，GPU）：

~~~bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl \
  .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae forward forward+vae
~~~

重画 fig19：

~~~bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.visualization.full_protocol_augmentation
~~~

## 5. 与旧结果的关系

| 实验 | real | vae | forward | forward+vae |
|---|---:|---:|---:|---:|
| §8.7 来源对照（无噪声/边缘/2 配置，单配置 3 seed） | 15.47 | 14.22 | 16.93 | 11.40 |
| 本轮完整旧协议（6 checkpoint） | 10.79 | **7.47** | 10.92 | 9.10 |

两行不能逐项直接相减，因为 §8.7 没有噪声/边缘/两配置/6 checkpoint/裁剪；
但排序变化很清楚：简化协议下混合臂最好，完整协议下 VAE-only 最好。
旧最终 7.47%/1.79% 是 VAE-only + 6 checkpoint + 裁剪，本轮 VAE-only 与它对上，
说明“旧协议有效”的主因可以落到 VAE、噪声、边缘与多配置，而不是正模型合成。

## 6. 限制

- 只跑了 1 个训练框架、1 组超参数；没有对 TCN 结构、学习率、epoch 做二次搜索，
  不能声称“任何正模型合成都无效”。
- 15 对配对来自 3 seed × 5 折，seed 与折之间不独立；p 值与 CI 只作描述，
  不是独立重复实验的推断。
- 正模型与 VAE 都只用 40 条训练行拟合；正模型的 40 条拟合样本很少，外推到
  边缘浓度时本身就不稳定。这是“正模型合成”的固有困难，不是实现 bug。
- 锁定 12 条没有参与本轮；结论只适用于 50 条开发集 OOF。若把 VAE-only
  作为最终候选，需要按预注册协议在锁定 12 条上只评估一次，不得用锁定结果
  回头调参。
- real 臂是干净基线，样本量 40；三个增强臂 320–332。real 与增强臂的差异
  包含样本量效应；只有三个增强臂之间的比较是“等曝光、等噪声/边缘”的来源比较。
- 6 checkpoint 的裁剪范围来自 50 条开发集标签；这是部署口径的近似，
  不是严格的无信息先验。

## 7. 本轮验证

- 根目录 `.venv/bin/python -m unittest discover -s tests`：67 项通过（1 skip）。
- `MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s sensor_array/tests`：
  8 项通过。
- `py_compile`：新实验/绘图脚本通过。
- `git diff --check -- '*.py' '*.md'`：干净。
- `sensor_array/artifacts` 与 `sensor_array/figures` 共 57 个文件，
  `SHA256SUMS` 57 行，`sha256sum -c` 全部成功；图件 26 个（13 组 PNG+SVG），
  0 PDF。

## 8. 下一步

1. **锁定 12 条一次性评估**：若 VAE-only 作为最终候选，先冻结协议与 checkpoint，
   再在锁定 12 条上评估一次，报宏 3σ/MAPE/中位/p90；不得用锁定结果调参。
2. **补 5 s 同协议对照**：本轮只做 10 s；5 s 的 VAE/正模型结论需要单独跑，
   不能从 10 s 外推。
3. **更多 seed**：3 seed 的 VAE sd 只有 0.29，已经较稳；若要写进论文，
   建议 5 seed 并报告 seed 方差。
4. **VAE 超参敏感性**：components/latent_dim/beta 只用了旧配置，未做搜索；
   若要继续提升，优先在开发集上做小网格，不要碰锁定集。
5. **正模型合成若要重试**：先解决 40 条训练折上的分布偏移（例如限制只在
   训练设计凸包内采样、或对边缘条件做拒绝采样），再做等曝光对照；不要直接把
   更多正模型合成混进 VAE 臂。
6. **文档收尾**：`learning_guide`/`beginner_guide` 中 MAP 章节仍是历史内容，
   计划第 8 周整节重写；硕士总结的逆问题章节也需同步完整旧协议结论。

## 9. 交接检查

- 当前分支 `experiment/forward-augmented-tcn`；本次提交见 `git log -1`。
- 工作区应 clean；若有未提交改动，先确认是否为本 handoff 之后的编辑。
- 不要删除 `artifacts/augmentation/protocols/old/` 与
  `artifacts/augmentation/protocols/full/`；前者是简化协议对照，后者是完整复刻。
- 不要用锁定 12 条做模型选择；不要补测/补采；不要写 ppm。
