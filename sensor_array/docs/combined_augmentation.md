# 正模型 + PCA-VAE 组合增强（2026-11-26）

> 分支 `experiment/forward-augmented-tcn`。本文记录“把正模型样本与
> PCA-VAE 样本放进同一次训练”的组合方案。目标是利用两者互补：VAE 保
> NH3/NO2 与整体形状多样性，正模型补 H2 与浓度边缘。
>
> 结论先说：**数据层直接拼接没有超过 PCA-VAE**。3 seed × 2 配置下，
> 最好的组合 `union_h2_180` 宏 3σ 7.70%，VAE 7.59%，配对 p=0.454；
> 组合把 H2 3σ 从 9.51% 降到 7.94%，但 NO2 从 8.15% 升到 9.55%。
> 按气体路由（H2 用组合、NH3/NO2 用 VAE）的宏上限约 7.06%，但那是
> 在 OOF 上按气体挑模型，不是单模型数据增强，不作为结论。
> 50 条开发集 OOF；锁定 12 条未参与选择。

## 1. 目的与假设

四协议消融（`archive/protocol_augmentation_20261126.md`）的结论是：
完整协议下 PCA-VAE 宏 3σ 7.59%，分层正模型 9.11%，50/50 混合 7.94%。
混合没有优于 VAE，因为混合把 VAE 的 120 条全气体样本减半，而正模型
在 NH3/NO2 上并不占优。

组合增强换一个方向：**不削减 VAE 的全气体样本**，只在 VAE 之外追加
正模型的 H2 样本。这样：

- VAE 继续提供 NH3/NO2 与三气体形状多样性，NO2 不因混合而减半；
- 正模型补充 H2 的浓度条件与曲线形状，尤其是 VAE 较弱的档位；
- 单次训练仍是一个 TCN，不需要按气体路由或两个模型。

假设：H2 的额外监督会通过共享 TCN 骨干改善 H2，而 NH3/NO2 至少保持
VAE 的水平；宏平均因此低于任一单独来源。第 5 节会看到，这个假设只
对 H2 成立，NO2 没有保住。

## 2. 方法

### 2.1 `union_h2`

每折的合成集由两部分拼成：

| 部分 | 条数 | 监督 | 来源 |
|---|---:|---|---|
| PCA-VAE 全气体 | 120 / 180 | H2/NH3/NO2 | 训练折拟合的 VAE，条件标签 |
| 分层正模型 H2-only | 30–240 | 仅 H2 | 方案 1 的分层随机效应正模型 |

H2-only 样本的损失权重为 `[1, 0, 0]`，VAE 样本为 `[1, 1, 1]`；
真实曲线与 0.1×MAD 噪声视图仍为 `[1, 1, 1]`。因此 NH3/NO2 的监督
只来自真实、噪声和 VAE，不会被正模型的 NH3/NO2 预测拉偏。

边缘复制同样分开：VAE 的边缘条件样本全气体，正模型的边缘样本只监督
H2。完整协议的其他部分不变：real 40 + noise 40 + 合成 + edge，
6→16 因果 TCN、三气体独立头、两套低浓度配置、3 seed。

### 2.2 样本量扫描

H2-only 样本数是唯一的组合超参数，先做 5 折 1 seed 扫描：

| H2-only 条数 | 宏 3σ | 宏 MAPE | 说明 |
|---:|---:|---:|---|
| 30 | 11.01 | 2.63 | 接近仅真实 |
| 60 | 10.51 | 2.24 | 仍弱于 VAE |
| 120 | 8.96 | 2.06 | 首次明显优于 VAE |
| **180** | **8.09** | **1.98** | 扫描最优 |
| 240 | 8.11 | 1.91 | 基本持平 180 |

同次扫描 VAE 9.71/2.32，仅真实 16.80/3.81。又试了“VAE 与 H2-only
同时加到 180/240”的平衡版本：180/180 宏 9.98、240/240 宏 8.22，
都优于 VAE，但仍不如 120/180。样本量扫描只用于选候选，1 seed 的
绝对值不能作为最终结论；第 5 节对 120 与 180 做了 3 seed × 2 配置全量。
代码只保留 `union_h2` 与 `union_h2_180`；30/60/240/平衡变体已删除，
扫描结果仍保存在 `combined_tune*`、`combined_bal` 目录，供追溯。

### 2.3 筛选过的负结果：anchor

先试过“VAE 采样 + 把 H2 通道按比例替换成正模型”的 anchor 方案：

- `vae_anchor_h2`：H2 通道 = 0.5×VAE + 0.5×正模型；
- `vae_anchor_h2_full`：H2 通道完全用正模型。

5 折 1 seed 筛选中，两者宏 3σ 分别为 14.28%、17.22%，同折 VAE 9.71%、
仅真实 16.80%。H2 通道的物理混合破坏了 VAE 样本的通道间联合结构，
NH3/NO2 也被拖坏，因此弃用；anchor 代码已删除。组合必须保留 VAE
样本的完整三通道，只在其外追加正模型样本。

### 2.4 协议与统计

- 数据：62 条，50 条开发集 5 折 OOF；锁定 12 条不参与。
- 窗口：10 s 前缀，0.2 s 网格。
- seed：20260923 / 20260924 / 20260925。
- 配置：低浓度指数 `p=1.0` 与 `p=0,0.3,1.0`。
- 集成：2 配置 × 3 seed = 6 checkpoint，预测先平均再裁剪到开发集范围。
- 指标：宏与逐气体 3σ、MAPE、中位、p90；配对单位为 seed×fold，
  100,000 次 bootstrap 百分位 95% CI，双侧 Wilcoxon。
- 比较：候选对 real、VAE；同时给出同 run 的 VAE 对 real 基线。

## 3. 全量结果（3 seed × 2 配置，6 checkpoint）

50 条开发集 OOF，宏 3σ / MAPE（%）：

| 臂 | 宏 3σ | 宏 MAPE | 宏中位 | 宏 p90 | H2 3σ | NH3 3σ | NO2 3σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 仅真实 | 11.23 | 2.94 | 2.38 | 6.20 | 11.39 | 6.99 | 15.31 |
| PCA-VAE | **7.59** | **1.89** | **1.37** | 4.13 | 9.51 | **5.10** | **8.15** |
| `union_h2`（120） | 7.71 | 1.90 | 1.39 | 3.67 | 8.76 | 4.97 | 9.41 |
| `union_h2_180`（180） | 7.70 | 1.89 | 1.51 | 3.70 | **7.94** | 5.61 | 9.55 |

配对（15 个 seed×fold，负值表示候选更好）：

| 候选 − 基线 | 宏 3σ 均值差 / pp | 改善 | 95% CI | Wilcoxon p |
|---|---:|---:|---|---:|
| union_h2(120) − real | −3.97 | 15/15 | [−5.30, −2.76] | 0.00006 |
| union_h2(120) − VAE | +0.04 | 8/15 | [−0.57, +0.73] | 0.934 |
| union_h2_180 − real | −4.03 | 14/15 | [−5.55, −2.66] | 0.00018 |
| union_h2_180 − VAE | −0.03 | 9/15 | [−0.72, +0.82] | 0.454 |

分气体配对（union_h2_180 − VAE）：H2 −1.34 pp（12/15，p=0.064）、
NH3 +0.15 pp（p=0.561）、NO2 +1.10 pp（p=0.169）。MAPE 的宏差
+0.03 pp（p=0.847）。两种组合与 VAE 的宏平均都统计打平；120 的
H2/NH3 略好、NO2 更差，180 把 H2 拉得更好、NH3 回吐，宏不变。

## 4. 解释：互补只在气体层面成立

- **H2 确实受益**。VAE 的 H2 形状多样性不足，H2-only 正模型样本补上了
  浓度条件与快慢差异；180 条时 H2 3σ 从 9.51% 降到 7.94%，12/15 对
  改善。这是正模型增强最稳定的部分。
- **NO2 被共享骨干拖累**。H2-only 样本不直接监督 NO2，但会更新共享
  TCN 特征；NO2 头仍只由 VAE 样本训练，输入分布却变了。结果是
  NO2 3σ 从 8.15% 升到 9.55%，5/15 对改善。NH3 在 120 时略好、
  180 时略差，方向不稳定。
- **宏平均打平不是“结合成功”**。组合相对仅真实显著改善（−4.03 pp），
  但没有超过 VAE；相对单独分层正模型的优势不显著（120 条版本
  −1.29 pp，p=0.107）。它只是把误差从 H2 挪到 NO2。若按气体路由：
  H2 用 union_h2_180、NH3/NO2 用 VAE，宏 3σ 约 7.06%，但这是 OOF 上
  按气体选模型，含选择偏差，且需要两个模型，不能当作单模型数据增强
  的收益。
- 样本量扫描的 180 优势没有在全量复现：1 seed 时 180 比 120 低
  0.87 pp，3 seed 时只低 0.01 pp。说明该超参数对 seed/配置平均不稳健。

## 5. 限制

- 15 个 seed×fold 对不是独立样本；fold 之间共享文件，置信区间偏窄。
- 锁定 12 条没有参与任何选择，也没有在这一轮评估；本文数字都是开发集 OOF。
- 组合臂的训练集比 VAE 多 120–180 条 H2-only 合成样本，不是等预算比较；
  它回答“组合是否更好”，不回答“每单位计算是否更好”。
- 样本量 30–240 的扫描只做了 1 seed；120/180 有 3 seed 全量，平衡版本
  180/180、240/240 只有 1 seed。
- 正模型合成仍依赖方案 1 的分层随机效应；没有重新做正模型结构搜索。
- VAE 与 TCN 的超参数沿用旧协议，没有为组合方案单独调参。

## 6. 产物与复现

| 内容 | 路径 |
|---|---|
| 120 条全量 | `artifacts/augmentation/combined_full/` |
| 180 条全量 | `artifacts/augmentation/combined_full180/` |
| 样本量扫描 | `artifacts/augmentation/combined_tune/`、`combined_tune2/`、`combined_bal/` |
| 汇总与配对 | 各目录 `report/`（`combined_report.py`） |
| 最终三图 | `combined_full180/report/union_h2_180/figs/`（PNG+SVG） |
| 脚本 | `tcn_union/experiments/hierarchical_augmentation.py`、`tcn_union/evaluation/combined_report.py`、`tcn_union/evaluation/final_report.py` |

180 条全量复现：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.hierarchical_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --configs 1.0 0,0.3,1.0 \
  --arms real vae union_h2_180 --protocol full \
  --output artifacts/augmentation/combined_full180
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.combined_report \
  --source artifacts/augmentation/combined_full180/predictions.npz \
  --folds artifacts/augmentation/combined_full180/folds.csv \
  --output artifacts/augmentation/combined_full180/report \
  --candidates union_h2_180 --baselines real vae
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.final_report \
  --source artifacts/augmentation/combined_full180/predictions.npz \
  --arm union_h2_180 --run-name combined_full180 \
  --output artifacts/augmentation/combined_full180/report/union_h2_180
```

## 7. 结论

数据层直接拼接正模型与 PCA-VAE **没有找到优于两者的单模型方案**。
最好的 `union_h2_180` 与 VAE 宏平均打平（7.70% vs 7.59%，p=0.454），
H2 更低（7.94% vs 9.51%，12/15，p=0.064）、NO2 更差（9.55% vs 8.15%，
p=0.169）；120 条版本同样是打平。组合相对仅真实显著改善，相对单独
分层正模型均值更低但不显著；“比 VAE 更好”这一目标没有达成。

下一步若还要沿组合方向走，需要让 NO2 头不受 H2-only 样本的共享特征
漂移影响，例如按气体隔离部分骨干、对 NO2 输出加 VAE 教师蒸馏，或
先训练 VAE-only 模型再冻结 NO2 相关层做 H2 微调。这些都不是当前
数据拼接能做到的；在验证之前，PCA-VAE 仍是单一来源里最强的选择。
