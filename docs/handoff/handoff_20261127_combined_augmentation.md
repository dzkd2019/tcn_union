# Handoff：正模型 + PCA-VAE 组合增强（2026-11-27）

## 0. 交接状态

- 仓库：`/home/mrawa/code/deep-learning/tcn_union`
- 分支：`experiment/pure-forward-augmentation`
- 本 handoff 写入前 HEAD：`a3edaa3 docs: prune stale handoffs and index sensor_array docs`
- 远端：`origin` = `git@github.com:dzkd2019/tcn_union.git`
- 本 handoff 写入后按用户指令 commit + push；以 `git log -1` / `git status` 为准
- 本轮提交信息：`feat(tcn): combine forward-model and PCA-VAE augmentation`
- 硬约束：仅 `data/0.csv`–`data/61.csv` 共 62 条，不补测/不补采/不重采；
  `TEST_IDS={2,7,15,20,23,27,31,38,41,46,55,60}` 只做最终描述性评估，
  本轮未参与；`.venv/bin/python`；PNG 300 dpi + SVG、不出 PDF；
  浓度写“原始单位”，不写 ppm。

## 1. 本轮目标与结论

用户要求：

1. 归档上一轮四种训练协议（syn / noise_syn / full / gas）的结果；
   后续不再沿“训练协议消融”方向扩展。
2. 探索把正模型增强与 PCA-VAE 放进同一次训练的数据层组合，
   目标是优于两者各自独立使用；即使超不过 VAE，也判断互补关系。

结论：

- **没有找到优于 PCA-VAE 的单模型数据层组合**。最好的
  `union_h2_180` 宏 3σ **7.70%**，PCA-VAE **7.59%**，配对差
  −0.03 pp（9/15 改善，95% CI [−0.72,+0.82]，p=0.454），统计打平。
- **互补只发生在气体层面**：H2 3σ 9.51% → **7.94%**（12/15，
  p=0.064），NO2 8.15% → 9.55%（p=0.169）；NH3 5.10% → 5.61%。
  组合把误差从 H2 挪到 NO2，宏平均不变。
- 120 条版本 `union_h2` 同样打平：7.71% vs 7.59%（p=0.934），
  H2 8.76%、NH3 4.97%、NO2 9.41%。
- 按气体路由（H2 用 union_h2_180、NH3/NO2 用 VAE）约 **7.06%**，
  但这是在 OOF 上按气体挑模型，含选择偏差，且需要两个模型，
  不能当作单模型数据增强的收益。
- PCA-VAE 仍是当前单一来源里最强的选择。

## 2. 归档

上一轮四协议结果已移到：

- `artifacts/archive/protocol_augmentation_20261126/`
  - `fullrun_syn/`、`fullrun_noise_syn/`、`fullrun_full/`、`fullrun_gas/`
  - `fullrun_comparison/`（四协议合并指标、配对、图）
  - `README.md`（归档说明与当前结论）
- 文档：`sensor_array/docs/archive/protocol_augmentation_20261126.md`
  （顶部有“已归档”banner，指向新主线 `combined_augmentation.md`）

后续不再沿训练协议方向扩展。

## 3. 组合方法

### 3.1 `union_h2`

每折保留完整数量的 PCA-VAE 全气体样本，再额外追加分层正模型的
H2-only 样本；**不削减 VAE 样本**，避免上一轮 50/50 混合把 VAE
的 NH3/NO2 覆盖减半。

| 部分 | 条数 | 监督 | 来源 |
|---|---:|---|---|
| PCA-VAE 全气体 | 120 / 180 | H2/NH3/NO2 | 训练折拟合的 VAE，条件标签 |
| 分层正模型 H2-only | 120 / 180 | 仅 H2 | 方案 1 的分层随机效应正模型 |

H2-only 样本的损失权重为 `[1,0,0]`，VAE 样本为 `[1,1,1]`；
真实曲线与 0.1×MAD 噪声视图仍为 `[1,1,1]`。NH3/NO2 的监督只来自
真实、噪声和 VAE，不会被正模型的 NH3/NO2 预测拉偏。

边缘复制同样分开：VAE 的边缘条件样本全气体，正模型的边缘样本只监督
H2。完整协议其余部分不变：real 40 + noise 40 + 合成 + edge，
6→16 因果 TCN、三气体独立头、两套低浓度配置、3 seed。

### 3.2 协议

| 项目 | 设置 |
|---|---|
| 数据 | 62 条，开发 50 条 5 折 OOF |
| 窗口 | 10 s 前缀，0.2 s 网格 |
| seed | 20260923 / 20260924 / 20260925 |
| 低浓度配置 | `p=1.0`；`p=0,0.3,1.0` |
| 集成 | 2 配置 × 3 seed = 6 checkpoint，先平均再裁剪 |
| 指标 | 宏/逐气体 3σ、MAPE、中位、p90 |
| 配对 | seed×fold 共 15 对；100,000 次 bootstrap 95% CI；双侧 Wilcoxon |
| 锁定 12 | 未参与任何选择或评估 |

## 4. 全量结果

50 条开发集 OOF，宏 3σ / MAPE（%）：

| 臂 | 宏 3σ | 宏 MAPE | 宏中位 | 宏 p90 | H2 3σ | NH3 3σ | NO2 3σ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 仅真实 | 11.23 | 2.94 | 2.38 | 6.20 | 11.39 | 6.99 | 15.31 |
| PCA-VAE | **7.59** | **1.89** | **1.37** | 4.13 | 9.51 | **5.10** | **8.15** |
| `union_h2`（120） | 7.71 | 1.90 | 1.39 | 3.67 | 8.76 | 4.97 | 9.41 |
| `union_h2_180`（180） | 7.70 | 1.89 | 1.51 | 3.70 | **7.94** | 5.61 | 9.55 |

配对（候选 − 基线，负值更好）：

| 候选 − 基线 | 宏 3σ 均值差 / pp | 改善 | 95% CI | Wilcoxon p |
|---|---:|---:|---|---:|
| union_h2(120) − real | −3.97 | 15/15 | [−5.30,−2.76] | 0.00006 |
| union_h2(120) − VAE | +0.04 | 8/15 | [−0.57,+0.73] | 0.934 |
| union_h2_180 − real | −4.03 | 14/15 | [−5.55,−2.66] | 0.00018 |
| union_h2_180 − VAE | −0.03 | 9/15 | [−0.72,+0.82] | 0.454 |

`union_h2_180 − VAE` 分气体：H2 −1.34 pp（12/15，p=0.064）、
NH3 +0.15 pp（p=0.561）、NO2 +1.10 pp（p=0.169）；MAPE 宏差
+0.03 pp（p=0.847）。

## 5. 负结果与筛选

### 5.1 anchor（已删除代码）

先试过“VAE 采样 + 把 H2 通道按比例替换成正模型”：

- `vae_anchor_h2`：H2 = 0.5×VAE + 0.5×正模型；
- `vae_anchor_h2_full`：H2 完全用正模型。

5 折 1 seed 下宏 3σ 分别 14.28%、17.22%（同折 VAE 9.71%、
仅真实 16.80%）。H2 通道的物理混合破坏 VAE 样本的通道间联合结构，
NH3/NO2 也被拖坏；已弃用并删除代码。

### 5.2 样本量扫描（1 seed × 5 折）

| H2-only 条数 | 宏 3σ | 宏 MAPE | 说明 |
|---:|---:|---:|---|
| 30 | 11.01 | 2.63 | 接近仅真实 |
| 60 | 10.51 | 2.24 | 仍弱于 VAE |
| 120 | 8.96 | 2.06 | 首次明显优于 VAE |
| **180** | **8.09** | **1.98** | 扫描最优 |
| 240 | 8.11 | 1.91 | 基本持平 180 |

同次扫描 VAE 9.71%、仅真实 16.80%。平衡版本 180/180 宏 9.98、
240/240 宏 8.22，均不如 120/180。扫描只用于选候选；3 seed 全量后
180 相对 120 的优势缩到 0.01 pp，不稳健。代码只保留 `union_h2`
与 `union_h2_180`，其余变体已删除，扫描结果仍保存在
`artifacts/augmentation/combined_tune*` 与 `combined_bal/`。

## 6. 解释

- **H2 确实受益**：VAE 的 H2 形状多样性不足，H2-only 正模型样本
  补上了浓度条件与快慢差异；180 条时 H2 3σ 9.51% → 7.94%。
- **NO2 被共享骨干拖累**：H2-only 样本不直接监督 NO2，但会更新
  共享 TCN 特征；NO2 头仍只由 VAE 样本训练，输入分布却变了。
  结果是 NO2 3σ 8.15% → 9.55%，仅 5/15 对改善。
- **宏平均打平不是“结合成功”**：组合相对仅真实显著改善，
  相对单独分层正模型均值更低但不显著（−1.29 pp，p=0.107）；
  它只是把误差从 H2 挪到 NO2。

## 7. 代码、文档与产物

### 7.1 代码

- `tcn_union/experiments/hierarchical_augmentation.py`
  - 新增 `_h2_weights`、`_h2_union`、`_h2_union_edge`；
  - `UNION_COUNTS = {"union_h2": (120,120), "union_h2_180": (120,180)}`；
  - `sample_weight` 传给 `_train`，H2-only 合成样本只监督 H2；
  - 修复 `_h2_union` 在 n_hier>120 时 target 条数不足导致的
    sample_weight/y_train 形状不一致。
- `tcn_union/evaluation/combined_report.py`
  - 读 `folds.csv` + `predictions.npz`，输出 `ensemble_metrics.csv`、
    `paired.csv`（100,000 次 bootstrap、双侧 Wilcoxon）。
- `tcn_union/evaluation/final_report.py`：AGENTS.md §10 规定的三张
  1×3 标准图（绝对误差、相对误差、均值 95% CI）。

### 7.2 文档

- 新结果文档：`sensor_array/docs/combined_augmentation.md`
- 归档文档：`sensor_array/docs/archive/protocol_augmentation_20261126.md`
- 图索引：`docs/plotting_guide_tcn_10s.md`（10s-F17–F19 与
  “2026-11-26 正模型 + PCA-VAE 组合增强”节）
- 索引同步：`README.md`、`sensor_array/README.md`、
  `sensor_array/docs/README.md`

### 7.3 结果目录

| 内容 | 路径 |
|---|---|
| 120 条全量 | `artifacts/augmentation/combined_full/` |
| 180 条全量 | `artifacts/augmentation/combined_full180/` |
| 样本量扫描 | `artifacts/augmentation/combined_tune/`、`combined_tune2/`、`combined_bal/` |
| 1 seed 筛选 | `artifacts/augmentation/combined_screen/` |
| 最终三图 | `combined_full180/report/union_h2_180/figs/` |
| 四协议归档 | `artifacts/archive/protocol_augmentation_20261126/` |

## 8. 复现

180 条全量：

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
```

最终标准图：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.final_report \
  --source artifacts/augmentation/combined_full180/predictions.npz \
  --arm union_h2_180 --run-name combined_full180 \
  --output artifacts/augmentation/combined_full180/report/union_h2_180
```

## 9. 验证

- `python -m unittest discover -s tests`：67 项通过（1 skipped）。
- `MPLCONFIGDIR=/tmp/mpl python -m unittest discover -s sensor_array/tests`：
  9 项通过。
- `py_compile` 新脚本通过；`git diff --check -- '*.py' '*.md'` 干净。
- 图件：`combined_full` 与 `combined_full180` 的 report 共 42 个
  `fig_*` 文件（PNG+SVG），0 PDF。
- 当前修改文档相对链接 0 坏链。
- `artifacts/full_protocol_smoke/` 与组合 smoke 目录已删除。

## 10. 限制

- 15 个 seed×fold 对不是独立样本；fold 之间共享文件，CI 偏窄。
- 组合臂训练集比 VAE 多 120–180 条 H2-only 合成样本，不是等预算
  比较；它回答“组合是否更好”，不回答“每单位计算是否更好”。
- 样本量 30–240 的扫描只做 1 seed；120/180 有 3 seed 全量。
- 正模型合成仍依赖方案 1 的分层随机效应，没有重新做正模型结构搜索。
- VAE 与 TCN 超参数沿用旧协议，没有为组合方案单独调参。
- 锁定 12 条没有参与任何选择，也没有在这一轮评估。

## 11. 下一步

1. **保 NO2**：组合的瓶颈是共享骨干被 H2-only 样本带偏。候选：
   隔离部分骨干、冻结 NO2 相关层、对 NO2 输出加 VAE 教师蒸馏，
   或先训练 VAE-only 模型再做 H2 微调。这些已超出数据层拼接。
2. **5 seed 复验**：如果继续主张 H2 收益，至少 5 seed；当前 3 seed
   的宏平均差异小于 seed 波动。
3. **锁定 12**：只在最终候选确定后评估一次，结果不得回调超参。
4. **不要**再沿训练协议消融方向扩展；四协议结果已归档。
5. PCA-VAE 仍是当前单一来源最强；任何新组合都要同 run 配对，
   不能跨 run 比较绝对值。
