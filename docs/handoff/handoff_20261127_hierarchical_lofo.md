# Handoff：分层正模型 62 折留一、四方案增强对比与 NO2 guard（2026-11-27）

## 0. 交接状态

- 仓库：`/home/mrawa/code/graduate/tcn_union`
- 分支：`experiment/no2-preserving-augmentation`
- 本 handoff 写入前 HEAD：`0bd89ea feat(sensor-array): split hierarchical LOFO curves by gas`
- 远端：`origin` = `git@github.com:dzkd2019/tcn_union.git`
- 本 handoff 写入后按用户指令 commit + push；以 `git log -1` / `git status -sb` 为准
- 工作区：写入前 clean；67 项主测试 + 9 项 sensor_array 测试通过
- 硬约束：仅 `data/0.csv`–`data/61.csv` 共 62 条，不补测/不补采/不重采；
  `TEST_IDS={2,7,15,20,23,27,31,38,41,46,55,60}` 只做最终描述性评估，
  本轮未参与任何选择；`.venv/bin/python`；PNG 300 dpi + SVG、不出 PDF；
  浓度写“原始单位”，不写 ppm。

## 1. 本轮完成的三件事

1. 给正模型加入**实验级随机效应**（方案 1），并做 62 折留一。
2. 把 PCA-VAE、仅正模型、VAE+正模型、NO2 guard 四条增强路线放进
   同一套评估规范（`AGENTS.md` §10），输出逐气体 MAPE 与散点图。
3. 针对“正模型增强把 NO2 拖坏”的问题，实现 **NO2 guard**：
   VAE 基座 + 冻结共享骨干与 NH3/NO2 头 + 只微调 H2 头。

结论先说：

- **分层随机效应在 in-sample 拟合上明显更紧，但 62 折留一几乎没有增益。**
  这说明实验间偏差是实验特异的，不能只由浓度预测。
- **NO2 guard 是当前唯一“同轮显著优于 VAE”的方案**，代价是 NO2 本身
  没有下降，只是被完整保住；H2 改善，宏平均下降。
- 四条增强路线的绝对数不能直接排名：它们来自不同轮次，VAE 基线不同。
  必须看同轮配对差。

## 2. 正模型：分层随机效应

### 2.1 模型

均值模型 \(f_j(t;c)\) 是双指数骨架 + 三路独立 MLP 残差，输入
\(c=(c_{H2},c_{NH3},c_{NO2})\) 与时间 \(t\)。在它上面挂三个实验级标量：

\[
r_{ij}(t)=e^{g_{ij}} f_j\!\left(\frac{t}{e^{\ell_{ij}}};c_i\right)+d_{ij}t+\varepsilon_{ij}(t)
\]

\[
R_{ij}(t)=R_{ij}(0)\left[1+r_{ij}(t)\right]
\]

- \(g_{ij}\)：对数增益，整条曲线乘 \(e^g\)；
- \(\ell_{ij}\)：对数时间尺度，时间轴拉伸 \(e^\ell\)；
- \(d_{ij}\)：线性漂移；
- \(\varepsilon\)：测量噪声。

随机效应对 \(\log c\) 的二次设计做条件均值：

\[
b_{ij}=(g_{ij},\ell_{ij},d_{ij})^\top
=A_j^\top\phi(c_i)+\eta_{ij},\quad
\eta_{ij}\sim N(0,\Sigma_j)
\]

实现：`sensor_array/core/hierarchical.py` 的
`HierarchicalRandomEffectsModel`；`ridge=1e-2`、`shrink=0.3`，
增益/时间尺度/漂移边界 0.7/0.7/2e-4。

### 2.2 62 折留一结果

每折用 61 条训练双指数 + 三路 MLP，再拟合随机效应；留出实验只用
**条件均值随机效应**（不采样、不用真值）。

| 模型（62 折） | H2 nRMSE | NH3 nRMSE | NO2 nRMSE | 宏平均 |
|---|---:|---:|---:|---:|
| 双指数 + 三路 MLP | 2.023% | 1.012% | 1.405% | 1.480% |
| 再加实验级随机效应 | 1.968% | 1.031% | 1.402% | **1.467%** |

- 宏平均只改善 **0.013 pp**；H2 略好、NH3 略差、NO2 基本不变；
- 逐实验宏 nRMSE：最小 0.72%、中位 1.45%、最大 9.15%；
  尾部集中在 H2=8000、NH3=10、NO2 较高的角点；
- in-sample 后验重构（已知该实验随机效应）：0.511% → **0.402%**，
  H2 61/62、NH3 62/62、NO2 62/62 改善。

**不要把 0.402% 当泛化误差。** 它是“已知该实验自己的增益/时间尺度/
漂移”时的拟合上界；公平留一是 1.467%。

### 2.3 如何向老师汇报

建议主线：

1. 旧模型宏平均 1.48%，但逐实验 0.72–9.15%，误差按实验整条聚集，
   不是白噪声；
2. 诊断：模型只看到 \(c,t\)，实验间有电极、老化、温湿度、流量等
   未观测差异；NH3 的 \(R(0)\) 有 9.46 倍漂移，说明差异真实存在；
3. 数学：把差异写成增益、时间尺度、漂移三个随机效应；
4. 结果：in-sample 0.402%，留一 1.467% vs 1.480%；
5. 解释：偏差可解释，但不可由浓度外推；它的价值在误差分解、
   不确定性、合成数据多样性，而不是让新实验点预测自动变准；
6. 下一步若要提升新实验精度，需要用前 1–3 秒做后验校准，
   或引入可观测协变量。

## 3. 四条增强路线统一对比

50 条开发集 OOF，10 s，6 checkpoint ensemble（2 配置 × 3 seed）：

| 方法 | 宏 MAPE | 宏 3σ | H2 MAPE | NH3 MAPE | NO2 MAPE | 同轮 VAE 3σ |
|---|---:|---:|---:|---:|---:|---:|
| PCA-VAE（combined 轮） | 1.886 | 7.589 | 2.326 | 1.246 | 2.087 | 7.589 |
| 仅正模型 joint10 | **1.746** | 7.866 | 1.568 | 1.038 | 2.632 | 7.992 |
| VAE+正模型 union_h2_180 | 1.893 | 7.700 | 1.865 | 1.375 | 2.439 | 7.589 |
| NO2 guard | 1.930 | **7.508** | 1.853 | 1.329 | 2.607 | 8.043 |

同轮配对（候选 − 同轮 VAE，负值更好）：

| 候选 | 宏 3σ 差 / pp | 宏 MAPE 差 / pp | 说明 |
|---|---:|---:|---|
| joint10 | −0.126 | **−0.235** | MAPE 最好，3σ 略差 |
| union_h2_180 | +0.111 | +0.007 | 打平，H2 改善、NO2 退化 |
| guard | **−0.535** | −0.130 | 3σ 最好，NO2 被完整保住 |

**注意**：四条路线来自不同轮次；表中“同轮 VAE 3σ”不同（7.589/
7.992/7.589/8.043）。绝对数不能直接说“guard 比 VAE 好 0.08 pp”；
应说“guard 相对同轮 VAE 改善 0.535 pp”。

## 4. NO2 guard

### 4.1 方法

1. 先训练 VAE-only 基座（real+noise+180 VAE+120 VAE 边缘，380 行）；
2. 冻结共享 TCN 骨干 `blocks` 与 NH3/NO2 头；
3. 只训练 H2 头；H2-only 正模型特征在 `torch.no_grad()` 下取，
   不反传到共享骨干；
4. union 数据 680 行（real+noise+180 VAE+180 分层正模型 H2-only+边缘）；
5. 先评估未微调 VAE 头作为 fallback，只有验证宏 3σ 改善才接受。

### 4.2 结果

| 臂 | 宏 3σ | 宏 MAPE | H2 | NH3 | NO2 |
|---|---:|---:|---:|---:|---:|
| 同轮 VAE | 8.04 | 2.06 | 9.04 | 5.14 | 9.94 |
| guard | **7.51** | **1.93** | **7.44** | 5.14 | 9.94 |

- 30 个 fold×seed×配置配对：21 改善、6 持平、3 折略差（最大 +0.11 pp），
  均值 −0.65 pp，Wilcoxon p=5.6×10⁻⁵；
- 3/3 seed 改善：VAE 8.55/10.12/11.52 → guard 7.76/9.63/10.90；
- NH3/NO2 逐位不变（因为冻结）；
- 结论：guard “保住 NO2、改善 H2”，但没有降低 NO2；与历史 union
  7.70% 不是同轮协议，不能直接宣称超过 union。

## 5. 文件与复现

### 正模型

- `sensor_array/core/hierarchical.py`
- `sensor_array/core/mlp_per_gas.py`
- `sensor_array/experiments/evaluate.py`（62 折，保存 `{file}_hierarchical`）
- `sensor_array/experiments/plot_hierarchical_lofo.py`（fig21–23）
- `sensor_array/experiments/evaluate_hierarchical_curves.py`（in-sample 图）
- `sensor_array/artifacts/evaluation/predictions.npz`
- `sensor_array/artifacts/evaluation/curve_summary.csv`
- `sensor_array/artifacts/evaluation/hierarchical_lofo_metrics.csv`
- `sensor_array/figures/model/fig21_hierarchical_lofo_H2.png/.svg`
- `sensor_array/figures/model/fig22_hierarchical_lofo_NH3.png/.svg`
- `sensor_array/figures/model/fig23_hierarchical_lofo_NO2.png/.svg`
- 文档：`sensor_array/docs/hierarchical_forward_fit.md`

### 逆模型增强

- `tcn_union/experiments/full_protocol_augmentation.py`（joint10 / VAE / union）
- `tcn_union/experiments/no2_guard.py`
- `tcn_union/experiments/analyze_no2_guard.py`
- `tcn_union/experiments/compare_augmentation.py`
- `artifacts/augmentation/comparison_20261127/`
- `artifacts/augmentation/no2_guard/full/`
- 文档：`sensor_array/docs/no2_preserving_augmentation.md`、
  `sensor_array/docs/augmentation_comparison_20261127.md`

### 命令

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate --folds 62
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.plot_hierarchical_lofo
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.no2_guard
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.compare_augmentation
```

## 6. 已知边界与下一步

1. **分层随机效应**：公平留一没有增益；不要再把 0.402% 当泛化。
   若继续，优先做“用前 1–3 秒估计随机效应后验”的校准实验，
   而不是继续加大网络。
2. **NO2 guard**：目前只保 NO2、不降 NO2；下一步可试
   “VAE 教师蒸馏 NH3/NO2 头”或“joint10 残差 PCA + guard”，
   但必须同轮 VAE 对照、50 条开发集 OOF、锁定 12 不动。
3. **四条路线**：论文里应写同轮配对差，不要跨轮排名；
   joint10 的 MAPE 最好，guard 的 3σ 最好，VAE 的 NO2 最好。
4. **锁定 12**：只在协议冻结后做一次性描述性评估，不参与选择。
5. **文档**：`sensor_array/docs/README.md` 已按当前结论排序；
   新增结果先更新该索引和 `AGENTS.md` §10 的图件规范。

## 7. 建议的接手顺序

1. 先读 `sensor_array/docs/hierarchical_forward_fit.md` 和
   `sensor_array/docs/augmentation_comparison_20261127.md`；
2. 看 `fig21–fig23` 与 `artifacts/augmentation/comparison_20261127/*/figs`；
3. 若要继续提升正模型：做 early-window 后验校准；
4. 若要继续提升逆模型：先在同一轮里复现 guard vs VAE vs union，
   再决定是否把 guard 作为最终方案；
5. 任何新评估都按 `AGENTS.md` §10：50 条开发集、5 折、3 seed、
   2 配置、6 checkpoint、逐气体 MAPE/NRMSE/3σ/95% CI、PNG+SVG。
