# Handoff：方案 1 分层随机效应增强（2026-11-26）

## 0. 交接状态

- 仓库：`/home/mrawa/code/graduate/tcn_union`
- 分支：`experiment/pure-forward-augmentation`
- 本 handoff 写入前 HEAD：`6a2b4ee feat(sensor-array): evaluate hierarchical random-effects augmentation`
- 远端：`origin` = `git@github.com:dzkd2019/tcn_union.git`，本地领先 2 个提交
- 本 handoff 写入后按用户指令 commit + push；以 `git log -1` / `git status` 为准
- 硬约束：仅 `data/0.csv`–`data/61.csv` 共 62 条，不补测/不补采/不重采；
  `TEST_IDS={2,7,15,20,23,27,31,38,41,46,55,60}` 只做最终描述性评估，
  本轮未参与；`.venv/bin/python`；PNG 300 dpi + SVG、不出 PDF；
  浓度写“原始单位”，不写 ppm。

## 1. 本轮目标与结论

用户要求：

1. 实现“方案 1”分层随机效应正模型，把每条实验的增益/时间尺度/漂移
   作为条件在浓度上的随机效应内化到正模型，而不是生成时单独做残差 PCA；
2. 额外评估“方案 1 + PCA-VAE 联合数据增强”；
3. 在此之前先整理 `artifacts/` 目录（已完成，见 §6）。

结论：

- **方案 1 相对仅真实有显著正增益**：6 checkpoint 集成宏 3σ
  11.24% → **8.99%**，逐折配对 −2.92 pp、23/30 改善、p<0.001；
  三种气体都下降。
- **方案 1 没有超过 PCA-VAE**：8.99% vs **7.83%**，配对 +0.78 pp、
  14/30、p=0.328。H2 更好（8.35 vs 9.08），NH3/NO2 更差。
- **方案 1 + VAE 也没有超过 PCA-VAE**：8.35% vs 7.83%，p=0.299；
  相对单独方案 1 的逐折改善 +0.07 pp、15/30、p=0.903，不能算稳定收益。
- 结论：方案 1 证明了“参数型随机效应”能补 H2 的实验间差异，但
  NH3/NO2 的剩余误差主要是曲线形状偏差，需要 joint10 的残差 PCA 或
  更强的形状先验；VAE 仍是当前最强单一来源。

## 2. 方案 1 模型

对第 \(i\) 条实验、第 \(j\) 路传感器：

\[
R_{ij}(t)=R_{0,ij}\left[1+\exp(g_{ij})\,h_j(t/\exp(\ell_{ij});c_i)+d_{ij}t\right]
\]

其中 \(h_j\) 是正模型（双指数 + MLP）的相对响应形状，\(g,\ell,d\) 为
实验级随机效应：

\[
g_{ij}=\alpha_j^\top\phi(c_i)+u_{ij},\quad
\ell_{ij}=\beta_j^\top\phi(c_i)+v_{ij},\quad
d_{ij}=\delta_j^\top\phi(c_i)+w_{ij}
\]

\((u,v,w)\sim N(0,\Sigma_j)\)，\(\Sigma_j\) 由训练折残差协方差向对角
收缩 0.3 得到。拟合时先对每条训练实验/气体单独拟合 \((g,\ell,d)\)
（soft-L1），再回归条件均值；生成时从条件正态采样并裁剪到训练范围。

## 3. 协议

与完整旧 TCN 协议一致，只改合成来源：

| 项目 | 设置 |
|---|---|
| 数据 | 62 条，开发 50 条 OOF |
| 折 | 5 折，每折训练 40、验证 10 |
| seed | 20260923 / 20260924 / 20260925 |
| 低浓度配置 | 1.0；0,0.3,1.0 |
| 臂 | real / vae / hier / hier+vae |
| 增强臂数据 | real + 0.1×MAD noise + 120 合成 + 120 边缘复制 |
| hier+vae | 60 hier + 60 VAE，边缘各半 |
| TCN | 6→16 因果 TCN，3 残差块，三气体独立头 |
| 目标变换 | H2 log，NH3 原空间，NO2 log |
| 早停 | 验证宏 3σ，patience 50，最多 500 epoch |
| 集成 | 2 配置 × 3 seed = 6 checkpoint，先平均再裁剪 |
| 锁定 12 | 未参与任何选择或评估 |

3σ = 50 条 OOF 相对误差标准差 × 3；宏平均为三气体算术平均。

## 4. 结果

6 checkpoint 集成（`artifacts/augmentation/hierarchical/full/ensemble.csv`）：

| 方案 | 宏 3σ / % | 宏 MAPE / % | H2 3σ | NH3 3σ | NO2 3σ |
|---|---:|---:|---:|---:|---:|
| 仅真实 | 11.24 | 2.95 | 11.31 | 7.37 | 15.04 |
| PCA-VAE | **7.83** | **2.07** | 9.08 | 5.34 | **9.07** |
| 方案 1（hier） | 8.99 | 2.19 | **8.35** | 6.80 | 11.82 |
| 方案 1 + VAE | 8.35 | 2.10 | 7.71 | 5.41 | 11.92 |

逐 seed 均值（6 个 OOF 的均值±sd）：仅真实 14.66±1.20；
PCA-VAE 10.23±0.90；hier 11.50±0.65；hier+vae 11.05±1.58。

逐折配对（30 个 seed×配置×折，负值表示候选更好；
`artifacts/augmentation/hierarchical/full/paired.csv`）：

| 候选 − 基线 | 均值差 / pp | 中位差 / pp | 改善折数 | Wilcoxon p |
|---|---:|---:|---:|---:|
| hier − real | −2.92 | −3.15 | 23/30 | 0.00009 |
| hier − vae | +0.78 | +0.23 | 14/30 | 0.328 |
| hier+vae − real | −2.85 | −2.63 | 25/30 | 0.00015 |
| hier+vae − vae | +0.85 | −0.06 | 16/30 | 0.299 |
| hier+vae − hier | +0.07 | +0.19 | 15/30 | 0.903 |

注意：30 个观测共享实验/折，p 值只作描述性，不是独立重复。

## 5. 重要修正：早期 pilot 作废

早期 1 seed/1 配置 pilot 曾得到“hier 10.95、hier+vae 10.57、
joint10 9.62”，看起来方案 1 优于 VAE。原因是当时增强臂只喂合成数据，
**漏掉了 `real + noise` 行**；修正为与完整协议一致的
`real + noise + synthetic + edge` 后才得到 §4 的数字。旧 pilot 产物已删除，
不得写进论文或与 §4 比较。

## 6. artifacts 整理（本轮前置工作）

`artifacts/` 原 39 MB、48 个顶层子目录、633 个 tracked 文件，含 34 个
历史 PDF、空目录、`__pycache__`、重复/过时协议。已重组为：

- `artifacts/data/{splits,concentration}`
- `artifacts/inverse/{models,eval_10s,eval_5s,prefix,legacy/...}`
- `artifacts/augmentation/protocols/{full,old,forward}`
- `artifacts/augmentation/pure_forward/{joint10,joint_boot,base,h2,routing}`
- `artifacts/augmentation/hierarchical/full/`（本轮）
- `artifacts/augmentation/diagnostics/...`
- `artifacts/augmentation/vae_gan/...`
- `artifacts/analysis/...`
- `artifacts/figures/...`
- `artifacts/archive/...`

新增 `artifacts/README.md`（目录说明、当前结论入口、旧→新路径映射）。
提交：`eba29ca refactor(artifacts): categorize experiment outputs and remove legacy PDFs`。
`sensor_array/artifacts/` 是另一套正模型产物，未动。

## 7. 代码与产物

新增：

- `sensor_array/core/hierarchical.py`：`HierarchicalRandomEffectsModel`
  （fit / sample / save / load）。
- `tcn_union/experiments/hierarchical_augmentation.py`：四臂实验 runner。
- `tcn_union/experiments/analyze_hierarchical.py`：6 checkpoint 集成与配对汇总。
- `sensor_array/tests/test_hierarchical.py`：dummy 正模型的形状/正值测试。
- `sensor_array/docs/hierarchical_augmentation.md`：方法、协议、结果、限制。

结果目录：

- `artifacts/augmentation/hierarchical/full/folds.csv`：逐 seed/配置/折。
- `.../oof.csv`、`.../summary.csv`、`.../ensemble.csv`、`.../paired.csv`。
- `.../predictions.npz`：6 个 checkpoint 的 OOF 预测。
- `artifacts/archive/logs/hierarchical_full.log`。

## 8. 复现

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl .venv/bin/python \
  -m tcn_union.experiments.hierarchical_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 \
  --configs 1.0 0,0.3,1.0 \
  --arms real vae hier hier+vae \
  --output artifacts/augmentation/hierarchical/full
```

汇总：

```bash
.venv/bin/python -m tcn_union.experiments.analyze_hierarchical \
  artifacts/augmentation/hierarchical/full
```

## 9. 验证

- `python -m unittest discover -s tests`：67 项通过（1 skip）。
- `python -m unittest discover -s sensor_array/tests`：9 项通过。
- `py_compile` 新脚本通过；`git diff --check` 干净。
- 全量 4 臂耗时约 68 min（`fold 4 done (4103s)`）。

## 10. 下一步

1. **方案 1 + joint10 残差**：把参数型随机效应（增益/时间尺度/漂移）
   与 joint10 的低秩形状残差合并，目标同时覆盖 H2 的增益差异和
   NO2 的形状差异；这是当前最可能超过 VAE 的路线。
2. **逐气体路由**：H2 用方案 1，NH3/NO2 用 VAE/joint10；需在开发集
   嵌套选择，不能看锁定集。
3. **公平性**：若要主张“方案 1 更好”，至少 5 seed；当前 3 seed 的
   宏平均差异小于 seed 波动。
4. **锁定 12**：只在最终候选确定后评估一次，结果不得回调超参。
5. **不要**重复早期 pilot 的错误：增强臂必须包含 `real + noise` 行；
   合成数据只改参数更新，不改 scaler/验证。
