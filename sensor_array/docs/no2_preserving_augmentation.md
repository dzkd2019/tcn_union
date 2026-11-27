# NO2 保持的 H2 扩充增强（guard）

## 1. 问题

组合增强 `union_h2_180` 用 180 条 VAE 全气体样本 + 180 条正模型 H2-only 样本训练一个 TCN。H2 从 9.51% 降到 7.94%（3σ），但 NO2 从 8.15% 升到 9.55%，宏平均与纯 VAE 打平（7.70% vs 7.59%）。原因是 H2-only 样本也更新了共享卷积骨干，NO2 头的输入分布被改变；NO2 头本身仍只由 VAE 样本训练。

## 2. 方法

guard 把“共享表征”和“H2 头”分开：

1. 先按完整协议训练 VAE-only 模型（real + noise + 180 VAE + VAE 边缘复制），保存权重与 z-score。
2. 冻结共享 `blocks` 和 NH3/NO2 头，只保留 H2 头可训练。
3. 用 union 数据（real + noise + 180 VAE + 180 分层正模型 H2-only + 边缘）微调 H2 头；共享特征在 H2-only 样本上 `detach`，不反传。
4. 先评估未微调的 H2 头，作为 fallback；只有验证宏 3σ 改善才接受微调结果，否则退回 VAE 头。

因此 NH3/NO2 的输出与 VAE 基线逐位一致，H2 头可以吸收 H2-only 样本；guard 的宏平均不会比 VAE 差。

## 3. 协议

10 s 前缀、50 条开发集 5 折、3 seed × 2 配置、6 checkpoint、裁剪；锁定 12 条不参与；指标为逐气体 3σ/MAPE 与三气体宏平均，按 AGENTS §10。

## 4. 结果

### 4.1 5 折 pilot（1 seed，1 配置）

| 臂 | 宏 3σ / % | H2 3σ / % | NH3 3σ / % | NO2 3σ / % | 宏 MAPE / % |
|---|---|---|---|---|---|
| VAE | 9.06 | 9.75 | 7.43 | 10.02 | 2.25 |
| guard | **8.29** | **7.41** | 7.43 | 10.02 | **2.06** |

guard 的 NH3/NO2 与 VAE 逐位相同（冻结骨干和两头），H2 从 9.75% 降到 7.41%，宏平均降 0.78 pp。逐折 3/5 改善、2/5 因 H2 头轻微过拟合而略差；因此正式版加入“先评估未微调 VAE 头、只有验证改善才接受”的 fallback。

### 4.2 全量（3 seed × 2 配置 × 5 折）

3 seed 预测平均后的 OOF（50 条开发集，6 checkpoint 集成口径）：

| 臂 | 宏 3σ / % | 宏 MAPE / % | H2 3σ / % | NH3 3σ / % | NO2 3σ / % |
|---|---|---|---|---|---|
| VAE | 8.04 | 2.06 | 9.04 | 5.14 | 9.94 |
| guard | **7.51** | **1.93** | **7.44** | 5.14 | 9.94 |

逐 seed OOF 宏 3σ：VAE 8.55/10.12/11.52，guard 7.76/9.63/10.90，3/3 改善；NH3/NO2 完全一致。30 个 fold×seed×配置配对中 21 改善、6 持平、3 折略差（最大 +0.11 pp），均值 −0.65 pp，Wilcoxon p=5.6×10⁻⁵。逐折均值（不做 seed 平均）为 VAE 10.17、guard 9.51；两者口径不同，正文用 ensemble 数字。

注意：这里的 VAE 基线（8.04%）与之前 union 实验的 7.59% 不能直接比，因为随机种子/聚合口径不同；guard 的合法比较是同一轮 VAE 基线。guard 的 7.51% 与历史 union 的 7.70% 接近但未做同轮配对，不能直接宣称超过 union。

## 5. 讨论

- guard 证明“H2-only 样本不必更新共享骨干”：冻结骨干和 NH3/NO2 头后，H2 仍能从 9.04% 降到 7.44%（ensemble），NO2 不再像 union_h2 那样从 8.15% 升到 9.55%。
- 代价是 H2 头的输入特征仍由 VAE 数据训练，H2 增益受限于表征；guard 比同轮 VAE 低 0.53 pp（7.51 vs 8.04），但没有改善 NO2 本身，只是不再拖累 NO2。
- 下一步可把冻结策略与 joint10 的残差 PCA 结合：共享骨干由 VAE/joint10 更新，H2 头单独吸收新组合样本；或对 NO2 头做教师蒸馏，允许骨干有限更新。若要宣称优于历史 union 的 7.70%，需在同一脚本/同一 seed 下补 union 臂做配对。

## 6. 复现

```bash
.venv/bin/python -m tcn_union.experiments.no2_guard \
  --folds 5 --seeds 20260923,20260924,20260925 \
  --configs "1.0;0,0.3,1.0" \
  --output artifacts/augmentation/no2_guard/full
```

产物：`artifacts/augmentation/no2_guard/full/{folds,oof}.csv`、`predictions.npz`、`run_config.json`。
