# 组合使用：把 PCA-VAE 与传感器模型生成结合

## 1. 问题

单独用传感器模型生成数据时，H₂ 的误差下降，但 NO₂ 的误差会上升。原因是传感器模型样本和 VAE 样本一起更新共享卷积骨干，NO₂ 输出头看到的输入分布被改变；NO₂ 头本身仍主要依赖 VAE 样本。

组合使用的目标不是让传感器模型替代 VAE，而是把两者结合：传感器模型生成与 PCA-VAE **各自对 H₂、NH₃、NO₂ 三种气体生成数据**；微调时冻结共享骨干和 NH₃/NO₂ 输出头，只更新 H₂ 输出头。这样 H₂ 头可以吸收传感器模型的增益，NH₃/NO₂ 的输出保持不变。

## 2. 方法

1. 先按完整协议训练 PCA-VAE 基座：真实 + 测量噪声 + 120 条 VAE 合成 + VAE 边缘复制，得到共享骨干、三气体输出头与 z-score。
2. 冻结共享 `blocks` 与 NH₃/NO₂ 头；只保留 H₂ 头可训练。
3. 用 120 条 VAE 三气体样本 + 180 条传感器模型三气体样本 + 边缘样本微调 H₂ 头。传感器模型按三气体浓度组合采样，但损失只把 H₂ 头计入；共享特征在传感器模型样本上 `detach`，不反传。这里的正模型是 **hierarchical 正模型**，即双指数骨架 + 三路独立 MLP 残差模型；合成曲线由该模型均值叠加三气体联合低秩残差采样（`forward_prob_joint10`，10 个主成分）得到。它不是早期 `HierarchicalRandomEffectsModel` 随机效应版本。
4. 最终组合预测的 H₂ 取自微调后的 H₂ 头，NH₃/NO₂ 取自 PCA-VAE 的多个 seed 平均预测。

因此 NH₃/NO₂ 的预测与 PCA-VAE 基线逐位一致；组合使用在 H₂ 上获得传感器模型的增益，同时保持 NH₃/NO₂ 不变。

## 3. 协议

- 10 s 前缀、50 条开发集、5 折实验级 OOF；每折 40 条训练、10 条验证。
- 多个 seed（20260923/24/25）× 两套低浓度配置（`1.0` 与 `0,0.3,1.0`）× 6 个 checkpoint；预测先平均再裁剪。
- 量程来自开发集 50 条；锁定 12 条不参与 OOF、早停或选型。
- 指标为逐气体 3σ/MAPE、三气体宏平均，按 `AGENTS.md` 第 10 节。

## 4. 结果

多个 seed 训练后取平均预测，50 条开发集 OOF：

| 方法 | 宏 3σ / % | 宏 MAPE / % | H₂ 3σ / % | NH₃ 3σ / % | NO₂ 3σ / % |
|---|---:|---:|---:|---:|---:|
| PCA-VAE | 7.472 | 1.791 | 9.313 | 4.948 | 8.156 |
| 传感器模型生成 | 7.866 | 1.746 | 8.082 | 4.346 | 11.169 |
| 组合使用 | **7.062** | **1.739** | 8.084 | 4.948 | 8.156 |

组合使用的宏 3σ 和宏 MAPE 均为三种中最低。H₂ 3σ 从 PCA-VAE 的 9.313% 降到 8.084%，与传感器模型生成的 8.082% 基本一致；NH₃/NO₂ 保持 PCA-VAE 的 4.948% / 8.156%。

## 5. 讨论

- 组合使用证明 H₂ 的增益不需要更新共享骨干：冻结骨干和 NH₃/NO₂ 头后，H₂ 3σ 仍从 9.313% 降到 8.084%，NO₂ 不再被拖差。
- 组合使用的宏 3σ 比 PCA-VAE 低 0.410 百分点，宏 MAPE 低 0.052 百分点；这部分增益来自 H₂ 头，NH₃/NO₂ 没有被改变。
- 与“数据层直接拼接”不同，组合使用是结构化的使用方式；当前不再保留拼接方案。

## 6. 复现

```bash
.venv/bin/python -m tcn_union.experiments.no2_guard \
  --folds 5 --seeds 20260923,20260924,20260925 \
  --configs "1.0;0,0.3,1.0" \
  --output artifacts/augmentation/no2_guard/full
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.analyze_no2_guard
```

产物：`artifacts/augmentation/no2_guard/full/{folds,oof,oof_seed,seed_mean}.csv`、`predictions.npz`、`run_config.json`；三方法汇总见 [`augmentation_comparison_20261127.md`](augmentation_comparison_20261127.md)。
