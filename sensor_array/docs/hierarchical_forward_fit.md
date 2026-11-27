# 带实验级随机效应的正模型：62 条曲线拟合检查

## 1. 这次评估回答什么

之前报告的正模型曲线误差是“仅浓度 → 曲线”的确定性模型：

- 全量 in-sample：宏 nRMSE 0.51%（H₂ 0.74 / NH₃ 0.32 / NO₂ 0.47）；
- 62 折留一实验：宏 nRMSE 1.48%（H₂ 2.02 / NH₃ 1.01 / NO₂ 1.40）。

方案 1 在正模型外面加了实验级随机效应（增益、时间尺度、漂移）。本次把
62 条实验各自的三个随机效应拟合出来，再用拟合值重构曲线，回答：

> 加上实验间偏差后，正模型能把 62 条曲线解释到什么程度？

## 2. 协议

- 基座：`sensor_array/artifacts/models/mlp_per_gas.json`（双指数骨架 + 三路独立 MLP，全量 62 条训练）。
- 随机效应：`HierarchicalRandomEffectsModel().fit(model, experiments, 10)`，每条实验每路气体拟合 `(log gain, log time-scale, drift)`，soft-L1。
- 重构：用拟合的随机效应逐条生成相对响应，不采样、不使用真值。
- nRMSE：逐实验逐气体 RMSE / 全部 62 条该气体相对响应的 RMS；再对 62 条取平均。
- 这是**后验重构**，不是留一实验预测；不能当作泛化误差。

## 3. 结果

| 模型 | H₂ / % | NH₃ / % | NO₂ / % | 宏 / % |
|---|---:|---:|---:|---:|
| 旧：仅浓度（in-sample） | 0.738 | 0.322 | 0.472 | 0.511 |
| 新：+ 实验级随机效应（拟合） | **0.549** | **0.272** | **0.384** | **0.402** |
| 相对下降 | 25.6% | 15.5% | 18.6% | 21.4% |

逐实验比较：

- H₂：61/62 条改善；
- NH₃：62/62 条改善；
- NO₂：62/62 条改善；
- 中位数：H₂ 0.63→0.52%，NH₃ 0.22→0.18%，NO₂ 0.45→0.38%。

图：

- `sensor_array/figures/model/fig12_hierarchical_62_curves.{png,svg}`：三气体各 62 条实测（灰）与重构（气体色）曲线；
- `sensor_array/figures/model/fig13_hierarchical_nrmse_compare.{png,svg}`：逐实验 nRMSE 新旧对比（1:1 线下方为改善）。

## 4. 怎么解释

- 实验级偏差确实主要落在 H₂：H₂ 的 nRMSE 下降最多（0.74→0.55），说明它的实验间差异主要是增益/时间尺度，而不是均值形状。
- NH₃/NO₂ 的改善较小（0.32→0.27、0.47→0.38），说明它们的剩余误差更多是**曲线形状**，不是三个标量随机效应能完全吸收的。
- 这只说明“在已知这条实验的随机效应时，模型能拟合得多好”。对未见实验，随机效应未知，必须从条件先验采样；那部分的误差应看留一实验，而不是这张图。

## 5. 复现

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate_hierarchical_curves
```

输出：`sensor_array/artifacts/evaluation/hierarchical_curves.csv`、
`hierarchical_curve_predictions.npz` 与上述两图。
