# 62 折留一评估：双指数骨架与三路独立 MLP 残差

## 1. 结论

- 62 折 LOGO 下，曲线宏 nRMSE：双指数骨架 **3.30%**，三路独立 MLP 残差组合 **1.48%**。
- 逐气体（H₂ / NH₃ / NO₂）：骨架 5.17% / 1.69% / 3.04%，组合 2.02% / 1.01% / 1.40%。
- 三路独立组合在 **59/62** 折改善宏 nRMSE，逐文件配对中位改善 **−1.80 百分点**。
- 当前没有独立锁定测试；MAP 浓度反演与区间校准已从本分支删除。

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
- 曲线 nRMSE 只衡量形状，不是浓度误差；本分支不再做 MAP 反演。
- 模型只在恒浓度阶跃下验证；时变浓度未验证。

## 6. 复现

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
```

## 7. 图

- [fig01_design_and_data](../figures/model/fig01_design_and_data.png)：设计与示例曲线。
- [fig03_curve_error](../figures/model/fig03_curve_error.png)：逐文件 nRMSE 与骨架/组合对比。
- [fig07_model_architecture](../figures/model/fig07_model_architecture.png)：模型架构。
- [fig09_concentration_design](../figures/model/fig09_concentration_design.png)：浓度组合覆盖。
- [fig10_nh3_initial_drift](../figures/model/fig10_nh3_initial_drift.png)：NH₃ 初值漂移。
