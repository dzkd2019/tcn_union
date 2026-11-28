# 十折验证：双指数骨架与三路独立 MLP 残差

本文件只回答正问题：给定浓度组合，模型能否重建 0–60 s 的三路相对响应曲线。
逆问题的三方法评估见 [`augmentation_comparison_20261127.md`](augmentation_comparison_20261127.md)。

## 1. 结论

- 十折验证下，三路曲线宏 nRMSE：双指数骨架 **3.30%**，加三路独立 MLP 残差后 **1.48%**。
- 逐气体（H₂ / NH₃ / NO₂）：骨架 5.17% / 1.69% / 3.04%，组合 2.02% / 1.01% / 1.40%。
- 三路独立组合在 **60/62** 条留出曲线改善宏 nRMSE，逐文件配对中位改善 **−1.77 百分点**。
- 骨架形式不是主要瓶颈：双指数与一阶在多次训练取平均后几乎等价（宏 1.5209% 对 1.5261%，p=0.273）；两个二阶骨架更差。
- 当前没有独立的锁定测试；十折验证是正问题唯一的泛化评估。

## 2. 协议

- 按实验划分 10 折，每次留出一折完整曲线，用其余实验训练；逐条留出曲线统计误差。
- 每折先拟合双指数骨架，再用**同一训练折**拟合三路独立 MLP；两模型共享训练数据。
- 骨架用逐实验形状初值 + 加权最小二乘联合拟合；MLP 用骨架输出作为输入，不接触测试折。
- 曲线误差：

$$
\mathrm{nRMSE}_{j}=\frac{\sqrt{\frac1T\sum_t(\hat y_j(t)-y_j(t))^2}}
{\sqrt{\frac1{NT}\sum_{i,t}y_{ij}(t)^2}},
$$

  分母是训练折该通道相对响应的 RMS；宏平均为三路算术平均。
- 预测数组、逐文件指标和汇总表分别写入 `sensor_array/artifacts/evaluation/predictions.npz`、`curve_metrics.csv`、`curve_summary.csv`。

## 3. 曲线结果

| 模型 | H₂ nRMSE / % | NH₃ nRMSE / % | NO₂ nRMSE / % | 宏 nRMSE / % |
|---|---:|---:|---:|---:|
| 双指数骨架 | 5.17 | 1.69 | 3.04 | 3.30 |
| 双指数 + 三路独立 MLP | **2.02** | **1.01** | **1.40** | **1.48** |

逐文件离散度（62 条实验）：

| 模型 | 气体 | 均值 / % | 中位 / % | p90 / % | 标准差 / % |
|---|---|---:|---:|---:|---:|
| 骨架 | H₂ | 5.17 | 4.03 | 8.36 | 3.76 |
| 骨架 | NH₃ | 1.69 | 1.40 | 2.78 | 1.22 |
| 骨架 | NO₂ | 3.04 | 2.62 | 4.27 | 1.44 |
| 组合 | H₂ | 2.02 | 1.08 | 3.54 | 2.55 |
| 组合 | NH₃ | 1.01 | 0.64 | 2.11 | 1.05 |
| 组合 | NO₂ | 1.41 | 1.21 | 2.14 | 0.77 |

组合相对骨架：宏 nRMSE 的改善集中在 H₂（5.17→2.02）和 NO₂（3.04→1.40），NH₃ 从 1.69 降到 1.01。

### 3.1 骨架对比

四骨架在十折验证、单次训练下的筛选结果：

| 骨架 | 宏 nRMSE / % | 参数量/通道 |
|---|---:|---:|
| 双指数 | **1.478** | 23 |
| 一阶 | 1.481 | 15 |
| 两个二阶并联 | 1.634 | — |
| 临界阻尼二阶 | 1.766 | — |

前两名各做多次训练并取平均：双指数 1.5209%，一阶 1.5261%；逐文件配对中位差 +0.0135 百分点，Wilcoxon p=0.273。两者差异不显著，因此保留双指数是因为它的快、慢两个时间尺度可解释，而不是因为显著更准；一阶是等价的 45 参数备选。

## 4. 解释与诊断

- **H₂ 的误差最大**：H₂ 的相对响应幅度最小（60 s 约 0.02–0.03），相同的绝对偏差会被放大；同时 H₂=32000 只有 4 条，且都在 NO₂=2，边角外推困难。
- **NH₃ 的误差最小**：响应幅度最大（60 s 约 0.3–0.4），信噪比高；但 NH₃ 的浓度与采集顺序高度相关（约 0.976），模型学到的是“该顺序下的等效浓度效应”，不能解释为纯浓度因果。
- **NO₂ 为负响应**：骨架用符号项处理，MLP 残差在相对响应上学习，不需要额外变换。
- **初始基线 (R_j(0)) 漂移**：相对化只去掉乘性基线，去不掉与浓度同步的缓慢漂移；fig10 显示 NH₃ 初值随实验序号上升，且与浓度档位同步。
- **误差集中**：少数实验（如 file 0）的宏 nRMSE 明显高于中位数，主要来自 H₂ 的高浓度边角；模型没有文件编号输入，不能靠记忆实验修正。

## 5. 限制

- 62 条实验、每个浓度组合 1 条、无重复；不能评估重复性、气体缺失、零浓度或跨器件泛化。
- 数据只有恒浓度阶跃；时变浓度、温度/湿度变化、长期漂移未验证。
- 三路 MLP 固定 600 轮、无早停；超参数只做了少量筛选，没有嵌套交叉验证。
- 曲线 nRMSE 的分母是训练折 RMS，不同折之间分母略有差异；跨模型比较必须在同一折内完成。
- 所有结论仅适用于 H₂ / NH₃ / NO₂ 和当前采集条件。

## 6. 复现

```bash
# 当前模型评估（GPU 训练 MLP，CPU 拟合骨架）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

# 四骨架筛选（十折验证口径）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first,second,second2 --seeds 0 --max-iter 600

# 前两名多次训练复跑
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first --seeds 20260923,20260924,20260925 --max-iter 600

# 重生成图件
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
```

## 7. 图

- [fig01_design_and_data](../figures/model/fig01_design_and_data.png)：62 条实验的设计点与 file 0 原始曲线。
- [fig03_curve_error](../figures/model/fig03_curve_error.png)：逐文件 nRMSE 箱线图与骨架/组合逐文件对比。
- [fig07_model_architecture](../figures/model/fig07_model_architecture.png)：双指数骨架 + 三路 MLP 的推理路径。
- [fig09_concentration_design](../figures/model/fig09_concentration_design.png)：62 个浓度组合的覆盖与缺失。
- [fig10_nh3_initial_drift](../figures/model/fig10_nh3_initial_drift.png)：NH₃ 初值漂移与浓度混淆。
- [fig12_all_curves_H2](../figures/model/fig12_all_curves_H2.png)：H₂ 的 62 条实测/十折验证预测。
- [fig13_all_curves_NH3](../figures/model/fig13_all_curves_NH3.png)：NH₃ 的 62 条实测/十折验证预测。
- [fig14_all_curves_NO2](../figures/model/fig14_all_curves_NO2.png)：NO₂ 的 62 条实测/十折验证预测。
- [fig15_curve_error_sigma](../figures/model/fig15_curve_error_sigma.png)：逐文件误差散点与 ±1σ/2σ/3σ。
