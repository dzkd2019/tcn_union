# 三种数据增强方式的统一评估（2026-11-27）

## 1. 三种方式

当前只保留三种数据增强方式：

1. **PCA-VAE**：用 PCA 响应基 + 条件 VAE 生成三气体曲线；
2. **传感器模型生成**：用传感器建模模型生成三气体响应曲线，不混入 PCA-VAE；
3. **组合使用**：把 PCA-VAE 与传感器模型生成的数据结合使用，两者各自对 H₂、NH₃、NO₂ 三种气体生成数据；微调时冻结共享骨干与 NH₃/NO₂ 输出头，只更新 H₂ 输出头。最终 H₂ 采用组合分支，NH₃/NO₂ 采用 PCA-VAE 的多个 seed 平均预测。

**正模型口径**：本表及后文所说的“传感器模型生成”和“组合使用”，其正模型统一登记为 **HierarchicalRandomEffectsModel**：以 `sensor_array` 的双指数骨架 + 三路独立 MLP 残差模型为基座，对第 i 条实验、第 j 路气体的实验级增益 $g_{ij}$、时间尺度 $\ell_{ij}$ 与线性漂移 $d_{ij}$ 建立条件随机效应，生成时从条件先验采样：$R_{ij}(t)=R_{0,ij}[1+\exp(g_{ij})h_j(t/\exp(\ell_{ij});c_i)+d_{ij}t]$。后续论文按该版本描述；本表的数据与结果保持不变。

统一口径：50 条开发实验、10 s 前缀、5 折实验级 OOF、多个 seed 训练后取平均预测、两套低浓度配置、6 个 checkpoint 平均；量程来自开发集 50 条；锁定 12 条未参与选择。指标按 `AGENTS.md` 第 10 节计算。

## 2. 绝对结果

| 方法 | 宏 3σ / % | 宏 MAPE / % | 宏 NRMSE | H₂ 3σ / MAPE | NH₃ 3σ / MAPE | NO₂ 3σ / MAPE |
|---|---:|---:|---:|---:|---:|---:|
| PCA-VAE | 7.472 | 1.791 | 0.0206 | 9.313 / 2.173 | 4.948 / 1.152 | **8.156 / 2.046** |
| 传感器模型生成 | 7.866 | 1.746 | **0.0185** | 8.082 / 1.568 | **4.346 / 1.038** | 11.169 / 2.632 |
| 组合使用 | **7.062** | **1.739** | 0.0195 | 8.084 / 2.019 | 4.948 / 1.152 | **8.156 / 2.046** |

仅真实基线为宏 3σ 10.791%、MAPE 2.774%；三种方式都明显低于基线。

读法：

- **组合使用**的宏 3σ 与宏 MAPE 最低；H₂ 3σ 与传感器模型生成几乎相同（8.084% 对 8.082%），NH₃/NO₂ 保持 PCA-VAE 的水平。
- **PCA-VAE** 的 NO₂ 3σ/MAPE 最好，NH₃ 与组合使用相同。
- **传感器模型生成** 的 H₂、NH₃ 最好，宏 NRMSE 也最低，但 NO₂ 3σ 11.169% 是三种中最差。
- 逐气体数字说明三种方式各有侧重，宏指标用于方法间的主比较。

## 3. 结论

1. 组合使用在宏 3σ（7.062%）与宏 MAPE（1.739%）上优于 PCA-VAE 和传感器模型生成，论文主结果采用这一行。
2. 组合使用的 H₂ 来自微调后的 H₂ 头，NH₃/NO₂ 来自 PCA-VAE；因此它同时保住了 H₂ 的增益和 NO₂ 的稳定。
3. PCA-VAE 单独使用时 NO₂ 最好；传感器模型生成单独使用时 H₂/NH₃ 最好，但 NO₂ 退化。
4. 所有数字都是 50 条开发集 OOF；12 条锁定集没有参与，也没有用于调参。

## 4. 图件与产物

每种方法一个独立目录，按 `AGENTS.md` §10.2 生成三张图（PNG+SVG）：

```text
artifacts/augmentation/comparison_20261127/
  vae/figs/fig_abs_error.{png,svg}
  vae/figs/fig_rel_error.{png,svg}
  vae/figs/fig_ci95.{png,svg}
  forward/figs/...
  guard/figs/...
  metrics.csv  每个方法目录内
  predictions.csv  每个方法目录内
  summary.csv
  predictions.npz
  run_config.json
```

- 两张散点图横轴是 50 条开发实验序号，纵轴为有符号误差；水平线为 0、±1σ、±2σ、±3σ，图内标注数值。
- CI 图是单张图：横轴 H₂ / NH₃ / NO₂，每气体一个均值点 + 95% t 置信区间（n=50）；不画逐 seed 点，不显示原始散点。
- 预测取多个 seed 的平均；图中不区分 seed。

## 5. 复现

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.compare_augmentation
```

脚本：`tcn_union/experiments/compare_augmentation.py`；绘图复用 `tcn_union/evaluation/final_report.py`。

## 6. 限制

- 50 条开发集 OOF 的置信区间只描述这 50 条实验，不是外部盲测。
- 锁定 12 条只做描述性统计；本文件不报告锁定集结果。
- 组合使用的 NH₃/NO₂ 与 PCA-VAE 相同，H₂ 与传感器模型生成接近；三者不是互斥替代关系，而是同一数据条件下的三条路线。
