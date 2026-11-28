# 正问题结果与图件快照（2026-11-28）

本文件记录当前 `sensor_array/artifacts/` 与 `sensor_array/figures/model/` 的内容。
逆问题 TCN 的产物在仓库根 `artifacts/augmentation/`，不属于本快照。

## 范围

| 目录 | 文件数 | 说明 |
|---|---:|---|
| `artifacts/models/` | 2 | 三路独立 MLP（推荐）、双指数骨架（对照） |
| `artifacts/models/comparison/` | 10 | 四骨架筛选模型：double/first 为单个 seed 初筛加多个 seed 复跑，second/second2 为单个 seed |
| `artifacts/evaluation/` | 5 | 十折验证曲线指标、汇总、预测、误差统计与运行日志 |
| `artifacts/comparison/` | 7 | 四骨架十折验证汇总与预测 |
| `artifacts/comparison/screening/` | 8 | 1 seed 四骨架筛选的审计副本 |
| `figures/model/` | 18 | 9 组 PNG + SVG，300 dpi，无 PDF |
| **合计** | **50** | `SHA256SUMS` 共 50 行 |

## 内容

- `artifacts/models/mlp_per_gas.json`：双指数 + 三路独立 MLP，推荐模型；
- `artifacts/models/double.json`：纯双指数骨架，对照模型；
- `artifacts/comparison/skeleton_summary.csv`：四骨架单 seed 初筛与
  double/first 多个 seed 取平均汇总；
- `artifacts/comparison/comparison_summary.md`：骨架对照的协议、配对统计与取舍；
- `artifacts/comparison/predictions_*.npz`：四个骨架的十折验证预测；
- `artifacts/models/comparison/*.json`：对照模型，命名含骨架与 seed；
- `artifacts/evaluation/curve_metrics.csv`：逐文件、逐模型的曲线 nRMSE；
- `artifacts/evaluation/curve_summary.csv`：曲线均值/中位/p90；
- `artifacts/evaluation/predictions.npz`：十折验证曲线预测；
- `artifacts/evaluation/curve_error_stats.csv`：逐气体误差离散度；
- `artifacts/evaluation/lofo62_run.log`：运行日志。

图件：

- `fig01_design_and_data`：浓度设计点与代表曲线；
- `fig03_curve_error`：十折验证曲线误差；
- `fig07_model_architecture`：双指数 + 三路 MLP 结构；
- `fig09_concentration_design`：62 条实验的浓度组合覆盖；
- `fig10_nh3_initial_drift`：NH₃ 初值随采集顺序漂移；
- `fig12_all_curves_H2`、`fig13_all_curves_NH3`、`fig14_all_curves_NO2`：逐气体 62 条实测与十折验证预测；
- `fig15_curve_error_sigma`：逐文件曲线误差与 ±σ 带。

## 校验

```bash
cd sensor_array
sha256sum -c SHA256SUMS
```

重新运行评估、骨架对照或绘图会改变文件字节，届时需重新生成本清单与校验和。
本快照不替代独立盲测。
