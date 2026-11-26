# 结果与图件快照（当前分支，2026-11-26）

本文件记录当前分支 `experiment/pure-forward-augmentation` 的 `sensor_array/artifacts/` 与
`sensor_array/figures/model/` 内容。MAP 反演、区间校准与降级产物已删除；旧模型
代码与图件已删除，历史版本可用 `git log` 查看。逆问题 TCN 的新产物在仓库根
`artifacts/augmentation/protocols/forward/`、`artifacts/augmentation/protocols/old/`
与 `artifacts/augmentation/protocols/full/`，不属于本快照的 SHA 校验范围。

## 范围

| 目录 | 文件数 | 说明 |
|---|---:|---|
| `artifacts/inverse/legacy/evaluation/` | 4 | 62 折曲线指标、汇总、预测与逐文件 σ 统计 |
| `artifacts/models/` | 2 | 三路独立（推荐）、双指数骨架（对照） |
| `artifacts/models/comparison/` | 10 | 骨架筛选模型：double/first 各 1+3 seed，second/second2 各 1 seed |
| `artifacts/comparison/` | 7 | 骨架对照汇总与四个骨架的 62 折预测 |
| `artifacts/comparison/screening/` | 8 | 1 seed 四骨架筛选的审计副本 |
| `figures/model/` | 26 | 13 组 PNG + SVG，300 dpi，无 PDF |
| **合计** | **57** | `SHA256SUMS` 共 57 行 |

图件编号为 fig01、fig03、fig07、fig09、fig10、fig12–fig19：数据设计、留一曲线
误差、模型架构、浓度组合覆盖、NH₃ 初值漂移、每气体 62 条曲线总图、逐文件误差
σ 散点、正模型增强 TCN 的四臂 OOF 与逐折配对、旧训练协议下的增强来源
对照，以及完整旧协议（噪声/边缘/两套低浓度/6 checkpoint）的四臂对照。
fig02/04/05/06/08/11 已删除。

## 内容

- `artifacts/inverse/legacy/evaluation/curve_metrics.csv`：逐文件、逐模型的曲线 nRMSE；
- `artifacts/inverse/legacy/evaluation/curve_summary.csv`：曲线均值/中位/p90；
- `artifacts/inverse/legacy/evaluation/predictions.npz`：62 折曲线预测；
- `artifacts/inverse/legacy/evaluation/curve_error_stats.csv`：逐气体误差离散度；
- `artifacts/models/mlp_per_gas.json`：双指数 + 三路独立 MLP，**推荐模型**；
- `artifacts/models/double.json`：纯双指数骨架，对照模型；
- `artifacts/comparison/skeleton_summary.csv`：四骨架 1 seed 与
  double/first 3 seed 汇总；
- `artifacts/comparison/comparison_summary.md`：骨架对照的协议、配对统计与取舍；
- `artifacts/comparison/predictions_*.npz`：四个骨架的 62 折预测；
- `artifacts/models/comparison/*.json`：对照模型，命名含骨架与 seed；
- `figures/model/fig01_design_and_data.png/.svg`；
- `figures/model/fig03_curve_error.png/.svg`；
- `figures/model/fig07_model_architecture.png/.svg`；
- `figures/model/fig09_concentration_design.png/.svg`；
- `figures/model/fig10_nh3_initial_drift.png/.svg`；
- `figures/model/fig12_all_curves_H2.png/.svg`；
- `figures/model/fig13_all_curves_NH3.png/.svg`；
- `figures/model/fig14_all_curves_NO2.png/.svg`；
- `figures/model/fig15_curve_error_sigma.png/.svg`；
- `figures/model/fig16_forward_augmentation.png/.svg`：四臂 OOF 宏 3σ/MAPE；
- `figures/model/fig17_forward_augmentation_paired.png/.svg`：逐折配对散点；
- `figures/model/fig18_old_protocol_augmentation.png/.svg`：旧训练协议下
  仅真实 / PCA-VAE / 正模型 / 混合臂的增强来源对照；
- `figures/model/fig19_full_protocol_augmentation.png/.svg`：完整旧协议
  （0.1×噪声 + 边缘 ×4 + 两套低浓度 + 6 checkpoint）的四臂对照。

根目录 `artifacts/augmentation/protocols/forward/` 另含 TCN 四臂的 `folds.csv`、`oof.csv`、
`summary.csv`、`real_dup.csv`、`real_dup_summary.csv`、`predictions.npz`、
`run_config.json`；根目录 `artifacts/augmentation/protocols/old/` 含增强来源对照的
`folds.csv`、`oof.csv`、`summary.csv`、`predictions.npz`、`run_config.json`；
根目录 `artifacts/augmentation/protocols/full/` 含完整旧协议四臂的
`folds.csv`、`fold_seed.csv`、`oof_seed.csv`、`oof.csv`、`summary.csv`、
`paired.csv`、`predictions.npz`、`summary.md`、`run_config.json`；
它们不在 `sensor_array/SHA256SUMS` 中。

## 校验

~~~bash
cd sensor_array
sha256sum -c SHA256SUMS
~~~

重新运行评估、骨架对照或绘图会改变文件字节，届时需重新生成本清单与校验和。
本快照不替代独立盲测，也没有独立锁定测试。
