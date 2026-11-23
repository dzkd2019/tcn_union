# 结果与图件快照（当前分支，2026-11-23）

本文件记录当前分支 `experiment/mlp-double-exponential` 的 `artifacts/` 与
`figures/model/` 内容。MAP 反演、区间校准与降级产物已删除；旧模型代码与图件
只保留在 `docs/archive/` 作历史归档。

## 范围

| 目录 | 文件数 | 说明 |
|---|---:|---|
| `artifacts/evaluation/` | 3 | 62 折曲线指标、汇总与预测 |
| `artifacts/models/` | 2 | 三路独立（推荐）、双指数骨架（对照） |
| `figures/model/` | 10 | 5 组 PNG + 同名 SVG，300 dpi，无 PDF |
| **合计** | **15** | `SHA256SUMS` 共 15 行 |

图件编号为 fig01、fig03、fig07、fig09、fig10：数据设计、留一曲线误差、
模型架构、浓度组合覆盖、NH₃ 初值漂移。fig02/04/05/06/08/11 已删除。

## 内容

- `artifacts/evaluation/curve_metrics.csv`：逐文件、逐模型的曲线 nRMSE；
- `artifacts/evaluation/curve_summary.csv`：曲线均值/中位/p90；
- `artifacts/evaluation/predictions.npz`：62 折曲线预测；
- `artifacts/models/mlp_per_gas.json`：双指数 + 三路独立 MLP，**推荐模型**；
- `artifacts/models/double.json`：纯双指数骨架，对照模型；
- `figures/model/fig01_design_and_data.png/.svg`；
- `figures/model/fig03_curve_error.png/.svg`；
- `figures/model/fig07_model_architecture.png/.svg`；
- `figures/model/fig09_concentration_design.png/.svg`；
- `figures/model/fig10_nh3_initial_drift.png/.svg`。

## 校验

~~~bash
cd sensor_array
sha256sum -c SHA256SUMS
~~~

重新运行评估或绘图会改变文件字节，届时需重新生成本清单与校验和。
本快照不替代独立盲测，也没有独立锁定测试。
