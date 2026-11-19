# 结果与图件快照（当前分支，2026-11-19）

本文件记录当前分支 `experiment/mlp-double-exponential` 的 `artifacts/` 与
`figures/model/` 内容。旧模型（一阶骨架、低秩残差、共享 MLP）的代码、artifacts、
图件与 P6 快照已从本分支删除，不在本清单中。

## 范围

| 目录 | 文件数 | 说明 |
|---|---:|---|
| `artifacts/evaluation/` | 5 | 双指数骨架与三路独立 MLP 的曲线、下游、汇总与预测 |
| `artifacts/models/` | 2 | 三路独立（推荐）、双指数骨架（对照） |
| `artifacts/calibration/` | 3 | P4 区间校准的逐条区间、汇总与分组复核 |
| `figures/model/` | 12 | 6 组 PNG + 同名 SVG，300 dpi，无 PDF |
| **合计** | **22** | `SHA256SUMS` 共 22 行 |

图件编号为 `fig01`–`fig06`：数据设计与示例曲线、曲线示例、留一曲线误差、
下游浓度误差、P4 区间校准、五个代表实验的生成曲线。所有图均同时有 PNG 与 SVG。

## 内容

- `artifacts/evaluation/curve_metrics.csv`：逐文件、逐模型的曲线 nRMSE；
- `artifacts/evaluation/curve_summary.csv`：曲线均值/中位/p90；
- `artifacts/evaluation/downstream.csv`：5 s / 10 s 逐文件浓度估计与标准误；
- `artifacts/evaluation/downstream_summary.csv`：下游中位/p90；
- `artifacts/evaluation/predictions.npz`：62 折曲线预测；
- `artifacts/models/mlp_per_gas.json`：双指数 + 三路独立 MLP，**推荐模型**；
- `artifacts/models/double.json`：纯双指数骨架，对照模型；
- `artifacts/calibration/intervals.csv`：逐条区间与覆盖标记；
- `artifacts/calibration/summary.csv`：按方法/数据划分/窗口/气体汇总；
- `artifacts/calibration/by_group.csv`：NH₃ 档位与采集块分组覆盖率；
- `figures/model/fig01_design_and_data.png/.svg`；
- `figures/model/fig02_curve_examples.png/.svg`；
- `figures/model/fig03_curve_error.png/.svg`；
- `figures/model/fig04_downstream.png/.svg`；
- `figures/model/fig05_interval_calibration.png/.svg`；
- `figures/model/fig06_representative_curves.png/.svg`。

## 校验

~~~bash
cd sensor_array
sha256sum -c SHA256SUMS
~~~

重新运行评估、校准或绘图会改变文件字节，届时需重新生成本清单与校验和。
本快照不替代独立盲测，也不改变 12 条锁定测试只作描述性复核的约定。
