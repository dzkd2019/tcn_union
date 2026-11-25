# 10 秒前缀 TCN 绘图索引

> 本文件登记 **TCN 响应加速（`tcn_union`）的 10 秒前缀图件**，与
> `sensor_array/docs/plotting_guide.md` 分开维护。5 秒前缀图件见
> [`plotting_guide_tcn_5s.md`](plotting_guide_tcn_5s.md)。

> **2026-11-25 规范变更**：对比实验的三种气体合并到同一张图，但三个指标
> 分开：`fig_abs_error`、`fig_rel_error`、`fig_ci95` 各一张 1×3 图
> （列为 H₂/NH₃/NO₂）。表中 10s-F2–F5 已按新规范重绘。

## 图件清单

| 图号 | 文件 | 绘图脚本 / 函数 | 数据源 | 版式 | 输出目录 |
|---|---|---|---|---|---|
| 10s-F1 | `fig_10s_relative_error_ci95.{png,svg}` | `artifacts/3sigma_noridge/scripts/plot_ci95_10s.py::main` | `artifacts/3sigma_noridge/all62_predictions.csv`（50 OOF + 12 锁定） | 1×3，H2/NH3/NO2；散点=单个实验，黑菱形/竖线=均值及 95% t 置信区间；OOF 与锁定分列 | `artifacts/3sigma_noridge/figs/` |
| 10s-F2 | `{fig_abs_error,fig_rel_error,fig_ci95}.{png,svg}` | `tcn_union/evaluation/final_report.py::run` | `artifacts/full_protocol_augmentation/predictions.npz`，arm=real | 3 张 1×3：绝对误差散点、相对误差散点、均值 95% CI；每张列为 H2/NH3/NO2；散点用 ensemble，CI 每 seed 一点；OOF 50，n=50 | `artifacts/full_protocol_augmentation/report/real/figs/` |
| 10s-F3 | `{fig_abs_error,fig_rel_error,fig_ci95}.{png,svg}` | `tcn_union/evaluation/final_report.py::run` | 同上，arm=vae | 同上 | `artifacts/full_protocol_augmentation/report/vae/figs/` |
| 10s-F4 | `{fig_abs_error,fig_rel_error,fig_ci95}.{png,svg}` | `tcn_union/evaluation/final_report.py::run` | `artifacts/routing_augmentation/predictions.npz`，arm=routing | 同上；routing 为 H2 用 forward+vae、NH3/NO2 用 vae | `artifacts/routing_augmentation/report/figs/` |
| 10s-F5 | `{fig_abs_error,fig_rel_error,fig_ci95}.{png,svg}` | `tcn_union/evaluation/final_report.py::run` | `artifacts/h2_augmentation_full/predictions.npz`，arm=forward_cond_h2 | 同上；H2-only 正模型 + VAE，3 seed × 2 配置 | `artifacts/h2_augmentation_full/report/forward_cond_h2/figs/` |

## 统一规范

- PNG 300 dpi + 同名 SVG，不输出 PDF。
- 字体 `Microsoft YaHei`，负号正常显示；不写 `ppm`（数据未注明单位）。
- 颜色：H2 `#1F77B4`，NH3 `#D98E04`，NO2 `#C0392B`。
- 95% 置信区间定义：对每组 \(n\) 个相对误差，
  \(\bar e \pm t_{0.975,n-1}\,s/\sqrt{n}\)，其中 \(s\) 为样本标准差。
  这里画的是**均值的 95% 置信区间**，不是单个实验的 95% 预测区间；
  单个实验的离散程度由散点本身展示。
- OOF 50 与锁定 12 分开统计，不混合；锁定 12 只作一次性测试描述。
- 所有图的数据都来自固定预测文件，绘图脚本不重新训练模型。

## 复现

```bash
.venv/bin/python artifacts/3sigma_noridge/scripts/plot_ci95_10s.py
```

## 相关文档

- 10 秒方案与 3σ 结果：[`docs/sigma-target/3sigma_方案.md`](sigma-target/3sigma_方案.md)
- 5 秒对照：[`docs/tcn_5s_evaluation.md`](tcn_5s_evaluation.md)

## 最终评估图件与目录规范（2026-11-24）

本节适用于每一次 TCN 浓度反演评估（OOF、验证、锁定测试、集成），与
`AGENTS.md` §10 配套；图件、指标、目录必须同时满足。

### 目录

每次评估运行一个独立目录：

```text
artifacts/<run_name>/
  predictions.csv
  metrics.csv
  figs/
    fig_abs_error.png/.svg
    fig_rel_error.png/.svg
    fig_ci95.png/.svg
```

`<run_name>` 至少包含模型、窗口、协议与 seed，例如
`tcn_10s_full_seed20260923`。同一次运行的所有图件放在同一个 `figs/`；
不与其他 run、其他窗口、其他模型混放。TCN 逆问题图件不放进
`sensor_array/figures/`（那里只放正问题图）。

### 必报指标（`metrics.csv`）

逐气体（H2/NH3/NO2）报告：`n`、`MAPE_percent`、`NRMSE`、
`three_sigma_percent`、`ci95_low_percent`、`ci95_high_percent`、
`mean_relative_percent`；并报告三气体宏平均 `macro_MAPE_percent`、
`macro_NRMSE`、`macro_three_sigma_percent`。定义与分母口径见 `AGENTS.md`
§10.1；NRMSE 的浓度量程来源必须写进 `metrics.csv` 旁注或运行配置。

### 必报图件

1. **有符号绝对误差散点图** `fig_abs_error`：1×3，列为 H₂/NH₃/NO₂；
   纵轴 \(\hat c-c\)。
2. **有符号相对误差散点图** `fig_rel_error`：1×3，列为 H₂/NH₃/NO₂；
   纵轴 \(100(\hat c-c)/c\)。
3. **均值与 95% 置信区间图** `fig_ci95`：1×3，列为 H₂/NH₃/NO₂；
   只画相对误差均值与 95% CI 误差棒，不显示原始散点；标注 \(n\)；多种子时每种一个点。

两张散点图横轴为实验序号；零线加粗，另画 \(\pm1\sigma,\pm2\sigma,
\pm3\sigma\) 并在图内标数值；不连线、不平滑；散点用 ensemble（无则主 seed）；
OOF 与锁定测试同图时用底色/分面区分。

PNG 300 dpi + 同名 SVG，不输出 PDF；不写 `ppm`。

### 登记

新增运行后，在本文件“图件清单”表补一行 run 级记录，并在图注中写明
`run_name`、绘图脚本/函数、数据源、版式与 `figs/` 路径；不得只生成图不更新索引。

### 2026-11-25 正模型增强与按气体路由

- 结果文档：`sensor_array/docs/forward_augmentation_20261125.md`
- 复现：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.routing
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.final_report \
    --source artifacts/routing_augmentation/predictions.npz \
    --output artifacts/routing_augmentation/report --arm routing \
    --run-name routing_augmentation
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.evaluation.final_report \
    --source artifacts/h2_augmentation_full/predictions.npz \
    --output artifacts/h2_augmentation_full/report/forward_cond_h2 \
    --arm forward_cond_h2 --run-name h2_augmentation_full
```

- 四个 run（real / vae / routing / forward_cond_h2）的指标分别见各目录 `metrics.csv`；
  NRMSE 量程统一用开发集 50 条逐气体极差，已写入 `run_config.json` 的
  `range_source=development_50`。锁定 12 条未参与本轮评估。
