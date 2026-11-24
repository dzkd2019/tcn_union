# 10 秒前缀 TCN 绘图索引

> 本文件登记 **TCN 响应加速（`tcn_union`）的 10 秒前缀图件**，与
> `sensor_array/docs/plotting_guide.md` 分开维护。5 秒前缀图件见
> [`plotting_guide_tcn_5s.md`](plotting_guide_tcn_5s.md)。

## 图件清单

| 图号 | 文件 | 绘图脚本 / 函数 | 数据源 | 版式 | 输出目录 |
|---|---|---|---|---|---|
| 10s-F1 | `fig_10s_relative_error_ci95.{png,svg}` | `artifacts/3sigma_noridge/scripts/plot_ci95_10s.py::main` | `artifacts/3sigma_noridge/all62_predictions.csv`（50 OOF + 12 锁定） | 1×3，H2/NH3/NO2；散点=单个实验，黑菱形/竖线=均值及 95% t 置信区间；OOF 与锁定分列 | `artifacts/3sigma_noridge/figs/` |

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
    fig_H2_signed_error_scatter.png/.svg
    fig_H2_ci95.png/.svg
    fig_NH3_signed_error_scatter.png/.svg
    fig_NH3_ci95.png/.svg
    fig_NO2_signed_error_scatter.png/.svg
    fig_NO2_ci95.png/.svg
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

1. **有符号误差散点图**：每气体 1×2；左为绝对误差 \(\hat c-c\)，右为相对误差
   \(100(\hat c-c)/c\)；横轴实验序号；零线加粗，另画 \(\pm1\sigma,\pm2\sigma,
   \pm3\sigma\) 并在图内标数值；不连线、不平滑；OOF 与锁定测试同图时用底色/分面区分。
2. **95% 置信区间图**：每气体 1×1；只画相对误差均值与 95% CI 误差棒，
   不显示原始散点；标注 \(n\)；多种子时每种一个点。

PNG 300 dpi + 同名 SVG，不输出 PDF；不写 `ppm`。

### 登记

新增运行后，在本文件“图件清单”表补一行 run 级记录，并在图注中写明
`run_name`、绘图脚本/函数、数据源、版式与 `figs/` 路径；不得只生成图不更新索引。
