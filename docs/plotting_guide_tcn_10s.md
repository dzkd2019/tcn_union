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
