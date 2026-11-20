# 5 秒前缀 TCN 绘图索引

> 本文件只登记 **TCN 响应加速（`tcn_union`）的 5 秒前缀实验**图件，与
> `sensor_array/docs/plotting_guide.md` 分开维护。10 秒前缀图件见
> [`plotting_guide_tcn_10s.md`](plotting_guide_tcn_10s.md)。

## 图件清单

| 图号 | 文件 | 绘图脚本 / 函数 | 数据源 | 版式 | 输出目录 |
|---|---|---|---|---|---|
| 5s-F1 | `fig_5s_relative_error_all62.{png,svg}` | `artifacts/3sigma_5s/scripts/plot_all62_5s.py::_draw_relative` | `artifacts/3sigma_5s/all62_predictions.csv`（50 OOF + 12 锁定） | 2×2，H2/NH3/NO2/三气体合并；±1/2/3σ 横线；锁定测试底色 | `artifacts/3sigma_5s/figs/` |
| 5s-F2 | `fig_5s_absolute_error_all62.{png,svg}` | 同上 `::_draw_absolute` | 同上 | 1×3，绝对误差（原始单位）；中位/p90；锁定测试底色 | 同上 |
| 5s-F3 | `fig_5s_vs_10s_metrics.{png,svg}` | 同上 `::_draw_metrics` | `artifacts/3sigma_5s/prefix_metrics.csv` + `artifacts/3sigma_noridge/all62_error_stats.csv` | 1×2，OOF 50 的 3σ 与 MAPE；5 s vs 10 s | 同上 |

## 统一规范

- PNG 300 dpi + 同名 SVG，不输出 PDF。
- 字体 `Microsoft YaHei`，负号正常显示；不写 `ppm`（数据未注明单位），绝对误差标为“原始单位”。
- 颜色：H2 `#1F77B4`，NH3 `#D98E04`，NO2 `#C0392B`；锁定测试底色 `#F2D9A6`。
- 3σ 定义：`3 * std(100*(pred-truth)/max(|truth|,1e-9), ddof=1)`；宏平均为三气体算术平均。
- 所有图的数据都来自固定预测文件，绘图脚本不重新训练模型。

## 复现

```bash
.venv/bin/python artifacts/3sigma_5s/scripts/plot_all62_5s.py
```

## 相关文档

- 评估协议与结果：[`docs/tcn_5s_evaluation.md`](tcn_5s_evaluation.md)
- 10 秒方案：[`docs/sigma-target/3sigma_方案.md`](sigma-target/3sigma_方案.md)
