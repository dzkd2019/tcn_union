# sensor_array 绘图指南

本文件逐图登记正问题图件的脚本、函数、数据源和版式。新增或修改图件时，必须在同一次改动中更新本文件。
当前只保留双指数骨架（`double`）与三路独立 MLP 残差（`mlp_per_gas`）两个模型；历史协议图已删除。

## 1. 运行入口

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
```

- 脚本：`sensor_array/plotting/model.py`；入口 `main()`。
- 输出目录：`sensor_array/figures/model/`。
- 每张图同时写 PNG（300 dpi）与同名 SVG，不输出 PDF。
- 数值与统计口径见 [`evaluation.md`](evaluation.md)。

## 2. 样式与保存

- 样式集中在 `sensor_array/plotting/style.py`：`_style()` 设置字体、字号、线宽；`_save(figure, name)` 保存 PNG + SVG。
- 字体为 `["Source Han Sans CN", "SimHei"]`：优先思源黑体，缺失时回退 SimHei。
- 气体配色：H₂ `#C0392B`，NH₃ `#E69F00`，NO₂ `#0072B2`；模型配色：`double` 蓝色 `#4C72B0`，`mlp_per_gas` 绿色 `#009E73`。
- 多面板图统一 `layout="constrained"`；模型图例放图底部，避免遮挡柱顶数值。

## 3. 逐图索引

| 图号 | 输出文件 | 绘图函数 | 数据源 | 版式 |
|---|---|---|---|---|
| fig01 | `figures/model/fig01_design_and_data.{png,svg}` | `_design(experiments)` | `core.data.load_experiments()`，原始 `data/0.csv`–`data/61.csv` | 1×2 |
| fig03 | `figures/model/fig03_curve_error.{png,svg}` | `_curve_error(curve)` | `artifacts/evaluation/curve_metrics.csv` | 1×2 |
| fig07 | `figures/model/fig07_model_architecture.{png,svg}` | `_architecture()` | 无数据依赖，固定架构 | 1×1 流程图 |
| fig09 | `figures/model/fig09_concentration_design.{png,svg}` | `_concentration_design(experiments)` | `core.data.load_experiments()` | 上 4 个 H₂ 切片 + 下 2 个平面汇总 |
| fig10 | `figures/model/fig10_nh3_initial_drift.{png,svg}` | `_nh3_initial_drift(experiments)` | `core.data.load_experiments()` | 1×2 |
| fig12 | `figures/model/fig12_all_curves_H2.{png,svg}` | `_all_curves(experiments)` | `artifacts/evaluation/predictions.npz` | 1×1 |
| fig13 | `figures/model/fig13_all_curves_NH3.{png,svg}` | `_all_curves(experiments)` | 同上 | 1×1 |
| fig14 | `figures/model/fig14_all_curves_NO2.{png,svg}` | `_all_curves(experiments)` | 同上 | 1×1 |
| fig15 | `figures/model/fig15_curve_error_sigma.{png,svg}` | `_curve_error_sigma(experiments)` | `predictions.npz` + 写 `curve_error_stats.csv` | 1×3 |

图号不连续（缺 fig02/04/05/06/08/11）是历史编号保留，不影响当前使用。

## 4. 逐图说明

### fig01 数据设计与示例曲线

- 左图：62 条实验的 (H₂, NH₃) 设计点，颜色为 NO₂ 浓度，双对数坐标；说明设计覆盖不均、H₂ 高端点少。
- 右图：file 0 的三路相对响应原始曲线，给出“数据长什么样”的直观印象。
- 调整入口：`_design()`；颜色 `cmap="viridis"`，点大小 `s=28`，示例文件 `experiments[0]`。

### fig03 十折验证曲线误差

- 左图：逐文件 nRMSE 箱线图，三种气体 × 两个模型；纵轴对数，关闭离群点显示。
- 右图：每条实验的宏平均 nRMSE，横轴双指数骨架、纵轴三路独立 MLP + 双指数；虚线 1:1，左上角标注改善曲线数。
- 调整入口：`_curve_error()`；箱线位置由 `positions` 控制，离群点由 `showfliers` 控制。

### fig07 模型架构

- 1×1 流程图：左侧浓度输入；上支双指数骨架，下支三路独立 MLP 残差；相加后由 (R_j(0)) 还原绝对输出。
- 骨架框给出双指数公式，残差框给出 7 维输入与 (7\to64\to64\to1) 宽度；图中不写参数数量、损失函数和训练细节。
- 调整入口：`_architecture()`；框位置/大小在 `box()` 调用里，箭头在 `arrow()` 调用里。

### fig09 浓度组合覆盖

- 上排：按 H₂ 的 4 个档位分片，每片是 NH₃×NO₂ 的 5×5 网格；格内 1 表示有 1 条实验，0 表示缺失。
- 下排左：H₂–NO₂ 平面，格内数字为跨 NH₃ 的有数据组合数（0–5）。
- 下排右：H₂–NH₃ 平面，格内数字为跨 NO₂ 的有数据组合数（0–5）。
- 用途：说明设计矩阵不是全因子、H₂=32000 几乎只覆盖 NO₂=2、NH₃=50 在 H₂=32000 缺失。
- 调整入口：`_concentration_design()`；二值颜色范围 `vmin=0, vmax=1`，下排汇总范围 `vmin=0, vmax=5`。

### fig10 NH₃ 初始响应漂移

- 左图：NH₃ 初始响应 (R(0)) 对实验序号，颜色为 NH₃ 设定浓度；标注 Pearson (r=0.989)、最大/最小 = 9.46×。
- 右图：同一批点对 NH₃ 设定浓度，颜色为实验序号；标注 Pearson (r=0.965)。
- 用途：说明 NH₃ 初值随采集顺序漂移，并与浓度档位同步上升，二者在设计上无法分离。
- 调整入口：`_nh3_initial_drift()`；颜色映射 `cmap="viridis"` / `cmap="plasma"`。

### fig12–fig14 每气体 62 条实测/十折验证预测

- 每气体一张：实测深灰 `#6F6F6F`（alpha 0.80），预测用该气体颜色；62 条全部画出，不挑选、不排序、不平滑。
- 预测来自十折验证的 `predictions.npz` 中 `mlp_per_gas`，不是全量 in-sample；只画当前主模型，不画骨架对照。
- 版式 1×1，图例只写“实测 / 预测”；标题为“H₂：62 条实测与十折验证预测”等。
- 调整入口：`_all_curves()`；线宽/透明度 `lw`、`alpha`，颜色来自 `GAS_COLORS`。

### fig15 逐文件曲线误差与 σ 带

- 三个面板分别为 H₂/NH₃/NO₂；x 为实验序号 0–61，y 为该文件的曲线 nRMSE（%）。
- 误差按逐气体全体 62 条实验的均方根尺度归一化：

$$
e_i=\frac{\sqrt{\mathrm{mean}_t(\hat y_i-y_i)^2}}{\sqrt{\mathrm{mean}_{k,t}y_k^2}}\times100.
$$

- 实线为 62 条实验的均值，虚线依次为 (\bar e\pm1\sigma,2\sigma,3\sigma)；σ 是实验间离散度，不是 95% 置信区间。
- 同一函数写 `artifacts/evaluation/curve_error_stats.csv`（每气体 mean/median/std/p90/min/max）；这些数字不放进散点图。
- 调整入口：`_curve_error_sigma()`；y 轴下限由 `set_ylim` 控制，σ 线型在 `(1, ":"), (2, "-."), (3, "--")`。

## 5. 新增或修改图件检查清单

1. 在 `sensor_array/plotting/model.py` 新增函数并在 `main()` 调用；函数名以 `_` 开头表示只在本模块使用。
2. 用 `_save(figure, "figXX_name")` 保存，确认 PNG 与 SVG 同名成对，不生成 PDF。
3. 在本文件第 3 节表格和第 4 节说明中登记图号、函数、数据源、版式。
4. 重新运行 `python -m sensor_array.plotting.model`，目视检查中文、下标、图例、数值标签是否被裁剪。
5. 若新增数据文件，在 `evaluation.md` 与 `README.md` 中同步数值和路径。
6. 逆问题图件不放在本目录：每次评估写到 `artifacts/augmentation/<run>/figs/`，登记在对应结果文档。
