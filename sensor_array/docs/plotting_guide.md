# sensor_array 绘图指南（当前模型）

本文件逐图记录绘图脚本、绘图函数、数据源和版式。每次新增图片或修改绘图脚本，都要同步更新本文件。
当前分支只保留两个模型：双指数骨架（<code>double</code>）与三路独立 MLP 残差 + 双指数（<code>mlp_per_gas</code>）。
历史脚本在 <code>docs/archive/plotting_guide.md</code> 中仅作归档。图片对应的数值与统计口径见
[<code>evaluation.md</code>](evaluation.md)。

## 1. 运行入口

~~~bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
~~~

脚本：<code>sensor_array/plotting/model.py</code>；入口 <code>main()</code>。
所有图片写入 <code>sensor_array/figures/model/</code>，同时输出 PNG（300 dpi）与同名 SVG，不输出 PDF。
本指南只登记仍然保留的图件；MAP 反演、区间校准与降级图已删除。

## 2. 样式与保存

- 样式集中在 <code>sensor_array/plotting/style.py</code>：
  <code>_style()</code> 设置字体、字号、线宽；<code>_save(figure, name)</code> 保存 PNG + SVG。
- 字体为 <code>["Noto Serif CJK SC"]</code>：同时覆盖拉丁字母、H₂/NH₃/NO₂ 下标与中文。
  不要改成单独 SimSun 或纯 Latin 的 Noto Serif，否则中文会变成方框。
- 模型配色：<code>double</code> 蓝色 <code>#4C72B0</code>，<code>mlp_per_gas</code> 绿色
  <code>#009E73</code>；气体配色在 <code>GAS_COLORS</code>。
- 多面板图统一用 <code>layout="constrained"</code>；图例放在图底部
  （<code>figure.legend(loc="outside lower center")</code>），避免遮挡柱顶数值。

## 3. 逐图索引

| 图号 | 输出文件 | 绘图脚本 | 绘图函数 | 数据源 | 版式 |
|---|---|---|---|---|---|
| fig01 | <code>figures/model/fig01_design_and_data.png/.svg</code> | <code>plotting/model.py</code> | <code>_design(experiments)</code>，由 <code>main()</code> 调用 | <code>core.data.load_experiments()</code>，原始 <code>data/0.csv</code>–<code>data/61.csv</code> | 1×2 |
| fig03 | <code>figures/model/fig03_curve_error.png/.svg</code> | <code>plotting/model.py</code> | <code>_curve_error(curve)</code> | <code>artifacts/evaluation/curve_metrics.csv</code> | 1×2 |
| fig07 | <code>figures/model/fig07_model_architecture.png/.svg</code> | <code>plotting/model.py</code> | <code>_architecture()</code> | 无数据依赖，固定架构 | 1×1 流程图 |
| fig09 | <code>figures/model/fig09_concentration_design.png/.svg</code> | <code>plotting/model.py</code> | <code>_concentration_design(experiments)</code> | <code>core.data.load_experiments()</code> | 上 4 个 H₂ 切片 + 下 2 个平面汇总 |
| fig10 | <code>figures/model/fig10_nh3_initial_drift.png/.svg</code> | <code>plotting/model.py</code> | <code>_nh3_initial_drift(experiments)</code> | <code>core.data.load_experiments()</code> | 1×2，漂移与浓度同步 |

## 4. 逐图说明

### fig01 数据设计与示例曲线

- 左图：62 条实验的 (H₂, NH₃) 设计点，颜色为 NO₂ 浓度，双对数坐标；用于说明设计覆盖不均、H₂ 高端点少。
- 右图：file 0 的三路相对响应原始曲线；用于给读者一个“数据长什么样”的直观印象。
- 调整入口：<code>_design()</code>；改颜色可改 <code>cmap="viridis"</code>，改点大小改
  <code>s=28</code>，改示例文件把 <code>experiments[0]</code> 换成其他下标。

### fig03 留一曲线误差

- 左图：逐文件 nRMSE 的箱线图，三种气体 × 两个模型；纵轴对数，关闭离群点显示。
- 右图：每条实验的宏平均 nRMSE，横轴双指数骨架、纵轴三路独立 MLP + 双指数；虚线为 1:1，
  左上角标注改善折数。
- 调整入口：<code>_curve_error()</code>；箱线图位置由 <code>positions</code> 控制，
  离群点由 <code>showfliers</code> 控制。

### fig07 模型架构

- 1×1 流程图：左边浓度输入，上支双指数骨架，下支三路独立 MLP 残差，二者相加后
  由 \(R_j(0)\) 还原绝对输出。
- 骨架框给出双指数公式，残差框给出 7 维输入与 \(7\to64\to64\to1\) 宽度；图中不写
  参数数量、损失函数和训练细节，避免把架构图变成方法正文。
- 调整入口：<code>_architecture()</code>；框的位置/大小在 <code>box()</code> 调用里，
  箭头在 <code>arrow()</code> 调用里。

### fig09 62 条实验的浓度组合覆盖

- 上排：按 H₂ 的 4 个档位分片，每片是 NH₃×NO₂ 的 5×5 网格；格内 1 表示该组合有 1 条实验，0 表示缺失。
- 下排左：H₂–NO₂ 平面，格内数字为“跨 NH₃ 的有数据组合数”（0–5）。
- 下排右：H₂–NH₃ 平面，格内数字为“跨 NO₂ 的有数据组合数”（0–5）。
- 62 个已覆盖组合都只有 1 条实验，没有重复；因此“数据量丰富”指该平面内覆盖的组合数多，“缺失”指格内为 0。
- 用途：说明设计矩阵不是全因子、H₂=32000 几乎只覆盖 NO₂=2、NH₃=50 在 H₂=32000 缺失；同时保留 NH₃ 分块信息。
- 调整入口：`_concentration_design()`；二值颜色范围 `vmin=0, vmax=1`，下排汇总范围 `vmin=0, vmax=5`。

### fig10 NH₃ 初始响应漂移

- 左图：NH₃ 初始响应 \(R(0)\) 对实验序号，颜色为 NH₃ 设定浓度；标注 Pearson \(r=0.989\)、最大/最小 = 9.46×。
- 右图：同一批点对 NH₃ 设定浓度，颜色为实验序号；标注 Pearson \(r=0.965\)。
- 用途：说明 NH₃ 初值随采集顺序漂移，并与 NH₃ 浓度档位同步上升，二者在设计上无法分离。
- 调整入口：`_nh3_initial_drift()`；颜色映射在 `cmap="viridis"` / `cmap="plasma"`。

## 5. 新增或修改图件检查清单

1. 在 <code>plotting/model.py</code> 中新增函数，并在 <code>main()</code> 调用；函数名以 <code>_</code> 开头表示只在本模块使用。
2. 用 <code>_save(figure, "figXX_name")</code> 保存，确认 PNG 与 SVG 同名成对，不生成 PDF。
3. 在本文件第 3 节表格和第 4 节说明中登记图号、函数、数据源、版式。
4. 重新运行 <code>python -m sensor_array.plotting.model</code>，目视检查中文、下标、图例、数值标签是否被裁剪。
5. 若新增数据文件，在 <code>evaluation.md</code> 与 <code>README.md</code> 中同步数值和路径。
