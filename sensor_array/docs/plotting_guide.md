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
<code>fig05</code> 只有在 <code>artifacts/calibration/summary.csv</code> 存在时才绘制；先运行
<code>.venv/bin/python -m sensor_array.analysis.calibration</code>。

## 2. 样式与保存

- 样式集中在 <code>sensor_array/plotting/style.py</code>：
  <code>_style()</code> 设置字体、字号、线宽；<code>_save(figure, name)</code> 保存 PNG + SVG。
- 字体为 <code>["Noto Serif CJK SC"]</code>：同时覆盖拉丁字母、H₂/NH₃/NO₂ 下标与中文。
  不要改成单独 SimSun 或纯 Latin 的 Noto Serif，否则中文会变成方框。
- 模型配色：<code>double</code> 蓝色 <code>#4C72B0</code>，<code>mlp_per_gas</code> 绿色
  <code>#009E73</code>；气体配色在 <code>GAS_COLORS</code>。
- 多面板图统一用 <code>layout="constrained"</code>；图例放在图底部
  （<code>figure.legend(loc="outside lower center")</code>），避免遮挡柱顶数值。
- 柱图纵轴上限约为最大柱高的 1.18 倍，数值标签放在柱顶上方 0.2 个单位。

## 3. 逐图索引

| 图号 | 输出文件 | 绘图脚本 | 绘图函数 | 数据源 | 版式 |
|---|---|---|---|---|---|
| fig01 | <code>figures/model/fig01_design_and_data.png/.svg</code> | <code>plotting/model.py</code> | <code>_design(experiments)</code>，由 <code>main()</code> 调用 | <code>core.data.load_experiments()</code>，原始 <code>data/0.csv</code>–<code>data/61.csv</code> | 1×2 |
| fig02 | <code>figures/model/fig02_curve_examples.png/.svg</code> | <code>plotting/model.py</code> | <code>_curve_examples(experiments, curve, predictions)</code> | <code>artifacts/evaluation/curve_metrics.csv</code>、<code>predictions.npz</code>、原始 CSV | 2×3 |
| fig03 | <code>figures/model/fig03_curve_error.png/.svg</code> | <code>plotting/model.py</code> | <code>_curve_error(curve)</code> | <code>artifacts/evaluation/curve_metrics.csv</code> | 1×2 |
| fig04 | <code>figures/model/fig04_downstream.png/.svg</code> | <code>plotting/model.py</code> | <code>_downstream(downstream)</code> | <code>artifacts/evaluation/downstream_summary.csv</code> | 1×3 |
| fig05 | <code>figures/model/fig05_interval_calibration.png/.svg</code> | <code>plotting/model.py</code> | <code>_interval_calibration(summary)</code> | <code>artifacts/calibration/summary.csv</code> | 1×3 |
| fig06 | <code>figures/model/fig06_representative_curves.png/.svg</code> | <code>plotting/model.py</code> | <code>_representative_curves(experiments, curve, predictions)</code> | <code>artifacts/evaluation/curve_metrics.csv</code>、<code>predictions.npz</code>、原始 CSV | 5×3（左侧案例标签） |
| fig07 | <code>figures/model/fig07_model_architecture.png/.svg</code> | <code>plotting/model.py</code> | <code>_architecture()</code> | 无数据依赖，固定架构 | 1×1 流程图 |
| fig08 | <code>figures/model/fig08_downstream_error_scatter.png/.svg</code> | <code>plotting/model.py</code> | <code>_downstream_error_scatter(downstream)</code> | <code>artifacts/evaluation/downstream.csv</code> | 2×3（上：相对误差；下：绝对误差） |
| fig09 | <code>figures/model/fig09_concentration_design.png/.svg</code> | <code>plotting/model.py</code> | <code>_concentration_design(experiments)</code> | <code>core.data.load_experiments()</code> | 上 4 个 H₂ 切片 + 下 2 个平面汇总 |
| fig10 | <code>figures/model/fig10_nh3_initial_drift.png/.svg</code> | <code>plotting/model.py</code> | <code>_nh3_initial_drift(experiments)</code> | <code>core.data.load_experiments()</code> | 1×2，漂移与浓度同步 |
| fig11 | <code>figures/model/fig11_sensor_degradation.png/.svg</code> | <code>plotting/model.py</code> | <code>_degradation()</code> | <code>artifacts/evaluation/degradation_summary.csv</code> | 3×4 热图（行=气体，列=传感器组合） |

## 4. 逐图说明

### fig01 数据设计与示例曲线

- 左图：62 条实验的 (H₂, NH₃) 设计点，颜色为 NO₂ 浓度，双对数坐标；用于说明设计覆盖不均、H₂ 高端点少。
- 右图：file 0 的三路相对响应原始曲线；用于给读者一个“数据长什么样”的直观印象。
- 调整入口：<code>_design()</code>；改颜色可改 <code>cmap="viridis"</code>，改点大小改
  <code>s=28</code>，改示例文件把 <code>experiments[0]</code> 换成其他下标。

### fig02 曲线示例

- 选两条实验：双指数骨架宏 nRMSE 最差的一条与中位的一条（按 <code>curve_metrics.csv</code> 的
  <code>nrmse_macro</code> 选取）；当前主模型的分层代表案例见 fig06。
- 两行三列：行为实验、列为 H₂/NH₃/NO₂；黑线实测，蓝线双指数骨架，绿线三路独立 MLP + 双指数。
- 调整入口：<code>_curve_examples()</code>；改选折逻辑在函数开头 <code>median_file</code>/<code>worst_file</code>，
  改线宽在 <code>lw</code>。

### fig03 留一曲线误差

- 左图：逐文件 nRMSE 的箱线图，三种气体 × 两个模型；纵轴对数，关闭离群点显示。
- 右图：每条实验的宏平均 nRMSE，横轴双指数骨架、纵轴三路独立 MLP + 双指数；虚线为 1:1，
  左上角标注改善折数（当前应为 59/62）。
- 调整入口：<code>_curve_error()</code>；箱线图位置由 <code>positions</code> 控制，
  离群点由 <code>showfliers</code> 控制。

### fig04 下游浓度误差

- 三个面板：10 s 中位绝对相对误差、5 s 中位绝对相对误差、10 s p90 绝对相对误差。
- 每个面板按气体分组，蓝色为双指数骨架，绿色为三路独立 MLP + 双指数；柱顶标数值。
- 调整入口：<code>_downstream()</code>；窗口与指标在 <code>panels</code> 元组中，
  纵轴上限由 <code>max(maxima) * 1.18</code> 控制。

### fig05 P4 区间校准

- 三个面板：开发集覆盖率、开发集中位区间宽度、锁定集覆盖率；均为 10 s。
- 四种方法：原始 Laplace 90%（灰）、全局共形（蓝）、NH₃ 档位共形（橙）、采集块共形（绿）；
  覆盖率面板画 90% 名义线。开发集上全局/NH₃ 共形为 90%–92%，采集块为 84%，
  图标题按这一事实标注。
- 区间在 log 浓度空间构造，宽度用相对宽度 <code>exp(q*se_log)-1</code>；校准只用开发 50 条，
  锁定 12 条只作描述性复核，不参与分位数拟合。
- 调整入口：<code>_interval_calibration()</code>；方法列表在 <code>methods</code>，
  图例标签在 <code>labels</code>，标题在 <code>figure.suptitle</code>。

### fig06 代表性生成曲线

- 五个案例：当前模型宏 nRMSE 的最优（file 31，0.38%）、中位（file 27，1.16%）、
  最差（file 0，7.03%），以及两个设计边界——NH₃ 最低档且 H₂/NO₂ 居中的 file 4
  （1.35%）、NH₃ 与 NO₂ 同时最高的 file 61（2.66%）。
- 版式 5×3：每行一条实验，列为 H₂/NH₃/NO₂；左侧标签给出文件号、案例类型、
  宏 nRMSE 与三路浓度；黑线实测，蓝线双指数骨架，绿线三路独立 MLP + 双指数。
- file 31 的 H₂ 通道展示骨架的过冲/平台误差被 MLP 修正；file 0 的 H₂ 展示
  H₂=32000 边角上实测仍在缓慢上升而两个模型都提前压平；file 61 的 NH₃ 在约 35 s
  有一次采集跳变，模型无法跟随，这是采集伪影而不是模型输出错误。
- 这五条只作展示，不参与选型或调参；选择规则写死在函数开头，数据固定为 62 条。
- 调整入口：<code>_representative_curves()</code>；案例列表在 <code>cases</code>，
  左侧标签在 <code>header.text()</code>，曲线线宽在 <code>lw</code>。

### fig07 模型架构

- 1×1 流程图：左边浓度输入，上支双指数骨架，下支三路独立 MLP 残差，二者相加后
  由 \(R_j(0)\) 还原绝对输出。
- 骨架框给出双指数公式，残差框给出 7 维输入与 \(7\to64\to64\to1\) 宽度；图中不写
  参数数量、损失函数和训练细节，避免把架构图变成方法正文。
- 调整入口：<code>_architecture()</code>；框的位置/大小在 <code>box()</code> 调用里，
  箭头在 <code>arrow()</code> 调用里。


### fig08 下游浓度反演误差散点

- 10 s、当前 `mlp_per_gas` 模型，62 条实验全部画出，不区分 OOF 与锁定测试；无图题。
- 上排：有符号相对误差 `%`，横轴实验序号；灰线为 ±1σ/±2σ/±3σ，左上角标注均值、σ、3σ，不写样本量。
- 下排：有符号绝对误差（原始单位，`theta_hat - theta_true`），零线与 ±1σ/±2σ/±3σ 灰线；左上角同样标注均值、σ、3σ，不写样本量。
- 调整入口：`_downstream_error_scatter()`；换窗口改 `tmax`，换模型改 `model` 过滤条件。

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

### fig11 少一路传感器的浓度反演误差

- 数据源：<code>artifacts/evaluation/degradation_summary.csv</code>；由 <code>python -m sensor_array.analysis.degradation</code> 生成。
- 版式：3×4 热图；行为 H₂/NH₃/NO₂，列为“完整 / 缺 H₂ / 缺 NH₃ / 缺 NO₂”；格内为 10 s 中位绝对相对误差与相对完整三路的倍数。
- **协议**：模型仍用三路响应训练；只在 MAP 反演时删掉一路残差。不是“去掉一路重新训练”，图注和正文必须写清楚。
- 数值：完整 1.5/1.2/3.8%；缺 H₂ 15.6/2.7/4.8%；缺 NH₃ 2.6/5.9/4.6%；缺 NO₂ 5.8/2.6/15.1%（H₂/NH₃/NO₂）。
- 调整入口：<code>_degradation()</code>；颜色范围按最大中位误差自动取 <code>vmax</code>，文字阈值 <code>0.55*vmax</code>。

## 5. 新增或修改图件检查清单

1. 在 <code>plotting/model.py</code> 中新增函数，并在 <code>main()</code> 调用；函数名以 <code>_</code> 开头表示只在本模块使用。
2. 用 <code>_save(figure, "figXX_name")</code> 保存，确认 PNG 与 SVG 同名成对，不生成 PDF。
3. 在本文件第 3 节表格和第 4 节说明中登记图号、函数、数据源、版式。
4. 重新运行 <code>python -m sensor_array.plotting.model</code>，目视检查中文、下标、图例、数值标签是否被裁剪。
5. 若新增数据文件，在 <code>evaluation.md</code> 与 <code>README.md</code> 中同步数值和路径。