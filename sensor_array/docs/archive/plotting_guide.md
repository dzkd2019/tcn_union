# sensor_array 绘图指南（逐图索引）

本文档只覆盖 sensor_array 项目。每新增一张图、修改绘图函数或调整输出目录时，必须同步更新本文件。

## 1. 运行入口

~~~bash
cd /home/mrawa/code/deep-learning/tcn_union
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.response
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.chapter
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model_comparison
~~~

- response.py：生成 fig01–fig07，写入 figures/archive/。
- chapter.py：生成 fig08–fig20、fig22–fig29，写入 figures/archive/；每张图按依赖的 CSV/NPZ 是否存在条件调用。
- model_comparison.py：生成 fig30、fig31，分别写入 figures/mlp_residual/ 与 figures/double_exponential/。
- 所有图件同时输出 PNG（300 dpi）与 SVG，不输出 PDF。
- response.py 与 chapter.py 出现 “Font family 'Noto Serif CJK SC' not found” 时回退到 SimSun，不影响输出。
- model_comparison.py 的图中文较多；当前 matplotlib 下 SimSun 在多面板图里会丢字，因此该模块显式使用 ["Noto Serif", "Source Han Sans CN"]（COMPARISON_FONTS），不要改回 SimSun。

## 2. 样式与保存

公共代码在 sensor_array/plotting/style.py：

- _style()：统一中文字体、字号、线宽、mathtext 和保存边界。
- model_comparison.py 的图用 layout="constrained"，不用 tight_layout；三面板图例统一放在图底（figure.legend，loc="outside lower center"），柱图不单独放轴内图例。
- _letter(ax, text)：面板左上角标题。
- _save(figure, name, directory=FIGURES_ARCHIVE)：保存 name.png 与 name.svg，并自动创建目录。
- GAS_LABEL、GAS_COLORS：气体名称与固定配色。
- MODEL_COLORS、MODEL_LABELS、MODEL_ORDER：主模型的颜色、图例名和顺序。

调整顺序建议：先改对应 figure_xxx 函数内部的 figsize、坐标轴、图例；需要全局字号或线宽时再改 _style()。不要在其他图函数里复制 _style()。

## 3. 逐图总表

| 图号 | 输出文件 | 绘图脚本 | 绘图函数 | 数据源 | 版式 |
|---|---|---|---|---|---|
| fig01 | figures/archive/fig01_data_overview.png/.svg | plotting/response.py | figure_data_overview(experiments) | data/0.csv–data/61.csv | 2×2 |
| fig02 | figures/archive/fig02_design_coverage.png/.svg | plotting/response.py | figure_design_coverage(experiments) | data/0.csv–data/61.csv | 2×2 |
| fig03 | figures/archive/fig03_shape_fit.png/.svg | plotting/response.py | figure_shape_fit(experiments, shapes) | data/0.csv–data/61.csv + fit_experiment_shapes | 1×2 |
| fig04 | figures/archive/fig04_logo_comparison.png/.svg | plotting/response.py | figure_logo_comparison(metrics) | artifacts/logo/logo_metrics.csv | 1×4 |
| fig05 | figures/archive/fig05_gain_tau.png/.svg | plotting/response.py | figure_gain_tau(experiments, shapes, model) | artifacts/models/model.json + data/0.csv–data/61.csv | 2×2 |
| fig06 | figures/archive/fig06_prediction_examples.png/.svg | plotting/response.py | figure_prediction_examples(experiments, metrics, predictions) | artifacts/logo/logo_metrics.csv + artifacts/logo/logo_predictions.npz | 3×3 |
| fig07 | figures/archive/fig07_absolute_response.png/.svg | plotting/response.py | figure_absolute(experiments, model) | artifacts/models/model.json + data/0.csv–data/61.csv | 1×3 |
| fig08 | figures/archive/fig08_hybrid_model.png/.svg | plotting/chapter.py | figure_hybrid(experiments, model) | artifacts/models/hybrid.json + artifacts/analysis/hybrid_sensitivity.csv + data/0.csv–data/61.csv | 2×3 |
| fig09 | figures/archive/fig09_crlb_window.png/.svg | plotting/chapter.py | figure_crlb(estimate) | artifacts/analysis/estimates.csv | 1×2 |
| fig10 | figures/archive/fig10_identifiability_map.png/.svg | plotting/chapter.py | figure_identifiability_map(experiments) | data/0.csv–data/61.csv | 1×3 |
| fig11 | figures/archive/fig11_error_map.png/.svg | plotting/chapter.py | figure_error_map(estimate, experiments) | artifacts/analysis/estimates.csv + data/0.csv–data/61.csv | 1×3 |
| fig12 | figures/archive/fig12_early_error.png/.svg | plotting/chapter.py | figure_early_error(estimate) | artifacts/analysis/estimates.csv | 1×2 |
| fig13 | figures/archive/fig13_coverage.png/.svg | plotting/chapter.py | figure_coverage(conformal) | artifacts/analysis/conformal.csv | 1×2 |
| fig14 | figures/archive/fig14_stopping.png/.svg | plotting/chapter.py | figure_stopping(stopping) | artifacts/analysis/stopping.csv | 1×2 |
| fig15 | figures/archive/fig15_degradation.png/.svg | plotting/chapter.py | figure_degradation() | artifacts/analysis/degradation.csv | 1×3 |
| fig16 | figures/archive/fig16_error_budget.png/.svg | plotting/chapter.py | figure_budget() | artifacts/analysis/error_budget.csv | 1×3 |
| fig17 | figures/archive/fig17_residual_map.png/.svg | plotting/chapter.py | figure_residual_map() | artifacts/analysis/residual_matrix.npz | 3×1 |
| fig18 | figures/archive/fig18_residual_pca.png/.svg | plotting/chapter.py | figure_pca() | artifacts/analysis/residual_pca_modes.csv | 2×2 |
| fig19 | figures/archive/fig19_residual_shape.png/.svg | plotting/chapter.py | figure_shape() | artifacts/analysis/residual_diagnostics.csv | 2×2 |
| fig20 | figures/archive/fig20_protocols.png/.svg | plotting/chapter.py | figure_protocols() | artifacts/validation/curve_summary.csv + artifacts/validation/estimate_summary.csv | 1×4 |
| fig21 | 预留，未生成 | — | — | P2 暂缓 | — |
| fig22 | figures/archive/fig22_structured_basis.png/.svg | plotting/chapter.py | figure_structured_basis(experiments) | artifacts/analysis/structured_basis.csv + data/0.csv–data/61.csv | 1×2 |
| fig23 | figures/archive/fig23_structured_residual.png/.svg | plotting/chapter.py | figure_structured_residual(experiments) | artifacts/validation/structured_logo_predictions.npz + data/0.csv–data/61.csv | 1×3 |
| fig24 | figures/archive/fig24_group_coverage.png/.svg | plotting/chapter.py | figure_group_coverage() | artifacts/analysis/coverage_by_group.csv | 1×2 |
| fig25 | figures/archive/fig25_bias_estimation.png/.svg | plotting/chapter.py | figure_bias_estimation() | artifacts/analysis/bias_summary.csv + artifacts/analysis/bias_estimates.csv | 1×3 |
| fig26 | figures/archive/fig26_stopping_decision.png/.svg | plotting/chapter.py | figure_stopping_decision() | artifacts/analysis/stopping_decision.csv + artifacts/analysis/stopping_decision_summary.csv | 1×2 |
| fig27 | figures/archive/fig27_fair_augmentation.png/.svg | plotting/chapter.py | figure_fair_augmentation() | artifacts/augmentation/fair_comparison.csv + artifacts/augmentation/fair_summary.csv | 1×3 |
| fig28 | figures/archive/fig28_forward_error_visual.png/.svg | plotting/chapter.py | figure_forward_error_visual(experiments) | data/0.csv–data/61.csv | 1×3 |
| fig29 | figures/archive/fig29_estimator_scatter.png/.svg | plotting/chapter.py | figure_estimator_scatter() | artifacts/analysis/estimates.csv | 1×3 |
| fig30 | figures/mlp_residual/fig30_mlp_residual_comparison.png/.svg | plotting/model_comparison.py | figure_mlp_residual() | artifacts/model_experiments/mlp_curve_metrics.csv + mlp_downstream.csv | 1×3，constrained |
| fig31 | figures/double_exponential/fig31_double_exponential_comparison.png/.svg | plotting/model_comparison.py | figure_double_exponential() | artifacts/model_experiments/double_curve_metrics.csv + double_downstream.csv | 1×3，constrained |

## 4. 逐图调整说明

### 4.1 数据与骨架（fig01–fig07）

- fig01 figure_data_overview：左上为基线漂移，其余三幅为三气体 62 条相对响应曲线。改 62 条曲线的透明度或抽样显示时，调整循环里的 alpha 和 linewidth。
- fig02 figure_design_coverage：左上为采集顺序与 NH3 档位，其余三幅为三组二维浓度设计平面。改点大小、档位分隔线和标注时，调整四个面板内的 scatter 与 axhline/axvline。
- fig03 figure_shape_fit：左为一阶模型对单条实验的拟合，右为逐实验拟合误差。改示例文件编号时，改函数内的 index。
- fig04 figure_logo_comparison：四个面板分别为宏平均、H2、NH3、NO2 的六模型误差。改模型顺序或颜色时，改 style.py 的 MODEL_ORDER/MODEL_COLORS，不要在函数内硬编码。
- fig05 figure_gain_tau：四面板为增益一致性、时间常数一致性、H2 受 NO2 抑制、NH3 交叉影响。改气泡大小、等高线或标注时，调整对应面板的 scatter/contour 参数。
- fig06 figure_prediction_examples：3×3，每行一个示例文件、每列一种气体，实测与预测对比。改示例文件时，改函数内的示例索引列表。
- fig07 figure_absolute：1×3，三气体的绝对响应曲线与模型。改纵轴是否为对数、是否裁剪 t 轴时，调整 set_yscale 与 set_xlim。

### 4.2 混合模型与可辨识性（fig08–fig14）

- fig08 figure_hybrid：上排为 K 的边际收益、B 样条基、单文件残差修正；下排为三气体实测/灰盒/混合曲线。示例文件由函数内 index = 21 控制。
- fig09 figure_crlb：左为噪声下限 CRLB，右为计入模型偏差后的实际不确定性。改窗长或气体时，调整 figure_crlb 内的筛选条件和柱/线样式。
- fig10 figure_identifiability_map：1×3，三气体在 5 s 窗的可辨识性平面。改色标范围或等值线时，调整 contourf 的 levels 与 colorbar。
- fig11 figure_error_map：1×3，三气体的估计误差随设计变量的分布。改误差定义或分箱时，调整 figure_error_map 内的计算和 scatter/hexbin。
- fig12 figure_early_error：1×2，早期误差与误差随时间收敛。改窗长集合时，调整函数内的 tmax 列表。
- fig13 figure_coverage：左为覆盖率、右为区间宽度。改名义覆盖率线时，调整 axhline 与图例文本。
- fig14 figure_stopping：左为能否给出足够窄的区间，右为达到目标所需时间。改目标宽度或停止判据时，调整函数内的阈值。

### 4.3 降级、误差预算与残差（fig15–fig20）

- fig15 figure_degradation：1×3，三气体在完整/缺通道条件下的误差。改通道组合或窗长时，调整图例与 x 轴刻度。
- fig16 figure_budget：1×3，三气体的误差预算（实测、混合生成、灰盒生成）。改曲线颜色或线型时，调整 plot 参数。
- fig17 figure_residual_map：3×1，三气体的折外残差热图。改色标范围时，调整 imshow 的 vmin/vmax；改时间/档位轴时，调整 extent。
- fig18 figure_pca：2×2，低秩方差占比、第一模式、第二模式、NO2 第二模式与幅度/时间常数。改模式编号时，调整函数内的 PC 索引。
- fig19 figure_residual_shape：2×2，增益/形状分解、误差集中度、设计稀疏度与误差、NO2 形状随幅度变化。改中位数/均值口径时，调整聚合函数。
- fig20 figure_protocols：1×4，曲线误差加三气体下游误差，比较留一实验/留一档位/留一采集块。改柱宽、误差棒或图例列数时，调整 bar 与 legend 参数。

### 4.4 结构化残差与 P4（fig22–fig29）

- fig22 figure_structured_basis：左为自由 B 样条基，右为按物理时段切分的结构化基。改基数量或时段边界时，同时改 core/hybrid.py 的 structured_basis 与本图调用。
- fig23 figure_structured_residual：1×3，三气体的逐文件 nRMSE 与文件序号，比较自由基与结构化基。改相关标注或纵轴范围时，调整 text 与 set_ylim。
- fig24 figure_group_coverage：左为区间宽度、右为覆盖率，比较全局、NH3 分档、采集块。改分组定义时，先改 analysis/estimation.py 的 group_conformal，再改本图标签。
- fig25 figure_bias_estimation：1×3，点误差、Laplace 区间覆盖、逐文件散点。改散点透明度或对角线范围时，调整 scatter 与 plot。
- fig26 figure_stopping_decision：左为停止时间曲线，右为 10 s 内决策结果堆叠柱。改堆叠顺序或颜色时，调整 bar 的 bottom 累加顺序。
- fig27 figure_fair_augmentation：1×3，宏平均 MAPE、10 s 分气体 ΔMAPE、5/10 s 配对差与 95% CI。改三臂标签或颜色时，调整函数内 labels/colors 字典。
- fig28 figure_forward_error_visual：1×3，三气体的实测曲线与混合生成曲线对比，示例为 file 0。改示例文件时，改函数内的 file_id。
- fig29 figure_estimator_scatter：1×3，三气体 10 s 估计对真值散点。改坐标范围或误差条时，调整 set_xlim/set_ylim 与 errorbar。

### 4.5 新方案对比（fig30–fig31）

结果与统计见 [`model_experiments.md`](model_experiments.md)；本节只说明绘图入口。图件不进入 P6 快照。

- fig30 figure_mlp_residual：
  - (a) 逐文件宏平均 nRMSE 散点：横轴低秩残差，纵轴 MLP 残差，双对数，虚线为 1:1；低于虚线表示 MLP 更好。左上角标出改善折数和中位差。
  - (b) 逐气体平均 nRMSE 分组柱：低秩残差 vs MLP 残差，柱顶标数值。
  - (c) 10 s 浓度误差：两模型在 H2/NH3/NO2 上的中位绝对相对误差。
  - 改散点范围、柱宽、数值标签格式时，分别调整 _scatter、_bars、_downstream_bars。
  - 三面板共用图底图例（两列）；柱图 y 轴上限为最大柱高的 1.18 倍，避免柱顶数值贴边。
  - 输出：figures/mlp_residual/fig30_mlp_residual_comparison.png 与 .svg。
- fig31 figure_double_exponential：
  - (a) 逐文件宏平均 nRMSE 散点：横轴当前低秩残差，纵轴双指数 + 低秩，双对数，虚线为 1:1。
  - (b) 逐气体平均 nRMSE 分组柱：一阶灰盒、双指数灰盒、低秩残差、双指数 + 低秩四组。
  - (c) 10 s 浓度误差：低秩残差 vs 双指数 + 低秩，直接比较同一残差修正下的骨架差异。
  - 改四组配色或图例列数时，调整 plotting/model_comparison.py 顶部的 STYLE 字典；图底图例为四列。
  - 输出：figures/double_exponential/fig31_double_exponential_comparison.png 与 .svg。

## 5. 目录与归档约定

- figures/archive/：P6 之前的主线图件，fig01–fig20、fig22–fig29；fig21 为暂缓的 P2 预留。
- figures/mlp_residual/：只放 MLP 残差实验图（当前 fig30）。
- figures/double_exponential/：只放双指数实验图（当前 fig31）。
- 不同方案的图不得混放在同一目录；新增方案时新建子目录，并同步更新本文件第 3 节和第 5 节。
- MANIFEST.md 与 SHA256SUMS 只覆盖 artifacts 与 figures/archive 的 P6 快照；新实验图件单独记录，是否纳入快照由当次任务决定。

## 6. 新增或修改图件的检查清单

1. 在对应 plotting 模块新增 figure_xxx，复用 style.py 的 _style/_letter/_save。
2. 明确数据源；缺文件时给出清晰报错或在 main() 中条件调用。
3. 同时输出 PNG 300 dpi 与 SVG，不输出 PDF。
4. 在 main() 中注册调用；若是新方案，写入独立子目录。
5. 更新本文件第 3 节表格与第 4 节面板说明；新增目录时更新第 5 节。
6. 运行对应模块，打开 PNG 目视检查中文、图例、数值标签是否遮挡；多面板图优先用 layout="constrained"。
7. 若图件进入快照，重跑 MANIFEST/SHA256SUMS 并核对数量和校验和。
