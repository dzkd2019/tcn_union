# Engineering Principles

## 0. Data Constraint

- 现有数据集（`data/0.csv`–`data/61.csv`，62 次实验）**不可扩充**：传感器已损坏。
- 任何方案、实验、论文与后续工作都**不得建议或依赖补测、补采、重新采集数据**。
- 只能在现有 62 条实验上做建模、验证、外推与仿真分析。

## 1. Simplicity First

- Always prefer the simplest correct solution that satisfies the current requirements.
- Optimize for readability, maintainability, and low cognitive complexity, not architectural sophistication.
- Prefer explicit and straightforward code over clever abstractions.

## 2. KISS and YAGNI

- Do not implement features that were not explicitly requested.
- Do not introduce speculative abstractions for hypothetical future requirements.
- Do not create interfaces, factories, strategies, providers, registries, or plugin systems when a direct implementation is sufficient.
- Do not add configuration options unless there is a real requirement for configurability.
- Do not introduce additional dependencies when existing tools or standard libraries are sufficient.

## 3. Architecture Discipline

- Reuse existing project architecture and conventions whenever possible.
- Do not introduce new architectural layers without a concrete justification.
- Avoid unnecessary wrappers, adapters, managers, services, and pass-through functions.
- Keep related logic together when separation would increase complexity.
- Introduce abstractions only when they provide demonstrable value, such as eliminating meaningful duplication or isolating a genuinely independent responsibility.

## 4. Implementation Discipline

- Inspect existing code before implementing new functionality.
- Prefer modifying existing code over creating new files and modules.
- Keep changes narrowly scoped to the requested task.
- Do not refactor unrelated code.
- Do not add defensive checks for impossible scenarios.
- Do not overengineer error handling.

## 5. Before Adding Complexity

Before introducing a new abstraction, dependency, module, or architectural pattern, ask:

1. Does the current requirement actually need this?
2. Can the requirement be solved using existing code?
3. Would a straightforward implementation be easier to understand?
4. Does the additional complexity provide a concrete benefit today?

If the simpler solution is sufficient, use it.

## 6. Completion Review

Before completing a task:

- Review the entire diff.
- Identify unnecessary abstractions, duplicated logic, speculative features, and redundant code.
- Simplify the implementation where appropriate.
- Ensure existing behavior is preserved unless a behavior change was requested.
- Run relevant tests and report their results honestly.

## 7. Design Justification

When introducing a new architectural layer or significant abstraction, briefly explain:

- What problem it solves.
- Why existing code cannot solve it adequately.
- Why a simpler alternative is insufficient.

Do not introduce complexity merely to demonstrate good software architecture.

## 8. 图表输出规范

- 绘图时同时输出 **PNG** 和 **SVG** 两种格式；
- 不再输出 PDF 图片；
- PNG 用于 PPT、网页和快速预览，SVG 用于论文/报告的矢量插图；
- 文件名保持一致，仅扩展名不同，例如 `fig_name.png` 与 `fig_name.svg`。
- 如果旧脚本仍输出 PDF，修改该脚本时一并改为 PNG + SVG。

## 9. 绘图文档维护（sensor_array）

- `sensor_array/docs/plotting_guide.md` 是 sensor_array 项目的逐图索引，必须与绘图代码同步维护。
- 每张图必须登记：图号、输出文件名、绘图脚本、绘图函数、数据源、版式与保存目录。
- 新增图件，或修改绘图脚本、绘图函数、数据源、版式、输出路径时，必须在同一次改动中更新该文档。
- 不同方案的图件按方案分目录存放，不把不同方案的图混在同一目录；具体目录约定见该文档。
- 图件格式仍遵守第 8 节：PNG 300 dpi + 同名 SVG，不输出 PDF。

## 10. 逆问题最终评估规范（TCN 浓度反演）

本节适用于**每一次 TCN 浓度反演评估**：OOF、验证集、锁定测试、集成预测，以及后续任何新协议。所有指标以**实验**为单位，不把同一实验内的多个时间点当独立样本；OOF 与锁定测试必须分开统计，不混合、不用锁定集调参。

### 10.1 必报指标

每次评估必须保存逐实验预测与汇总表（建议 `metrics.csv`），逐气体报告：

- MAPE（%）：\(\mathrm{MAPE}_j=\frac{100}{n}\sum_i |\hat c_{ij}-c_{ij}|/|c_{ij}|\)；
- NRMSE：\(\mathrm{NRMSE}_j=\sqrt{\frac1n\sum_i(\hat c_{ij}-c_{ij})^2}/R_j\)，其中 \(R_j\) 为该气体的浓度量程；必须注明量程来自训练折、开发集还是测试集，所有对比臂使用同一量程；
- 3σ（%）：\(3\times\mathrm{std}(r_{ij},\mathrm{ddof}=1)\)，其中 \(r_{ij}=100(\hat c_{ij}-c_{ij})/c_{ij}\)；3σ 是实验间离散度，**不是**置信区间；
- 95% 置信区间（%）：\(\bar r_j\pm t_{0.975,n-1}\,s_j/\sqrt n\)，其中 \(s_j\) 为 \(r_{ij}\) 的样本标准差；这是**均值**的置信区间，必须同时报告 \(n\)；
- 三气体宏平均：MAPE、NRMSE、3σ 分别取三气体算术平均；宏 3σ 是“三个 3σ 的平均”，不是对宏平均相对误差再乘 3。

MAPE 在任一 \(c_{ij}=0\) 时无定义：写空值并说明原因，不得用极小值替代或静默丢弃。所有指标保留原始精度（不要只给四舍五入后的百分数），并在结果文档中写明预测来源、窗口长度、seed、折数与预测集。

### 10.2 必报图件

逐气体各输出两组图，统一放在本次评估运行的 `figs/` 子目录：

1. **有符号误差散点图**：每气体一张 1×2 面板；(a) 有符号绝对误差 \(\hat c-c\)，(b) 有符号相对误差 \(100(\hat c-c)/c\)；横轴为实验序号，零线加粗，另画 \(\pm1\sigma,\pm2\sigma,\pm3\sigma\) 水平线并在图内标注数值；不连线、不做平滑；OOF 与锁定测试若同图，必须用底色或分面明确区分。
2. **95% 置信区间图**：每气体一张；只画均值与 95% CI 误差棒，**不显示原始散点**；纵轴为相对误差均值（%），图内标注 \(n\)；多种子时每个种子一个点，不把种子当实验重复。

图件格式遵守第 8 节：PNG 300 dpi + 同名 SVG，不输出 PDF。颜色、字体、下标与第 8/9 节一致；不得写 `ppm`（数据未注明单位）。

### 10.3 目录与登记

- 每次评估运行一个独立目录，建议 `artifacts/<run_name>/`，其中 `<run_name>` 至少包含模型、窗口、协议与 seed（例如 `tcn_10s_full_seed20260923`）；该次运行的预测、`metrics.csv`、脚本与 `figs/` 全部放在这个目录内。
- 该次运行的所有图件必须放在同一个 `figs/` 子目录，不与其他运行、其他模型或其他窗口的图混放。
- TCN 逆问题的图件不放进 `sensor_array/figures/`（那里只放正问题图）；逐图登记到 `docs/plotting_guide_tcn_10s.md` 或对应窗口的 plotting guide，登记内容包括 run_name、图文件名、绘图脚本/函数、数据源、版式与 `figs/` 路径。
- 新增或修改图件、指标定义、量程来源时，必须在同一次改动中同步 plotting guide 与结果文档；不允许“图已生成、索引未更新”。

### 10.4 执行时机

- 任何对外汇报、论文表格或锁定测试结论，必须先满足 10.1–10.3；缺一项即视为评估未完成。
- 锁定 12 条只在最终候选上评估一次；评估后不得据其调整模型、seed、阈值或图件选择。
- 若指标因数据或定义不可计算，必须在结果中显式写“不可计算/原因”，不能用 0 或空图代替。
