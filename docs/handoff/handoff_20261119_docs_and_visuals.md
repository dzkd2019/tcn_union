# Handoff — 2026-11-19 — 学习文档、两阶段总结与直观图

> **接手状态（2026-11-19）**：下方“未提交/未跟踪”清单已由提交 c571640 全部纳入 git；工作区实际为 /home/mrawa/code/deep-learning/tcn_union，HEAD 为 c571640。fig28–fig29 已纳入讲解图，MANIFEST / SHA256SUMS 同步为 28 组、56 个图件（124 行）。
>
> **模型分支变更（2026-11-19 晚）**：当前分支 `experiment/mlp-double-exponential` 已改为双指数骨架 + 三路独立 MLP 残差（默认推荐三路独立组合，纯骨架作对照）；旧模型代码、artifacts、figures 与 MANIFEST/SHA 记录均已清理，本文件描述的图表与路径只作历史参考。当前模型见 [`sensor_array/docs/model.md`](../../sensor_array/docs/model.md) 与 [`evaluation.md`](../../sensor_array/docs/evaluation.md)。
>
> **最新交接**：[`handoff_20261119_mlp_per_gas.md`](handoff_20261119_mlp_per_gas.md)。本文件只保留历史图表会话记录。

## 0. 会话状态

- 工作区：`/home/mrawa/code/graduate/tcn_union`
- 分支：`feature/sensor-array-thesis`；HEAD：`75a112a`（`feat(sensor-array)：完成 P1/P3/P4/P5/P6 并同步文档与图表`）
- 没有正在运行的后台任务。
- 用户最后一条实质请求是 `/handoff`（m03270）；再往前依次是：
  - 重写 `learning_guide`（m02764）；
  - 扩展 `beginner_guide` 的 §3.7/§3.8（m03015）、§5/§6（m03058）、§4.4（m03136）；
  - 解释“曲线 nRMSE 与估计器浓度误差”的直观对比（m03172）。
- 本会话**只改文档、加两张解释图**，没有改模型、没有重跑实验。

### 0.1 未提交改动（下一步要先处理）

```text
 M sensor_array/docs/beginner_guide.md
 M sensor_array/docs/learning_guide.md
 M sensor_array/plotting/chapter.py
?? docs/硕士课题总结_响应加速与阵列建模.md
?? sensor_array/figures/archive/fig28_forward_error_visual.png
?? sensor_array/figures/archive/fig28_forward_error_visual.svg
?? sensor_array/figures/archive/fig29_estimator_scatter.png
?? sensor_array/figures/archive/fig29_estimator_scatter.svg
```

本 handoff 文件本身也会是 untracked。

### 0.2 验证状态

- `.venv/bin/python -m unittest discover -s tests`：**67 项 OK**（属于 `tcn_union`，不是 `sensor_array` 的测试）。
- `git diff --check -- '*.md' '*.py'`：通过。
- `sensor_array/plotting/chapter.py`：`py_compile` 通过，fig28/fig29 已实际渲染。
- `sensor_array` 仍没有独立自动化测试；文档改动没有测试可跑。

## 1. 先读这些，避免重复

- 主交接：`/home/mrawa/code/graduate/tcn_union/handoff_20261117.md`
  - 数据约束、A+B、P1/P3/P4/P5/P6 结果、目录/命令、已知限制。
  - 注意：它的 §0 还写 HEAD `0e2a9b1` 和“改动未提交”，git 快照已过期；以本文件的 `75a112a` 为准。
- 结论与证据：`sensor_array/docs/chapter_results.md`
- 路线图与 P2 暂缓理由：`sensor_array/docs/roadmap.md`
- 论文叙事版学习文档：`sensor_array/docs/learning_guide.md`
- 概念双解手册：`sensor_array/docs/beginner_guide.md`
- 两阶段合并总结：`docs/硕士课题总结_响应加速与阵列建模.md`
- 项目入口：`sensor_array/README.md`

上一份增量 handoff `docs/handoff/handoff_20261118_beginner_guide.md` 的 §0 也过期了：它写 HEAD `0e2a9b1`、P1–P6 未提交、beginner_guide 是 untracked；实际这些已经在 `75a112a` 中。它的工作区路径还写成 `/home/mrawa/code/deep-learning/tcn_union`，实际是 `/home/mrawa/code/graduate/tcn_union`。

不要在本文件里重复 `chapter_results.md` 的六模型表、P1–P6 数字或 `roadmap.md` 的里程碑；需要时直接引用。

## 2. 本次增量

### 2.1 `learning_guide.md` 重写（1471 行，未提交）

按硕士论文一章的结构重写，不写摘要：

1. 绪论（电子鼻故事、正/逆问题、数据约束、核心问题、贡献）；
2. 数据与协议（设计混淆、相对响应、LOGO/留档位/留块、指标）；
3. 从最简模型开始（均值曲线、两阶段、直接岭回归、TCN）；
4. 一阶灰盒（ODE、45 参数、联合拟合）；
5. 混合模型（低秩残差、K=5、结构化对照）；
6. 可辨识性（噪声、FIM/CRLB、盲基线）；
7. 估计（MAP/软墙/Laplace/CV+/停止/误差预算/降级/剖面）；
8. 正模型反哺逆模型（P5 公平 B、旧 B/循环一致性、指标与标准化教训）；
9. 讨论；10. 结论与展望；
附录 A 公式、B 代码/图/命令、C 术语、D 自测 15 题、E 5 分钟汇报提纲。

§4.5/§4.6 专门解释了“逐条拟合误差（占全幅值）”和“62 折 LOGO nRMSE”为什么不是同一个量：前者是每条曲线自己的 3 参数 in-sample 形状拟合，后者是 61 条共享 45 参数的留出实验误差；分母也分别是全幅值/训练折 RMS。

### 2.2 `beginner_guide.md` 扩写（1339 行，未提交）

- §3.7/§3.8：标准差/标准误/置信区间、覆盖率/宽度；约 7704 字符。
- §5/§6：可辨识性、CRLB、噪声/MAD、MAP/软墙、Laplace、协方差、CV+、分组共形、剖面、决策/期望损失；约 11614 字符。
- §4.4：残差/低秩/B 样条，改成 4.4.1–4.4.6 六个小节，约 7222 字符：
  - 残差的定义与残差矩阵；
  - 低秩分解 `R_j ≈ Φ B_j`、`B_j = β_j X_all^T`；
  - B 样条 Cox–de Boor、局部支撑、钳位、为什么丢掉第一条基；
  - 时间基 × 浓度特征的乘积、岭回归闭式解、两步拟合、K=5、结果与边界。
- 行内公式统一用 `$...$`；不要再把数学符号放进反引号。

### 2.3 两张直观图（新增，未提交）

文件：

- `sensor_array/figures/archive/fig28_forward_error_visual.{png,svg}`
- `sensor_array/figures/archive/fig29_estimator_scatter.{png,svg}`

实现：`sensor_array/plotting/chapter.py` 新增

- `figure_forward_error_visual(experiments)`：file 0（正模型最差折外曲线）的实测 vs 混合生成，画原始传感器单位，并标注 LOGO nRMSE、RMSE、占全幅值比例；
- `figure_estimator_scatter()`：10 s 窗、62 折 LOGO 的真实曲线浓度估计对真值散点，三气体各一格，标注中位/p90/最差 file。

两点用途：

1. 说明“nRMSE 5%”在原始输出上可能只是很小的一段，但换成相对信号可能很大；
2. 说明“曲线最差”不等于“浓度估计最差”：file 0 曲线最差但 10 s 估计 17.8/9.5/17.1%，file 3 曲线只有 6.0% 但估计 23.7/48.5/46.4%。

`chapter.py` 的 `main()` 已把这两个函数接进去，但 **fig28/fig29 不在原计划 fig01–fig27 的编号体系里**；如果论文图号已冻结，考虑把它们作为附录/讲解图，不要硬插进章节图序。

### 2.4 两阶段合并总结（386 行，untracked）

`docs/硕士课题总结_响应加速与阵列建模.md`

- 摘要 + 关键词；
- 1 引言（为什么 60 s→10 s、两个问题、贡献）；
- 2 数据/指标/协议；
- 3 第一阶段 TCN 响应加速（3σ 方案、加权、VAE、结果、遗留问题）；
- 4 第二阶段阵列建模（灰盒、混合、CRLB、10 s 估计、误差预算、P5）；
- 5 两阶段递进（“能不能→为什么/多可信→闭环”）；
- 6 讨论/局限；7 结论；资料索引。
- 嵌入 3 张现有图；链接、公式定界符、无“补测/补采/ppm”检查通过。

## 3. 硬约束（继续保留）

- 只能使用 `data/0.csv`–`data/61.csv` 共 62 条实验；不得建议补测、补采、重新采集。
- 12 条锁定测试 `TEST_IDS={2, 7, 15, 20, 23, 27, 31, 38, 41, 46, 55, 60}` 只做描述性统计，不用于选模型、调参或挑方法。
- 浓度单位未注明，统一写“原始单位”，不要写 ppm。
- 运行 Python 用 `.venv/bin/python`。
- 不 commit、不 push，除非用户明确要求。
- 不改 `pyproject.toml`、`uv.lock` 和 `tcn_union/` 下的其他文件，除非任务确实需要。
- 图同时输出 PNG + SVG，不出 PDF。
- P2 仍暂缓；重启条件见 `roadmap.md`，不要仅因“想提升指标”重启。

## 4. 建议下一步

1. **先决定是否提交本次文档/图**。如果要提交，建议一个提交只做文档与图：
   `docs: rewrite learning guide, expand beginner guide, add intuition figures`
   提交前跑 `git diff --check -- '*.md' '*.py'` 和 `python -m unittest discover -s tests`。
2. **通读 `learning_guide.md` §4.5/§4.6**：这是用户最近问的“1–2% vs 3.47%”的核心；确认与 `chapter_results.md` 的六模型表一致。
3. **通读 `beginner_guide.md` §4.4**：确认残差矩阵、`B_j`、B 样条丢首条基的公式与 `core/hybrid.py` 的 `bspline_basis` 实现一致。
4. **如果进入论文写作**：用 `docs/硕士课题总结_响应加速与阵列建模.md` 作骨架，用 `chapter_results.md` 取证据；不要把 `beginner_guide` 的“通俗说法”直接抄进论文正文。
5. **如果继续研究**：P2 仍暂缓；P4/P5/P6 已完成，除非用户明确要求，不要重跑六模型或 P5 训练。

## 5. Suggested skills

下一个 agent 视任务调用：

- `ml-paper-writing`：把两阶段总结扩成论文章节/摘要/投稿稿。
- `academic-plotting`：论文级图件与图注；本项目仍要求 PNG+SVG。
- `code-review`：合并前审查本次 `chapter.py` 与文档改动。
- `writing-for-agents`：如果还要维护 handoff/AGENTS.md 这类给 agent 读的文档。
- `handoff`：下一次会话结束时继续写增量交接。