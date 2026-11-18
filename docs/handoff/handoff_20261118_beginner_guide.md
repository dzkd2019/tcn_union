# Handoff — 2026-11-18 — 新手概念文档

## 0. 会话状态

- 工作区：/home/mrawa/code/deep-learning/tcn_union
- 当前没有正在运行的后台实验或分析任务。
- Git HEAD 仍是 0e2a9b1；P1/P3/P4/P5/P6 的改动均未提交，也未 push。
- 用户最后一条指令是 /handoff（m03118）；再上一条实质请求是新增一份面向新手的学习文档（m02967）。
- 本次会话的增量工作：新增 sensor_array/docs/beginner_guide.md，并完成链接、NMAE 定义修正和验证。没有改模型代码，没有新图，没有新实验。

## 1. 先读这些，避免重复

- 项目全量交接：/home/mrawa/code/deep-learning/tcn_union/handoff_20261117.md
  这里已有数据约束、P1–P6 状态、目录结构、复现命令和已知限制；本文件只记录本次增量。
- 结果与证据：/home/mrawa/code/deep-learning/tcn_union/sensor_array/docs/chapter_results.md
- 路线图与 P2 暂缓理由：/home/mrawa/code/deep-learning/tcn_union/sensor_array/docs/roadmap.md
- 项目叙事版学习文档：/home/mrawa/code/deep-learning/tcn_union/sensor_array/docs/learning_guide.md
- 本次新增的概念双解文档：/home/mrawa/code/deep-learning/tcn_union/sensor_array/docs/beginner_guide.md
- 项目入口：/home/mrawa/code/deep-learning/tcn_union/sensor_array/README.md

不要在本文件里重复 chapter_results 的数字或 roadmap 的里程碑；需要时直接引用上述文件。

## 2. 本次增量（已完成）

### 2.1 新文档

- 文件：sensor_array/docs/beginner_guide.md
- 标题：《从零读懂气体传感器阵列：概念双解手册》
- 规模：698 行，约 19.3 KB。
- 定位：面向没有传感器建模和统计基础的新手；不是改写 learning_guide，而是逐概念解释。
- 结构：0 读法/符号；1 测什么；2 验证协议；3 指标；4 模型；5 逆问题与不确定度；6 决策；7 合成数据增强；8 统计与可复现；9 项目数字地图；10 常见误区 12 条；11 术语双解速查表；12 自测 6 题。
- 每张概念卡片统一格式：
  - **严谨说法**：定义、公式、前提、适用条件；
  - **通俗说法**：日常类比；
  - **边界**：类比在哪里失效、结论何时不成立。
- 关键公式覆盖：相对响应、一阶惯性、MAPE/NMAE/nRMSE、Fisher/CRLB、MAP、Laplace Hessian、白化、共形、剖面似然、期望损失。
- 项目数字与 chapter_results 的 §8、§13 对齐；浓度只写“原始单位”，没有写 ppm。

### 2.2 文档入口

- sensor_array/docs/learning_guide.md 开头的“配套”块加入 beginner_guide.md 链接。
- sensor_array/README.md 的导读、目录树、代码说明三处加入 beginner_guide.md。
- handoff_20261117.md 的 §3.4 文档清单加入 beginner_guide.md。

### 2.3 顺手修正

- sensor_array/docs/chapter_results.md:15 原先把 NMAE 写成“平均绝对相对误差”，与代码和同文件第 180 行不一致。
- 已改为：NMAE = 平均绝对误差 / 训练折该气体浓度量程。
- 依据：sensor_array/experiments/fair_augmentation.py 中 nmae = mean(|err|/label_range)；同一文件的 §3.3 也这样定义。

### 2.4 验证

- .venv/bin/python -m unittest discover -s tests：Ran 67 tests，OK。
- git diff --check -- '*.md'：exit 0。
- 新文档自定义校验通过：16 组公式定界符成对、反引号成对、无控制字符、无尾随空白、所有相对链接可解析。
- 新文档目前是 git untracked 文件；没有 commit，没有 push。

### 2.5 经验

- 第一次用 TypeScript 模板字符串写 Markdown 时，\theta / \hat 被当成转义序列，分别变成了制表符和 hat。已修复并全量检查；以后程序化生成公式应使用原始字符串或双反斜杠。
- 本次没有新增图件；如果后续加图，仍按 AGENTS.md §8 输出 PNG 300 dpi + SVG，不出 PDF。

## 3. 硬约束（继续保留）

- 只能使用 data/0.csv–data/61.csv 共 62 条实验；不得建议补测、补采或扩充数据。
- 12 条锁定测试 TEST_IDS={2, 7, 15, 20, 23, 27, 31, 38, 41, 46, 55, 60} 只做描述性统计，不用于选模型、调参或挑方法。
- 运行 Python 用 .venv/bin/python，不要用 uv run。
- 不 commit、不 push，除非用户明确要求。
- 不动 pyproject.toml、uv.lock 和 tcn_union/ 下的其他文件；本任务范围是 tcn_union/sensor_array/ 的文档。
- 浓度单位未注明，统一写“原始单位”，不要写 ppm。
- P2 仍暂缓；重启条件见 roadmap 和 chapter_results，不要仅因“想提升指标”重启。

## 4. 建议下一步

当前没有未完成的代码任务。根据用户下一句话选择：

1. 如果继续文档：先让用户确认目标读者（完全新手 / 有 ML 基础 / 论文读者），再决定是精简 beginner_guide，还是把它拆成“概念篇 + 项目篇”。不要默认继续加长。
2. 如果进入论文写作：从 chapter_results 提取证据，beginner_guide 只作为概念解释素材，不要把“通俗说法”直接写进论文正文。
3. 如果继续研究：P2 仍暂缓；P4/P5/P6 已完成，除非用户明确要求，不要重跑六模型或 P5 训练。
4. 修改任何文档前，先读 handoff_20261117.md 对应章节，避免和已有结论冲突。

## 5. Suggested skills

Next agent should call the Skill tool for these when relevant:

- deja-history — before editing, recall this project's settled constraints and past decisions (especially the no-new-data, no-ppm, no-commit rules).
- academic-plotting — only if new figures are requested; it knows the PNG+SVG convention and publication style.
- scientific-writing — if the beginner guide is being adapted into paper prose or a report section.
- code-review — if the next session reviews the accumulated P1–P6 diff rather than writing docs.

## 6. 交接检查

- [x] 新文档路径和结构已记录
- [x] 入口链接已记录
- [x] NMAE 修正已记录
- [x] 测试与文档校验结果已记录
- [x] 未提交/未 push 状态已记录
- [x] 硬约束和 P2 暂缓已记录
- [ ] 用户确认下一步方向（待下一会话）
