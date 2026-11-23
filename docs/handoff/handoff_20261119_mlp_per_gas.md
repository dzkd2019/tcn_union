# Handoff：双指数 + 三路独立 MLP（2026-11-19 晚）

> **历史交接（2026-11-19）**：fig06 与 MAP/区间内容已在本分支后续清理中删除；
> 当前状态见 [handoff_20261123_map_removal_and_plan.md](handoff_20261123_map_removal_and_plan.md)
> 与 [plan_20261123.md](../../sensor_array/docs/plan_20261123.md)。

> 面向明天工位开发机继续开发。
>
> 根目录的 `handoff_20261117.md` 是同日更长的详细记录；本文件只保留开工必需的状态、结论、命令和下一步。

## 0. 快照

- 仓库根：`/home/mrawa/code/deep-learning/tcn_union`；工位路径不同时，以 `git rev-parse --show-toplevel` 为准。
- 分支：`experiment/mlp-double-exponential`
- HEAD：`121eb05 feat(sensor-array): switch to double-exponential + per-gas MLP residual`
- 工作区：已提交并推送到 `origin/experiment/mlp-double-exponential`；本次新增
  `fig06` 与对应文档修改尚未提交（以 `git status` 为准）。
- 后台任务：无。
- 环境：`.venv/bin/python`（CPython 3.14.10）；绘图前设 `MPLCONFIGDIR=/tmp/mpl`。
- GPU：AMD Radeon RX 7800 XT 16 GB，ROCm 7.2；`.venv` 中 torch 2.14.1+rocm7.2。
  三路独立 MLP 训练时 `device=cuda`；无 GPU 会自动回退 CPU，只是更慢。
- 图表规范：PNG 300 dpi + 同名 SVG，不输出 PDF。
- 数据：`data/0.csv`–`data/61.csv`，共 62 条；传感器已损坏，**不可补测、补采或重采**。

## 1. 当前模型与结果

当前只保留两套模型：

| 模型 | 曲线宏 nRMSE | 10 s 宏中位 ARE | 5 s 宏中位 ARE |
|---|---:|---:|---:|
| 纯双指数骨架 | 3.30% | 4.99% | 7.42% |
| **双指数 + 三路独立 MLP** | **1.48%** | **2.14%** | **2.96%** |

- 默认推荐模型：`sensor_array/artifacts/models/mlp_per_gas.json`；`double.json` 保留作骨架对照。
- 相对骨架：曲线 **59/62** 折改善（中位 −1.80 pp，Wilcoxon p=1.36e−11）；
  10 s 下游 **47/62**（−2.14 pp，p=3.93e−7）；5 s **45/62**（−3.14 pp，p=2.50e−6）。
- 开发 50 / 锁定 12：曲线 47/50、12/12；10 s 下游 36/50、11/12；5 s 35/50、10/12。
- 残差诊断：预测残差与真实残差相关 **0.866 / 0.722 / 0.883**，标准差比 0.79 / 0.79 / 0.96。
- 限制：单 seed、固定 600 轮、无早停；下游 \(\sigma\) 为训练折 in-sample；未重做停止/剖面/嵌套校准。
- 共享三输出 MLP 已从代码和产物中删除，不再作为主线。

P4 区间校准（10 s，90%）：

| 方法 | 开发覆盖 H₂/NH₃/NO₂ | 开发中位宽度 / % | 锁定覆盖 / % |
|---|---:|---:|---:|
| 原始 Laplace | 32.0 / 30.0 / 6.0% | 0.6 / 0.7 / 0.9 | 8.3 / 41.7 / 25.0 |
| **全局共形（默认）** | **92.0 / 92.0 / 92.0%** | **3.5 / 3.4 / 8.7** | **100.0 / 91.7 / 100.0** |
| NH₃ 档位共形 | 90.0 / 90.0 / 90.0% | 3.9 / 5.4 / 9.6 | 100.0 / 91.7 / 100.0 |
| 采集块共形 | 84.0 / 84.0 / 84.0% | 3.5 / 3.0 / 9.2 | 91.7 / 91.7 / 100.0 |

- 默认全局共形；NH₃ 档位/采集块只作诊断。
- 5 s 趋势相同：Laplace 34.0 / 46.0 / 32.0%，全局共形 92.0%，宽度 6.5 / 9.5 / 13.7%。
- 未重做：停止策略、剖面似然、嵌套协方差。

## 2. 今晚新增的文档与图件改动

1. `sensor_array/docs/learning_guide.md` §7.3 已扩写为
   **“TCN 加速与当前 MAP：两种逆模型”**，写入同口径比较、三层加速意义、两者互补关系和严格对照方案。
2. `AGENTS.md` 新增 **§9 绘图文档维护（sensor_array）**：
   `sensor_array/docs/plotting_guide.md` 必须逐图登记图号、输出文件、绘图脚本、绘图函数、数据源、版式与目录；
   新增图或改绘图脚本/数据源/版式/输出路径时，同一次改动必须更新该文档；不同方案图件分目录。
3. 本 handoff 新建于 `docs/handoff/`，不再把根目录旧 handoff 当最新入口。
4. 新增 `fig06_representative_curves`：当前模型最优（file 31）、中位（file 27）、
   最差（file 0）、NH₃ 低端（file 4）与 NH₃/NO₂ 高端（file 61）五个案例，每行三路气体，
   同时画实测、纯骨架与三路独立组合；`plotting_guide.md`、`evaluation.md`、
   `README.md` 已同步。

## 3. 与 TCN 的关系（今天讨论的结论）

**当前 MAP 与 TCN 都是逆模型，不是“正模型替代逆模型”。**

- TCN：amortized inference，训练时把映射学进网络，推理一次前向；
- 当前 MAP：analysis-by-synthesis，每条曲线调用正模型做逐样本优化。

同口径 10 s 指标（3σ = 逐气体相对误差标准差的 3 倍）：

| 口径 | TCN（第一阶段） | 当前三路独立 + MAP |
|---|---:|---:|
| 锁定 12 宏 3σ | **7.11%** | 8.60% |
| 锁定 12 MAPE | **1.67%** | 2.16% |
| 开发 50 宏 3σ | **7.47%** | 13.14%\* |
| 开发 50 MAPE | **1.79%** | 3.10%\* |
| 62 点合并宏 3σ | **7.69%** | 12.35% |

\* 当前“开发 50”的 62 折 LOGO 训练集包含锁定 12；当前“锁定 12”每折训练集包含另外 11 条锁定文件。

这不是“50 条训练并冻结、12 条盲测”的严格协议；TCN 的 7.11% / 1.67% 才是该协议的结果。

当前 3σ/MAPE 数字是从 `sensor_array/artifacts/evaluation/downstream.csv` 按 TCN 同一公式重算的，尚未改动 `evaluation.md`。

结论：**当前中位误差低，但尾部仍不如 TCN；TCN 的响应加速、计算摊销和尾部优化意义都还在。**

当前模型的价值是可解释、中位精度好、有区间、支持 5 s；两者应互补，而不是删除 TCN。

## 4. 代码地图

- `sensor_array/core/shapes.py`：逐实验一阶形状初值。
- `sensor_array/core/double_exponential.py`：双指数骨架，69 参数（23/通道），
  `fit/predict_relative/predict_response/save/load`。
- `sensor_array/core/mlp_per_gas.py`：三路独立 MLP，三个 `7→64→64→1` 网络，
  每路 4,737 参数、合计 14,211；PyTorch GPU 全批 600 轮，NumPy 推理。
- `sensor_array/analysis/estimation.py`：噪声估计、浓度边界、MAP + Laplace。
- `sensor_array/analysis/calibration.py`：P4 区间校准（Laplace / 全局 / NH₃ 档位 / 采集块共形）。
- `sensor_array/experiments/evaluate.py`：62 折 LOGO、曲线与 5/10 s 下游评估。
- `sensor_array/plotting/model.py`：fig01–fig06。
- 已删除：`core/hybrid.py`、`core/grey_box.py`、`core/baselines.py`、`core/mlp_double.py`、
  `analysis/identifiability.py`、`analysis/residuals.py`、`analysis/degradation.py`、
  `experiments/logo.py`、`experiments/validation.py`、`experiments/structured.py`、
  `experiments/inverse_augmentation.py`、`experiments/fair_augmentation.py`、
  `experiments/model_comparison.py`、`plotting/chapter.py`、`plotting/response.py`、
  `plotting/model_comparison.py`。

## 5. 产物与文档

- 评估：`sensor_array/artifacts/evaluation/{curve_metrics,curve_summary,downstream,downstream_summary}.csv`、
  `predictions.npz`；
- 模型：`sensor_array/artifacts/models/double.json`、`mlp_per_gas.json`；
- 校准：`sensor_array/artifacts/calibration/{intervals,summary,by_group}.csv`；
- 图：`sensor_array/figures/model/fig01_design_and_data`、`fig02_curve_examples`、
  `fig03_curve_error`、`fig04_downstream`、`fig05_interval_calibration`、
  `fig06_representative_curves`，各 PNG + SVG；
- 文档：`sensor_array/docs/model.md`、`evaluation.md`、`plotting_guide.md`、
  `learning_guide.md`、`beginner_guide.md`、`AGENTS.md`；
- 快照：`sensor_array/MANIFEST.md`、`sensor_array/SHA256SUMS`（10 个 artifacts + 12 个图件 = 22 行）。

## 6. 明天开工先做这 5 件事

```bash
cd /home/mrawa/code/deep-learning/tcn_union
git status --short
git branch --show-current
git log -1 --oneline
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

1. 确认仍在 `experiment/mlp-double-exponential`，HEAD 为 `121eb05`；若路径/分支不同，先不要改代码。
2. 读 `docs/handoff/handoff_20261119_mlp_per_gas.md`（本文件）、`learning_guide.md` §7.3、
   `AGENTS.md` §8–§9、`sensor_array/docs/plotting_guide.md`。
3. 确认 GPU：`torch.cuda.is_available()` 应为 True；无 GPU 会自动回退 CPU，但 62 折会明显变慢。
4. 先审 `git status`，**不要直接 `git add -A`**：当前包含大量旧模型删除与未跟踪的新代码/产物。
5. 从下面“下一步”选一项开工，不要同时改模型/图/文档。

## 7. 下一步（按优先级）

1. **严格 TCN 对照**：只用 50 条开发集训练并冻结当前模型，在 12 条锁定集上**只评估一次**，
   同时报告宏 3σ、MAPE、中位误差和单样本延迟；锁定集不再调参。
   这是回答“当前 MAP 是否真的超过 TCN”的唯一干净协议。
2. **3–5 seed 复验**：确认 1.48% / 2.14% / 2.96% 不是单 seed 偶然；若稳定，再把固定 600 轮改成按折早停。
3. **P4 收尾**：在当前模型上重做停止策略与剖面似然；协方差校准若要恢复，先做嵌套验证。
4. **论文/答辩**：主模型用三路独立组合，纯骨架作对照；图统一 fig01–fig06，按 `plotting_guide.md` 复现。
5. **提交前**：审 `git status` 与 `git diff --stat`；不要把旧 archive、根 `artifacts/` 或 mock 图带进来。

## 8. 复现命令

```bash
# 62 折评估（推荐模型）；本次约 766 s（12.8 min）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

# P4 区间校准
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.analysis.calibration

# 图（PNG + SVG）
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model

# 单元测试
.venv/bin/python -m unittest discover -s tests
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s sensor_array/tests

# 快照
cd sensor_array && sha256sum -c SHA256SUMS
```

已验证（2026-11-19）：62 折评估 exit 0（766 s）；校准 exit 0；67 项根测试 OK（1 skipped）；
6 项 sensor_array 测试 OK；compileall、模型 save/load、`git diff --check` 通过；
图件 12 个（6 组 PNG+SVG），0 PDF；SHA256SUMS 22 行全部成功。

## 9. 不可违反的约定

- 只使用现有 62 条实验；不补测、不补采、不重采。
- 锁定测试 `{2,7,15,20,23,27,31,38,41,46,55,60}` 只做描述性复核，不参与选型、调参或分位数拟合。
- 浓度单位未注明，统一写“原始单位”，不写 ppm。
- 不修改 `tcn_union/training/gan_train.py`、`tcn_union/augmentation/vae.py`、
  `tests/test_vae.py`、`pyproject.toml`、`uv.lock`；不碰仓库根 `artifacts/`。
- 不 commit、不 push，除非用户明确要求。
- 图件 PNG 300 dpi + SVG，不出 PDF；新增/改图必须同步 `plotting_guide.md`。
