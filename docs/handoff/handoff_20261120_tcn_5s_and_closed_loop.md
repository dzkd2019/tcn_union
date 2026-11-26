# Handoff：5 s 前缀 TCN 与正模型闭环（2026-11-20）

> **历史交接（2026-11-20）**：本文件的闭环产物与 fig06 已在本分支后续清理中删除；
> 当前状态见 [handoff_20261123_map_removal_and_plan.md](handoff_20261123_map_removal_and_plan.md)
> 与 [plan_20261123.md](../../sensor_array/docs/plan_20261123.md)。

## 0. 一句话

在 `experiment/mlp-double-exponential` 分支上，5 s 前缀 TCN 已按 10 s 的 3σ 配方完成训练/评估，并用“当前双指数 + 三路 MLP 正模型”做了 5 s/10 s 的合成增强闭环对照；10 s 既有代码和产物全部保留。当前最需要接着做的是：闭环结果的多种子复现，以及 9 s 前缀是否值得补 final/test。

## 1. 仓库状态

- 分支：`experiment/mlp-double-exponential`
- HEAD：`746295b feat(tcn): add 10 s relative-error 95% CI figure`
- 工作区：clean
- 远端：`origin/experiment/mlp-double-exponential` 落后本地 3 个提交（尚未 push）
- 分支历史：

```text
746295b 10 s 相对误差 95% CI 图
d94820e 闭环协议修正 + 结果
59e21f2 5 s 实验 + 闭环初版
4fe724a sensor-array：fig06 + 文档/校验清单
```

- 原 `experiment/tcn-5s` 已被删除；上述 5 s/闭环提交已包含在当前分支。
- 数据硬约束：只使用 `data/0.csv`–`data/61.csv` 共 62 条实验，**数据不再扩充**；锁定 12 条（`2,7,15,20,23,27,31,38,41,46,55,60`）不参与调参/选型，只做一次性测试描述。

## 2. 已完成的工作

### 2.1 5 s 前缀 TCN（3σ 配方）

协议：与 10 s 最终方案相同，两配置（A `p=1.0`；B `0,0.3,1.0`）× 3 seed（20260923/24/25），VAE + 噪声 + 边缘增强、逐气体头、裁剪到开发集范围；OOF 50，锁定 12 只评估一次。

| 前缀 | OOF 50 宏 3σ | 锁定 12 宏 3σ | OOF MAPE | 测试 MAPE |
|---|---:|---:|---:|---:|
| 10 s（既有基线） | **7.47%** | **7.11%** | 1.79% | 1.67% |
| 5 s | **12.99%** | **11.85%** | 2.98% | 2.77% |

5 s 逐气体 OOF 3σ：H₂ 12.64 / NH₃ 9.09 / NO₂ 17.25；锁定 12：11.80 / 8.57 / 15.17。只有 NH₃ 低于 10%，H₂/NO₂ 不满足逐气体 3σ<10%。

9 s 补充（3 seed × 2 配置，OOF only）：**10.07 / 5.90 / 10.30，宏 8.76%**；6/7/8 s 单 seed 筛选宏 13.54/14.52/12.91%，更差。9 s 只差一点，但还没有 final + 锁定测试，不能现在宣称“9 s 可用”。

代码改动：
- `tcn_union/core/data.py::make_features` 支持任意 0.2 s 整数倍前缀；
- `tcn_union/training/gan_train.py` 新增 `--seconds`（2–10，默认 10）并透传到 config/checkpoint/selection；
- `tcn_union/evaluation/ensemble_evaluate.py` 窗口放宽到 1–60 s；
- `tcn_union/augmentation/{gan,vae,pca_generator,direct_vae}.py` 的 seconds 校验放宽。

### 2.2 当前正模型闭环增强

初版曾得到“合成让结果变差”，复核发现是协议问题，不是归一化 bug：

1. 合成时 `R0` 随机配，破坏了 `raw_delta` 绝对通道与标签/基线的关系；
2. `real+synthetic` 把合成和真实混在一起拟合输入 scaler；
3. 初版只做 2:1 直接混合，真实样本曝光减半，且没有 pretrain+finetune。

修正后：前 40 条合成与训练实验同浓度、同 `R0`；所有臂共用真实训练折 scaler；新增 `real+synthetic(1:1)` 与 `pretrain+finetune`（200 合成 + 100 真实）。

| 窗口 | 臂 | 宏 3σ / % | 宏 MAPE / % | 配对 p（vs real） |
|---|---|---:|---:|---:|
| 5 s | real | 15.51 ± 5.04 | 4.70 | — |
| 5 s | real+synthetic(1:1) | **11.57 ± 1.88** | 3.43 | 0.112 |
| 5 s | pretrain+finetune | 14.48 ± 2.22 | 4.04 | 0.596 |
| 10 s | real | 14.50 ± 5.36 | 4.27 | — |
| 10 s | real+synthetic(1:1) | 13.61 ± 5.18 | 3.65 | 0.554 |
| 10 s | pretrain+finetune | **12.26 ± 1.54** | 3.44 | 0.323 |

结论：合成曲线**有改善迹象，但不是稳定显著增强**；5 折、1 seed 的 p 值都 >0.1，改善主要来自把“real 特别差”的折 2 方差压下来。它更像正则化/降方差，不是提供新的浓度信息。

### 2.3 10 s 相对误差 95% CI 图

- 图：`artifacts/inverse/eval_10s/figs/fig_10s_relative_error_ci95.{png,svg}`
- 脚本：`artifacts/inverse/eval_10s/scripts/plot_ci95_10s.py`
- 索引：`docs/plotting_guide_tcn_10s.md`
- 定义：每组相对误差的均值 ± \(t_{0.975,n-1}s/\sqrt n\)；OOF 50 与锁定 12 分列。画的是**均值的 CI**，不是单次实验的 95% 预测区间；单次波动看散点、标准差或 3σ。

## 3. 关键文件

- 5 s 方案与结果：`docs/tcn_5s_evaluation.md`
- 闭环协议与修正结果：`docs/closed_loop_current_forward.md`
- 5 s 绘图索引：`docs/plotting_guide_tcn_5s.md`
- 10 s 绘图索引：`docs/plotting_guide_tcn_10s.md`
- 10 s 原方案：`docs/sigma-target/3sigma_方案.md`
- 旧交接（sensor_array 主模型）：`docs/handoff/handoff_20261119_mlp_per_gas.md`

产物：
- `artifacts/inverse/eval_5s/`：`oof/`、`final/`、`evaluation/`、`all62_predictions.csv`、`all62_error_stats.csv`、`figs/`、`scripts/`
- `artifacts/inverse/prefix/6s..9s/`
- `artifacts/analysis/closed_loop/closed_loop.csv`
- 10 s 既有产物：`artifacts/inverse/eval_10s/`（未改动，保留）

## 4. 复现命令

```bash
# 5 s 闭环对照（GPU 推荐；当前脚本 5 折 × 2 窗口 × 3 臂，约 15 min）
mkdir -p artifacts/analysis/closed_loop
.venv/bin/python -m tcn_union.experiments.closed_loop

# 5 s 误差图
.venv/bin/python artifacts/inverse/eval_5s/scripts/plot_all62_5s.py

# 10 s 95% CI 图
.venv/bin/python artifacts/inverse/eval_10s/scripts/plot_ci95_10s.py

# 回归测试
.venv/bin/python -m unittest discover -s tests
```

## 5. 下一步（按优先级）

1. **闭环多种子复现**：把 `closed_loop.py` 的 `SEED` 扩到 3 个（20260923/24/25），重跑 5 s/10 s 三臂，报告均值±标准差和配对检验；确认 5 s 1:1、10 s pretrain 的改善是否跨种子。当前是 1 seed，不能写进主结论。
2. **9 s 是否补 final/test**：9 s OOF 宏 8.76%，H₂/NO₂ 仍略高于 10%。如果目标是“最短且逐气体<10%”，需要补 9 s 的 final + 锁定 12 一次性评估；如果保持 10 s 主方案，则 9 s 只作为探索记录。
3. **最短前缀**：6–8 s 单 seed 已明显更差；不要在没有多种子/最终评估的情况下继续扫更短前缀。
4. **push/PR**：本地领先 origin 3 个提交；push 前确认 `artifacts/` 体积和是否需要提交 5 s OOF 中间产物。
5. **不要重跑 10 s 主方案**：10 s 产物是冻结基线，除统一绘图外不要覆盖。

## 6. 环境

- `.venv/bin/python`：CPython 3.14.10
- `torch 2.14.0+cu130`；GTX 1650 4 GB（`cuda True`）。CPU 也能跑，但 MLP/闭环训练明显更慢。
- 绘图：`MPLCONFIGDIR=/tmp/mpl`，中文字体 `Microsoft YaHei`/`Noto Serif CJK SC`；PNG 300 dpi + 同名 SVG，不输出 PDF。
- 浓度不写 `ppm`（数据未注明单位）；绝对误差写“原始单位”。

## 7. 容易踩的坑

- **输入标准化**：`tcn_union/training/train.py::_train_network` 默认对输入/标签做 z-score；`scaler_stats` 可传入真实训练折统计量，`initial_state` 可载入预训练权重。不要再写“没有归一化”的自定义训练循环。
- **合成 `R0`**：`raw_delta` 前 3 通道是绝对响应，随机配 `R0` 会破坏基线-标签关系；同浓度合成必须同 `R0`。
- **真实曝光**：1:1 混合会让每个 epoch 的真实样本减半；比较时必须说明预算，不能把“1:1 不显著”当成“合成无用”。
- **3σ 不是 CI**：3σ = 3×标准差，描述单次波动；均值 CI 会窄一个数量级。
- **锁定 12**：只做一次性测试描述；任何调参、选型、分位数拟合都不能碰。
- **图件索引**：5 s 与 10 s 的 plotting guide 分开维护，且与 `sensor_array/docs/plotting_guide.md` 分开。

## 8. Suggested skills

- `deja-history`：改代码/写文档前查项目历史决策，避免重复踩坑。
- `ml-training-recipes`：多种子、早停、预算对齐和训练诊断。
- `academic-plotting`：把这批图整理成论文/汇报版式时使用。
- `code-review`：合并/推送前审查 5 s 代码改动与闭环协议。
