# Handoff：逆问题最终评估规范与 forward_seen 对比

> 最新交接（2026-11-25 纯正模型 joint10）：[handoff_20261125_pure_forward_augmentation.md](handoff_20261125_pure_forward_augmentation.md)。本文件保留 11-24 历史状态。

- 日期：2026-11-24
- 当前分支：`experiment/forward-augmented-tcn`
- 当前 HEAD：`828fc6c feat(tcn): complete full old-protocol augmentation comparison`
- 远端：`origin` = `git@github.com:dzkd2019/tcn_union.git`
- 工作区：本次提交前有未提交的评估规范、forward_seen 实验代码/产物与文档；提交后以 `git log -1` 和 `git status` 为准
- 数据约束：62 条不可扩充，不补测/不补采/不重采；不写 `ppm`

## 1. 本次完成

### 1.1 把最终评估要求写入 `AGENTS.md`

新增 `AGENTS.md` §10「逆问题最终评估规范（TCN 浓度反演）」，适用于 OOF、验证、锁定测试与集成预测：

- §10.1 必报指标：逐气体 `MAPE`、`NRMSE`、`3σ`、`95% CI`，以及三气体宏平均 `MAPE/NRMSE/3σ`；每次评估保存逐实验预测与 `metrics.csv`。
- §10.2 必报图件：每气体一张 1×2 的有符号误差散点图（绝对误差 + 相对误差，含 ±1σ/±2σ/±3σ 线），每气体一张只画均值与 95% CI 的图（不显示原始散点）。
- §10.3 目录：每次评估一个 `artifacts/<run_name>/`，所有图放在该目录的 `figs/`，不跨运行混放；TCN 图不放入 `sensor_array/figures/`。
- §10.4 时机：对外汇报/论文/锁定结论前必须满足 §10.1–§10.3；锁定 12 只评估一次，评估后不得据此调参。

关键口径：

- `3σ = 3 × std(相对误差, ddof=1)`，是实验间离散度，不是置信区间；
- `95% CI = mean(相对误差) ± t(0.975,n-1) × std/√n`，是**均值**的区间，必须写 `n`；
- `NRMSE = RMSE / 浓度量程`，量程来源（训练折/开发集/测试集）必须写明，各臂一致；
- `MAPE` 在浓度 0 时无定义，写空值并说明原因。

### 1.2 同步绘图指南

- `docs/plotting_guide_tcn_10s.md` 增加「最终评估图件与目录规范（2026-11-24）」：目录模板、`metrics.csv` 字段、每气体图名、登记要求。
- `sensor_array/docs/plotting_guide.md` §5 增加一条：TCN 逆问题图按上述规范放在 `artifacts/<run_name>/figs/`，不放进正问题图目录。

### 1.3 forward_seen 与 PCA-VAE 的初步对比

问题：如果正模型只在训练集已有的 40 个浓度标签上合成，和 PCA-VAE 比如何？

设置：50 条开发集、5 折、10 s、3 seed、1 套低浓度配置；两者都合成 320 条（40 real + 40 noise + 120 base + 120 edge），共用真实训练折 scaler；锁定 12 不参与。

| 方法 | 宏 3σ（15 折均值） | 中位 3σ | 宏 MAPE |
|---|---:|---:|---:|
| real | 12.76% | — | 3.48% |
| PCA-VAE | **9.26% ± 2.20** | **8.53%** | **2.36%** |
| forward_seen | 10.12% ± 3.74 | 8.96% | 2.55% |

逐折配对 `forward_seen − VAE`：均值 **+0.86 pp**（正模型更差），6/15 折改善，Wilcoxon **p=0.454**。单种子 pilot 曾出现 10.03% vs 10.08% 的“打平”，三种子后消失。

结论：只使用已有标签时，正模型没有优于 PCA-VAE；同标签同 \(R_0\) 的曲线近似重复，VAE 的 latent 提供形状多样性。正模型的价值仍在**新浓度组合**（H₂）。

完整说明：`sensor_array/docs/forward_seen_vs_vae.md`。

## 2. 本次代码改动

`tcn_union/experiments/full_protocol_augmentation.py`：

- 新增 `forward_seen` 臂：`_training_concentrations()` 从 40 个训练标签有放回抽 120 个条件；`_initial_for_labels()` 用对应真实 \(R_0\)；`forward_seen_base` 合成绝对响应；再加 `edge_forward`。
- `_paired_rows` 加 `if "real" not in arms: return []`，并把 `forward_seen` 视为增强臂，避免只跑 `vae/forward_seen` 时汇总阶段 `StopIteration`。
- 此前已有 `forward_pretrain` / `forward_unseen` / `forward_h2`、`--real-scaler`、`_transformed_labels`、`_train_pretrain`。

`tcn_union/training/train.py`：

- `_train_network` 新增可选 `sample_weight`（形状同 `y`，逐样本×气体乘到 SmoothL1），默认 `None` 行为不变；用于 `forward_h2` 只给 H₂ 头监督。

本次评估规范只写入文档，**尚未实现**对应的指标/绘图代码；下一次 TCN 评估需要先补 `metrics.csv` 与 §10.2 的图。

## 3. 当前 forward 增强进展

- 完整旧协议四臂（10 s，3 seed × 5 折 × 2 配置 × 6 checkpoint）：real 10.79% 3σ / 2.77% MAPE；VAE 7.47 / 1.79；forward 10.92 / 2.61；forward+vae 9.10 / 2.11。VAE-only 最好，正模型单独无增益。
- `forward_h2`（合成只监督 H₂ 头）：H₂ 12.19 → 10.69%，3/3 seed 改善；但 NH₃/NO₂ 变差，整体 14.47 → 16.35。
- 逐气体组合（H₂ 用 `forward_h2`，NH₃/NO₂ 用 real）：3 seed 级 OOF 宏 14.47 → 13.55（+0.92 pp）；3-seed 平均预测 11.28 → 10.81；fold 配对 9/15、p=0.277，幅度小、不显著。
- `forward_seen`（只在已有标签上合成）：如上，未超过 VAE。

结论：正模型的可复现增益目前只出现在 H₂ 与新浓度组合方向，且幅度小、fold 级不显著；不能把 `forward_seen` 当成正模型增益的主证据。

## 4. 下一步

1. **先补评估代码**：按 `AGENTS.md` §10 实现统一的 `metrics.csv` 与每气体三张图（有符号绝对/相对散点 + 95% CI），并在新 run 目录落盘；不要直接用旧图改口径。
2. **决定 forward 主线**：若继续追“正模型必须有增益”，优先做 `forward_seen_r0`（随机 \(R_0\)）1 seed pilot，验证是否只是 \(R_0\) 太死板；否则回到“新组合 + H₂ 专用”的 2 配置/6 checkpoint 完整协议。
3. **控制实验**：补 `real_dup` 等样本量对照，区分“更多曝光”与“正模型合成”的贡献；不要用锁定集选方案。
4. **最终候选**：只对最终方案跑一次锁定 12，按 §10 输出指标与图，之后不再调参。

## 5. 复现

```bash
# forward_seen 单种子 pilot
.venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 --configs 1.0 \
  --arms real vae forward_seen --real-scaler \
  --output artifacts/augmentation/diagnostics/forward_seen_pilot

# forward_seen 三种子（只跑 vae/forward_seen；real 基线复用同协议结果）
.venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --configs 1.0 \
  --arms vae forward_seen --real-scaler \
  --output artifacts/augmentation/diagnostics/forward_seen_seeds
```

- 数据/预测：`artifacts/augmentation/diagnostics/forward_seen_pilot/`、`artifacts/augmentation/diagnostics/forward_seen_seeds/`、`artifacts/augmentation/diagnostics/forward_seen_seeds.log`、`artifacts/augmentation/diagnostics/forward_seen_seeds/folds_from_log.csv`
- 对比文档：`sensor_array/docs/forward_seen_vs_vae.md`
- 评估规范：`AGENTS.md` §10；图目录：`docs/plotting_guide_tcn_10s.md`

## 6. 已知坑

- **单种子不可靠**：forward_seen 单种子打平，三种子后 VAE 更好；后续至少 3 seed。
- **3 seed 那次没跑 real 臂**：`real` 基线取自同协议同种子同折的已有结果；汇总阶段因缺 real 报错，逐折日志完整，数字由日志解析；`_paired_rows` 已修复。
- **合成臂真实曝光**：1:1 混合会让真实样本曝光减半；比较时同时报等真实曝光/等样本量版本。
- **R0 与 scaler**：合成臂必须用真实训练折 scaler；R0 要么同标签真实值，要么只从训练折经验分布抽，不能碰验证/锁定。
- **锁定 12**：不参与任何调参、早停、图件选择；最终只评估一次。
- **图件规范**：PNG 300 dpi + 同名 SVG；不输出 PDF；不写 `ppm`；每次 run 一个 `figs/` 并更新 plotting guide。

## 7. 交接检查

- [ ] `git status` 干净（本次提交后）
- [x] `AGENTS.md` §10 已写入
- [x] `docs/plotting_guide_tcn_10s.md` 与 `sensor_array/docs/plotting_guide.md` 已同步
- [x] forward_seen 3 seed 结果与文档已生成
- [ ] 评估指标/绘图代码尚未按 §10 实现（下一次评估前必须补）
- [ ] 锁定 12 尚未评估
