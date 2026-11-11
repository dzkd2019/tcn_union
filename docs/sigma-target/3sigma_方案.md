# 无 Ridge 方案：用低浓度损失加权替代额外线性预测

> 分支：`experiment/3sigma-noridge`（从精简方案 `24597f3` 切出）  
> 动机：Ridge 是一个额外的线性模型，部署/解释都显得“多了一层”；希望在**只保留 TCN 模型族**的前提下，用训练目标（低浓度 Loss 加权）而不是后处理线性模型来降 3σ。  
> 结论：无 Ridge 方案在 OOF 与 12 点锁定测试集上**逐气体 3σ 均 <10%**（OOF 9.31 / 4.95 / 8.16，测试 6.46 / 5.71 / 9.16），62 点合并口径 8.86 / 5.05 / 8.59；相比“TCN + Ridge”方案 OOF 略差（7.47% vs 7.06%），测试宏平均 7.11% vs 7.18%，符合“性能略微下降”的预期。
>
> **状态**：该方案已合并到 `dev` 分支并推送到 `origin`；旧迭代分支与旧产物已清理。  
> **测试集读取声明**：本方案的测试结果是整个 3σ 任务的**第 7 次读取**。候选在读取测试集前已用 OOF 完全冻结（配置、加权强度、seed 数、集成方式都只在 OOF 上选择），没有根据测试结果调参。

---

## 1. 为什么去掉 Ridge，换什么

精简方案（上一分支）的最终流水线是：

```text
单配置 TCN（逐气体目标 + 逐气体头 + 边缘 VAE 扩充）× 3 seed
→ Ridge（57 维手工特征，w=0.4）
→ 裁剪
```

其中 Ridge 在 OOF 上贡献很大（TCN 8.78% → 7.37%，6 seed；3 seed 下 8.41% → 7.06%），但它是**独立于 TCN 的第二个模型**：需要额外维护 57 维特征、在 log 空间拟合线性回归、再做加权融合。用户希望去掉这层，改为**提升 TCN 自身的训练目标**，即重新使用之前验证过的“增大低浓度 Loss 权重”思路。

无 Ridge 方案的流水线：

```text
TCN 配置 A（低浓度加权 p=1.0）+ 配置 B（逐气体加权 0,0.3,1.0）
各训练 3 个 seed → 6 个 checkpoint
→ 逐 seed 预测平均 → 裁剪
（没有 Ridge，没有任何额外模型）
```

两套配置共享其余设置：10 s 因果 TCN（16 宽）、6 通道 `raw_delta`、逐气体目标 `log,none,log`、逐气体输出头、边缘 VAE 扩充 4×、噪声 0.1× + VAE 合成 3×、3σ 早停。

---

## 2. 原理：低浓度 Loss 加权怎么降 3σ

### 2.1 3σ 偏大的根源

相对误差 `rel = (pred - true)/true` 在低浓度处对绝对误差极其敏感：NO2 只有 2–10 ppm、NH3 10–50 ppm，同样的绝对误差在低浓度点会产生大得多的 `rel`，这些点决定 3σ 的尾部。原训练损失是 SmoothL1 的平均，高浓度样本（H2 8000–32000）天然贡献更大的绝对误差、占据更多梯度；低浓度样本的“相对误差重要性”没有被显式表达。

### 2.2 权重定义

对每个训练样本 `i` 和气体 `g`，用训练折（当前五折的 train fold）的逐气体中位数 `m_g` 定义：

```text
w_i,g = clip( (m_g / max(y_i,g, 1e-9)) ** p_g , 0.25, 4.0 )
```

- `p_g = 0`：该气体不加权（权重恒为 1）；
- `p_g > 0`：浓度低于中位数的样本权重 >1，高于中位数 <1；`p_g` 越大，低浓度强调越强；
- 裁剪到 `[0.25, 4.0]`：避免某个极低浓度点主导整个 batch，也避免高浓度样本被过度忽略；
- 权重只用于**训练损失**，验证/测试指标始终在原始浓度口径计算；
- `p_g` 可以逐气体设置（`--low-conc-power 1.0` 或 `--low-conc-power 0,0.3,1.0`）。

加权后的损失：

```text
loss = Σ_i Σ_g w_i,g · SmoothL1(pred_i,g, target_i,g) / Σ_i Σ_g w_i,g
```

这样低浓度/小分母样本在梯度中的占比上升，模型被逼着在这些点上更准确，直接压 3σ 的尾部。由于 H2/NO2 已经在 log 目标空间训练（log 空间误差 ≈ 相对误差），加权主要补足 NH3（原空间）以及低浓度段的相对误差；对 NO2 的低浓度尾部收益最明显（见 §3 消融）。

### 2.3 为什么用两个加权强度的集成

单个加权强度在 OOF 上“顾此失彼”：

- `p=1.0`（所有气体）NO2 明显改善（10.65% → 8.11%），但 H2 变差（9.32% → 9.95%）；
- `p=0,0.3,1.0`（H2 不加权、NH3 轻加权、NO2 全加权）H2 9.62%、NO2 9.01%，但两个都贴线；
- 把两套配置的预测按 seed 平均后，H2 9.31%、NO2 8.16%，两个气体都留出余量。

两套配置共享同样结构，只是“低浓度强调程度”不同，平均相当于对加权强度做了一次平滑，避免单套权重在 H2/NO2 之间来回摇摆。

---

## 3. OOF 筛选（全部在开发集上完成）

每个候选 3 seed（20260923–25），预测先按 seed 平均，再裁剪到开发集浓度范围；**不使用 Ridge**。

| 候选 | H2 | NH3 | NO2 | 宏平均 3σ |
|---|---:|---:|---:|---:|
| 无加权 TCN（对照） | 9.32 | 4.41 | 10.65 | 8.13 |
| p=0.5（所有气体） | 9.89 | 5.49 | 10.55 | 8.64 |
| p=1.0（所有气体） | 9.95 | 5.24 | 8.11 | 7.77 |
| p=1.0 + 边缘扩充 8× | 9.06 | 6.13 | 9.37 | 8.18 |
| p=0.75 + 边缘扩充 8× | 10.30 | 6.50 | 10.92 | 9.24 |
| p=1.0 + noise 0.2 | 8.58 | 5.13 | 10.35 | 8.02 |
| 逐气体 p=0,0.3,1.0 | 9.62 | 5.48 | 9.01 | 8.04 |
| **最终：p=1.0 与 p=0,0.3,1.0 两配置 × 3 seed 平均** | **9.31** | **4.95** | **8.16** | **7.47** |

观察：

- 单配置里 p=1.0 和 p=0,0.3,1.0 都能做到“逐气体全部 <10%”，但 H2 或 NO2 只剩 0.1–0.4 pp 余量；
- 两配置平均后余量扩大到 H2 0.69 pp、NO2 1.84 pp，因此最终选它；
- 边缘扩充 8×、noise 0.2 等变体没有带来稳定改善，未采用。

原始筛选数字保存在 `artifacts/3sigma_noridge/ablation/low_conc_screening.json`。

---

## 4. 最终结果

### 4.1 OOF（50 个开发实验，3 seed，裁剪，无 Ridge）

| 方法 | H2 | NH3 | NO2 | 宏平均 3σ | MAPE |
|---|---:|---:|---:|---:|---:|
| 无 Ridge 方案（p=1.0 + p=0,0.3,1.0 平均） | 9.31 | 4.95 | 8.16 | **7.47%** | 1.79 |
| 对照：TCN + Ridge（旧精简方案，代码已移除） | 7.63 | 4.28 | 9.28 | 7.06% | 1.70 |

OOF 变差 0.41 pp，逐气体仍全部 <10%；NH3 4.95% 仍达到 5% 目标，H2/NO2 未到 5%。

### 4.2 锁定测试集（12 个实验，第 7 次读取）

| 方法 | H2 | NH3 | NO2 | 宏平均 3σ | macro-MAPE |
|---|---:|---:|---:|---:|---:|
| 旧方案（none+加权，3 seed） | 10.75 | 6.65 | 16.14 | 11.18 | 2.64 |
| v4 混合（第 4 次） | 11.00 | 3.71 | 10.04 | 8.25 | 1.99 |
| v5 混合（第 5 次） | 10.60 | 3.80 | 10.10 | 8.17 | 1.88 |
| TCN + Ridge（第 6 次） | 9.86 | 2.72 | 8.96 | 7.18 | 1.67 |
| **无 Ridge（第 7 次）** | **6.46** | **5.71** | **9.16** | **7.11** | 1.67 |

- **逐气体 3σ 全部 <10%**；宏平均 7.11%，与 Ridge 方案基本持平（略好 0.07 pp）；
- 对比 Ridge 方案：H2 更低（6.46 vs 9.86），NH3、NO2 略高（5.71 vs 2.72；9.16 vs 8.96），属于“性能略微下降/重分配”；
- 5% 目标只有 NH3（5.71% 未达到，Ridge 方案 2.72% 达到）；H2 6.46%、NO2 9.16% 距 5% 仍远；
- 单 seed 测试宏 3σ 分别为 9.21 / 11.56 / 8.76，**只有 3 seed 平均后才全部过线**，与精简方案一致依赖 seed 集成；
- 12 点测试集中 NO2 9.16% 距门槛只有 0.84 pp，H2 OOF 9.31% 也贴线，结论对抽样敏感。

### 4.3 62 点合并口径（50 OOF + 12 测试）

| 方案 | H2 | NH3 | NO2 | 合并（186 值） |
|---|---:|---:|---:|---:|
| 旧方案（`final_vae_hybrid`） | 18.47 | 12.73 | 20.37 | 17.49 |
| TCN + Ridge（精简方案） | 8.08 | 4.01 | 9.22 | 7.45 |
| **无 Ridge 方案** | **8.86** | **5.05** | **8.59** | **7.69** |

![无 Ridge 方案 62 点误差散点](../../artifacts/3sigma_noridge/figs/fig_noridge_all62_error_sigma_by_gas.png)

![旧方案 vs 无 Ridge 方案 62 点对比](../../artifacts/3sigma_noridge/figs/fig_noridge_vs_baseline_all62.png)

横轴 0–49 为 OOF、50–61 为锁定测试（浅黄底色）；水平线为均值 ±1σ/±2σ/±3σ。相比旧方案，H2/NO2 离散带显著收窄；NO2 测试段仍有 2 个点接近 +10%，是最紧的部分。

---

## 5. 实现

### 5.1 新增的训练参数

| 参数 | 位置 | 说明 |
|---|---|---|
| `--low-conc-power` | `tcn_union/training/gan_train.py` CLI | 单值或 `H2,NH3,NO2` 三值，如 `1.0`、`0,0.3,1.0`；0 表示关闭 |
| `_low_conc_powers()` | `tcn_union/training/train.py` | 解析/校验逐气体加权强度 |
| `_low_conc_weights()` | `tcn_union/training/train.py` | 计算 `[N,3]` 权重 `clip((median/y)^p, 0.25, 4)` |
| 加权 SmoothL1 | `tcn_union/training/train.py::_train_network` | `loss = Σ w·SmoothL1 / Σ w`；辅助头（如启用）同样加权 |
| selection 记录 | `gan_train.py` | `selection.json` 记录 `low_conc_power`，便于复现 |

权重只在训练折的标签上计算（无泄漏），验证/测试始终原始浓度口径。`p=0`（默认）时数据集仍为二元组、损失仍为 `mean`，与旧实现逐位一致。

### 5.2 最终候选

两套配置（其余参数完全相同）：

```text
配置 A（p=1.0）：--low-conc-power 1.0
配置 B（p=0,0.3,1.0）：--low-conc-power 0,0.3,1.0
共同：--target-transform log,none,log --per-gas-head --edge-copies 4
      --modes noise_gan --generator vae --vae-components 6 --vae-beta 0.1
      --noise-scale 0.1 --synthetic-ratio 3.0 --validation-metric three_sigma
```

- 每个配置 3 个 seed（20260923–25），共 **6 个 TCN checkpoint**；
- 推理：逐 seed 预测平均 → 裁剪到开发集浓度范围；**不使用 Ridge**；
- 评估：`python -m tcn_union.evaluation.ensemble_evaluate --clip-to-development-range`（TCN 集平均 + 裁剪；模块已不含 Ridge 代码）。

### 5.3 目录结构与代码清理

```text
artifacts/3sigma_noridge/
├── oof/seed_2026092{3,4,5}/noise_gan/oof.csv   # 两配置平均后的逐 seed OOF
├── final/p10/noise_gan/10s/seed_*.pt           # 配置 A 的 3 个 checkpoint
├── final/q/noise_gan/10s/seed_*.pt             # 配置 B 的 3 个 checkpoint
├── evaluation/{evaluation.json,test_predictions.csv,test_metrics.csv}
├── all62_predictions.csv / all62_error_stats.csv
├── figs/{fig_noridge_all62_error_sigma_by_gas,fig_noridge_vs_baseline_all62}.{png,svg}
├── ablation/low_conc_screening.json            # OOF 筛选原始数字
└── scripts/{build_oof.py,plot_all62.py}        # 无 Ridge 方案的复现脚本
```

本次清理删除的旧代码/产物（不再保留兼容）：

- 代码：`tcn_union/evaluation/hybrid_evaluate.py`（Ridge 混合评估）、`report_three_sigma.py`、`plot_variance_comparison.py`；新增无 Ridge 的 `tcn_union/evaluation/ensemble_evaluate.py`；
- 脚本：旧 62 点/方法对比脚本、旧 `artifacts/3sigma/` 下的分析脚本；
- 产物：`artifacts/3sigma/`（TCN+Ridge 迭代）、`artifacts/3sigma_diagnosis/`、`artifacts/variance_search/`；
- 文档：v1–v5 的旧 3σ 记录与诊断文档（历史保留在 git）；
- 测试：`tests/test_hybrid_evaluate.py` 由无 Ridge 的 `tests/test_ensemble_evaluate.py` 取代，全量测试 **67 个通过**。

---

## 6. 测试集读取与可信度

| 版本 | 组成 | 测试逐气体（H2/NH3/NO2） | 宏 3σ |
|---|---|---|---:|
| 旧方案 | none+加权，3 seed | 10.75 / 6.65 / 16.14 | 11.18 |
| v1–v5 | 多配置/多结构迭代（5 次读取，旧文档/产物已清理） | — | 9.74 → 8.17 |
| TCN + Ridge（精简方案） | 单配置 × 3 seed + Ridge + 裁剪（第 6 次） | 9.86 / 2.72 / 8.96 | 7.18 |
| **无 Ridge（本方案）** | 2 配置 × 3 seed + 裁剪（第 7 次） | **6.46 / 5.71 / 9.16** | **7.11** |

**诚实说明**：

- 无 Ridge 候选的配置、低浓度加权强度、双配置集成方式、seed 数、裁剪全部在 OOF 上确定，候选在测试评估前冻结；
- 第 7 次读取是本次任务下的一次性最终评估，没有根据测试结果再调参；若继续用该测试集迭代，结果将失去泛化意义；
- 单 seed 测试不过线（9.21 / 11.56 / 8.76），结论依赖 3 seed 集成；
- 测试集累计读取 7 次，`7.11%` 应视为“该 12 点测试集上的最终结果”，更可信的验证需要新的独立锁定集。

---

## 7. 复现

```bash
export UV_NO_SYNC=1

# 1) 配置 A（p=1.0）五折 OOF + 3 seed 最终模型
for SEED in 20260923 20260924 20260925; do
  uv run --no-sync python -m tcn_union.training.gan_train \
    --manifest artifacts/splits/manifest.csv \
    --output-dir artifacts/3sigma_noridge/oof_raw/p10/seed_${SEED} \
    --device cuda --seed ${SEED} --oof-only \
    --modes noise_gan --generator vae --vae-components 6 --vae-beta 0.1 \
    --noise-scale 0.1 --synthetic-ratio 3.0 --edge-copies 4 \
    --target-transform log,none,log --per-gas-head \
    --low-conc-power 1.0 --validation-metric three_sigma
done

# 2) 配置 B（p=0,0.3,1.0）同样跑 3 个 seed，输出到 oof_raw/q
#    把上面循环里的 --output-dir 改为 .../oof_raw/q/seed_${SEED}，
#    并把 --low-conc-power 改为 0,0.3,1.0

# 3) 两配置各训练 3 seed 最终模型（分别输出到 final/p10、final/q）
uv run --no-sync python -m tcn_union.training.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/3sigma_noridge/final/p10 \
  --device cuda --seed 20260923 \
  --final-seeds 20260923 20260924 20260925 \
  --modes noise_gan --generator vae --vae-components 6 --vae-beta 0.1 \
  --noise-scale 0.1 --synthetic-ratio 3.0 --edge-copies 4 \
  --target-transform log,none,log --per-gas-head \
  --low-conc-power 1.0 --validation-metric three_sigma

# 4) 逐 seed 的 OOF 两配置平均（得到 artifacts/3sigma_noridge/oof）
uv run --no-sync python artifacts/3sigma_noridge/scripts/build_oof.py

# 5) 锁定测试评估（第 7 次读取；已有 evaluation.json 时不要重复跑）
uv run --no-sync python -m tcn_union.evaluation.ensemble_evaluate \
  --manifest artifacts/splits/manifest.csv \
  --model-dirs artifacts/3sigma_noridge/final/p10/noise_gan \
               artifacts/3sigma_noridge/final/q/noise_gan \
  --output-dir artifacts/3sigma_noridge/evaluation \
  --device cuda --windows 10 \
  --seeds 20260923 20260924 20260925 \
  --clip-to-development-range

# 6) 62 点散点图 + 统计（PNG+SVG）
uv run --no-sync python artifacts/3sigma_noridge/scripts/plot_all62.py
```

---

## 8. 局限与后续

1. **测试集已读 7 次**：无 Ridge 结果只能作为该 12 点测试集上的最终数字，不能据此继续迭代；
2. **NO2 9.16% 与 OOF H2 9.31% 贴线**：换一批测试点可能重新超 10%；需要新的独立锁定集复验；
3. **依赖 3 seed 集成**：单 seed 不过线，部署需保留全部 6 个 checkpoint；
4. **双配置集成的解释成本**：虽然去掉了 Ridge，但仍需解释“两个损失加权强度”；若想单模型，可退回 `p=1.0` 单配置（OOF H2 9.95 / NO2 8.11 也全部 <10%，但余量更小）；
5. **5% 目标未达**：NH3 5.71%（Ridge 方案 2.72% 更好），H2/NO2 仍远高于 5%；
6. **数据侧仍是根本瓶颈**：要更稳的 <10%（尤其 H2/NO2），需要补 H2=32000、NO2 低浓度与稳定难例的真实重复实验，或建立新的锁定集。

---

## 9. 一句话总结

用**低浓度损失加权**（`(median/y)^p` 裁剪到 0.25–4，逐气体可配）替代 Ridge，并把两种加权强度（`p=1.0` 与 `0,0.3,1.0`）× 3 seed 的 TCN 预测平均后裁剪：OOF 逐气体 **9.31 / 4.95 / 8.16**（宏 7.47%），测试逐气体 **6.46 / 5.71 / 9.16**（宏 7.11%），62 点合并 **8.86 / 5.05 / 8.59**；不再需要 Ridge，指标仍在阈值内。
