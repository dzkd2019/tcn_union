# GAN 扩充 + 辅助监督再评估：实施方案

## 1. 背景与决定

老师指出 cutout 违背传感器器件的物理规律。因此新分支 `feature/gan-augmentation`：

- 从 `dev` 切出，**删除 cutout 训练代码、cutout 搜索脚本和 cutout 测试**；
- 保留通用的 TCN 训练、五折 OOF、锁定测试评估；
- 用条件 GAN 在训练折内部合成 10 秒响应，替代 cutout；
- 重新评估 5 秒辅助监督在“GAN 方案”中是否仍有收益；
- 评估指标增加逐气体 RMSE 散点图。

> 这一页只给方案和验收标准，不预设 GAN 一定有效。GAN 在小样本传感器数据上很可能失败，必须先过物理合理性和下游 OOF 两道关。

## 2. 新分支的代码结构

| 文件 | 作用 |
|---|---|
| `tcn_union/gan.py` | 条件 WGAN-GP：`(浓度标签, 噪声) -> 10 秒三路响应`，只允许用训练折拟合 |
| `tcn_union/gan_quality.py` | GAN 生成质量快速检查：平滑度、初值/末端、最近邻、多样性、真实/合成曲线图 |
| `tcn_union/gan_train.py` | `clean / aux / gan / gan_aux` 四种模式的五折 OOF 与最终重训 |
| `tcn_union/train.py` | 通用 TCN 训练循环，保留辅助监督和可选 EMA，已删除 cutout |
| `tcn_union/augmentation_evaluate.py` | 冻结后评估，支持只评估 10 秒窗口 |
| `tcn_union/plot_rmse_scatter.py` | 逐气体真实值-预测值散点图，标注 RMSE/MAE |

### GAN 的物理约束

GAN 本身不自动满足传感器物理。当前实现加入：

- **平滑惩罚**：惩罚二阶差分，减少高频锯齿；
- **范围惩罚**：把生成值拉回训练集逐通道观测范围；
- **条件约束**：只从训练集的真实浓度三元组采样条件，不凭空插值出未验证的浓度组合；
- **折内拟合**：每折只用 40 个训练实验拟合 GAN，验证折不参与；最终模型只在 50 个开发实验上拟合 GAN，测试集不参与。

这些约束只能减少明显不合理的生成曲线，**不能证明生成样本符合传感器物理**。因此还需要第 4 节的生成质量检查。

## 3. 辅助监督的再评估

### 3.1 四种模式

| 模式 | 真实数据 | GAN 合成 | 5 秒辅助头 | 用途 |
|---|---|---|---|---|
| `clean` | 是 | 否 | 否 | 新分支基线 |
| `aux` | 是 | 否 | 是 | 单独看 aux 在无 GAN 时的作用 |
| `gan` | 是 | 是 | 否 | 单独看 GAN 的作用 |
| `gan_aux` | 是 | 是 | 是 | 看 GAN 与 aux 是否有交互收益 |

### 3.2 训练协议

- 固定 5 折，每折 40 训 / 10 验，实验级划分不变；
- 每折的 GAN 只在该折 40 个训练实验上拟合；
- TCN 训练集 = 40 条真实 + `synthetic_ratio × 40` 条合成，默认比例 1.0；
- 验证集始终是 10 条真实曲线；
- 早停只看验证 macro-NMAE；
- 最终轮数取 5 折最佳 epoch 中位数，再用全部 50 个开发实验重训 3 个 seed；
- 测试集只在所有模式和超参数冻结后评估。

### 3.3 先做单 seed 试点，再做三 seed

1. 先跑一个 seed 的 `clean/aux/gan/gan_aux` 五折 OOF，确认 GAN 不崩、指标可比较；
2. 如 GAN 模式有希望，再跑三个 seed；
3. 最后才读锁定测试集，报告三 seed 集成和逐气体 RMSE 散点图；
4. 不在测试集上回头调 GAN 比例、辅助权重或 epoch。

## 4. GAN 生成质量检查（进入下游前）

至少检查以下几项，任一明显失败就不要继续下游实验：

1. **数值与形状**：无 NaN/Inf，形状为 `[N,3,51]`，时间轴固定 0.2 秒；
2. **范围**：生成值落在训练集逐通道观测范围内；
3. **平滑度**：生成曲线的二阶差分分布不应显著大于真实曲线；
4. **初值/末端分布**：生成样本的 `t=0` 与 10 秒末端分布不能偏离真实分布太远；
5. **条件一致性**：同一浓度条件下，生成样本的均值应随浓度单调或至少同向变化；
6. **多样性**：不能所有生成样本都塌缩到同一条曲线（看样本间距离/方差）；
7. **可视化**：每个气体抽若干真实/生成曲线叠加，人工检查是否像传感器响应，而不是噪声或直线。

这些检查是“排除明显不合理”，不是“证明物理正确”。如果 GAN 只是记住了 40 条训练曲线，也要在报告中说明。

## 5. 评估指标

### 5.1 主指标

- 开发集五折 OOF macro-NMAE（越低越好）；
- 逐气体 MAE、RMSE、NMAE；
- 锁定测试集三 seed 集成的逐气体 RMSE、MAE、R²、MAPE；
- 以实验为单位的配对 bootstrap，比较 `gan_aux` 与 `gan`、`aux`、`clean`。

### 5.2 逐气体 RMSE 散点图

`plot_rmse_scatter.py` 读取 `test_predictions.csv`，为 H2/NH3/NO2 各画一张真实浓度 vs 预测浓度散点图：

- 虚线为 `y=x`；
- 面板标题标注该气体 RMSE、MAE、样本数；
- 输出 `rmse_scatter_by_gas.png` 和 `.pdf`。

示例命令见第 6 节。

## 6. 复现命令

```bash
# 0) 先看 GAN 生成质量（不跑下游）
uv run python -m tcn_union.gan_quality \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_quality \
  --samples 100 --gan-epochs 300

# 1) 单 seed 四模式五折 OOF（先试点）
uv run python -m tcn_union.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_ablation_pilot \
  --modes clean aux gan gan_aux \
  --synthetic-ratio 1.0 --gan-epochs 300 --max-epochs 500 \
  --seed 20260923 --oof-only

# 2) 三 seed 最终重训（确认方案后）
uv run python -m tcn_union.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_augmentation \
  --modes gan_aux \
  --synthetic-ratio 1.0 --gan-epochs 300 --max-epochs 500

# 3) 冻结后评估 10 秒 GAN 模型（baseline 用旧 clean checkpoint）
uv run python -m tcn_union.augmentation_evaluate \
  --manifest artifacts/splits/manifest.csv \
  --baseline-dir artifacts/tcn \
  --model-dir artifacts/gan_augmentation/gan_aux \
  --output-dir artifacts/gan_evaluation --windows 10 --rmse-plots

# 4) 也可以单独重画逐气体 RMSE 散点图
uv run python -m tcn_union.plot_rmse_scatter \
  --predictions artifacts/gan_evaluation/test_predictions.csv \
  --output-dir artifacts/gan_evaluation/figures \
  --model candidate --window 10 --seed ensemble
```

## 7. 验收标准与停止规则

**GAN 可以进入下一阶段的条件**（全部满足）：

- 生成质量检查没有明显失败；
- `gan` 的五折 OOF macro-NMAE 不差于 `clean`，或至少逐气体 RMSE 有稳定改善；
- 三个 seed 的方差没有明显变差；
- 锁定测试集上不出现明显退化。

**辅助监督保留的条件**：

- `gan_aux` 相对 `gan` 的 OOF 改善在三个 seed 上方向一致；
- 逐气体 RMSE 至少两个气体不变差；
- 配对 bootstrap 的区间不出现明显反向证据；
- 若收益只出现在单个 seed 或单个气体，报告为“不稳定”，不写成确定结论。

**停止规则**：

- GAN 生成质量检查失败 → 不进入下游，改为报告负结果；
- GAN 下游 OOF 明显变差 → 停止，不调测试集；
- aux 在 `gan_aux` 中没有稳定增益 → 新方案可以不保留 aux，或只保留为主模型的轻量正则项。

## 8. 风险与需要老师确认的点

- **样本量**：50 个开发实验训练 GAN 很容易模式坍塌；当前 GAN 是“小样本探索”，不是成熟生成模型；
- **物理正确性**：平滑/范围约束只是弱先验，不能保证生成曲线对应真实传感器动力学；
- **条件外推**：GAN 只见过训练集的浓度组合，不能可靠生成测试集中未出现的新组合；
- **评价标准**：建议老师确认“生成质量”的最低要求（曲线形态、初值、稳态、噪声水平），否则容易只凭下游指标挑 GAN；
- **是否允许合成样本进入最终训练集**：需要明确报告“真实 50 条 + 合成 N 条”，不能把合成样本说成真实实验。

## 9. 建议的下一步顺序

1. 先跑 `clean` / `aux` 的 OOF，确认回退后的基线可复现；
2. 用 1 个 seed、`synthetic_ratio=0.5/1.0` 试 GAN，先看生成曲线质量；
3. 质量过关后跑四模式单 seed OOF，比较 `gan` 与 `gan_aux`；
4. 有希望再跑三 seed 和锁定测试，生成逐气体 RMSE 散点图；
5. 把负结果也写进汇报：如果 GAN 不稳定，说明“小样本 GAN 的局限”，而不是硬说有效。
