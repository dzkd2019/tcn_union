> 历史阶段性结果：本文记录“纯噪声”对照。最终方案已升级为 PCA-VAE + 噪声，见 [`vae_augmentation_results.md`](vae_augmentation_results.md)。

# 噪声 / GAN 扩充对比结果与最终选择

## 1. 协议

- 数据：固定 50 个开发实验 + 12 个锁定测试实验；
- 五折：固定实验级划分，每折 40 训 / 10 验；
- 只使用开发集做方案选择；
- 噪声只加在训练折的原始响应上，验证/测试保持真实曲线；
- 最终方案冻结后，才读取 12 个测试实验；
- 测试集评估使用三 seed 集成。

本轮比较了三类方案：

1. **真实噪声增强**：从训练折二阶差分 MAD 估计测量噪声，按倍率加高斯噪声；
2. **条件 GAN**：MLP / control-point / 端点锚定 / 条件均值残差多个版本；
3. **噪声 + GAN 组合**，以及 **5 秒辅助监督**。

## 2. 噪声强度估计

对训练折的 10 秒响应做二阶差分，用 MAD 估计高斯噪声标准差：

| 气体 | 估计噪声 σ（原始单位） |
|---|---:|
| H2 | 0.0138 |
| NH3 | 0.0905 |
| NO2 | 0.0191 |

`noise_scale=0.1` 表示只加约 0.1 倍估计噪声，即 H2/NH3/NO2 约
`0.0014 / 0.009 / 0.0019` 原始单位，远小于早期 cutout 实验里按初值比例加的
噪声（当时约 2–12 单位）。

## 3. 开发集 OOF 结果（seed 20260923）

| 方案 | OOF macro-NMAE | 相对 clean |
|---|---:|---:|
| clean | 0.027097 | — |
| noise 0.05 | 0.025054 | −7.5% |
| **noise 0.1** | **0.023127** | **−14.7%** |
| noise 0.15 | 0.024737 | −8.7% |
| noise 0.25 | 0.024892 | −8.1% |
| noise 0.5 | 0.023206 | −14.4% |
| noise 1.0 | 0.026520 | −2.1% |
| noise 2.0 | 0.026491 | −2.2% |
| noise 0.1, copies=2 | 0.024773 | −8.6% |
| aux（clean + 5 秒辅助） | 0.024954 | −7.9% |
| noise 0.1 + aux | 0.024577 | −9.3% |
| GAN（条件均值残差） | 0.030237 | +11.6% |
| noise 0.1 + GAN | 0.027831 | +2.7% |
| noise 0.1 + EMA 0.99 | 0.023946 | −11.6% |

**结论：**

- 噪声强度不是越大越好；`0.1×` 和 `0.5×` 最好，`0.1×` 略优；
- **5 秒辅助监督在 clean 上有效**（0.024954 vs 0.027097），但在 best noise 上反而变差（0.024577 vs 0.023127）——说明噪声已经提供了足够正则化，aux 是重复正则，存在负交互；
- GAN 单独和与噪声组合都明显变差，不采用；
- EMA 在噪声方案上也没有改善。

因此本轮最终选择：**noise 0.1，copies=1，不加 aux，不加 EMA**。

## 4. 噪声 0.1 的种子稳健性

| seed | OOF macro-NMAE |
|---|---:|
| 20260923 | 0.023127 |
| 20260924 | 0.027673 |
| 20260925 | 0.026407 |
| 均值 | **0.025736** |

三个 seed 的均值低于旧 clean 的三种子均值 `0.028348`；但 noise 的种子方差更大（约 `0.00235` vs clean `0.00121`），所以不能宣称“每个 seed 都稳定大幅改善”，只能说是固定划分上的平均改善。

## 5. 锁定测试集结果

最终模型：noise 0.1，三 seed 集成，10 秒窗口。

| 方案 | macro-NMAE | macro-MAPE / % |
|---|---:|---:|
| clean 10 秒基线 | 0.024807 | 4.6476 |
| **noise 0.1 三 seed 集成** | **0.018002** | **3.4714** |

逐气体：

| 气体 | NMAE | RMSE（原始单位） |
|---|---:|---:|
| H2 | 0.02228 | 695.94 |
| NH3 | 0.01075 | 0.5196 |
| NO2 | 0.02098 | 0.2159 |

配对 bootstrap（以实验为单位，noise − clean）：

- 均值差：`−0.006805`；
- 95% 区间：`[−0.011877, −0.001711]`，不跨 0。

三个种子的测试 macro-NMAE：`0.024546 / 0.025581 / 0.021651`；集成后 `0.018002`。

逐气体 RMSE 散点图：
[`artifacts/final_noise/evaluation/figures/candidate_10s_rmse_scatter_by_gas.png`](../artifacts/final_noise/evaluation/figures/candidate_10s_rmse_scatter_by_gas.png)

## 6. GAN 为什么没有采用

条件 GAN 试了：

- MLP 直接输出；
- control-point 平滑生成；
- 端点锚定；
- 条件均值 + 残差 GAN。

最好的一版（条件均值 + 残差 + 端点锚定）质量指标为：
平滑度比值约 12.8（目标接近 1），NH3 初值/末端条件相关约 0.978/0.981，
视觉上比最初版本合理，但下游 OOF 仍为 **0.030237**，比 clean 差。

因此按“质量门 + 下游 OOF”双重标准，GAN 不进入最终方案。详细质量结果见
[`gan_quality_result.md`](gan_quality_result.md)。

## 7. 结论与边界

- 最终采用 **真实测量噪声 0.1×**，删除 cutout，不保留 aux；
- 测试集三 seed 集成从 clean 的 `0.024807` 降到 `0.018002`，bootstrap 区间支持改善；
- 但测试集只有 12 个实验，且此前被查看过；不能称为外部盲测或跨器件泛化；
- 噪声是“测试时不可见、只在训练时加”的增强，推理输入仍是原始曲线；
- 与旧 cutout+EMA 的 `0.016692` 相比，本方案没有更好；但 cutout 已被老师否定，本方案是在**不使用 cutout**前提下的当前最佳。

## 8. 复现

```bash
# 开发集 OOF
uv run python -m tcn_union.training.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/final_noise \
  --modes noise --noise-scale 0.1 --noise-copies 1 \
  --max-epochs 500 --oof-only

# 三 seed 最终模型
uv run python -m tcn_union.training.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/final_noise \
  --modes noise --noise-scale 0.1 --noise-copies 1 \
  --max-epochs 500

# 锁定测试集 + RMSE 散点图
uv run python -m tcn_union.evaluation.augmentation_evaluate \
  --manifest artifacts/splits/manifest.csv \
  --baseline-dir artifacts/tcn \
  --model-dir artifacts/final_noise/noise \
  --output-dir artifacts/final_noise/evaluation \
  --windows 10 --rmse-plots
```
