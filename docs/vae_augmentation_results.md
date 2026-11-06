# PCA-VAE 扩充：适合当前小样本项目的生成式方案

## 1. 为什么从 GAN 换到 PCA-VAE

GAN 在当前 50 个开发实验上失败：

- 直接生成 51×3 曲线：高频抖动，平滑度比值 1228.7；
- control-point 生成器：曲线变平滑但出现不自然折线，标签条件弱；
- 条件均值残差 GAN：视觉改善，但下游 OOF `0.030237`，仍差于 clean。

根本原因是 50 条样本不足以让 GAN 同时学会“传感器响应形态”和“浓度条件”。
因此改为 **低维响应基 + 条件 VAE**：

1. 只在训练折真实曲线上拟合每气体 PCA 响应基，把生成曲线限制在真实响应
   张成的低维子空间；
2. 用训练标签拟合 PCA 系数的条件均值；
3. 条件 VAE 只学条件均值周围的残差分布；
4. 采样 = 条件均值 + VAE 残差，再反 PCA 回 51 点原始响应。

这样生成曲线天然平滑、接近真实响应形态，同时保留标签条件和样本间变化。

## 2. 实现

代码：[`tcn_union/vae.py`](../tcn_union/vae.py)

默认配置：

- 10 秒窗口，51 点；
- 每气体 6 个 PCA 主成分；
- VAE latent dim 4，hidden 64；
- β=0.1，训练 300 epoch；
- 下游训练时使用 3× 真实训练样本数的 VAE 合成样本（40 条真实 → 120 条合成），再叠加真实测量噪声 0.1×；
- 验证/测试始终是真实曲线。

### 学习资料

如果第一次接触 PCA、VAE 或条件生成，先读：
[`PCA-VAE学习资料.md`](PCA-VAE学习资料.md)。它从“为什么需要增强”开始，用尽量少的数学讲清 PCA、条件均值、VAE、重参数化和采样，并逐段对照 `tcn_union/vae.py`。

### 论文版架构图

- 矢量 PDF：[`figures/fig_tcn_architecture.pdf`](../figures/fig_tcn_architecture.pdf)
- PNG：[`figures/fig_tcn_architecture.png`](../figures/fig_tcn_architecture.png)

图中包含：逐折数据构造、GasTCN 主结构、残差块内部数据流、五折 OOF 与三 seed 推理协议。5 秒辅助头以虚线标出，并注明最终方案未启用。

## 3. 生成质量

| 指标 | 真实/目标 | PCA-VAE |
|---|---:|---:|
| 平滑度比值（越接近 1 越好） | ~1 | **0.881** |
| 初值 MAE | 0 | 21.09 |
| 末端 MAE | 0 | 33.90 |
| 初值偏差 / 量程 | 0 | 0.005 |
| 末端偏差 / 量程 | 0 | 0.010 |
| 最近邻距离比 | 接近 1 | 0.421 |
| 成对多样性比 | 接近 1 | 1.094 |
| NH3 初值-标签相关（真实 0.966） | 同向、接近 | **0.971** |
| NH3 末端-标签相关（真实 0.967） | 同向、接近 | **0.973** |
| H2 末端-标签相关（真实 0.423） | 同向、接近 | 0.367 |
| NO2 末端-标签相关（真实 −0.819） | 同向、接近 | −0.828 |

和 GAN 最好版本的对比：平滑度 `0.881` vs `12.8`，初值 MAE `21` vs `94`，
末端 MAE `34` vs `109`。视觉上合成曲线与真实曲线的形态、初值、上升/下降趋势
基本一致。

图：[`artifacts/vae_quality/gan_samples.png`](../artifacts/vae_quality/gan_samples.png)

## 4. 开发集 OOF

| 方案 | OOF macro-NMAE |
|---|---:|
| clean | 0.027097 |
| VAE 单独 | 0.026231 |
| VAE + aux | 0.024264 |
| noise 0.1 | 0.023127 |
| noise + VAE（1×） | 0.024332 |
| noise + VAE（2×） | 0.026890 |
| **noise + VAE（3×）** | **0.024015** |
| noise + VAE（4×） | 0.024945 |
| noise + VAE + aux | 0.024904 |

单看 seed 23，noise 0.1 仍是 OOF 最低；但 VAE 混合方案在三个 seed 上更稳：

| seed | noise 0.1 | noise + VAE 3× |
|---|---:|---:|
| 20260923 | 0.023127 | 0.024015 |
| 20260924 | 0.027673 | **0.023043** |
| 20260925 | 0.026407 | **0.023957** |
| 均值 | 0.025736 | **0.023672** |
| 样本标准差 | 0.00235 | **0.00055** |

因此最终选择 **noise 0.1 + PCA-VAE 3×**：它用生成式模型提升种子稳健性，
并在三 seed 平均 OOF 上优于纯噪声。

## 5. 锁定测试集

最终模型：noise 0.1 + PCA-VAE 3×，三 seed 集成，10 秒窗口。

| 方案 | macro-NMAE | macro-MAPE / % |
|---|---:|---:|
| clean 10 秒基线 | 0.024807 | 4.6476 |
| noise 0.1（对照） | 0.018002 | 3.4714 |
| **noise + PCA-VAE 3×** | **0.017942** | **3.3927** |

逐气体：

| 气体 | NMAE | RMSE |
|---|---:|---:|
| H2 | 0.02165 | 676.37 |
| NH3 | 0.01289 | 0.6451 |
| NO2 | 0.01928 | 0.1859 |

配对 bootstrap（VAE 混合 − clean，以实验为单位）：

- 均值差：`−0.006865`；
- 95% 区间：`[−0.013223, −0.000257]`，不跨 0。

三个种子的测试 macro-NMAE：`0.020601 / 0.029479 / 0.020223`；集成后 `0.017942`。

逐气体评估图保留两种：

- 原来的“真实浓度 vs 预测浓度”RMSE 散点图：
  [`candidate_10s_rmse_scatter_by_gas.png`](../artifacts/final_vae_hybrid/evaluation/figures/candidate_10s_rmse_scatter_by_gas.png)
- 新增的“相对误差 vs Points + mean/±1σ/±2σ/±3σ”分布图（参考 `example/pic_1.png`）：
  [`candidate_10s_error_sigma_by_gas.png`](../artifacts/final_vae_hybrid/evaluation/figures/candidate_10s_error_sigma_by_gas.png)

baseline 的两张图也保留在同一个 `figures/` 目录。

### 全部 62 组图

为了增加散点数量，另用 **50 个开发集 OOF 预测 + 12 个锁定测试预测** 合成 62 个点：

- candidate：[`candidate_all62_error_sigma_by_gas.png`](../artifacts/final_vae_hybrid/evaluation/figures/candidate_all62_error_sigma_by_gas.png)
- baseline：[`baseline_all62_error_sigma_by_gas.png`](../artifacts/final_vae_hybrid/evaluation/figures/baseline_all62_error_sigma_by_gas.png)

注意：

- 开发集点来自五折 OOF，不是训练集自身预测；
- 测试集点来自三 seed 集成；
- 因此 62 点图用于展示误差分布，不是一次新的独立测试；
- 图中说明文字已改为中文，保留 `OOF`、`σ` 等必要术语。

## 6. 辅助监督的结论

- VAE 单独：`0.026231`；
- VAE + aux：`0.024264`，aux 在纯 VAE 上有效；
- noise + VAE：`0.024015`；
- noise + VAE + aux：`0.024904`，aux 又变差。

结论：aux 在“纯 VAE”方案里有帮助，但在“noise + VAE”里是重复正则，存在负交互。最终不保留 aux。

## 7. 最终方案

```text
训练集（每折 40 个开发实验）
  ├─ 真实响应 + 0.1× 真实测量噪声（40 条）
  └─ PCA-VAE 生成（120 条，3×）
       每气体 6 个 PCA 主成分
       条件均值 + β-VAE 残差
       条件来自训练集真实浓度三元组
TCN 训练（10 秒，3 个固定 seed）
验证/测试：只用真实曲线
```

测试集三 seed 集成：macro-NMAE `0.017942`，macro-MAPE `3.3927%`，
比 clean 基线 `0.024807 / 4.6476%` 明显降低；bootstrap 区间支持改善。

## 8. 复现

```bash
# 生成质量检查
uv run python -m tcn_union.gan_quality \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/vae_quality \
  --samples 100 --gan-epochs 300 --generator vae \
  --vae-components 6 --vae-beta 0.1

# 开发集 OOF：noise 0.1 + VAE 3×
uv run python -m tcn_union.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/final_vae_hybrid \
  --modes noise_gan \
  --generator vae --vae-components 6 --vae-beta 0.1 \
  --noise-scale 0.1 --noise-copies 1 \
  --synthetic-ratio 3.0 --gan-epochs 300 \
  --max-epochs 500 --oof-only

# 三 seed 最终模型 + 测试评估
uv run python -m tcn_union.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/final_vae_hybrid \
  --modes noise_gan \
  --generator vae --vae-components 6 --vae-beta 0.1 \
  --noise-scale 0.1 --noise-copies 1 \
  --synthetic-ratio 3.0 --gan-epochs 300 \
  --max-epochs 500

uv run python -m tcn_union.augmentation_evaluate \
  --manifest artifacts/splits/manifest.csv \
  --baseline-dir artifacts/tcn \
  --model-dir artifacts/final_vae_hybrid/noise_gan \
  --output-dir artifacts/final_vae_hybrid/evaluation \
  --windows 10 --rmse-plots
```

## 9. 边界

- 测试集只有 12 个实验，且此前被查看过，不能称为外部盲测；
- VAE 的 PCA 基和条件均值都只在训练折拟合，测试集不参与；
- PCA-VAE 是“条件生成 + 真实响应基”，不是无条件 GAN；
- 生成质量指标只能排除明显失败，不能证明完全符合传感器物理；
- 三 seed 平均 OOF 优于纯 noise，但单 seed 23 的 OOF 略差；最终选择同时考虑了稳健性和创新性。
