# PCA-VAE 数据增强

PCA-VAE 是逆问题保留的三种增强方式之一。它先用训练折真实曲线拟合每气体的 PCA
响应基，再用条件 VAE 学习 PCA 系数中标签解释不了的残差，最后把采样结果反变换回
响应曲线。目标是扩充 10 s 前缀样本，同时把生成曲线限制在真实响应张成的低维子空间内。

## 1. 为什么不用直接生成原始曲线

每条训练曲线只有 51 个时间点 × 3 个通道，50 条开发实验要支撑 5 折训练。直接对
原始曲线做生成模型，生成器很容易产生不自然的形状：高频抖动、通道间不一致、
或超出训练浓度范围的响应。PCA 响应基把每条曲线压缩成少量系数，生成器只学系数，
形状由真实主方向约束。

## 2. 方法

### 2.1 每气体 PCA 响应基

对每个气体通道分别做 PCA，不把三种气体混在一起：

1. 取训练折中每条实验的 0–10 s 相对响应，得到 \(n\times51\) 矩阵；
2. 减去该通道的平均曲线 \(\mu_j\)；
3. 对中心化矩阵做 SVD，取前 \(K=6\) 个主方向 \(V_j\)；
4. 每条曲线表示为 \(z_{ij}=V_j(y_{ij}-\mu_j)\)，即 6 维 PCA 系数。

默认 `VAEConfig.components=6`；K 的累计解释方差与五折重构误差可用
`tcn_union.evaluation.analyze_pca_components` 复算。

### 2.2 标签条件均值

浓度标签标准化后，用线性回归预测 6 维 PCA 系数：

$$
\hat z_j(c)=W_j^\top[1,c_{\mathrm{std}}].
$$

这一步给出“给定浓度时该气体响应的平均形状”。它只建模标签可解释的部分。

### 2.3 条件 VAE 学残差

VAE 的输入是 PCA 系数残差 \(r_j=z_j-\hat z_j(c)\) 与标准化标签，输出是重建的
残差系数。结构为：

- 编码器：\((18+3)\to64\to64\to(\mu,\log\sigma^2)\in\mathbb R^4\)；
- 解码器：\((4+3)\to64\to64\to18\)；
- 损失：重建均方误差 \(+\beta\cdot\mathrm{KL}\)，\(\beta=0.1\)；
- 训练：Adam，学习率 \(10^{-3}\)，batch 16，300 epoch，梯度裁剪 10。

18 = 3 气体 × 6 主成分。编码器与解码器都使用 ReLU。

### 2.4 采样

采样时先从训练标签中有放回抽浓度条件（或给定指定浓度），再从标准正态抽
\(\epsilon\in\mathbb R^4\)，解码得到残差，加上标签条件均值，最后：

$$
z_j=\hat z_j(c)+\mathrm{Decoder}(\epsilon,c),\qquad
y_j=\mu_j+V_j^\top z_j.
$$

生成曲线按训练折逐通道最小/最大值裁剪，避免外推。标签不生成新组合，只从训练
标签中抽样；边缘浓度组合可显式指定，用于边缘复制。

## 3. 训练协议

当前 10 s 逆问题使用完整协议：

- 50 条开发实验，5 折实验级 OOF，锁定 12 条不参与；
- 多个 seed：20260923 / 20260924 / 20260925；
- 两套低浓度损失加权配置：`1.0` 与 `0,0.3,1.0`；
- 每个增强臂 120 条合成曲线（3×40）；
- 0.1×MAD 真实测量噪声视图；
- 边缘浓度复制 ×4；
- 6 个 checkpoint 平均，多个 seed 训练后取平均预测；
- 预测裁剪到开发集逐气体浓度范围。

PCA-VAE 只在训练折上拟合，验证折不参与 PCA、条件均值或 VAE 训练。

## 4. 结果

10 s、50 条开发集 OOF、开发集量程、使用多个 seed 训练后取平均预测：

| 指标 | 宏平均 | H₂ | NH₃ | NO₂ |
|---|---:|---:|---:|---:|
| 3σ / % | **7.472** | 9.313 | 4.948 | **8.156** |
| MAPE / % | **1.791** | 2.173 | 1.152 | **2.046** |

仅真实基线为宏 3σ 10.791%、MAPE 2.774%。PCA-VAE 在三气体上都降低了
3σ 与 MAPE；它的 NO₂ 3σ/MAPE 最好，NH₃ 与组合使用相同，宏 MAPE 1.791% 略高于组合使用的 1.739%。

组合使用的绝对结果见三方法对比表；三种方法都在同一开发集上给出 OOF 预测，
按绝对指标直接比较。

完整三方法对比见
[augmentation_comparison_20261127.md](../sensor_array/docs/augmentation_comparison_20261127.md)。

## 5. 复现

从仓库根目录运行，使用项目虚拟环境：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae forward_prob_joint10 \
  --output artifacts/augmentation/pure_forward/joint10
```

输出在 `artifacts/augmentation/pure_forward/joint10/`；只要 PCA-VAE 与仅真实时，
把 `--arms` 改为 `real vae`，`--output` 改为 `artifacts/augmentation/protocols/full`。
三方法汇总：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.compare_augmentation
```

## 6. 限制

- 50 条训练实验、每个浓度组合 1 条；PCA 基与 VAE 都只能覆盖训练分布附近。
- K=6 是默认配置；更大的 K 保留更多形状细节，但在小样本上更容易过拟合。
- 条件均值是线性回归，不能表达强非线性浓度交互；残差 VAE 只补偿剩余变化。
- 生成曲线按训练折范围裁剪，不会外推到未见浓度组合。
- 所有结论只适用于当前 62 条实验、三种气体与 10 s 前缀。