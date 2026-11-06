# PCA-VAE 数据增强学习资料

> 面向：深度学习新手、数学基础一般、需要读懂本项目 `tcn_union/vae.py` 的读者。  
> 目标：先理解“为什么这样做”，再理解“每一步在做什么”，最后能对照代码说出每个变量的含义。

---

## 0. 先记住一句话

本项目里的 PCA-VAE 不是“让 VAE 直接生成整条传感器曲线”，而是：

> **先用真实曲线的主成分（PCA）把曲线压缩成几个系数；再用条件 VAE 学这些系数的残差分布；采样后把系数还原成曲线。**

可以把它想成：

```text
真实曲线  →  PCA 压缩成 6 个系数  →  条件均值解释标签影响
                                      ↓
                               VAE 学剩下的随机差异
                                      ↓
采样：条件均值 + VAE 残差  →  PCA 还原  →  合成曲线
```

它要解决的是：**只有 50 条真实实验，直接训练 TCN 容易过拟合；直接上 GAN 又生成不出自然的传感器曲线。**

---

## 1. 为什么需要数据增强

当前数据：

- 50 个开发实验；
- 每个实验有 H2、NH3、NO2 三路响应；
- 每个实验取 10 秒前缀，共 51 个采样点；
- 每个实验有 3 个浓度标签。

也就是训练集大约只有：

```text
50 条曲线 × 3 个气体 × 51 个时间点
```

对于深度网络来说非常少。数据增强的目标是：

- 不改变标签；
- 生成“看起来像真实传感器响应”的新曲线；
- 让模型见到更多变化，减少死记硬背。

### 为什么不用 cutout

cutout 是在曲线中间挖掉一段。它的问题：

- 传感器响应是连续物理过程，挖掉一段再用直线连接，会破坏真实动力学；
- 对气体传感器来说，这种曲线不一定对应任何真实浓度；
- 老师指出它违背器件物理规律，因此本项目已经删除。

### 为什么不用直接 GAN

GAN 直接生成 `51×3` 曲线，在 50 条样本上很难同时学会：

- 曲线要平滑；
- 曲线要符合传感器响应形态；
- 曲线要随浓度标签变化；
- 曲线之间还要有合理差异。

本项目试过 GAN，生成曲线出现高频抖动或不自然折线，因此没有采用。

### 为什么 PCA-VAE 更适合

PCA 先把生成空间限制在“真实曲线张成的低维空间”里：

- 生成曲线不会是任意形状，而是真实曲线主成分的组合；
- 天然比较平滑；
- 参数量小，50 条样本也能训练；
- 再用 VAE 学系数上的随机变化，保留样本多样性。

---

## 2. 数据长什么样

代码入口：

```python
responses, labels = _prefix_arrays(rows, config.seconds)
```

对应 [`vae.py` 第 142–154 行](../tcn_union/vae.py#L142-L154)。

对每个实验：

```python
response[mask].T
```

原始 `response` 形状是：

```text
[时间, 3] = [51, 3]
```

转置后变成：

```text
[3, 51] = [气体, 时间]
```

最后堆起来：

```text
responses: [N, 3, 51]
labels:    [N, 3]
```

其中：

- `N` 是训练实验数；
- 3 是 H2、NH3、NO2；
- 51 是 10 秒、5 Hz 的采样点数；
- `labels` 是三个浓度。

> 注意：这里不是把 51 个时间点当成 51 个独立样本，而是“一个实验 = 一条曲线 = 一个样本”。

---

## 3. PCA：把一条曲线压缩成几个系数

### 3.1 PCA 的直觉

假设每个气体有 50 条曲线，每条 51 个点。我们想用几个“基本形状”来表示它们。

PCA 做的事：

1. 先求所有曲线的平均形状；
2. 看每条曲线相对平均形状差在哪里；
3. 找到差异最大的几个方向；
4. 用这几个方向的组合来近似每条曲线。

例如，某个主成分可能表示“整体上升快慢”，另一个表示“中间有没有弯曲”。

### 3.2 代码怎么算

[`vae.py` 第 176–185 行](../tcn_union/vae.py#L176-L185)：

```python
for gas in range(3):
    flat = responses[:, gas, :]          # [N, 51]
    mean = flat.mean(axis=0)             # [51]
    centered = flat - mean[None, :]      # 去均值
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:components]              # [K, 51]
    coefficient = centered @ basis.T     # [N, K]
```

逐行解释：

- `flat`：取出某一个气体的所有曲线；
- `mean`：这个气体的平均曲线；
- `centered`：每条曲线减去平均曲线，得到“形状差异”；
- `np.linalg.svd`：奇异值分解，用来找主方向；
- `vt[:components]`：取前 `K` 个主方向，代码默认 `K=6`；
- `coefficient`：每条曲线在这 6 个方向上的坐标。

实际代码还会限制主成分数量：

```python
components = min(config.components, n - 1, length)
```

也就是 `K = min(6, 样本数-1, 51)`。每折有 40 个训练样本，所以实际 `K=6`；如果样本更少，K 会自动变小，避免 PCA 过拟合。

### 3.3 用公式表示

对某个气体，设曲线为 \(x_i\)，平均曲线为 \(\mu\)，主方向为 \(v_1,\dots,v_K\)。

去均值：

\[
\tilde x_i = x_i-\mu
\]

投影得到系数：

\[
c_{i,k}=\tilde x_i\cdot v_k
\]

还原：

\[
\hat x_i=\mu+\sum_{k=1}^{K} c_{i,k}v_k
\]

如果 \(K=51\)，可以完全还原；如果 \(K=6\)，就是“用 6 个数字概括一条 51 点曲线”。

### 3.4 三个气体分别做 PCA

代码里 `for gas in range(3)` 表示：

- H2 有自己的一套均值和主方向；
- NH3 有自己的一套；
- NO2 有自己的一套。

因为三个气体的响应幅度、变化趋势差别很大，混在一起做 PCA 不合适。

所以：

```text
coefficients: [N, 3, K]
means:        [3, 51]
components:   [3, K, 51]
```

---

## 4. 条件均值：先学“标签决定的主要趋势”

PCA 系数还是会随浓度变化。代码先做一个简单的线性模型：

\[
\text{系数} \approx a + b_1\cdot \text{H2标签} + b_2\cdot \text{NH3标签} + b_3\cdot \text{NO2标签}
\]

这不是深度网络，只是一个线性回归，用来抓住“浓度 → 曲线主趋势”的关系。

### 4.1 标签标准化

[`vae.py` 第 190–193 行](../tcn_union/vae.py#L190-L193)：

```python
label_mean = labels.mean(axis=0)
label_std = labels.std(axis=0)
labels_z = (labels - label_mean) / label_std
```

为什么要标准化？

- H2 浓度是 8000–32000；
- NH3 是 10–50；
- NO2 是 2–10。

如果不标准化，线性回归会被 H2 的大数值主导。标准化后三个标签尺度接近。

### 4.2 构造设计矩阵

[`vae.py` 第 197–199 行](../tcn_union/vae.py#L197-L199)：

```python
design = np.concatenate((np.ones((n, 1)), labels_z), axis=1)
mean_coef = np.linalg.lstsq(design, coefficient_z.reshape(n, -1), rcond=None)[0]
base = (design @ mean_coef).reshape(n, 3, components)
```

`design` 的形状是 `[N, 4]`：

```text
[1, H2_z, NH3_z, NO2_z]
```

`mean_coef` 是线性回归系数，形状 `[4, 3K]`。

`base` 就是“只由标签预测出来的 PCA 系数”，形状 `[N, 3, K]`。

### 4.3 残差

[`vae.py` 第 200 行](../tcn_union/vae.py#L200)：

```python
residual = coefficient_z - base
```

含义：

```text
真实系数 = 标签能解释的部分（base） + 标签解释不了的部分（residual）
```

残差包括：

- 实验之间的个体差异；
- 传感器噪声；
- 未观测到的环境因素；
- 模型没学到的非线性。

VAE 只学这个残差，而不是重新学生成整条曲线。

---

## 5. VAE：学习残差的分布

### 5.1 VAE 是什么

VAE（变分自编码器）由两部分组成：

- **编码器 Encoder**：把输入压缩成一个隐变量分布；
- **解码器 Decoder**：从隐变量还原输入。

普通自编码器只做“压缩 → 还原”。VAE 还要求隐变量服从一个简单的分布（通常是标准正态分布），这样我们才能从分布里采样，生成新样本。

### 5.2 编码器

[`vae.py` 第 52–66 行](../tcn_union/vae.py#L52-L66)：

```python
class _Encoder(nn.Module):
    ...
    self.mu = nn.Linear(hidden_dim, latent_dim)
    self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, coefficient, condition):
        hidden = self.net(torch.cat((coefficient, condition), dim=1))
        return self.mu(hidden), self.logvar(hidden)
```

输入：

- `coefficient`：残差系数，长度 `3K=18`；
- `condition`：三个标准化标签，长度 3。

输出：

- `mu`：隐变量均值；
- `logvar`：隐变量对数方差。

为什么输出 `logvar` 而不是直接输出标准差？

- 方差必须大于 0；
- 网络输出 `logvar` 可以取任意实数；
- 用 `exp(0.5 * logvar)` 得到标准差，天然为正。

### 5.3 重参数化技巧

[`vae.py` 第 223–226 行](../tcn_union/vae.py#L223-L226)：

```python
mu, logvar = encoder(batch_residual, batch_label)
std = torch.exp(0.5 * logvar)
noise = torch.randn_like(std)
latent = mu + std * noise
```

公式：

\[
z=\mu+\sigma\odot\epsilon,\qquad \epsilon\sim\mathcal N(0,I)
\]

为什么要这样写？

- 如果直接“从正态分布采样”，采样操作不可导，反向传播会断；
- 写成 `mu + std * noise` 后，随机性在 `noise` 里，`mu` 和 `std` 仍然可导；
- 这就是 VAE 的“重参数化技巧”。

### 5.4 解码器

[`vae.py` 第 69–81 行](../tcn_union/vae.py#L69-L81)：

```python
class _Decoder(nn.Module):
    ...
    def forward(self, latent, condition):
        return self.net(torch.cat((latent, condition), dim=1))
```

输入：

- `latent`：隐变量 `z`；
- `condition`：三个标准化标签。

输出：

- 残差系数，长度 `3K=18`。

### 5.5 损失函数

[`vae.py` 第 227–230 行](../tcn_union/vae.py#L227-L230)：

```python
reconstructed = decoder(latent, batch_label)
reconstruction = ((reconstructed - batch_residual) ** 2).mean()
kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
loss = reconstruction + config.beta * kl
```

两个损失：

1. **重构损失**：解码器能不能还原残差；
2. **KL 散度**：编码器输出的分布不要离标准正态太远。

KL 公式：

\[
D_{KL}
=\frac{1}{2}\sum_j
\left(
\mu_j^2+\sigma_j^2-1-\log\sigma_j^2
\right)
\]

代码里把 `logvar` 直接代入，所以写成了：

```text
-0.5 * (1 + logvar - mu^2 - exp(logvar))
```

`beta` 控制两者的平衡：

- `beta` 大：更强调隐变量接近标准正态，生成更平滑，但重构可能变差；
- `beta` 小：更强调重构，细节更多，但隐空间可能不够规整。

当前配置 `beta=0.1`。

---

## 6. 采样：怎么生成一条新曲线

[`vae.py` 第 113–136 行](../tcn_union/vae.py#L113-L136)：

```python
latent = torch.randn(count, latent_dim)
condition = labels_z[random_indices]
residual = decoder(latent, condition)
base = design @ mean_coef
coefficients = (base + residual) * coefficient_std
responses = means + coefficients @ components
responses = clip(responses, output_min, output_max)
```

完整步骤：

1. 从标准正态分布采样 `latent`；
2. 从训练集真实标签里随机取一个条件；
3. 解码器生成残差；
4. 用标签算条件均值 `base`；
5. 相加得到标准化系数；
6. 乘回 `coefficient_std`；
7. 用 PCA 均值和主方向还原曲线；
8. 裁剪到训练集观测范围；
9. 得到合成曲线和对应标签。

### 为什么条件从训练标签里取？

代码：

```python
indices = torch.randint(0, len(labels_z), (count,))
condition = labels_z[indices]
```

原因：

- 只生成训练集中真实出现过的浓度组合；
- 不凭空插值出未验证的浓度；
- 合成样本的标签是真实的浓度三元组。

这也意味着：它不能可靠外推到训练集没有的浓度组合。

---

## 7. 整体训练流程

```text
输入：某一折的 40 个训练实验
  1. 取 10 秒响应，整理成 [40, 3, 51]
  2. 每个气体分别做 PCA，得到 [40, 3, 6] 系数
  3. 标签标准化，线性回归得到条件均值 base
  4. 残差 = 系数 - base
  5. 训练 VAE：编码残差 + 标签 → 隐变量；解码 → 残差
  6. 采样 120 条：条件均值 + VAE 残差 → 反 PCA → 曲线
  7. 生成 TCN 特征，与 40 条真实 + 40 条噪声视图一起训练
```

验证/测试：

- 只用真实曲线；
- 不参与 PCA、条件均值、VAE 的训练。

---

## 8. 代码逐段对照

| 代码位置 | 作用 | 关键变量 |
|---|---|---|
| [`vae.py` 27–49 行](../tcn_union/vae.py#L27-L49) | 配置 | `components=6`, `latent_dim=4`, `beta=0.1` |
| [`vae.py` 52–66 行](../tcn_union/vae.py#L52-L66) | 编码器 | `mu`, `logvar` |
| [`vae.py` 69–81 行](../tcn_union/vae.py#L69-L81) | 解码器 | `latent`, `condition` |
| [`vae.py` 142–154 行](../tcn_union/vae.py#L142-L154) | 读取前缀 | `responses [N,3,51]`, `labels [N,3]` |
| [`vae.py` 176–188 行](../tcn_union/vae.py#L176-L188) | PCA | `means`, `components`, `coefficients` |
| [`vae.py` 190–200 行](../tcn_union/vae.py#L190-L200) | 条件均值 + 残差 | `labels_z`, `mean_coef`, `base`, `residual` |
| [`vae.py` 202–237 行](../tcn_union/vae.py#L202-L237) | 训练 VAE | `reconstruction`, `kl`, `loss` |
| [`vae.py` 239–254 行](../tcn_union/vae.py#L239-L254) | 保存统计量 | `output_min/max`, `means`, `components` |
| [`vae.py` 113–136 行](../tcn_union/vae.py#L113-L136) | 采样还原 | `base + residual`, 反 PCA |
| [`vae.py` 257–272 行](../tcn_union/vae.py#L257-L272) | 生成 TCN 特征 | `make_features` |

---

## 9. 一个很小的数值例子

假设只考虑一个气体，只有 3 条曲线，每个曲线 3 个时间点：

```text
曲线 1: [1, 2, 3]
曲线 2: [2, 3, 4]
曲线 3: [3, 4, 5]
```

平均曲线：

```text
[2, 3, 4]
```

去均值：

```text
曲线 1: [-1, -1, -1]
曲线 2: [ 0,  0,  0]
曲线 3: [ 1,  1,  1]
```

这三条差异都沿着同一个方向 `[1,1,1]`，所以只需要 1 个主成分就能完全表示。

每条曲线的系数就是：

```text
曲线 1: -1
曲线 2:  0
曲线 3:  1
```

还原时：

```text
平均曲线 + 系数 × 主方向
```

VAE 要学的就是这些系数（这里是 1 维）的分布和随机变化。

真实数据当然复杂得多，但核心思想一样：**用少数几个数字表示一条曲线。**

---

## 10. 常见误解

### 误解 1：VAE 直接生成原始曲线

不是。这里 VAE 生成的是 **PCA 系数的残差**，最后还要反 PCA 才变成曲线。

### 误解 2：隐变量就是浓度

不是。浓度是条件 `condition`；隐变量 `latent` 表示“在给定浓度下，这条曲线还能怎么变化”。

### 误解 3：PCA 主成分一定有物理意义

不一定。主成分只是“变化最大的方向”，可能混合了多个物理因素。不要把它们直接解释成某个传感器机理。

### 误解 4：生成样本一定符合物理

PCA 只保证曲线在真实曲线的线性组合范围内，不能保证完全符合传感器动力学。所以还需要生成质量检查。

### 误解 5：合成越多越好

不是。合成比例过高可能引入偏差。当前用 3×（40 条真实 → 120 条合成），是通过实验选的。

### 误解 6：测试集也用了增强

没有。PCA、条件均值、VAE 都只在训练折上拟合；验证/测试始终是真实曲线。

---

## 11. 术语表

| 术语 | 通俗解释 |
|---|---|
| PCA | 主成分分析，用少数几个方向概括数据 |
| 主成分 | 数据变化最大的方向 |
| 系数 | 一条曲线在每个主方向上的坐标 |
| 重构 | 用均值和系数把曲线近似还原 |
| VAE | 变分自编码器，能学习数据分布并采样 |
| Encoder | 把输入压缩成隐变量分布 |
| Decoder | 从隐变量还原输入 |
| 隐变量 latent | 压缩后的低维表示 |
| KL 散度 | 衡量两个分布差多少 |
| 重参数化 | 让随机采样也能反向传播的技巧 |
| 条件 condition | 生成时额外给的信息，这里是三个浓度标签 |
| 残差 residual | 标签解释不了的那部分变化 |
| epoch | 把训练集完整过一遍 |
| batch | 一次送进网络的一小批样本 |
| fold | 交叉验证的一折 |
| OOF | 折外预测，模型没见过的验证样本预测 |
| seed | 随机种子，控制可复现性 |

---

> 如果准备向老师汇报，方法选择理由、幻灯片结构和三分钟口头稿见 [`PCA-VAE汇报提纲.md`](PCA-VAE汇报提纲.md)。

## 12. 建议的学习顺序

如果第一次接触，建议按这个顺序：

1. 先运行 `tcn_union/vae.py` 里的 `fit_vae`，打印每个变量的形状；
2. 只保留 `components=2`，把 PCA 系数画成二维散点，理解“曲线 → 点”；
3. 用 `TrainedVAE.sample` 生成 5 条曲线，和真实曲线画在一起；
4. 看 `reconstruction` 和 `kl` 随 epoch 的变化；
5. 再看 `tcn_union/gan_train.py` 怎么把合成特征拼进训练集；
6. 最后看 `docs/vae_augmentation_results.md` 了解为什么选 3×、0.1 噪声。

### 小练习

1. 把 `components` 从 6 改成 2，观察生成曲线是不是变“单一”；
2. 把 `beta` 从 0.1 改成 0.001，观察重构和 KL；
3. 把 `latent_dim` 从 4 改成 8，观察生成多样性；
4. 生成 10 条曲线，计算它们和真实曲线的最近邻距离，判断是否只是在复制训练样本。

> 注意：这些练习只用于理解，不要直接拿测试集调参。

---

## 13. 一句话总结

> **PCA 负责“把曲线压缩成少数系数并保证形态自然”，线性条件均值负责“让系数跟随浓度标签”，VAE 负责“学习系数上剩余的随机变化”。三者组合，才是本项目使用的 PCA-VAE 数据增强。**
