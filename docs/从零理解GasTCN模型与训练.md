# 从零理解 GasTCN：模型定义与训练

> 面向：第一次读这个项目、深度学习和数学基础一般的读者。  
> 目标：读完后能说清楚“输入是什么、模型怎么算、loss 怎么来的、五折和三种子是什么意思、测试集为什么不能碰”。

---

## 0. 先看整体流程

```text
一次实验 CSV
  → 取 0–10 秒三路响应
  → 构造 raw_delta 特征（6 通道 × 51 点）
  → 因果 TCN
  → 取最后一个时间位置的隐藏向量
  → 线性头
  → H2 / NH3 / NO2 三个浓度
```

一句话：

> **输入是一小段三路传感器响应曲线，输出是三种气体的浓度；模型只允许看当前时刻和过去，不能看未来。**

对应的论文版架构图：

- [fig_tcn_architecture.pdf](../figures/fig_tcn_architecture.pdf)
- [fig_tcn_architecture.png](../figures/fig_tcn_architecture.png)

---

## 1. 问题定义

### 1.1 输入和输出

| 项目 | 形状 | 含义 |
|---|---|---|
| 一次实验 | `[299, 7]` | 299 行 × 7 列 |
| 时间 | `[299]` | 0–59.6 秒，每 0.2 秒一个点 |
| 响应 | `[299, 3]` | H2、NH3、NO2 三路传感器响应 |
| 标签 | `[3]` | H2、NH3、NO2 三个浓度 |
| 10 秒特征 | `[6, 51]` | 6 通道 × 51 个时间点 |
| 5 秒特征 | `[6, 26]` | 6 通道 × 26 个时间点 |
| 模型输出 | `[3]` | 三个标准化浓度 |

### 1.2 为什么是 51 个点

采样间隔是 0.2 秒：

\[
10\text{ 秒} / 0.2 + 1 = 51
\]

加 1 是因为包含 `t=0` 和 `t=10` 两个端点。

5 秒同理：

\[
5 / 0.2 + 1 = 26
\]

### 1.3 关键约束：只能用前缀

预测 10 秒浓度时，模型只能看 `0–10` 秒；不能看 10 秒之后的响应。

这不是“建议”，而是代码硬约束：

- 输入特征只截取到目标时刻；
- 卷积只在左侧补零；
- 输出只读最后一个时间位置。

---

## 2. 数据读取与特征构造

核心文件：[`tcn_union/core/data.py`](../tcn_union/core/data.py)

### 2.1 读取一次实验

[`load_run()`](../tcn_union/core/data.py#L60-L84) 做这些检查：

- CSV 列顺序必须是：

```text
time, H2, NH3, NO2, conc1, conc2, conc3
```

- 时间必须从 0 开始；
- 时间必须严格递增；
- 间隔必须是 0.2 秒；
- 浓度必须非负；
- 同一次实验内浓度标签不能变化；
- 所有数值必须有限。

返回：

```python
time:      [299]
responses: [299, 3]
labels:    [3]
```

### 2.2 构造模型输入

[`make_features()`](../tcn_union/core/data.py#L147-L195) 做两件事：

1. 截取前缀；
2. 计算 `raw_delta` 六通道。

先取初值：

```python
baseline = prefix[0]          # [3]
```

计算相对增量：

\[
\text{delta}(t)=\frac{x(t)-x(0)}{\max(|x(0)|,10^{-8})}
\]

然后拼接：

```python
channels = np.concatenate((prefix.T, delta.T), axis=0)
```

最终：

```text
[3 原始响应; 3 相对增量] = [6, 51]
```

通道顺序：

| 通道 | 含义 |
|---|---|
| 0 | H2 原始响应 |
| 1 | NH3 原始响应 |
| 2 | NO2 原始响应 |
| 3 | H2 相对增量 |
| 4 | NH3 相对增量 |
| 5 | NO2 相对增量 |

### 2.3 为什么要 `raw_delta`

只用原始响应：

- 不同气体量级差异大；
- 模型可能只记住绝对幅度。

只用增量：

- 丢失初始响应的绝对信息；
- 对低浓度可能不够。

`raw_delta` 同时保留：

- **绝对水平**：初始响应；
- **相对变化**：响应上升/下降的形状。

---

## 3. 一维因果卷积的直觉

### 3.1 普通卷积

一维卷积用一个长度为 `k` 的窗口在时间轴上滑动：

```text
输入:  x0  x1  x2  x3  x4
窗口:  w0  w1  w2
输出:  y2 = w0*x0 + w1*x1 + w2*x2
```

如果窗口包含未来点，就会泄漏未来信息。

### 3.2 因果卷积

代码在左侧补零：

```python
padding = (kernel_size - 1) * dilation
return layer(F.pad(x, (padding, 0)))
```

假设 `k=5`、`d=1`，左侧补 4 个零：

```text
0 0 0 0 x0 x1 x2 ...
```

这样输出时刻 `t` 只依赖 `t-4..t`，不会看 `t+1`。

### 3.3 膨胀卷积

膨胀率 `d` 让卷积核跳着看：

```text
d=1: 只看相邻点
d=2: 隔一个点看
d=4: 隔三个点看
```

同样 `k=5`，`d=4` 的窗口覆盖：

\[
(k-1)\times d = 4\times4=16
\]

个时间间隔，感受野更大，但参数量不变。

### 3.4 感受野公式

每个卷积层增加：

\[
(k-1)\times d
\]

每个残差块有两层卷积，所以增加：

\[
2(k-1)d
\]

三个块 `d=1,2,4`，总感受野：

\[
1 + 2(k-1)(1+2+4)
\]

代入 `k=5`：

\[
1+2\times4\times7=57
\]

即模型输出位置最多能看到过去 57 个采样点。

- 10 秒输入有 51 点，57 足够覆盖整个前缀；
- 5 秒输入有 26 点，也被完整覆盖。

---

## 4. 模型定义

核心文件：[`tcn_union/core/model.py`](../tcn_union/core/model.py)

### 4.1 残差块

[`CausalResidualBlock`](../tcn_union/core/model.py#L13-L51)：

```python
residual = self.projection(x)
hidden = self.dropout1(F.relu(self._causal_conv(x, self.conv1)))
hidden = self.dropout2(F.relu(self._causal_conv(hidden, self.conv2)))
return F.relu(residual + hidden)
```

公式：

\[
y=\mathrm{ReLU}\left(\mathrm{proj}(x)+F(x)\right)
\]

其中 `F(x)` 是两层因果卷积。

为什么要有残差连接：

- 让梯度更容易传回浅层；
- 允许模型学习“在输入基础上修改”，而不是完全重写；
- 训练更稳定。

为什么有投影：

- 第一块输入是 6 通道，输出要求 16 通道；
- 用 `1×1 Conv` 把通道数对齐；
- 后面两块输入已经是 16 通道，用 `Identity` 即可。

### 4.2 三个残差块

[`GasTCN.__init__`](../tcn_union/core/model.py#L62-L75)：

```python
self.blocks = nn.Sequential(
    CausalResidualBlock(in_channels, dilation=1),
    CausalResidualBlock(16, dilation=2),
    CausalResidualBlock(16, dilation=4),
)
```

通道数：

```text
输入 6 → 16 → 16 → 16
```

膨胀率：

```text
1 → 2 → 4
```

### 4.3 输出头

```python
self.head = nn.Sequential(
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 3),
)
```

输入是最后一个时间位置的 16 维隐藏向量：

```python
return self.head(hidden[:, :, -1])
```

形状变化：

```text
[B, 6, 51]
  → blocks
[B, 16, 51]
  → 取最后一个时间位置
[B, 16]
  → head
[B, 3]
```

`B` 是 batch size，训练时通常是 8。

### 4.4 为什么只读最后一个位置

做 10 秒预测时，最后一个位置就是 `t=10 秒` 的隐藏状态。

它已经通过因果卷积“看过”前面所有时间点，所以：

- 不需要全局池化；
- 不需要把未来点也喂进去；
- 推理接口简单：给 10 秒前缀，输出 3 个浓度。

### 4.5 辅助头（可选）

[`forward_positions()`](../tcn_union/core/model.py#L89-L108) 允许在隐藏序列的多个过去位置解码：

```python
hidden = self.blocks(x)
outputs = [self.head(hidden[:, :, index]) for index in positions]
```

例如 `positions=(25,)` 表示在 5 秒位置也做一次预测。

当前最终方案没有启用 aux；它只是模型保留的一个训练接口。原因见 [`docs/PCA-VAE汇报提纲.md`](PCA-VAE汇报提纲.md)。

---

## 5. 训练流程

核心文件：[`tcn_union/training/train.py`](../tcn_union/training/train.py)

### 5.1 训练前先划分数据

- 50 个开发实验；
- 12 个锁定测试实验；
- 开发集再分 5 折，每折 10 个验证实验；
- 每折训练 40 / 验证 10。

训练时只使用开发集；测试集完全不参与。

### 5.2 标准化（z-score）

[`fit_scalers()`](../tcn_union/training/train.py#L50-L68)：

\[
x'=\frac{x-\mu_x}{\sigma_x}
\]

\[
y'=\frac{y-\mu_y}{\sigma_y}
\]

关键：

- 均值和标准差**只用训练折**拟合；
- 验证集用训练折的统计量；
- 不能把验证/测试数据混进来算均值。

输入统计量按“通道 × 时间”计算：

```python
x_mean = np.mean(x_train, axis=(0, 2))  # [6]
x_std  = np.std(x_train, axis=(0, 2))   # [6]
```

标签统计量按气体：

```python
y_mean = np.mean(y_train, axis=0)  # [3]
y_std  = np.std(y_train, axis=0)   # [3]
```

### 5.3 DataLoader

[`_train_network()`](../tcn_union/training/train.py#L154-L301) 里：

```python
dataset = TensorDataset(torch.from_numpy(x_train_z), torch.from_numpy(y_train_z))
loader = DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)
```

每个 batch：

```text
x: [8, 6, 51]
y: [8, 3]
```

`shuffle=True` 每个 epoch 打乱训练样本，减少顺序影响。

### 5.4 前向传播和 loss

```python
prediction = model(batch_x)       # [8,3]
loss = criterion(prediction, batch_y)
```

损失函数：

```python
criterion = nn.SmoothL1Loss(beta=1.0, reduction="mean")
```

SmoothL1 的定义：

\[
\text{loss}(e)=
\begin{cases}
0.5e^2, & |e|<1\\
|e|-0.5, & |e|\ge1
\end{cases}
\]

它比平方误差更不容易被极端误差主导，适合小样本。

### 5.5 反向传播和优化器

```python
optimizer = torch.optim.AdamW(
    model.parameters(), lr=1e-3, weight_decay=1e-3, betas=(0.9,0.999)
)
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

- `loss.backward()`：算梯度；
- `clip_grad_norm_`：梯度太大时裁剪，防止训练爆炸；
- `optimizer.step()`：更新参数。

### 5.6 验证和早停

每个 epoch 结束后：

1. `model.eval()`；
2. 在验证集上前向；
3. 反标准化回原始浓度；
4. 计算验证 macro-NMAE；
5. 如果更好，保存当前权重；
6. 连续 50 轮没改善就停止。

代码：

```python
score, _ = _macro_nmae(y_validation, pred, train_range_y)
if score < best_score:
    best_score = score
    best_epoch = epoch
    best_state = ...
    stale_epochs = 0
else:
    stale_epochs += 1
if early_stop and stale_epochs >= PATIENCE:
    break
```

### 5.7 训练后恢复最佳权重

```python
model.load_state_dict(best_state)
return model, stats, best_epoch, best_score, best_prediction
```

注意：返回的不是“最后一轮”的权重，而是**验证集最好那一轮**的权重。

---

## 6. 五折 OOF 和最终重训

核心文件：[`tcn_union/training/gan_train.py`](../tcn_union/training/gan_train.py)

### 6.1 五折 OOF

[`cross_validate()`](../tcn_union/training/gan_train.py#L163-L270)：

```python
for fold in range(5):
    train_rows = [row for row in dev_rows if int(row["fold"]) != fold]
    val_rows   = [row for row in dev_rows if int(row["fold"]) == fold]
```

每折：

- 40 个训练实验；
- 10 个验证实验；
- 用验证集早停；
- 记录最佳 epoch 和验证预测。

5 折跑完后：

- 每个开发实验恰好被预测一次；
- 这 50 个预测叫 **OOF（out-of-fold）预测**；
- OOF 比“训练集自评”更可信。

### 6.2 最终轮数

```python
epochs = math.ceil(float(np.median([row["best_epoch"] for row in fold_scores])))
```

含义：

- 收集 5 折的最佳 epoch；
- 取中位数；
- 向上取整；
- 作为全开发集重训的固定轮数。

为什么用中位数：

- 比平均值更不容易被某一折的极端 epoch 影响；
- 让最终模型训练轮数稳定。

### 6.3 三个 seed

```python
SEEDS = [20260923, 20260924, 20260925]
```

每个 seed 单独训练一个模型，最后：

```python
ensemble = np.mean(seed_predictions, axis=0)
```

注意：

- 不是把三个模型的参数平均；
- 是三个模型分别预测，再平均预测值；
- 三个 seed 改变初始化、数据打乱、Dropout 等随机性；
- 平均预测可以降低方差。

### 6.4 最终训练不使用早停

最终模型在全部 50 个开发实验上训练：

- 不再有验证集；
- 不再早停；
- 使用 5 折中位数 epoch；
- 三个 seed 各训一个模型。

---

## 7. 数据增强（可选）

当前最终方案还会在训练折内加入：

- 真实测量噪声 0.1×；
- PCA 响应基 + 条件 VAE 合成样本 3×。

关键点：

- 只加在训练折；
- 验证/测试始终是真实曲线；
- 增强不改变标签；
- 生成器也只在训练折上拟合。

这部分原理见：

- [`docs/PCA-VAE学习资料.md`](PCA-VAE学习资料.md)
- [`docs/生成式增强对比实验.md`](生成式增强对比实验.md)

如果第一次读代码，可以先跳过增强，只看 `train.py` 的 clean 训练。

---

## 8. 评估

核心文件：[`tcn_union/evaluation/evaluate.py`](../tcn_union/evaluation/evaluate.py)

### 8.1 反标准化

模型输出是标准化后的浓度：

\[
\hat y'
\]

要变回原始浓度：

\[
\hat y=\hat y'\cdot\sigma_y+\mu_y
\]

### 8.2 指标

- **MAE**：平均绝对误差；
- **RMSE**：均方根误差；
- **NMAE**：MAE 除以开发集该气体浓度极差；
- **macro-NMAE**：三种气体 NMAE 的平均；
- **MAPE**：平均绝对百分比误差。

`macro-NMAE` 越低越好。

### 8.3 测试集

测试集只在方案冻结后使用：

- 不参与训练；
- 不参与早停；
- 不参与选 epoch；
- 不参与选增强方案。

---

## 9. 代码阅读顺序

建议按下面顺序读：

| 顺序 | 文件 | 重点 |
|---|---|---|
| 1 | [`data.py`](../tcn_union/core/data.py) | `load_run`、`make_features` |
| 2 | [`model.py`](../tcn_union/core/model.py) | `CausalResidualBlock`、`GasTCN.forward` |
| 3 | [`train.py`](../tcn_union/training/train.py) | `fit_scalers`、`_train_network` |
| 4 | [`gan_train.py`](../tcn_union/training/gan_train.py) | `cross_validate`、`train_final` |
| 5 | [`evaluate.py`](../tcn_union/evaluation/evaluate.py) | `_metric_summary`、测试集评估 |
| 6 | [`predict.py`](../tcn_union/cli/predict.py) | 单文件推理 |

### 最小代码练习

在读完整训练循环前，可以先跑这个形状检查：

```python
from pathlib import Path
from tcn_union.core.data import load_run, make_features
from tcn_union.core.model import GasTCN
import torch

time, response, labels = load_run(Path("data/0.csv"))
x = make_features(time, response, 10.0, "raw_delta")
print("features:", x.shape)          # (6, 51)

model = GasTCN(in_channels=6)
batch = torch.from_numpy(x).unsqueeze(0)  # (1,6,51)
print("hidden:", model.blocks(batch).shape)  # (1,16,51)
print("output:", model(batch).shape)         # (1,3)
```

---

## 10. 小练习

1. 把 `make_features` 的 `seconds` 从 10 改成 5，打印形状，应该是 `(6,26)`。
2. 把 `GasTCN` 的第二个残差块膨胀率从 2 改成 1，重新算感受野。
3. 打印 `_train_network` 每个 epoch 的 loss，观察是否下降。
4. 把 `batch_size` 从 8 改成 4，观察训练时间。
5. 把 `PATIENCE` 从 50 改成 10，观察早停轮数。
6. 把 `seed` 改成另一个值，观察结果变化。

> 这些练习只用于理解代码；不要用测试集来调这些参数。

---

## 11. 常见误解

### 误解 1：模型看到了整条 59.6 秒曲线

不是。模型只看 `0–10` 秒前缀。

### 误解 2：卷积是双向的

不是。所有卷积只在左侧补零，是单向因果卷积。

### 误解 3：输出是原始浓度

不是。模型输出标准化浓度，训练器/推理器再用 checkpoint 中的均值和标准差反标准化。

### 误解 4：五折是随机切的

不是。五折和测试集都在 manifest 中固定，避免反复随机切分造成泄漏。

### 误解 5：三个 seed 是把参数平均

不是。是三个模型分别预测，再平均预测值。

### 误解 6：测试集可以用来选 epoch

不可以。测试集只在方案冻结后评估。

### 误解 7：增强样本也进入验证/测试

不会。增强只作用于训练折，验证/测试始终是真实曲线。

---

## 12. 术语表

| 术语 | 通俗解释 |
|---|---|
| 张量 tensor | 多维数组，例如 `[batch, channel, time]` |
| batch | 一次送进网络的一小批样本 |
| channel | 特征通道，这里是 6 路 |
| 卷积 kernel | 在时间轴上滑动的小窗口 |
| dilation | 卷积核跳着看的间隔 |
| 感受野 | 一个输出位置最多能看到的输入时间范围 |
| 残差连接 | 输出 = 输入 + 变换，帮助训练 |
| z-score | 减均值、除标准差 |
| loss | 预测和真实值的误差 |
| 反向传播 | 从 loss 算梯度、更新参数 |
| epoch | 训练集完整过一遍 |
| 早停 | 验证集不再变好就停止 |
| OOF | 折外预测，模型没见过的验证样本预测 |
| seed | 随机种子，控制可复现性 |
| ensemble | 多个模型预测平均 |
| macro-NMAE | 三种气体 NMAE 的平均 |

---

## 13. 一页纸总结

> 模型输入是 10 秒、6 通道、51 点的传感器特征；它先经过三个因果残差块，每个块有两层 `k=5`、膨胀率 `1/2/4` 的卷积，通道数为 16；总感受野 57 点；最后只取第 51 个时间位置的 16 维向量，经 `16→16→3` 的线性头输出三种浓度。  
>   
> 训练时只用 40 个训练实验拟合 z-score，用 batch size 8、AdamW、SmoothL1 训练；每个 epoch 在 10 个验证实验上算 macro-NMAE，保留最好权重，50 轮不改善就早停；五折跑完后取最佳 epoch 中位数，在全部 50 个开发实验上重训 3 个 seed，预测时三个 seed 平均。测试集只在最后评估，不参与任何选择。
