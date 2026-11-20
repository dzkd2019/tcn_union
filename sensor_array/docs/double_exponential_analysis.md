# 双指数骨架：参数化、初值与拟合

本文只拆解 **纯双指数骨架**（<code>DoubleExponentialArrayModel</code>，产物
<code>artifacts/models/double.json</code>）。完整组合模型（骨架 + 三路独立 MLP 残差）
见 [model.md](model.md)，评估数字见 [evaluation.md](evaluation.md)。

一句话：骨架把每条响应曲线拆成“快指数 + 慢指数”的加权和，所有参数都是浓度的
光滑函数；先用逐实验单指数拟合得到初值，再用一次加权最小二乘把 69 个参数一起
拟合出来。

## 1. 目标量

对第 \(j\) 路传感器（\(j=1,2,3\) 对应 H₂、NH₃、NO₂），记原始读数为 \(R_j(t)\)，
拟合目标是**有符号相对响应**

$$
y_j(t)=\frac{R_j(t)-R_j(0)}{R_j(0)}.
$$

这里 \(y_j\) 已经带方向：H₂、NH₃ 为正，NO₂ 为负。模型内部再把方向拆出来：

$$
y_j(t)=s_j\,\tilde y_j(t),\qquad s=(+1,+1,-1),
$$

\(\tilde y_j\ge 0\) 是“幅度”。这样做的好处是：

1. 去掉实验间可达一个数量级的乘性基线差异（NH₃ 的 \(R_j(0)\) 从 786 到 7437）；
2. 三路传感器可以用同一套“正值幅度”参数化；
3. 绝对输出由 \(R_j(t)=R_j(0)[1+y_j(t)]\) 还原，\(R_j(0)\) 作为显式输入，不由浓度预测。

> 数据没有注明浓度单位，本文不写 ppm；浓度只用原始单位。

## 2. 为什么是双指数

单指数

$$
\tilde y(t)=A\left(1-e^{-(t-d)_+/\tau}\right)
$$

在大多数实验上已经能解释主要形状，但对 H₂ 和 NO₂ 会留下系统性的“先快后慢”
偏差：起始段上升/下降比单指数快，后段又拖着一个更慢的尾巴。双指数把它拆成

$$
\tilde y_j(t)=A_j(\mathbf c)\left[
f_j(\mathbf c)\left(1-e^{-(t-d_j)_+/\tau^{\mathrm{fast}}_j(\mathbf c)}\right)
+(1-f_j(\mathbf c))\left(1-e^{-(t-d_j)_+/\tau^{\mathrm{slow}}_j(\mathbf c)}\right)
\right].
$$

它有三个好处：

- **仍是一阶系统的叠加**：每个分量都有明确的稳态幅度和时间常数，可解释；
- **嵌套单指数**：当 \(f_j\to1\)，或 \(\tau^{\mathrm{slow}}_j\to\tau^{\mathrm{fast}}_j\)，模型退化为
  单指数；数据不需要第二个时间尺度时，它不会强行多出一个；
- **快慢分离**：H₂ 的快分量约 3 s、慢分量约 18 s；NO₂ 以约 12 s 的慢分量为主，
  只保留很小的快分量；NH₃ 在当前数据上几乎就是单指数（\(f\approx1\)）。

这比直接换成 stretched exponential 或二阶系统更保守：参数仍有“幅度、时间常数、
混合比例、延迟”的物理含义，且参数数只从 45 增加到 69。

## 3. 参数化：每个参数长什么样

### 3.1 浓度特征

令

$$
u_k=\ln\frac{c_k}{c_k^{\mathrm{ref}}},\qquad
\mathbf c^{\mathrm{ref}}=(16000,\,30,\,6),
$$

参考浓度取设计域的中心附近。用对数浓度而不是原始浓度，是因为响应幅度随浓度
更接近幂律/饱和，而不是线性；除以参考浓度让 \(u=0\) 落在数据中心，改善数值条件。

两个特征向量：

$$
X(\mathbf c)=\big[1,u_1,u_2,u_3,u_1^2,u_2^2,u_3^2,u_1u_2,u_1u_3,u_2u_3\big]\in\mathbb R^{10},
$$

$$
U(\mathbf c)=\big[1,u_1,u_2,u_3\big]\in\mathbb R^{4}.
$$

\(X\) 是二次单项式（含交叉项），用来描述增益的饱和与交叉敏感；\(U\) 是一次，
用来描述时间常数的平滑变化。

### 3.2 稳态幅度 \(A_j\)

$$
A_j(\mathbf c)=\exp\!\big(\beta_j^\top X(\mathbf c)\big),\qquad \beta_j\in\mathbb R^{10}.
$$

- **为什么用 exp**：幅度必须为正；exp 把无约束的 \(\beta_j\) 映射到 \((0,\infty)\)，
  优化器不需要边界约束。
- **为什么是二次**：H₂ 的增益在 H₂ 高端会饱和，而且被 NO₂ 强烈抑制（NO₂ 从 2 到 10
  时 H₂ 增益可降一个量级），所以需要二次项和 \(u_{\mathrm H_2}u_{\mathrm{NO_2}}\) 交叉项；
  纯线性不够。
- **为什么不再高次**：62 条曲线、每条 299 点，但浓度组合只有 62 个；三次项会把
  实验间漂移也拟合进去，留一误差反而变大。
- 10 个系数中，\(\beta_{j0}\) 是参考浓度处的 \(\ln A_j\)，其余 9 个是弹性和交互项。

### 3.3 快时间常数 \(\tau^{\mathrm{fast}}_j\)

$$
\tau^{\mathrm{fast}}_j(\mathbf c)=\exp\!\big(\gamma_j^\top U(\mathbf c)\big),\qquad \gamma_j\in\mathbb R^{4}.
$$

- 时间常数必须为正，同样用 exp；
- 只给一次项：时间常数的浓度趋势比增益平缓，62 条曲线不足以稳定支撑二次曲面；
- 一次项让 \(\tau^{\mathrm{fast}}\) 随浓度单调变化，外推时不会出现意外的峰谷。

### 3.4 慢时间常数 \(\tau^{\mathrm{slow}}_j\)

不直接参数化 \(\tau^{\mathrm{slow}}\)，而是参数化**慢快差**：

$$
\tau^{\mathrm{slow}}_j(\mathbf c)=\tau^{\mathrm{fast}}_j(\mathbf c)
+\exp\!\big(\delta_j^\top U(\mathbf c)\big),\qquad \delta_j\in\mathbb R^{4}.
$$

- 这样天然保证 \(\tau^{\mathrm{slow}}_j>\tau^{\mathrm{fast}}_j\)，不会出现两个指数“交换身份”；
- 如果分别用两个独立的 exp 参数化，优化器可能在迭代中把 fast/slow 标签互换，
  得到等价但不可解释的解；
- \(\delta_j\) 的指数形式保证差值正，同时允许差值随浓度变化。

### 3.5 混合权重 \(f_j\)

$$
f_j(\mathbf c)=\operatorname{logit}^{-1}\!\big(\eta_j^\top U(\mathbf c)\big)
=\frac{1}{1+e^{-\eta_j^\top U(\mathbf c)}},\qquad \eta_j\in\mathbb R^{4}.
$$

- \(f_j\in(0,1)\) 自动满足；
- \(f_j\to1\) 表示只剩快分量，\(f_j\to0\) 表示只剩慢分量，两端都退化为单指数；
- logistic 的线性指数仍然可解释：\(\eta\) 控制“快分量占比”随浓度的变化。

### 3.6 纯延迟 \(d_j\)

每路一个标量，\(d_j\in[0,2]\) s，三路分别约 0.25/0.22/0.11 s。

- 延迟是传感器/气路的传输属性，主要与传感器有关，不随浓度大幅变化；
- 62 条曲线不足以识别“延迟 × 浓度”的交互，强行参数化会过拟合；
- 只给 3 个延迟参数，并在联合拟合中加 \([0,2]\) s 的盒约束。

### 3.7 参数总数

| 部分 | 每路个数 | 三路合计 |
|---|---:|---:|
| \(\beta_j\)（增益二次） | 10 | 30 |
| \(\gamma_j\)（快时间常数一次） | 4 | 12 |
| \(\delta_j\)（慢快差一次） | 4 | 12 |
| \(\eta_j\)（混合权重一次） | 4 | 12 |
| \(d_j\)（延迟） | 1 | 3 |
| **合计** | **23** | **69** |

方向 \(s_j\) 由全部实验的响应方向固定，不作为待拟合参数；否则它会和 \(A_j>0\)
完全冗余。

## 4. 初值：先逐条曲线，再回归到浓度

联合拟合有 69 个参数，直接用单位初值容易落入局部极小。当前实现分两步。

### 4.1 逐实验单指数拟合

对每条实验、每路传感器，先拟合

$$
\tilde y(t)=A\left(1-e^{-(t-d)_+/\tau}\right),
$$

其中目标已经乘过 \(s_j\)，所以 \(A>0\)。实现是
<code>core/shapes.py::_fit_one_shape</code>：

- 初值 \(A=\max(\tilde y(59.6),10^{-4})\)，\(\tau=10\) s；
- 延迟从 0 和 0.3 s 两个起点分别拟合，取残差小的；
- 边界：\(A\in[0,10]\)，\(\tau\in[0.1,120]\) s，\(d\in[0,2]\) s；
- 输出每条实验的 \((A_i,\tau_i,d_i)\)，三个数组都是 \((62,3)\)。

这一步把“每条曲线自己的快慢和幅度”先估出来，给联合拟合一个接近数据的起点。

### 4.2 把逐实验参数回归到浓度

对每路 \(j\)，用岭回归把逐实验参数变成浓度函数：

$$
\ln A_i \approx \beta_j^\top X(\mathbf c_i),\qquad
\ln\frac{\tau_i}{2}\approx \gamma_j^\top U(\mathbf c_i),\qquad
\ln(1.5\tau_i)\approx \delta_j^\top U(\mathbf c_i).
$$

- \(\tau^{\mathrm{fast}}\) 初值取 \(\tau_i/2\)，慢快差初值取 \(1.5\tau_i\)，相当于把单指数
  时间常数拆成一个更快的分量和一个更慢的分量；混合权重初值 \(f=0.5\)；
- 延迟初值取逐实验中位数（三路分别约 0.25/0.22/0.11 s）；
- 岭回归的截距不惩罚，\(\lambda=10^{-3}\)；特征共线时（例如 NH₃ 与采集顺序）
  岭惩罚让系数保持有限。

这一步给出 \(\beta_j,\gamma_j,\delta_j,\eta_j=0\) 和 \(d_j\) 的初值，然后才进入联合优化。

## 5. 联合拟合：69 个参数一起解

### 5.1 目标函数

记 \(\hat y_j(t;\theta)\) 为骨架预测，\(\theta\) 为 69 维参数。残差向量由两部分拼成：

$$
r(\theta)=
\begin{bmatrix}
\sqrt{w_j}\,\big(\hat y_j(t_i;\theta)-y_{ij}\big)\\[2pt]
\sqrt{\lambda}\,\mathbf 1_{\text{non-intercept}}\odot\theta
\end{bmatrix}.
$$

- **通道权重** \(w_j=1/\overline{y_j^2}\)，再归一化到均值 1。三路的相对幅度差很多
  （H₂ 约 1%，NH₃ 约 45%，NO₂ 约 9%），不加权时最小二乘会几乎只拟合 NH₃；
  加权后三路的 nRMSE 量级被拉到同一档。
- **岭惩罚** 只加在非截距系数上；延迟不惩罚。\(\lambda=10^{-3}\) 是轻正则，主要用来
  稳定共线方向，不改变主趋势。
- 残差里没有对 \(R_j(0)\) 建模：目标量已经是相对响应。

### 5.2 优化器

- <code>scipy.optimize.least_squares</code>，TRF；
- 只有延迟有盒约束 \([0,2]\) s；\(A,\tau,f\) 的正性/有界性由 exp/logit 保证；
- <code>x_scale="jac"</code>，让 69 个量级差异很大的参数共用同一个信任域；
- <code>max_nfev=600</code>；当前全量拟合实际 **48 次函数评估**收敛，cost ≈ 0.0181。

没有多起点：初值来自逐曲线拟合，已经落在合理盆地；多起点会增加成本，且对当前
数据没有可见收益。

### 5.3 最终参数

在参考浓度 \(\mathbf c^{\mathrm{ref}}=(16000,30,6)\) 处，把 69 个系数换算成可读量：

| 气体 | \(A_j\) | \(\tau^{\mathrm{fast}}\) / s | \(\tau^{\mathrm{slow}}\) / s | \(f_j\) | \(d_j\) / s |
|---|---:|---:|---:|---:|---:|
| H₂ | 0.0113 | 2.90 | 18.10 | 0.946 | 0.245 |
| NH₃ | 0.4515 | 21.38 | 21.40 | 1.000 | 0.222 |
| NO₂ | 0.0970 | 1.04 | 11.74 | 0.110 | 0.108 |

读法：

- **H₂**：约 95% 的幅度走 2.9 s 的快分量，剩 5% 走 18 s 的慢尾巴；所以曲线先快速
  抬升，后面还有缓慢爬升。
- **NH₃**：\(f\approx1\)，慢快差约 0.02 s，双指数退化为单指数，时间常数约 21.4 s；
  这不是“双指数参数浪费”，而是数据告诉我们在当前设计下 NH₃ 只有一个可分辨尺度。
- **NO₂**：只有约 11% 的幅度走 1.0 s 的快分量，89% 走 11.7 s 的慢分量；负号由
  \(s_{\mathrm{NO_2}}=-1\) 给出，幅度 \(A\) 仍为正。
- 三路延迟都小于 0.25 s，和“气路传输 + 传感器起始响应”量级一致。

完整的 69 个系数在 <code>artifacts/models/double.json</code>：<code>gain_coef</code>
(3×10)、<code>tau_fast_coef</code> (3×4)、<code>delta_coef</code> (3×4)、
<code>mix_coef</code> (3×4)、<code>delay</code> (3,)。

## 6. 这个参数化解决了什么、没解决什么

**解决了**

- 正性/有界性：exp、logit、差值参数化让优化过程中不会出现负幅度、负时间常数、
  快慢交换；
- 通道不平衡：加权最小二乘让三路等权；
- 小样本：二次只给增益，时间常数只给一次，69 个参数由 62 条曲线共享；
- 可解释：\(A,\tau,f,d\) 都能单独拿出来做机理分析，MLP 残差关闭后骨架仍可用。

**没解决**

- NH₃ 浓度与采集顺序完全混淆（相关系数约 0.976），\(\beta_{\mathrm{NH_3}}\) 只能解释为
  “该采集顺序下的等效增益”，不是纯灵敏度；
- H₂=32000 只有 4 条且都在 NO₂=2，H₂ 高端与 NO₂ 的交互没有数据支撑，外推不可靠；
- 延迟被假设为与浓度无关；如果真实延迟随浓度变化，这一项会吸收到 \(A\) 或 \(\tau\) 里；
- 骨架在边角实验（file 0）仍有系统偏差，这正是后面 MLP 残差存在的原因。

## 7. 复现

```python
from sensor_array import DoubleExponentialArrayModel
from sensor_array.core.data import load_experiments

experiments = load_experiments()
model = DoubleExponentialArrayModel().fit(experiments)   # 逐实验初值 + 联合拟合
model.save("sensor_array/artifacts/models/double.json")

relative = model.predict_relative([[16000, 30, 6]], experiments[0].time)
gain = model.gain([[16000, 30, 6]])
tau_fast, tau_slow, mix = model.time_constants([[16000, 30, 6]])
```

当前全量拟合：69 参数，48 次函数评估，cost ≈ 0.0181；62 折 LOGO 曲线宏 nRMSE
3.30%（H₂ 5.17%、NH₃ 1.69%、NO₂ 3.04%）。
