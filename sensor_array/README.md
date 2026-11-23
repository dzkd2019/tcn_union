# 传感器阵列动态响应建模（双指数骨架 + 三路独立 MLP 残差）

给定 H₂、NH₃、NO₂ 的浓度组合，预测三路传感器在 0–60 s 内的完整相对响应曲线。
当前正式模型是 **双指数响应骨架 + 三路独立 MLP 残差**；纯双指数骨架作为对照。
62 折留一评估中，组合把曲线宏 nRMSE 从骨架的 3.30% 降到 **1.48%**。
默认推荐 `artifacts/models/mlp_per_gas.json`。

骨架对照已完成：双指数与一阶在 3 seed 下几乎等价（宏 1.5209% 对 1.5261%），
两个二阶骨架更差；双指数保留作数据生成骨架，一阶为 45 参数等价备选。
正模型生成数据增强逆 TCN 已完成首轮四臂、旧训练协议增强来源对照与完整旧协议复刻；MAP 反演、区间校准与降级分析已从本分支删除。

## 1. 结论

- 数据只有三种气体（H₂、NH₃、NO₂）；模型、实验与结论都只针对这三种气体。
- 建模目标是相对响应 \(y_j(t)=(R_j(t)-R_j(0))/R_j(0)\)。基线 \(R_j(0)\) 随实验漂移，
  不能由浓度唯一确定，因此作为显式输入；绝对响应为 \(R_j(t)=R_j(0)[1+y_j(t)]\)。
- 双指数骨架每通道 23 个参数（增益二次 10、快时间常数一次 4、慢快差一次 4、
  混合权重一次 4、延迟 1），三通道共 69 个。三路独立残差是三个
  \(7\to64\to64\to1\) 网络，每路 4,737 参数，合计 14,211；输入为三个对数浓度、
  归一化时间与骨架的三路预测，不使用文件编号。
- 62 折留一实验（LOGO）下，曲线宏 nRMSE：双指数骨架 **3.30%**、三路独立组合
  **1.48%**；逐气体（H₂ / NH₃ / NO₂）骨架为 5.17% / 1.69% / 3.04%，
  组合为 2.02% / 1.01% / 1.40%。
- 三路独立相对骨架：曲线宏 **59/62** 折改善（中位 −1.80 百分点）。
- 骨架对照：1 seed 筛选下双指数 1.478%、一阶 1.481%、两个二阶并联 1.634%、
  临界阻尼二阶 1.766%；前两名 3 seed 宏均值 1.5209 ± 0.0154% 与 1.5261 ± 0.0233%，
  逐文件配对 p=0.273，差异与 seed 方差同量级。双指数保留作为数据生成骨架，
  一阶是等价的 45 参数备选；骨架形式不是当前误差的主要瓶颈。
- 模型架构：图 fig07 画浓度输入、双指数骨架、三路 MLP 残差与绝对输出还原，
  只保留推理路径，不写训练细节。
- 数据问题：图 fig09 展示 62 条实验的浓度组合覆盖（哪些组合有、哪些缺失），
  fig10 展示 NH₃ 初值随采集顺序的漂移。
- 曲线总览：fig12–fig14 把每气体的 62 条实测与 62 折留一预测全部叠画；fig15 给出
  逐文件误差散点与均值 ±1σ/2σ/3σ，并写出 <code>curve_error_stats.csv</code>。
  fig16/fig17 是逆问题 TCN 的正模型增强四臂结果与逐折配对；fig18 是旧训练协议下
  仅真实 / PCA-VAE / 正模型 / 正模型 + PCA-VAE 的增强来源对照；fig19 是完整旧协议
  （噪声/边缘/两配置/6 checkpoint）的四臂对照。四张图详见
  <code>docs/evaluation.md</code> §8；它们属于实验分支，不影响前向默认模型。
- 旧训练协议增强来源（10 s，3 seed × 5 折，等 120 条合成量、不含噪声/边缘）：
  仅真实 15.47%、PCA-VAE only 14.22%（9/15，p=0.804）、正模型 only 16.93%
  （5/15，p=0.107）、正模型 + PCA-VAE **11.40%**（14/15，p=0.00018，3 seed
  集成 8.27%）。该对照只到 8.27%，完整旧协议（噪声/边缘/两配置/裁剪）见下条。
  单一来源不足，组合才稳定。
- 完整旧协议复刻（10 s，0.1×MAD 噪声 + 边缘 ×4 + 两套低浓度配置 +
  6 checkpoint + 裁剪）：仅真实 10.79%、PCA-VAE only **7.47%**（14/15，
  p=0.00012，MAPE 1.79%）、正模型 only 10.92%（p=0.524）、正模型 + PCA-VAE
  9.10%（14/15，p=0.0084）。PCA-VAE only 与旧最终 7.47%/1.79% 对齐；
  正模型合成在完整协议下没有增益，混合反而把 VAE 结果拉差；详见
  [`docs/full_protocol_augmentation.md`](docs/full_protocol_augmentation.md) 与
  fig19。
- 限制：单 seed、三路独立固定 600 轮且无早停、没有独立锁定测试；
  62 条实验、每个浓度组合 1 条、无重复；NH₃ 浓度与采集顺序高度混淆；
  H₂=32000 只有 4 条且都在 NO₂=2；这些限制不能靠模型弥补。

## 2. 数据事实

| 项目 | 数值 |
|---|---|
| 实验数 | 62（`data/0.csv`–`data/61.csv`） |
| 采样 | 0–59.6 s，299 点，5 Hz |
| 输入浓度 | H₂ ∈ {8000, 16000, 24000, 32000}；NH₃ ∈ {10,20,30,40,50}；NO₂ ∈ {2,4,6,8,10} |
| 覆盖 | 62/100 个浓度组合，每个组合 1 条，无重复 |
| 输出 | 三路传感器响应 H2、NH3、NO2（列名是通道名，不是只对同名气体敏感） |

浓度单位未注明，本文统一写“原始单位”，不写 ppm。浓度组合覆盖不均：
H₂ 四档分别 22/23/13/4 条；NH₃ 五档 7/14/14/14/13 条；NO₂ 五档 16/12/14/11/9 条。
NH₃ 按档位分块顺序采集，与文件序号相关系数约 0.976；H₂、NO₂ 与序号的相关性只有
−0.110、0.129。

三路初始基线 \(R_j(0)\) 随实验序号漂移（NH₃ 通道可达 9.5 倍），相对化去掉乘性基线，
但去不掉与浓度同步的缓慢漂移。三路响应在 60 s 内单调：H₂、NH₃ 上升，NO₂ 下降；
NH₃ 在 60 s 末仍有初始斜率的约 7.4%，尚未完全进入稳态。

## 3. 模型

### 3.1 双指数骨架

$$
y_j(t)=s_j A_j(\mathbf c)\left[
f_j(\mathbf c)\left(1-e^{-(t-d_j)_+/\tau^{\mathrm{fast}}_j(\mathbf c)}\right)
+(1-f_j(\mathbf c))\left(1-e^{-(t-d_j)_+/\tau^{\mathrm{slow}}_j(\mathbf c)}\right)
\right],
$$

\(s=(+1,+1,-1)\)，\(d_j\in[0,2]\) s，\(0<\tau^{\mathrm{fast}}<\tau^{\mathrm{slow}}\)，
\(f_j\in(0,1)\)。\(\ln A_j\) 是 \(\ln(c/c^{\mathrm{ref}})\) 的二次多项式，
\(\ln\tau^{\mathrm{fast}}_j\)、\(\ln(\tau^{\mathrm{slow}}_j-\tau^{\mathrm{fast}}_j)\) 与
\(\operatorname{logit} f_j\) 是一次多项式；参考浓度 \(c^{\mathrm{ref}}=(16000,30,6)\)。
完整公式、参数计数、初值与优化见 [`docs/model.md`](docs/model.md)；逐参数分析见
[`docs/double_exponential_analysis.md`](docs/double_exponential_analysis.md)。

### 3.2 三路独立 MLP 残差

每个气体通道各自学习骨架残差 \(r_j=y_j-\hat y^{\mathrm{skel}}_j\)，输入为三个对数浓度、
归一化时间以及骨架在当前点的三路预测，共 7 维；\(t_{\max}\) 固定为训练曲线的 60 s，
推断 5 s / 10 s 前缀时也沿用同一个 \(t_{\max}\)。三路网络之间不共享任何权重，
最终预测为骨架加残差。

每路 \(7\to64\to64\to1\)，`tanh` 激活、Adam、权重衰减 \(10^{-4}\)，
PyTorch GPU 全批 600 轮、无早停；训练结束后权重拷回 NumPy 做推理。
加入骨架输出是为了让残差网络知道当前点落在骨架的哪一段。

## 4. 评估协议

- **留一实验（LOGO）**：每次留出 1 条完整曲线，用其余 61 条训练，预测留出曲线；62 折。
  不能随机拆时间点，否则同一条曲线的信息会泄漏。
- 每折先拟合双指数骨架，再用同一训练集拟合三路独立 MLP；两个模型使用完全相同的训练折。
- 曲线指标：逐通道
  \(\mathrm{nRMSE}=\mathrm{RMSE}/\mathrm{RMS}_{\text{训练折}}\)，宏平均为三路算术平均。
- 主判断使用全部 62 折；不设独立锁定测试，NH₃ 与采集顺序的混淆仍写进限制。

完整结果表、逐文件配对与限制见 [`docs/evaluation.md`](docs/evaluation.md)。

## 5. 复现

~~~bash
# 62 折评估（三路独立 MLP，GPU 训练；单线程 BLAS 避免 CPU 被占满）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

# 四骨架 62 折筛选（约 40 分钟）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first,second,second2 --seeds 0 --max-iter 600

# 前两名 3 seed 复跑（约 44 分钟）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first --seeds 20260923,20260924,20260925 --max-iter 600

# 完整旧协议四臂（0.1 噪声 + 边缘 ×4 + 两套低浓度配置 + 6 checkpoint；约 50 分钟）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae forward forward+vae

# 图（PNG + SVG）
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.visualization.full_protocol_augmentation
~~~

`--folds 3` 可用于快速调试。全量评估会重写 `artifacts/evaluation/curve_metrics.csv`、
`curve_summary.csv`、`predictions.npz` 和 `artifacts/models/*.json`。

## 6. 目录

```text
sensor_array/
  core/          data.py  skeletons.py  double_exponential.py  mlp_per_gas.py
                 residual.py  metrics.py  shapes.py
  experiments/   evaluate.py  compare_skeletons.py
  plotting/      model.py  style.py
  artifacts/     models/  evaluation/  comparison/
  figures/model/ fig01、fig03、fig07、fig09、fig10、fig12–fig19（PNG+SVG）
  docs/          model.md  evaluation.md  double_exponential_analysis.md
                 plotting_guide.md  plan_20261123.md  archive/
  tests/         test_model.py
```

## 7. 限制

- 62 条实验、每个浓度组合 1 条、无重复；不能靠模型弥补设计缺口。
- NH₃ 浓度与采集顺序完全混淆（相关系数约 0.976），NH₃ 的“浓度效应”只能解释为
  该采集顺序下的等效效应。
- H₂=32000 只有 4 条且都在 NO₂=2；边角外推不可靠。
- 曲线 nRMSE 是相对形状误差，不是浓度误差；本分支不再报 MAP 浓度反演。
- 骨架对比的 1 seed 筛选偏乐观；前两名 3 seed 的差异与 seed 方差同量级，
  不能据此声称双指数显著优于一阶。
- 模型只在恒浓度阶跃下验证；时变浓度未验证。
- 逆问题 TCN 的完整旧协议只跑 1 套超参数、3 seed × 5 折（15 对），
  种子与折不独立；正模型合成在 40 条训练行上拟合，不能据此声称“任何正模型
  合成都无效”，只能说明当前实现在完整旧协议下没有增益。
