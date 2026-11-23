# 传感器阵列动态响应建模（双指数骨架 + 三路独立 MLP 残差）

给定 H₂、NH₃、NO₂ 的浓度组合，预测三路传感器在 0–60 s 内的完整相对响应曲线。
当前分支的正式模型是 **双指数响应骨架 + 三路独立 MLP 残差**；纯双指数骨架作为对照。
62 折留一评估中，组合把曲线宏 nRMSE 从骨架的 3.30% 降到 **1.48%**，10 s 下游
三气体宏中位绝对相对误差从 4.99% 降到 **2.14%**，5 s 从 7.42% 降到 **2.96%**。
默认推荐 <code>artifacts/models/mlp_per_gas.json</code>。

P4 区间校准已在同一批折外估计上重做：10 s 原始 Laplace 90% 区间的开发集覆盖只有
32.0% / 30.0% / 6.0%（H₂ / NH₃ / NO₂）；用开发集 50 条拟合全局共形后为 92.0%，
中位宽度 3.5% / 3.4% / 8.7%，锁定 12 条覆盖 91.7%–100%。12 条锁定测试不参与校准。
旧的一阶灰盒、低秩 B 样条残差、直接岭回归与 TCN 代码均已从本分支删除；旧文档只保留在
<code>docs/archive/</code> 作历史归档。

## 1. 结论

- 数据只有三种气体（H₂、NH₃、NO₂）；模型、实验与结论都只针对这三种气体。
- 建模目标是相对响应 \(y_j(t)=(R_j(t)-R_j(0))/R_j(0)\)。基线 \(R_j(0)\) 随实验漂移，
  不能由浓度唯一确定，因此作为显式输入；绝对响应为 \(R_j(t)=R_j(0)[1+y_j(t)]\)。
- 双指数骨架每通道 23 个参数（增益二次 10、快时间常数一次 4、慢快差一次 4、
  混合权重一次 4、延迟 1），三通道共 69 个。三路独立残差是三个
  \(7\to64\to64\to1\) 网络，每路 4,737 参数，合计 14,211；输入为三个对数浓度、
  归一化时间与骨架的三路预测，不使用文件编号。
- 62 折留一实验（LOGO）下，曲线宏 nRMSE：双指数骨架 **3.30%**、三路独立组合
  **1.48%**；10 s 下游三气体宏中位绝对相对误差分别为 **4.99% / 2.14%**，
  5 s 为 **7.42% / 2.96%**。
- 三路独立相对骨架：曲线宏 **59/62** 折改善（中位 −1.80 百分点），10 s 下游宏
  **47/62**（中位 −2.14），5 s 宏 **45/62**（中位 −3.14）。开发集 50 条与
  锁定测试 12 条方向一致，锁定只作描述性复核。
- P4 区间校准：10 s 原始 Laplace 90% 的开发覆盖只有 32.0% / 30.0% / 6.0%；
  全局共形为 92.0%，中位宽度 3.5% / 3.4% / 8.7%。NH₃ 档位共形没有更窄
  （3.9% / 5.4% / 9.6%），采集块共形的开发覆盖只有 84.0%。因此全局共形作默认，
  分组共形只作诊断；区间不改变点估计。
- 代表案例：图 fig06 选出当前模型宏误差的最优、中位、最差三条，以及 NH₃ 低端
  与 NH₃/NO₂ 高端两个设计边界；每行三路气体同时画实测、骨架与组合模型。
- 模型架构：图 fig07 画浓度输入、双指数骨架、三路 MLP 残差与绝对输出还原，
  只保留推理路径，不写训练细节。
- 下游误差：图 fig08 把 62 条实验的浓度反演相对/绝对误差画成散点，不区分 OOF 与锁定测试；
- 数据问题：图 fig09 展示 62 条实验的浓度组合覆盖（哪些组合有、哪些缺失），图 fig10 展示 NH₃ 初值随采集顺序的漂移；图 fig11 展示少一路传感器时的浓度反演误差。
- 限制：单 seed、三路独立固定 600 轮且无早停、下游似然使用训练集自报残差尺度、
  区间只用开发 50 条校准；三路独立同时改变了容量与训练方式，不能把全部收益
  单独归因于“三路独立”。
- 62 条实验、每个浓度组合 1 条、无重复；NH₃ 浓度与采集顺序高度混淆；
  H₂=32000 只有 4 条且都在 NO₂=2；这些限制不能靠模型弥补。

## 2. 数据事实

| 项目 | 数值 |
|---|---|
| 实验数 | 62（<code>data/0.csv</code>–<code>data/61.csv</code>） |
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
完整公式、参数计数、初值与优化见 [<code>docs/model.md</code>](docs/model.md)；逐参数分析见 [<code>docs/double_exponential_analysis.md</code>](docs/double_exponential_analysis.md)。

### 3.2 三路独立 MLP 残差

每个气体通道各自学习骨架残差 \(r_j=y_j-\hat y^{\mathrm{skel}}_j\)，输入为三个对数浓度、
归一化时间以及骨架在当前点的三路预测，共 7 维；\(t_{\max}\) 固定为训练曲线的 60 s，
推断 5 s / 10 s 前缀时也沿用同一个 \(t_{\max}\)。三路网络之间不共享任何权重，
最终预测为骨架加残差。

每路 \(7\to64\to64\to1\)，<code>tanh</code> 激活、Adam、权重衰减 \(10^{-4}\)，
PyTorch GPU 全批 600 轮、无早停；训练结束后权重拷回 NumPy，推理与下游 MAP 在 CPU
完成。加入骨架输出是为了让残差网络知道当前点落在骨架的哪一段。逐文件配对、
诊断与限制见 [<code>docs/evaluation.md</code>](docs/evaluation.md)。

## 4. 评估协议

- **留一实验（LOGO）**：每次留出 1 条完整曲线，用其余 61 条训练，预测留出曲线；62 折。
  不能随机拆时间点，否则同一条曲线的信息会泄漏。
- 每折先拟合双指数骨架，再用同一训练集拟合三路独立 MLP；两个模型使用完全相同的训练折。
- 曲线指标：逐通道
  \(\mathrm{nRMSE}=\mathrm{RMSE}/\mathrm{RMS}_{\text{训练折}}\)，宏平均为三路算术平均。
- 下游指标：在 5 s、10 s 前缀上用同一套 MAP 估计浓度。MAP 残差为数据加权误差 +
  对数正态先验 + 边界软墙；每个模型使用训练折 in-sample 残差尺度
  \(\sigma_{\mathrm{eff}}=\sqrt{\sigma_{\mathrm{noise}}^2+\sigma_{\mathrm{model}}^2}\)。
- 主判断使用全部 62 折；开发集 50 条与锁定测试 12 条只作描述性复核，不用于选型。
  锁定测试是 <code>{2,7,15,20,23,27,31,38,41,46,55,60}</code>。
- 区间校准只用开发集 50 条拟合共形分位数；12 条锁定测试不参与分位数拟合，
  只报告覆盖与宽度。

完整结果表、逐文件配对统计、开发/锁定拆分、P4 区间校准与限制见
[<code>docs/evaluation.md</code>](docs/evaluation.md)。

## 5. 复现

~~~bash
# 62 折评估（三路独立 MLP，GPU 训练；单线程 BLAS 避免 CPU 被占满）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

# P4 区间校准（只读评估 CSV，不重新训练）
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.analysis.calibration

# 只跑前 3 折做冒烟
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate --folds 3 --max-iter 50

# 图（PNG + SVG）
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model

# 新模型单元测试
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s sensor_array/tests -v
~~~

完整 62 折评估本次约 766 s（12.8 min）；若同时有其他负载，时间会变长。
模型保存到 <code>artifacts/models/double.json</code> 与
<code>mlp_per_gas.json</code>。评估 CSV 在 <code>artifacts/evaluation/</code>，
区间 CSV 在 <code>artifacts/calibration/</code>，图在 <code>figures/model/</code>。

推荐模型（三路独立 MLP + 双指数）：

~~~python
from sensor_array import PerGasMLPDoubleExponentialModel, load_experiments

model = PerGasMLPDoubleExponentialModel.load(
    "sensor_array/artifacts/models/mlp_per_gas.json"
)
experiment = load_experiments()[0]
relative = model.predict_relative(experiment.concentration[None], experiment.time)[0]
absolute = model.predict_response(
    experiment.concentration[None], experiment.time, experiment.initial[None]
)[0]
~~~

对照模型（骨架）：

~~~python
from sensor_array import DoubleExponentialArrayModel, load_experiments

model = DoubleExponentialArrayModel.load("sensor_array/artifacts/models/double.json")
experiment = load_experiments()[0]
relative = model.predict_relative(experiment.concentration[None], experiment.time)[0]
~~~

## 6. 代码结构

~~~
sensor_array/
  core/
    data.py                 读取 CSV、相对响应、设计表
    shapes.py               逐实验形状初值、岭回归、浓度特征
    double_exponential.py   双指数骨架（69 参数）
    mlp_per_gas.py          双指数骨架 + 三路独立 MLP 残差（PyTorch GPU）
    metrics.py              曲线 nRMSE
  analysis/
    estimation.py           噪声估计、MAP + Laplace、浓度边界与起点
    calibration.py          P4 区间校准（Laplace、全局/分组共形）
  experiments/
    evaluate.py             62 折 LOGO、5/10 s 下游、模型保存
  plotting/
    style.py                字体、配色、PNG+SVG 保存
    model.py                fig01–fig11
  tests/
    test_model.py           形状、双指数、per-gas MLP、保存/加载、MAP
  docs/
    model.md                模型公式、参数、接口
    double_exponential_analysis.md  参数化、初值、联合拟合与最终参数
    evaluation.md           评估结果、P4 校准与限制
    plotting_guide.md       逐图脚本与函数索引
    archive/                旧模型历史文档（代码已删除）
  artifacts/
    models/                 double.json、mlp_per_gas.json
    evaluation/             curve_*、downstream_*、predictions.npz
    calibration/            intervals.csv、summary.csv、by_group.csv
  figures/
    model/                  fig01–fig11，PNG + SVG
  MANIFEST.md / SHA256SUMS  当前快照
~~~

## 7. 限制

1. **单 seed、无早停**：MLP 结果的随机性没有被多种子平均；固定 600 轮，轮数没有按折选择。
2. **下游似然尺度自报**：<code>sigma_model</code> 是训练折 in-sample 残差，可能偏乐观；
   这会同时影响两个模型，但不能解释为纯结构差异。
3. **区间校准仍有限**：分位数只用开发集 50 条，分组样本最少只有 6–7 条；锁定 12 条
   只作描述。停止策略、剖面似然与嵌套协方差没有在本模型上重做。
4. **非严格嵌套**：浓度边界、噪声估计与 MAP 起点使用全部 62 条，属于共享 nuisance；
   留一折只保证模型拟合不使用留出文件。
5. **NH₃ 混淆与 H₂ 高端稀疏**：见第 2 节；模型不能弥补设计缺陷。
6. **三气体限定**：不对第四种气体做任何声称；所有分析只基于现有 62 条实验。

## 8. 相关文档

- [模型细节](docs/model.md)
- [双指数参数化与拟合](docs/double_exponential_analysis.md)
- [评估结果与 P4 校准](docs/evaluation.md)
- [逐图索引](docs/plotting_guide.md)
- [快照说明](MANIFEST.md)
- [历史归档](docs/archive/)