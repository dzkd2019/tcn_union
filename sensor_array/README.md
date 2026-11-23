# 传感器阵列动态响应建模（双指数骨架 + 三路独立 MLP 残差）

给定 H₂、NH₃、NO₂ 的浓度组合，预测三路传感器在 0–60 s 内的完整相对响应曲线。
当前正式模型是 **双指数响应骨架 + 三路独立 MLP 残差**；纯双指数骨架作为对照。
62 折留一评估中，组合把曲线宏 nRMSE 从骨架的 3.30% 降到 **1.48%**。
默认推荐 `artifacts/models/mlp_per_gas.json`。

一阶、二阶等骨架对照，以及正模型生成数据增强逆 TCN 的工作尚未开始；计划见
[`docs/plan_20261123.md`](docs/plan_20261123.md)。MAP 反演、区间校准与降级分析已从本分支删除。

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
- 模型架构：图 fig07 画浓度输入、双指数骨架、三路 MLP 残差与绝对输出还原，
  只保留推理路径，不写训练细节。
- 数据问题：图 fig09 展示 62 条实验的浓度组合覆盖（哪些组合有、哪些缺失），
  fig10 展示 NH₃ 初值随采集顺序的漂移。
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

# 图（PNG + SVG）
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
~~~

`--folds 3` 可用于快速调试。全量评估会重写 `artifacts/evaluation/curve_metrics.csv`、
`curve_summary.csv`、`predictions.npz` 和 `artifacts/models/*.json`。

## 6. 目录

```text
sensor_array/
  core/          data.py  double_exponential.py  mlp_per_gas.py  metrics.py  shapes.py
  experiments/   evaluate.py
  plotting/      model.py  style.py
  artifacts/     models/  evaluation/
  figures/model/ fig01、fig03、fig07、fig09、fig10（PNG+SVG）
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
- 模型只在恒浓度阶跃下验证；时变浓度未验证。
