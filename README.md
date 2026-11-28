# 三气体传感器阵列：动态响应建模与早期浓度反演

本仓库围绕 62 次三气体（H₂ / NH₃ / NO₂）传感器实验，包含两个相互衔接的任务：

1. **正问题**（`sensor_array/`）：给定浓度组合，预测三路传感器 0–60 s 的相对响应曲线；
2. **逆问题**（`tcn_union/`）：给定 10 s 响应前缀，反演三种气体浓度，并比较三种数据增强方式。

数据只有 `data/0.csv`–`data/61.csv` 共 62 条实验，**不可扩充、不补测**；传感器已损坏。浓度单位统一记为 ppm。

---

## 1. 当前结论

### 1.1 正问题：双指数骨架 + 三路独立 MLP 残差

- 建模对象是相对响应 $y_j(t)=(R_j(t)-R_j(0))/R_j(0)$；$R_j(0)$ 随实验漂移，作为显式输入保留。
- 骨架为双指数（快、慢两个一阶分量），每通道 23 个参数、三通道 69 个；三路独立 MLP 残差各 $7\to64\to64\to1$，每路 4,737 参数，合计 14,211。
- 十折验证曲线宏 nRMSE：双指数骨架 **3.30%**，加三路 MLP 后 **1.48%**；逐气体 H₂ / NH₃ / NO₂ 为 **2.02% / 1.01% / 1.40%**，60/62 条留出曲线改善。
- 骨架对比：单次训练初筛双指数 1.478%、一阶 1.481%；前两名多次训练取平均后为 1.5209% 与 1.5261%，逐文件配对 p=0.273，差异不显著。保留双指数是因其可解释的快慢时间尺度，而不是显著更准。
- 当前模型文件：`sensor_array/artifacts/models/mlp_per_gas.json`；骨架对照 `sensor_array/artifacts/models/double.json`。

### 1.2 逆问题：三种数据增强方式

统一口径：50 条开发实验、10 s 前缀、5 折实验级 OOF、多个 seed 训练后取平均预测、两套低浓度配置、6 个 checkpoint 平均；量程来自开发集 50 条；锁定 12 条未参与选择。

| 方法 | 宏 3σ / % | 宏 MAPE / % | H₂ 3σ / MAPE | NH₃ 3σ / MAPE | NO₂ 3σ / MAPE |
|---|---:|---:|---:|---:|---:|
| PCA-VAE | 7.472 | 1.791 | 9.313 / 2.173 | 4.948 / 1.152 | **8.156 / 2.046** |
| 传感器模型生成 | 7.866 | 1.746 | 8.082 / 1.568 | **4.346 / 1.038** | 11.169 / 2.632 |
| 组合使用 | **7.062** | **1.739** | 8.084 / 2.019 | 4.948 / 1.152 | **8.156 / 2.046** |

读法：

- **组合使用的宏 3σ（7.062%）与宏 MAPE（1.739%）均为三种中最低**；它的 H₂ 3σ 与传感器模型生成几乎相同，NH₃/NO₂ 保持 PCA-VAE 的水平。
- **PCA-VAE** 的 NO₂ 3σ/MAPE 最好（8.156% / 2.046%）；**传感器模型生成** 的 H₂、NH₃ 最好，但 NO₂ 明显变差（11.169% / 2.632%）。
- 三种方法都在同一开发集上给出 OOF 预测，直接按上表绝对指标比较。
- 组合使用的含义：把 PCA-VAE 与传感器模型生成的数据结合使用；微调时冻结共享骨干和 NH₃/NO₂ 输出头，只更新 H₂ 输出头，最终 H₂ 采用组合分支，NH₃/NO₂ 采用 PCA-VAE 的多个 seed 平均预测。
- 这里的正模型在项目中统一称 **hierarchical 正模型**：双指数骨架 + 三路独立 MLP 残差，合成时叠加三气体联合低秩残差采样（`forward_prob_joint10`）；不是早期随机效应版 `HierarchicalRandomEffectsModel`。
- 三种方式都相对“仅真实”有明显改善；论文主结果使用组合使用，另两种作为组成方法与对照。

---

## 2. 数据与固定划分

- 62 个文件，每个 299 行、7 列；时间 0–59.6 s，采样间隔 0.2 s（5 Hz）。
- 列：`time`、三路响应 `H2/NH3/NO2`、三路浓度标签 `conc1/conc2/conc3`。
- 浓度档位：H₂ {8000, 16000, 24000, 32000}，NH₃ {10,20,30,40,50}，NO₂ {2,4,6,8,10}；62 个组合互不重复，每个组合 1 条。
- 锁定测试集固定为 `2, 7, 15, 20, 23, 27, 31, 38, 41, 46, 55, 60`，其余 50 条为开发集；划分清单在 `artifacts/data/splits/manifest.csv`。
- 锁定集只做描述性统计，不用于选方法、seed 或超参数。

主要设计限制：

- NH₃ 浓度与采集顺序高度相关（约 0.976），NH₃ 的“浓度效应”只能解释为该采集顺序下的等效效应；
- H₂=32000 只有 4 条，且都在 NO₂=2，边角外推不可靠；
- 每条浓度组合只有 1 条实验，无法评估重复性、气体缺失、零浓度或跨器件泛化。

---

## 3. 正问题模型

### 3.1 双指数骨架

$$
y_j(t)=s_j A_j(\mathbf c)\left[
f_j(\mathbf c)\left(1-e^{-(t-d_j)_+/\tau^{\mathrm{fast}}_j(\mathbf c)}\right)
+(1-f_j(\mathbf c))\left(1-e^{-(t-d_j)_+/\tau^{\mathrm{slow}}_j(\mathbf c)}\right)
\right],
$$

$s=(+1,+1,-1)$，$d_j\in[0,2]$ s，$0<\tau^{\mathrm{fast}}<\tau^{\mathrm{slow}}$，$f_j\in(0,1)$。$\ln A_j$ 是 $\ln(c/c^{\mathrm{ref}})$ 的二次多项式，$\ln\tau^{\mathrm{fast}}_j$、$\ln(\tau^{\mathrm{slow}}_j-\tau^{\mathrm{fast}}_j)$ 与 $\operatorname{logit} f_j$ 是一次多项式，参考浓度 $c^{\mathrm{ref}}=(16000,30,6)$。

### 3.2 三路独立 MLP 残差

每个气体通道各自学习骨架残差，输入为三个对数浓度、归一化时间与骨架的三路预测，共 7 维；三路网络不共享权重。每路 $7\to64\to64\to1$，`tanh` 激活，Adam，权重衰减 $10^{-4}$，固定 600 轮、无早停；训练后权重拷回 NumPy 推理。

### 3.3 评估

十折验证：按实验划分 10 折，每次留出一折完整曲线，用其余曲线训练；同一折内骨架和 MLP 使用完全相同的训练集。逐条留出曲线的误差按通道 $\mathrm{nRMSE}=\mathrm{RMSE}/\mathrm{RMS}_{\text{训练折}}$ 计算，宏平均为三通道算术平均。

完整公式、参数和逐参数分析见 [`sensor_array/docs/model.md`](sensor_array/docs/model.md) 与 [`sensor_array/docs/double_exponential_analysis.md`](sensor_array/docs/double_exponential_analysis.md)；结果与限制见 [`sensor_array/docs/evaluation.md`](sensor_array/docs/evaluation.md)。

---

## 4. 逆问题训练与评估

- 网络：因果 TCN，3 个残差块、16 通道、膨胀率 1/2/4，感受野 57 个采样点，覆盖 10 s 输入；输出三路标准化浓度。
- 训练：SmoothL1，AdamW（lr $10^{-3}$，weight decay $10^{-3}$），batch 8，梯度裁剪 1，最多 500 epoch，验证连续 50 epoch 不改善则早停。
- 增强：PCA 响应基 + 条件 VAE 生成曲线；真实测量噪声（0.1×MAD）；边缘浓度复制；两套低浓度损失加权配置；6 个 checkpoint 平均。
- 指标：逐气体 MAPE、NRMSE（注明量程）、3σ、95% 置信区间与 n，宏平均为三气体算术平均；3σ 是实验间离散度，不是置信区间。
- 每次评估固定输出三张图：有符号绝对误差、有符号相对误差、三气体均值与 95% CI；PNG 300 dpi + SVG，不输出 PDF。

指标定义与图件规范见 [`AGENTS.md`](AGENTS.md) 第 10 节；三种方法的完整对比见 [`sensor_array/docs/augmentation_comparison_20261127.md`](sensor_array/docs/augmentation_comparison_20261127.md)。

---

## 5. 复现

全部命令从仓库根目录运行，使用项目虚拟环境；不要用 `uv run`（沙箱内 uv 缓存不可写）。

```bash
# 正问题：评估（GPU 训练 MLP，CPU 拟合双指数）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
  --folds 62 --max-iter 600

# 正问题：四骨架筛选（十折验证口径）
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.compare_skeletons \
  --folds 62 --models double,first,second,second2 --seeds 0 --max-iter 600

# 逆问题：PCA-VAE 与仅真实
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae \
  --output artifacts/augmentation/protocols/full

# 逆问题：传感器模型生成
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 --seconds 10 --epochs 500 \
  --configs 1.0 0,0.3,1.0 --arms real vae forward_prob_joint10 \
  --output artifacts/augmentation/pure_forward/joint10

# 逆问题：组合使用
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.no2_guard \
  --folds 5 --seeds 20260923,20260924,20260925 \
  --configs "1.0;0,0.3,1.0" \
  --output artifacts/augmentation/no2_guard/full

# 逆问题：三方法汇总与图件
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m tcn_union.experiments.compare_augmentation

# 测试
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m unittest discover -s sensor_array/tests -t .
```

图件重生成：

```bash
MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
```

---

## 6. 目录

```text
data/                       62 条实验 CSV
sensor_array/               正问题：数据、双指数骨架、三路 MLP、评估与绘图
  core/                     data.py  skeletons.py  double_exponential.py
                            mlp_per_gas.py  residual.py  metrics.py  shapes.py
  experiments/              evaluate.py  compare_skeletons.py
  plotting/                 model.py  style.py
  artifacts/                models/  evaluation/  comparison/
  figures/model/            fig01、fig03、fig07、fig09、fig10、fig12–fig15
  docs/                     模型、评估、骨架分析、绘图指南、增强对比
tcn_union/                  逆问题：TCN、PCA-VAE、三种增强、评估与绘图
  core/                     data.py  model.py  device.py
  augmentation/             noise.py  vae.py
  training/                 train.py  gan_train.py
  evaluation/               evaluate.py  final_report.py  analyze_pca_components.py
  experiments/              full_protocol_augmentation.py  no2_guard.py
                            compare_augmentation.py  analyze_no2_guard.py
  visualization/            plot_model_architecture.py  plot_pca_k_selection.py
                            plot_pca_vae_math_cards.py  plot_style.py
  cli/                      split.py  predict.py
artifacts/
  data/splits/              固定划分与文件哈希
  data/concentration/       浓度组合覆盖与分布
  augmentation/             三种方法的 OOF、预测、图件与汇总
  figures/                  TCN 结构图与 PCA-VAE 讲解图
sensor_array/artifacts/     正问题模型、十折验证指标与骨架对照
docs/                       项目级文档索引与 PCA-VAE 说明
```

各子目录的代码职责见 [`sensor_array/README.md`](sensor_array/README.md)、[`tcn_union/README.md`](tcn_union/README.md) 和 [`docs/README.md`](docs/README.md)。

---

## 7. 限制与适用范围

- 62 条实验不可扩充；不能靠补测、补采或重新采集解决覆盖缺口。
- 正问题只验证恒浓度阶跃；时变浓度、跨器件、跨日期未验证。
- 逆问题的 12 条锁定测试只做描述性统计；结果支持内部可行性判断，不等同于外部盲测。
- 逆问题三种方法都只在 50 条开发集上做 5 折 OOF，结论限定在当前开发集与固定折划分，不外推到新器件或新浓度组合。
- 所有结论仅适用于 H₂ / NH₃ / NO₂ 三种气体和当前采集条件。