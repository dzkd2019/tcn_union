# 三气体早期响应浓度预测

本项目使用 H2、NH3、NO2 三路传感器响应的早期前缀，预测一次实验的三种浓度标签。当前实现包含数据校验、实验级固定划分、因果 TCN、Ridge 对照、锁定测试评估、预测 CLI 和结果绘图。

## 当前结论

在预先锁定的 12 个测试实验上，5 秒 TCN 的主种子 macro-NMAE 为 **0.02627**，早期特征 Ridge 为 **0.03422**；10 秒 TCN 为 **0.03072**，略高于 10 秒早期 Ridge 的 **0.03003**。因此，旧方案的结果支持继续验证 5 秒方案，但尚不足以证明外部泛化或满足应用误差要求。数据文件未注明传感器响应与浓度物理单位，本文均保留原始单位，不将浓度标为 ppm。

> **当前分支 `feature/gan-augmentation`**：已回退 cutout（老师指出其违背传感器物理规律），改用条件 GAN 在训练折内扩充 10 秒响应，并重新评估 5 秒辅助监督。新方案的动机、GAN 约束、消融矩阵和验收标准见 [`docs/gan_augmentation_plan.md`](docs/gan_augmentation_plan.md)。cutout 的历史实现保留在 `dev`/旧提交中，本分支不再使用。

## 数据与校验

### 文件格式

`data/0.csv` 至 `data/61.csv` 共 62 次实验，每个文件 299 行、7 列，时间为 0–59.6 秒、采样间隔 0.2 秒（5 Hz），总计 18,538 个采样行。

| 列 | 含义 | 模型使用方式 |
|---|---|---|
| `time` | 时间（秒） | 只用于截取固定前缀 |
| `H2`, `NH3`, `NO2` | 三路传感器响应 | 预测输入 |
| `conc1`, `conc2`, `conc3` | H2、NH3、NO2 浓度标签 | 仅训练/评估监督信号；预测时不读取 |

按实验记录，CSV 起点对应响应开始变化，实验间充分清洗恢复。预测窗口从这个响应起点计算，不包含通气至响应出现的气路或传感延迟。`t=0` 响应仅作为初值，不假定为纯空气基线。

### 输入校验

训练数据加载时会检查表头、7 列数值、有限值、非负且文件内恒定的浓度标签、从 0 起始的严格递增时间，以及 0.2 秒间隔（绝对容差 `1e-6`）。划分生成还检查文件集合、SHA-256、数值重复响应和重复浓度三元组。缺失或新增文件、重复实验、标签变化、采样异常都会报错，不会静默删行、补值或重抽划分。

当前 62 个文件的浓度三元组互不相同。浓度档位及实验数：

- H2：8000/16000/24000/32000，各 22/23/13/4 次。
- NH3：10/20/30/40/50，各 7/14/14/14/13 次。
- NO2：2/4/6/8/10，各 16/12/14/11/9 次。

没有零浓度、单气体实验或重复浓度组合。

## 固定实验级划分

先固定整次实验的归属，再构造其前缀特征；同一文件的时间点不会被拆到不同集合。测试集固定为：

```text
2, 7, 15, 20, 23, 27, 31, 38, 41, 46, 55, 60
```

开发集为其余 50 个文件：

```text
0, 1, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19,
21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 35, 36, 37, 39,
40, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54, 56, 57,
58, 59, 61
```

五折验证的验证文件如下。每折用其余 40 个开发文件训练：

| 折 | 验证文件 ID |
|---|---|
| 0 | `5, 6, 16, 19, 22, 24, 42, 45, 50, 59` |
| 1 | `4, 9, 12, 14, 26, 34, 35, 43, 54, 58` |
| 2 | `0, 13, 18, 21, 32, 33, 40, 47, 52, 53` |
| 3 | `3, 10, 11, 28, 30, 36, 37, 39, 49, 51` |
| 4 | `1, 8, 17, 25, 29, 44, 48, 56, 57, 61` |

测试集仅按三个气体各自的浓度档位覆盖构造，不保证所有测试浓度组合位于开发集联合凸包内部。测试集档位计数为：H2 `4/4/3/1`（8000/16000/24000/32000），NH3 `1/3/3/3/2`（10/20/30/40/50），NO2 `3/2/3/2/2`（2/4/6/8/10）。

固定划分与文件哈希写入 `artifacts/splits/manifest.csv`；行数、采样信息与哈希写入 `artifacts/splits/inventory.csv`。显式清单是唯一划分依据，训练后不得重新抽取测试集。

## 前缀特征与归一化

分别训练 5 秒和 10 秒模型，窗口均包括 `t=0` 和终点：输入长度分别为 26、51 个时间点。每个气体通道以初值 `b_j = R_j(0)` 计算相对变化：

```text
d_j(t) = (R_j(t) - b_j) / max(abs(b_j), 1e-8)
```

两种输入模式用于开发集消融和选型：

- `raw_delta`：按 H2/NH3/NO2 顺序的三路原始响应，再接三路相对变化，共 6 通道。
- `delta`：仅三路相对变化，共 3 通道。

不使用全曲线最大值或末端值，不做居中平滑、未来点插值或自动全曲线起点检测。预测加载器只要求 `time,H2,NH3,NO2` 四列，并在达到目标时间后停止读取响应数据；多出的浓度列不会参与预测。

每折、每时限、每输入模式分别以训练文件拟合逐输入通道均值/总体标准差和逐气体标签 z-score。输入统计跨训练文件及前缀时间点计算；标签统计按训练文件计算。标准差小于 `1e-8` 时取 1。验证集、测试集与预测数据只应用训练所得统计量，不参与拟合。

## TCN 结构

`GasTCN` 接收 `[batch, channels, time]`，输出三个标准化浓度。网络包含：

- 3 个残差块，通道数均为 16，膨胀率依次为 1、2、4。
- 每块含两层 `Conv1d`，卷积核大小 5、stride 1；卷积前仅在左侧补零，保证因果性。
- 每层卷积后为 ReLU 和 Dropout(0.1)。首块用 1×1 卷积投影残差通道，其余残差为 identity；块输出为 `ReLU(主分支 + 残差)`。
- 不使用 BatchNorm 或跨时间归一化。只取最后一个有效时间点，经 `Linear(16,16) → ReLU → Linear(16,3)` 输出。

理论感受野为 `1 + 2 × (5−1) × (1+2+4) = 57` 个采样点，可覆盖完整 10 秒输入。标签反 z-score 后输出原始浓度单位；预测不截断负数，也不夹到训练范围。

## 训练、选型与基线

### TCN 训练

- 输入时限：5 秒、10 秒；模式：`raw_delta`、`delta`，共四种开发集组合。
- 固定五折实验级交叉验证。每 epoch 以验证集三气体原始单位 MAE 分别除以对应训练折浓度极差，再取三气体平均作为 macro-NMAE。任一训练折某气体极差为零则报错。
- 损失：标准化标签上的 `SmoothL1Loss(beta=1, reduction="mean")`。
- 优化器：AdamW，学习率 `1e-3`、权重衰减 `1e-3`、betas `(0.9, 0.999)`；batch size 8；梯度范数裁剪 1；固定学习率；最多 500 epochs，验证分数连续 50 epochs 未改善时停止。
- 训练种子 `20260923`，训练前重置 Python、NumPy、Torch RNG。按折验证分数选择各窗口的输入模式；相同分数时选 `delta`。选择完成后，使用全部 50 个开发文件重训，epoch 数取所选组合五折最佳 epoch 的中位数向上取整。
- 最终每个窗口训练种子 `20260923/20260924/20260925`。主结果预先固定为 `20260923`，其他两种子用于展示随机性，不择优、不集成。默认 CPU float32。

### 对照模型

1. 开发集标签均值预测。
2. 仅用三路 `t=0` 原始响应的 Ridge。
3. 每个前缀窗口的 9 维早期 Ridge 特征：三路初值、三路终点相对初值的增量、三路时间线性最小二乘斜率。

Ridge 的输入与标签均按训练折 z-score，带未惩罚截距；正则化 `alpha` 在 `{0.01, 0.1, 1, 10, 100}` 上用相同五折 macro-NMAE 选择，平分时选较大的 alpha。最终 t=0 Ridge 选得 `alpha=1`；5 秒和 10 秒早期 Ridge 均选得 `alpha=0.01`。

## 指标

- **MAE**：平均绝对误差，保留每个气体的原始单位。
- **RMSE**：平方误差均值的平方根，对较大误差更敏感。
- **MAPE**：每个目标按 `100 × mean(|预测−真实| / |真实|)` 计算，单位为百分比；真实值包含 0 时该气体的 MAPE 记为 `null` 并说明原因。MAPE 按真实浓度归一化，低浓度样本会产生更大的相对权重，接近 0 时敏感；不等同于 NMAE。
- **R²**：测试目标方差非零时计算；方差为零时输出 `null` 和原因。
- **NMAE/NRMSE**：MAE/RMSE 除以该气体全部开发集浓度极差。
- **macro-NMAE**：H2、NH3、NO2 三个 NMAE 的算术平均，避免 H2 较大的数值量级主导综合分数。
- **macro-MAPE**：三个气体 MAPE 的算术平均；任一气体因目标为 0 而未定义时，macro-MAPE 也记为 `null`。
- **配对 bootstrap**：对主种子 TCN 与各基线的逐实验 macro-NMAE 差做 2000 次百分位 bootstrap，随机种子 `20260923`；重采样单位是实验，不是时间点。

## 实测结果

以下为锁定 12 个测试实验、主种子 `20260923` 的逐气体结果。NMAE 分母使用开发集极差（H2 24000、NH3 40、NO2 8）。

| 窗口 | 气体 | MAE | RMSE | R² | NMAE | MAPE (%) |
|---|---|---:|---:|---:|---:|---:|
| 5 秒 | H2 | 715.038 | 904.897 | 0.98594 | 2.979% | 6.0786 |
| 5 秒 | NH3 | 0.6267 | 0.7079 | 0.99660 | 1.567% | 2.7119 |
| 5 秒 | NO2 | 0.2668 | 0.3313 | 0.98609 | 3.334% | 6.3791 |
| 10 秒 | H2 | 613.504 | 766.884 | 0.98990 | 2.556% | 3.8875 |
| 10 秒 | NH3 | 1.0915 | 1.5034 | 0.98465 | 2.729% | 5.8410 |
| 10 秒 | NO2 | 0.3144 | 0.4075 | 0.97895 | 3.930% | 8.3910 |

| 方法 | 5 秒 macro-NMAE | 10 秒 macro-NMAE | 5 秒 macro-MAPE (%) | 10 秒 macro-MAPE (%) |
|---|---:|---:|---:|---:|
| 标签均值 | 0.27176 | 0.27176 | 52.07 | 52.07 |
| t=0 Ridge | 0.14907 | 0.14907 | 28.56 | 28.56 |
| 早期特征 Ridge | 0.03422 | 0.03003 | 7.09 | 6.17 |
| TCN 主种子 | **0.02627** | 0.03072 | **5.06** | **6.04** |
 
MAPE 用于补充比较，不参与模型选型。5 秒下 TCN 的 macro-MAPE 低于早期 Ridge；10 秒下 TCN 也略低，但与 macro-NMAE 的排序不同，反映两种指标对误差和目标值的加权方式不同。

TCN 三种子 macro-NMAE：5 秒 `0.02627 / 0.02476 / 0.02261`；10 秒 `0.03072 / 0.02831 / 0.03188`。5 秒 TCN 相对早期特征 Ridge 的配对差值（TCN−Ridge）为 `−0.00795`，95% 区间 `[−0.01445, −0.00076]`；10 秒为 `+0.00069`，区间 `[−0.00976, +0.01039]`。测试样本少，区间及差值应谨慎解释。

开发集选型的五折平均 macro-NMAE：5 秒 `raw_delta=0.02761`、`delta=0.04582`；10 秒 `raw_delta=0.02710`、`delta=0.04172`。两个时限均选中 `raw_delta`，说明当前模型从绝对初始响应获得了有用信息；这本身不能证明泄漏。全数据探索中，初始 H2/NH3 响应与 NH3 标签的相关系数约为 0.9751/0.9652；但初值与文件编号的相关系数为 0.9932/0.9898，NH3 标签与文件编号为 0.9766，控制文件编号后偏相关降至 0.1971/−0.0264。该关联很可能是采集顺序/批次漂移造成的统计捷径，不能当作物理交叉敏感；需用随机顺序、不同日期/器件的新数据复核。

主种子测试集没有负浓度预测，但有少数预测超出当前已测浓度范围；这些值未被裁剪，不代表模型已验证外推能力。当前记录末端未验证为稳态，因此 5 秒、10 秒相对于 59.6 秒记录窗口缩短约 91.6%、83.2%，不能称为相对稳态响应的加速倍数。

## 当前分支：PCA-VAE + 噪声扩充

本分支删除了 cutout。先尝试 GAN 扩充，但生成质量门和下游 OOF 均未通过；随后改用 **PCA 响应基 + 条件 VAE**，并叠加真实测量噪声，作为最终生成式扩充方案。

阅读顺序：

1. `tcn_union/core/data.py`：CSV 校验、读取、`raw_delta` 特征；
2. `tcn_union/core/model.py`：因果 TCN；
3. `tcn_union/training/train.py`：训练循环、辅助监督、可选 EMA、checkpoint；
4. `tcn_union/augmentation/gan.py`：条件 WGAN-GP，生成 `(浓度标签) -> 10 秒三路响应`；
5. `tcn_union/augmentation/noise.py`：从真实曲线估计测量噪声并生成训练噪声视图；
6. `tcn_union/evaluation/gan_quality.py`：生成质量排雷（平滑度、初值/末端、最近邻、多样性、曲线图）；
7. `tcn_union/augmentation/vae.py`：PCA 响应基 + 条件 VAE，生成物理形态合理的响应曲线；
8. `tcn_union/training/gan_train.py`：clean/aux/noise/生成式/组合模式的五折 OOF 与最终重训；
9. `tcn_union/evaluation/augmentation_evaluate.py`：冻结后锁定测试评估；
10. `tcn_union/visualization/plot_rmse_scatter.py`：逐气体真实值-预测值 RMSE 散点图。

GAN 的弱物理先验只有平滑惩罚、范围惩罚和“条件只取训练集真实浓度三元组”；这些不能证明生成曲线物理正确，所以必须先过生成质量检查，再进入下游 OOF。详细方案见 [`docs/gan_augmentation_plan.md`](docs/gan_augmentation_plan.md)。

GAN 在当前 50 个开发实验上生成质量不合格、下游 OOF 也差于 clean，因此没有采用 GAN。为了兼顾创新性和小样本稳定性，最终采用 **PCA 响应基 + 条件 VAE + 真实测量噪声**：

- 生成质量：平滑度比值 `0.881`（越接近 1 越好），NH3 初值/末端条件相关 `0.971/0.973`；
- 训练集：40 条真实 + 0.1× 真实测量噪声 + 3× PCA-VAE 合成（每折 120 条）；
- seed 23 OOF：clean `0.027097` → `0.024015`；三 seed 平均：clean `0.028348` → `0.023672`；
- 锁定测试集三 seed 集成：clean `0.024807` → `0.017942`，macro-MAPE `4.6476%` → `3.3927%`；
- 逐气体 MAPE / NMAE / 归一化 RMSE 对比图见 [`fig_final_gas_metrics.png`](artifacts/final_vae_hybrid/evaluation/figures/fig_final_gas_metrics.png)；
- aux 在 noise+VAE 上反而变差，最终不保留 aux。

完整对比见 [`docs/vae_augmentation_results.md`](docs/vae_augmentation_results.md)；GAN 质量失败记录见 [`docs/gan_quality_result.md`](docs/gan_quality_result.md)；纯噪声对照见 [`docs/noise_augmentation_results.md`](docs/noise_augmentation_results.md)。

PCA-VAE 的零基础原理与代码逐段讲解见 [`docs/PCA-VAE学习资料.md`](docs/PCA-VAE学习资料.md)。

向老师汇报的方法选择理由、幻灯片结构和口头稿见 [`docs/PCA-VAE汇报提纲.md`](docs/PCA-VAE汇报提纲.md)。

GAN 与 PCA-VAE 的结构框图、过程讲解和可直接照读的口头稿见 [`docs/GAN与PCA-VAE数据增强汇报稿.md`](docs/GAN与PCA-VAE数据增强汇报稿.md)。

从数据和模型定义开始、逐步理解训练代码的学习资料见 [`docs/从零理解GasTCN模型与训练.md`](docs/从零理解GasTCN模型与训练.md)。

代码目录与模块职责见 [`tcn_union/README.md`](tcn_union/README.md)。

GAN / PCA / VAE / PCA+VAE 的生成质量与下游测试对比见 [`docs/生成式增强对比实验.md`](docs/生成式增强对比实验.md)。

PCA 为什么选 6 个主成分、解释方差/重构误差/下游 OOF 证据见 [`docs/PCA主成分数选择证据.md`](docs/PCA主成分数选择证据.md)。

## GPU / ROCm 支持

代码不再写死 CUDA：所有训练/评估命令的 `--device` 默认是 `auto`，会按以下顺序选择：

- 有可用 NVIDIA CUDA → `cuda`；
- 有可用 AMD ROCm → 仍然使用 PyTorch 的 `cuda` 设备名（ROCm 版 PyTorch 沿用这套 API）；
- 都没有 → `cpu`。

### NVIDIA CUDA 机器

```bash
uv sync
uv run python -m tcn_union.training.gan_train ... --device auto
```

### AMD ROCm 机器

```bash
# 默认：ROCm 7.2 + Python 3.14 + torch 2.14.1
bash scripts/setup_rocm.sh

# 运行训练/评估时加 --no-sync，避免 uv 把 ROCm torch 换回 lock 中的版本
uv run --no-sync python -m tcn_union.training.gan_train ... --device auto
```

脚本默认从**南京大学 PyTorch wheel 镜像**安装：

```text
https://mirrors.nju.edu.cn/pytorch/whl/rocm7.2
```

也可以用腾讯镜像或显式指定版本：

```bash
ROCM_INDEX=https://mirrors.tencent.com/pytorch-wheels/whl/rocm7.2 \
TORCH_SPEC=torch==2.14.1+rocm7.2 bash scripts/setup_rocm.sh
```

注意：

- `torch==2.14.1+rocm7.2` 提供 Python 3.14 的 wheel；如果机器只有 ROCm 6.2，需要匹配 Python 3.12 和 `torch==2.5.1+rocm6.2`，不要混用版本；
- 如果系统检测到 `/dev/kfd` 或 `rocm-smi`，但当前 PyTorch 是 CUDA 构建，`--device auto` 会直接报错并提示运行 `scripts/setup_rocm.sh`，不会静默退回 CPU；
- 如果国内 PyPI 也慢，可在 `uv sync` 前设置 `UV_DEFAULT_INDEX=https://mirrors.aliyun.com/pypi/simple`（只影响非 torch 的 Python 包）。

设备检查：

```bash
uv run --no-sync python -c "import torch; print(torch.__version__, torch.version.cuda, torch.version.hip, torch.cuda.is_available())"
```

## 复现命令

从项目根目录运行：

> ROCm 用户：先执行 `bash scripts/setup_rocm.sh`，然后把下面命令中的 `uv run` 全部替换为 `uv run --no-sync`，避免 uv 重新同步并覆盖 ROCm 版 torch。

```bash
# 生成固定清单与 inventory
uv run python -m tcn_union.cli.split --data-dir data --output-dir artifacts/splits

# 最终 10 秒模型：noise + PCA-VAE 五折 OOF + 三 seed 重训
uv run python -m tcn_union.training.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/final_vae_hybrid \
  --modes noise_gan --generator vae \
  --noise-scale 0.1 --synthetic-ratio 3.0 \
  --gan-epochs 300 --max-epochs 500

# 锁定测试集评估与基线比较
uv run python -m tcn_union.evaluation.evaluate --manifest artifacts/splits/manifest.csv --model-dir artifacts/tcn --output-dir artifacts/evaluation --device auto

# 单文件预测示例（10 秒模型将路径中的 5s 改为 10s）
uv run python -m tcn_union.cli.predict --checkpoint artifacts/tcn/5s/seed_20260923.pt --csv data/2.csv

# 重新绘制测试结果图
uv run python -m tcn_union.visualization.plot_results --results-dir artifacts/evaluation --output-dir artifacts/evaluation/figures

# 最终方案逐气体 MAPE / NMAE / 归一化 RMSE
uv run python -m tcn_union.visualization.plot_final_gas_metrics \
  --evaluation artifacts/final_vae_hybrid/evaluation/evaluation.json \
  --output-dir artifacts/final_vae_hybrid/evaluation/figures

# 回归检查
uv run python -m unittest discover -s tests -v
```

GAN 方案的复现命令：

```bash
# 单 seed 四模式五折 OOF（先试点）
uv run python -m tcn_union.training.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_ablation_pilot \
  --modes clean aux gan gan_aux \
  --synthetic-ratio 1.0 --gan-epochs 300 --max-epochs 500 \
  --seed 20260923 --oof-only

# 三 seed 最终重训（确认方案后）
uv run python -m tcn_union.training.gan_train \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_augmentation \
  --modes gan_aux \
  --synthetic-ratio 1.0 --gan-epochs 300 --max-epochs 500

# 冻结后评估 10 秒 GAN 模型
uv run python -m tcn_union.evaluation.augmentation_evaluate \
  --manifest artifacts/splits/manifest.csv \
  --baseline-dir artifacts/tcn \
  --model-dir artifacts/gan_augmentation/gan_aux \
  --output-dir artifacts/gan_evaluation --windows 10 --rmse-plots

# 也可以单独重画逐气体 RMSE 散点图
uv run python -m tcn_union.visualization.plot_rmse_scatter \
  --predictions artifacts/gan_evaluation/test_predictions.csv \
  --output-dir artifacts/gan_evaluation/figures \
  --model candidate --window 10 --seed ensemble
```

预测 CLI 输出 JSON，包括 `seconds`、`H2`、`NH3`、`NO2` 和 `unit: "original"`。模型 checkpoint 保存架构、模式、通道顺序、训练文件哈希、训练统计量、种子、epoch 与软件版本。

## 产物

- `artifacts/splits/manifest.csv`、`inventory.csv`：实验划分、浓度标签和输入文件哈希。
- `artifacts/tcn/`：旧 clean 基线 checkpoint、五折分数与 OOF，用于新方案对照。
- `artifacts/gan_ablation_pilot/`：GAN 四模式单 seed 试点的 OOF 与 selection。
- `artifacts/gan_augmentation/`：确认方案后的最终 GAN checkpoint。
- `artifacts/final_vae_hybrid/`：最终选中的 PCA-VAE + noise 0.1 方案：三 seed checkpoint、OOF、锁定测试指标与 RMSE 散点图。
- `artifacts/final_noise/`：纯 noise 0.1 对照：三 seed checkpoint、OOF 和测试指标。
- `artifacts/gan_quality_mlp/`、`artifacts/gan_quality_cp/`：GAN 质量失败记录。
- `artifacts/gan_evaluation/`：如后续重新启用 GAN，用于冻结测试指标与散点图。
- `artifacts/augmentation_search/summary.csv`：cutout 时代的历史搜索汇总，仅作追溯，不再使用。

## 限制与适用范围

测试集为按浓度档位覆盖构造的 12 次实验，H2=32000 仅有一个测试文件；不等同于外部盲测。62 个文件均为不同浓度组合，无法评估重复性、气体缺失识别、零浓度/单气体情况或跨器件泛化。所有文件都曾用于数据质量审查和初始响应相关性探索，因此测试集也不是完全未查看的外部数据。模型结果目前只支持在这些采集条件下的内部可行性判断；部署前需要用户定义逐气体误差容限，并增加独立日期/器件、重复组合和边界浓度实验。
