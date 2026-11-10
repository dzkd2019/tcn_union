# GAN 生成质量检查：第一轮结果

## 1. 检查设置

- 数据：50 个开发实验，10 秒前缀（51 个采样点）；
- GAN：条件 WGAN-GP，条件为三气体浓度标签；
- 采样：100 条合成曲线；
- 训练：300 epoch；
- 测试集：未读取；
- 只做生成质量检查，没有进入下游 `gan/gan_aux` OOF。

检查了两个生成器版本：

1. **MLP 直接输出 51 点**：最早的实现；
2. **control-point 生成器**：先生成 11 个控制点，再线性插值到 51 点，试图强制曲线平滑。

## 2. 指标对比

| 指标 | 真实/目标 | MLP 直接输出 | control-point |
|---|---:|---:|---:|
| 平滑度比值（越小越接近真实） | ~1 | **1228.7** | **81.1** |
| 初值 MAE（原始单位，三气体平均） | 0 | 234.5 | 107.9 |
| 末端 MAE（原始单位，三气体平均） | 0 | 485.2 | 109.5 |
| 初值偏差 / 该气体量程 | 0 | 0.039 | 0.028 |
| 末端偏差 / 该气体量程 | 0 | 0.084 | 0.044 |
| 最近邻距离比（合成/真实） | 接近 1 | 1.37 | 2.30 |
| 成对多样性比（合成/真实） | 接近 1 | 0.31 | 0.94 |
| NH3 初值-标签相关（真实=0.966） | 同向、接近 | **0.411** | **−0.075** |
| NH3 末端-标签相关（真实=0.967） | 同向、接近 | **−0.346** | **0.095** |
| H2 末端-标签相关（真实=0.423） | 同向、接近 | 0.137 | −0.164 |
| NO2 末端-标签相关（真实=−0.819） | 同向、接近 | 0.210 | 0.100 |

## 3. 直观检查

- MLP 版本：生成曲线明显高频抖动，像噪声叠加，不像平滑的传感器响应；
- control-point 版本：曲线变平滑了，但大量曲线在中间呈 V 形折线，且不同浓度条件下生成曲线几乎一样，**条件信息没有真正进入生成结果**。

图：

- MLP：[`artifacts/gan_quality_mlp/gan_samples.png`](../artifacts/gan_quality_mlp/gan_samples.png)
- control-point：[`artifacts/gan_quality_cp/gan_samples.png`](../artifacts/gan_quality_cp/gan_samples.png)

## 4. 结论

**第一轮 GAN 没有通过质量门槛，不能进入下游 TCN 实验。**

主要问题：

1. 平滑度仍远离真实曲线；
2. 关键气体（尤其 NH3）的“浓度-响应”条件关系没有学出来；
3. control-point 虽然改善了平滑度和多样性，但把曲线变成了不自然的折线，且标签条件更弱；
4. 50 个开发实验对条件 GAN 来说太少，模式坍塌/条件忽略风险很高。

因此没有跑 `gan` / `gan_aux` 的五折 OOF，也没有任何“GAN 提升指标”的结论。

## 5. 下一步可选修复方向

按代价从低到高：

1. **加强条件机制**：FiLM / projection discriminator，让标签进入生成器和判别器的每一层，而不是只在输入/输出拼接；
2. **条件均值 + 残差 GAN**：先用标签拟合一条条件均值曲线，GAN 只生成残差；这样至少保证标签主趋势；
3. **物理参数化生成器**：用一阶/二阶响应模型生成曲线，GAN 只学参数分布，避免生成任意形状；
4. **增加真实实验**：50 条对 GAN 太少，优先补数据比继续调 GAN 更可靠；
5. 如果仍要做纯 GAN，需要先和老师确认“生成质量”的验收标准（平滑度、初值、稳态、条件相关性、多样性），避免只看下游指标。

## 6. 复现

```bash
# MLP 版本（旧实现，需 checkout 4f3329a 之前的 gan.py）
uv run python -m tcn_union.evaluation.gan_quality \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_quality_mlp \
  --samples 100 --gan-epochs 300

# control-point 版本（当前实现）
uv run python -m tcn_union.evaluation.gan_quality \
  --manifest artifacts/splits/manifest.csv \
  --output-dir artifacts/gan_quality_cp \
  --samples 100 --gan-epochs 300
```
