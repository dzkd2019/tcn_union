# sensor_array 文档索引

`sensor_array/` 负责正问题：给定 H₂ / NH₃ / NO₂ 浓度组合，预测三路传感器 0–60 s 的相对响应曲线。
逆问题的数据增强对比在 `tcn_union/` 与 `artifacts/augmentation/`，不在本目录展开。

## 1. 阅读顺序

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [`model.md`](model.md) | 当前正式模型：双指数骨架 + 三路独立 MLP 的公式、参数、训练与推理 |
| 2 | [`double_exponential_analysis.md`](double_exponential_analysis.md) | 纯双指数骨架的参数化、初值、拟合与逐参数诊断 |
| 3 | [`evaluation.md`](evaluation.md) | 十折验证的曲线结果、逐文件配对、解释与限制 |
| 4 | [`plotting_guide.md`](plotting_guide.md) | 当前图件（fig01/03/07/09/10/12–15）的脚本、函数、数据源与版式 |
| 5 | [`augmentation_comparison_20261127.md`](augmentation_comparison_20261127.md) | 逆问题三种增强方式（PCA-VAE / 传感器模型生成 / 组合使用）的统一评估 |
| 6 | [`no2_preserving_augmentation.md`](no2_preserving_augmentation.md) | 组合使用的动机、冻结策略、协议与结果 |

## 2. 当前结论

- 正问题：十折验证曲线宏 nRMSE，双指数骨架 3.30%，加三路独立 MLP 后 **1.48%**；逐气体 2.02 / 1.01 / 1.40%。
- 逆问题：10 s 前缀、50 条开发集、5 折 OOF、多个 seed 训练取平均、6 个 checkpoint 平均、开发集量程。
  PCA-VAE 宏 3σ 7.472% / MAPE 1.791%；传感器模型生成 7.866% / 1.746%；组合使用 7.062% / 1.739%，宏指标最低。
- 组合使用在 H₂ 上接近传感器模型生成，NH₃/NO₂ 保持 PCA-VAE 的水平；三种方法按绝对指标直接比较。

## 3. 目录约定

```text
sensor_array/
  core/            数据、骨架、MLP、残差、指标
  experiments/     evaluate.py  compare_skeletons.py
  plotting/        model.py  style.py
  artifacts/       models/  evaluation/  comparison/
  figures/model/   正问题图件（PNG + SVG）
  docs/            本目录
```

- 正问题图件只放 `sensor_array/figures/model/`，按 `figXX_name.{png,svg}` 命名。
- 每次评估的预测与指标放 `sensor_array/artifacts/evaluation/`，不把不同模型的结果混在一个文件里。
- 逆问题图件放各自的 `artifacts/augmentation/<run>/figs/`，不放进本目录。
- 图件格式统一 PNG 300 dpi + 同名 SVG，不输出 PDF。
