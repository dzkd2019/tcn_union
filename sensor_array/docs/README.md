# sensor_array 文档索引

本目录只保留当前分支仍在使用的文档。已删除的历史文档（旧 MAP/区间校准、
P1–P6 路线图、旧章节草稿、旧绘图指南）可在 git 历史中找回。

## 1. 先读什么

按这个顺序读，可以从前向模型一路走到逆模型增强：

| 顺序 | 文档 | 内容 |
|---|---|---|
| 1 | [learning_guide.md](learning_guide.md) | 主学习文档：问题、数据、双指数骨架、三路 MLP、62 折结果、代码地图 |
| 2 | [beginner_guide.md](beginner_guide.md) | 概念双解手册：每个概念的“严谨说法 / 通俗说法 / 边界” |
| 3 | [model.md](model.md) | 当前唯一正式模型：双指数 + 三路独立 MLP 的公式、参数、训练 |
| 4 | [double_exponential_analysis.md](double_exponential_analysis.md) | 纯双指数骨架的参数化、初值与拟合细节 |
| 5 | [evaluation.md](evaluation.md) | 62 折 LOGO 的曲线结果、逐文件配对、限制；§8 是逆模型增强实验入口 |
| 6 | [plotting_guide.md](plotting_guide.md) | fig01–fig19 的脚本、函数、数据源与版式 |

> `learning_guide.md` 与 `beginner_guide.md` 中涉及 MAP / Laplace / 共形区间 /
> 传感器降级的章节是历史方法记录；这些代码已删除，当前模型只做前向曲线预测。

## 2. 逆模型增强实验（按时间）

这些文档都是“正模型合成 → 训练 TCN 反演”的实验记录。训练协议消融
（syn/noise_syn/full/gas）已归档到 `archive/protocol_augmentation_20261126.md`，
后续不再沿协议方向扩展；当前主线是正模型与 PCA-VAE 的组合增强
`combined_augmentation.md`。结论是组合与 VAE 打平（7.70% vs 7.59%），
没有超过 VAE；前面的文档用于说明协议演进和负结果。

| 文档 | 时间 | 结论一句话 |
|---|---|---|
| [full_protocol_augmentation.md](full_protocol_augmentation.md) | 11-23 | 完整旧协议下，仅真实 10.79%、PCA-VAE 7.47%、正模型 10.92%、混合 9.10%；正模型单独无增益 |
| [forward_augmentation_20261125.md](forward_augmentation_20261125.md) | 11-25 | 条件 R0、H2-only 与按气体路由；H2 有稳定增益，整体仍打平 VAE |
| [forward_seen_vs_vae.md](forward_seen_vs_vae.md) | 11-24 | 只用已有 40 个浓度标签时，正模型 10.12% 未超过 PCA-VAE 9.26% |
| [pure_forward_augmentation.md](pure_forward_augmentation.md) | 11-25 | 纯正模型 joint10：7.87%，略优于 VAE 7.99%，但分气体不全面占优 |
| [hierarchical_augmentation.md](hierarchical_augmentation.md) | 11-26 | 方案 1 分层随机效应：相对仅真实显著改善，但 full 协议下未超过 VAE；详细结果已并入协议消融文档 |
| [archive/protocol_augmentation_20261126.md](archive/protocol_augmentation_20261126.md) | 11-26 | 四协议消融：full+VAE 7.59% 最强；gas+VAE 7.88% 但 hier 退化；分层只在 H2 稳定占优 |
| [combined_augmentation.md](combined_augmentation.md) | 11-26 | VAE + H2-only 正模型拼接：最好 7.70%，与 VAE 7.59% 打平；H2 更好、NO2 更差 |

## 3. 计划与历史边界

- [plan_20261123.md](plan_20261123.md)：11-23 与导师讨论后的 8 周计划；
  其中骨架对比、62 条总图、σ 散点、增强实验已执行，剩余项以
  `hierarchical_augmentation.md` §10 和最新 handoff 为准。
- 历史 handoff：`docs/handoff/` 只保留最近三份（11-24、11-25、11-26）。
- 已删除：旧 `archive/`（MAP 章节、P1–P6 路线图、旧章节草稿、旧绘图指南）。
  这些内容不再维护；需要时用 `git log -- sensor_array/docs/archive` 查历史。

## 4. 当前结论（2026-11-27）

- 前向曲线：双指数骨架 + 三路独立 MLP，62 折宏 nRMSE 1.48%
  （骨架 3.30%）；H2/NH3/NO2 为 2.02/1.01/1.40%。
- 逆模型增强：full + PCA-VAE 最强（7.59% 宏 3σ，NO2 8.15%）；gas + VAE
  7.88%（NH3 最好 3.89%，NO2 9.73%）；full + hier+VAE 7.94%（H2 最好 7.18%，
  NO2 11.32%）；方案 1 单独 9.11%。分层随机效应只在 H2 稳定占优；joint10
  7.87% 与 VAE 接近，分气体不全面占优。
- 正模型 + VAE 拼接：最好 `union_h2_180` 7.70%，VAE 7.59%（p=0.454）；
  H2 7.94% 优于 VAE 9.51%，NO2 9.55% 差于 8.15%。数据层拼接没有超过
  VAE；按气体路由的 7.06% 是 OOF 选模型，不是单模型结论。
- 锁定 12 条未参与上述任何选择；只允许在最终候选确定后做一次描述性评估。
