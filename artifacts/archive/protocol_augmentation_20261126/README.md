# 四训练协议消融（已归档）

2026-11-26 的四协议（syn / noise_syn / full / gas）结果，已停止扩展。

- `fullrun_syn/`：real + synthetic，无噪声、无边缘
- `fullrun_noise_syn/`：real + noise + synthetic
- `fullrun_full/`：real + noise + synthetic + edge
- `fullrun_gas/`：H2 扩充、NH3/NO2 只用已有标签
- `fullrun_comparison/`：四协议合并指标、配对统计与图

结论：full + PCA-VAE 最好（宏 3σ 7.59%），分层随机效应只在 H2 稳定占优；
详见 `sensor_array/docs/archive/protocol_augmentation_20261126.md`。

后续不再沿训练协议方向扩展，改做正模型与 PCA-VAE 的组合增强。
组合结果：最好 `union_h2_180` 宏 3σ 7.70%，未超过 VAE 7.59%
（p=0.454）；H2 更好、NO2 更差。详见
`sensor_array/docs/combined_augmentation.md`，结果在
`artifacts/augmentation/combined_full180/`。
