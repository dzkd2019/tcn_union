# 项目文档索引

本目录放跨正问题与逆问题的项目级说明；各子项目的详细文档在对应目录内。

## 项目级

- [PCA-VAE 数据增强](PCA-VAE.md)：逆问题保留的三种增强方式之一，含 PCA 响应基、
  条件 VAE、训练协议、结果与限制。
- [逆问题三种方法对比](../sensor_array/docs/augmentation_comparison_20261127.md)：
  PCA-VAE、传感器模型生成、组合使用的统一评估。
- [组合使用](../sensor_array/docs/no2_preserving_augmentation.md)：传感器模型生成与
  PCA-VAE 结合、冻结共享骨干与 NH₃/NO₂ 头、只更新 H₂ 头的做法与结果。
- [评估与绘图规范](../AGENTS.md)：指标定义、必报图件、目录与口径。

## 正问题（sensor_array）

- [模型与数据](../sensor_array/README.md)
- [双指数骨架](../sensor_array/docs/model.md)
- [骨架逐参数分析](../sensor_array/docs/double_exponential_analysis.md)
- [十折验证](../sensor_array/docs/evaluation.md)
- [逐图索引](../sensor_array/docs/plotting_guide.md)

## 复现产物

- [artifacts 目录说明](../artifacts/README.md)
- [正问题模型与评估产物](../sensor_array/artifacts/)
