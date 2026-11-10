# tcn_union 代码结构

按职责拆分为七个子包，避免所有模块平铺在 `tcn_union/` 根目录。

```text
tcn_union/
├── core/            # 数据、模型、设备等基础能力
│   ├── data.py      # CSV 读取、前缀截断、raw_delta 特征
│   ├── model.py     # 因果 TCN
│   └── device.py    # CUDA / ROCm / CPU 设备选择
├── augmentation/    # 训练数据增强与生成器
│   ├── noise.py     # 真实测量噪声估计与加噪
│   ├── vae.py       # PCA 响应基 + 条件 VAE
│   ├── gan.py       # 条件 WGAN-GP
│   ├── pca_generator.py
│   └── direct_vae.py
├── training/        # 训练入口与训练循环
│   ├── train.py
│   └── gan_train.py # 最终训练入口（五折 OOF / 3 seed）
├── evaluation/      # 评估与误差诊断
│   ├── evaluate.py
│   ├── augmentation_evaluate.py
│   ├── gan_quality.py
│   ├── generative_metrics.py
│   ├── analyze_pca_components.py
│   └── analyze_error_concentration.py
├── visualization/   # 所有绘图脚本
├── experiments/     # 生成式增强对比实验
└── cli/             # 数据划分与单文件预测
```

## 常用入口

```bash
# 最终训练
uv run python -m tcn_union.training.gan_train --help

# 锁定测试评估
uv run python -m tcn_union.evaluation.evaluate --help

# 单文件预测
uv run python -m tcn_union.cli.predict --help

# 数据划分
uv run python -m tcn_union.cli.split --help
```

## import 约定

跨子包使用绝对导入：

```python
from tcn_union.core.data import make_features
from tcn_union.core.model import GasTCN
from tcn_union.augmentation.vae import VAEConfig, fit_vae
from tcn_union.training.train import fit_scalers
```

这样从任意脚本或 notebook 导入时，路径都明确。
