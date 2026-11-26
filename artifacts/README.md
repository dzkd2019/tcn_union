# artifacts 目录说明

本目录保存 TCN 逆问题、正模型增强和诊断实验的**可复现产物**。  
原则：

- 只保留当前结论需要的 CSV / NPZ / JSON / 模型和图表；
- 图只保留 PNG + SVG，不再新增 PDF；
- 过时、被新协议取代的实验放入 `archive/`，不删除历史证据；
- 新实验按“任务 → 实验族 → 具体运行”放置，避免根目录继续膨胀。

## 目录结构

```text
artifacts/
├── data/                         # 数据本身的事实
│   ├── splits/                   # 50/12 划分、清单
│   └── concentration/            # 浓度覆盖与分布图
│
├── inverse/                      # TCN 逆问题
│   ├── models/                   # 5 s / 10 s 的 3-seed checkpoint 与选择结果
│   ├── eval_10s/                 # 10 s 最终评估（原 3sigma_noridge）
│   ├── eval_5s/                  # 5 s 最终评估（原 3sigma_5s）
│   ├── prefix/                   # 6–9 s 前缀长度探索
│   └── legacy/                   # 旧口径评估，仅作历史对照
│
├── augmentation/                 # 数据增强实验
│   ├── protocols/                # 旧协议 / 完整协议 / forward 旧实现
│   ├── pure_forward/             # 纯正模型增强
│   │   ├── joint10/              # 当前最好的纯正模型方案
│   │   ├── joint_boot/           # bootstrap 负结果
│   │   ├── base/                 # 早期 pure forward
│   │   ├── h2/                   # H2 专用/逐气体路由
│   │   └── routing/              # 路由实验
│   ├── hierarchical/            # 方案 1：分层随机效应合成
│   ├── diagnostics/              # forward_seen / h2 / unseen / 条件残差筛查
│   └── vae_gan/                  # PCA-VAE、GAN、noise 与旧对比
│
├── analysis/                     # 非“训练”类分析
│   ├── error/                    # 误差-浓度诊断
│   ├── ablations/                # 组件消融、逐气体头消融
│   └── closed_loop/              # 闭环/路由诊断
│
├── figures/                      # 报告级图
│   ├── report/                   # 旧汇报图
│   ├── pca_vae/                  # PCA-VAE 讲解图
│   └── architecture/             # TCN 结构图
│
└── archive/                      # 被取代但保留的证据
    ├── augmentation_final/
    ├── augmentation_search/
    ├── improvement/
    └── logs/
```

## 当前结论对应的入口

| 用途 | 目录 / 文件 |
|---|---|
| 数据划分 | `data/splits/manifest.csv` |
| 10 s 逆问题最终评估 | `inverse/eval_10s/` |
| 5 s 逆问题最终评估 | `inverse/eval_5s/` |
| 纯正模型最好方案 | `augmentation/pure_forward/joint10/` |
| 完整旧协议四臂 | `augmentation/protocols/full/` |
| 旧协议（历史对照） | `augmentation/protocols/old/` |
| 方案 1 分层随机效应 | `augmentation/hierarchical/` |
| 正模型诊断 | `augmentation/diagnostics/` |
| PCA-VAE / GAN | `augmentation/vae_gan/` |
| 误差与消融 | `analysis/` |

`joint10` 的最终数字以 `augmentation/pure_forward/joint10/summary.csv` 和
`sensor_array/docs/pure_forward_augmentation.md` 为准；`archive/` 中的数字不再作为结论。

## 路径迁移对照

| 旧路径 | 新路径 |
|---|---|
| `artifacts/splits` | `artifacts/data/splits` |
| `artifacts/data_distribution` | `artifacts/data/concentration` |
| `artifacts/tcn` | `artifacts/inverse/models` |
| `artifacts/3sigma_noridge` | `artifacts/inverse/eval_10s` |
| `artifacts/3sigma_5s` | `artifacts/inverse/eval_5s` |
| `artifacts/prefix` | `artifacts/inverse/prefix` |
| `artifacts/evaluation` | `artifacts/inverse/legacy/evaluation` |
| `artifacts/evaluation_final` | `artifacts/inverse/legacy/evaluation_final` |
| `artifacts/full_protocol_augmentation` | `artifacts/augmentation/protocols/full` |
| `artifacts/old_protocol_augmentation` | `artifacts/augmentation/protocols/old` |
| `artifacts/forward_augmentation` | `artifacts/augmentation/protocols/forward` |
| `artifacts/joint10_full` | `artifacts/augmentation/pure_forward/joint10` |
| `artifacts/joint_boot_full` | `artifacts/augmentation/pure_forward/joint_boot` |
| `artifacts/pure_forward_full` | `artifacts/augmentation/pure_forward/base` |
| `artifacts/h2_augmentation_full` | `artifacts/augmentation/pure_forward/h2` |
| `artifacts/forward_seen_*` | `artifacts/augmentation/diagnostics/forward_seen_*` |
| `artifacts/forward_h2_*` | `artifacts/augmentation/diagnostics/forward_h2_*` |
| `artifacts/forward_unseen_*` | `artifacts/augmentation/diagnostics/forward_unseen_*` |
| `artifacts/cond_*` | `artifacts/augmentation/diagnostics/cond_*` |
| `artifacts/pca_analysis` | `artifacts/augmentation/vae_gan/pca` |
| `artifacts/final_vae_hybrid` | `artifacts/augmentation/vae_gan/final_vae_hybrid` |
| `artifacts/final_noise` | `artifacts/augmentation/vae_gan/final_noise` |
| `artifacts/downstream_comparison` | `artifacts/augmentation/vae_gan/comparisons/downstream` |
| `artifacts/generative_comparison` | `artifacts/augmentation/vae_gan/comparisons/generative` |
| `artifacts/error_analysis` | `artifacts/analysis/error` |
| `artifacts/component_ablation` | `artifacts/analysis/ablations/component` |
| `artifacts/per_gas_head_ablation` | `artifacts/analysis/ablations/per_gas_head` |
| `artifacts/closed_loop` | `artifacts/analysis/closed_loop` |
| `artifacts/report` | `artifacts/figures/report` |
| `artifacts/augmentation_search` | `artifacts/archive/augmentation_search` |
| `artifacts/augmentation_final` | `artifacts/archive/augmentation_final` |
| `artifacts/improvement` | `artifacts/archive/improvement` |

## 本次清理

- 删除 34 个历史 PDF（PNG 保留，按 AGENTS §8 以后只输出 PNG + SVG）；
- 删除空目录与 `__pycache__`；
- 日志统一到 `archive/logs/`；
- 旧评估进入 `inverse/legacy/`，旧协议进入 `augmentation/protocols/`，不再与当前结果并列。
