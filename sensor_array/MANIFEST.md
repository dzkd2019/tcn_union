# 结果与图件快照（P6，2026-11-18）

本文件记录 P6 收尾时的 `artifacts/` 与 `figures/` 内容。仓库尚未提交，因此“冻结”指记录当前工作树中的文件清单与 SHA-256，不是 git tag。

## 范围

| 目录 | 文件数 | 说明 |
|---|---:|---|
| `artifacts/analysis/` | 26 | LOGO 估计、共形、残差诊断、P4 协方差/分组/停止/剖面 |
| `artifacts/augmentation/` | 13 | 旧逆模型增强与 P5 公平 B（含 5 个 seed） |
| `artifacts/logo/` | 17 | 六模型 62 折预测、指标与设计表 |
| `artifacts/models/` | 3 | 默认混合模型、结构化残差模型、旧模型 |
| `artifacts/validation/` | 9 | 留一档位/区块、结构化残差、曲线汇总 |
| `figures/` | 52 | 26 个 PNG + 26 个 SVG，无 PDF |

图件编号覆盖 `fig01`–`fig20` 与 `fig22`–`fig27`，共 26 组；`fig21` 留给暂缓的 P2（分层模型未创建），因此没有文件。所有图均为 300 dpi PNG + 同名 SVG。

## 校验

```bash
cd sensor_array
sha256sum -c SHA256SUMS
```

`SHA256SUMS` 共 120 行（68 个 artifacts 文件 + 52 个图件）。重新运行分析或绘图会改变文件字节，届时需重新生成校验和；本快照不替代独立盲测，也不改变 12 条锁定测试未被使用的约定。
