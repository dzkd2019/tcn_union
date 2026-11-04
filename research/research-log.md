# Research Log

## 2026-11-04

- 建立 `feature/model-improvement` 分支。
- 基线：OOF seed 23/24/25 = 0.024592 / 0.031708 / 0.028850；测试三 seed 集成 = 0.018321。
- 锁定假设：EMA 权重平均可降低种子方差；先只跑 seed 24 的 OOF，避免直接消耗测试集。
- 通过标准见 `experiments/ema-seed-robustness/protocol.md`。
