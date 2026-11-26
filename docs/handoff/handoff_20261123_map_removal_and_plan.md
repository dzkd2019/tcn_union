# Handoff：传感器阵列前向模型清理与下一阶段计划

> 最新交接：完整旧协议四臂对照见
> [`handoff_20261123_full_protocol_augmentation.md`](handoff_20261123_full_protocol_augmentation.md)。
> 本文件保留到“旧协议来源对照”为止的历史记录。

- 日期：2026-11-23
- 当前分支：`experiment/forward-augmented-tcn`（从 `experiment/mlp-double-exponential` 的
  `20538ff` 切出）
- 当前 HEAD：`20538ff refactor(sensor-array): remove MAP evaluation and add next-step plan`
- 未提交：骨架对比、fig12–fig19、正模型增强 TCN、旧协议与完整旧协议增强对照、
  `docs/evaluation.md` §8 等；最新状态见 §8–§9 与
  `handoff_20261123_full_protocol_augmentation.md`，提交后以 `git log -1` 为准
- 远端：`origin` = `git@github.com:dzkd2019/tcn_union.git`

## 1. 导师决定（2026-11-23）

1. 主模型保留**双指数骨架 + 三路独立 MLP 残差**；增加一阶 + MLP、二阶 + MLP、
   两个二阶 + MLP 的对比。
2. 每种气体把 62 条实测曲线与 62 条预测曲线画在同一张图里，不挑选案例；
   当前模型只画主模型，不画纯骨架或旧模型。
3. 曲线误差要画 σ 散点图，并另存每气体中位数/平均数；统计数字不放在散点图里。
4. MAP 反演与区间校准不再需要，删除。
5. 用第 1 项中最好的骨架生成合成数据，训练逆 TCN；对比 real、正模型、
   PCA-VAE、正模型 + PCA-VAE 四种数据方案，必须复用之前 PCA-VAE 的训练协议。

完整计划见 [`sensor_array/docs/plan_20261123.md`](../../sensor_array/docs/plan_20261123.md)。
不要在本文件里重复计划细节。

## 2. 本次已完成

### 2.1 删除 MAP/区间/降级

- 删除 `sensor_array/analysis/` 整个目录：
  `estimation.py`（MAP/Laplace）、`calibration.py`（共形区间）、
  `degradation.py`（少一路传感器）。
- `sensor_array/experiments/evaluate.py` 改为**只做曲线评估**：
  不再计算 MAP、downstream、区间或降级；只写
  `curve_metrics.csv`、`curve_summary.csv`、`predictions.npz` 和全量模型。
- 删除产物：`artifacts/calibration/`、`artifacts/inverse/legacy/evaluation/downstream*.csv`、
  `degradation*.csv`。
- 删除图件：fig02、fig04、fig05、fig06、fig08、fig11 及其绘图函数。
- 删除文档：`docs/汇报_数据问题与MAP实验.md`。
- `paths.py` 删除 `CALIBRATION`；`tests/test_model.py` 删除 MAP 测试。

### 2.2 文档与清单

- `sensor_array/README.md`、`docs/evaluation.md`、`docs/model.md`、
  `docs/plotting_guide.md` 改为前向模型口径。
- `sensor_array/MANIFEST.md` 与 `SHA256SUMS` 按当前 15 个文件重建。
- `learning_guide.md`、`beginner_guide.md` 顶部加了“MAP/区间章节已废弃”横幅；
  正文中的 MAP 章节尚未整节删除，计划中列为文档重写项。
- `docs/硕士课题总结_响应加速与阵列建模.md` 顶部加了修订说明，§4.5/§5.3 的 MAP
  内容待重写。

### 2.3 验证

- `python -m unittest discover -s tests`：67 项通过（1 skip）。
- `python -m unittest sensor_array.tests.test_model`：5 项通过。
- 3 折快速评估通过（只验证流程；全量 curve artifacts 已恢复，未被 3 折覆盖）。
- 当前图件：fig01、fig03、fig07、fig09、fig10（PNG+SVG）。

## 3. 当前模型与文件

- 模型：双指数骨架（69 参数）+ 三路独立 MLP 残差（每路 7→64→64→1，
  合计 14,211 参数）。
- 62 折 LOGO 曲线宏 nRMSE：骨架 3.30%，组合 1.48%；
  H₂/NH₃/NO₂ 组合 2.02/1.01/1.40%。
- 推荐模型：`sensor_array/artifacts/models/mlp_per_gas.json`；
  对照：`double.json`。
- 代码结构：
  - `sensor_array/core/{data,double_exponential,mlp_per_gas,metrics,shapes}.py`
  - `sensor_array/experiments/evaluate.py`
  - `sensor_array/plotting/{model,style}.py`
  - `sensor_array/tests/test_model.py`
- 复现命令：
  ```bash
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.experiments.evaluate \
    --folds 62 --max-iter 600
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m sensor_array.plotting.model
  ```
- 环境：Python 3.14.10（`.venv/bin/python`）；torch 2.14.0+cu130；
  GTX 1650 4 GB 可用；MLP 在 GPU 训练，推理在 CPU。

## 4. 下一步

严格按计划执行，顺序不要乱：

1. **先确认“两个二阶 + MLP”定义**：当前计划按“两个二阶系统并联”实现；
   如果导师指“两个不同的二阶模型分别训练”，先改计划再写代码。
2. 实现 `core/skeletons.py`：一阶、二阶、两个二阶；通用残差类复用现有 MLP，
   不复制四份网络。
3. 四骨架 62 折 1 seed 筛选；前两名 3 seed 复跑；用宏 nRMSE 均值选最佳。
4. 每气体 62 条实测/预测总图（3 张，不挑选案例）。
5. 曲线误差 σ 散点图 + `curve_error_stats.csv`（mean/median/std/p90）。
6. 用最佳骨架 + MLP 生成合成数据，在 50/12 开发/锁定划分、5 折、3 seed 下
   对比 real / forward / PCA-VAE / forward+PCA-VAE；复用 PCA-VAE 训练协议。
7. 更新 `plotting_guide.md`、`MANIFEST.md`、`SHA256SUMS`；重写 learning/beginner
   文档，删掉 MAP 章节；重写硕士课题总结的 §4.5/§5.3。

## 5. 已知坑

- **不要把锁定 12 条用于调参或合成**：逆 TCN 的 50/12 划分只在最终评估时用 12；
  正模型合成必须由每折训练 40 条拟合。
- **R0 与 scaler**：旧闭环初版曾随机配 R0、混用 scaler，导致负结果；新实验必须
  “同浓度同 R0”或只用训练折 R0 分布，所有臂共用真实训练折 scaler。
- **合成臂真实曝光**：1:1 混合会让真实样本曝光减半；必须同时报“等真实曝光”版本。
- **MLP seed 方差**：不要用单 seed 判定骨架优劣；前两名 3 seed。
- **5 s PCA-VAE**：先确认 `seconds` 支持；不能用截断 10 s 曲线冒充 5 s。
- **数据约束**：62 条不可扩，不补测/不补采/不重采；不写 ppm。
- **图件规范**：PNG 300 dpi + 同名 SVG；新增/改图同步 `plotting_guide.md`；
  不同方案图件分目录。

## 6. 建议技能

- `simply-kiss`：先读，保持骨架对比与增强实验的最小实现，避免过度抽象。
- `ml-training-recipes`：实现 62 折/5 折训练与 seed 复现时参考。
- `academic-plotting`：画每气体 62 条总图与 σ 散点图时参考版式。
- `code-review`：提交前按仓库标准与计划做一次 review。

## 7. 交接检查

- [ ] `git status` 干净（提交后；当前有 42 项未提交改动）。
- [x] 67 项测试 + sensor_array 8 项测试通过。
- [x] `sensor_array/docs/plan_20261123.md` 已读，执行到第 4 节。
- [x] 确认“两个二阶”定义后再开始写 `skeletons.py`（按两个二阶系统并联实现）。

## 8. 执行进展（2026-11-23 晚，覆盖上文 §2.3/§3/§4 的过时状态）

- 当前 HEAD：`20538ff refactor(sensor-array): remove MAP evaluation and add next-step plan`；
  骨架对比与 fig12–fig15 尚未提交（`git status` 共 42 项改动）。
- 四骨架实现：`core/skeletons.py`、`core/residual.py`、`experiments/compare_skeletons.py`；
  单元测试 8 项通过。
- 1 seed 四骨架（62 折宏 nRMSE）：双指数 1.478%、一阶 1.481%、
  两个二阶并联 1.634%、二阶 1.766%。
- 前两名 3 seed：双指数 1.5209 ± 0.0154%、一阶 1.5261 ± 0.0233%；
  逐文件配对 p=0.273，差异与 seed 方差同量级。选双指数作数据生成骨架，
  一阶为等价备选；结论“骨架形式不是当前误差主要瓶颈”。
- 新增图 fig12–fig15；当前 9 组 18 个图件、0 PDF。
- MANIFEST/SHA 当时按 49 个文件重建；加入 fig16/17 后为 53 个，加入 fig18 后为
  55 个，加入 fig19 与完整旧协议后为 57 个文件，见 §9。
- `sensor_array/docs/plan_20261123.md` §0.1 已记录执行状态。
- 下一步：按计划第 5 节新建 `experiment/forward-augmented-tcn` 分支，
  做 real / forward / pca_vae / forward+pca_vae × 3 seed × 5 折；
  锁定 12 条只在最终候选上评估一次。
- 交接检查：根 67 项测试 OK（1 skip）、sensor_array 8 项 OK、
  `git diff --check -- '*.py' '*.md'` 干净；未 commit、未 push。

## 9. 正模型增强 TCN 结果（2026-11-23，已完成）

- 分支：`experiment/forward-augmented-tcn`，从 `20538ff` 切出；未 commit、未 push。
- 协议：50 条开发集、5 折（40 训练/10 验证）；四臂
  `real` / `forward` / `pca_vae` / `forward+pca_vae`；3 个 base seed
  20260923/24/25，`seed + 1000 * fold`；5 s 与 10 s；TCN 500 轮上限、
  早停 patience 50，所有臂共用真实训练折 scaler。锁定 12 未参与 OOF。
- 10 s OOF 宏 3σ/MAPE（3 seed mean ± sd，%）：`real` 15.88±1.33 / 4.11±0.41；
  `forward` 18.03±2.73 / 4.44±0.58；`pca_vae` 16.67±0.25 / 4.41±0.12；
  `forward+pca_vae` 19.85±1.47 / 5.03±0.23。
- 5 s OOF 宏 3σ/MAPE：`real` 17.01±1.62 / 4.49±0.50；`forward` 20.49±0.45 / 5.22±0.08；
  `pca_vae` 17.02±1.41 / 4.35±0.21；`forward+pca_vae` 20.35±3.34 / 5.21±0.79。
- 逐 (seed, fold) 15 对、候选减 `real`：`pca_vae` 5 s 宏 3σ 中位 −0.98 pp、10/15 改善、
  p=0.151；10 s +0.53 pp、6/15、p=0.421。`forward` 5 s +3.91 pp、p=0.048；
  `forward+pca_vae` 10 s +3.09 pp、p=0.005。没有增强臂在 3σ 上稳定优于 `real`。
- 双倍真实样本对照 `real_dup`（真实训练折重复一次，n=80）：逐折平均 5 s
  14.24/4.07、10 s 13.25/3.64，优于 `real` 与 `pca_vae`；说明更多真实曝光/更新
  本身能降误差，pca 的 5 s 小幅优势不能单独归因于合成数据。
- 结论：**当前简化协议内**不采用合成增强；但 pca_vae 臂只用了 1× 合成，
  缺旧最终方案的 0.1× 噪声、3× + 边缘 ×4、两套低浓度配置、6 checkpoint 平均
  与裁剪。旧协议同数据单 seed 10 s OOF 为 10.22% 3σ，本轮 pca_vae 为 16.65%、
  real 为 15.16%，旧最终 7.47% 是 2 配置 × 3 seed 集成；因此首轮负结果不能
  否定旧 PCA-VAE 结论，需补旧协议复现臂后再下结论。锁定 12 未进入候选评估，
  `locked_forward.py` 未运行。
- 产物：`artifacts/augmentation/protocols/forward/{folds,oof,summary,real_dup,real_dup_summary}.csv`、
  `predictions.npz`、`run_config.json`；代码 `tcn_union/experiments/forward_augmentation.py`、
  `real_control.py`、`locked_forward.py`；图 `fig16_forward_augmentation`、
  `fig17_forward_augmentation_paired`。

### 9.1 旧训练协议下的增强来源对照（2026-11-23 补做）

- 目的：隔离“增强来源”，不把新特征协议与 1× 合成混在一起。协议为旧 TCN
  训练协议（`three_sigma` 早停、`log,none,log`、逐气体头、`low_conc_power=1.0`、
  500 轮），每折 40 训练，合成总数固定 120（3×）；**不加**噪声、边缘复制、
  两套低浓度配置。四臂 `real` / `vae` / `forward` / `forward+vae`，3 seed × 5 折，
  10 s。脚本 `tcn_union/experiments/old_protocol_augmentation.py`。
- 结果（seed 均值±sd / 3 seed 集成，%）：`real` 15.47±1.73 / 11.70；
  `vae` 14.22±1.12 / 10.98；`forward` 16.93±2.03 / 13.70；
  `forward+vae` **11.40±1.32 / 8.27**。
- 配对（15 个 seed×fold，候选−real）：`vae` 9/15、中位 −0.05、p=0.804；
  `forward` 5/15、+1.61、p=0.107；`forward+vae` 14/15、**−2.42**、
  **p=0.00018**，MAPE 12/15、−0.87、p=0.00085。
- 结论：VAE-only 方向改善但不显著；正模型-only 更差；**正模型 + VAE 才稳定
  改善**。这和旧“组合方案有效”一致，也说明首轮简化负结果不能否定旧结论；
  该对照只到 8.27%，不能等同旧 7.47%；完整旧协议已在 §8.8 / fig19 复刻，
  VAE-only 6 checkpoint 为 7.47%/1.79%，正模型无增益。锁定 12 仍未评估。
- 图 `fig18_old_protocol_augmentation`；产物
  `artifacts/augmentation/protocols/old/{folds,oof,summary}.csv`、`predictions.npz`、
  `run_config.json`。

- 当前图件 13 组 26 文件（fig01、03、07、09、10、12–19），0 PDF；
  MANIFEST/SHA 已按 57 个文件重建并通过 `sha256sum -c`。
- 后续：若要重试合成，需要严格等真实曝光对照、更多 seed，并先解决正模型
  在 40 条训练折上的分布偏移；learning/beginner/硕士总结中 MAP 章节仍未整节重写。
