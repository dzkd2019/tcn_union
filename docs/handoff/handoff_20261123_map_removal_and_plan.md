# Handoff：传感器阵列前向模型清理与下一阶段计划

- 日期：2026-11-23
- 分支：`experiment/mlp-double-exponential`
- 本文件提交前 HEAD：`f0e363e feat(sensor-array): add sensor degradation and MAP derivation`
- 本次提交：删除 MAP/区间/降级，加入下一阶段计划；提交后以 `git log -1` 为准
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
- 删除产物：`artifacts/calibration/`、`artifacts/evaluation/downstream*.csv`、
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

- [ ] `git status` 干净（本 handoff 提交后）。
- [ ] 67 项测试 + sensor_array 5 项测试通过。
- [ ] `sensor_array/docs/plan_20261123.md` 已读。
- [ ] 确认“两个二阶”定义后再开始写 `skeletons.py`。
