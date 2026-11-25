# Handoff：纯正模型增强 joint10（2026-11-25）

## 0. 交接状态

- 仓库：/home/mrawa/code/deep-learning/tcn_union
- 分支：experiment/pure-forward-augmentation
- 起点提交：d0a4a25（feat(tcn): evaluate forward-model augmentation variants and routing）
- 本次提交前工作区：5 个已修改文件 + 4 个未跟踪路径（3 个 artifacts 目录、1 个文档）
- 本 handoff 写入后，按用户指令 git commit + git push，然后 shutdown 关机。
- 硬约束：仅 data/0.csv–data/61.csv 共 62 条，不补测/补采；
  TEST_IDS={2,7,15,20,23,27,31,38,41,46,55,60} 只做最终描述性评估；
  .venv/bin/python；PNG 300 dpi + SVG、不出 PDF；浓度写“原始单位”，不写 ppm；
  不改 tcn_union/augmentation/vae.py、tests/test_vae.py、pyproject.toml、uv.lock。

## 1. 本轮目标与结论

用户目标：探索“纯正模型增强”，使 H2/NH3/NO2 三种气体上的 TCN 逆模型
都相对仅真实显著改善，并与 PCA-VAE 对照；最终要求纯正模型三气体都
明显超过 VAE。

结论：**严格目标未达成。**

- 最好的纯正模型方案是 joint10（正模型均值 + 三气体联合残差的
  10 主成分条件采样）。完整旧协议 5 折 × 3 seed × 2 配置、
  6 checkpoint 集成：宏 3σ 7.866%、MAPE 1.746%，略好于 PCA-VAE 的
  7.992%/1.980%，但分气体不全面占优。
- 分气体 3σ：joint10 8.082/4.346/11.169%，PCA-VAE
  10.066/4.334/9.575%，仅真实 11.000/6.165/14.619%（H2/NH3/NO2）。
  H2 方向更好，NH3 持平，NO2 更差。
- joint10−仅真实：宏 12/15 改善，中位 −2.928 百分点，p=0.0151；
  H2 p=0.0554、NH3 p=0.00263、NO2 p=0.0730。三气体方向都改善，
  只有 NH3 达到常规显著。
- joint10−PCA-VAE：宏 8/15，中位 −0.019，p=0.7615；H2 p=0.0946、
  NH3 p=0.8469、NO2 p=0.3591（NO2 中位 +1.857）。没有任何气体
  显著超过 VAE。
- 因此可写成：纯正模型把三气体都拉到优于仅真实；但不能稳定超过
  VAE，NO2 仍需要 VAE 的实验形状多样性。

## 2. 协议（完整旧 TCN 协议）

| 项目 | 设置 |
|---|---|
| 数据 | 62 条，开发 50 条 OOF |
| 折 | 5 折，每折训练 40、验证 10 |
| seed | 20260923 / 20260924 / 20260925 |
| 低浓度配置 | 1.0；0,0.3,1.0 |
| 合成量 | 基础 3×40=120 条 + 边缘复制 |
| 噪声 | 增强臂 0.1×MAD 噪声；仅真实臂干净 |
| TCN | 6→16 因果 TCN，3 残差块，三气体独立头 |
| 目标变换 | H2 log，NH3 原空间，NO2 log |
| 早停 | 验证宏 3σ，patience 50，最多 500 epoch |
| 集成 | 2 配置 × 3 seed = 6 checkpoint，先平均再裁剪 |
| 裁剪 | 开发集逐气体 min/max |
| 锁定 12 | 未参与任何选择或评估 |

3σ = 50 条 OOF 相对误差标准差 × 3；宏平均为三气体算术平均。

## 3. 关键结果表

6 checkpoint 集成：

| 方案 | 宏 3σ / % | 宏 MAPE / % | 中位 / % | p90 / % |
|---|---:|---:|---:|---:|
| 仅真实 | 10.595 | 2.745 | 2.361 | 6.178 |
| PCA-VAE | 7.992 | 1.980 | 1.659 | 4.273 |
| joint5 | 7.956 | 1.896 | 1.391 | 4.203 |
| joint10 | 7.866 | 1.746 | 1.269 | 4.376 |

3 seed 自身均值±sd（非集成）：仅真实 13.411±1.148；
PCA-VAE 9.812±0.406；joint5 9.838±0.585；joint10 9.684±0.908。
joint10 的 seed 方差大于 VAE，宏平均差异小于 seed 波动。

## 4. 负结果：bootstrap 与单 seed 陷阱

- joint10 把联合 PCA 高斯采样换成训练折经验残差 bootstrap：全量 3 seed
  集成宏 3σ 8.826%、MAPE 2.078%，H2/NH3/NO2 8.560/5.678/12.240%；
  比 joint10 差，已弃用，产物保留在 artifacts/joint_boot_full/。
- 16 主成分、协方差收缩 1.0、normal/bootstrap 50/50、逐气体拼接等
  5 折单 seed 均未稳定超过 joint10；未进入 3 seed 全量。
- 单 seed 筛选多次误导（joint5 曾 7.61、bootstrap NO2 曾 8.03），
  后续判断只信 3 seed × 2 配置全量。

## 5. 代码改动

- tcn_union/experiments/full_protocol_augmentation.py：
  - 新增/保留纯正模型臂 forward_prob_joint、forward_prob_joint10；
  - JOINT_SAMPLE = os.environ.get("FORWARD_JOINT_SAMPLE", "normal")，
    支持 normal/bootstrap 两种联合残差采样；
  - 删除未使用的 forward_mix、JOINT_COMPONENTS、JOINT_SHRINK 与
    mix/per-gas 分支；
  - forward_resid / forward_nn 的精确训练 R0 与实验臂保留（已记录负结果）。
- tcn_union/evaluation/final_report.py：按 AGENTS.md §10.2 输出
  fig_abs_error、fig_rel_error、fig_ci95 三张 1×3 图（PNG+SVG）。
- 删除未使用 tcn_union/evaluation/pure_forward_route.py。
- 清理后 1 折 smoke：8.508/2.103，与原 joint10 fold0 一致；py_compile 通过。

## 6. 产物

保留：

- artifacts/joint10_full/：主结果，含 folds.csv、fold_seed.csv、oof.csv、
  summary.csv、paired.csv、predictions.npz、run_config.json；
  report/{real,vae,forward_prob_joint,forward_prob_joint10}/ 各含
  metrics.csv、predictions.csv、run_config.json 与 figs/ 三张图；
- artifacts/pure_forward_full/：joint5 全量与 3 臂报告；
- artifacts/joint_boot_full/：bootstrap 负结果与报告。

已删除所有 screen/smoke 目录（joint10_screen、joint16_screen、
joint_mix_screen、joint_pergas_screen、resid_screen、pure_smoke 等）。

结果文档：sensor_array/docs/pure_forward_augmentation.md。
图索引：docs/plotting_guide_tcn_10s.md 的 10s-F6–F13。

## 7. 复现

主实验：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/mpl .venv/bin/python \
  -m tcn_union.experiments.full_protocol_augmentation \
  --folds 5 --seeds 20260923 20260924 20260925 \
  --configs 1.0 0,0.3,1.0 \
  --arms real vae forward_prob_joint forward_prob_joint10 \
  --real-scaler --output artifacts/joint10_full
```

报告：

```bash
for arm in real vae forward_prob_joint forward_prob_joint10; do
  MPLCONFIGDIR=/tmp/mpl .venv/bin/python \
    -m tcn_union.evaluation.final_report \
    --source artifacts/joint10_full/predictions.npz \
    --output artifacts/joint10_full/report/$arm \
    --arm $arm --window 10 --run-name joint10_full
done
```

bootstrap 负结果：设置 FORWARD_JOINT_SAMPLE=bootstrap，其余相同。

## 8. 验证

- 根 .venv/bin/python -m unittest discover -s tests：67 tests OK（1 skipped）。
- MPLCONFIGDIR=/tmp/mpl .venv/bin/python -m unittest discover -s sensor_array/tests：
  8 tests OK。
- py_compile 新脚本通过；git diff --check -- '*.py' '*.md' 干净。
- sensor_array/SHA256SUMS 57 行校验通过（本轮未改 sensor_array 的
  artifacts/figures，新增 TCN 产物在仓库根 artifacts/）。
- 新增 8 个 report run、每个 6 个图文件，共 48 个新图；均 PNG+SVG。

## 9. 下一步

1. 若必须坚持纯正模型：加入显式实验形状先验，例如按浓度条件建模
   时间常数/延迟的随机效应，或训练一个只生成形状、不生成标签的
   生成器；joint10 已证明仅靠低秩残差不够。
2. 若允许混合：H2 用 joint10、NH3/NO2 用 VAE 的路由已接近可用，
   但属于混合方案，不是纯正模型。
3. 任何最终候选只在锁定 12 条上评估一次，不得用锁定结果回调超参。
4. 若要继续提升 VAE 对照的公平性，可补 5 个 seed 或更多折，
   但必须保持锁定 12 条不参与选择。

