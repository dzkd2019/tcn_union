# 归档文档

本目录保存 2026-11-18 之前的章节与旧模型实验记录。学习文档已迁回 `docs/` 并更新为当前模型，见 `../learning_guide.md` 与 `../beginner_guide.md`。

当前分支已改为**双指数骨架 + 三路独立 MLP 残差**（默认推荐三路独立组合），并删除了旧的一阶灰盒、低秩 B 样条残差、直接岭回归、TCN 与旧对照实验代码及其 artifacts。这里的文档只用于追溯旧结论和写法，其中引用的代码、CSV、figures 路径可能已不存在，不应作为当前实现或复现依据。

当前模型与评估请看：

- [`../model.md`](../model.md)：模型结构、参数与接口；
- [`../evaluation.md`](../evaluation.md)：62 折协议、结果与限制；
- [`../learning_guide.md`](../learning_guide.md)：当前模型的学习版叙事；
- [`../beginner_guide.md`](../beginner_guide.md)：概念双解手册；
- [`../plotting_guide.md`](../plotting_guide.md)：当前五张图的绘图入口。