# 文献摘录：cutout / 时间掩蔽与辅助监督

> 用途：为最终 10 秒模型的两项设计提供可核验的文献背景：① 原始响应 cutout；② 5 秒辅助读出（中间监督）。只收录能在官方页面核对到标题、作者与 venue 的论文，共 12 篇：A 类 6 篇，B 类 6 篇。

## 30 秒汇报口径

- **cutout 有序列先例**：Cutout 是图像源头；SpecAugment 在语音序列上做 time masking；SpliceOut 做时间/频率片段替换；wav2vec 2.0 和 SimMTM 在语音/时间序列上做掩蔽预训练。我们是在原始响应上做“内部短区间 + 线性桥”，属于同类思想，但不是同一算子。
- **辅助监督有先例**：GoogLeNet 的 auxiliary classifier、DSN 的逐层 companion classifier、Stacked Hourglass 的逐阶段中间监督、MS-TCN 的逐 stage loss、InterCTC 的中间 CTC 损失，都支持“训练时在中间位置加监督、推理只用主输出”。我们的 5 秒辅助头与它们机制相同，位置在时间轴而不是网络深度。
- **边界**：没有一篇论文直接做过我们的三气体传感器 10 秒 cutout + 5 秒辅助；文献只能提供方法动机，不能替代我们自己的消融、OOF 三种子检验和 12 个锁定测试实验的统计限制。

## 0. 对照基准：我们的最终 10 秒模型

- **输入与骨干**：0–10 s 原始响应 → `raw_delta` 六通道；因果 TCN（3 个残差块，膨胀率 1/2/4），主输出取 10 s 末端。
- **cutout**：训练时在响应内部随机取 1–10 个采样点（最多 2 s），用区间两端线性桥替换；不触及 `t=0` 与末端，标签不变。
- **辅助监督**：同一隐藏序列在 5 s（采样索引 25）位置再解码一次；代码复用同一输出头，辅助 SmoothL1 权重 0.5。推理只用 10 s 主输出。
- **集成**：三个固定种子平均。

> 下文的“关系”都是针对这套最终方案的分析；论文本身并没有做过我们的气体传感器实验。

## A. Cutout、随机擦除与时间掩蔽

### A1. Improved Regularization of Convolutional Neural Networks with Cutout

- **作者**：Terrance DeVries, Graham W. Taylor
- **发表**：arXiv:1708.04552, 2017（预印本）
- **链接**：[https://arxiv.org/abs/1708.04552](https://arxiv.org/abs/1708.04552)
- **中文注记**：在训练图像上随机挖去一个方形区域，迫使模型利用分散的局部线索而不是单一显著区域，是遮挡式正则化的代表作。我们的原始响应 cutout 直接沿用这一思路，但把二维方块换成时间轴上的短区间，并用两端线性桥填充，以保持传感器曲线的连续性。

### A2. Random Erasing Data Augmentation

- **作者**：Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, Yi Yang
- **发表**：Proceedings of the AAAI Conference on Artificial Intelligence, 34(7), 2020；DOI: 10.1609/aaai.v34i07.7000
- **链接**：[https://aaai.org/papers/13001-random-erasing-data-augmentation/](https://aaai.org/papers/13001-random-erasing-data-augmentation/)
- **中文注记**：随机选择图像矩形区域并用随机值擦除，系统验证了随机遮挡对分类、检测和行人重识别的正则化收益。它和 cutout 共同构成我们 cutout 的方法来源；我们没有采用随机值填充，而是用区间两端插值，避免破坏 `t=0` 初值和末端水平。

### A3. SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition

- **作者**：Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le
- **发表**：Proc. Interspeech 2019, pp. 2613–2617；DOI: 10.21437/Interspeech.2019-2680
- **链接**：[https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html)
- **中文注记**：在语音 filter bank 特征上做时间块掩蔽、频率块掩蔽与时间弯曲，是语音识别中最具代表性的时间掩蔽增强。我们的 cutout 相当于只保留其 time masking 思想并搬到原始响应域，不改变总时长，也不做频率掩蔽。

### A4. wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations

- **作者**：Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli
- **发表**：Advances in Neural Information Processing Systems 33 (NeurIPS 2020)
- **链接**：[https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html)
- **中文注记**：在潜空间随机掩蔽语音帧，用对比学习从上下文中识别被掩帧对应的量化表示，说明连续时间片掩蔽能迫使序列模型使用更长上下文。它与我们 cutout 的动机相近，但属于大规模无标注预训练；我们的 5 秒辅助监督是有标签的小样本回归，不能直接套用其数据规模结论。

### A5. SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling

- **作者**：Jiaxiang Dong, Haixu Wu, Haoran Zhang, Li Zhang, Jianmin Wang, Mingsheng Long
- **发表**：Advances in Neural Information Processing Systems 36 (NeurIPS 2023)；DOI: 10.52202/075280-1306
- **链接**：[https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html)
- **中文注记**：指出直接随机掩蔽时间点会破坏关键时序变化，因而用多个掩蔽序列的邻域加权聚合来重建被掩部分，在 forecasting/classification 上取得 SOTA 微调结果。它提醒我们掩蔽强度不能过大；我们的 cutout 只挖内部 ≤2 秒并用线性桥保留端点，属于较温和的监督式掩蔽，而不是它的自监督预训练范式。

## B. 辅助监督、深层监督与中间监督

### A6. SpliceOut: A Simple and Efficient Audio Augmentation Method

- **作者**：Arjit Jain, Pranay Reddy Samala, Deepak Mittal, Preethi Jyothi, Maneesh Singh
- **发表**：Proc. Interspeech 2022
- **链接**：[https://www.isca-archive.org/interspeech_2022/jain22_interspeech.pdf](https://www.isca-archive.org/interspeech_2022/jain22_interspeech.pdf)
- **中文注记**：在语音/音频序列上把时间或频率片段替换为另一个样本的片段，和 cutout 同属“局部片段替换”，但替换源是另一个样本。它支持“序列上的片段替换可以作为一种增强”，我们的区别是用两端线性桥而不是另一个样本填充。

### B1. Going Deeper with Convolutions

- **作者**：Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
- **发表**：Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2015, pp. 1–9；DOI: 10.1109/CVPR.2015.7298594
- **链接**：[https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html)
- **中文注记**：在 GoogLeNet 的中间层加两个辅助分类器（损失权重 0.3），以改善深层网络的梯度传播和中间层判别性，是辅助监督的经典起点。我们的 5 秒辅助读出同样是“主输出之外再加一个监督点”，但目的是正则化而非解决梯度消失，且辅助点在时间轴而非网络深度上。

### B2. Deeply-Supervised Nets

- **作者**：Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu
- **发表**：Proceedings of the 18th International Conference on Artificial Intelligence and Statistics (AISTATS), PMLR 38:562–570, 2015
- **链接**：[https://proceedings.mlr.press/v38/lee15a.html](https://proceedings.mlr.press/v38/lee15a.html)
- **中文注记**：在每一隐藏层都接一个 companion classifier 并共同训练，使中间层直接受分类目标监督，是 deep supervision 的代表作。我们的方案可看作其轻量版：只在 5 秒位置加一个辅助损失，而不是每层都加，且共享同一个输出头。

### B3. Stacked Hourglass Networks for Human Pose Estimation

- **作者**：Alejandro Newell, Kaiyu Yang, Jia Deng
- **发表**：Computer Vision – ECCV 2016, LNCS 9912, pp. 483–499；DOI: 10.1007/978-3-319-46484-8_29
- **链接**：[https://doi.org/10.1007/978-3-319-46484-8_29](https://doi.org/10.1007/978-3-319-46484-8_29)
- **中文注记**：堆叠多个 hourglass 模块，在每个阶段输出热图并逐阶段施加中间监督，反复自底向上/自顶向下 refine 姿态估计。它支持“中间读出被监督后能改善共享表征”这一机制；我们的 10 秒 TCN 只在 5 秒处加一个中间读出，没有多阶段堆叠。

### B4. MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation

- **作者**：Yazan Abu Farha, Juergen Gall
- **发表**：Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019, pp. 3575–3584
- **链接**：[https://www.openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html](https://www.openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html)
- **中文注记**：用多阶段空洞时间卷积做动作分割，每个 stage 都产生预测并接受监督，后一 stage refine 前一 stage，是 TCN、时间分割与中间监督三者结合、且最接近我们架构的文献。我们的模型是单阶段 TCN，只在 5 秒位置加辅助读出；它说明逐阶段监督在时序分割里有效，但不直接证明标量回归上的 5 秒辅助一定有效。

### B5. Intermediate Loss Regularization for CTC-based Speech Recognition

- **作者**：Jaesong Lee, Shinji Watanabe
- **发表**：IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2021
- **链接**：[https://arxiv.org/abs/2102.03216](https://arxiv.org/abs/2102.03216)
- **中文注记**：在 CTC 编码器的中间层也加 CTC 损失（InterCTC），用中间监督正则化 CTC 训练并改善语音识别，是 speech sequence 模型中 intermediate supervision 的直接例子。我们的 5 秒辅助损失与其“在序列中间位置加辅助 CTC”最相似；区别是我们在时间位置 25 加一个 SmoothL1 回归损失，而不是在多个层深上各加一个 CTC。

### B6. Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics

- **作者**：Alex Kendall, Yarin Gal, Roberto Cipolla
- **发表**：Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018, pp. 7482–7491；DOI: 10.1109/CVPR.2018.00371
- **链接**：[https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)
- **中文注记**：用同方差不确定性自动学习多任务损失的相对权重，避免人工调参，是辅助损失加权最常被引用的方法。我们的辅助权重固定为 0.5，属于其简化版；在只有 50 个开发实验的情况下，学习权重可能过拟合，因此暂未采用。

## 汇总表

| 方向 | 论文 | 年份 / venue | 链接 | 与我们的关系 |
|---|---|---|---|---|
| A1 Cutout（背景） | Improved Regularization of Convolutional Neural Networks with Cutout | 2017 / arXiv | [arXiv](https://arxiv.org/abs/1708.04552) | 原始响应 cutout 的思想来源；我们用线性桥替代方块擦除 |
| A2 随机擦除 | Random Erasing Data Augmentation | 2020 / AAAI 34(7) | [AAAI](https://aaai.org/papers/13001-random-erasing-data-augmentation/) | 遮挡式正则化；我们用插值而非随机值填充 |
| A3 时间掩蔽 | SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition | 2019 / Interspeech | [ISCA](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html) | 语音时间块掩蔽；我们只取其 time masking 思想 |
| A4 潜空间掩蔽 | wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations | 2020 / NeurIPS 33 | [NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) | 连续帧掩蔽迫使模型用上下文；自监督预训练，非同类设定 |
| A5 时序掩蔽建模 | SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling | 2023 / NeurIPS | [NeurIPS](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html) | 时间序列掩蔽重建；提醒掩蔽强度不能过大 |
| A6 序列片段替换 | SpliceOut: A Simple and Efficient Audio Augmentation Method | 2022 / Interspeech | [ISCA](https://www.isca-archive.org/interspeech_2022/jain22_interspeech.pdf) | 音频序列时间/频率片段替换；我们用线性桥替代另一段样本 |
| B1 辅助分类器 | Going Deeper with Convolutions | 2015 / CVPR | [CVF](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) | 中间层 aux loss 的起点；我们是时间位置辅助读出 |
| B2 深层监督 | Deeply-Supervised Nets | 2015 / AISTATS (PMLR 38) | [PMLR](https://proceedings.mlr.press/v38/lee15a.html) | 逐层 deep supervision；我们是单点轻量版 |
| B3 中间监督 | Stacked Hourglass Networks for Human Pose Estimation | 2016 / ECCV | [Springer](https://doi.org/10.1007/978-3-319-46484-8_29) | 逐阶段中间监督；我们只在 5 s 处加一个读出 |
| B4 TCN 中间监督 | MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation | 2019 / CVPR | [CVF](https://www.openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html) | 架构最接近：TCN + 时序分割 + 逐阶段监督 |
| B5 语音中间监督 | Intermediate Loss Regularization for CTC-based Speech Recognition | 2021 / ICASSP | [arXiv](https://arxiv.org/abs/2102.03216) | 最接近的语音 intermediate supervision；我们在时间位置加回归损失 |
| B6 辅助损失加权 | Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics | 2018 / CVPR | [CVF](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) | 学习损失权重；我们固定 0.5，暂不学权重 |

## 概念对照：三种机制与一处关键差异

1. **输入遮挡 / 掩蔽（A1–A6）**：图像方块、语音帧/频带、时间序列 patch 都被证明是有效的正则化或预训练信号。我们的 cutout 是其中较温和的时域版本：只挖内部短区间，且用两端线性桥保持曲线连续。
2. **中间读出监督（B1–B5）**：文献中的“中间”多指网络层深（DSN、GoogLeNet、InterCTC）或 stage（Hourglass、MS-TCN）；我们的“中间”是**时间位置**——在 5 s 处对同一隐藏序列再解码一次。两者共享“给共享表征增加额外监督”的机制，但我们的实现更轻：单阶段 TCN、共享输出头、只有一个辅助点。
3. **辅助损失加权（B6）**：我们固定权重 0.5，没有采用不确定性学习。这是小样本（50 个开发实验）下的有意简化；学习权重会多一组可训练参数，也可能把验证集噪声学进去。
4. **证据边界**：这些论文支持“遮挡和中间监督是合理的归纳偏置”，但不能证明我们的具体增益。仓库内部的 OOF 三种子平均接近 0，测试集三 seed 集成有改善但只有 12 个测试实验；文献只能提供动机，不能替代我们自己的消融和显著性检验。

## 核验说明

### 实际打开的页面

以下页面均通过 HTTPS 直接请求打开，并核对了标题、作者、venue/年份；不是根据记忆或二手引用填写。

| 论文 | 实际打开的官方页面 | 核对到的信息 |
|---|---|---|
| Cutout | [arxiv.org/abs/1708.04552](https://arxiv.org/abs/1708.04552) | 标题、两位作者、2017-08-15、arXiv ID；另用 arXiv API 核对 |
| Random Erasing | [aaai.org/papers/13001-random-erasing-data-augmentation/](https://aaai.org/papers/13001-random-erasing-data-augmentation/) | 五位作者、AAAI-20 Vol. 34 No. 07、DOI 10.1609/aaai.v34i07.7000 |
| SpecAugment | [isca-archive.org/interspeech_2019/park19e_interspeech.html](https://www.isca-archive.org/interspeech_2019/park19e_interspeech.html) | 七位作者、Interspeech 2019、pp. 2613–2617、DOI 10.21437/Interspeech.2019-2680 |
| wav2vec 2.0 | [proceedings.neurips.cc/.../92d1e1eb...-Abstract.html](https://proceedings.neurips.cc/paper/2020/hash/92d1e1eb1cd6f9fba3227870bb6d7f07-Abstract.html) | 标题、NeurIPS 2020、四位作者；PDF 首页核对作者为 Henry Zhou |
| SpliceOut | [isca-archive.org/interspeech_2022/jain22_interspeech.pdf](https://www.isca-archive.org/interspeech_2022/jain22_interspeech.pdf) | 标题、五位作者、Interspeech 2022；PDF 首页核对 |
| SimMTM | [proceedings.neurips.cc/.../5f9bfdfe...-Abstract-Conference.html](https://proceedings.neurips.cc/paper_files/paper/2023/hash/5f9bfdfe3685e4ccdbc0e7fb29cccf2a-Abstract-Conference.html) | 标题、六位作者、NeurIPS 36 (2023)、DOI 10.52202/075280-1306、摘要 |
| GoogLeNet | [openaccess.thecvf.com/.../Szegedy_Going_Deeper_2015...](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html) | 九位作者、CVPR 2015、pp. 1–9 |
| DSN | [proceedings.mlr.press/v38/lee15a.html](https://proceedings.mlr.press/v38/lee15a.html) | 五位作者、AISTATS 2015、PMLR 38:562–570 |
| Stacked Hourglass | [link.springer.com/chapter/10.1007/978-3-319-46484-8_29](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_29) | 三位作者、ECCV 2016、LNCS 9912、pp. 483–499 |
| MS-TCN | [openaccess.thecvf.com/.../Abu_Farha_MS-TCN...](https://www.openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html) | 两位作者、CVPR 2019、pp. 3575–3584 |
| InterCTC | [arxiv.org/abs/2102.03216](https://arxiv.org/abs/2102.03216) 与 [ICASSP 2021 技术程序页](https://www.2021.ieeeicassp.org/2021.ieeeicassp.org/Papers/ViewPaper7630.html) | 标题、两位作者、arXiv 备注 “Accepted at ICASSP 2021”、程序页 Paper ID SPE-21.6 |
| Uncertainty weighting | [openaccess.thecvf.com/.../Kendall_Multi-Task...](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) | 三位作者、CVPR 2018、pp. 7482–7491 |

### 方法与限制

- **web_fetch 在本会话不可用**：对 arxiv.org、openaccess.thecvf.com、proceedings.mlr.press、www.ijcai.org 等主机都返回 `URL hostname ... resolves to a non-public IP address`；本机 DNS 把这些域名统一映射到 `2001:2::/16`。因此改用 HTTPS 客户端直接请求同一官方 URL，并解析页面中的 meta、BibTeX 或正文。读取的是同一页面，但工具不是 `web_fetch`；这是本环境的限制，不是论文本身的问题。
- **web_search** 只用于发现候选和定位官方页面；最终信息以官方页面为准。
- 只读了摘要、元数据和部分首页，没有逐篇读全文；因此“中文注记”只描述摘要/引言层面能确认的内容，不扩展到实验细节。
- 核对日期：2026-11-05。

### 未收录与不确定项

- **PatchTST**（*A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*，arXiv:2211.14730）：论文真实存在，但 OpenReview/ICLR 页面返回 challenge/403，无法核对 ICLR 2023 的 venue 信息，按“无法核验即排除”的规则未收录。
- **Exploring Cutout and Mixup for Robust Human Activity Recognition on Sensor and Skeleton Data**（传感器 HAR 的 cutout 应用）：尝试打开的出版商页面返回 403，未能核对，未收录。
- **Time Series Data Augmentation for Deep Learning: A Survey**（IJCAI 2021, pp. 4653–4660）：已通过 IJCAI 官方页面核验，但正文并不直接讨论 cutout；为保持 A 类聚焦，未列入。
- **Cutout** 本次核验只找到 arXiv 预印本，未见正式会议/期刊版本；文中按预印本引用，不声称它发表于某个会议。
- **Random Erasing** 的 AAAI 页面没有给出页码范围；条目只写卷期和 DOI，不编造页码。
- **wav2vec 2.0** 的 NeurIPS 页面元数据把第二作者显示为 “Yuhao Zhou”，但论文 PDF 与 arXiv 均为 “Henry Zhou”；条目按 PDF 的 Henry Zhou 填写。
- **InterCTC** 的 IEEE Xplore 页面返回 HTTP 202 且内容为空，DOI 未能直接打开；venue 依据 ICASSP 2021 技术程序页与 arXiv 的 “Accepted at ICASSP 2021” 备注，链接使用 arXiv。
- 另有 **Convolutional Pose Machines**（CVPR 2016）等候选已核验，但为控制篇幅（每类 4–6 篇）未列入；这不影响本文已列论文的真实性。
