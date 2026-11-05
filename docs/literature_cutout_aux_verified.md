# Cutout 与辅助监督：可核验的序列处理相关文献

> 结论先说：我没有找到“气体传感器浓度回归 + 10 秒 cutout + 5 秒辅助头”完全相同的论文。下面这些是方法层面的相邻工作：**序列/时间序列里的时间遮挡或掩码**，以及**序列模型里的中间监督/辅助损失**。它们可以用来支撑“这两类操作在序列建模中都有先例”，但不能说我们的组合已经被某篇论文直接验证。

## 1. Cutout / 时间掩码 / 序列遮挡

| 论文 | 年份 / 出处 | 与我们方法的关系 |
|---|---|---|
| **Improved Regularization of Convolutional Neural Networks with Cutout**<br>Terrance DeVries, Graham W. Taylor | arXiv:1708.04552, 2017 | cutout 方法的源头；在图像上随机挖掉一块。我们的做法是把“挖掉”搬到原始响应时间轴，并用两端线性桥保持连续。 |
| **SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition**<br>Daniel S. Park, William Chan, Yu Zhang, Chung-Cheng Chiu, Barret Zoph, Ekin D. Cubuk, Quoc V. Le | Interspeech 2019，DOI: 10.21437/Interspeech.2019-2680 | 语音序列里的 time masking：直接把连续时间段的特征置零。说明“遮挡连续时间片”是序列任务的标准增强。 |
| **SpliceOut: A Simple and Efficient Audio Augmentation Method**<br>Arjit Jain, Pranay Reddy Samala, Deepak Mittal, Preethi Jyothi, Maneesh Singh | Interspeech 2022 | 在语音/音频序列上把时间或频率片段替换为另一段；和 cutout 同属“局部片段替换”，但替换源是另一个样本。 |
| **TS2Vec: Towards Universal Representation of Time Series**<br>Zhihan Yue, Yujing Wang, Juanyong Duan, Tianmeng Yang, Congrui Huang, Yunhai Tong, Bixiong Xu | AAAI 2022，DOI: 10.1609/aaai.v36i8.20881 | 时间序列上的 timestamp masking + random cropping，用于对比学习。它不是监督式 cutout，但直接证明“时间序列的时间片掩码”是有效增强。 |
| **Time Series Data Augmentation for Deep Learning: A Survey**<br>Qingsong Wen, Liang Sun, Fan Yang, Xiaomin Song, Jingkun Gao, Xue Wang, Huan Xu | IJCAI 2021，DOI: 10.24963/ijcai.2021/631 | 时间序列增强综述，可用于说明裁剪/掩码类增强在时间序列中的位置；它不是专门讲 cutout 的论文。 |

**和我们的差异**：SpecAugment 是置零，SpliceOut 是拼接另一个样本，TS2Vec 多在隐空间做掩码；我们是在原始响应上做“内部片段 + 线性桥”，保留 `t=0` 和终点。因此引用时应说“同类思想”，不要说“同一方法”。

## 2. 辅助监督 / 中间监督 / 辅助损失

| 论文 | 年份 / 出处 | 与我们方法的关系 |
|---|---|---|
| **MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation**<br>Yazan Abu Farha, Juergen Gall | CVPR 2019 | 时序卷积网络的多阶段结构；每个 stage 后都加 loss layer，后一阶段 refinement。和我们的“10 秒主输出 + 5 秒辅助输出”最接近：都是在序列主干的不同阶段/位置加监督。 |
| **Deeply-Supervised Nets**<br>Chen-Yu Lee, Saining Xie, Patrick Gallagher, Zhengyou Zhang, Zhuowen Tu | AISTATS 2015，PMLR 38:562–570 | 在隐藏层加分类器做 deep supervision，说明“中间层也需要直接监督”是通用做法。 |
| **Going Deeper with Convolutions**<br>Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich | CVPR 2015 | GoogLeNet 的 auxiliary classifiers：中间层加辅助分类器，损失权重 0.3，推理时丢弃。和我们的 5 秒辅助头最相似：训练时加权辅助，推理只保留主输出。 |
| **Semi-supervised Multitask Learning for Sequence Labeling**<br>Marek Rei | ACL 2017，DOI: 10.18653/v1/P17-1194 | 序列标注里的 secondary training objective（辅助语言模型目标），证明序列模型可以靠辅助损失受益。 |
| **A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks**<br>Kazuma Hashimoto, Caiming Xiong, Yoshimasa Tsuruoka, Richard Socher | EMNLP 2017，DOI: 10.18653/v1/D17-1206 | 多层多任务序列模型，低层任务作为高层任务的辅助监督；说明序列模型里的分层/辅助监督有先例。 |

## 3. 汇报时怎么用

可以这样讲，比较稳：

1. **cutout 不是我们拍脑袋**：Cutout、SpecAugment、SpliceOut、TS2Vec 都说明“在序列/时间轴上遮挡或替换局部片段”是常见增强；我们是把它落到原始传感器响应上，并用线性桥避免把曲线弄断。
2. **辅助头也不是我们独有**：GoogLeNet 的 auxiliary classifier 是“中间层加辅助损失、推理丢弃”；MS-TCN 是“每个 stage 都加 loss”；Rei/Hashimoto 是“序列任务加辅助目标”。我们的 5 秒辅助头与这些工作的共同点是：**训练时加中间监督，推理只用主输出**。
3. **要主动说明差异**：没有一篇论文是在三气体传感器 10 秒响应上做 cutout + 5 秒辅助；我们的贡献是把这两类方法组合到“早期响应浓度回归”这个具体问题上，并用 OOF/锁定测试验证，而不是声称提出了全新的 cutout 或辅助监督机制。

## 4. 核验说明

以下链接在整理时实际打开过，用于核对标题、作者、年份或 DOI：

- Cutout: <https://arxiv.org/abs/1708.04552>
- SpecAugment: <https://arxiv.org/abs/1904.08779>，DOI: <https://doi.org/10.21437/Interspeech.2019-2680>
- SpliceOut: <https://www.isca-archive.org/interspeech_2022/jain22_interspeech.pdf>
- TS2Vec: <https://doi.org/10.1609/aaai.v36i8.20881>，<https://ojs.aaai.org/index.php/AAAI/article/view/20881>
- MS-TCN: <https://www.openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html>
- Deeply-Supervised Nets: <https://proceedings.mlr.press/v38/lee15a.html>
- Going Deeper with Convolutions: <https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html>
- Semi-supervised Multitask Learning for Sequence Labeling: <https://aclanthology.org/P17-1194/>
- A Joint Many-Task Model: <https://aclanthology.org/D17-1206/>
- Time Series Data Augmentation: A Survey: <https://arxiv.org/abs/2002.12478>，DOI: <https://doi.org/10.24963/ijcai.2021/631>

说明：

- Cutout 原文是图像任务，SpecAugment/SpliceOut 是语音/音频序列，TS2Vec 是时间序列自监督；它们不是“气体传感器”论文。
- MS-TCN 是视频时序卷积，不是传感器时间序列；它的“每个 stage 加 loss”与我们的“5 秒辅助头”是结构类比。
- “Time Series Data Augmentation: A Survey”只作为综述背景，不声称它专门验证 cutout。
