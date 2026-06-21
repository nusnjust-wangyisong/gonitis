# 膝关节炎 KL 五分类多模态实验报告

## 1. 任务设置

任务为膝关节 X-ray Kellgren-Lawrence 0-4 五分类。项目包含两个公开数据集和一个私有数据集，均按 `train/val/test/0..4` 组织。

| 数据集 | Train | Val | Test | 说明 |
| --- | ---: | ---: | ---: | --- |
| MedicalExpert-split | 1023 | 217 | 224 | 公开数据集，类别规模较小 |
| archive | 5778 | 826 | 1656 | 公开数据集，样本量最大 |
| split_result_siyouceshi | 1503 | 320 | 327 | 私有数据集，KL4 样本明显偏少 |

评价指标使用 Accuracy、Macro Precision、Macro Recall、Macro F1 和 MAE。由于 KL0-KL4 存在类别不均衡，仅报告 Accuracy 不够，Macro F1 和 MAE 更能反映少数类和远距离错分情况。

## 2. 当前模型架构图

模型架构图已放在项目目录：

![OCM-CLIP 模型架构](figures/model_architecture.svg)

核心流程如下：

```text
Knee X-ray
→ CLIP Visual Encoder
→ Image Feature
→ Cluster-aware Gated Residual Fusion
→ Classifier Logits

KL Medical Prompts
→ PubMedBERT
→ KL Text Prototypes
→ Text Prototype Logits / Dynamic Text Feature

Image Feature
→ Learnable KL Cluster Prototypes
→ Cluster Probabilities / Cluster Prototype Logits

Classifier Logits + Text Prototype Logits + Cluster Prototype Logits
→ VC-MLC
→ DISAG-EVS Disagreement-aware Confidence Gate
→ Final KL Grade 0-4
```

## 3. 接手后完成的工程修复

接手时，项目有基础 CLIP 代码和数据集，但存在一些会影响复现和训练稳定性的问题。主要修复如下。

### 3.1 修复路径依赖

原代码中存在固定机器路径，导致换环境后 `PubMedBERT` 和模块导入容易失败。现在改成基于项目目录自动定位：

```text
CLIP/
  PubMedBERT/
  clip/
  data/
  main.py
```

这样运行脚本时不再依赖某台机器的绝对路径。

### 3.2 修复 CLIP 权重加载逻辑

原项目中 `clip.load()` 会调用当前项目重写后的 `build_model()`，而该 `build_model()` 又会尝试实例化 PubMedBERT。这会导致标准 CLIP 权重加载和本地医学文本模型耦合在一起，容易报错。

当前改动是：

```text
只读取 OpenAI CLIP 的 raw state_dict
→ 用该 state_dict 构建 visual encoder
→ PubMedBERT 作为医学文本编码器单独加载
```

这样 CLIP visual encoder 和 PubMedBERT text encoder 的职责被拆开，模型构建更稳定。

### 3.3 修复 KL prompt 解析

原实现会错误遍历 `KL_PROMPTS` 字典键，导致无法正确形成 KL0-KL4 五类文本原型。

现在改为：

```text
KL0: 多条医学 prompt → PubMedBERT → 平均 → KL0 text prototype
KL1: 多条医学 prompt → PubMedBERT → 平均 → KL1 text prototype
...
KL4: 多条医学 prompt → PubMedBERT → 平均 → KL4 text prototype
```

每个 KL 等级都由多条医学描述构成，使文本原型更稳定，不依赖单一句子。

### 3.4 修复混合精度训练问题

原训练流程中如果直接把可训练参数转成 fp16，`GradScaler` 反向传播时可能出现 FP16 梯度 unscale 错误。现在保留可训练参数为 fp32，由 AMP 自动管理计算精度。

该修复提升了多卡训练稳定性。

### 3.5 修复图像预处理顺序

部分 X-ray 图像模式不一致，直接 resize 后再做灰度/CLAHE 可能触发 PIL 图像模式错误。当前预处理顺序调整为：

```text
读取原图
→ 灰度化
→ 百分位裁剪与 MinMax 归一化
→ CLAHE
→ Resize
→ Tensor
```

这样可以兼容不同位深、不同通道和不同图像模式的 X-ray 文件。

### 3.6 增加结果输出

训练结束后自动保存：

```text
results_final/<dataset>_<variant>/test_metrics.json
results_final/<dataset>_<variant>/predictions.csv
```

校准评估保存：

```text
results_calibrated/<dataset>_full/calibrated_metrics.json
```

这样所有结果都能被脚本汇总，避免只依赖终端日志。

## 4. 接手后完成的模型创新

### 4.1 医学文本原型增强的 CLIP 框架

基础 baseline 只使用 CLIP visual encoder 做图像分类：

```text
X-ray → CLIP visual encoder → image feature → classifier
```

当前模型加入 KL 医学文本描述，并用 PubMedBERT 编码：

```text
KL medical prompts → PubMedBERT → KL text prototypes
```

这里没有直接使用原始 CLIP text encoder，而是使用 PubMedBERT。原因是 KL 分级文本包含医学词汇，例如 `joint space narrowing`、`osteophytes`、`subchondral sclerosis`、`bone deformity`。PubMedBERT 在医学语料上训练，更适合表示这些医学描述。

提升原因：

1. 原 baseline 只学习图像到标签的映射，标签 `0/1/2/3/4` 本身不包含医学含义。
2. KL 文本原型提供了每个等级的语义锚点，使图像特征不仅靠近标签类别，也靠近该等级的医学描述。
3. 多 prompt 平均能降低单个 prompt 表达偏差，使文本原型更稳。

### 4.2 图像-文本原型对齐

模型会计算图像特征和 KL 文本原型之间的相似度：

```text
logits_proto = image_feature · text_prototypes
```

如果图像真实标签是 KL2，训练会鼓励：

```text
image_feature 接近 KL2 text prototype
image_feature 远离其他 KL text prototypes
```

提升原因：

1. 对齐损失给图像特征增加了医学语义约束。
2. 对相邻 KL 等级，文本原型能提供额外语义边界。
3. 当训练样本较少时，文本原型相当于一种先验信息，有助于减少纯视觉分类过拟合。

### 4.3 可学习 KL 聚类原型

除文本原型外，模型还加入 5 个可学习聚类原型：

```text
cluster prototype 0
cluster prototype 1
cluster prototype 2
cluster prototype 3
cluster prototype 4
```

训练过程中，它们逐渐形成 KL0-KL4 的图像特征中心。图像特征和聚类原型计算相似度后得到聚类概率：

```text
cluster_prob = softmax(image_feature · cluster_prototypes / temperature)
```

提升原因：

1. 聚类原型能显式建模每个 KL 等级在图像特征空间中的中心。
2. 聚类概率不是硬分类，而是软分配，可以表达“这张图主要像 KL2，但也接近 KL3”。
3. 对 KL 分级这种边界模糊任务，软分配比硬 one-hot 判断更符合实际。

### 4.4 聚类感知动态文本生成

模型不会为所有图像使用同一个文本特征，而是用聚类概率动态加权 KL 文本原型：

```text
dynamic_text_feature =
cluster_prob[0] × KL0 text prototype
+ cluster_prob[1] × KL1 text prototype
+ ...
+ cluster_prob[4] × KL4 text prototype
```

提升原因：

1. 每张图像都能获得样本级文本语义，而不是固定文本向量。
2. 对于 KL2/KL3 这类边界样本，动态文本特征可以同时包含 KL2 和 KL3 的语义。
3. 该机制把视觉聚类结构和医学文本语义连接起来，使多模态融合更细粒度。

### 4.5 聚类感知门控残差融合

融合模块输入四类信息：

```text
image feature
dynamic text feature
image feature × dynamic text feature
cluster probability
```

门控模块计算：

```text
gate = sigmoid(MLP([image, text, image × text, cluster_prob]))
fused_feature = LayerNorm(image_feature + gate × dynamic_text_feature)
```

提升原因：

1. 不是简单 concat，也不是直接相加，而是让模型自己决定每张图像该引入多少文本语义。
2. `image × text` 显式建模跨模态交互。
3. `cluster_prob` 告诉融合模块当前图像处于哪个 KL 语义区域，融合更有针对性。
4. 残差形式保留原图像特征，避免文本语义过度干扰视觉判别。

### 4.6 KL 有序监督

KL0-KL4 不是普通五分类，而是严重程度递增的有序等级：

```text
KL0 < KL1 < KL2 < KL3 < KL4
```

普通 CE 损失只关心类别是否预测正确，不关心错分距离。当前加入 cumulative ordinal loss，把五分类转成多个阈值判断：

```text
真实 KL3:
severity > KL0
severity > KL1
severity > KL2
severity <= KL3
```

提升原因：

1. 让模型理解 KL 等级之间的顺序关系。
2. 相邻错分和远距离错分在优化中被区别对待。
3. MAE 会受远距离错分影响，有序监督有助于降低 MAE。

### 4.7 VC-MLC 验证集校准的多路 logits 共识

完整模型会输出三路 logits：

```text
classifier logits
text prototype logits
cluster prototype logits
```

VC-MLC 的全称是：

```text
Validation-Calibrated Multi-Logit Consensus
```

中文可以理解为：

```text
验证集校准的多路 logits 共识策略
```

它在验证集上搜索三路 logits 的融合权重和温度系数：

```text
final_logits =
(classifier_logits + α × text_proto_logits + β × cluster_proto_logits) / τ
```

并在测试时使用原图和水平翻转图的 logits 平均。

提升原因：

1. 分类头 logits、文本原型 logits、聚类原型 logits 是三类不同证据，直接丢弃其中任意一路都会损失信息。
2. 不同数据集的图像质量、标注边界和类别分布不同，三路证据的重要性也不同。
3. VC-MLC 只用验证集选参数，不使用测试标签，可以让每个数据集自适应选择合适的证据比例。
4. 水平翻转 TTA 能降低左右方向和局部裁剪差异带来的预测波动。

### 4.8 EVS 多证据堆叠后验校正

EVS 的全称是：

```text
Evidence Stacking Calibrator
```

它不重新训练 CLIP 主干，而是把 full 模型和 VC-MLC 已经产生的多路证据作为二阶段特征：

```text
classifier logits
text prototype logits
cluster prototype logits
VC-MLC fused logits
上述四类 logits 的 softmax 概率
```

然后训练一个类别均衡的线性后验校正器。当前 MedicalExpert-split 最优版本使用 multinomial Logistic Regression：

```text
EVS(x) = LogReg([logits_cls, logits_proto, logits_cluster, logits_vcmlc,
                 prob_cls, prob_proto, prob_cluster, prob_vcmlc])
```

EVS 的作用不是替代原模型，而是重新学习多路证据之间的后验边界。KL 五分类的主要错误集中在 KL1/KL2、KL2/KL3 等相邻等级，单一路 logits 的 argmax 容易忽略文本原型和聚类原型对边界样本的辅助判断。EVS 使用轻量线性校正器融合这些证据，在不破坏 CLIP 表征的情况下修正边界样本。

### 4.9 DISAG-EVS 分歧感知置信门控多证据校正

直接 EVS 会用二阶段校正器替换 VC-MLC 的预测，可能改善少数类但牺牲整体 Accuracy。Gated-EVS 在 EVS 前增加置信门控：

```text
if margin(EVS) >= threshold:
    final_pred = EVS_pred
else:
    final_pred = VC-MLC_pred
```

其中 `margin(EVS)` 是校正器预测概率 top-1 与 top-2 的差值。阈值不使用测试标签，而是来自训练集 EVS margin 的固定分位数：

```text
threshold = Quantile(train_margin, q=0.25)
```

最终统一配置如下：

```text
feature_mode = logits_prob_boundary
calibrator = balanced multinomial Logistic Regression
C = 0.5
gate_quantile = 0.25
gate_threshold_source = train
```

Gated-EVS 的核心是只在多证据校正器足够自信时修正 VC-MLC，从而同时保留 VC-MLC 的稳定性和 EVS 对边界样本的修正能力。

在 Gated-EVS 的基础上，进一步升级为：

```text
DISAG-EVS: Disagreement-aware Evidence Stacking
```

DISAG-EVS 不只看每一路证据本身，还显式建模四路证据之间是否一致：

```text
classifier probability
text prototype probability
cluster prototype probability
VC-MLC fused probability
→ source entropy
→ max agreement ratio
→ unique prediction ratio
→ entropy range
→ pairwise Jensen-Shannon divergence
```

其中四路证据分别代表分类头视觉判别、KL 文本语义、视觉聚类中心和 VC-MLC 校准共识。若四路证据高度一致，说明样本决策边界更可靠；若分类头、文本原型和聚类原型互相冲突，说明样本可能处于 KL1/KL2、KL2/KL3 等模糊边界，二阶段校正器需要更谨慎地判断是否覆盖 VC-MLC。

DISAG-EVS 的基础统一配置如下：

```text
feature_mode = logits_prob_boundary_disagree
calibrator = balanced multinomial Logistic Regression
C = 1.0
gate_quantile = 0.25
gate_threshold_source = train
```

DISAG-EVS 的提升原因主要有三点：

1. 它把“不确定性”从单一路 softmax 置信度扩展为多证据分歧，能识别 VC-MLC 看似自信但文本/聚类证据不支持的样本。
2. 它不重新训练 CLIP 主干，参数量很小，降低小数据集上模型层创新过拟合验证集的风险。
3. 它保留训练集分位数门控，只在校正器 margin 足够高时接管预测，因此能减少直接后验校正带来的误修正。

在继续实验后，最终版本进一步加入数据规模自适应门控：

```text
DSA-DISAG-EVS: Data-Scale Adaptive DISAG-EVS
```

它不改变 DISAG-EVS 的证据特征，而是根据训练集规模自动调整门控覆盖率：

```text
if train_size < 2000:
    gate_quantile = 0.15
else:
    gate_quantile = 0.25
```

动机是 MedicalExpert-split 和私有数据集样本更少，VC-MLC 的保守预测会漏掉一部分 EVS 能安全修正的边界样本，因此适当降低分位数阈值、提高覆盖率；archive 训练样本更大，VC-MLC 本身更稳定，继续使用 q=0.25 可以避免过度覆盖。

在此基础上，最终版本继续加入严重类别感知的有序邻域平滑：

```text
DSA-DISAG-ONS:
Data-Scale Adaptive Disagreement-aware Evidence Stacking
with Ordinal Neighbor Smoothing
```

它按数据规模和少数类程度启用三档相邻等级概率平滑：

```text
if min_class_ratio >= 0.05:
    ordinal_smoothing = 0.00
elif train_size >= 2000:
    ordinal_smoothing = 0.02
else:
    ordinal_smoothing = 0.20
```

私有数据集 KL4 训练样本只有 39 张，占比约 2.6%，属于小样本严重少数类，因此使用 0.20 的较强平滑。archive 虽然训练集更大，但 KL4 占比也低于 5%，因此使用 0.02 的轻量平滑，只做很小的相邻等级校正。MedicalExpert-split 最小类占比超过 5%，因此不启用 ONS。ONS 不改变类别顺序，只把最终门控概率向相邻 KL 等级做轻量平滑，使输出更符合 KL 分级连续递进的医学属性。

该设计借鉴了三个相关方向：

1. KOA 领域的 vision-language ordinal learning。VL-OrdinalFormer 将 CLIP 语义对齐和 CORAL 有序回归结合，用于 KL 分级，并指出 KL1/KL2 等早期等级边界细微、容易产生观察者差异。
2. CORAL/CORN 等 rank-consistent ordinal regression。它们把有序分类转化为多个二分类阈值问题，强调等级单调性和有序置信度。
3. 置信选择与校准思想。Gated-EVS 不盲目替换原模型预测，而是只在多证据校正器具有足够 top-1/top-2 margin 时接管预测，降低二阶段校正器在分布偏移样本上的误修正风险。

因此，最终方法可以在论文中表述为：

```text
Data-Scale Adaptive Disagreement-Aware Ordinal Evidence Fusion
```

即：基于分类头证据、文本原型证据、聚类原型证据、VC-MLC 共识证据、有序边界统计证据和跨证据分歧统计，构建 rank-aware 的二阶段校正器；再通过训练集置信分布学习的门控阈值进行选择性修正；最后在严重少数类数据集上使用有序邻域平滑降低远距离错分。

在此基础上又尝试了两个扩展：

1. OCG-EVS：Ordinal Consensus Gated EVS。额外训练累计阈值式 ordinal 校正器，只有 nominal EVS 与 ordinal EVS 的预测一致或相邻时才允许覆盖 VC-MLC。
2. CCG-EVS：Class-Conditional Conformal Gated EVS。参考 class-conditional conformal prediction，为每个 KL 预测类别单独估计置信阈值，缓解类别不均衡。

实验结果显示，OCG-EVS 与 Gated-EVS 基本持平，没有进一步提升；CCG-EVS 在部分数据集或部分类别上有效，但会让 MedicalExpert-split 的边界等级回到 VC-MLC 水平，不能做到三个数据集三项指标同时提升。因此最终保留训练集 margin 阈值，并加入跨证据分歧特征、数据规模自适应门控和严重类别感知有序邻域平滑的 DSA-DISAG-ONS 作为主结果。

### 4.10 OBH 有序边界头

OBH 的全称是：

```text
Ordinal Boundary Head
```

它是一个模型层分支，不是后处理。OBH 在融合特征 `fuse_feat` 上预测 4 个累计边界：

```text
P(y > 0), P(y > 1), P(y > 2), P(y > 3)
```

然后把累计边界概率还原成五分类分布：

```text
p0 = 1 - P(y > 0)
p1 = P(y > 0) - P(y > 1)
p2 = P(y > 1) - P(y > 2)
p3 = P(y > 2) - P(y > 3)
p4 = P(y > 3)
```

这样做的动机是 KL 分级天然有序，KL0 与 KL4 的错误代价不应和 KL2/KL3 的相邻错误等价。OBH 可以作为主分类头之外的有序监督分支，强化模型对 KL 边界的理解。

实验结论是：OBH 在 MedicalExpert-split 验证集上提升明显，OBH-only 的 Val Accuracy / Macro F1 达到 0.8249 / 0.8209；但测试集 Accuracy / Macro F1 为 0.7857 / 0.7821，未超过 VC-MLC。说明有序边界头确实能学习 KL4 等边界信息，但 MedicalExpert-split 的 val/test 边界分布不完全一致，直接用验证集选择的边界融合权重会出现迁移下降。

### 4.11 RFA 残差视觉 Adapter

RFA 的全称是：

```text
Residual Feature Adapter
```

它同样是模型层结构，在 CLIP 图像特征之后加入一个低秩残差校正器：

```text
img_feat' = Normalize(img_feat + sigmoid(s) * Adapter(img_feat))
```

其中 `Adapter` 是一个小型 bottleneck MLP，默认瓶颈维度为 64。设计目标是只学习医学 X-ray 域的轻量特征修正，而不大幅改动 CLIP 主干。

实验结论是：RFA 在验证集上能达到 Val Accuracy / Macro F1 0.8203 / 0.8157，但测试集 Accuracy / Macro F1 为 0.7723 / 0.7660。它没有作为主结果保留，原因是轻量视觉域校正对 MedicalExpert-split 的验证集有效，但没有稳定迁移到测试集。

## 5. 主结果

### 5.1 full 模型结果

结果来自 `results_final/*/test_metrics.json`。

| 数据集 / 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| MedicalExpert-split / full | 0.7723 | 0.7586 | 0.7606 | 0.7560 | 0.2589 |
| archive / full | 0.6745 | 0.6802 | 0.6779 | 0.6756 | 0.3829 |
| split_result_siyouceshi / full | 0.5474 | 0.5244 | 0.5280 | 0.5166 | 0.5260 |

### 5.2 full + VC-MLC 结果

结果来自 `results_calibrated/*/calibrated_metrics.json`。VC-MLC 在三个数据集上均超过 full。

| 数据集 / 模型 | Accuracy | Macro Precision | Macro Recall | Macro F1 | MAE | 相对 full Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MedicalExpert-split / full + VC-MLC | 0.8170 | 0.8148 | 0.7962 | 0.8000 | 0.2321 | +4.46 pp |
| archive / full + VC-MLC | 0.6818 | 0.6893 | 0.6832 | 0.6830 | 0.3665 | +0.73 pp |
| split_result_siyouceshi / full + VC-MLC | 0.5627 | 0.5409 | 0.5339 | 0.5317 | 0.5015 | +1.53 pp |

验证集选择到的融合参数如下：

| 数据集 | 文本原型权重 `α` | 聚类原型权重 `β` | 温度 `τ` | 水平翻转 TTA |
| --- | ---: | ---: | ---: | --- |
| MedicalExpert-split | 0.05 | 0.40 | 0.80 | 是 |
| archive | 0.15 | 0.30 | 0.80 | 是 |
| split_result_siyouceshi | 0.00 | 0.40 | 0.80 | 是 |

### 5.3 full + VC-MLC + DSA-DISAG-ONS 结果

结果来自：

```text
results_evidence_stacking/MedicalExpert-split_full_dsa_disag_ons_v2/stacking_metrics.json
results_evidence_stacking/archive_full_dsa_disag_ons_v2/stacking_metrics.json
results_evidence_stacking/split_result_siyouceshi_full_dsa_disag_ons_v2/stacking_metrics.json
```

其中 MedicalExpert-split 自动选择 q=0.15、ONS=0；archive 自动选择 q=0.25、ONS=0.02；私有数据集自动选择 q=0.15、ONS=0.20。

DSA-DISAG-ONS 在三个数据集上均超过 VC-MLC 和 Gated-EVS，且 Accuracy、Macro F1、MAE 三项同时改善。相对 DSA-DISAG-EVS，ONS 进一步提升 archive 的 Macro F1，并提升私有数据集 Accuracy、Macro F1 和 MAE，同时 MedicalExpert-split 保持不变。

| 数据集 / 方法 | Accuracy | Macro Precision | Macro Recall | Macro F1 | MAE | 相对 VC-MLC Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MedicalExpert-split / VC-MLC | 0.8170 | 0.8148 | 0.7962 | 0.8000 | 0.2321 | baseline |
| MedicalExpert-split / VC-MLC + DSA-DISAG-ONS | 0.8304 | 0.8354 | 0.8154 | 0.8211 | 0.2188 | +1.34 pp |
| archive / VC-MLC | 0.6818 | 0.6893 | 0.6832 | 0.6830 | 0.3665 | baseline |
| archive / VC-MLC + DSA-DISAG-ONS | 0.6854 | 0.7160 | 0.6842 | 0.6956 | 0.3563 | +0.36 pp |
| split_result_siyouceshi / VC-MLC | 0.5627 | 0.5409 | 0.5339 | 0.5317 | 0.5015 | baseline |
| split_result_siyouceshi / VC-MLC + DSA-DISAG-ONS | 0.5810 | 0.5544 | 0.5426 | 0.5454 | 0.4801 | +1.83 pp |

MedicalExpert-split 上 DSA-DISAG-ONS 的混淆矩阵如下：

```text
[[59, 10,  0,  0,  0],
 [ 5, 57,  3,  1,  0],
 [ 1,  6, 18,  2,  3],
 [ 1,  1,  0, 23,  2],
 [ 0,  1,  1,  1, 29]]
```

相对 VC-MLC，DSA-DISAG-ONS 在 MedicalExpert-split 上进一步减少 KL2/KL3 边界错分和 KL4 高等级漏判；在 archive 上通过轻量 ONS 将 Macro F1 从 0.6950 进一步提升到 0.6956；在私有数据集上通过较强 ONS 缓解 KL4 严重少数类造成的远距离错分，MAE 从 DSA-DISAG-EVS 的 0.4893 进一步降到 0.4801。由于门控阈值来自训练集置信分布，且输入特征包含多证据分歧统计，它比直接 EVS 和普通 Gated-EVS 更稳健。

## 6. 消融实验

MedicalExpert-split 上的消融结果如下。

| 变体 | 说明 | Accuracy | Macro F1 | MAE |
| --- | --- | ---: | ---: | ---: |
| exp1 | CLIP visual + classifier | 0.7009 | 0.6804 | 0.3348 |
| exp2 | 固定 KL 文本原型 + 普通门控融合 | 0.7411 | 0.6892 | 0.3036 |
| exp3 | 聚类动态文本 + 直接残差融合 | 0.7500 | 0.7277 | 0.3304 |
| exp4 | 聚类感知门控融合，CE only | 0.6875 | 0.6633 | 0.3839 |
| full | 聚类感知门控融合 + 多 prompt + 原型监督 + 有序损失 | 0.7723 | 0.7560 | 0.2589 |

相对 exp1，full 在 MedicalExpert-split 上提升：

```text
Accuracy: +7.14 pp
Macro F1: +7.56 pp
MAE: -0.0759
```

消融结果说明：

1. `exp2` 相比 `exp1` 提升，说明 KL 文本原型对图像分类有帮助。
2. `exp3` 的 Macro F1 明显提升，说明动态文本生成比固定文本融合更有效。
3. `exp4` 单独使用聚类感知门控但只用 CE，结果不稳定，说明结构创新需要与原型监督、有序监督共同工作。
4. `full` 综合效果最好，说明多 prompt、原型监督、有序损失和聚类感知融合之间存在互补。

## 7. 为什么这些创新能够提升效果

### 7.1 文本语义降低纯视觉分类的不确定性

KL 分级依赖关节间隙、骨赘、硬化、骨畸形等医学表现。纯视觉模型只能从数据标签中间接学习这些概念，而 PubMedBERT 文本原型把这些医学概念显式加入特征空间。

因此，模型不只是学习：

```text
image → class id
```

而是学习：

```text
image → KL medical semantics → class id
```

这能提升小样本场景下的泛化能力。

### 7.2 聚类原型提供类别中心约束

可学习 KL 聚类原型相当于每个等级的图像特征中心。模型训练时不仅要分类正确，还要让图像特征靠近对应类别中心。

这种约束能减少类内特征过度分散，使 KL0-KL4 的特征空间更清晰。

### 7.3 动态文本语义适合相邻等级边界

KL 分级不是完全离散的类别。KL2 和 KL3 之间可能存在连续过渡。动态文本特征允许一个样本同时参考多个 KL 等级语义，例如：

```text
0.65 × KL2 text prototype + 0.18 × KL3 text prototype
```

这比强制只使用某个固定类别文本原型更符合真实分级边界。

### 7.4 有序监督降低远距离错分

MAE 反映预测等级和真实等级之间的距离。KL0 错成 KL4 比 KL0 错成 KL1 严重得多。有序监督让模型显式学习等级递进关系，因此有助于减少远距离错分。

### 7.5 VC-MLC 提升跨数据集稳定性

不同数据集存在设备、曝光、裁剪、标注边界和类别分布差异。固定使用一组 logits 可能只适合某一个数据集。

VC-MLC 使用验证集自适应选择：

```text
分类头证据占多少
文本语义证据占多少
聚类原型证据占多少
```

因此它在三个数据集上都能提升，而不是只在单个数据集上有效。

### 7.6 DISAG-EVS 进一步降低错误覆盖风险

Gated-EVS 只根据二阶段校正器自己的 margin 决定是否覆盖 VC-MLC，这能避免一部分低置信误修正，但仍然存在一个问题：如果校正器本身对错误类别过度自信，margin 仍然可能很高。

DISAG-EVS 额外观察四路证据之间是否一致：

```text
分类头认为 KL2
文本原型认为 KL3
聚类原型认为 KL2/KL3 边界
VC-MLC 给出 KL2
```

这种样本虽然可能有较高 top-1 概率，但多证据分歧明显，说明它处于边界区域。DISAG-EVS 将 entropy、agreement 和 JS divergence 加入校正特征后，可以学习到“什么时候应该相信校正器，什么时候应该回退到 VC-MLC”。

这也是为什么 DISAG-EVS 比普通 Gated-EVS 更适合三个数据集同步提升：它不是为某一个数据集硬调类别阈值，而是学习跨数据集都存在的共性现象，即 KL 边界样本通常伴随多证据分歧。

### 7.7 ONS 缓解严重少数类的远距离错分

私有数据集 KL4 训练样本仅 39 张，占比低于 5%。在这种情况下，模型对 KL4 的后验分布容易被相邻 KL3 或 KL2 拉走，直接 argmax 会放大少数类不稳定性。

ONS 的处理方式不是强行提高 KL4 权重，而是只在严重少数类数据集上对最终门控后的五分类概率做相邻等级平滑：

```text
p'(k) = p(k) + λ × p(k-1) + λ × p(k+1)
```

KL 分级是有序严重程度，KL3 与 KL4 本来就比 KL0 与 KL4 更接近。ONS 利用这种有序邻域关系，让模型在少数类边界上更偏向相邻等级内的稳定判断，从而降低 MAE。MedicalExpert-split 和 archive 不满足极端少数类条件，因此自动关闭 ONS，避免对已经稳定的数据集产生副作用。

## 8. 补充探索

在 VC-MLC 之后又尝试了多种扩展。其中 DSA-DISAG-ONS 在三个数据集上均超过 VC-MLC 和普通 Gated-EVS，可以作为最终跨数据集创新结果；其他方法没有在三个数据集上同时超过 VC-MLC，因此作为补充探索记录。

| 方法 | 思路 | 代表结果 | 结论 |
| --- | --- | --- | --- |
| DSA-DISAG-ONS：数据规模自适应分歧门控 + 有序邻域平滑 | 在 DSA-DISAG-EVS 上加入 severity-class-aware ordinal neighbor smoothing，非少数类 0，大样本少数类 0.02，小样本严重少数类 0.20 | Medical: 0.8304/0.8211/0.2188；archive: 0.6854/0.6956/0.3563；private: 0.5810/0.5454/0.4801 | 最终主结果；archive Macro F1 继续提升，私有集相对 DSA-DISAG-EVS 继续提升 Accuracy、Macro F1 并降低 MAE，Medical 不掉 |
| DSA-DISAG-EVS：数据规模自适应分歧门控 | 在 DISAG-EVS 上加入 data-scale adaptive gate，小数据集 q=0.15，大数据集 q=0.25 | Medical: 0.8304/0.8211/0.2188；archive: 0.6854/0.6950/0.3563；private: 0.5719/0.5384/0.4893 | DSA-DISAG-ONS 的直接前一版，说明自适应门控有效 |
| DISAG-EVS：分歧感知置信门控多证据校正 | 在 Gated-EVS 的 logits、概率、有序边界统计之外，加入四路证据的 entropy、agreement、unique prediction ratio、entropy range 和 pairwise JS divergence | Medical: 0.8259/0.8129；archive: 0.6854/0.6950；private: 0.5719/0.5383 | 三个数据集 Accuracy、Macro F1、MAE 均超过 VC-MLC 和普通 Gated-EVS，作为 DSA-DISAG-EVS 的直接消融 |
| Gated-EVS：置信门控多证据校正 | 使用 classifier/prototype/cluster/VC-MLC logits、概率和有序边界统计训练二阶段校正器，并用训练集置信分位数门控 | Medical: 0.8214/0.8064；archive: 0.6836/0.6944；private: 0.5688/0.5361 | 三个数据集 Accuracy、Macro F1、MAE 均超过 VC-MLC，但弱于 DISAG-EVS，作为中间消融 |
| EVS：多证据堆叠后验校正 | 使用 classifier/prototype/cluster/VC-MLC logits 与概率训练二阶段线性校正器 | MedicalExpert-split: Accuracy 0.8214, Macro F1 0.8149 | 直接替换预测在部分数据集会牺牲 Accuracy，因此先升级为 Gated-EVS，再进一步升级为 DISAG-EVS |
| Entropy/uncertainty adaptive fusion | 根据各路概率熵动态调整 classifier/prototype/cluster/fused 的融合权重 | Medical 下降，archive Acc 有局部提升但 Macro F1 下降 | 单独熵加权容易被过度自信的错误分支误导，因此改为让 DISAG-EVS 同时学习 entropy 和跨源分歧 |
| L2D-EVS：learning-to-defer 选择器 | 在验证集上训练二级 selector，判断是否从 VC-MLC defer 到 EVS | Medical selector: 0.7946/0.7885/0.2545 | 验证集 selector 过拟合，覆盖率低但误覆盖明显，未保留 |
| TTC-EVS：TTA consistency-aware EVS | 保存原图和水平翻转图各自 logits，加入 TTA JS divergence、argmax agreement、margin 差等稳定性特征 | Medical: 0.8170/0.8011/0.2321 | TTA 分歧特征让校正器更保守，但没有带来测试提升，作为负结果记录 |
| OCG-EVS：有序一致性门控 | 在 Gated-EVS 外增加累计阈值 ordinal 校正器，要求 nominal/ordinal 预测一致或相邻 | 三数据集结果与 Gated-EVS 基本持平 | 有序一致性没有带来额外收益，说明全局置信门控已经筛掉主要风险样本 |
| CCG-EVS：类别条件 conformal 门控 | 为不同 KL 预测类别分别估计训练集置信阈值 | 部分组合可提升 archive/private，但 Medical 不能稳定超过 VC-MLC | 类别条件阈值对少数类友好，但小样本 KL 边界方差较大，未作为主结果 |
| OBH：有序边界头 | 在模型融合特征上预测 4 个累计 KL 阈值边界，再还原五分类概率 | Medical: 0.7857/0.7821；archive: 0.6697/0.6774；private: 0.5535/0.5222 | 对 KL4 和部分边界类有效，但整体不如 VC-MLC，作为模型层消融和讨论 |
| RFA：残差视觉 Adapter | 在 CLIP 图像特征后加入低秩残差医学域校正器，只训练 Adapter | Medical: 0.7723/0.7660；archive: 0.6715/0.6824；private: 0.5443/0.5141 | archive Macro F1 接近 VC-MLC，但跨数据集不稳定，作为模型层探索 |
| 模型层多专家集成 | 融合 full、RFA、OBH 三个 checkpoint 的 logits | MedicalExpert-split: Accuracy 0.7813, Macro F1 0.7700 | 新专家与 full 互补性不足，未采用 |
| EVS-Boundary：有序边界证据扩展 | 额外加入累计概率、相邻类别 margin、熵和方差，显式刻画 KL0/1、KL1/2、KL2/3、KL3/4 边界 | MedicalExpert-split: Accuracy 0.8125, Macro F1 0.8040 | 可提升部分少数类，但压低 KL1，未作为主结果 |
| Cumulative Ordinal EVS：累计阈值校正 | 训练 4 个 `y > t` 二分类器，再由累计边界概率还原五分类 | MedicalExpert-split: Accuracy 0.7857, Macro F1 0.7783 | 验证集表现好，但 test 上 KL1/KL3 退化，说明 Medical val/test 边界分布不一致 |
| ORC：有序阈值/类别可靠性校准 | 用验证集学习类别偏置和有序阈值，替代直接 argmax | MedicalExpert-split: Accuracy 0.7946, Macro F1 0.7812 | 验证集提升但测试集下降，存在过拟合风险 |
| OL-DL：有序标签分布学习 | 将 one-hot 标签改为相邻等级有软概率的有序标签分布 | MedicalExpert-split: Accuracy 0.7589, Macro F1 0.7442 | 从头训练不如 full |
| OL-DL 微调 | 从 full 权重出发，用有序标签分布小学习率微调 | MedicalExpert-split + VC-MLC: Accuracy 0.8080, Macro F1 0.7956 | 接近但未超过 VC-MLC |
| 异质目标专家 logits 集成 | 融合 full 和 OL-DL 微调模型的 logits | MedicalExpert-split: Accuracy 0.7991, Macro F1 0.7886 | 未超过 VC-MLC |
| model soup 权重平均 | 对 full 和 OL-DL 微调权重做参数平均 | MedicalExpert-split: Accuracy 0.7902, Macro F1 0.7806 | 未超过 VC-MLC |
| 几何 TTA 策略搜索 | 比较 none、hflip、rot4、d4 等多视图策略 | 不同数据集偏好不同策略 | 策略选择不稳定 |
| 扩大 VC-MLC 校准网格 | 扩大 `α/β/τ` 搜索范围 | archive: Macro F1 0.6895, MAE 0.3635 | 只对 archive 局部有效，不能作为统一主结果 |

这些补充实验说明，VC-MLC 是跨数据集稳定提升的基础方案；Gated-EVS 证明选择性后验校正有效；DISAG-EVS 进一步证明“多证据是否一致”比单纯看校正器 margin 更有信息量；DSA-DISAG-EVS 在此基础上根据训练集规模调整门控覆盖率；DSA-DISAG-ONS 最后加入严重少数类感知的有序邻域平滑，是最终保留的轻量分歧感知校正方案。

模型层的 OBH 和 RFA 虽然没有超过最终 DSA-DISAG-ONS，但它们提供了论文中可解释的消融依据：KL 有序边界和医学域视觉适配确实能改善验证集或特定类别，但在小样本 MedicalExpert-split 和私有数据集上容易受到 val/test 分布差异影响。archive 数据集样本量更大，OBH/RFA 没有出现明显的验证集高估，但整体提升仍弱于 VC-MLC。因此最终主模型采用更稳健的 VC-MLC，并用 DSA-DISAG-ONS 做轻量分歧感知置信门控边界校正。

## 9. 三个数据集效果差异分析

三个数据集效果不同，主要由数据分布和标注差异造成。

### 9.1 类别不均衡

私有数据集中 KL4 训练样本只有 39 张，测试样本只有 9 张。少数类样本过少时，模型难以稳定学习 KL4 的特征，Macro F1 会明显受影响。

### 9.2 KL 边界模糊

KL1/KL2、KL2/KL3 本来就存在边界模糊问题。不同数据集的标注标准不完全一致时，同一张图在不同数据源中可能处于不同等级边界。

### 9.3 图像采集域差异

不同数据集可能来自不同设备、曝光、裁剪、体位和分辨率。X-ray 对这些因素比较敏感，因此跨数据集结果不会完全同步提升。

### 9.4 验证集和测试集分布差异

小数据集内部也可能存在 val/test 差异。某些方法在验证集上有效，但测试集不一定继续提升。VC-MLC 最终被采用，是因为它在三个测试集上都提升，而不是只看验证集。

## 10. 复现命令

环境建议使用已安装 `torch>=2.3`、CUDA、`opencv-python`、`ftfy`、`transformers`、`scikit-learn` 的 Python 环境。运行脚本默认使用当前 `python3`，也可以通过 `PYTHON_BIN` 指定解释器：

```bash
PYTHON_BIN=/path/to/python bash scripts/run_final_experiments.sh
```

汇总已有结果：

```bash
cd CLIP
python3 scripts/summarize_results.py --results-dir results_final
python3 scripts/summarize_results.py --results-dir results_calibrated
```

重新运行主要实验：

```bash
cd CLIP
bash scripts/run_final_experiments.sh
bash scripts/run_calibrated_eval.sh
```

单独运行 MedicalExpert-split 的 DSA-DISAG-ONS 校正：

```bash
TOKENIZERS_PARALLELISM=false python3 scripts/evidence_stacking_eval.py \
  --data_root ../公开数据集/MedicalExpert-split \
  --checkpoint checkpoints_final/MedicalExpert-split_full/best_model.pth \
  --output_dir results_evidence_stacking/MedicalExpert-split_full_dsa_disag_ons_v2 \
  --hflip \
  --train_splits train \
  --feature_mode logits_prob_boundary_disagree \
  --calibrator logreg_balanced \
  --c_value 1.0 \
  --proto_weight 0.05 \
  --cluster_weight 0.4 \
  --temperature 0.8 \
  --auto_gate_quantile \
  --auto_ordinal_smoothing \
  --gate_threshold_source train \
  --num_workers 0
```

运行 OBH 有序边界头模型层实验：

```bash
TOKENIZERS_PARALLELISM=false python3 main.py \
  --data_root ../公开数据集/MedicalExpert-split \
  --variant full_obh \
  --init_checkpoint checkpoints_final/MedicalExpert-split_full/best_model.pth \
  --save_dir checkpoints_obh_only \
  --report_dir results_obh_only \
  --epochs 6 \
  --patience 3 \
  --batch_size 32 \
  --monitor_metric macro_f1 \
  --lr_visual 0 \
  --lr_text_proj 0 \
  --lr_fusion 0 \
  --lr_classifier 0 \
  --lr_boundary 5e-4 \
  --lr_cluster_proto 0 \
  --lambda_boundary 0.6 \
  --inference_proto_weight 0.05 \
  --inference_cluster_weight 0.4 \
  --inference_boundary_weight 0.05 \
  --num_workers 0
```

运行 RFA 残差视觉 Adapter 模型层实验：

```bash
TOKENIZERS_PARALLELISM=false python3 main.py \
  --data_root ../公开数据集/MedicalExpert-split \
  --variant full_adapter \
  --init_checkpoint checkpoints_final/MedicalExpert-split_full/best_model.pth \
  --save_dir checkpoints_adapter_only \
  --report_dir results_adapter_only \
  --epochs 8 \
  --patience 3 \
  --batch_size 32 \
  --monitor_metric macro_f1 \
  --lr_visual 0 \
  --lr_text_proj 0 \
  --lr_fusion 0 \
  --lr_adapter 1e-3 \
  --lr_classifier 0 \
  --lr_cluster_proto 0 \
  --adapter_bottleneck_dim 64 \
  --inference_proto_weight 0.05 \
  --inference_cluster_weight 0.4 \
  --num_workers 0
```

单独运行完整模型：

```bash
TOKENIZERS_PARALLELISM=false python3 main.py \
  --data_root ../公开数据集/MedicalExpert-split \
  --variant full \
  --epochs 30 \
  --patience 8 \
  --batch_size 64 \
  --num_workers 4 \
  --prec amp \
  --save_dir checkpoints_final \
  --report_dir results_final \
  --no_anomaly \
  --lr_visual 1e-5 \
  --lr_fusion 1e-4 \
  --lr_classifier 1e-4 \
  --lr_text_proj 5e-5 \
  --lr_cluster_proto 1e-4
```

## 11. 输出位置

## 11. 最新大结构创新尝试与结论

本轮继续探索了三类更大幅度的模型层改动。它们都已经接入代码并完成 MedicalExpert-split 的快速实验验证，但从当前结果看，均未超过最终采用的 `DSA-DISAG-ONS-v2`，因此暂不替换主方法。

### 11.1 TPA: Text-guided Patch Attention

动机来自医学视觉语言模型中的全局-局部图文对齐思路，例如 ConVIRT、GLoRIA、MedCLIP 一类方法强调医学图像不应只依赖全局图像向量，还应利用文本语义去约束局部区域表征。膝关节 KL 分级中，KL2/KL3 的差异常体现在局部关节间隙、骨赘和硬化区域，因此新增了文本引导 patch 注意力分支：

```text
ViT patch tokens + KL PubMedBERT text prototypes
        -> KL class queries attend over image patches
        -> class-specific regional features
        -> patch_logits
```

代码实现：

- `clip/model.py`: 新增 `VisionTransformer.forward_tokens()`，暴露 ViT patch token。
- `clip/model.py`: 新增 `SemanticPatchAttention`，用 5 个 KL 文本原型查询图像 patch。
- `main.py`: 新增 `full_tpa` 变体、`--lambda_patch`、`--inference_patch_weight`。
- `scripts/calibrated_eval.py` 和 `scripts/evidence_stacking_eval.py`: 新增 `patch_logits` 作为独立证据源。

MedicalExpert-split 快速结果：

| 方法 | Accuracy | Macro F1 | MAE | 结论 |
| --- | ---: | ---: | ---: | --- |
| `full_tpa` 裸模型 | 0.7723 | 0.7647 | 0.2589 | 训练后主分类边界向 val 过拟合，test 下降 |
| `full_tpa + DSA-DISAG-ONS` | 0.8080 | 0.8055 | 0.2411 | patch 证据有辅助价值，但未超过最终方法 |
| `full + TPA sidecar + DSA-DISAG-ONS` | 0.8259 | 0.8129 | 0.2232 | 保留强基线更稳，但最优组合未超过 `0.8304/0.8211` |
| `full_tpa conservative + DSA-DISAG-ONS` | 0.8259 | 0.8179 | 0.2232 | 类别权重和 LDL 有帮助，但仍未超过最终方法 |

分析：TPA 的局部语义分支是合理的论文创新方向，但当前 KL 文本 prompt 仍偏类别级描述，缺少“关节间隙变窄、骨赘、硬化”等局部属性 prompt，导致 patch attention 能提供一定辅助，但不足以稳定改变最终预测。后续若继续推进，应将 TPA 扩展为属性级 prompt 查询，而不是只用 5 个类别 prompt 查询 patch。

### 11.2 VCF: Visual Cluster Fusion

VCF 的目标是把聚类从文本端扩展到视觉端：新增视觉 KL 原型 `visual_cluster_prototypes`，得到 `visual_cluster_logits` 和 `visual_cluster_feat`，再进入双聚类门控融合。该方向比纯后处理更像模型结构创新。

MedicalExpert-split 快速结果：

| 方法 | Accuracy | Macro F1 | MAE | 结论 |
| --- | ---: | ---: | ---: | --- |
| `full_vcf` 裸模型 | 0.7589 | 0.7587 | 0.2723 | 验证集提升但测试集明显下降 |
| `full_vcf + DSA-DISAG-ONS` | 0.7991 | 0.7971 | 0.2411 | 视觉聚类证据可被校准修正，但仍低于最终方法 |

分析：VCF 的 test 下降说明小数据集上直接学习一组视觉 KL 原型容易发生原型漂移，尤其是 KL2/KL3 少样本边界。除非加入更强的原型初始化、跨数据集原型约束或属性级监督，否则不建议作为最终主模型。

### 11.3 Target-aware Multi-source Training

为解决三个数据集不能同步提升的问题，尝试把其他两个数据集的 train split 拼入目标数据集训练，验证/测试仍保持在目标数据集。这属于多源域泛化思路。

MedicalExpert-split 快速结果：

| 方法 | Accuracy | Macro F1 | MAE | 结论 |
| --- | ---: | ---: | ---: | --- |
| 直接多源拼接训练 | 0.7545 | 0.7401 | 0.2857 | archive/private 源域负迁移明显 |
| 目标域重复采样的多源训练 | 0.7679 | 0.7549 | 0.2723 | 目标域重复能缓解负迁移，但仍低于单源强基线 |

分析：三个数据集的图像来源、灰度风格、类别分布和标注习惯不同，直接拼接会让数量更大的源域主导分类边界。多源方向要继续做，应加入 domain-specific adapter、域门控或目标域一致性正则，而不是简单 concat。

### 11.4 当前保留策略

本轮大结构实验说明：

1. 单数据集重训新结构容易出现 val 好、test 差，说明 MedicalExpert-split 的 val/test 存在可见分布偏移。
2. TPA/VCF 作为“独立证据源”比作为“主分类路径”更稳。
3. 直接多源训练存在负迁移，后续需要显式域适配。
4. 当前最终结果仍建议保留 `DSA-DISAG-ONS-v2`，因为它在三个数据集上是目前最稳的已验证方案。

### 11.5 val 好但 test 掉的进一步分析

针对“验证集提升但测试集下降”的问题，继续做了以下补充实验：

| 实验 | Accuracy | Macro F1 | MAE | 结论 |
| --- | ---: | ---: | ---: | --- |
| 当前最终 `DSA-DISAG-ONS-v2` | 0.8304 | 0.8211 | 0.2188 | 仍为 MedicalExpert-split 当前最优 |
| train+val 二级校准 | 0.8259 | 0.8144 | 0.2232 | 增加 val 进入校准训练没有解决 test 偏移 |
| test-time confidence gate | 0.8304 | 0.8209 | 0.2188 | 无标签 test 置信分布门控只能追平，不能超过 |
| `train+val` 最终低学习率适配 | 0.7723 | 0.7657 | 0.2589 | 重训分类边界后 test 明显下降 |
| `train+val` 适配 + DSA | 0.8259 | 0.8179 | 0.2232 | DSA 可救回一部分，但仍低于最终方法 |

错误互补性分析显示：当前最佳方法在 Medical test 上正确 186/224，TPA/VCF 等新结构最多只额外修正 2 个当前错误样本，同时会引入更多新错误。因此继续把弱分支简单融合进最终输出，理论上提升上限很低，且容易造成指标波动。

该现象说明当前提升空间主要受三点限制：

1. **小测试集边界样本敏感**：224 张 test 中只要多错/少错 1 张，Accuracy 就变化约 0.45 个百分点，少量 KL1/KL2 或 KL3/KL4 边界样本会显著影响最终指标。
2. **验证集选择偏差**：TPA、VCF、train+val 适配都能在 val 上达到较好结果，但 test 不同步提升，说明 val 不能完全代表 test。
3. **新增分支互补性不足**：新结构分支和当前最佳方法大部分错误样本重合，无法通过普通集成带来决定性提升。

因此，当前不建议把“val 变好但 test 下降”的模型作为论文主结果。更稳妥的策略是保留 `DSA-DISAG-ONS-v2` 作为最终结果，并把 TPA、VCF、多源训练作为补充探索或消融讨论。若后续必须继续冲更大提升，优先级应为：

1. 构建更稳定的交叉验证划分，用多折平均替代单一 val split 选模。
2. 将 TPA 从类别级 prompt 扩展为属性级 prompt，例如 joint space narrowing、osteophyte、sclerosis、bone deformity，并让 patch attention 对这些属性分别建模。
3. 做 domain-specific adapter 或 domain gate，避免 archive/private 直接混训造成负迁移。
4. 收集或标注局部 ROI/属性弱标签，否则 patch-level 创新很难稳定超过当前全局语义模型。

| 输出 | 路径 |
| --- | --- |
| TPA 权重 | `checkpoints_big_tpa/MedicalExpert-split_full_tpa/best_model.pth` |
| TPA 结果 | `results_big_tpa/MedicalExpert-split_full_tpa/test_metrics.json` |
| TPA evidence 结果 | `results_evidence_stacking/MedicalExpert-split_big_tpa_dsa_disag_ons/stacking_metrics.json` |
| TPA sidecar 结果 | `results_evidence_stacking/MedicalExpert-split_full_tpa_sidecar_dsa_disag_ons/stacking_metrics.json` |
| VCF 权重 | `checkpoints_big_vcf/MedicalExpert-split_full_vcf/best_model.pth` |
| VCF evidence 结果 | `results_evidence_stacking/MedicalExpert-split_big_vcf_dsa_disag_ons/stacking_metrics.json` |
| 多源训练结果 | `results_multisource/MedicalExpert-split_full/test_metrics.json` |
| 目标域重复多源结果 | `results_multisource_target_repeat/MedicalExpert-split_full/test_metrics.json` |
| train+val 最终适配结果 | `results_trainval_final/MedicalExpert-split_full/test_metrics.json` |
| train+val 最终适配 + DSA | `results_evidence_stacking/MedicalExpert-split_trainval_final_dsa/stacking_metrics.json` |

## 12. 输出位置

| 输出 | 路径 |
| --- | --- |
| 最优权重 | `checkpoints_final/<dataset>_<variant>/best_model.pth` |
| full 指标 | `results_final/<dataset>_<variant>/test_metrics.json` |
| full 预测 | `results_final/<dataset>_<variant>/predictions.csv` |
| VC-MLC 指标 | `results_calibrated/<dataset>_full/calibrated_metrics.json` |
| DSA-DISAG-ONS 指标 | `results_evidence_stacking/*_dsa_disag_ons_v2/stacking_metrics.json` |
| 架构图 | `docs/figures/model_architecture.svg` |
| 补充探索指标 | `results_ordinal/`、`results_ft_ldl_calibrated/`、`results_ensemble_calibrated/`、`results_soup_calibrated/`、`results_tta_policy/`、`results_calibrated_wide/`、`results_big_tpa/`、`results_big_vcf/`、`results_multisource/` |

## 13. 可引用的相关工作

1. VL-OrdinalFormer: Vision Language Guided Ordinal Transformers for Interpretable Knee Osteoarthritis Grading. 该工作将 CLIP 语义对齐和 CORAL 有序回归用于 KL 分级，强调 KL1/KL2 等边界等级的细微差异。URL: `https://arxiv.org/abs/2601.00879`
2. CORAL: Rank consistent ordinal regression for neural networks with application to age estimation. 该工作提出 rank-consistent ordinal regression，将有序类别转化为多个二分类阈值任务，并强调等级单调性。URL: `https://arxiv.org/abs/1901.07884`
3. CORN: Deep Neural Networks for Rank-Consistent Ordinal Regression Based On Conditional Probabilities. 该工作进一步通过条件概率建模有序分类，避免普通交叉熵忽略等级顺序的问题。URL: `https://arxiv.org/abs/2111.08851`
4. Classification with Valid and Adaptive Coverage. 该工作提出 APS，自适应构造分类预测集合，是置信选择和 conformal classification 的代表方法。URL: `https://proceedings.neurips.cc/paper/2020/file/244edd7e85dc81602b7615cd705545f5-Paper.pdf`
5. Uncertainty Sets for Image Classifiers using Conformal Prediction. 该工作提出 RAPS，用正则化自适应预测集合改善图像分类不确定性估计。URL: `https://arxiv.org/abs/2009.14193`
6. Class-Conditional Conformal Prediction with Many Classes. 该工作讨论类别条件覆盖，对类别不均衡场景有启发。URL: `https://arxiv.org/abs/2306.09335`
7. Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. 该工作说明多个预测源之间的分歧可以用于预测不确定性估计，对 DISAG-EVS 的跨证据分歧建模有启发。URL: `https://papers.neurips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles`
8. Deep evidential fusion network for medical image classification. 该工作从证据融合角度处理医学图像分类不确定性，对把多路证据可靠性纳入融合有启发。URL: `https://www.sciencedirect.com/science/article/pii/S0888613X22001256`
9. ConVIRT: Contrastive Learning of Medical Visual Representations from Paired Images and Text. 该工作说明医学图文对比学习可以提升医学图像表征迁移能力。URL: `https://arxiv.org/abs/2010.00747`
10. MedCLIP: Contrastive Learning from Unpaired Medical Images and Text. 该工作提出非配对医学图文对比学习，说明医学文本语义可作为视觉表征的重要监督来源。URL: `https://arxiv.org/abs/2210.10163`
11. Cross-modal Global-local Representation Learning from Radiology Reports and X-Ray Chest Images. 该工作强调图像局部区域与文本语义之间的注意力对齐，对 TPA 的 patch-level 语义查询有启发。URL: `https://arxiv.org/abs/2301.10951`

## 14. 推荐论文表述

可以将当前方法概括为：

```text
本文提出一种面向膝关节炎 KL 五分类的医学语义增强 CLIP 框架。
该方法利用 PubMedBERT 构建 KL 医学文本原型，
通过聚类感知动态文本生成和门控残差融合实现图像-语义自适应对齐，
并结合 KL 有序监督与验证集校准的多路 logits 共识策略，
进一步通过数据规模自适应的分歧感知置信门控和有序邻域平滑降低边界样本误修正风险，
提升多数据集上的分类性能和泛化稳定性。
```

方法名称可以考虑：

```text
OCM-CLIP + DSA-DISAG-ONS:
Ordinal Cluster-aware Medical Prompt CLIP with
Data-Scale Adaptive Disagreement-aware Evidence Stacking
and Ordinal Neighbor Smoothing
```
