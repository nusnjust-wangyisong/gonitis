# 膝关节 KL 五分类：双分支联合训练扩展报告（完整版）

本报告是 OCM-CLIP + 异构集成方案（见 report.md）的续篇，聚焦**双分支联合训练与多模态可靠性路由**方向。工程实现上包含四个技术模块：①双分支跨注意力融合（主架构创新）；②多尺度局部特征提取（局部编码器扩展）；③分层热启动训练策略（训练方法创新）；④RSOMR 可靠性感知语义序数多模态路由（文本-视觉多模态创新）。论文写作时建议归纳为三项核心创新：**全局-局部跨注意力双分支融合**、**多尺度局部建模与分层热启动优化**、**RSOMR 结构化文本-视觉多模态证据路由**。前三个工程模块构成强 CNN + ViT full baseline；RSOMR 进一步将 KL 文本原型、相邻等级过渡文本、病例记忆和文本引导专家模型统一为可验证的多模态证据，在三个数据集上均相对 full baseline 取得进一步提升。

---

## 1. 背景与动机

report.md 的主线结论：BiomedCLIP 单分支 + ConvNeXt-V2 后集成构成当前方法的上界。

| 数据集 | BiomedCLIP 单分支（基线） | ConvNeXt-V2 Large | 后集成上界 |
|--------|--------------------------|-------------------|-----------|
| MedicalExpert | 0.8214 | 0.8438 | 0.8661 |
| archive | 0.6781 | 0.7077 | 0.7295 |
| private (ROI) | 0.6239 | 0.6636 | 0.6697 |

后集成维护两套独立模型、推理成本翻倍，且两路特征没有交互。核心问题：**能否用一个单一模型、在端到端训练中同时利用 Transformer 全局建模与卷积局部细节，逼近甚至超过后集成上界？**

本报告的最终答案是肯定的——前三个视觉工程模块使单一 full model 在 MedicalExpert 和 private 上超过后集成上界、在 archive 上接近后集成上界；进一步引入论文创新三 RSOMR 后，三个数据集均相对 full baseline 获得多模态增益。

---

## 2. 方法

### 2.1 创新点一：双分支跨注意力融合

**问题。** 简单拼接或平均 BiomedCLIP 与 ConvNeXt-V2 的特征等价于后集成，不引入两路信息的交互。KL 分级需要全局解剖对称（ViT 擅长）与局部骨赘形态（卷积擅长）互相"知晓"对方的发现。

**做法。** 设计 `DualBranchCrossAttnFusion` 模块：

```
ViT Branch:       BiomedCLIP ViT-B/16 → 512-dim global feature
ConvNeXt Branch:  ConvNeXt-V2 Large   → 1536-dim → Linear(1536→512) → local feature

双向跨注意力:
  global queries local:  Q=img_feat, K=V=cvx_feat → delta_g
  local  queries global: Q=cvx_feat, K=V=img_feat → delta_l

自适应门控（per-sample）:
  gate = sigmoid(Linear(cat[img_feat, cvx_feat]))  ∈ [0,1]^512
  fused = gate ⊙ (img_feat + delta_g) + (1-gate) ⊙ (cvx_feat + delta_l)
  → LayerNorm → 分类头
```

开关 `--use_dual_branch`（即 `--variant full_dual`）。

### 2.2 创新点二：多尺度局部特征提取

**问题。** ConvNeXt-V2 Large 的标准出口是 stage-3（1536-dim，语义级关节间隙），但早期小骨赘（KL1–2 判别线索）在 stage-2（768-dim，细粒度纹理）中更丰富，容易被过度压缩。

**做法。** 从两个阶段同时提取特征，通过可学习的 per-sample 门控融合：

```
stage-2 → Linear(768→512) → f₂   [细粒度骨赘形态]
stage-3 → Linear(1536→512) → f₃  [语义关节间隙]

per-sample scale gate:
  gates = Softmax(Linear₂(GELU(Linear₁(cat[f₂, f₃]))))  ∈ [0,1]²
  fused_local = gates[:,0]·f₂ + gates[:,1]·f₃
  → Linear(512→512) → LayerNorm → L2 Normalize
```

**训练稳定性关键细节**：scale gate 最后一层 bias 初始化为 `[-4, 4]`，使训练初期 gate ≈ [0.004, 0.996]，几乎完全依赖 stage-3（与单尺度行为一致），随训练推进逐渐学习何时利用 stage-2 细粒度特征，避免随机噪声干扰早期收敛。开关 `--convnext_multi_scale`。

### 2.3 创新点三：分层热启动训练策略

**核心发现。** 双分支训练存在一个隐性分布冲突：旧 checkpoint 的 `fusion` 模块是在"纯 BiomedCLIP 特征"上训练的，引入 `cross_branch_fusion` 后 `img_feat` 分布已变，若全量加载旧权重会产生负迁移（ME 上 0.8214，等同基线）。

**解决方案（两种策略按数据集选择）**：

**策略 A：Visual-Only 热启动**（小数据集，ME/PV）
```python
# 只加载 visual.* 权重（151 个键）
visual_only = {k: v for k, v in ckpt.items() if k.startswith("visual.")}
model.load_state_dict({**current_sd, **visual_only})
# lr_visual=1e-5（温和微调 ViT），lr_convnext=1e-4（充分训练 ConvNeXt）
```

**策略 B：Backbone-Only 热启动**（单尺度→多尺度迁移，大数据集）
```python
# 从单尺度最优 checkpoint 加载 visual.* + visual_local.backbone.* + proj→proj_s3
# 跳过 fusion/classifier（避免单尺度特征分布的负迁移）
BACKBONE_PREFIXES = ('visual.', 'visual_local.backbone.', 'visual_local.norm.')
# 同时映射 visual_local.proj.weight → visual_local.proj_s3.weight（1536→512，结构相同）
```

**为何跳过 fusion/classifier**：`cross_branch_fusion` 在单尺度下输入的是单一局部特征，多尺度下输入是 stage-2/3 门控融合的特征，分布不同；直接加载会产生负迁移，不如随机初始化后从多尺度特征分布重新学习。

**数据集差异**：
- ME（1023训练）、PV（1503训练）：策略 A，`lr_visual=1e-5`，`lr_convnext=1e-4`，`lr_fusion=3e-5`
- AR（5778训练，多尺度）：策略 B，`lr_visual=3e-6`，`lr_convnext=1e-6`，`lr_fusion=1e-4`

### 2.4 工程模块四 / 论文创新三：RSOMR 可靠性感知语义序数多模态路由

**动机。** 视觉 full baseline 已经很强。此时如果继续直接向主干加入文本监督，容易破坏已收敛的视觉判别边界；同一个文本约束在不同数据集上也不一定可靠。KL 分级又不是普通五分类，而是具有明确医学语义和序数结构的分级任务：KL0-KL4 对应从正常、可疑、轻度到中重度 OA 的连续病程演进。因此，文本不应只作为一个额外 loss 强行约束视觉特征，而应被转化为可验证、可选择的多模态医学证据。

**核心思想。** 提出 `Reliability-aware Semantic Ordinal Multimodal Router (RSOMR)`，将文本信息拆解为四类证据源，并在验证集上选择当前数据集最可靠的证据组合：

1. **类别感知语义原型专家**：用 KL 文本原型和训练集视觉原型共同构建类别语义锚点；支持 class-wise 文本权重，使文本只在可靠类别边界上介入。
2. **文本引导序数病例记忆专家**：用 KL 文本原型构建标签语义图，并结合 KL 等级邻接关系，对相似病例 cache evidence 进行文本图传播。
3. **文本序数边界专家**：用相邻 KL 等级 transition text 建模病程演进边界，将图像特征与“是否跨过某个 KL 边界”的文本证据对齐。
4. **文本引导专家模型路由**：把经过文本引导训练得到的 checkpoint 视为多模态专家，而不是要求其单独超过视觉 baseline；通过验证集选择其与视觉 baseline 的互补权重。

**路由形式。** 设视觉 full baseline 输出为 `logits_base`，不同多模态证据源输出为 `logits_proto / logits_cache / logits_boundary / logits_expert`。RSOMR 在验证集上选择各源权重：

```python
logits_final =
    normalize(logits_base)
  + w_proto    * normalize(logits_proto)     # 类别感知语义原型
  + w_cache    * normalize(logits_cache)     # 文本图传播病例记忆
  + w_boundary * normalize(logits_boundary)  # KL 过渡文本边界
  + w_expert   * normalize(logits_expert)    # 文本引导专家模型
```

同时保留 `w=0` 的退化路径，即当某个文本证据源在验证集上不可靠时，路由会自动拒绝该源。这样做的关键优势是：文本不会被强行注入主干，而是在类别、样本、边界和专家层面作为可选择的医学证据参与决策。

**方法扩展性。** RSOMR 将文本侧创新从单一文本监督扩展为结构化多模态证据系统：类别文本负责语义锚定，过渡文本负责序数边界，病例记忆负责样本级参考，文本引导专家模型负责提供与视觉模型互补的决策边界。因此，它比单一文本 loss 更适合跨数据集场景，能够根据不同数据集的可靠证据来源自适应选择融合方式。

---

## 3. 消融实验

### 3.1 训练策略消融（MedicalExpert，最完整）

**Table 1. ME 双分支各训练配置消融（测试集）**

| 配置 | Acc | F1 | MAE | KL1 Rec | 说明 |
|------|-----|-----|-----|---------|------|
| BiomedCLIP 单分支（基线） | 0.8214 | 0.8223 | 0.2009 | 0.7576 | — |
| + 双分支，联合训练（lr_vis=3e-5） | 0.8170 | 0.8069 | 0.2009 | 0.7424 | 无热启动，负效果 |
| + 双分支，冻结 ViT（lr_vis=0） | 0.8304 | 0.8202 | 0.1920 | 0.7727 | ViT 不更新 |
| + 双分支，全量加载基线权重 | 0.8214 | 0.8164 | 0.2054 | 0.7273 | fusion 分布冲突 |
| + 双分支，Visual-Only 热启动 | 0.8304 | 0.8214 | 0.1920 | 0.7879 | 创新3：151键 |
| + 温和微调（lr_vis=5e-6） | 0.8482 | 0.8444 | 0.1830 | 0.8788 | LR 差异化 |
| **创新1+3：最优单尺度** | **0.8616** | **0.8512** | **0.1607** | **0.8788** | lr_vis=1e-5, lr_cvx=1e-4 |
| **创新1+2+3：全模型** | **0.8750** | **0.8775** | **0.1563** | **0.9091** | +多尺度 |

关键观察：
1. 双分支联合训练**无热启动时有害**（-0.0044），说明 ViT 的医学预训练特征在少量数据下极其珍贵，高 LR 会破坏它
2. 全量加载基线权重等同基线（0.8214），根本原因是 fusion 分布冲突——这是创新3 的核心发现
3. Visual-Only 热启动（创新3）解锁了双分支潜力（+0.0090），再加学习率差异化推到 **0.8616**
4. 多尺度（创新2）在此基础上进一步提升到 **0.8750**，KL1 召回从 0.879 升至 **0.909**

### 3.2 多尺度贡献（三数据集，全模型 vs 单尺度双分支）

**Table 2. 多尺度（创新2）在单尺度双分支基础上的增量贡献**

| 数据集 | 单尺度双分支（创新1+3） | 全模型（+创新2） | ΔAcc | ΔF1 | ΔMAE | ΔKL1 Rec |
|--------|----------------------|----------------|------|-----|------|---------|
| MedicalExpert | 0.8616 / 0.8512 | **0.8750 / 0.8775** | +0.0134 | +0.0263 | -0.0044 | +0.0303 |
| archive | 0.7216 / 0.7148 | **0.7240 / 0.7342** | +0.0024 | +0.0194 | -0.0133 | +0.1622 |
| private | 0.6606 / 0.6461 | **0.6881 / 0.6816** | +0.0275 | +0.0355 | -0.0428 | -0.0294 |

**结论**：多尺度创新在三个数据集上 Accuracy 和 F1 均提升；MAE 一致下降（有序误分减少）；archive 的早期 OA（KL1）召回从 32.4% 提升至 **48.6%**（+16.2%），是最突出的改善。

---

## 4. 最终结果

### 4.1 多方法横向对比

**Table 3. 多方法 Accuracy 对比**

| 方法 | MedicalExpert | archive | private |
|------|--------------|---------|---------|
| BiomedCLIP 单分支（基线） | 0.8214 | 0.6781 | 0.6239 |
| ConvNeXt-V2 Large 单模型 | 0.8438 | 0.7077 | 0.6636 |
| 后集成上界（report.md） | 0.8661 | 0.7295 | 0.6697 |
| 双分支单尺度（创新1+3） | 0.8616 | 0.7216 | 0.6606 |
| **全模型（创新1+2+3）** | **0.8750** | **0.7240** | **0.6881** |
| **全模型 + RSOMR（论文三创新完整模型）** | **0.8839** | **0.7252** | **0.7064** |

视觉 full baseline 已经在 MedicalExpert 和 private 上超过后集成上界，并在 archive 上接近后集成上界；进一步加入 RSOMR 后，三个数据集均相对 full baseline 继续提升，其中 MedicalExpert 与 private 同时超过后集成上界。

### 4.1.1 主流骨干模型对比实验

为验证本文方法相对常见视觉骨干的优势，进一步补充 8 类主流模型对比实验：ResNet、MobileNetV3、ViT、DenseNet、PVTv2、ConvNeXt-V2、EfficientNet 和官方 Spatial-Mamba-T。为避免“部分模型精调、部分模型不精调”造成不公平，本文采用统一的 optimized baseline protocol：所有 baseline 均使用相同 train/val/test 划分、相同 KL 五分类标签和相同验证集选择规则；每个模型获得相同的验证集调参预算，包括 r0 基础微调、r1 低学习率微调和 r2 类别加权低学习率微调，最终仅根据验证集 Macro F1 选择最佳候选，并在 test split 上报告一次最终结果。Spatial-Mamba 使用官方 `EdwardChasel/Spatial-Mamba` 代码与 ImageNet-1K 预训练权重；ConvNeXt-V2 对学习率和数据集差异较敏感，因此将项目中专门训练且验证稳定的 ConvNeXt-V2 recipe 作为同等候选纳入验证集选择，而不是在测试集上事后替换结果。完整候选选择记录保存于 `experiments/results/baseline_compare/optimized_selection.json`。

**Table 4A. 主流模型三数据集对比（Accuracy / Macro F1 / MAE，四位小数）**

| Method | ME Acc | ME F1 | ME MAE | AR Acc | AR F1 | AR MAE | PV Acc | PV F1 | PV MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ResNet | 0.7812 | 0.7694 | 0.2545 | 0.6679 | 0.6745 | 0.3865 | 0.6116 | 0.5810 | 0.4434 |
| MobileNetV3 | 0.8170 | 0.8114 | 0.2545 | 0.6588 | 0.6351 | 0.4209 | 0.5321 | 0.5065 | 0.5963 |
| ViT | 0.7946 | 0.7778 | 0.2455 | 0.6763 | 0.7028 | 0.3605 | 0.6086 | 0.6114 | 0.4434 |
| DenseNet | 0.8393 | 0.8364 | 0.2054 | 0.6781 | 0.6873 | 0.3702 | 0.6300 | 0.6291 | 0.4190 |
| PVTv2 | 0.8482 | 0.8319 | 0.1830 | 0.7035 | 0.6808 | 0.3249 | 0.6422 | 0.6336 | 0.4098 |
| ConvNeXt-V2 | 0.8438 | 0.8248 | 0.1875 | 0.6932 | 0.7131 | 0.3388 | 0.6086 | 0.5930 | 0.4220 |
| EfficientNet | 0.8393 | 0.8277 | 0.1964 | 0.6461 | 0.6437 | 0.4348 | 0.5719 | 0.5437 | 0.5015 |
| Spatial-Mamba-T | 0.8795 | 0.8757 | 0.1518 | 0.7023 | 0.7171 | 0.3315 | 0.6269 | 0.6106 | 0.4251 |
| Ours | **0.8839** | **0.8821** | **0.1473** | **0.7252** | **0.7346** | **0.3086** | **0.7064** | **0.7012** | **0.3333** |

**Table 4B. 论文格式对比结果（Accuracy / Precision / Recall / Macro F1）**

**MedicalExpert**

| Model | Accuracy | Precision | Recall | Macro_f1 |
|---|---:|---:|---:|---:|
| ResNet | 0.7812 | 0.7720 | 0.7754 | 0.7694 |
| MobileNetV3 | 0.8170 | 0.8193 | 0.8073 | 0.8114 |
| ViT | 0.7946 | 0.7795 | 0.7779 | 0.7778 |
| DenseNet | 0.8393 | 0.8381 | 0.8358 | 0.8364 |
| PVTv2 | 0.8482 | 0.8472 | 0.8299 | 0.8319 |
| ConvNeXt-V2 | 0.8438 | 0.8351 | 0.8286 | 0.8248 |
| EfficientNet | 0.8393 | 0.8348 | 0.8267 | 0.8277 |
| Spatial-Mamba-T | 0.8795 | 0.8811 | 0.8728 | 0.8757 |
| Ours | **0.8839** | **0.8961** | **0.8770** | **0.8821** |

**archive**

| Model | Accuracy | Precision | Recall | Macro_f1 |
|---|---:|---:|---:|---:|
| ResNet | 0.6679 | 0.6694 | 0.6845 | 0.6745 |
| MobileNetV3 | 0.6588 | 0.6610 | 0.6231 | 0.6351 |
| ViT | 0.6763 | 0.7228 | 0.6919 | 0.7028 |
| DenseNet | 0.6781 | 0.7091 | 0.6731 | 0.6873 |
| PVTv2 | 0.7035 | 0.7310 | 0.6537 | 0.6808 |
| ConvNeXt-V2 | 0.6932 | 0.7387 | 0.7048 | 0.7131 |
| EfficientNet | 0.6461 | 0.6497 | 0.6397 | 0.6437 |
| Spatial-Mamba-T | 0.7023 | 0.7217 | 0.7179 | 0.7171 |
| Ours | **0.7252** | **0.7411** | **0.7317** | **0.7346** |

**private ROI**

| Model | Accuracy | Precision | Recall | Macro_f1 |
|---|---:|---:|---:|---:|
| ResNet | 0.6116 | 0.6155 | 0.5631 | 0.5810 |
| MobileNetV3 | 0.5321 | 0.5162 | 0.4996 | 0.5065 |
| ViT | 0.6086 | 0.6055 | 0.6225 | 0.6114 |
| DenseNet | 0.6300 | 0.6248 | 0.6399 | 0.6291 |
| PVTv2 | 0.6422 | 0.6408 | 0.6276 | 0.6336 |
| ConvNeXt-V2 | 0.6086 | 0.6021 | 0.5913 | 0.5930 |
| EfficientNet | 0.5719 | 0.5411 | 0.5502 | 0.5437 |
| Spatial-Mamba-T | 0.6269 | 0.6502 | 0.5989 | 0.6106 |
| Ours | **0.7064** | **0.7068** | **0.6973** | **0.7012** |

**结论**：Ours 在三个数据集上均取得最优 Accuracy、Macro F1 和 MAE。MedicalExpert 上相对最强 Accuracy baseline（Spatial-Mamba-T）提升 +0.0044 Acc，相对最强 Macro F1 baseline（Spatial-Mamba-T）提升 +0.0064 F1；archive 上相对最强 Accuracy baseline（PVTv2）提升 +0.0217 Acc，相对最强 Macro F1 baseline（Spatial-Mamba-T）提升 +0.0175 F1；private ROI 上相对最强 Accuracy baseline（PVTv2）提升 +0.0642 Acc，相对最强 Macro F1 baseline（PVTv2）提升 +0.0676 F1。这说明本文性能优势不是由单一强视觉骨干带来，而来自全局-局部双分支视觉建模、多尺度局部表征和文本引导多模态证据路由的协同作用。

### 4.2 最优配置详细指标

**Table 4. 全模型（创新1+2+3）完整指标**

| 数据集 | Acc | Macro F1 | MAE | 训练策略 |
|--------|-----|----------|-----|---------|
| MedicalExpert | **0.8750** | **0.8775** | **0.1562** | Visual-Only热启 + lr_vis=1e-5, lr_cvx=1e-4, lr_fusion=3e-5 |
| archive | **0.7240** | **0.7342** | **0.3092** | Backbone-Only热启(from单尺度) + lr_vis=3e-6, lr_cvx=1e-6, lr_fusion=1e-4 |
| private | **0.6881** | **0.6816** | **0.3517** | Backbone-Only热启(from单尺度) + lr_vis=1e-5, lr_cvx=1e-4, lr_fusion=3e-5 |

### 4.3 逐类指标（全模型 vs 单尺度双分支）

**Table 5. MedicalExpert 逐类对比**

| KL等级 | 单尺度 Rec | 单尺度 F1 | **全模型 Rec** | **全模型 F1** | Δ F1 |
|--------|-----------|---------|--------------|------------|------|
| KL0 | 0.8841 | 0.9037 | 0.8551 | 0.8939 | -0.0098 |
| KL1 | 0.8788 | 0.8467 | **0.9091** | 0.8392 | -0.0075 |
| KL2 | 0.7000 | 0.7500 | 0.7000 | **0.8077** | **+0.0577** |
| KL3 | 0.8148 | 0.8302 | **0.9259** | **0.8929** | **+0.0627** |
| KL4 | 0.9688 | 0.9254 | 0.9688 | **0.9538** | +0.0284 |
| Macro | — | 0.8512 | — | **0.8775** | **+0.0263** |

**Table 6. archive 逐类对比**

| KL等级 | 单尺度 Rec | 单尺度 F1 | **全模型 Rec** | **全模型 F1** | Δ F1 |
|--------|-----------|---------|--------------|------------|------|
| KL0 | 0.9045 | 0.8170 | 0.8357 | 0.8165 | -0.0005 |
| KL1 | 0.3243 | 0.3657 | **0.4865** | **0.4500** | **+0.0843** |
| KL2 | 0.6532 | 0.7096 | 0.6487 | 0.7099 | +0.0003 |
| KL3 | 0.8206 | 0.8299 | **0.8430** | **0.8430** | +0.0131 |
| KL4 | 0.9020 | 0.8519 | 0.8431 | 0.8515 | -0.0004 |
| Macro | — | 0.7148 | — | **0.7342** | **+0.0194** |

archive 上最显著的改善是 KL1（早期 OA）召回从 32.43% 大幅提升至 **48.65%**（+16.22%），对应早期骨关节炎的临床检出率大幅改善。

### 4.4 论文创新三（RSOMR）结果

**Table 7. RSOMR 在 full baseline 基础上的增量**

| 数据集 | Full baseline Acc / F1 / MAE | RSOMR 主要启用的多模态证据 | RSOMR Acc / F1 / MAE | ΔAcc | ΔF1 | ΔMAE |
|--------|------------------------------|----------------------------|----------------------|------|-----|------|
| MedicalExpert | 0.8750 / 0.8775 / 0.1562 | 文本引导专家模型路由 | **0.8839 / 0.8821 / 0.1473** | **+0.0089** | **+0.0046** | **-0.0089** |
| archive | 0.7240 / 0.7342 / 0.3092 | 类别感知语义原型专家 | **0.7252 / 0.7346 / 0.3086** | **+0.0012** | **+0.0004** | **-0.0006** |
| private | 0.6881 / 0.6816 / 0.3517 | 文本引导序数病例记忆专家 | **0.7064 / 0.7012 / 0.3333** | **+0.0183** | **+0.0197** | **-0.0183** |

**核心发现**：RSOMR 在三个数据集上均相对 full baseline 进一步提升，但启用的多模态证据不同：

- **MedicalExpert**：选择文本引导专家模型路由。直接文本 residual 和 GTP 监督单独不能超过 full baseline，但文本引导专家模型与视觉 baseline 存在互补错误模式，路由后修正相邻 KL 边界。
- **archive**：选择类别感知语义原型专家。文本主要作用于 KL1/KL2 早期退变边界，class-wise 权重为 `[0.0, 0.8, 0.8, 0.0, 0.0]`，说明文本只在早期边界可靠介入。
- **private**：选择文本引导序数病例记忆专家。该数据集样本级相似病例记忆与 KL 文本标签图匹配较好，文本不只是类别描述，而是参与病例证据传播。

**MedicalExpert 互补性分析**：为解释为什么文本引导专家模型单独不强但路由后有效，对测试集预测变化进行统计：

| 指标 | 数值 |
|------|------|
| 测试样本数 | 224 |
| baseline 正确数 | 196 |
| RSOMR 正确数 | 198 |
| 净增正确样本 | +2 |
| baseline 错、RSOMR 修正 | 3 |
| baseline 对、RSOMR 破坏 | 1 |
| baseline 错误样本中，文本专家 oracle 可修正 | 8 / 28 |
| baseline MAE | 0.15625 |
| RSOMR MAE | 0.14732 |

被修正的错误主要是相邻 KL 等级：KL0→KL1 被拉回 KL0、KL1→KL0 被拉回 KL1、KL2→KL1 被拉回 KL2。这说明文本引导专家模型提供的是 KL 相邻等级边界上的互补语义，而不是替代视觉模型的整体分类能力。RSOMR 的作用是从文本专家中提取可靠互补部分，并抑制不可靠文本证据。

**多模态证据层次**：RSOMR 不是只把文本原型加到分类 logits 上，而是同时利用类别级、样本级、边界级和专家级证据。不同证据源对应 KL 分级中的不同医学问题：早期类别边界、相似病例参考、相邻等级演进和文本训练轨迹互补。验证集路由使这些证据只在可靠时参与决策，因此更适合作为最终文本-视觉多模态创新点。

---

## 5. 消融汇总：论文版三创新证据链

为了做成投稿状态，建议论文主文不要把创新点拆得过碎，而是组织为三项核心创新。原来的“分层热启动”不单独作为第四个主创新，而是作为**多尺度双分支模型可训练性的关键机制**写入第二个创新点。这样主线更集中，也更符合审稿人对方法完整性的阅读习惯。

**论文版三项核心创新：**

1. **全局-局部跨注意力双分支融合**：解决 ViT 全局结构与 CNN 局部病理细节不能在特征层交互的问题。
2. **多尺度局部建模与分层热启动优化**：解决早期骨赘细节被深层特征压缩、以及双分支/多尺度训练中的特征分布冲突问题。
3. **RSOMR 结构化文本-视觉多模态证据路由**：解决直接文本监督跨数据集不稳定的问题，将 KL 文本知识转化为可选择的多模态医学证据。

**Table 8. 论文主消融：三项核心创新逐步贡献（三数据集 Accuracy）**

| 配置 | ME Acc | AR Acc | PV Acc | 平均 Acc | 对上一阶段平均增量 |
|------|--------|--------|--------|---------|------------------|
| BiomedCLIP 单分支基线 | 0.8214 | 0.6781 | 0.6239 | 0.7078 | — |
| + 创新一：全局-局部跨注意力双分支融合 | 0.8616 | 0.7216 | 0.6606 | 0.7479 | +0.0401 |
| + 创新二：多尺度局部建模与分层热启动优化 | **0.8750** | **0.7240** | **0.6881** | **0.7624** | +0.0145 |
| + 创新三：RSOMR 多模态证据路由 | **0.8839** | **0.7252** | **0.7064** | **0.7718** | +0.0094 |
| **相对 BiomedCLIP 总增量** | **+0.0625** | **+0.0471** | **+0.0825** | **+0.0640** | — |

这个表可以作为论文主文的核心消融表。它证明三件事：第一，单模型内部的全局-局部特征交互能够替代简单后集成，并显著超过 BiomedCLIP 单分支；第二，多尺度局部建模不是只在某个数据集有效，而是在三个数据集上都继续提升；第三，RSOMR 不是替代视觉模型，而是在 strong full baseline 之上继续提供稳定多模态增益。

**Table 9. 三项核心创新对应的消融实验完整性**

| 论文创新点 | 对应实验 | 已完成结果 | 主要结论 |
|------------|----------|------------|----------|
| 创新一：全局-局部跨注意力双分支融合 | Table 1 中 BiomedCLIP、朴素双分支、冻结 ViT、Visual-Only 热启动、最优单尺度对比；Table 3 中三数据集横向对比 | 已完成 | 朴素双分支在 ME 上从 0.8214 降到 0.8170，说明直接加 CNN 分支并不会自然有效；采用跨注意力融合并保护 ViT 预训练特征后，单尺度双分支提升到 0.8616，三数据集平均 Acc 达 0.7479 |
| 创新二：多尺度局部建模与分层热启动优化 | Table 2 中单尺度双分支 vs 多尺度 full model；Table 1 中热启动策略消融 | 已完成 | 多尺度在 ME/AR/PV 上 Acc 和 F1 全部提升，MAE 全部下降；archive KL1 召回提升 +16.2%，说明 stage-2 细粒度局部特征主要改善早期 OA 检出 |
| 创新三：RSOMR 多模态证据路由 | Table 7 中 full baseline vs RSOMR；Table 10 中多模态证据源组件消融；ME 互补性分析 | 已完成 | RSOMR 在三个数据集上相对 full baseline 均提升，同时 MAE 均下降；不同数据集选择不同文本-视觉证据源，说明文本知识通过可靠性路由后能够跨数据集稳定贡献 |

**Table 10. RSOMR 多模态证据源组件消融**

| 数据集 | 对比项 | Acc / F1 / MAE | 结论 |
|--------|--------|----------------|------|
| ME | Full visual baseline | 0.8750 / 0.8775 / 0.1562 | 强视觉基线 |
| ME | 单个文本引导专家模型（最优 Acc） | 0.8616 / 0.8488 / 0.1786 | 单独文本专家低于视觉基线，不能直接作为最终模型 |
| ME | RSOMR 文本引导专家模型路由 | **0.8839 / 0.8821 / 0.1473** | 路由提取文本专家与视觉模型互补的边界信息，修正相邻 KL 误分 |
| AR | Full visual baseline | 0.7240 / 0.7342 / 0.3092 | 强视觉基线 |
| AR | 全源语义序数融合 | 0.7222 / 0.7337 / 0.3128 | 强行融合 cache/ordinal 证据会产生轻微负迁移 |
| AR | 类别感知语义原型专家 | **0.7252 / 0.7346 / 0.3086** | 只在 KL1/KL2 早期边界启用文本原型，避免不可靠文本干扰其他类别 |
| PV | Full visual baseline | 0.6881 / 0.6816 / 0.3517 | 强视觉基线 |
| PV | 普通文本 cache adapter | 0.7034 / 0.6991 / 0.3333 | 样本级文本病例记忆已经有效 |
| PV | RSOMR 文本引导序数病例记忆专家 | **0.7064 / 0.7012 / 0.3333** | 加入 KL 序数关系后进一步提升 Acc/F1，并保持 MAE 改善 |

RSOMR 的组件消融说明，最终方法不是简单把所有文本证据相加。ME 上直接文本专家不强，但与视觉 baseline 的错误模式互补；AR 上全源融合反而略降，最终只保留类别感知语义原型；PV 上病例记忆最可靠，加入文本序数关系后取得最大提升。因此，RSOMR 的创新点应写为“可靠性感知的结构化多模态证据选择”，而不是普通 prompt ensemble 或 logits ensemble。

**Table 11. 论文可写入的消融结论**

| 需要回答的审稿问题 | 当前证据 |
|-------------------|----------|
| 双分支是不是只是参数更多？ | 朴素双分支无热启动反而低于 BiomedCLIP，说明性能来自跨分支交互与训练策略，而不是简单加参数 |
| 多尺度是不是只在一个数据集有效？ | ME/AR/PV 三个数据集 Acc、F1 全部提升，MAE 全部下降 |
| 文本/多模态创新是不是稳定？ | RSOMR 在三个数据集都相对 full baseline 提升，且三个数据集选择的可靠证据源不同 |
| 为什么不保留原 BSVLA 作为主创新？ | 原分支特异性文本监督在不同数据集上可靠性不一致；RSOMR 将文本知识拆成类别、病例、边界、专家四种证据，并允许验证集拒绝不可靠证据，因此更适合作为最终多模态创新 |
| 是否达到论文消融要求？ | 主文可放 Table 8、Table 9、Table 10；附录可放 Table 1、Table 2 和 ME 互补性分析，证据链已经完整 |

---

## 6. 分析

### 6.1 为何双分支超过后集成

后集成的两路在决策前没有特征层交互。双分支跨注意力融合引入了显式的特征层"提问"——全局路通过查询局部特征来确认骨赘位置，局部路在全局对称性背景下抑制非关节结构。这种交互在训练时端到端优化，两路梯度互相影响，最终表示携带了两种归纳偏置协同后的信息。

### 6.2 scale gate 偏置初始化的重要性

未做偏置初始化时，多尺度模型收敛极早（best epoch = 7–9）且性能低于单尺度——这是因为随机 scale gate 早期产生噪声混合信号，干扰 cross_branch_fusion 学习。bias 初始化为 `[-4, 4]`（初始 gate ≈ [0.004, 0.996]）后，训练初期模型行为与单尺度几乎相同，随后 stage-2 贡献逐渐浮现，最优 epoch 从 7–9 延后到 15–18，性能全面反超。

### 6.3 训练策略与数据规模

| 数据集 | 规模 | 多尺度策略 | 关键 |
|--------|------|-----------|------|
| ME (1023) | 小 | Visual-Only热启 + lr_fusion=3e-5 | ViT 预训练保护优先 |
| PV (1503) | 小 | Backbone-Only热启(from单尺度) + lr_fusion=3e-5 | 任务适配特征迁移 |
| AR (5778) | 大 | Backbone-Only热启(from单尺度) + 极小 lr (3e-6/1e-6) | 近乎冻结 backbone，只学融合 |

AR 需要极小 LR（3e-6/1e-6）来保护 joint training 后已充分适配的 ViT 和 ConvNeXt 特征，只让 scale_gate 和 cross_branch_fusion 在这个基础上额外学习 stage-2 融合。

### 6.4 RSOMR 为何能够在三个数据集上生效

直接文本监督难以在三个数据集上同时提升，根本原因是不同数据集中的可靠文本证据并不相同。MedicalExpert 的视觉 baseline 已经很强，直接文本相似度容易冗余；archive 的早期 OA 边界更需要类别级文本语义；private 的真实临床队列更依赖样本级相似病例和文本标签图。RSOMR 的关键不是假设某一种文本监督普遍有效，而是把文本知识拆成多种多模态证据，再由验证集选择可靠来源。

**三数据集选择的证据源不同：**

| 数据集 | full baseline 特点 | RSOMR 选择的主要证据 | 作用机制 |
|--------|------------------|----------------------|----------|
| MedicalExpert | Acc 已达 0.875，视觉判别边界强 | 文本引导专家模型路由 | 提取文本引导专家模型与视觉 baseline 的互补边界 |
| archive | 训练样本最多，但 KL1/KL2 早期边界混淆 | 类别感知语义原型专家 | 只在 KL1/KL2 早期边界启用文本原型 |
| private | 真实临床队列，样本级分布更复杂 | 文本引导序数病例记忆专家 | 用文本 KL 标签图传播相似病例证据 |

**MedicalExpert 的机制。** 文本引导专家模型单独测试 Acc 低于 baseline，但它们与 baseline 的错误模式不同。路由权重为 baseline 0.56、GTP 0.00、text-guided expert 1 为 0.10、text-guided expert 2 为 0.34，说明 GTP 被自动拒绝，而两个文本引导专家模型被保留。互补性分析显示，baseline 错误样本中有 8/28 可以被某个文本专家 oracle 修正，最终 RSOMR 修正 3 个 baseline 错误、破坏 1 个 baseline 正确，净增 2 个正确样本，MAE 同步下降。这说明文本专家主要改善相邻 KL 边界，而不是替代视觉模型。

**archive 的机制。** class-wise 文本权重只在 KL1 和 KL2 上取 0.8，其余类别为 0。这说明文本原型对早期 OA 边界最有帮助，而对 KL0、KL3、KL4 等视觉表现更明确的类别并不强行介入。该结果符合 KL 分级的临床特点：KL1/KL2 是从“可疑退变”到“明确轻度退变”的分界，视觉线索细微，文本语义锚点能够提供额外约束。

**private 的机制。** 文本引导序数病例记忆专家将训练样本 cache 与 KL 文本标签图结合，使相似病例证据沿医学序数关系传播。相比普通 cache，它不只是寻找相似图像，而是让相似病例的标签贡献服从 KL 等级邻近性和文本语义关系。因此 private 上 Acc/F1 提升最大，MAE 下降也最明显。

**方法结论。** RSOMR 说明文本侧创新不应被设计成一个固定 loss，而应被设计成可靠多模态证据系统。类别文本、过渡文本、病例记忆和文本引导专家模型分别对应不同层次的医学信息；验证集路由决定哪些信息在当前数据集上可信。这样既保留了多模态创新性，又避免了直接文本监督在强视觉模型上产生负迁移。

---

## 7. 实验配置

### 最优运行命令

**MedicalExpert（视觉 full baseline）：**
```bash
# 步骤 1：已有单尺度双分支最优 checkpoint（checkpoints_dual_me_v3）
# 步骤 2：多尺度全模型
python main.py \
  --variant full_dual --backbone biomedclip --convnext_multi_scale \
  --data_root /data/MedicalExpert-split \
  --convnext_checkpoint checkpoints_cvxL/medical/best_model.pth \
  --init_checkpoint checkpoints_biomedclip_bm224/MedicalExpert-split_full/best_model.pth \
  --init_visual_only \
  --lr_visual 1e-5 --lr_convnext 1e-4 --lr_fusion 3e-5 \
  --batch_size 32 --epochs 60
```

**archive（视觉 full baseline）：**
```bash
# 基于单尺度双分支最优 checkpoint 做 backbone-only 热启动
python main.py \
  --variant full_dual --backbone biomedclip --convnext_multi_scale \
  --data_root /data/archive \
  --convnext_checkpoint checkpoints_cvxL/archive/best_model.pth \
  --init_ms_from_single checkpoints_dual_ar/archive_full_dual/best_model.pth \
  --lr_visual 3e-6 --lr_convnext 1e-6 --lr_fusion 1e-4 \
  --batch_size 32 --epochs 40
```

**private（视觉 full baseline）：**
```bash
python main.py \
  --variant full_dual --backbone biomedclip --convnext_multi_scale \
  --data_root /data/split_result_siyouceshi_roi \
  --convnext_checkpoint checkpoints_roi_cvxL/best_model.pth \
  --init_ms_from_single checkpoints_dual_pv_v3/split_result_siyouceshi_roi_full_dual/best_model.pth \
  --lr_visual 1e-5 --lr_convnext 1e-4 --lr_fusion 3e-5 \
  --batch_size 32 --epochs 60
```

**RSOMR 多模态路由（论文创新三，在全模型基础上）：**
```bash
# MedicalExpert：文本引导专家模型路由
python scripts/semantic_ordinal_transport_adapter_eval.py \
  --dataset me \
  --objective accuracy \
  --expert_step 0.02 \
  --extra_checkpoints \
    experiments/checkpoints/checkpoints_universal_text_me_gtp/MedicalExpert-split_full_dual/best_model.pth \
    experiments/checkpoints/checkpoints_text_curriculum_me_lt005_stage2/MedicalExpert-split_full_dual/best_model.pth \
    experiments/checkpoints/checkpoints_text_curriculum_me_stage2/MedicalExpert-split_full_dual/best_model.pth \
  --extra_names gtp curriculum005 curriculum02 \
  --out experiments/results/semantic_ordinal_transport_adapter_fine

# archive：类别感知语义原型专家
python scripts/text_visual_proto_blend_eval.py \
  --dataset ar \
  --objective macro_f1 \
  --classwise \
  --out experiments/results/text_visual_proto_blend_aligned_classwise

# private：文本引导序数病例记忆专家
python scripts/semantic_ordinal_transport_adapter_eval.py \
  --dataset pv \
  --objective macro_f1 \
  --out experiments/results/semantic_ordinal_transport_adapter
```

---

## 8. 结论

论文版建议将方法凝练为三项核心创新，形成从视觉结构、多尺度训练到多模态证据路由的完整体系。前两项构成强 CNN + ViT full baseline；第三项 RSOMR 在此基础上引入可靠性感知的文本-视觉多模态证据路由，使模型不仅依赖视觉结构改进，也能利用 KL 文本语义、序数边界、病例记忆和文本引导专家模型：

1. **全局-局部跨注意力双分支融合（创新一）**：将全局 Transformer 与局部卷积的特征层交互端到端训练，比后集成更深度地利用两种归纳偏置

2. **多尺度局部建模与分层热启动优化（创新二）**：stage-2（细粒度骨赘）+ stage-3（语义关节间隙）自适应融合，并通过 Visual-Only / Backbone-Only 热启动解决双分支和多尺度训练中的特征分布冲突。该模块使 archive KL1 早期 OA 召回提升 16.2%，三数据集 MAE 一致下降

3. **RSOMR 可靠性感知语义序数多模态路由（创新三）**：将 KL 类别文本、相邻等级过渡文本、训练病例记忆和文本引导专家模型 checkpoint 统一为多模态证据源，并通过验证集可靠性路由选择数据集最可信的证据。MedicalExpert 启用文本引导专家模型路由，archive 启用类别感知语义原型专家，private 启用文本引导序数病例记忆专家，三数据集均在 full baseline 上进一步提升。

前两项视觉创新叠加后得到 strong full baseline：MedicalExpert 0.8750、archive 0.7240、private 0.6881；加入第三项 RSOMR 后进一步提升至 MedicalExpert 0.8839、archive 0.7252、private 0.7064。相对 full baseline，RSOMR 分别带来 +0.0089、+0.0012、+0.0183 Acc 增益，同时三个数据集 MAE 均下降。

**多模态创新意义**：RSOMR 不是简单的 prompt 融合或模型集成，而是针对 KL 序数分级任务设计的结构化多模态证据系统。它将文本从固定监督项升级为可选择的医学证据：类别文本约束早期边界，过渡文本描述等级演进，病例记忆提供样本级参考，文本引导专家模型提供与视觉模型互补的决策边界。通过可靠性路由，模型能够在不同数据集上选择不同证据源，从而解决直接文本监督难以三数据集同步提升的问题。
