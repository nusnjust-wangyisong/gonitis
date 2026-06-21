"""
ablation_models.py
==================
膝关节 X-ray KL 分级 —— 消融实验模型定义

四个实验变体，对应论文消融分析：

  Exp1  CLIP visual + Classifier
        基础视觉分类 baseline，无任何文本信号。

  Exp2  CLIP visual + KL text prototype + 普通 gated fusion
        验证固定 KL 文本语义是否对分类有帮助；
        使用静态文本原型（非聚类动态生成），
        门控网络仅用 img_feat 与 text_feat 的拼接，无聚类概率分支。

  Exp3  CLIP visual + cluster_prob 动态文本生成 + 直接残差融合
        验证聚类动态文本生成是否优于静态原型；
        移除门控 MLP，改用简单逐元素残差相加。

  Exp4  CLIP visual + cluster-aware gated fusion + CE only
        验证完整聚类感知门控融合模块的有效性（无有序损失）。

Full  CLIPFusionContrastiveModel（原始实现，CE + Ordinal）
        已在 model.py 中定义，此处不重复，直接使用即可。

用法示例
--------
from ablation_models import (
    Exp1Classifier,
    Exp2PlainGatedFusion,
    Exp3DynamicTextResidual,
    Exp4ClusterAwareGatedFusionCEOnly,
    build_ablation_model,
)

state_dict = torch.load("clip_weights.pt", map_location="cpu")
model = build_ablation_model("exp1", state_dict).cuda()
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer

from data.prompt import KL_PROMPTS
from .model import build_model, build_kl_prompt_groups, PUBMEDBERT_PATH

# ══════════════════════════════════════════════════════════════════════════════
# Exp1: CLIP visual + Classifier  ——  baseline，无任何文本信号
# ══════════════════════════════════════════════════════════════════════════════

class Exp1Classifier(nn.Module):
    """
    最简 baseline：
      image → CLIP visual encoder → img_feat [B, D]
            → MLP Classifier     → logits [B, 5]
    损失：CE only
    """

    NUM_CLASSES = 5

    def __init__(self,
                 clip_model: CLIP,
                 embed_dim: int = 512,
                 cls_hidden_dim: int = 256,
                 cls_dropout: float = 0.2):
        super().__init__()

        self.visual = clip_model.visual

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, self.NUM_CLASSES),
        )

        self.ce_loss_fn = nn.CrossEntropyLoss()

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.visual(image.type(self.dtype)).float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def forward(self,
                image: torch.Tensor,
                labels: torch.Tensor = None) -> dict:
        img_feat   = self.encode_image(image)          # [B, D]
        logits_cls = self.classifier(img_feat)          # [B, 5]

        out = {"logits_cls": logits_cls, "img_feat": img_feat}

        if self.training and labels is not None:
            out["loss"] = self.ce_loss_fn(
                logits_cls.clamp(-20.0, 20.0), labels.long()
            )
        return out

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(image)["logits_cls"].argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Exp2: CLIP visual + KL text prototype + 普通 gated fusion
#       门控输入：[img_feat ‖ text_feat ‖ img_feat⊙text_feat]（无聚类概率）
#       验证固定 KL 文本语义是否有效
# ══════════════════════════════════════════════════════════════════════════════

class PlainGatedFusion(nn.Module):
    """
    普通门控融合（无聚类感知）：

        gate_input = concat(img_feat, text_feat, img_feat ⊙ text_feat)
                   ∈ R^(B × 3D)
        gate       = sigmoid( MLP(gate_input) )  ∈ R^(B × D)
        fused      = LayerNorm( img_feat + gate × text_feat )
        fuse_feat  = L2-normalize( fused )
    """

    def __init__(self,
                 embed_dim: int,
                 hidden_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        # 注意：gate_input_dim = 3D（无聚类概率 C）
        gate_input_dim = embed_dim * 3

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self,
                img_feat: torch.Tensor,   # [B, D]
                text_feat: torch.Tensor,  # [B, D]  固定 KL 原型的加权平均或直接使用
                ) -> torch.Tensor:
        img_feat  = torch.nan_to_num(img_feat.float(),  nan=0.0, posinf=1.0, neginf=-1.0)
        text_feat = torch.nan_to_num(text_feat.float(), nan=0.0, posinf=1.0, neginf=-1.0)

        interaction = img_feat * text_feat
        gate_input  = torch.cat([img_feat, text_feat, interaction], dim=-1)
        gate        = self.gate_mlp(gate_input)
        gate        = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)

        fused       = self.layer_norm(img_feat + gate * text_feat)
        fused       = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        return fused / fused.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class Exp2PlainGatedFusion(nn.Module):
    """
    Exp2：CLIP visual + 固定 KL 文本原型 + 普通门控融合

    文本原型生成：
      - 对每条 KL_PROMPTS 编码，得到 text_feat [5, D]（已冻结）。
      - 用 img_feat 与各原型的相似度做 softmax，得到加权文本特征
        (等价于"软最近邻"但没有可学习聚类中心，保留文本语义同时
         又能动态地选择原型，与 Exp3 的聚类动态生成形成对比)。
      - 注意：这里没有可学习聚类原型矩阵，是与 Exp3 的核心区别。
    """

    TEMPERATURE = 0.07
    NUM_CLASSES = 5

    def __init__(self,
                 clip_model: CLIP,
                 embed_dim: int = 512,
                 fusion_hidden_dim: int = None,
                 fusion_dropout: float = 0.1,
                 cls_hidden_dim: int = 256,
                 cls_dropout: float = 0.2,
                 freeze_text_encoder: bool = True,
                 pubmedbert_path: str = PUBMEDBERT_PATH):
        super().__init__()

        self.visual          = clip_model.visual
        self.text_encoder    = clip_model.text_encoder
        self.text_projection = clip_model.text_projection

        self.text_encoder.float()
        self.text_projection.float()

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # 注意：Exp2 没有可学习聚类原型，直接使用静态文本原型做相似度加权
        self.fusion = PlainGatedFusion(
            embed_dim=embed_dim,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, self.NUM_CLASSES),
        )

        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.register_buffer("text_feat", torch.zeros(self.NUM_CLASSES, embed_dim))
        self._text_feat_initialized = False
        self._pubmedbert_path = pubmedbert_path

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @torch.no_grad()
    def _init_text_prototypes(self):
        tokenizer = AutoTokenizer.from_pretrained(self._pubmedbert_path)
        device = self.text_feat.device
        prompt_groups = build_kl_prompt_groups(KL_PROMPTS)
        if len(prompt_groups) != self.NUM_CLASSES:
            raise ValueError(f"Expected {self.NUM_CLASSES} KL prompt groups, got {len(prompt_groups)}.")
        all_feats = []
        for prompts in prompt_groups:
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            self.text_encoder.float()
            self.text_projection.float()
            with torch.cuda.amp.autocast(enabled=False):
                outputs   = self.text_encoder(input_ids=input_ids,
                                              attention_mask=attention_mask)
                hidden    = outputs.last_hidden_state.float()
                cls_out   = hidden[:, 0, :]
                mask_exp  = attention_mask.unsqueeze(-1).float()
                mean_pool = (hidden * mask_exp).sum(dim=1) \
                            / mask_exp.sum(dim=1).clamp(min=1e-9)
                combined  = (cls_out + mean_pool) / 2.0
                feat      = self.text_projection(combined.float())
                feat      = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                feat      = feat.mean(dim=0)
                feat      = feat / feat.norm(dim=-1).clamp(min=1e-9)
            all_feats.append(feat)
        self.text_feat.copy_(torch.stack(all_feats, dim=0))
        self._text_feat_initialized = True

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.visual(image.type(self.dtype)).float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def forward(self,
                image: torch.Tensor,
                labels: torch.Tensor = None) -> dict:
        if not self._text_feat_initialized:
            self._init_text_prototypes()

        img_feat = self.encode_image(image)  # [B, D]

        # ── 用静态文本原型做 softmax 加权（无可学习聚类中心）──────────
        sim           = img_feat @ self.text_feat.T / self.TEMPERATURE  # [B, 5]
        text_weights  = F.softmax(sim, dim=-1)                          # [B, 5]
        text_feat_dyn = text_weights @ self.text_feat                   # [B, D]
        text_feat_dyn = text_feat_dyn / text_feat_dyn.norm(
            dim=-1, keepdim=True).clamp(min=1e-9)

        # ── 普通门控融合（无聚类感知，无 cluster_prob 输入）─────────────
        fuse_feat  = self.fusion(img_feat, text_feat_dyn)
        logits_cls = self.classifier(fuse_feat)

        out = {"logits_cls": logits_cls, "img_feat": img_feat,
               "text_weights": text_weights, "fuse_feat": fuse_feat}

        if self.training and labels is not None:
            out["loss"] = self.ce_loss_fn(
                logits_cls.clamp(-20.0, 20.0), labels.long()
            )
        return out

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(image)["logits_cls"].argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Exp3: CLIP visual + cluster_prob 动态文本生成 + 直接残差融合
#       保留可学习聚类原型，但去掉门控 MLP，改为简单 LayerNorm 残差相加
#       验证聚类动态文本生成是否有效（对比 Exp2）
# ══════════════════════════════════════════════════════════════════════════════

class DirectResidualFusion(nn.Module):
    """
    直接残差融合（无门控 MLP）：

        fused      = LayerNorm( img_feat + cluster_txt_feat )
        fuse_feat  = L2-normalize( fused )

    与 ClusterAwareResidualGatedFusion 的唯一区别：
    去掉了 gate_mlp，直接等权相加，验证"动态门控权重"是否必要。
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self,
                img_feat: torch.Tensor,
                cluster_txt_feat: torch.Tensor) -> torch.Tensor:
        img_feat         = torch.nan_to_num(img_feat.float(),         nan=0.0, posinf=1.0, neginf=-1.0)
        cluster_txt_feat = torch.nan_to_num(cluster_txt_feat.float(), nan=0.0, posinf=1.0, neginf=-1.0)

        fused = self.layer_norm(img_feat + cluster_txt_feat)
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        return fused / fused.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class Exp3DynamicTextResidual(nn.Module):
    """
    Exp3：CLIP visual + 聚类动态文本生成 + 直接残差融合

    与完整模型 CLIPFusionContrastiveModel 的区别：
      - 保留可学习聚类原型矩阵（cluster_prototypes）
      - 保留 cluster_prob 动态文本生成
      - 将 ClusterAwareResidualGatedFusion 替换为 DirectResidualFusion
        （无门控 MLP，无 cluster_prob 拼接到 gate 输入）
    """

    TEMPERATURE = 0.07
    NUM_CLASSES = 5

    def __init__(self,
                 clip_model: CLIP,
                 embed_dim: int = 512,
                 cls_hidden_dim: int = 256,
                 cls_dropout: float = 0.2,
                 freeze_text_encoder: bool = True,
                 pubmedbert_path: str = PUBMEDBERT_PATH):
        super().__init__()

        self.visual          = clip_model.visual
        self.text_encoder    = clip_model.text_encoder
        self.text_projection = clip_model.text_projection

        self.text_encoder.float()
        self.text_projection.float()

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ── 可学习聚类原型（与完整模型相同）────────────────────────────
        cluster_init = F.normalize(torch.randn(self.NUM_CLASSES, embed_dim), dim=-1)
        self.cluster_prototypes = nn.Parameter(cluster_init)

        # ── 直接残差融合（无门控）──────────────────────────────────────
        self.fusion = DirectResidualFusion(embed_dim=embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, self.NUM_CLASSES),
        )

        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.register_buffer("text_feat", torch.zeros(self.NUM_CLASSES, embed_dim))
        self._text_feat_initialized = False
        self._pubmedbert_path = pubmedbert_path

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @torch.no_grad()
    def _init_text_prototypes(self):
        tokenizer = AutoTokenizer.from_pretrained(self._pubmedbert_path)
        device = self.text_feat.device
        prompt_groups = build_kl_prompt_groups(KL_PROMPTS)
        if len(prompt_groups) != self.NUM_CLASSES:
            raise ValueError(f"Expected {self.NUM_CLASSES} KL prompt groups, got {len(prompt_groups)}.")
        all_feats = []
        for prompts in prompt_groups:
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            self.text_encoder.float()
            self.text_projection.float()
            with torch.cuda.amp.autocast(enabled=False):
                outputs   = self.text_encoder(input_ids=input_ids,
                                              attention_mask=attention_mask)
                hidden    = outputs.last_hidden_state.float()
                cls_out   = hidden[:, 0, :]
                mask_exp  = attention_mask.unsqueeze(-1).float()
                mean_pool = (hidden * mask_exp).sum(dim=1) \
                            / mask_exp.sum(dim=1).clamp(min=1e-9)
                combined  = (cls_out + mean_pool) / 2.0
                feat      = self.text_projection(combined.float())
                feat      = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                feat      = feat.mean(dim=0)
                feat      = feat / feat.norm(dim=-1).clamp(min=1e-9)
            all_feats.append(feat)
        self.text_feat.copy_(torch.stack(all_feats, dim=0))
        self._text_feat_initialized = True

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.visual(image.type(self.dtype)).float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def forward(self,
                image: torch.Tensor,
                labels: torch.Tensor = None) -> dict:
        if not self._text_feat_initialized:
            self._init_text_prototypes()

        img_feat = self.encode_image(image)  # [B, D]

        # ── 聚类软分配概率（与完整模型相同）────────────────────────────
        norm_proto     = F.normalize(self.cluster_prototypes, dim=-1)
        cluster_logits = img_feat @ norm_proto.T / self.TEMPERATURE
        cluster_prob   = F.softmax(cluster_logits, dim=-1)             # [B, 5]

        # ── 聚类动态文本生成（与完整模型相同）───────────────────────────
        cluster_txt_feat = cluster_prob @ self.text_feat               # [B, D]
        cluster_txt_feat = cluster_txt_feat / cluster_txt_feat.norm(
            dim=-1, keepdim=True).clamp(min=1e-9)

        # ── 直接残差融合（无门控 MLP，区别于完整模型）───────────────────
        fuse_feat  = self.fusion(img_feat, cluster_txt_feat)
        logits_cls = self.classifier(fuse_feat)

        out = {"logits_cls": logits_cls, "img_feat": img_feat,
               "cluster_prob": cluster_prob, "fuse_feat": fuse_feat}

        if self.training and labels is not None:
            out["loss"] = self.ce_loss_fn(
                logits_cls.clamp(-20.0, 20.0), labels.long()
            )
        return out

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(image)["logits_cls"].argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# Exp4: CLIP visual + cluster-aware gated fusion + CE only（无 Ordinal Loss）
#       验证完整聚类感知融合模块的有效性；与 Full 模型区别仅在于去掉 Ordinal 损失
# ══════════════════════════════════════════════════════════════════════════════

class Exp4ClusterAwareGatedFusionCEOnly(nn.Module):
    """
    Exp4：CLIP visual + 聚类感知门控融合 + CE only

    与完整模型 CLIPFusionContrastiveModel 的区别：
      - LAMBDA_ORDINAL = 0，即彻底去掉 OrdinalCELoss 项
      - 保留所有其他组件（聚类原型、动态文本生成、ClusterAwareGatedFusion）
    通过对比 Exp4 与 Full 模型，可单独评估 Ordinal Loss 的贡献。
    通过对比 Exp3 与 Exp4，可单独评估门控融合模块的贡献。
    """

    # ── 唯一与完整模型不同的地方 ──────────────────────────────────────
    LAMBDA_ORDINAL = 0.0   # ← Full 模型为 0.2；置 0 即关闭 Ordinal Loss

    TEMPERATURE = 0.07
    NUM_CLASSES = 5

    def __init__(self,
                 clip_model: CLIP,
                 embed_dim: int = 512,
                 fusion_hidden_dim: int = None,
                 fusion_dropout: float = 0.1,
                 cls_hidden_dim: int = 256,
                 cls_dropout: float = 0.2,
                 freeze_text_encoder: bool = True,
                 pubmedbert_path: str = PUBMEDBERT_PATH):
        super().__init__()

        # ── 与完整模型完全相同的结构 ─────────────────────────────────
        from .model import ClusterAwareResidualGatedFusion  # 复用原始融合模块

        self.visual          = clip_model.visual
        self.text_encoder    = clip_model.text_encoder
        self.text_projection = clip_model.text_projection

        self.text_encoder.float()
        self.text_projection.float()

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        cluster_init = F.normalize(torch.randn(self.NUM_CLASSES, embed_dim), dim=-1)
        self.cluster_prototypes = nn.Parameter(cluster_init)

        # ── 与完整模型相同的 ClusterAwareResidualGatedFusion ──────────
        self.fusion = ClusterAwareResidualGatedFusion(
            embed_dim=embed_dim,
            num_classes=self.NUM_CLASSES,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, self.NUM_CLASSES),
        )

        # ── 只保留 CE，不初始化 OrdinalCELoss ─────────────────────────
        self.ce_loss_fn = nn.CrossEntropyLoss()

        self.register_buffer("text_feat", torch.zeros(self.NUM_CLASSES, embed_dim))
        self._text_feat_initialized = False
        self._pubmedbert_path = pubmedbert_path

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @torch.no_grad()
    def _init_text_prototypes(self):
        tokenizer = AutoTokenizer.from_pretrained(self._pubmedbert_path)
        device = self.text_feat.device
        prompt_groups = build_kl_prompt_groups(KL_PROMPTS)
        if len(prompt_groups) != self.NUM_CLASSES:
            raise ValueError(f"Expected {self.NUM_CLASSES} KL prompt groups, got {len(prompt_groups)}.")
        all_feats = []
        for prompts in prompt_groups:
            enc = tokenizer(prompts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            self.text_encoder.float()
            self.text_projection.float()
            with torch.cuda.amp.autocast(enabled=False):
                outputs   = self.text_encoder(input_ids=input_ids,
                                              attention_mask=attention_mask)
                hidden    = outputs.last_hidden_state.float()
                cls_out   = hidden[:, 0, :]
                mask_exp  = attention_mask.unsqueeze(-1).float()
                mean_pool = (hidden * mask_exp).sum(dim=1) \
                            / mask_exp.sum(dim=1).clamp(min=1e-9)
                combined  = (cls_out + mean_pool) / 2.0
                feat      = self.text_projection(combined.float())
                feat      = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                feat      = feat.mean(dim=0)
                feat      = feat / feat.norm(dim=-1).clamp(min=1e-9)
            all_feats.append(feat)
        self.text_feat.copy_(torch.stack(all_feats, dim=0))
        self._text_feat_initialized = True

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.visual(image.type(self.dtype)).float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def forward(self,
                image: torch.Tensor,
                labels: torch.Tensor = None) -> dict:
        if not self._text_feat_initialized:
            self._init_text_prototypes()

        img_feat = self.encode_image(image)

        norm_proto     = F.normalize(self.cluster_prototypes, dim=-1)
        cluster_logits = img_feat @ norm_proto.T / self.TEMPERATURE
        cluster_prob   = F.softmax(cluster_logits, dim=-1)

        cluster_txt_feat = cluster_prob @ self.text_feat
        cluster_txt_feat = cluster_txt_feat / cluster_txt_feat.norm(
            dim=-1, keepdim=True).clamp(min=1e-9)

        fuse_feat  = self.fusion(img_feat, cluster_txt_feat, cluster_prob)
        logits_cls = self.classifier(fuse_feat)

        out = {"logits_cls": logits_cls, "img_feat": img_feat,
               "cluster_prob": cluster_prob, "fuse_feat": fuse_feat}

        if self.training and labels is not None:
            # ── 只用 CE，LAMBDA_ORDINAL = 0.0 ─────────────────────────
            loss = self.ce_loss_fn(logits_cls.clamp(-20.0, 20.0), labels.long())
            out["loss"] = loss
        return out

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(image)["logits_cls"].argmax(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 统一工厂函数
# ══════════════════════════════════════════════════════════════════════════════

def build_ablation_model(
    variant: str,
    state_dict: dict,
    pubmedbert_path: str = PUBMEDBERT_PATH,
    freeze_text_encoder: bool = True,
    fusion_dropout: float = 0.1,
    cls_dropout: float = 0.2,
) -> nn.Module:
    """
    统一工厂函数，通过 variant 字符串返回对应消融实验模型。

    Parameters
    ----------
    variant : str
        "exp1" | "exp2" | "exp3" | "exp4" | "full"
    state_dict : dict
        原始 CLIP 预训练权重（从 .pt 文件加载）。
    pubmedbert_path : str
        PubMedBERT 本地路径。
    freeze_text_encoder : bool
        是否冻结文本编码器（Exp1 无文本编码器，此参数不影响）。
    fusion_dropout / cls_dropout : float
        Dropout 超参。

    Returns
    -------
    nn.Module，已载入视觉权重，处于 train() 模式。

    示例
    ----
    state_dict = torch.load("clip_weights.pt", map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    for v in ["exp1", "exp2", "exp3", "exp4", "full"]:
        model = build_ablation_model(v, state_dict).cuda()
        # 若需要文本原型初始化（exp2/3/4/full）：
        if hasattr(model, "_init_text_prototypes"):
            model._init_text_prototypes()
    """
    variant = variant.lower().strip()

    clip_base = build_model(state_dict, pubmedbert_path=pubmedbert_path)
    embed_dim = state_dict["text_projection"].shape[1]

    if variant == "exp1":
        model = Exp1Classifier(
            clip_model=clip_base,
            embed_dim=embed_dim,
            cls_dropout=cls_dropout,
        )

    elif variant == "exp2":
        model = Exp2PlainGatedFusion(
            clip_model=clip_base,
            embed_dim=embed_dim,
            fusion_dropout=fusion_dropout,
            cls_dropout=cls_dropout,
            freeze_text_encoder=freeze_text_encoder,
            pubmedbert_path=pubmedbert_path,
        )

    elif variant == "exp3":
        model = Exp3DynamicTextResidual(
            clip_model=clip_base,
            embed_dim=embed_dim,
            cls_dropout=cls_dropout,
            freeze_text_encoder=freeze_text_encoder,
            pubmedbert_path=pubmedbert_path,
        )

    elif variant == "exp4":
        model = Exp4ClusterAwareGatedFusionCEOnly(
            clip_model=clip_base,
            embed_dim=embed_dim,
            fusion_dropout=fusion_dropout,
            cls_dropout=cls_dropout,
            freeze_text_encoder=freeze_text_encoder,
            pubmedbert_path=pubmedbert_path,
        )

    elif variant == "full":
        # 完整模型，直接用原始 build_fusion_model
        from .model import build_fusion_model
        return build_fusion_model(
            state_dict=state_dict,
            pubmedbert_path=pubmedbert_path,
            freeze_text_encoder=freeze_text_encoder,
            fusion_dropout=fusion_dropout,
            cls_dropout=cls_dropout,
        ).train()

    else:
        raise ValueError(
            f"未知 variant='{variant}'，可选：exp1 | exp2 | exp3 | exp4 | full"
        )

    return model.train()
