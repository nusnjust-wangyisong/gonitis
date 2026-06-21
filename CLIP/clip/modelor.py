from collections import OrderedDict
from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel, AutoTokenizer
from data.prompt import KL_PROMPTS

CLIP_ROOT = Path(__file__).resolve().parents[1]


def build_kl_prompt_groups(prompt_bank, dataset_name: str = "KneeOA"):
    if isinstance(prompt_bank, dict):
        entry = prompt_bank.get(dataset_name, prompt_bank)
        templates = entry.get("templates", ["{}"])
        classnames = entry.get("slide_classnames") or entry.get("classnames")
        if classnames is None:
            raise ValueError("KL prompt dict must contain 'slide_classnames' or 'classnames'.")
        prompt_groups = []
        for names in classnames:
            if isinstance(names, str):
                names = [names]
            texts = []
            for template in templates:
                for name in names:
                    texts.append(template.format(name) if "{}" in template else f"{template} {name}")
            prompt_groups.append(texts)
        return prompt_groups
    if isinstance(prompt_bank, (list, tuple)):
        if prompt_bank and isinstance(prompt_bank[0], (list, tuple)):
            return [list(x) for x in prompt_bank]
        return [[str(x)] for x in prompt_bank]
    raise TypeError(f"Unsupported KL prompt format: {type(prompt_bank)}")

# ===== 以下类无需修改 =====
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride
        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)
        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)
        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)
        return x


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return x


# ===== 损失函数 =====

class OrdinalCELoss(nn.Module):
    """
    有序回归损失：将 K 分类问题转化为 K-1 个二分类问题，
    对 logits[:, :K-1] 与累积二值标签做 BCE，保持 KL 等级有序关系。
    """
    def __init__(self, num_classes: int, clamp_value: float = 20.0):
        super().__init__()
        self.num_classes = num_classes
        self.clamp_value = clamp_value

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        logits = torch.nan_to_num(logits, nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value)
        logits = logits.clamp(min=-self.clamp_value, max=self.clamp_value)
        targets = targets.long()

        cumulative_targets = torch.stack(
            [targets > k for k in range(self.num_classes - 1)], dim=1
        ).float().to(logits.device)

        cumulative_logits = logits[:, :self.num_classes - 1]
        loss = F.binary_cross_entropy_with_logits(cumulative_logits, cumulative_targets)
        return loss


# ===== 核心：原始CLIP类（保留用于build_model） =====

PUBMEDBERT_PATH = str(CLIP_ROOT / "PubMedBERT")

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 pubmedbert_path: str = PUBMEDBERT_PATH,
                 text_dropout: float = 0.1,
                 ):
        super().__init__()

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.text_encoder = AutoModel.from_pretrained(
            pubmedbert_path, use_safetensors=False
        )
        pubmedbert_hidden_size = self.text_encoder.config.hidden_size

        self.text_projection = nn.Sequential(
            nn.Dropout(text_dropout),
            nn.Linear(pubmedbert_hidden_size, embed_dim, bias=False)
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    @property
    def dtype(self):
        return torch.float32

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        image = image.float()
        feat  = self.visual(image).float()
        feat  = torch.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
        feat  = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return feat

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        self.text_encoder.float()
        self.text_projection.float()

        device = next(self.text_encoder.parameters()).device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        with torch.cuda.amp.autocast(enabled=False):
            outputs   = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden    = outputs.last_hidden_state.float()
            cls_output = hidden[:, 0, :]

            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                mean_pool = (hidden * mask_expanded).sum(dim=1) \
                            / mask_expanded.sum(dim=1).clamp(min=1e-9)
            else:
                mean_pool = hidden.mean(dim=1)

            combined      = (cls_output + mean_pool) / 2.0
            text_features = self.text_projection(combined.float())
        return text_features

    def forward(self, image, input_ids, attention_mask=None):
        image_features = self.encode_image(image)
        text_features  = self.encode_text(input_ids, attention_mask)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=1, keepdim=True)

        logit_scale       = self.logit_scale.exp()
        logits_per_image  = logit_scale * image_features @ text_features.t()
        logits_per_text   = logits_per_image.t()

        return logits_per_image, logits_per_text


# ===== 新增核心模块 =====

class ClusterAwareResidualGatedFusion(nn.Module):
    """
    聚类感知门控残差融合模块。

    将四路信号拼接后学习动态融合门控权重：

        gate_input = concat(img_feat, cluster_txt_feat,
                            img_feat ⊙ cluster_txt_feat, cluster_prob)
                   ∈ R^(B × (3D + C))

        gate       = sigmoid( MLP(gate_input) )   ∈ R^(B × D)
        fused      = LayerNorm( img_feat + gate × cluster_txt_feat )
        fuse_feat  = L2-normalize( fused )
    """
    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 hidden_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        gate_input_dim = embed_dim * 3 + num_classes

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self,
                img_feat: torch.Tensor,          # [B, D]
                cluster_txt_feat: torch.Tensor,  # [B, D]
                cluster_prob: torch.Tensor,       # [B, C]
                ) -> torch.Tensor:
        img_feat         = torch.nan_to_num(img_feat.float(),         nan=0.0, posinf=1.0,  neginf=-1.0)
        cluster_txt_feat = torch.nan_to_num(cluster_txt_feat.float(), nan=0.0, posinf=1.0,  neginf=-1.0)
        cluster_prob     = torch.nan_to_num(cluster_prob.float(),     nan=0.0, posinf=1.0,  neginf=0.0)

        interaction  = img_feat * cluster_txt_feat
        gate_input   = torch.cat([img_feat, cluster_txt_feat,
                                  interaction, cluster_prob], dim=-1)
        gate_input   = torch.nan_to_num(gate_input, nan=0.0, posinf=1.0, neginf=-1.0)

        gate         = self.gate_mlp(gate_input)
        gate         = torch.nan_to_num(gate, nan=0.5, posinf=1.0, neginf=0.0)

        fused        = self.layer_norm(img_feat + gate * cluster_txt_feat)
        fused        = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        fuse_feat    = fused / fused.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return fuse_feat


class CLIPFusionContrastiveModel(nn.Module):
    """
    膝关节 X-ray KL 分级模型（KL 0–4 五分类）。

    架构概述
    --------
    1. 图像编码器（CLIP visual backbone）
       image → img_feat [B, D]，L2 归一化。

    2. 文本原型（KL_PROMPTS × PubMedBERT text encoder）
       预先对 5 条 KL prompt 编码，得到冻结文本原型 text_feat [5, D]。

    3. 可学习聚类原型矩阵
       cluster_prototypes [5, D]，随训练更新，表示每个 KL 等级的聚类中心。

    4. 聚类软分配概率
       cluster_prob = softmax( img_feat @ cluster_prototypes.T / T ) ∈ R^(B × 5)

    5. 动态文本语义生成
       cluster_txt_feat = cluster_prob @ text_feat ∈ R^(B × D)

    6. ClusterAwareResidualGatedFusion
       将 img_feat、cluster_txt_feat、逐元素乘积和 cluster_prob 拼接，
       学习动态门控权重，残差融合后得到 fuse_feat [B, D]。

    7. MLP 分类器
       logits_cls = Classifier(fuse_feat) → [B, 5]

    8. 原型对齐 logits
       logits_proto = img_feat @ text_feat.T / T → [B, 5]

    训练损失
    --------
    loss = CE(logits_cls, labels)                              # 主分类损失
         + λ_ordinal * OrdinalLoss(logits_cls, labels)        # 有序等级惩罚
    """

    TEMPERATURE    = 0.07
    LAMBDA_ORDINAL = 0.2
    NUM_CLASSES    = 5

    def __init__(self,
                 clip_model: CLIP,
                 embed_dim: int = 512,
                 fusion_hidden_dim: int = None,
                 fusion_dropout: float = 0.1,
                 cls_hidden_dim: int = 256,
                 cls_dropout: float = 0.2,
                 freeze_text_encoder: bool = True,
                 pubmedbert_path: str = PUBMEDBERT_PATH,
                 ):
        super().__init__()

        # ── 视觉编码器 ────────────────────────────────────────────────
        self.visual = clip_model.visual

        # ── 文本编码器（PubMedBERT） ──────────────────────────────────
        self.text_encoder    = clip_model.text_encoder
        self.text_projection = clip_model.text_projection

        self.text_encoder.float()
        self.text_projection.float()

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ── 可学习聚类原型矩阵 [C, D] ────────────────────────────────
        cluster_init = F.normalize(
            torch.randn(self.NUM_CLASSES, embed_dim), dim=-1
        )
        self.cluster_prototypes = nn.Parameter(cluster_init)

        # ── 聚类感知门控融合模块 ──────────────────────────────────────
        self.fusion = ClusterAwareResidualGatedFusion(
            embed_dim=embed_dim,
            num_classes=self.NUM_CLASSES,
            hidden_dim=fusion_hidden_dim,
            dropout=fusion_dropout,
        )

        # ── MLP 分类器 ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, cls_hidden_dim),
            nn.GELU(),
            nn.Dropout(cls_dropout),
            nn.Linear(cls_hidden_dim, self.NUM_CLASSES),
        )

        # ── 损失函数 ──────────────────────────────────────────────────
        self.ordinal_loss_fn = OrdinalCELoss(num_classes=self.NUM_CLASSES)
        self.ce_loss_fn      = nn.CrossEntropyLoss()

        # ── 预计算文本原型 ─────────────────────────────────────────────
        self.register_buffer("text_feat", torch.zeros(self.NUM_CLASSES, embed_dim))
        self._text_feat_initialized = False

        self._pubmedbert_path = pubmedbert_path
        self._embed_dim       = embed_dim

    # ------------------------------------------------------------------
    # 文本原型初始化
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_text_prototypes(self):
        tokenizer = AutoTokenizer.from_pretrained(self._pubmedbert_path)
        device    = self.text_feat.device
        prompt_groups = build_kl_prompt_groups(KL_PROMPTS)
        if len(prompt_groups) != self.NUM_CLASSES:
            raise ValueError(f"Expected {self.NUM_CLASSES} KL prompt groups, got {len(prompt_groups)}.")

        all_feats = []
        for prompts in prompt_groups:
            enc = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            self.text_encoder.float()
            self.text_projection.float()

            with torch.cuda.amp.autocast(enabled=False):
                outputs   = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden    = outputs.last_hidden_state.float()

                cls_out   = hidden[:, 0, :]
                mask_exp  = attention_mask.unsqueeze(-1).float()
                mean_pool = (hidden * mask_exp).sum(dim=1) \
                            / mask_exp.sum(dim=1).clamp(min=1e-9)
                combined  = (cls_out + mean_pool) / 2.0

                feat = self.text_projection(combined.float())
                feat = feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
                feat = feat.mean(dim=0)
                feat = feat / feat.norm(dim=-1).clamp(min=1e-9)
            all_feats.append(feat)

        self.text_feat.copy_(torch.stack(all_feats, dim=0))
        self._text_feat_initialized = True

    def refresh_text_prototypes(self):
        """公开接口：手动刷新文本原型（文本编码器解冻微调后调用）。"""
        self._init_text_prototypes()

    # ------------------------------------------------------------------
    # 编码方法
    # ------------------------------------------------------------------

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.visual(image.type(self.dtype)).float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(self,
                image: torch.Tensor,
                labels: torch.Tensor = None
                ) -> dict:
        if not self._text_feat_initialized:
            self._init_text_prototypes()

        # 1. 图像特征
        img_feat = self.encode_image(image)                               # [B, D]

        # 2. 文本原型对齐 logits
        logits_proto = img_feat @ self.text_feat.T / self.TEMPERATURE     # [B, 5]

        # 3. 聚类软分配概率
        norm_cluster_proto = F.normalize(self.cluster_prototypes, dim=-1)
        cluster_logits     = img_feat @ norm_cluster_proto.T / self.TEMPERATURE
        cluster_prob       = F.softmax(cluster_logits, dim=-1)            # [B, 5]

        # 4. 动态文本语义生成
        cluster_txt_feat = cluster_prob @ self.text_feat                  # [B, D]
        cluster_txt_feat = cluster_txt_feat / cluster_txt_feat.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)

        # 5. 聚类感知门控残差融合
        fuse_feat = self.fusion(img_feat, cluster_txt_feat, cluster_prob) # [B, D]

        # 6. MLP 分类
        logits_cls = self.classifier(fuse_feat)                           # [B, 5]

        out = {
            "logits_cls"   : logits_cls,
            "logits_proto" : logits_proto,
            "cluster_prob" : cluster_prob,
            "fuse_feat"    : fuse_feat,
            "img_feat"     : img_feat,
        }

        # 7. 训练损失
        if self.training and labels is not None:
            loss = self._compute_loss(logits_cls, labels)
            out["loss"] = loss

        return out

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def _compute_loss(self,
                      logits_cls: torch.Tensor,
                      labels: torch.Tensor) -> torch.Tensor:
        """
        两项损失联合优化：

        1. loss_ce_cls  : 主分类 CE，监督 fuse_feat → logits_cls 的分类边界。
        2. loss_ordinal : 有序回归损失，保持 KL 分级的单调连续关系。
        """
        logits_cls = torch.nan_to_num(
            logits_cls.float(), nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)

        labels = labels.long()

        loss_ce_cls  = self.ce_loss_fn(logits_cls, labels)
        loss_ordinal = self.ordinal_loss_fn(logits_cls, labels)

        total = loss_ce_cls + self.LAMBDA_ORDINAL * loss_ordinal

        return torch.nan_to_num(total, nan=0.0, posinf=100.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        self.eval()
        out = self.forward(image, labels=None)
        return out["logits_cls"].argmax(dim=-1)


# ===== CLIP 辅助函数 =====

def convert_weights(model: nn.Module):
    def _convert_module_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    for module_name, module in model.named_modules():
        if module_name.startswith("text_encoder") or module_name.startswith("text_projection"):
            continue
        _convert_module_to_fp16(module)

    if hasattr(model, "text_encoder"):
        model.text_encoder.float()
    if hasattr(model, "text_projection"):
        model.text_projection.float()


def build_model(state_dict: dict, pubmedbert_path: str = PUBMEDBERT_PATH):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width      = state_dict["visual.conv1.weight"].shape[0]
        vision_layers     = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size         = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution  = vision_patch_size * grid_size
    else:
        counts: list      = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers     = tuple(counts)
        vision_width      = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width      = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution  = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        pubmedbert_path=pubmedbert_path,
    )

    text_keys = {
        "token_embedding.weight", "positional_embedding", "ln_final.weight",
        "ln_final.bias", "text_projection"
    }
    visual_state_dict = {
        k: v for k, v in state_dict.items()
        if not any(k == tk or k.startswith("transformer.") for tk in text_keys)
           and k not in ["input_resolution", "context_length", "vocab_size"]
    }

    missing, unexpected = model.load_state_dict(visual_state_dict, strict=False)
    print(f"[build_model] Missing keys: {[k for k in missing if 'text_encoder' in k or 'text_projection' in k]}")
    print(f"[build_model] Unexpected keys: {unexpected}")

    model.float()
    return model.eval()


def build_fusion_model(state_dict: dict,
                       pubmedbert_path: str = PUBMEDBERT_PATH,
                       freeze_text_encoder: bool = True,
                       fusion_dropout: float = 0.1,
                       cls_dropout: float = 0.2,
                       ) -> "CLIPFusionContrastiveModel":
    """
    一步到位：从 CLIP state_dict 构建 CLIPFusionContrastiveModel。

    典型用法
    --------
    state_dict = torch.load("clip_weights.pt", map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = build_fusion_model(state_dict).to("cuda")
    model.refresh_text_prototypes()
    """
    clip_base = build_model(state_dict, pubmedbert_path=pubmedbert_path)
    embed_dim = state_dict["text_projection"].shape[1]

    fusion_model = CLIPFusionContrastiveModel(
        clip_model=clip_base,
        embed_dim=embed_dim,
        fusion_dropout=fusion_dropout,
        cls_dropout=cls_dropout,
        freeze_text_encoder=freeze_text_encoder,
        pubmedbert_path=pubmedbert_path,
    )

    return fusion_model.train()


# ===== 优化器（分层学习率） =====

def build_optimizer(model: CLIPFusionContrastiveModel,
                    lr_text_encoder: float = 1e-5,
                    lr_visual: float = 5e-5,
                    lr_text_proj: float = 1e-4,
                    lr_fusion: float = 1e-4,
                    lr_classifier: float = 1e-4,
                    lr_cluster_proto: float = 1e-4,
                    weight_decay: float = 0.01) -> torch.optim.AdamW:
    """
    分层学习率 AdamW 优化器。
    """
    param_groups = [
        {"params": model.visual.parameters(),
         "lr": lr_visual,        "name": "visual"},
        {"params": model.text_projection.parameters(),
         "lr": lr_text_proj,     "name": "text_projection"},
        {"params": model.fusion.parameters(),
         "lr": lr_fusion,        "name": "fusion"},
        {"params": model.classifier.parameters(),
         "lr": lr_classifier,    "name": "classifier"},
        {"params": [model.cluster_prototypes],
         "lr": lr_cluster_proto, "name": "cluster_prototypes"},
    ]

    trainable_text_params = [p for p in model.text_encoder.parameters() if p.requires_grad]
    if trainable_text_params:
        param_groups.append({
            "params": trainable_text_params,
            "lr": lr_text_encoder, "name": "text_encoder"
        })

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
