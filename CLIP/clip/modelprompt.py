from collections import OrderedDict
from typing import Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoTokenizer, AutoModel


# =============================================================================
# 视觉编码器相关类（完全保留原始 CLIP 代码，无任何修改）
# =============================================================================

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
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
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
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
        self.positional_embedding = nn.Parameter(
            torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
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
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
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

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
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
    """Subclass torch's LayerNorm to handle fp16."""

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
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None else None
        )
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
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int,
                 layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=width,
            kernel_size=patch_size, stride=patch_size, bias=False
        )

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(
            scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width)
        )
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) +
             torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1
        )  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


# =============================================================================
# 新增：PubMedBERT 文本编码器
# 替换原 CLIP 中基于 token_embedding + Transformer 的文本编码器
# =============================================================================

class PubMedBERTEncoder(nn.Module):
    """
    使用本地 PubMedBERT 作为文本编码器，替换原 CLIP 的文本 Transformer。

    主要变化：
      - 输入从 token id 张量改为原始字符串列表，tokenize 在内部完成
      - 句子表示从 eot token 特征改为 BERT [CLS] token 特征（语义等价）
      - 增加线性投影层将 BERT 的 hidden_size (768) 对齐到 CLIP embed_dim
    """

    def __init__(self, model_path: str, embed_dim: int, freeze: bool = False):
        """
        Args:
            model_path: PubMedBERT 本地权重路径
            embed_dim:  目标嵌入维度，需与视觉编码器输出维度一致
            freeze:     是否冻结 PubMedBERT 全部参数（仅训练投影层）
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.bert = AutoModel.from_pretrained(model_path)

        if freeze:
            for param in self.bert.parameters():
                param.requires_grad = False

        # PubMedBERT hidden_size 通常为 768，投影到 CLIP embed_dim
        bert_hidden_size = self.bert.config.hidden_size
        self.projection = nn.Linear(bert_hidden_size, embed_dim)

        # 记录输出维度，方便外部查询
        self.output_dim = embed_dim

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        """
        Args:
            texts:  字符串列表，例如 ["KL grade 0", "KL grade 1", ...]
            device: 当前运算设备
        Returns:
            text_features: FloatTensor，形状 [len(texts), embed_dim]
        """
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        outputs = self.bert(**encoded)

        # 取 [CLS] token 作为整句表示（对应原 CLIP 的 eot_token 特征提取）
        cls_features = outputs.last_hidden_state[:, 0, :]  # [B, hidden_size]

        # 投影到目标维度
        text_features = self.projection(cls_features)      # [B, embed_dim]
        return text_features


# =============================================================================
# 修改后的 CLIP 主类：视觉编码器保持不变，文本编码器替换为 PubMedBERT
# =============================================================================

class CLIPWithPubMedBERT(nn.Module):
    """
    与原 CLIP 的主要差异：
      1. 文本编码器由 (token_embedding + Transformer + text_projection) 替换为
         PubMedBERTEncoder（本地 PubMedBERT + 线性投影）
      2. encode_text() 接受字符串列表而非 token id 张量
      3. 删除了 context_length / vocab_size / token_embedding /
         positional_embedding / ln_final / text_projection 等原文本侧参数
      4. 视觉编码器及其初始化逻辑与原 CLIP 完全一致
    """

    def __init__(self,
                 embed_dim: int,
                 # ---------- 视觉侧（与原 CLIP 完全一致）----------
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # ---------- 文本侧（替换为 PubMedBERT）----------
                 pubmedbert_path: str,
                 freeze_text: bool = False,
                 freeze_vision: bool = False):
        """
        Args:
            embed_dim:         图文共享嵌入空间的维度
            image_resolution:  输入图像分辨率（正方形边长）
            vision_layers:     视觉编码器层数；若为 tuple 则使用 ModifiedResNet，
                               若为 int 则使用 VisionTransformer
            vision_width:      视觉编码器宽度
            vision_patch_size: ViT patch 大小（仅 VisionTransformer 时有效）
            pubmedbert_path:   PubMedBERT 本地路径
            freeze_text:       是否冻结 PubMedBERT 权重
            freeze_vision:     是否冻结视觉编码器权重
        """
        super().__init__()

        # ------------------------------------------------------------------ #
        # 视觉编码器（与原 CLIP.__init__ 逻辑完全一致）
        # ------------------------------------------------------------------ #
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

        if freeze_vision:
            for param in self.visual.parameters():
                param.requires_grad = False

        # ------------------------------------------------------------------ #
        # 文本编码器（替换为 PubMedBERT）
        # ------------------------------------------------------------------ #
        self.text_encoder = PubMedBERTEncoder(
            model_path=pubmedbert_path,
            embed_dim=embed_dim,
            freeze=freeze_text
        )

        # 可学习温度参数（与原 CLIP 完全一致）
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 初始化视觉编码器权重（与原 CLIP.initialize_parameters 一致）
        self._initialize_visual_parameters()

    def _initialize_visual_parameters(self):
        """仅初始化视觉编码器参数，文本侧由 PubMedBERT 自带预训练权重。"""
        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [
                self.visual.layer1, self.visual.layer2,
                self.visual.layer3, self.visual.layer4
            ]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像编码，与原 CLIP.encode_image 完全一致。"""
        return self.visual(image.type(self.dtype))

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        文本编码。
        原 CLIP 接受 token id 张量；此处改为接受字符串列表，
        tokenize 由 PubMedBERTEncoder 内部完成。
        """
        device = next(self.parameters()).device
        features = self.text_encoder(texts, device)  # [B, embed_dim]，float32
        return features.type(self.dtype)

    def forward(self, image: torch.Tensor, texts: List[str]):
        """
        Args:
            image: FloatTensor [B, 3, H, W]
            texts: 长度为 B 的字符串列表
        Returns:
            logits_per_image: [B, B]  图对文相似度矩阵
            logits_per_text:  [B, B]  文对图相似度矩阵
        """
        image_features = self.encode_image(image)
        text_features = self.encode_text(texts)

        # L2 归一化（与原 CLIP 一致）
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        # 余弦相似度作为 logits（与原 CLIP 一致）
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


# =============================================================================
# 膝关节炎五分类器（复用上述模型）
# =============================================================================

class KneeOAClassifier(nn.Module):
    """
    基于 CLIPWithPubMedBERT 的膝关节炎 KL 分级分类器，支持两种推理模式：

    zero_shot=True:
        直接计算图像与 5 条 KL 提示文本的余弦相似度，取最高者作为预测类别。
        无需额外训练，适合快速验证。

    zero_shot=False（默认，推荐）:
        在图像嵌入后接 MLP 分类头进行有监督微调，适合正式训练。
    """

    # KL 0–4 级的医学提示文本
    KL_PROMPTS: List[str] = [
        "knee X-ray showing no signs of osteoarthritis, Kellgren-Lawrence grade 0",
        "knee X-ray showing doubtful joint space narrowing, Kellgren-Lawrence grade 1",
        "knee X-ray with definite osteophytes and possible joint space narrowing, "
        "Kellgren-Lawrence grade 2",
        "knee X-ray with moderate multiple osteophytes and definite joint space narrowing, "
        "Kellgren-Lawrence grade 3",
        "knee X-ray with large osteophytes and severe joint space narrowing or bone deformity, "
        "Kellgren-Lawrence grade 4",
    ]

    def __init__(self,
                 clip_model: CLIPWithPubMedBERT,
                 num_classes: int = 5,
                 zero_shot: bool = False,
                 dropout: float = 0.3):
        """
        Args:
            clip_model:   已实例化的 CLIPWithPubMedBERT
            num_classes:  分类数（KL 分级默认为 5）
            zero_shot:    是否使用零样本推理
            dropout:      MLP 分类头的 Dropout 概率
        """
        super().__init__()
        self.clip = clip_model
        self.zero_shot = zero_shot

        if not zero_shot:
            embed_dim = clip_model.visual.output_dim
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, 256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, num_classes)
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: FloatTensor [B, 3, H, W]，需已完成归一化
        Returns:
            logits: [B, num_classes]
        """
        if self.zero_shot:
            # Zero-shot：图像与各 KL 提示的相似度直接作为 logits
            logits, _ = self.clip(image, self.KL_PROMPTS)
            return logits                                   # [B, 5]
        else:
            # 有监督：图像嵌入 → MLP 分类头
            image_features = self.clip.encode_image(image).float()
            return self.classifier(image_features)          # [B, 5]


# =============================================================================
# 工具函数：复用原 CLIP 权重中的视觉编码器部分
# =============================================================================

def convert_weights(model: nn.Module):
    """将模型中适用的参数转换为 fp16（与原 CLIP 完全一致）。"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias", "bias_k", "bias_v"
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model_with_pubmedbert(
        clip_state_dict: dict,
        pubmedbert_path: str,
        freeze_text: bool = False,
        freeze_vision: bool = False
) -> CLIPWithPubMedBERT:
    """
    从原始 CLIP state_dict 推断视觉编码器配置，构建 CLIPWithPubMedBERT。
    文本侧参数从 state_dict 中丢弃，改用本地 PubMedBERT。

    Args:
        clip_state_dict:  原始 CLIP 模型的 state_dict（通过 torch.load 加载）
        pubmedbert_path:  PubMedBERT 本地路径
        freeze_text:      是否冻结 PubMedBERT
        freeze_vision:    是否冻结视觉编码器
    Returns:
        model: CLIPWithPubMedBERT 实例（已加载视觉编码器权重，eval 模式）
    """
    # ---- 判断视觉编码器类型（与原 build_model 逻辑一致）----
    vit = "visual.proj" in clip_state_dict

    if vit:
        vision_width = clip_state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in clip_state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = clip_state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (clip_state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(set(k.split(".")[2] for k in clip_state_dict
                    if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = clip_state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (clip_state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert output_width ** 2 + 1 == \
               clip_state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = clip_state_dict["text_projection"].shape[1]

    # ---- 构建模型 ----
    model = CLIPWithPubMedBERT(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        pubmedbert_path=pubmedbert_path,
        freeze_text=freeze_text,
        freeze_vision=freeze_vision
    )

    # ---- 只加载视觉编码器权重，丢弃文本侧权重 ----
    visual_state = {
        k.replace("visual.", ""): v
        for k, v in clip_state_dict.items()
        if k.startswith("visual.")
    }
    model.visual.load_state_dict(visual_state, strict=True)

    # logit_scale 也可以从原始权重中恢复
    if "logit_scale" in clip_state_dict:
        model.logit_scale.data = clip_state_dict["logit_scale"]

    convert_weights(model)
    return model.eval()


