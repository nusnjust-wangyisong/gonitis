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

# ── 分支特异性文本描述（每级 8 条，与 promptyuan.py 量级对齐）─────────────────
# 全局分支（ViT）：聚焦双侧对称性、整体关节间隙与骨性结构排列
# 每条描述刻意回避骨赘/硬化等局部病理词汇，强化 ViT 对全局解剖结构的监督信号
ANATOMY_TEXTS = [
    # KL0 — 正常
    [
        "Bilateral knee joint spaces are symmetric and well-preserved, with smooth and regular articular contours maintained throughout.",
        "Both knee joints demonstrate normal and equal joint space width, with preserved alignment of femoral and tibial articular surfaces.",
        "Normal bilateral joint space distribution observed; femoral condyles and tibial plateaus are symmetrically aligned with no structural deviation.",
        "Tibiofemoral joint spaces are symmetric and within normal limits; overall skeletal alignment and mechanical axis are normal.",
        "Articular surfaces of both knees are smooth and well-defined; bilateral joint space is uniform and symmetrically maintained.",
        "Joint space width is symmetric bilaterally; bone ends are normally shaped with no deformity of the articular contour.",
        "Both knees show normal joint space width, well-aligned articular surfaces, and preserved bilateral skeletal symmetry.",
        "Radiographic examination shows symmetric bilateral joint spaces with smooth articular surfaces and normal mechanical axis alignment.",
    ],
    # KL1 — 可疑
    [
        "Bilateral knee joint space shows possible mild asymmetry; overall joint space width remains largely preserved with maintained skeletal alignment.",
        "Joint space is predominantly maintained bilaterally, with questionable slight narrowing; articular contours remain largely smooth and symmetric.",
        "Slight possible reduction in joint space width observed, while overall bilateral alignment is preserved.",
        "Joint space width is largely symmetric with a possible equivocal narrowing; skeletal alignment and mechanical axis remain normal.",
        "Articular surfaces appear smooth with possible subtle joint space reduction; bilateral knee alignment is maintained.",
        "Minimal possible joint space asymmetry noted; overall tibiofemoral alignment and bone end morphology remain within normal limits.",
        "Possible early reduction in tibiofemoral joint space width; bilateral joint symmetry and skeletal alignment are essentially preserved.",
        "Questionable mild joint space narrowing in one or both knees; overall bilateral articular alignment and mechanical axis remain normal.",
    ],
    # KL2 — 轻度
    [
        "Definite reduction in joint space width is noted; bilateral knee alignment is largely maintained with possible mild asymmetry.",
        "Joint space narrowing is present, primarily in the medial or lateral compartment; overall bilateral skeletal alignment is preserved.",
        "Tibiofemoral joint space shows clear narrowing; articular surface alignment is maintained with mild asymmetry.",
        "Clear joint space reduction observed, predominantly in one compartment; bilateral skeletal alignment and mechanical axis are largely maintained.",
        "Definite narrowing of tibiofemoral joint space is present; bilateral skeletal alignment shows possible early asymmetry.",
        "Joint space is clearly reduced compared to normal; overall articular alignment and bilateral symmetry are largely preserved.",
        "Tibiofemoral joint space demonstrates definite narrowing; bilateral knee alignment shows possible mild deviations from symmetric.",
        "Clear joint space reduction is present in one or both knees; overall skeletal alignment and bilateral symmetry are maintained.",
    ],
    # KL3 — 中度
    [
        "Definite and moderate joint space narrowing is present; skeletal alignment shows moderate asymmetry with early structural deformity.",
        "Significant reduction in tibiofemoral joint space observed; bilateral knee alignment demonstrates moderate deformity or asymmetry.",
        "Moderate joint space narrowing is evident, with bilateral skeletal alignment showing clear asymmetry or varus/valgus deviation.",
        "Joint space is moderately narrowed throughout; bilateral mechanical axis shows deformity and structural malalignment.",
        "Tibiofemoral joint space demonstrates substantial narrowing; bilateral skeletal alignment shows moderate structural deviation.",
        "Moderate reduction in joint space is present; overall bilateral knee alignment is disrupted with evident asymmetry.",
        "Joint space shows significant narrowing in one or both compartments; bilateral skeletal alignment is moderately compromised.",
        "Moderate to severe joint space narrowing is present; bilateral knee alignment and mechanical axis demonstrate significant deviation.",
    ],
    # KL4 — 重度
    [
        "Severe or complete obliteration of joint space is present; bilateral skeletal alignment shows marked deformity and asymmetry.",
        "Tibiofemoral joint space is nearly or completely absent; skeletal alignment demonstrates severe deformity of bone ends.",
        "Joint space is markedly or completely lost; bilateral mechanical axis shows severe structural deformity.",
        "Severe narrowing or complete loss of joint space is evident; overall bilateral knee alignment is grossly deformed.",
        "Joint space obliteration is present; bilateral skeletal structure and articular alignment demonstrate advanced deformity.",
        "Complete or near-complete joint space loss is observed; bilateral knee morphology is severely distorted with marked malalignment.",
        "Joint space is absent or markedly obliterated; bilateral skeletal alignment shows severe deformity of articular ends.",
        "Severe joint space loss is present throughout; bilateral mechanical axis is severely deviated with gross articular deformity.",
    ],
]

# 局部分支（ConvNeXt）：聚焦骨赘形态、软骨下骨硬化与软骨局部病理
# 每条描述刻意回避双侧对称/整体排列词汇，强化 ConvNeXt 对局部病理细节的监督信号
PATHOLOGY_TEXTS = [
    # KL0 — 正常
    [
        "No osteophyte formation is observed at any joint margin; subchondral bone is smooth without sclerosis, and cartilage appears intact.",
        "Articular bone surfaces show no osteophytes or bony spurs; trabecular pattern is homogeneous with no subchondral densification.",
        "No marginal bony projections are present; subchondral bone density is uniform without sclerosis, cysts, or degenerative changes.",
        "Absence of osteophytes at tibial and femoral margins; bone density is normal with no subchondral sclerosis or trabecular disruption.",
        "No bony outgrowths or marginal lipping observed; subchondral bone shows no increased density or cystic changes.",
        "Joint margins are smooth without osteophyte formation; subchondral bone is intact with uniform density and no degenerative trabecular changes.",
        "No osteophytic lipping or marginal bony spurs at any knee joint margin; subchondral bone and cartilage are unremarkable.",
        "Bone surfaces are free of osteophytes; trabecular pattern is normal, subchondral bone is smooth, and no cartilage thinning is evident.",
    ],
    # KL1 — 可疑
    [
        "Small or doubtful marginal osteophyte is present at one or more joint margins; subchondral bone shows minimal change with no significant sclerosis.",
        "Possible early osteophytic lipping at the tibial or femoral condylar margins; subchondral bone density remains largely normal.",
        "Questionable small bony spur at joint margin; trabecular bone is intact and subchondral layer shows no significant densification.",
        "Minimal osteophyte formation is questionably present; subchondral bone shows no sclerosis or cysts, and cartilage thinning is absent.",
        "Faint lip-shaped osteophyte is visible at joint margin; subchondral bone density is normal with no degenerative trabecular changes.",
        "Small possible osteophyte at tibial or femoral margin; bone surface changes are minimal with no significant subchondral involvement.",
        "Marginal bony projection of doubtful significance; subchondral bone is without sclerosis or cysts, trabecular pattern intact.",
        "Early small osteophyte possibly present at joint periphery; subchondral bone and cartilage show no significant pathological change.",
    ],
    # KL2 — 轻度
    [
        "Definite small osteophytes present at femoral and tibial margins; mild subchondral sclerosis is present with possible minimal cartilage thinning.",
        "Clear osteophytic formation at joint margins; subchondral bone shows mild increased density without significant cysts or erosion.",
        "Definite marginal osteophytes of small to moderate size; subchondral sclerosis is mild, with no prominent cysts or severe cartilage loss.",
        "Small definite osteophytes at joint margins; subchondral bone density is mildly increased with possible early cartilage changes.",
        "Definite lip-shaped osteophytes visible; subchondral bone shows mild sclerosis and minimal cartilage thinning without bone cysts.",
        "Clear osteophyte formation at tibial plateau and femoral condyle; mild subchondral bone changes present without significant cartilage destruction.",
        "Definite small to moderate osteophytes at joint margins; subchondral layer shows mild sclerosis with possible minimal cartilage thinning.",
        "Small definite marginal osteophytes present; mild subchondral sclerosis is noted with minimal cartilage loss and no prominent bone cysts.",
    ],
    # KL3 — 中度
    [
        "Multiple moderate to large osteophytes at joint margins; subchondral sclerosis is moderate to severe with subchondral bone cysts present.",
        "Multiple osteophytes are visible at tibial and femoral margins; subchondral bone shows significant densification and bone cyst formation.",
        "Multiple bony spurs of varying size; subchondral sclerosis is moderate, with bone cysts and significant cartilage loss evident.",
        "Several moderate osteophytes present; subchondral bone sclerosis is prominent, with cystic changes and substantial cartilage degradation.",
        "Multiple marginal osteophytes of moderate size; subchondral bone shows moderate to severe sclerosis and subchondral cysts.",
        "Multiple osteophytes along tibial and femoral joint margins; subchondral densification is present with bone cysts and cartilage loss.",
        "Prominent multiple osteophytes at joint margins; moderate subchondral sclerosis with bone cyst formation and notable cartilage thinning.",
        "Multiple moderate osteophytes visible; subchondral bone shows significant sclerosis, bone cysts, and moderate to severe cartilage loss.",
    ],
    # KL4 — 重度
    [
        "Large multiple osteophytes at all joint margins; subchondral bone shows severe sclerosis with prominent bone cysts and near-complete cartilage destruction.",
        "Extensive large osteophytes at tibial and femoral margins; severe subchondral sclerosis with large bone cysts and advanced cartilage loss.",
        "Multiple large bony spurs and marginal osteophytes; severe subchondral densification, prominent bone cysts, and extensive cartilage destruction.",
        "Large osteophytes throughout joint margins; subchondral bone shows severe sclerosis and extensive cystic degeneration with cartilage obliteration.",
        "Diffuse large osteophyte formation; severe subchondral bone sclerosis with prominent cysts and complete or near-complete cartilage loss.",
        "Extensive osteophytes of large size at all margins; severe subchondral sclerosis, large bone cysts, and advanced bone end destruction.",
        "Multiple large osteophytes throughout; severe subchondral bone sclerosis with prominent cystic changes and severe cartilage loss.",
        "Large and extensive marginal osteophytes; subchondral bone shows severe sclerosis, large cysts, and complete articular cartilage destruction.",
    ],
]

# 等级过渡文本（GTP）：描述相邻 KL 等级之间的病理演进过程
TRANSITION_TEXTS = [
    "transition from normal to doubtful osteoarthritis with first appearance of marginal osteophyte formation",
    "transition from doubtful to minimal osteoarthritis with definite osteophytes and possible joint space reduction",
    "transition from minimal to moderate osteoarthritis with multiple osteophytes and definite joint space narrowing",
    "transition from moderate to severe osteoarthritis with extensive bone changes and severe joint space loss",
]

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

    def forward_tokens(self, x: torch.Tensor):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_post(x)
        if self.proj is not None:
            x = x @ self.proj
        return x[:, 0, :], x[:, 1:, :]

    def forward(self, x: torch.Tensor):
        cls_token, _ = self.forward_tokens(x)
        return cls_token


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

def build_kl_prompt_groups(prompt_bank, dataset_name: str = "KneeOA"):
    """
    Normalize prompt definitions to List[List[str]], one prompt list per KL class.
    Supports both the local nested prompt dict and a simple list of class prompts.
    """
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


class OrdinalCELoss(nn.Module):
    """
    BCE loss over cumulative KL thresholds. It penalizes ordinal distance between
    adjacent KL grades more appropriately than plain nominal classification.
    """
    def __init__(self, num_classes: int, clamp_value: float = 20.0):
        super().__init__()
        self.num_classes = num_classes
        self.clamp_value = clamp_value

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(
            logits.float(), nan=0.0, posinf=self.clamp_value, neginf=-self.clamp_value
        ).clamp(-self.clamp_value, self.clamp_value)
        targets = targets.long()
        cumulative_targets = torch.stack(
            [targets > k for k in range(self.num_classes - 1)], dim=1
        ).float().to(logits.device)
        cumulative_logits = logits[:, :self.num_classes - 1]
        return F.binary_cross_entropy_with_logits(cumulative_logits, cumulative_targets)


class OrdinalLabelDistributionLoss(nn.Module):
    """
    Cross entropy with an ordinal Gaussian target distribution.
    Adjacent KL grades receive more target mass than distant grades, which is
    better aligned with ordered KL severity than uniform label smoothing.
    """
    def __init__(self, num_classes: int, sigma: float = 0.8):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = float(sigma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        logits = logits.clamp(-20.0, 20.0)
        targets = targets.long()
        class_ids = torch.arange(self.num_classes, device=logits.device).float()
        distances = class_ids.unsqueeze(0) - targets.float().unsqueeze(1)
        soft_targets = torch.exp(-0.5 * (distances / max(self.sigma, 1e-6)) ** 2)
        soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-9)
        log_probs = F.log_softmax(logits, dim=1)
        return -(soft_targets * log_probs).sum(dim=1).mean()


class OrdinalBoundaryHead(nn.Module):
    """
    Predict cumulative KL boundaries P(y > 0), ..., P(y > 3) from the fused
    visual-language feature. The cumulative probabilities can be converted back
    to a five-class distribution while preserving the ordered KL structure.
    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_classes: int = 5, dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes - 1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat.float())


class ResidualFeatureAdapter(nn.Module):
    """
    Low-rank residual adapter for CLIP image features. It gives the model a
    small trainable medical-domain correction while keeping the pretrained
    visual backbone and downstream classifier largely intact.
    """
    def __init__(self, embed_dim: int, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.adapter_logit = nn.Parameter(torch.tensor(-4.0))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        residual = self.net(feat.float())
        scale = torch.sigmoid(self.adapter_logit)
        adapted = feat.float() + scale * residual
        adapted = torch.nan_to_num(adapted, nan=0.0, posinf=1.0, neginf=-1.0)
        return adapted / adapted.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def boundary_logits_to_class_logits(boundary_logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert cumulative boundary logits into nominal class logits.
    p0 = 1 - P(y>0), p1 = P(y>0)-P(y>1), ..., p4 = P(y>3).
    A cumulative minimum enforces monotonic boundaries during inference.
    """
    p_gt = torch.sigmoid(boundary_logits.float())
    p_gt = torch.cummin(p_gt, dim=1).values
    probs = torch.zeros(p_gt.shape[0], p_gt.shape[1] + 1, device=p_gt.device, dtype=p_gt.dtype)
    probs[:, 0] = 1.0 - p_gt[:, 0]
    for idx in range(1, probs.shape[1] - 1):
        probs[:, idx] = p_gt[:, idx - 1] - p_gt[:, idx]
    probs[:, -1] = p_gt[:, -1]
    probs = probs.clamp(min=eps)
    probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=eps)
    return probs.log()


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


class DualClusterAwareResidualGatedFusion(nn.Module):
    """
    Dual visual-text cluster gated fusion.

    Compared with ClusterAwareResidualGatedFusion, this module injects both:
      1) dynamic KL text feature from text prototypes
      2) visual cluster feature from learnable visual KL prototypes

    The gate is sample-adaptive and receives visual/text interactions plus
    cluster probabilities, so ambiguous KL boundary samples can use different
    evidence ratios.
    """
    def __init__(self,
                 embed_dim: int,
                 num_classes: int,
                 hidden_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim * 2
        gate_input_dim = embed_dim * 6 + num_classes * 2

        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim * 2),
            nn.Sigmoid(),
        )
        self.visual_residual_logit = nn.Parameter(torch.tensor(-5.0))
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self,
                img_feat: torch.Tensor,
                cluster_txt_feat: torch.Tensor,
                visual_cluster_feat: torch.Tensor,
                cluster_prob: torch.Tensor,
                visual_cluster_prob: torch.Tensor) -> torch.Tensor:
        img_feat = torch.nan_to_num(img_feat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        cluster_txt_feat = torch.nan_to_num(cluster_txt_feat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        visual_cluster_feat = torch.nan_to_num(visual_cluster_feat.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        cluster_prob = torch.nan_to_num(cluster_prob.float(), nan=0.0, posinf=1.0, neginf=0.0)
        visual_cluster_prob = torch.nan_to_num(visual_cluster_prob.float(), nan=0.0, posinf=1.0, neginf=0.0)

        gate_input = torch.cat([
            img_feat,
            cluster_txt_feat,
            visual_cluster_feat,
            img_feat * cluster_txt_feat,
            img_feat * visual_cluster_feat,
            cluster_txt_feat * visual_cluster_feat,
            cluster_prob,
            visual_cluster_prob,
        ], dim=-1)
        gate_input = torch.nan_to_num(gate_input, nan=0.0, posinf=1.0, neginf=-1.0)

        gates = self.gate_mlp(gate_input)
        gate_text, gate_visual = gates.chunk(2, dim=-1)
        visual_scale = torch.sigmoid(self.visual_residual_logit)
        fused = self.layer_norm(
            img_feat
            + gate_text * cluster_txt_feat
            + visual_scale * gate_visual * visual_cluster_feat
        )
        fused = torch.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=-1.0)
        return fused / fused.norm(dim=-1, keepdim=True).clamp(min=1e-6)


class SemanticPatchAttention(nn.Module):
    """
    KL text guided patch attention over ViT patch tokens.

    Each KL text prototype acts as a semantic query over image patches. The
    resulting class-specific regional features produce an additional set of
    logits that focuses on local radiographic cues instead of only the global
    CLS token.
    """
    def __init__(self,
                 embed_dim: int,
                 num_classes: int = 5,
                 bottleneck_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.patch_adapter = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, bottleneck_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, embed_dim),
        )
        self.adapter_logit = nn.Parameter(torch.tensor(-4.0))
        self.attn_logit_scale = nn.Parameter(torch.tensor(np.log(10.0)))
        self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.07)))

    def forward(self, patch_feat: torch.Tensor, text_feat: torch.Tensor):
        patch_feat = torch.nan_to_num(
            patch_feat.float(), nan=0.0, posinf=1.0, neginf=-1.0
        )
        patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        residual = self.patch_adapter(patch_feat)
        patch_feat = patch_feat + torch.sigmoid(self.adapter_logit) * residual
        patch_feat = torch.nan_to_num(patch_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        text_feat = F.normalize(text_feat.float(), dim=-1)
        attn_scale = self.attn_logit_scale.exp().clamp(1.0, 50.0)
        attn_scores = torch.einsum("bpd,cd->bcp", patch_feat, text_feat) * attn_scale
        attn = F.softmax(attn_scores, dim=-1)
        class_patch_feat = torch.einsum("bcp,bpd->bcd", attn, patch_feat)
        class_patch_feat = F.normalize(class_patch_feat, dim=-1)

        patch_logits = (class_patch_feat * text_feat.unsqueeze(0)).sum(dim=-1)
        patch_logits = patch_logits * self.logit_scale.exp().clamp(1.0, 100.0)
        patch_prob = F.softmax(patch_logits, dim=-1)
        pooled_patch_feat = (patch_prob.unsqueeze(-1) * class_patch_feat).sum(dim=1)
        pooled_patch_feat = F.normalize(pooled_patch_feat, dim=-1)
        return patch_logits, pooled_patch_feat, attn


def prototype_separation_loss(prototypes: torch.Tensor, margin: float = 0.15) -> torch.Tensor:
    """Penalize overly similar class prototypes to reduce prototype collapse."""
    proto = F.normalize(prototypes.float(), dim=-1)
    sim = proto @ proto.t()
    eye = torch.eye(sim.shape[0], device=sim.device, dtype=torch.bool)
    off_diag = sim.masked_fill(eye, -1.0)
    return F.relu(off_diag - margin).pow(2).mean()


def emd_ordinal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Squared Earth Mover's Distance loss for ordinal KL grading.

    Penalizes predictions proportional to ordinal distance from the true grade.
    CDF formulation: L = mean( ||CDF_pred - CDF_true||^2 )

    Unlike cross-entropy which treats all misclassifications equally, EMD loss
    assigns higher penalty to predictions farther from the true grade on the
    0-4 ordinal scale (e.g., KL0 predicted as KL4 is penalized much more than
    KL0 predicted as KL1).
    """
    probs = F.softmax(logits.float(), dim=-1)          # (B, K)
    K = logits.size(1)
    target_one_hot = F.one_hot(labels.long(), K).float().to(logits.device)
    pred_cdf   = torch.cumsum(probs, dim=1)            # (B, K)
    target_cdf = torch.cumsum(target_one_hot, dim=1)   # (B, K)
    return (pred_cdf - target_cdf).pow(2).mean()


def branch_decoupling_loss(feat_global: torch.Tensor, feat_local: torch.Tensor) -> torch.Tensor:
    """Complementarity regularization for dual-branch fusion.

    Encourages ViT (global) and ConvNeXt (local) pre-fusion features to
    extract complementary, non-redundant information. Penalizes the squared
    per-sample cosine similarity between the two branches.

    Without this, both branches can converge to similar representations under
    joint training, reducing the benefit of the dual-branch architecture.
    """
    f_g = F.normalize(feat_global.float(), dim=-1)  # (B, D)
    f_l = F.normalize(feat_local.float(), dim=-1)   # (B, D)
    cos_sim = (f_g * f_l).sum(dim=1)                # (B,)
    return cos_sim.pow(2).mean()


class OrdinalPromptLearner(nn.Module):
    """
    CoOp-style learnable text prompts with ordinal monotonicity constraint.

    For each KL grade k (0–4) the PubMedBERT input becomes:
        [CLS] [ctx_1] … [ctx_n] [grade_token_1] … [grade_token_m] [SEP]

    [ctx_i] are shared learnable context vectors injected into BERT's
    embedding space.  Gradient flows through ctx while BERT weights stay
    frozen.  An ordinal triplet loss encourages adjacent grade prototypes
    to be more similar than distant ones, preserving KL severity ordering.

    Parameters
    ----------
    text_encoder   : PubMedBERT – frozen reference, NOT owned by this module
    text_projection: linear projection – frozen reference, NOT owned
    tokenizer_path : path to PubMedBERT tokenizer
    embed_dim      : projected feature dimension (512)
    num_classes    : KL grades (5)
    n_ctx          : number of learnable context tokens
    dropout        : dropout applied to context tokens during training
    """

    def __init__(self,
                 text_encoder: nn.Module,
                 text_projection: nn.Module,
                 tokenizer_path: str,
                 embed_dim: int = 512,
                 num_classes: int = 5,
                 n_ctx: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.n_ctx = n_ctx
        self.embed_dim = embed_dim
        self._tokenizer_path = tokenizer_path

        hidden_size = text_encoder.config.hidden_size

        # Shared learnable context tokens [n_ctx, hidden_size]
        self.ctx = nn.Parameter(torch.empty(n_ctx, hidden_size))
        nn.init.normal_(self.ctx, std=0.02)
        self.drop = nn.Dropout(dropout)

        # Store encoder/projection as plain dict entries to avoid submodule
        # registration; their params already belong to the parent model.
        self.__dict__['_text_encoder_ref'] = text_encoder
        self.__dict__['_text_proj_ref'] = text_projection

        # Token cache – built lazily on first forward
        self._cache_built = False
        self.register_buffer('_cached_ids',  torch.zeros(1, dtype=torch.long), persistent=False)
        self.register_buffer('_cached_mask', torch.zeros(1, dtype=torch.long), persistent=False)

    def _build_cache(self, device: torch.device):
        from transformers import AutoTokenizer
        groups = build_kl_prompt_groups(KL_PROMPTS)
        tok    = AutoTokenizer.from_pretrained(self._tokenizer_path)
        texts  = [g[0] for g in groups]
        enc    = tok(texts, return_tensors='pt', padding=True,
                     truncation=True, max_length=100)
        self._cached_ids  = enc['input_ids'].to(device)
        self._cached_mask = enc['attention_mask'].to(device)
        self._cache_built = True

    def forward(self) -> torch.Tensor:
        """Prompted text features for all KL grades: [num_classes, embed_dim]."""
        device = self.ctx.device
        if not self._cache_built or self._cached_ids.device != device:
            self._build_cache(device)

        enc  = self._text_encoder_ref
        proj = self._text_proj_ref
        enc.float(); proj.float()

        C    = self.num_classes
        ids  = self._cached_ids    # [C, L]
        mask = self._cached_mask   # [C, L]

        with torch.cuda.amp.autocast(enabled=False):
            # 1. Word embeddings (frozen lookup)
            tok_emb = enc.embeddings.word_embeddings(ids).float()   # [C, L, H]

            # 2. Inject: [CLS] + [ctx×n] + [remaining tokens]
            ctx_exp = self.ctx.unsqueeze(0).expand(C, -1, -1).float()
            ctx_exp = self.drop(ctx_exp)
            combined_emb = torch.cat([
                tok_emb[:, :1, :],   # CLS
                ctx_exp,             # learnable context
                tok_emb[:, 1:, :],   # rest of tokens
            ], dim=1)                                                # [C, 1+n+L-1, H]

            # 3. Extended attention mask
            ctx_mask = torch.ones(C, self.n_ctx, dtype=mask.dtype, device=device)
            ext_mask = torch.cat([mask[:, :1], ctx_mask, mask[:, 1:]], dim=1)

            # 4. Position + token-type embeddings
            S       = combined_emb.shape[1]
            pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(C, -1)
            tt_ids  = torch.zeros_like(pos_ids)
            pos_emb = enc.embeddings.position_embeddings(pos_ids).float()
            tt_emb  = enc.embeddings.token_type_embeddings(tt_ids).float()
            full_emb = enc.embeddings.LayerNorm(combined_emb + pos_emb + tt_emb)
            full_emb = enc.embeddings.dropout(full_emb)

            # 5. Encoder forward
            bert_mask = (1.0 - ext_mask[:, None, None, :].float()) * -10000.0
            hidden = enc.encoder(full_emb, attention_mask=bert_mask).last_hidden_state.float()

            # 6. Pool: (CLS + masked mean) / 2
            cls_out   = hidden[:, 0, :]
            m_exp     = ext_mask.unsqueeze(-1).float()
            mean_pool = (hidden * m_exp).sum(1) / m_exp.sum(1).clamp(min=1e-9)
            pooled    = (cls_out + mean_pool) / 2.0

            # 7. Project → L2-normalise
            feat = F.normalize(proj(pooled), dim=-1)                # [C, D]

        return feat

    @staticmethod
    def ordinal_contrastive_loss(text_feat: torch.Tensor,
                                  margin: float = 0.1) -> torch.Tensor:
        """
        Ordinal triplet loss.  For every ordered triple (i < j < k):
            sim(T_i, T_j) ≥ sim(T_i, T_k) + margin   (closer is more similar)
            sim(T_k, T_j) ≥ sim(T_k, T_i) + margin
        Enforces the KL severity ordering in text prototype space.
        """
        feat = F.normalize(text_feat.float(), dim=-1)
        sim  = feat @ feat.T
        C    = feat.shape[0]
        loss = feat.new_zeros(1).squeeze()
        n    = 0
        for i in range(C):
            for j in range(i + 1, C):
                for k in range(j + 1, C):
                    loss = loss + F.relu(sim[i, k] - sim[i, j] + margin)
                    loss = loss + F.relu(sim[k, i] - sim[k, j] + margin)
                    n   += 2
        return loss / max(n, 1)


def ordinal_prototype_loss(prototypes: torch.Tensor,
                           margin: float = 0.1) -> torch.Tensor:
    """
    Ordinal monotonicity loss for cluster prototypes.

    Same triplet formulation as OrdinalPromptLearner.ordinal_contrastive_loss:
    for every ordered triple (i < j < k), prototype j should be more similar
    to i than prototype k, preserving the KL severity ordering in cluster space.
    """
    return OrdinalPromptLearner.ordinal_contrastive_loss(prototypes, margin=margin)


def text_ordinal_soft_targets(text_feat: torch.Tensor,
                              labels: torch.Tensor,
                              sigma: float = 0.8,
                              alpha: float = 0.5) -> torch.Tensor:
    """
    OTDD：从文本原型的"序数轴投影"生成软标签分布。

    动机：PubMedBERT 直接编码的 KL 文本原型相似度矩阵存在 ~17% 序数单调性违反
    （KL0 与 KL4 的相似度反而高于与 KL2），直接用相似度矩阵蒸馏会注入噪声。
    但把各等级原型投影到主序数方向（KL0→KL4）后投影值是单调的，且间距非均匀
    （临床上 KL0↔KL1、KL3↔KL4 跨度大，中间三级更模糊）。本函数提取这一"序数
    一致"分量作为软标签：以投影坐标差的高斯核作为相邻等级的目标质量。

    由于低等级在序数轴上接近坍缩，纯投影软标签会把质量过度分摊到 KL0/1/2，
    削弱真实类别。故与 one-hot 按 alpha 混合，确保真实等级始终占主导，文本
    几何仅调制相邻等级的残差质量（序数感知的标签平滑）。

    Parameters
    ----------
    text_feat : [C, D] 文本原型（会被 L2 归一化）
    labels    : [B]    每个样本的真实 KL 等级
    sigma     : 高斯核带宽（投影坐标已按等级跨度归一化到 [0, C-1]）
    alpha     : 文本软标签占比；one-hot 占 (1-alpha)。真实类别质量 ≥ 1-alpha。

    Returns
    -------
    soft_targets : [B, C]，每行和为 1
    """
    feat = F.normalize(text_feat.float(), dim=-1)
    C = feat.shape[0]
    # 主序数方向：用 KL0→KL_{C-1} 的原型差向量
    axis = F.normalize(feat[-1] - feat[0], dim=0)            # [D]
    proj = feat @ axis                                        # [C]，单调坐标
    # 归一化到 [0, C-1]，使 sigma 的尺度与整数等级可比
    proj = (proj - proj.min()) / (proj.max() - proj.min()).clamp(min=1e-9) * (C - 1)
    coords = proj.detach()                                    # 软标签不回传梯度到文本
    labels = labels.long()
    tgt_coords = coords[labels]                               # [B]
    dist = coords.unsqueeze(0) - tgt_coords.unsqueeze(1)      # [B, C]
    soft = torch.exp(-0.5 * (dist / max(sigma, 1e-6)) ** 2)
    soft = soft / soft.sum(dim=1, keepdim=True).clamp(min=1e-9)
    # 与 one-hot 混合，保证真实类别主导
    onehot = F.one_hot(labels, num_classes=C).float()
    soft = (1.0 - alpha) * onehot + alpha * soft
    return soft


class ConvNeXtV2LocalEncoder(nn.Module):
    """
    ConvNeXt-V2 backbone for local feature extraction (CNN branch).

    Single-scale mode (multi_scale=False): uses global avg-pool of the final
    stage only, projecting to embed_dim.

    Multi-scale mode (multi_scale=True): extracts features from the two deepest
    ConvNeXt stages, pools each independently, and fuses via learned per-sample
    attention weights before projecting. This captures both fine-grained local
    cues (osteophyte morphology, stage N-1) and higher-level spatial structure
    (joint-space narrowing, stage N), which operate at different granularities
    and are both diagnostically relevant for KL grading.
    """

    def __init__(self, embed_dim: int = 512,
                 convnext_variant: str = 'convnextv2_large',
                 checkpoint_path: str = '',
                 multi_scale: bool = False):
        super().__init__()
        import timm
        self.multi_scale = multi_scale

        if multi_scale:
            # Use the same num_classes=0 backbone so the domain checkpoint loads
            # with full key compatibility.  In forward() we manually traverse
            # stem → stages and pool intermediate outputs from stage-2 and stage-3
            # instead of calling backbone() which would do a single global pool.
            backbone = timm.create_model(
                convnext_variant, pretrained=False, num_classes=0, global_pool='avg'
            )
            # Probe channel widths from the backbone stage output dims
            # convnextv2_large stages: [192, 384, 768, 1536]
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                h = backbone.stem(dummy)
                for i, stage in enumerate(backbone.stages):
                    h = stage(h)
                    if i == 2:
                        c_s2 = h.shape[1]
                c_s3 = backbone.num_features
            if checkpoint_path:
                raw = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(raw, dict) and 'state_dict' in raw:
                    raw = raw['state_dict']
                sd = {k: v for k, v in raw.items() if not k.startswith('head.')}
                missing, unexpected = backbone.load_state_dict(sd, strict=False)
                print(f"[ConvNeXtV2-MS] loaded {checkpoint_path} "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
            self.backbone = backbone
            # Each scale projected to embed_dim, then fused
            self.proj_s2 = nn.Linear(c_s2, embed_dim, bias=False)
            self.proj_s3 = nn.Linear(c_s3, embed_dim, bias=False)
            # Learned per-sample scale gate: [B, 2] → softmax → weighted sum
            self.scale_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, 2),
                nn.Softmax(dim=-1),
            )
            # Bias init: start very strongly stage-3 biased (gate≈[0.004, 0.996]) so
            # early training is nearly identical to single-scale until proj_s2 warms up.
            # scale_gate[2] is Linear(embed_dim, 2); index 0→stage-2, 1→stage-3.
            import torch as _torch
            self.scale_gate[2].bias.data = _torch.tensor([-4.0, 4.0])
            self.proj = nn.Linear(embed_dim, embed_dim, bias=False)
            self.norm = nn.LayerNorm(embed_dim)
        else:
            backbone = timm.create_model(
                convnext_variant, pretrained=False, num_classes=0, global_pool='avg'
            )
            in_features = backbone.num_features  # 1536 Large / 1024 Base / 768 Tiny
            if checkpoint_path:
                raw = torch.load(checkpoint_path, map_location='cpu')
                if isinstance(raw, dict) and 'state_dict' in raw:
                    raw = raw['state_dict']
                sd = {k: v for k, v in raw.items() if not k.startswith('head.')}
                missing, unexpected = backbone.load_state_dict(sd, strict=False)
                print(f"[ConvNeXtV2] loaded {checkpoint_path} "
                      f"(missing={len(missing)}, unexpected={len(unexpected)})")
            self.backbone = backbone
            self.proj = nn.Linear(in_features, embed_dim, bias=False)
            self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_scale:
            # Manual stage traversal: stem → stage0 → stage1 → stage2 → stage3
            # Pool stage-2 output (fine-grained local) and stage-3 (semantic global)
            h = self.backbone.stem(x.float())
            for i, stage in enumerate(self.backbone.stages):
                h = stage(h)
                if i == 2:
                    f2 = h.mean(dim=[-2, -1])       # [B, c_s2]
            # stage-3 output h: apply optional norm_pre then global avg pool
            norm_pre = getattr(self.backbone, 'norm_pre', None)
            h_normed = norm_pre(h) if norm_pre is not None else h
            f3 = h_normed.mean(dim=[-2, -1])        # [B, c_s3]
            f2 = self.proj_s2(f2)                   # [B, embed_dim]
            f3 = self.proj_s3(f3)                   # [B, embed_dim]
            gates = self.scale_gate(torch.cat([f2, f3], dim=-1))  # [B, 2]
            fused = gates[:, 0:1] * f2 + gates[:, 1:2] * f3      # [B, embed_dim]
            feat = self.proj(fused)
            feat = self.norm(feat)
        else:
            feat = self.backbone(x.float())          # [B, in_features]
            feat = self.proj(feat)                   # [B, embed_dim]
            feat = self.norm(feat)
        return F.normalize(feat, dim=-1)


class DualBranchCrossAttnFusion(nn.Module):
    """
    Bidirectional cross-modal fusion for Transformer (global) and CNN (local) features.

    Each branch reads complementary information from the other via learned cross
    projections, then an adaptive per-sample gate decides the final contribution:
    - ViT (global): absorbs local fine-grained spatial cues from ConvNeXt
    - ConvNeXt (local): absorbs global semantic context from ViT
    A concatenation projection path additionally captures high-order interactions.
    """

    def __init__(self, embed_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.g_reads_l = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.l_reads_g = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Dropout(dropout)
        )
        self.norm_g = nn.LayerNorm(embed_dim)
        self.norm_l = nn.LayerNorm(embed_dim)
        # Per-sample adaptive gate
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1),
        )
        # Concatenation projection for high-order mixing
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, global_feat: torch.Tensor, local_feat: torch.Tensor) -> torch.Tensor:
        # Cross-enhance: each branch reads from the other
        g = self.norm_g(global_feat + self.g_reads_l(local_feat))
        l = self.norm_l(local_feat + self.l_reads_g(global_feat))
        # Adaptive per-sample gating
        gates = self.gate(torch.cat([g, l], dim=-1))        # [B, 2]
        gated = gates[:, 0:1] * g + gates[:, 1:2] * l      # [B, D]
        # Concatenation projection path
        concat_proj = self.out_proj(torch.cat([g, l], dim=-1))  # [B, D]
        return F.normalize(gated + concat_proj, dim=-1)


class OrdinalVisualContrastiveLoss(nn.Module):
    """
    Ordinal-aware supervised contrastive loss on visual features.

    Pulls same-KL-grade visual embeddings together (standard SupCon positive pairs)
    and pushes different-KL-grade embeddings apart with weight proportional to
    ordinal distance: KL0 vs KL4 (dist=4) is pushed harder than KL0 vs KL1 (dist=1).

    This directly shapes the visual feature space so that adjacent grades (KL0/KL1,
    KL1/KL2, etc.) are still separable, addressing the visual similarity problem
    that text-side constraints cannot fix.
    """

    def __init__(self, temperature: float = 0.1, ordinal_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.ordinal_weight = ordinal_weight

    def forward(self, visual_feat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        visual_feat : [B, D]  visual embeddings (need not be pre-normalized)
        labels      : [B]     ordinal class labels 0 .. C-1
        """
        feat = F.normalize(visual_feat.float(), dim=-1)
        B = feat.shape[0]

        sim = feat @ feat.T / self.temperature          # [B, B]

        lbl = labels.float().view(-1)
        dist = (lbl.unsqueeze(0) - lbl.unsqueeze(1)).abs()  # [B, B]

        eye = torch.eye(B, device=feat.device, dtype=torch.float)
        pos_mask = ((dist == 0).float()) * (1.0 - eye)  # same grade, not self

        n_pos = pos_mask.sum(dim=1)
        valid = n_pos > 0
        if not valid.any():
            return feat.sum() * 0.0

        # Ordinal-weighted negative contribution:
        # weight(i,j) = 1 + ordinal_weight * |yi - yj|  for negatives
        neg_only = (1.0 - pos_mask) * (1.0 - eye)
        neg_weight = (1.0 + self.ordinal_weight * dist) * neg_only

        exp_sim = torch.exp(sim)
        weighted_denom = (exp_sim * pos_mask + exp_sim * neg_weight).sum(dim=1)  # [B]

        log_prob = sim - torch.log(weighted_denom.unsqueeze(1) + 1e-8)  # [B, B]
        mean_log_pos = (log_prob * pos_mask).sum(dim=1) / (n_pos + 1e-8)  # [B]

        return -mean_log_pos[valid].mean()


class BranchTextAlignHead(nn.Module):
    """
    BSVLA：把某个视觉分支的预融合特征投影到 PubMedBERT 文本空间的小 MLP 头。

    旧版"方向B"直接用裸分支特征与文本原型点积——但 ConvNeXt 不是 CLIP 模型，
    其特征空间与文本空间无对应关系，点积≈噪声。本投影头提供从分支特征空间到
    文本空间的【可学习桥梁】，是让分支特异性文本对齐真正生效的关键缺失组件。
    """
    def __init__(self, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


def branch_neg_entropy(logits: torch.Tensor) -> torch.Tensor:
    """返回 -H(softmax(logits)) 的均值；最小化它 = 最大化熵 = 推向均匀分布。

    用于跨分支解耦：让 off-branch（ViT 对病理 / ConvNeXt 对解剖）的预测趋于
    无信息（均匀），从而强制每个分支只承载其专属语义。
    """
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20., 20.)
    p = F.softmax(logits, dim=-1)
    logp = F.log_softmax(logits, dim=-1)
    return (p * logp).sum(dim=-1).mean()


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
    loss = CE(logits_cls, labels)    # 主分类损失
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
                 pubmedbert_path: str = PUBMEDBERT_PATH,
                 label_smoothing: float = 0.05,
                 lambda_proto: float = 0.2,
                 lambda_cluster: float = 0.1,
                 lambda_ordinal: float = 0.2,
                 lambda_ldl: float = 0.0,
                 ldl_sigma: float = 0.8,
                 logit_adjust_tau: float = 0.0,
                 enable_feature_adapter: bool = False,
                 adapter_bottleneck_dim: int = 64,
                 enable_ordinal_boundary: bool = False,
                 lambda_boundary: float = 0.0,
                 enable_visual_cluster: bool = False,
                 lambda_visual_cluster: float = 0.0,
                 enable_semantic_patch: bool = False,
                 lambda_patch: float = 0.0,
                 patch_adapter_bottleneck_dim: int = 64,
                 lambda_proto_separation: float = 0.0,
                 inference_proto_weight: float = 0.0,
                 inference_cluster_weight: float = 0.0,
                 inference_visual_cluster_weight: float = 0.0,
                 inference_patch_weight: float = 0.0,
                 inference_boundary_weight: float = 0.0,
                 enable_prompt_learning: bool = False,
                 n_ctx: int = 4,
                 lambda_ordinal_prompt: float = 0.0,
                 lambda_ordinal_proto: float = 0.0,
                 lambda_visual_con: float = 0.0,
                 visual_con_temperature: float = 0.1,
                 visual_con_ordinal_weight: float = 1.0,
                 enable_dual_branch: bool = False,
                 convnext_variant: str = 'convnextv2_large',
                 convnext_checkpoint: str = '',
                 convnext_multi_scale: bool = False,
                 lambda_emd: float = 0.0,
                 lambda_decouple: float = 0.0,
                 lambda_branch_text: float = 0.0,
                 branch_text_detach: bool = False,
                 inference_branch_text_weight: float = 0.0,
                 lambda_transition: float = 0.0,
                 inference_transition_weight: float = 0.0,
                 adaptive_text_proto: bool = False,
                 lambda_text_ord: float = 0.05,
                 text_anchor_cluster: bool = False,
                 lambda_text_distill: float = 0.0,
                 text_distill_sigma: float = 0.8,
                 enable_bsvla: bool = False,
                 lambda_bsvla: float = 0.0,
                 lambda_bsvla_disent: float = 0.0,
                 inference_bsvla_weight: float = 0.0,
                 ):
        super().__init__()

        # ── 视觉编码器 ────────────────────────────────────────────────
        self.visual = clip_model.visual
        self.enable_feature_adapter = bool(enable_feature_adapter)
        if self.enable_feature_adapter:
            self.feature_adapter = ResidualFeatureAdapter(
                embed_dim=embed_dim,
                bottleneck_dim=adapter_bottleneck_dim,
                dropout=fusion_dropout,
            )
        self.enable_semantic_patch = bool(enable_semantic_patch)
        if self.enable_semantic_patch:
            self.semantic_patch_attention = SemanticPatchAttention(
                embed_dim=embed_dim,
                num_classes=self.NUM_CLASSES,
                bottleneck_dim=patch_adapter_bottleneck_dim,
                dropout=fusion_dropout,
            )
            self.patch_residual_logit = nn.Parameter(torch.tensor(-5.0))

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
        self.enable_visual_cluster = bool(enable_visual_cluster)
        if self.enable_visual_cluster:
            visual_cluster_init = cluster_init.clone()
            self.visual_cluster_prototypes = nn.Parameter(visual_cluster_init)

        # ── 聚类感知门控融合模块 ──────────────────────────────────────
        if self.enable_visual_cluster:
            self.fusion = DualClusterAwareResidualGatedFusion(
                embed_dim=embed_dim,
                num_classes=self.NUM_CLASSES,
                hidden_dim=fusion_hidden_dim,
                dropout=fusion_dropout,
            )
        else:
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
        self.enable_ordinal_boundary = bool(enable_ordinal_boundary)
        if self.enable_ordinal_boundary:
            self.boundary_head = OrdinalBoundaryHead(
                embed_dim=embed_dim,
                hidden_dim=cls_hidden_dim,
                num_classes=self.NUM_CLASSES,
                dropout=cls_dropout,
            )

        # ── 损失函数 ──────────────────────────────────────────────────
        self.ce_loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.ordinal_loss_fn = OrdinalCELoss(num_classes=self.NUM_CLASSES)
        self.ldl_loss_fn = OrdinalLabelDistributionLoss(
            num_classes=self.NUM_CLASSES, sigma=ldl_sigma
        )
        self.lambda_proto = float(lambda_proto)
        self.lambda_cluster = float(lambda_cluster)
        self.lambda_ordinal = float(lambda_ordinal)
        self.lambda_ldl = float(lambda_ldl)
        self.logit_adjust_tau = float(logit_adjust_tau)
        # 类别对数先验（Logit Adjustment，Menon 2020）；默认均匀，训练前由
        # set_class_prior() 用训练集类别频率填充。训练时把 tau*log_prior 加进
        # 主分类 CE 的 logits，推理时用原始 logits，从而提升少数类（如 KL4）召回。
        self.register_buffer("class_log_prior", torch.zeros(self.NUM_CLASSES))
        self.lambda_boundary = float(lambda_boundary)
        self.lambda_visual_cluster = float(lambda_visual_cluster)
        self.lambda_patch = float(lambda_patch)
        self.lambda_proto_separation = float(lambda_proto_separation)
        self.lambda_ordinal_prompt = float(lambda_ordinal_prompt)
        self.lambda_ordinal_proto = float(lambda_ordinal_proto)
        self.lambda_visual_con = float(lambda_visual_con)
        if self.lambda_visual_con > 0:
            self.ordinal_visual_con = OrdinalVisualContrastiveLoss(
                temperature=visual_con_temperature,
                ordinal_weight=visual_con_ordinal_weight,
            )
        self.lambda_emd = float(lambda_emd)
        self.lambda_decouple = float(lambda_decouple)
        self.lambda_branch_text = float(lambda_branch_text)
        self.branch_text_detach = bool(branch_text_detach)
        self.inference_branch_text_weight = float(inference_branch_text_weight)
        self.lambda_transition = float(lambda_transition)
        self.inference_transition_weight = float(inference_transition_weight)
        self.adaptive_text_proto = bool(adaptive_text_proto)
        self.lambda_text_ord = float(lambda_text_ord)
        # OTACP：用文本原型初始化聚类原型（文本锚定）；OTDD：序数文本分布蒸馏
        self.text_anchor_cluster = bool(text_anchor_cluster)
        self.lambda_text_distill = float(lambda_text_distill)
        self.text_distill_sigma = float(text_distill_sigma)
        # ── 分支特异性文本原型 ──────────────────────────────────────────
        # branch_text_detach=True 时改为可学习 Parameter，梯度只流向文本原型
        # （视觉特征在 forward 里 detach），实现"文本原型适配视觉分布"
        if self.branch_text_detach:
            self.global_text_feat = nn.Parameter(torch.zeros(self.NUM_CLASSES, embed_dim))
            self.local_text_feat  = nn.Parameter(torch.zeros(self.NUM_CLASSES, embed_dim))
        else:
            self.register_buffer("global_text_feat", torch.zeros(self.NUM_CLASSES, embed_dim))
            self.register_buffer("local_text_feat",  torch.zeros(self.NUM_CLASSES, embed_dim))
        self.register_buffer("transition_text_feat", torch.zeros(self.NUM_CLASSES - 1, embed_dim))
        # ── 双分支：CNN（局部）+ Transformer（全局）联合训练 ─────────────
        self.enable_dual_branch = bool(enable_dual_branch)
        if self.enable_dual_branch:
            self.visual_local = ConvNeXtV2LocalEncoder(
                embed_dim=embed_dim,
                convnext_variant=convnext_variant,
                checkpoint_path=convnext_checkpoint,
                multi_scale=bool(convnext_multi_scale),
            )
            self.cross_branch_fusion = DualBranchCrossAttnFusion(
                embed_dim=embed_dim,
                dropout=fusion_dropout,
            )
        # ── BSVLA：双分支语义专化对齐 ───────────────────────────────────
        # 投影头把各分支预融合特征映射到文本空间，对比损失强制 ViT↔解剖、
        # ConvNeXt↔病理，并通过最大熵解耦让 off-branch 预测无信息。
        self.enable_bsvla = bool(enable_bsvla) and self.enable_dual_branch
        self.lambda_bsvla = float(lambda_bsvla)
        self.lambda_bsvla_disent = float(lambda_bsvla_disent)
        self.inference_bsvla_weight = float(inference_bsvla_weight)
        if self.enable_bsvla:
            self.vit_text_head = BranchTextAlignHead(embed_dim, fusion_dropout)
            self.cvx_text_head = BranchTextAlignHead(embed_dim, fusion_dropout)
        self.inference_proto_weight = float(inference_proto_weight)
        self.inference_cluster_weight = float(inference_cluster_weight)
        self.inference_visual_cluster_weight = float(inference_visual_cluster_weight)
        self.inference_patch_weight = float(inference_patch_weight)
        self.inference_boundary_weight = float(inference_boundary_weight)

        # ── 预计算文本原型 ─────────────────────────────────────────────
        self.register_buffer("text_feat", torch.zeros(self.NUM_CLASSES, embed_dim))
        self._text_feat_initialized = False
        # 可学习文本原型校准向量（ATPC）：从零初始化，叠加在 PubMedBERT 原型上，
        # 通过分类梯度自动在视觉特征空间中展开文本原型，配合序数约束防止乱序。
        if self.adaptive_text_proto:
            self.text_feat_delta = nn.Parameter(torch.zeros(self.NUM_CLASSES, embed_dim))

        self._pubmedbert_path = pubmedbert_path
        self._embed_dim       = embed_dim

        # ── 有序软提示学习（可选） ────────────────────────────────────
        self.enable_prompt_learning = bool(enable_prompt_learning)
        if self.enable_prompt_learning:
            self.prompt_learner = OrdinalPromptLearner(
                text_encoder=self.text_encoder,
                text_projection=self.text_projection,
                tokenizer_path=pubmedbert_path,
                embed_dim=embed_dim,
                num_classes=self.NUM_CLASSES,
                n_ctx=n_ctx,
                dropout=fusion_dropout,
            )

    # ------------------------------------------------------------------
    # 文本原型初始化
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _init_text_prototypes(self):
        tokenizer = AutoTokenizer.from_pretrained(self._pubmedbert_path)
        device    = self.text_feat.device
        prompt_groups = build_kl_prompt_groups(KL_PROMPTS)
        if len(prompt_groups) != self.NUM_CLASSES:
            raise ValueError(
                f"Expected {self.NUM_CLASSES} KL prompt groups, got {len(prompt_groups)}."
            )

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

        # 编码分支特异性文本（anatomy / pathology / transition）
        # texts 可以是 List[str]（每项一条）或 List[List[str]]（每项多条取平均）
        def _encode_grade_texts(texts):
            """编码一个 KL 等级对应的多条描述，返回平均特征向量。"""
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=128)
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
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
            return feat

        def _encode_text_list(grade_texts):
            """grade_texts: List[str] 或 List[List[str]]，每项对应一个 KL 等级或一条过渡文本。"""
            feats = []
            for item in grade_texts:
                texts = item if isinstance(item, list) else [item]
                feats.append(_encode_grade_texts(texts))
            return torch.stack(feats, dim=0)

        self.global_text_feat.copy_(_encode_text_list(ANATOMY_TEXTS))
        self.local_text_feat.copy_(_encode_text_list(PATHOLOGY_TEXTS))
        self.transition_text_feat.copy_(_encode_text_list(TRANSITION_TEXTS))

        # OTACP：用文本原型锚定聚类原型，使驱动融合的原型从第0轮即带有
        # 文本序数语义（替代随机 randn 初始化）。仅在聚类原型仍为初始状态
        # （未从 checkpoint 加载）时执行，避免覆盖已训练权重。
        if self.text_anchor_cluster:
            self.cluster_prototypes.data.copy_(self.text_feat.detach())
            if self.enable_visual_cluster and hasattr(self, "visual_cluster_prototypes"):
                self.visual_cluster_prototypes.data.copy_(self.text_feat.detach())

        self._text_feat_initialized = True

    def refresh_text_prototypes(self):
        """公开接口：手动刷新文本原型（文本编码器解冻微调后调用）。"""
        self._init_text_prototypes()

    def _get_calibrated_text_feat(self) -> torch.Tensor:
        """返回校准后的文本原型：ATPC 启用时叠加可学习偏移量并重新 L2 归一化。"""
        if self.adaptive_text_proto and hasattr(self, 'text_feat_delta'):
            raw = self.text_feat.detach().float() + self.text_feat_delta.float()
            return F.normalize(raw, dim=-1)
        return self.text_feat

    # ------------------------------------------------------------------
    # 编码方法
    # ------------------------------------------------------------------

    @property
    def dtype(self):
        # OpenAI CLIP 视觉塔有 conv1；BiomedCLIP(open_clip TimmModel) 没有。
        # DataParallel 副本上 next(parameters()) 可能因 generator 为空抛 StopIteration，
        # 改用 list 取首元素避免此问题。
        visual = self.visual
        if hasattr(visual, "conv1") and hasattr(visual.conv1, "weight"):
            return visual.conv1.weight.dtype
        params = list(visual.parameters())
        return params[0].dtype if params else torch.float32

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        feat = self.visual(image.type(self.dtype)).float()
        return feat / feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    def encode_image_with_patches(self, image: torch.Tensor):
        if not hasattr(self.visual, "forward_tokens"):
            return self.encode_image(image), None
        cls_feat, patch_feat = self.visual.forward_tokens(image.type(self.dtype))
        cls_feat = cls_feat.float()
        patch_feat = patch_feat.float()
        cls_feat = cls_feat / cls_feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        patch_feat = patch_feat / patch_feat.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        return cls_feat, patch_feat

    # ------------------------------------------------------------------
    # 前向传播
    # ------------------------------------------------------------------

    def forward(self,
                image: torch.Tensor,
                labels: torch.Tensor = None
                ) -> dict:
        """
        Parameters
        ----------
        image  : [B, 3, H, W]
        labels : [B]  训练时必须提供；推理时可为 None

        Returns
        -------
        dict，包含：
          - "logits_cls"   : [B, 5]
          - "logits_proto" : [B, 5]
          - "cluster_prob" : [B, 5]
          - "fuse_feat"    : [B, D]
          - "img_feat"     : [B, D]
          - "loss"         : scalar（仅训练模式且 labels 不为 None 时存在）
        """
        if not self._text_feat_initialized:
            self._init_text_prototypes()

        # 0. 文本原型：提示学习时动态计算；ATPC 时加可学习校准偏移；否则固定 buffer
        if self.enable_prompt_learning:
            text_feat = self.prompt_learner()        # [C, D], gradient flows through ctx
        else:
            text_feat = self._get_calibrated_text_feat()  # [C, D]，ATPC 时包含可学习 delta

        # 1. 图像特征
        if self.enable_semantic_patch:
            img_feat, patch_tokens = self.encode_image_with_patches(image)
        else:
            img_feat = self.encode_image(image)                           # [B, D]
            patch_tokens = None
        if self.enable_feature_adapter:
            img_feat = self.feature_adapter(img_feat)
        # 双分支：CNN 局部分支 + 跨分支融合
        pre_fusion_img_feat = None
        pre_fusion_local_feat = None
        if self.enable_dual_branch:
            local_feat = self.visual_local(image)                        # [B, D]
            pre_fusion_img_feat = img_feat                               # save for decouple loss
            pre_fusion_local_feat = local_feat                           # save for decouple loss
            img_feat = self.cross_branch_fusion(img_feat, local_feat)    # [B, D]

        # 2. 文本原型对齐 logits
        logits_proto = img_feat @ text_feat.T / self.TEMPERATURE          # [B, 5]

        # 分支特异性文本对齐（方向B）：ViT↔解剖文本，ConvNeXt↔病理文本
        # branch_text_detach=True 时视觉特征 detach，梯度只更新可学习文本原型
        logits_anatomy  = None
        logits_pathology = None
        _need_branch_text = (
            self.enable_dual_branch and pre_fusion_img_feat is not None and
            (self.lambda_branch_text > 0 or self.inference_branch_text_weight > 0)
        )
        if _need_branch_text:
            vit_q  = pre_fusion_img_feat.detach()  if self.branch_text_detach else pre_fusion_img_feat
            cvx_q  = pre_fusion_local_feat.detach() if self.branch_text_detach else pre_fusion_local_feat
            logits_anatomy  = vit_q  @ F.normalize(self.global_text_feat, dim=-1).T / self.TEMPERATURE
            logits_pathology = cvx_q @ F.normalize(self.local_text_feat,  dim=-1).T / self.TEMPERATURE

        # BSVLA：双分支语义专化对齐（修复版方向B）
        # 投影头把分支特征映射进文本空间，再与解剖/病理文本原型对齐。
        # 同时计算 off-branch（ViT→病理 / ConvNeXt→解剖）logits 用于最大熵解耦。
        bsvla_logits = None
        if self.enable_bsvla and pre_fusion_img_feat is not None and (
            self.lambda_bsvla > 0 or self.inference_bsvla_weight > 0
        ):
            z_vit = F.normalize(self.vit_text_head(pre_fusion_img_feat), dim=-1)
            z_cvx = F.normalize(self.cvx_text_head(pre_fusion_local_feat), dim=-1)
            anat = F.normalize(self.global_text_feat.float(), dim=-1)
            path = F.normalize(self.local_text_feat.float(),  dim=-1)
            bsvla_logits = {
                "vit_anat": z_vit @ anat.T / self.TEMPERATURE,   # ViT→解剖（专化，对齐）
                "cvx_path": z_cvx @ path.T / self.TEMPERATURE,   # ConvNeXt→病理（专化，对齐）
                "vit_path": z_vit @ path.T / self.TEMPERATURE,   # ViT→病理（应无信息）
                "cvx_anat": z_cvx @ anat.T / self.TEMPERATURE,   # ConvNeXt→解剖（应无信息）
            }

        # 等级过渡提示（方向A）：融合特征与过渡文本的相似度作为 P(y>k) 的 logit

        # 3. 聚类软分配概率
        norm_cluster_proto = F.normalize(self.cluster_prototypes, dim=-1)
        cluster_logits     = img_feat @ norm_cluster_proto.T / self.TEMPERATURE
        cluster_prob       = F.softmax(cluster_logits, dim=-1)            # [B, 5]

        # 4. 动态文本语义生成
        cluster_txt_feat = cluster_prob @ text_feat                       # [B, D]
        cluster_txt_feat = cluster_txt_feat / cluster_txt_feat.norm(
            dim=-1, keepdim=True
        ).clamp(min=1e-9)

        visual_cluster_logits = None
        visual_cluster_prob = None
        visual_cluster_feat = None

        # 5. 可选：视觉端 KL 原型记忆增强
        if self.enable_visual_cluster:
            norm_visual_cluster_proto = F.normalize(self.visual_cluster_prototypes, dim=-1)
            visual_cluster_logits = img_feat @ norm_visual_cluster_proto.T / self.TEMPERATURE
            visual_cluster_prob = F.softmax(visual_cluster_logits, dim=-1)
            visual_cluster_feat = visual_cluster_prob @ norm_visual_cluster_proto
            visual_cluster_feat = visual_cluster_feat / visual_cluster_feat.norm(
                dim=-1, keepdim=True
            ).clamp(min=1e-9)

            fuse_feat = self.fusion(
                img_feat,
                cluster_txt_feat,
                visual_cluster_feat,
                cluster_prob,
                visual_cluster_prob,
            )
        else:
            fuse_feat = self.fusion(img_feat, cluster_txt_feat, cluster_prob) # [B, D]

        patch_logits = None
        patch_feat = None
        patch_attn = None
        if self.enable_semantic_patch and patch_tokens is not None:
            patch_logits, patch_feat, patch_attn = self.semantic_patch_attention(
                patch_tokens,
                text_feat,
            )
            patch_scale = torch.sigmoid(self.patch_residual_logit)
            fuse_feat = fuse_feat + patch_scale * patch_feat
            fuse_feat = torch.nan_to_num(fuse_feat, nan=0.0, posinf=1.0, neginf=-1.0)
            fuse_feat = fuse_feat / fuse_feat.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        # 等级过渡 logits：fuse_feat 与过渡文本的相似度 → [B, 4] 二元阈值 logits
        transition_logits = None
        if self.lambda_transition > 0 or self.inference_transition_weight > 0:
            transition_logits = fuse_feat @ \
                F.normalize(self.transition_text_feat, dim=-1).T / self.TEMPERATURE  # [B,4]

        # 6. MLP 分类 + 可选 KL 有序边界头
        logits_classifier = self.classifier(fuse_feat)                    # [B, 5]
        boundary_logits = None
        boundary_class_logits = None
        if self.enable_ordinal_boundary:
            boundary_logits = self.boundary_head(fuse_feat)
            boundary_class_logits = boundary_logits_to_class_logits(boundary_logits)


        logits_cls = logits_classifier
        if not self.training:
            logits_cls = (
                logits_classifier
                + self.inference_proto_weight * logits_proto
                + self.inference_cluster_weight * cluster_logits
            )
            if visual_cluster_logits is not None:
                logits_cls = logits_cls + self.inference_visual_cluster_weight * visual_cluster_logits
            if patch_logits is not None:
                logits_cls = logits_cls + self.inference_patch_weight * patch_logits
            if boundary_class_logits is not None:
                logits_cls = logits_cls + self.inference_boundary_weight * boundary_class_logits
            # 过渡 logits → 类 logits：P(y=k) ∝ P(y>k-1) - P(y>k)
            if transition_logits is not None and self.inference_transition_weight > 0:
                trans_probs = torch.sigmoid(transition_logits)           # [B, 4]
                ones  = torch.ones(trans_probs.size(0), 1, device=trans_probs.device)
                zeros = torch.zeros(trans_probs.size(0), 1, device=trans_probs.device)
                cdf   = torch.cat([ones, trans_probs, zeros], dim=1)     # [B, 6]
                trans_class_logits = (cdf[:, :-1] - cdf[:, 1:]).clamp(min=1e-9).log()  # [B, 5]
                logits_cls = logits_cls + self.inference_transition_weight * trans_class_logits
            # 分支文本原型集成：anatomy(ViT) + pathology(ConvNeXt) 平均后与主分类器集成
            if logits_anatomy is not None and self.inference_branch_text_weight > 0:
                logits_cls = logits_cls + self.inference_branch_text_weight * (
                    logits_anatomy + logits_pathology
                ) * 0.5
            # BSVLA 专化对齐 logits 集成：ViT→解剖 + ConvNeXt→病理
            if bsvla_logits is not None and self.inference_bsvla_weight > 0:
                logits_cls = logits_cls + self.inference_bsvla_weight * (
                    bsvla_logits["vit_anat"] + bsvla_logits["cvx_path"]
                ) * 0.5

        out = {
            "logits_cls"   : logits_cls,
            "logits_classifier": logits_classifier,
            "logits_proto" : logits_proto,
            "cluster_logits": cluster_logits,
            "cluster_prob" : cluster_prob,
            "fuse_feat"    : fuse_feat,
            "img_feat"     : img_feat,
        }
        if boundary_logits is not None:
            out["boundary_logits"] = boundary_logits
            out["boundary_class_logits"] = boundary_class_logits
        if visual_cluster_logits is not None:
            out["visual_cluster_logits"] = visual_cluster_logits
            out["visual_cluster_prob"] = visual_cluster_prob
            out["visual_cluster_feat"] = visual_cluster_feat
        if patch_logits is not None:
            out["patch_logits"] = patch_logits
            out["patch_feat"] = patch_feat
            out["patch_attn"] = patch_attn

        # 7. 训练损失
        if self.training and labels is not None:
            loss = self._compute_loss(
                logits_classifier, labels,
                logits_proto=logits_proto,
                cluster_logits=cluster_logits,
                visual_cluster_logits=visual_cluster_logits,
                patch_logits=patch_logits,
                boundary_logits=boundary_logits,
                text_feat=text_feat,
                visual_feat=img_feat,
                pre_fusion_img_feat=pre_fusion_img_feat,
                pre_fusion_local_feat=pre_fusion_local_feat,
                logits_anatomy=logits_anatomy,
                logits_pathology=logits_pathology,
                transition_logits=transition_logits,
                bsvla_logits=bsvla_logits,
            )
            out["loss"] = loss

        return out

    # ------------------------------------------------------------------
    # 损失计算
    # ------------------------------------------------------------------

    def _compute_loss(self,
                      logits_cls: torch.Tensor,
                      labels: torch.Tensor,
                      logits_proto: torch.Tensor = None,
                      cluster_logits: torch.Tensor = None,
                      visual_cluster_logits: torch.Tensor = None,
                      patch_logits: torch.Tensor = None,
                      boundary_logits: torch.Tensor = None,
                      text_feat: torch.Tensor = None,
                      visual_feat: torch.Tensor = None,
                      pre_fusion_img_feat: torch.Tensor = None,
                      pre_fusion_local_feat: torch.Tensor = None,
                      logits_anatomy: torch.Tensor = None,
                      logits_pathology: torch.Tensor = None,
                      transition_logits: torch.Tensor = None,
                      bsvla_logits: dict = None) -> torch.Tensor:
        """
        Joint loss for KL grading:
        CE(classifier) + ordinal threshold loss + text prototype CE + cluster CE
        + ordinal prompt loss (when enable_prompt_learning) + ordinal prototype loss
        + EMD ordinal loss + branch decoupling regularization.
        """
        logits_cls = torch.nan_to_num(
            logits_cls.float(), nan=0.0, posinf=20.0, neginf=-20.0
        ).clamp(-20.0, 20.0)

        labels = labels.long()

        # 主分类 CE：可选 Logit Adjustment（训练时在 logits 上加 tau*log_prior）
        if self.logit_adjust_tau > 0:
            ce_logits = logits_cls + self.logit_adjust_tau * self.class_log_prior.to(logits_cls.dtype)
        else:
            ce_logits = logits_cls
        total = self.ce_loss_fn(ce_logits, labels)
        if self.lambda_ldl > 0:
            total = total + self.lambda_ldl * self.ldl_loss_fn(logits_cls, labels)
        if self.lambda_ordinal > 0:
            total = total + self.lambda_ordinal * self.ordinal_loss_fn(logits_cls, labels)
        if boundary_logits is not None and self.lambda_boundary > 0:
            total = total + self.lambda_boundary * self.ordinal_loss_fn(boundary_logits, labels)
        if logits_proto is not None and self.lambda_proto > 0:
            total = total + self.lambda_proto * self.ce_loss_fn(
                logits_proto.float().clamp(-20.0, 20.0), labels
            )
        if cluster_logits is not None and self.lambda_cluster > 0:
            total = total + self.lambda_cluster * self.ce_loss_fn(
                cluster_logits.float().clamp(-20.0, 20.0), labels
            )
        if visual_cluster_logits is not None and self.lambda_visual_cluster > 0:
            total = total + self.lambda_visual_cluster * self.ce_loss_fn(
                visual_cluster_logits.float().clamp(-20.0, 20.0), labels
            )
        if patch_logits is not None and self.lambda_patch > 0:
            total = total + self.lambda_patch * self.ce_loss_fn(
                patch_logits.float().clamp(-20.0, 20.0), labels
            )
        if self.enable_visual_cluster and self.lambda_proto_separation > 0:
            total = total + self.lambda_proto_separation * (
                prototype_separation_loss(self.cluster_prototypes)
                + prototype_separation_loss(self.visual_cluster_prototypes)
            )
        # 有序提示损失：让相邻 KL 等级的文本原型比远距等级更相似
        if self.enable_prompt_learning and self.lambda_ordinal_prompt > 0 \
                and text_feat is not None:
            total = total + self.lambda_ordinal_prompt * \
                OrdinalPromptLearner.ordinal_contrastive_loss(text_feat)
        # 有序聚类原型损失：保持聚类原型的 KL 等级单调性
        if self.lambda_ordinal_proto > 0:
            total = total + self.lambda_ordinal_proto * \
                ordinal_prototype_loss(self.cluster_prototypes)
        # 有序视觉对比损失：在视觉特征空间按序数距离加权推开不同 KL 级别的特征
        if self.lambda_visual_con > 0 and visual_feat is not None:
            total = total + self.lambda_visual_con * \
                self.ordinal_visual_con(visual_feat, labels)
        # 序数 EMD 损失：惩罚预测分布与真实分布的 CDF 偏差，远误差罚更重
        if self.lambda_emd > 0:
            total = total + self.lambda_emd * emd_ordinal_loss(logits_cls, labels)
        # 互补解耦正则：惩罚 ViT 与 ConvNeXt 预融合特征的余弦相似度，防止两路退化为冗余表示
        if self.lambda_decouple > 0 \
                and pre_fusion_img_feat is not None \
                and pre_fusion_local_feat is not None:
            total = total + self.lambda_decouple * \
                branch_decoupling_loss(pre_fusion_img_feat, pre_fusion_local_feat)
        # 分支特异性文本对齐（方向B）：ViT↔解剖文本，ConvNeXt↔病理文本
        if self.lambda_branch_text > 0:
            if logits_anatomy is not None:
                total = total + self.lambda_branch_text * self.ce_loss_fn(
                    logits_anatomy.float().clamp(-20., 20.), labels)
            if logits_pathology is not None:
                total = total + self.lambda_branch_text * self.ce_loss_fn(
                    logits_pathology.float().clamp(-20., 20.), labels)
        # BSVLA：双分支语义专化对齐（修复版方向B）
        #   1) 专化对齐 CE：ViT→解剖、ConvNeXt→病理 都应能正确预测 KL 等级
        #   2) 跨分支最大熵解耦：ViT→病理、ConvNeXt→解剖 应无信息（趋于均匀），
        #      强制每个分支只承载其专属语义，配合投影头实现真正的语义专化。
        if self.enable_bsvla and self.lambda_bsvla > 0 and bsvla_logits is not None:
            total = total + self.lambda_bsvla * (
                self.ce_loss_fn(bsvla_logits["vit_anat"].float().clamp(-20., 20.), labels)
                + self.ce_loss_fn(bsvla_logits["cvx_path"].float().clamp(-20., 20.), labels)
            )
            if self.lambda_bsvla_disent > 0:
                total = total + self.lambda_bsvla_disent * (
                    branch_neg_entropy(bsvla_logits["vit_path"])
                    + branch_neg_entropy(bsvla_logits["cvx_anat"])
                )
        # OTDD：序数文本分布蒸馏。从文本原型的序数轴投影构造软标签，对主分类
        # logits 做 KL 蒸馏，把文本编码的"等级间临床模糊度"作为序数监督注入。
        if self.lambda_text_distill > 0 and text_feat is not None:
            soft = text_ordinal_soft_targets(
                text_feat, labels, sigma=self.text_distill_sigma
            )
            log_probs = F.log_softmax(logits_cls, dim=1)
            total = total + self.lambda_text_distill * \
                (-(soft * log_probs).sum(dim=1).mean())
        # ATPC 序数文本原型约束：校准后的文本原型在固定序数轴上必须单调递增
        # 使用 PubMedBERT 固定原型的 KL0→KL4 方向作为序数参考轴，防止文本原型乱序
        if self.adaptive_text_proto and self.lambda_text_ord > 0 and text_feat is not None:
            with torch.no_grad():
                base_ax = self.text_feat[-1].float() - self.text_feat[0].float()
                ordinal_ax = F.normalize(base_ax, dim=0)   # [D], 固定参考轴
            projs = text_feat.float() @ ordinal_ax         # [C], 各等级投影值
            margin = 0.01
            text_ord_loss = sum(
                F.relu(margin - (projs[i + 1] - projs[i]))
                for i in range(self.NUM_CLASSES - 1)
            )
            total = total + self.lambda_text_ord * text_ord_loss
        # 等级过渡提示损失（方向A）：4个二元阈值 CE，P(y>k) vs (label>k)
        if self.lambda_transition > 0 and transition_logits is not None:
            K = transition_logits.size(1)  # 4
            trans_loss = sum(
                F.binary_cross_entropy_with_logits(
                    transition_logits[:, k],
                    (labels > k).float()
                )
                for k in range(K)
            ) / K
            total = total + self.lambda_transition * trans_loss

        return torch.nan_to_num(total, nan=0.0, posinf=100.0, neginf=0.0)

    # ------------------------------------------------------------------
    # 预测接口
    # ------------------------------------------------------------------

    def set_class_prior(self, counts):
        """用训练集各类样本数设置 Logit Adjustment 的对数先验。"""
        counts = torch.as_tensor(counts, dtype=torch.float32)
        freq = counts.clamp(min=1.0)
        freq = freq / freq.sum()
        self.class_log_prior = torch.log(freq).to(self.class_log_prior.device)
        return self.class_log_prior

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
    text_missing = [k for k in missing if 'text_encoder' in k or 'text_projection' in k]
    print(f"[build_model] Missing text keys: {len(text_missing)} (expected for PubMedBERT replacement)")
    print(f"[build_model] Unexpected keys: {len(unexpected)}")

    model.float()
    return model.eval()


BIOMEDCLIP_HF = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def build_biomedclip_visual(img_size: int = 224) -> nn.Module:
    """
    构建 BiomedCLIP 视觉塔（open_clip 的 TimmModel，ViT-B/16）。

    输出特征维度为 512，与 OpenAI CLIP ViT-B/16 的投影维度一致，因此可以
    直接替换 CLIPFusionContrastiveModel 的 self.visual，而文本原型、聚类
    原型、门控融合与分类器等下游模块全部无需改动。

    BiomedCLIP 的文本端本身就是 PubMedBERT，与本项目本地加载的 PubMedBERT
    路线天然契合：这里只替换视觉塔，文本侧保持现有实现。
    """
    try:
        import open_clip
    except ImportError as e:
        raise ImportError(
            "使用 --backbone biomedclip 需要安装 open_clip_torch："
            "pip install open_clip_torch"
        ) from e

    model, _ = open_clip.create_model_from_pretrained(BIOMEDCLIP_HF)
    visual = model.visual
    visual.float()

    if img_size and img_size != 224:
        # BiomedCLIP 原生 224。非原生分辨率：放宽 patch_embed 尺寸校验，
        # 并把位置编码双三次插值到新的 patch 网格（标准 ViT pos-embed 插值）。
        import math
        tm = visual.trunk
        patch = tm.patch_embed.patch_size[0]
        if img_size % patch != 0:
            raise ValueError(f"img_size={img_size} 必须能被 patch_size={patch} 整除")
        new_grid = img_size // patch
        tm.patch_embed.strict_img_size = False
        tm.patch_embed.img_size = (img_size, img_size)
        tm.patch_embed.grid_size = (new_grid, new_grid)

        npre = tm.num_prefix_tokens
        pe = tm.pos_embed
        cls_pe, grid_pe = pe[:, :npre], pe[:, npre:]
        old_grid = int(math.sqrt(grid_pe.shape[1]))
        C = grid_pe.shape[-1]
        grid_pe = grid_pe.reshape(1, old_grid, old_grid, C).permute(0, 3, 1, 2)
        grid_pe = F.interpolate(grid_pe, size=(new_grid, new_grid),
                                mode="bicubic", align_corners=False)
        grid_pe = grid_pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, C)
        tm.pos_embed = nn.Parameter(torch.cat([cls_pe, grid_pe], dim=1))
        print(f"[backbone] BiomedCLIP 位置编码已插值到 {new_grid}x{new_grid} 网格 "
              f"(img_size={img_size})")

    return visual


def build_fusion_model(state_dict: dict,
                       pubmedbert_path: str = PUBMEDBERT_PATH,
                       backbone: str = "openai_clip",
                       img_size: int = 224,
                       freeze_text_encoder: bool = True,
                       fusion_dropout: float = 0.1,
                       cls_dropout: float = 0.2,
                       label_smoothing: float = 0.05,
                       lambda_proto: float = 0.2,
                       lambda_cluster: float = 0.1,
                       lambda_ordinal: float = 0.2,
                       lambda_ldl: float = 0.0,
                       ldl_sigma: float = 0.8,
                       logit_adjust_tau: float = 0.0,
                       enable_feature_adapter: bool = False,
                       adapter_bottleneck_dim: int = 64,
                       enable_ordinal_boundary: bool = False,
                       lambda_boundary: float = 0.0,
                       enable_visual_cluster: bool = False,
                       lambda_visual_cluster: float = 0.0,
                       enable_semantic_patch: bool = False,
                       lambda_patch: float = 0.0,
                       patch_adapter_bottleneck_dim: int = 64,
                       lambda_proto_separation: float = 0.0,
                       inference_proto_weight: float = 0.0,
                       inference_cluster_weight: float = 0.0,
                       inference_visual_cluster_weight: float = 0.0,
                       inference_patch_weight: float = 0.0,
                       inference_boundary_weight: float = 0.0,
                       enable_prompt_learning: bool = False,
                       n_ctx: int = 4,
                       lambda_ordinal_prompt: float = 0.0,
                       lambda_ordinal_proto: float = 0.0,
                       lambda_visual_con: float = 0.0,
                       visual_con_temperature: float = 0.1,
                       visual_con_ordinal_weight: float = 1.0,
                       enable_dual_branch: bool = False,
                       convnext_variant: str = 'convnextv2_large',
                       convnext_checkpoint: str = '',
                       convnext_multi_scale: bool = False,
                       lambda_emd: float = 0.0,
                       lambda_decouple: float = 0.0,
                       lambda_branch_text: float = 0.0,
                       branch_text_detach: bool = False,
                       inference_branch_text_weight: float = 0.0,
                       lambda_transition: float = 0.0,
                       inference_transition_weight: float = 0.0,
                       adaptive_text_proto: bool = False,
                       lambda_text_ord: float = 0.05,
                       text_anchor_cluster: bool = False,
                       lambda_text_distill: float = 0.0,
                       text_distill_sigma: float = 0.8,
                       enable_bsvla: bool = False,
                       lambda_bsvla: float = 0.0,
                       lambda_bsvla_disent: float = 0.0,
                       inference_bsvla_weight: float = 0.0,
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

    if backbone == "biomedclip":
        # 只替换视觉塔：用领域预训练的 BiomedCLIP ViT-B/16 取代通用 OpenAI CLIP
        # 视觉塔；文本编码器 / text_projection / 融合 / 分类器全部保持不变。
        biomed_visual = build_biomedclip_visual(img_size=img_size)
        with torch.no_grad():
            probe = torch.zeros(1, 3, img_size, img_size)
            out_dim = biomed_visual(probe).shape[-1]
        if out_dim != embed_dim:
            raise ValueError(
                f"BiomedCLIP 视觉输出维度 {out_dim} 与 embed_dim {embed_dim} 不一致，"
                f"无法直接替换视觉塔。"
            )
        clip_base.visual = biomed_visual
        print(f"[backbone] 已替换为 BiomedCLIP 视觉塔 (dim={out_dim}, img_size={img_size})")
    elif backbone != "openai_clip":
        raise ValueError(f"未知 backbone: {backbone}")

    fusion_model = CLIPFusionContrastiveModel(
        clip_model=clip_base,
        embed_dim=embed_dim,
        fusion_dropout=fusion_dropout,
        cls_dropout=cls_dropout,
        freeze_text_encoder=freeze_text_encoder,
        pubmedbert_path=pubmedbert_path,
        label_smoothing=label_smoothing,
        lambda_proto=lambda_proto,
        lambda_cluster=lambda_cluster,
        lambda_ordinal=lambda_ordinal,
        lambda_ldl=lambda_ldl,
        ldl_sigma=ldl_sigma,
        logit_adjust_tau=logit_adjust_tau,
        enable_feature_adapter=enable_feature_adapter,
        adapter_bottleneck_dim=adapter_bottleneck_dim,
        enable_ordinal_boundary=enable_ordinal_boundary,
        lambda_boundary=lambda_boundary,
        enable_visual_cluster=enable_visual_cluster,
        lambda_visual_cluster=lambda_visual_cluster,
        enable_semantic_patch=enable_semantic_patch,
        lambda_patch=lambda_patch,
        patch_adapter_bottleneck_dim=patch_adapter_bottleneck_dim,
        lambda_proto_separation=lambda_proto_separation,
        inference_proto_weight=inference_proto_weight,
        inference_cluster_weight=inference_cluster_weight,
        inference_visual_cluster_weight=inference_visual_cluster_weight,
        inference_patch_weight=inference_patch_weight,
        inference_boundary_weight=inference_boundary_weight,
        enable_prompt_learning=enable_prompt_learning,
        n_ctx=n_ctx,
        lambda_ordinal_prompt=lambda_ordinal_prompt,
        lambda_ordinal_proto=lambda_ordinal_proto,
        lambda_visual_con=lambda_visual_con,
        visual_con_temperature=visual_con_temperature,
        visual_con_ordinal_weight=visual_con_ordinal_weight,
        enable_dual_branch=enable_dual_branch,
        convnext_variant=convnext_variant,
        convnext_checkpoint=convnext_checkpoint,
        convnext_multi_scale=convnext_multi_scale,
        lambda_emd=lambda_emd,
        lambda_decouple=lambda_decouple,
        lambda_branch_text=lambda_branch_text,
        branch_text_detach=branch_text_detach,
        inference_branch_text_weight=inference_branch_text_weight,
        lambda_transition=lambda_transition,
        inference_transition_weight=inference_transition_weight,
        adaptive_text_proto=adaptive_text_proto,
        lambda_text_ord=lambda_text_ord,
        text_anchor_cluster=text_anchor_cluster,
        lambda_text_distill=lambda_text_distill,
        text_distill_sigma=text_distill_sigma,
        enable_bsvla=enable_bsvla,
        lambda_bsvla=lambda_bsvla,
        lambda_bsvla_disent=lambda_bsvla_disent,
        inference_bsvla_weight=inference_bsvla_weight,
    )

    return fusion_model.train()


# ===== 优化器（分层学习率） =====

def build_optimizer(model: CLIPFusionContrastiveModel,
                    lr_text_encoder: float = 1e-5,
                    lr_visual: float = 5e-5,
                    lr_text_proj: float = 1e-4,
                    lr_fusion: float = 1e-4,
                    lr_classifier: float = 1e-4,
                    lr_adapter: float = None,
                    lr_boundary: float = None,
                    lr_cluster_proto: float = 1e-4,
                    lr_prompt: float = 3e-4,
                    lr_convnext: float = 1e-5,
                    weight_decay: float = 0.01) -> torch.optim.AdamW:
    """
    分层学习率 AdamW 优化器。
    """
    param_groups = []
    if hasattr(model, "visual"):
        param_groups.append({
            "params": list(model.visual.parameters()),
            "lr": lr_visual, "name": "visual",
        })
    if hasattr(model, "text_projection"):
        param_groups.append({
            "params": list(model.text_projection.parameters()),
            "lr": lr_text_proj, "name": "text_projection",
        })
    if hasattr(model, "fusion"):
        param_groups.append({
            "params": list(model.fusion.parameters()),
            "lr": lr_fusion, "name": "fusion",
        })
    if hasattr(model, "feature_adapter"):
        param_groups.append({
            "params": list(model.feature_adapter.parameters()),
            "lr": lr_fusion if lr_adapter is None else lr_adapter, "name": "feature_adapter",
        })
    if hasattr(model, "semantic_patch_attention"):
        patch_params = list(model.semantic_patch_attention.parameters())
        if hasattr(model, "patch_residual_logit"):
            patch_params.append(model.patch_residual_logit)
        param_groups.append({
            "params": patch_params,
            "lr": lr_fusion, "name": "semantic_patch_attention",
        })
    if hasattr(model, "classifier"):
        param_groups.append({
            "params": list(model.classifier.parameters()),
            "lr": lr_classifier, "name": "classifier",
        })
    if hasattr(model, "boundary_head"):
        param_groups.append({
            "params": list(model.boundary_head.parameters()),
            "lr": lr_classifier if lr_boundary is None else lr_boundary, "name": "boundary_head",
        })
    if hasattr(model, "cluster_prototypes"):
        param_groups.append({
            "params": [model.cluster_prototypes],
            "lr": lr_cluster_proto, "name": "cluster_prototypes",
        })
    if hasattr(model, "visual_cluster_prototypes"):
        param_groups.append({
            "params": [model.visual_cluster_prototypes],
            "lr": lr_cluster_proto, "name": "visual_cluster_prototypes",
        })
    # 有序提示学习：只训练可学习的 ctx 向量，text_encoder/projection 已在各自组里
    if hasattr(model, "prompt_learner"):
        param_groups.append({
            "params": [model.prompt_learner.ctx],
            "lr": lr_prompt, "name": "prompt_learner_ctx",
        })
    # 双分支：ConvNeXt backbone 用较低 LR（域微调初始化），projection 和 fusion 用常规 LR
    if hasattr(model, "visual_local"):
        param_groups.append({
            "params": list(model.visual_local.backbone.parameters()),
            "lr": lr_convnext, "name": "visual_local_backbone",
        })
        if getattr(model.visual_local, 'multi_scale', False):
            # 多尺度：proj_s3 对应单尺度 proj（可热启），proj_s2/scale_gate/proj 是新参数
            param_groups.append({
                "params": (list(model.visual_local.proj_s3.parameters())
                          + list(model.visual_local.norm.parameters())),
                "lr": lr_fusion, "name": "visual_local_proj_s3",
            })
            param_groups.append({
                "params": (list(model.visual_local.proj_s2.parameters())
                          + list(model.visual_local.scale_gate.parameters())
                          + list(model.visual_local.proj.parameters())),
                "lr": lr_fusion, "name": "visual_local_ms_new",
            })
        else:
            param_groups.append({
                "params": (list(model.visual_local.proj.parameters())
                          + list(model.visual_local.norm.parameters())),
                "lr": lr_fusion, "name": "visual_local_proj",
            })
    if hasattr(model, "cross_branch_fusion"):
        param_groups.append({
            "params": list(model.cross_branch_fusion.parameters()),
            "lr": lr_fusion, "name": "cross_branch_fusion",
        })
    # BSVLA：双分支语义专化投影头（新增参数，用 fusion 学习率）
    if getattr(model, "enable_bsvla", False) and hasattr(model, "vit_text_head"):
        param_groups.append({
            "params": (list(model.vit_text_head.parameters())
                       + list(model.cvx_text_head.parameters())),
            "lr": lr_fusion, "name": "bsvla_heads",
        })
    # ATPC：可学习文本原型校准向量，使用与 cluster_prototypes 相同的学习率
    if hasattr(model, "text_feat_delta") and model.adaptive_text_proto:
        param_groups.append({
            "params": [model.text_feat_delta],
            "lr": lr_cluster_proto, "name": "text_feat_delta",
        })

    trainable_text_params = (
        [p for p in model.text_encoder.parameters() if p.requires_grad]
        if hasattr(model, "text_encoder") else []
    )
    if trainable_text_params:
        param_groups.append({
            "params": trainable_text_params,
            "lr": lr_text_encoder, "name": "text_encoder"
        })

    param_groups = [
        pg for pg in param_groups
        if any(p.requires_grad for p in pg["params"])
    ]
    if not param_groups:
        raise ValueError("No trainable parameter groups were found.")

    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)
