import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np

import gzip
import html
import os
from functools import lru_cache

import ftfy
import regex as re

from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# =====================================================================
#  以下 SimpleTokenizer / BPE 工具函数保持不变，此处省略（与原文相同）
# =====================================================================


def load_options(cfg):
    backbone_name = cfg.MODEL.EVO.ENC_NAME
    if backbone_name == "omnivore_swinT" or backbone_name == "omnivore_swinS":
        z_dim = 768
    elif backbone_name == "omnivore_swinB":
        z_dim = 1024
    else:
        z_dim = cfg.MODEL.EVO.ENC_OUT_DIM

    bias_flag  = cfg.MODEL.EVO.BIAS
    frames     = cfg.INPUT.FRAMES
    input_size = cfg.INPUT.SIZE

    if cfg.MODEL.EVO.ACT == "relu":
        act = nn.ReLU
    elif cfg.MODEL.EVO.ACT == "gelu":
        act = nn.GELU

    return z_dim, bias_flag, frames, input_size, act


# ======================================================================
#  Swin-UNet 基础模块（与原文完全相同，保持不变）
# ======================================================================

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features    or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x);  x = self.act(x);  x = self.drop(x)
        x = self.fc2(x);  x = self.drop(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
                   W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim         = dim // num_heads
        self.scale       = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) *
                        (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten    = torch.flatten(coords, 1)
        relative_coords   = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords   = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads,
                                   C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW   = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) \
                   + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x);  x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7,
                 shift_size=0, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim              = dim
        self.input_resolution = input_resolution
        self.num_heads        = num_heads
        self.window_size      = window_size
        self.shift_size       = shift_size
        self.mlp_ratio        = mlp_ratio

        if min(self.input_resolution) <= self.window_size:
            self.shift_size  = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn  = WindowAttention(
            dim, window_size=to_2tuple(self.window_size),
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            H, W      = self.input_resolution
            img_mask  = torch.zeros((1, H, W, 1))
            h_slices  = (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
            w_slices  = (slice(0, -self.window_size),
                         slice(-self.window_size, -self.shift_size),
                         slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt;  cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask    = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask    = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                    .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W    = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size),
                                   dims=(1, 2))
        else:
            shifted_x = x

        x_windows  = window_partition(shifted_x, self.window_size)
        x_windows  = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x    = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, expand_scale=2,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.expand_scale     = expand_scale
        self.dim              = dim
        self.dim_scale        = dim_scale
        self.expand           = nn.Linear(dim, expand_scale * dim, bias=False)
        self.norm             = norm_layer(dim // dim_scale)

    def forward(self, x):
        H, W = self.input_resolution
        x    = self.expand(x)
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c) -> b (h p1) (w p2) c',
                      p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x


# ======================================================================
#  ★ 新增模块 1：LesionQueryAttention
#    三类病灶（骨赘 / 关节间隙 / 软骨退化）对应三个可学习 query，
#    通过交叉注意力从空间特征图中提取各自的响应区域。
# ======================================================================

class LesionQueryAttention(nn.Module):
    """
    Cross-attention module with learnable lesion-type queries.

    三个可学习的病灶查询向量分别代表：
      - Query 0: 骨赘 (Osteophyte)      — 骨边缘尖锐凸起
      - Query 1: 关节间隙 (Joint Space)  — 间隙狭窄/消失
      - Query 2: 软骨退化 (Cartilage)    — 软骨下骨纹理变化

    每个 query 与空间 token 做多头交叉注意力，
    输出 [B, K, L] 的空间激活图（K=3），
    后续用于加权增强掩码。
    """

    def __init__(self, dim: int, num_lesion_types: int = 3, num_heads: int = 8,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须可被 num_heads 整除"

        self.num_lesion_types = num_lesion_types
        self.num_heads        = num_heads
        self.head_dim         = dim // num_heads
        self.scale            = self.head_dim ** -0.5

        # 可学习的病灶类型 query，形状 [K, dim]
        self.lesion_queries = nn.Parameter(
            torch.randn(num_lesion_types, dim) * 0.02
        )

        self.q_proj   = nn.Linear(dim, dim, bias=False)
        self.k_proj   = nn.Linear(dim, dim, bias=False)
        self.v_proj   = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.norm     = norm_layer(dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, L, C]  —— 空间特征序列（L = H * W）
        Returns:
            lesion_context: [B, K, C]  —— 每类病灶的聚合特征
            lesion_attn:    [B, K, L]  —— 每类病灶的空间激活图
        """
        B, L, C = x.shape
        K       = self.num_lesion_types

        # 将可学习 query 扩展到 batch 维度
        queries = self.lesion_queries.unsqueeze(0).expand(B, -1, -1)  # [B, K, C]

        q = self.q_proj(queries)  # [B, K, C]
        k = self.k_proj(x)        # [B, L, C]
        v = self.v_proj(x)        # [B, L, C]

        # 拆分多头
        def split_heads(t, seq_len):
            return t.reshape(B, seq_len, self.num_heads, self.head_dim) \
                    .permute(0, 2, 1, 3)   # [B, H, seq_len, head_dim]

        q = split_heads(q, K)   # [B, H, K, head_dim]
        k = split_heads(k, L)   # [B, H, L, head_dim]
        v = split_heads(v, L)   # [B, H, L, head_dim]

        # 交叉注意力：query(K) × key(L)
        attn = (q @ k.transpose(-2, -1)) * self.scale   # [B, H, K, L]
        attn = attn.softmax(dim=-1)

        # 各病灶的空间激活图：对多头取均值
        lesion_attn = attn.mean(dim=1)   # [B, K, L]

        # 加权聚合 value
        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, K, C)  # [B, K, C]
        out = self.out_proj(out)
        out = self.norm(out)

        return out, lesion_attn


# ======================================================================
#  ★ 新增模块 2：EdgeEnhancementModule
#    可学习的 Sobel 型卷积，专门检测骨赘的尖锐边缘轮廓。
#    初始化为标准 Sobel 核，训练中允许微调以适应 X-ray 纹理。
# ======================================================================

class EdgeEnhancementModule(nn.Module):
    """
    Learnable edge-enhancement for osteophyte detection.

    骨赘表现为骨边缘的尖锐凸起，在影像中对应强梯度区域。
    本模块使用可学习的 Sobel 风格卷积提取高频边缘响应，
    并与主干特征融合，使掩码在骨边缘处产生更高激活。

    输入/输出形状均为 [B, 1, H, W]（单通道灰度激活图）。
    """

    def __init__(self, learnable: bool = True):
        super().__init__()

        # 标准 Sobel 核（水平 + 垂直）
        sobel_x = torch.tensor([[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                 [ 0,  0,  0],
                                 [ 1,  2,  1]], dtype=torch.float32)

        # [out_channels, in_channels, kH, kW]
        kernel = torch.stack([sobel_x, sobel_y], dim=0).unsqueeze(1)  # [2, 1, 3, 3]

        if learnable:
            self.edge_conv = nn.Conv2d(1, 2, kernel_size=3, padding=1, bias=False)
            self.edge_conv.weight = nn.Parameter(kernel)
        else:
            self.register_buffer('fixed_kernel', kernel)
            self.edge_conv = None

        self.learnable = learnable

        # 融合边缘响应 → 单通道掩码修正项
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W]
        Returns:
            edge_map: [B, 1, H, W]，边缘增强响应图（骨赘高激活区域）
        """
        if self.learnable:
            edges = self.edge_conv(x)            # [B, 2, H, W]
        else:
            edges = F.conv2d(x, self.fixed_kernel, padding=1)

        edge_magnitude = self.fusion(edges)      # [B, 1, H, W]
        return edge_magnitude


# ======================================================================
#  ★ 新增模块 3：MultiScaleLesionFusion
#    将三路特征（基础掩码 / 病灶 query 激活 / 边缘图）
#    通过可学习权重加权融合，输出最终的病灶增强掩码。
# ======================================================================

class MultiScaleLesionFusion(nn.Module):
    """
    Weighted fusion of three lesion-aware feature streams:
      Stream A: base channel-projected mask    (从 Swin decoder 来的基础响应)
      Stream B: lesion query attention map     (三类病灶的交叉注意力激活)
      Stream C: edge enhancement map           (骨赘边缘)

    三路特征通过可学习标量权重融合，权重经 Softmax 归一化，
    确保融合结果可解释（各路贡献之和为 1）。
    """

    def __init__(self, input_size: tuple):
        super().__init__()
        self.H, self.W = input_size

        # 三路融合权重（可学习标量），Softmax 归一化
        self.fusion_weights = nn.Parameter(torch.ones(3))

        # 将三路 concat 后做轻量 1×1 卷积精细化
        self.refine = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(8, 1, kernel_size=1, bias=False),
        )

    def forward(self,
                base_map:    torch.Tensor,
                lesion_map:  torch.Tensor,
                edge_map:    torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_map:   [B, 1, H, W]  基础解码器输出
            lesion_map: [B, 1, H, W]  病灶 query 激活（三路求和后）
            edge_map:   [B, 1, H, W]  边缘增强输出
        Returns:
            fused:      [B, 1, H, W]  融合后的病灶感知掩码
        """
        w = self.fusion_weights.softmax(dim=0)   # [3]，归一化权重

        weighted = w[0] * base_map + w[1] * lesion_map + w[2] * edge_map

        # 将三路 concat 送入轻量卷积做空间精细化
        concat = torch.cat([base_map, lesion_map, edge_map], dim=1)  # [B, 3, H, W]
        refined = self.refine(concat)                                  # [B, 1, H, W]

        # 残差融合：加权叠加 + 精细化分支
        fused = weighted + refined
        return fused


# ======================================================================
#  ★ 核心类：Lesion_Aware_Prompt_Generator
#    替换原 EVo_Mask_Generator，针对骨科影像病灶区域增强设计。
#
#  主要改动：
#    1. 通道投影输出改为 3 通道（对应三类病灶），再加权合并为 1 通道
#    2. 新增 LesionQueryAttention，在 14×14 尺度插入交叉注意力
#    3. 新增 EdgeEnhancementModule，在 224×224 尺度提取骨赘边缘
#    4. 新增 MultiScaleLesionFusion，三路融合输出最终掩码
#    5. 强度修正（intensity_revision）保持与原版一致
# ======================================================================

class Lesion_Aware_Prompt_Generator(nn.Module):
    """
    Lesion-Aware Spatial Prompt Generator for musculoskeletal imaging.

    针对三类典型骨科病变进行空间掩码增强：
      - 骨赘 (Osteophyte)        : 骨边缘尖锐增生，由 EdgeEnhancement 专项响应
      - 关节间隙 (Joint Space)    : 间隙狭窄，由 LesionQueryAttention Query-1 响应
      - 软骨退化 (Cartilage Deg.) : 纹理及密度变化，由 LesionQueryAttention Query-2 响应

    数据流:
        z [B, z_dim, T/2, 7, 7]
          → projection2spatial [B, 7, 7, z_dim]
          → upsample1 + block1 + LesionQueryAttention [B, 14*14, z_dim//2]
          → upsample2 + block2 [B, 56*56, z_dim//8]
          → upsample3 [B, 224*224, z_dim//32]
          → channel_projection(3 ch) [B, 224, 224, 3]
          → EdgeEnhancement
          → MultiScaleLesionFusion
          → softmax + min-max norm
          → mask [B, 3, T, 224, 224]
    """

    def __init__(self, cfg):
        super().__init__()

        z_dim, bias_flag, frames, input_size, act = load_options(cfg)

        self.z_dim       = z_dim
        self.frames      = frames
        self.input_size  = input_size
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.intensity_revision = cfg.MODEL.EVO.SPATIAL_IR

        # ── 空间分辨率定义 ──────────────────────────────────────────────
        self.input_resolution = (7, 7)
        swin_res1 = tuple([r * 2  for r in self.input_resolution])   # 14 × 14
        swin_res2 = tuple([r * 8  for r in self.input_resolution])   # 56 × 56
        # 最终分辨率 = input_size，通常 224 × 224

        depth       = 2
        num_heads   = 8
        window_size = 7
        mlp_ratio   = 4
        norm_layer  = nn.LayerNorm

        # ── 时序 → 空间投影 ────────────────────────────────────────────
        self.projection2spatial = nn.Sequential(
            nn.Linear(self.frames // 2, self.frames // 2, bias=bias_flag),
            act(),
            nn.Linear(self.frames // 2, 1, bias=bias_flag),
            act(),
            nn.LayerNorm([self.input_resolution[0],
                          self.input_resolution[1],
                          self.z_dim, 1]),
        )

        # ── Swin-UNet 上采样 ───────────────────────────────────────────
        self.upsample1 = PatchExpand(self.input_resolution, dim=z_dim,
                                     dim_scale=2, expand_scale=2,
                                     norm_layer=norm_layer)
        self.upsample2 = PatchExpand(swin_res1, dim=z_dim // 2,
                                     dim_scale=4, expand_scale=4,
                                     norm_layer=norm_layer)
        self.upsample3 = PatchExpand(swin_res2, dim=z_dim // 8,
                                     dim_scale=4, expand_scale=4,
                                     norm_layer=norm_layer)

        # ── Swin Transformer 块 ────────────────────────────────────────
        self.block1 = nn.ModuleList([
            SwinTransformerBlock(
                dim=z_dim // 2, input_resolution=swin_res1,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.block2 = nn.ModuleList([
            SwinTransformerBlock(
                dim=z_dim // 8, input_resolution=swin_res2,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, norm_layer=norm_layer)
            for i in range(depth)
        ])

        # ── ★ 新增：病灶 Query 注意力（在 14×14 尺度插入）─────────────
        # 三个可学习 query 分别对应骨赘 / 关节间隙 / 软骨退化
        self.lesion_query_attn = LesionQueryAttention(
            dim=z_dim // 2,
            num_lesion_types=3,
            num_heads=min(num_heads, z_dim // 2 // 32),  # 保证 head_dim ≥ 32
            norm_layer=norm_layer,
        )

        # ── 通道投影（改为输出 3 通道，对应三类病灶）────────────────────
        if z_dim in [768, 1024]:
            ch_mid = z_dim // 96
            ch_mid = max(ch_mid, 3)       # 保证中间层不小于输出维度
            self.channel_projection = nn.Sequential(
                nn.Linear(z_dim // 32, ch_mid, bias=bias_flag),
                act(),
                nn.Linear(ch_mid, 3, bias=bias_flag),  # 3 通道 → 3 类病灶
            )
        elif z_dim == 128:
            self.channel_projection = nn.Sequential(
                nn.Linear(z_dim // 32, z_dim // 32, bias=bias_flag),
                act(),
                nn.Linear(z_dim // 32, 3, bias=bias_flag),
            )
        else:
            # 通用分支
            self.channel_projection = nn.Sequential(
                nn.Linear(z_dim // 32, max(z_dim // 64, 8), bias=bias_flag),
                act(),
                nn.Linear(max(z_dim // 64, 8), 3, bias=bias_flag),
            )

        # ── ★ 新增：边缘增强（骨赘检测，在 224×224 尺度操作）──────────
        self.edge_enhancement = EdgeEnhancementModule(learnable=True)

        # ── ★ 新增：多尺度病灶融合 ────────────────────────────────────
        self.lesion_fusion = MultiScaleLesionFusion(input_size=input_size)

        # ── 最终归一化 ─────────────────────────────────────────────────
        self.softmax = nn.Softmax(dim=1)

    # ------------------------------------------------------------------

    def _build_lesion_map(self,
                          lesion_attn: torch.Tensor,
                          H: int, W: int) -> torch.Tensor:
        """
        将 LesionQueryAttention 输出的激活图 [B, K, L]
        上采样到目标分辨率并求和，得到 [B, 1, H, W]。
        """
        B, K, L = lesion_attn.shape
        # L = h_feat * w_feat（注意力操作时的特征图尺寸）
        h_feat = w_feat = int(L ** 0.5)

        # [B, K, h_feat, w_feat]
        attn_map = lesion_attn.view(B, K, h_feat, w_feat)

        # 上采样到目标 224 × 224
        attn_map = F.interpolate(attn_map, size=(H, W),
                                 mode='bilinear', align_corners=False)

        # 三类病灶激活图求和后压缩为单通道
        lesion_map = attn_map.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        return lesion_map

    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor):
        """
        Args:
            z: [B, z_dim, T//2, 7, 7]  — Omnivore 提取的时空特征
        Returns:
            mask: [B, 3, T, H, W]       — 病灶感知增强掩码（已归一化到 [0,1]）
        """
        H_out, W_out = self.input_size

        # ── Step 1: 时序 → 空间投影 ────────────────────────────────────
        # [B, z_dim, T//2, 7, 7] → [B, 7, 7, z_dim]
        z = self.projection2spatial(z.permute(0, 3, 4, 1, 2))
        z = z.squeeze(-1).reshape(-1,
            self.input_resolution[0] * self.input_resolution[1], self.z_dim)

        # ── Step 2: 上采样 ×2 + Swin Block（14×14 尺度）────────────────
        z = self.upsample1(z)                # [B, 14*14, z_dim//2]
        for blk in self.block1:
            z = blk(z)

        # ── Step 3: ★ 病灶 Query 交叉注意力（在 14×14 尺度）────────────
        _, lesion_attn = self.lesion_query_attn(z)   # lesion_attn: [B, 3, 196]

        # ── Step 4: 上采样 ×4 + Swin Block（56×56 尺度）────────────────
        z = self.upsample2(z)                # [B, 56*56, z_dim//8]
        for blk in self.block2:
            z = blk(z)

        # ── Step 5: 上采样 ×4（224×224 尺度）───────────────────────────
        z = self.upsample3(z)                # [B, 224*224, z_dim//32]

        # ── Step 6: 通道投影（3 通道 → 三类病灶基础图）─────────────────
        z = self.channel_projection(z)       # [B, 224*224, 3]

        B = z.shape[0]
        # [B, H, W, 3] → [B, 3, H, W]，逐类病灶空间图
        lesion_ch_maps = z.reshape(B, H_out, W_out, 3).permute(0, 3, 1, 2)

        # 基础掩码：三通道求均值压缩为单通道
        base_map = lesion_ch_maps.mean(dim=1, keepdim=True)  # [B, 1, H, W]

        # ── Step 7: ★ 边缘增强（骨赘检测）─────────────────────────────
        edge_map = self.edge_enhancement(base_map)           # [B, 1, H, W]

        # ── Step 8: ★ 病灶 Query 激活图上采样 ──────────────────────────
        lesion_query_map = self._build_lesion_map(
            lesion_attn, H_out, W_out)                       # [B, 1, H, W]

        # ── Step 9: ★ 多尺度病灶融合 ────────────────────────────────────
        fused = self.lesion_fusion(base_map,
                                   lesion_query_map,
                                   edge_map)                  # [B, 1, H, W]

        # ── Step 10: SoftMax + Min-Max 归一化 ───────────────────────────
        mask_flat = fused.squeeze(1).reshape(B, -1)   # [B, H*W]
        mask_flat = self.softmax(mask_flat * self.logit_scale)

        max_probs = mask_flat.max(dim=-1).values.view(B, 1, 1, 1, 1)
        min_probs = mask_flat.min(dim=-1).values.view(B, 1, 1, 1, 1)

        mask = mask_flat.reshape(B, 1, 1, H_out, W_out)
        mask = torch.div(mask - min_probs, max_probs - min_probs + 1e-8)

        # ── Step 11: 扩展到完整时序帧维度 ──────────────────────────────
        mask = mask.expand(-1, 3, self.frames, -1, -1)   # [B, 3, T, H, W]

        return mask


# ======================================================================
#  下游类保持与原版接口完全兼容（仅替换内部 Generator）
# ======================================================================

class EVo_Context_Generator(nn.Module):
    """时序上下文 Prompt（保持原版不变）"""

    def __init__(self, cfg):
        super().__init__()
        z_dim, bias_flag, _, _, act = load_options(cfg)

        if "RN50"  in cfg.MODEL.BACKBONE.IMAGE.NAME: latent_dim = 1024
        elif "L/14" in cfg.MODEL.BACKBONE.IMAGE.NAME: latent_dim = 768
        else:                                          latent_dim = 512

        self.context_scale = nn.Parameter(torch.ones([]) * 0.001)
        self.feature_adaptation = nn.Sequential(
            nn.Linear(z_dim, latent_dim, bias=bias_flag),
            act(),
            nn.LayerNorm([latent_dim])
        )

    def forward(self, z):
        z = torch.mean(z, [-3, -2, -1])
        context_prompt = self.feature_adaptation(z) * self.context_scale
        return context_prompt.unsqueeze(1)


class EVo_Prompts(nn.Module):
    """组合 Spatial + Temporal Prompt，接口与原版完全相同"""

    def __init__(self, cfg):
        super().__init__()
        # ★ 使用 Lesion_Aware_Prompt_Generator 替换 EVo_Mask_Generator
        self.dec_s = Lesion_Aware_Prompt_Generator(cfg)
        self.dec_t = EVo_Context_Generator(cfg)

    def forward(self, z):
        mask_s         = self.dec_s(z)    # [B, 3, T, H, W]
        context_prompt = self.dec_t(z)    # [B, 1, latent_dim]
        return mask_s, context_prompt


class EVoPrompt(nn.Module):
    """
    顶层模型，接口与原版完全相同。
    内部将 EVo_Mask_Generator 替换为 Lesion_Aware_Prompt_Generator。
    """

    def __init__(self, cfg):
        super().__init__()
        self.mean     = cfg.INPUT.PIXEL_MEAN
        self.std      = cfg.INPUT.PIXEL_STD
        self.enc_name = cfg.MODEL.EVO.ENC_NAME

        self.enc = torch.hub.load("facebookresearch/omnivore", model=self.enc_name)
        for param in self.enc.parameters():
            param.requires_grad = False

        dec_type = {
            'Mask':    Lesion_Aware_Prompt_Generator,   # ★ 替换
            'Context': EVo_Context_Generator,
            'Both':    EVo_Prompts,
        }
        self.decoder_type       = cfg.MODEL.EVO.DEC_TYPE
        self.dec                = dec_type[cfg.MODEL.EVO.DEC_TYPE](cfg)
        self.prompt_aggregation = cfg.MODEL.EVO.PROMPT_AGGREGATION

        self._init_weights(cfg)

    def _init_weights(self, cfg):
        init_type = cfg.MODEL.EVO.PROMPT_INIT
        if init_type == "constant":
            for m in self.dec.modules():
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 0.02)
        elif init_type == "zero":
            for m in self.dec.modules():
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.zeros_(m.weight)
        elif init_type == "kaiming":
            for m in self.dec.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                    if m.weight is not None:
                        nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor):
        if self.enc_name.startswith("omnivore"):
            z = self.enc.trunk(x, out_feat_keys=['stage3'])[0]
        else:
            z = self.enc(x)
  
        if self.decoder_type == "Both":
            p_s, p_t = self.dec(z)
            return x * p_s, p_t

        elif self.decoder_type == "Context":
            p_t = self.dec(z)
            return x, p_t

        else:   # "Mask" → Lesion_Aware_Prompt_Generator
            p_s = self.dec(z)
            return x * p_s